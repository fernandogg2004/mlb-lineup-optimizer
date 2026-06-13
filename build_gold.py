"""
build_gold.py  [DEPRECATED — usar scripts/build_gold_v3.py]
=============
ATENCIÓN: este pipeline v2 está deprecado. El Gold de entrenamiento activo es
features_train_v3.parquet, generado por scripts/build_gold_v3.py (8 clases,
features compartidas con serving via src/features/shared_features.py).
Se conserva solo como referencia histórica; no reentrenar con su salida.

Reconstruye el parquet de features de entrenamiento (Gold) a partir de
todos los Silver disponibles.

Para cada PA en Silver calcula:
  - Rolling stats (7d / 15d / 30d) con shift(1) anti-leakage
  - EWMA xwoba (alpha=0.2 y 0.5) con shift(1) anti-leakage
  - Stats platoon career-to-date (vs mismo lado de lanzador) con shrinkage James-Stein
  - Per-PA Statcast raw (xwoba, launch_speed, launch_angle) — se guardan en Gold
    pero el modelo las excluye por target-leakage (ver model_at_bat.py)

Uso:
    python build_gold.py                              # usa todos los Silver disponibles
    python build_gold.py --output data/gold/features_train_v2.parquet
    python build_gold.py --val-check                  # muestra stats del archivo generado

Tiempo estimado: 10-20 minutos para 10 temporadas (~3M PAs).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

SILVER_BASE  = ROOT / "data" / "silver" / "plate_appearances"
DEFAULT_OUT  = ROOT / "data" / "gold" / "features_train_v2.parquet"

# James-Stein shrinkage thresholds (Lichtman "The Book")
_STAB_T = {"woba": 200, "k_rate": 60, "bb_rate": 120, "babip": 500, "iso": 160}
_LEAGUE  = {"woba": 0.318, "k_rate": 0.224, "bb_rate": 0.083, "babip": 0.298, "iso": 0.147}


# ---------------------------------------------------------------------------
# Load Silver
# ---------------------------------------------------------------------------

def load_silver() -> pl.DataFrame:
    parts = [
        pl.read_parquet(p, hive_partitioning=False)
        for p in sorted(SILVER_BASE.glob("season=*/data.parquet"))
    ]
    if not parts:
        raise FileNotFoundError(f"No hay Silver Parquets en {SILVER_BASE}")
    df = pl.concat(parts, how="diagonal_relaxed")
    print(f"  Silver cargado: {len(df):,} PAs  |  "
          f"temporadas {sorted(df['season'].unique().to_list())}", flush=True)
    return df.sort(["batter_id", "game_date"])


# ---------------------------------------------------------------------------
# Rolling features (shift-based, anti-leakage)
# ---------------------------------------------------------------------------

def _build_daily(silver: pl.DataFrame) -> pl.DataFrame:
    """Agrega al grano diario (1 fila por batter × game_date).

    Re-agrega tras _aggregate_to_daily porque esa función agrupa también por
    batter_stand: los switch hitters producían DOS filas por fecha y el join
    posterior por (batter_id, game_date) duplicaba PAs en el Gold.
    """
    from src.features.features_rolling import _add_pa_event_flags, _aggregate_to_daily
    daily = (
        _aggregate_to_daily(_add_pa_event_flags(silver.lazy()))
        .collect()
    )
    return (
        daily.group_by(["batter_id", "game_date", "season"])
        .agg([
            pl.col("pa").sum(), pl.col("k").sum(), pl.col("bb").sum(),
            pl.col("hr").sum(), pl.col("hits").sum(),
            pl.col("hard_hits").sum(), pl.col("bip").sum(),
            # Media ponderada aproximada de las métricas por-juego
            pl.col("xwoba_mean").drop_nulls().mean(),
            pl.col("launch_speed_mean").drop_nulls().mean(),
            pl.col("launch_angle_mean").drop_nulls().mean(),
        ])
        .sort(["batter_id", "game_date"])
    )


def _rolling_shift(daily: pl.DataFrame, window: int, min_pa: int, sfx: str) -> pl.DataFrame:
    """Ventana rolling de N juegos con shift(1) — anti-leakage garantizado."""
    agg_cols = ["pa", "k", "bb", "hr", "hits", "hard_hits", "bip",
                "xwoba_mean", "launch_speed_mean"]

    shifted = daily.sort(["batter_id", "game_date"]).with_columns([
        pl.col(c).shift(1).over("batter_id").alias(f"{c}_s") for c in agg_cols
    ])

    rolled = shifted.with_columns([
        pl.col(f"{c}_s")
        .rolling_sum(window_size=window, min_periods=1)
        .over("batter_id")
        .alias(f"{c}_r{sfx}")
        if c not in ("xwoba_mean", "launch_speed_mean")
        else
        pl.col(f"{c}_s")
        .rolling_mean(window_size=window, min_periods=1)
        .over("batter_id")
        .alias(f"{c}_r{sfx}")
        for c in agg_cols
    ])

    pa_col = f"pa_r{sfx}"
    result = rolled.with_columns([
        pl.col(f"pa_r{sfx}").alias(f"pa_{sfx}"),
        pl.when(pl.col(pa_col) >= min_pa)
            .then(pl.col(f"k_r{sfx}") / pl.col(pa_col).clip(lower_bound=1))
            .otherwise(None).alias(f"k_rate_{sfx}"),
        pl.when(pl.col(pa_col) >= min_pa)
            .then(pl.col(f"bb_r{sfx}") / pl.col(pa_col).clip(lower_bound=1))
            .otherwise(None).alias(f"bb_rate_{sfx}"),
        pl.when(pl.col(pa_col) >= min_pa)
            .then(pl.col(f"hr_r{sfx}") / pl.col(pa_col).clip(lower_bound=1))
            .otherwise(None).alias(f"hr_rate_{sfx}"),
        pl.when(pl.col(pa_col) >= min_pa)
            .then(pl.col(f"hard_hits_r{sfx}") / pl.col(f"bip_r{sfx}").clip(lower_bound=1))
            .otherwise(None).alias(f"hard_hit_rate_{sfx}"),
        pl.when(pl.col(pa_col) >= min_pa)
            .then(pl.col(f"xwoba_mean_r{sfx}"))
            .otherwise(None).alias(f"xwoba_{sfx}"),
        pl.when(pl.col(pa_col) >= min_pa)
            .then(pl.col(f"launch_speed_mean_r{sfx}"))
            .otherwise(None).alias(f"launch_speed_{sfx}"),
    ]).select(
        ["batter_id", "game_date",
         f"pa_{sfx}", f"k_rate_{sfx}", f"bb_rate_{sfx}", f"hr_rate_{sfx}",
         f"hard_hit_rate_{sfx}", f"xwoba_{sfx}", f"launch_speed_{sfx}"]
    )
    return result


def _ewma_shift(daily: pl.DataFrame) -> pl.DataFrame:
    """EWMA xwoba con shift(1) anti-leakage."""
    return daily.sort(["batter_id", "game_date"]).with_columns([
        pl.col("xwoba_mean").shift(1)
            .ewm_mean(alpha=0.2, min_periods=1, adjust=True)
            .over("batter_id").alias("xwoba_ewma_alpha02"),
        pl.col("xwoba_mean").shift(1)
            .ewm_mean(alpha=0.5, min_periods=1, adjust=True)
            .over("batter_id").alias("xwoba_ewma_alpha05"),
    ]).select(["batter_id", "game_date", "xwoba_ewma_alpha02", "xwoba_ewma_alpha05"])


# ---------------------------------------------------------------------------
# Platoon + James-Stein stabilization (career-to-date, shift anti-leakage)
# ---------------------------------------------------------------------------

def _build_platoon_stabilized(silver: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula stats platoon career-to-date (mismo lado del lanzador) con shrinkage.

    Para cada PA (batter_id, game_date, pitcher_throws):
      - Acumula PAs, K, BB, HR, hits, bip, xwoba hasta (game_date - 1 PA) del mismo pitcher_throws
      - Aplica James-Stein: stat_stab = mu + (1-B)*(raw - mu), B = T/(T + PA_obs)
    """
    from src.features.features_rolling import _add_pa_event_flags

    # Agregar flags para calcular hits e iso
    flags = _add_pa_event_flags(silver.lazy()).collect()

    # Agregar is_hit y is_tb para estimar woba/iso simplificados
    flags = flags.with_columns([
        pl.col("pa_result").is_in(["single", "double", "triple", "home_run"])
            .cast(pl.Int8).alias("is_hit"),
        # Total bases aproximado para ISO = SLG - AVG
        (
            (pl.col("pa_result") == "single").cast(pl.Int8) * 1 +
            (pl.col("pa_result") == "double").cast(pl.Int8) * 2 +
            (pl.col("pa_result") == "triple").cast(pl.Int8) * 3 +
            (pl.col("pa_result") == "home_run").cast(pl.Int8) * 4
        ).alias("tb"),
    ])

    # Sort para cumsum correcto
    flags = flags.sort(["batter_id", "pitcher_throws", "game_date"])

    # Cumsum de cada stat por (batter_id, pitcher_throws)
    flags = flags.with_columns([
        pl.col("pa_flag").cum_sum().over(["batter_id", "pitcher_throws"]).alias("cum_pa"),
        pl.col("k_flag").cum_sum().over(["batter_id", "pitcher_throws"]).alias("cum_k"),
        pl.col("bb_flag").cum_sum().over(["batter_id", "pitcher_throws"]).alias("cum_bb"),
        pl.col("hr_flag").cum_sum().over(["batter_id", "pitcher_throws"]).alias("cum_hr"),
        pl.col("is_hit").cum_sum().over(["batter_id", "pitcher_throws"]).alias("cum_hits"),
        pl.col("bip_flag").cum_sum().over(["batter_id", "pitcher_throws"]).alias("cum_bip"),
        pl.col("tb").cum_sum().over(["batter_id", "pitcher_throws"]).alias("cum_tb"),
        # xwoba: media acumulada = (suma_prev) / (pa_prev). Más fácil con cumsum de xwoba.
        pl.col("xwoba").fill_null(0.0).cum_sum()
            .over(["batter_id", "pitcher_throws"]).alias("cum_xwoba"),
        pl.col("xwoba").is_not_null().cast(pl.Int32).cum_sum()
            .over(["batter_id", "pitcher_throws"]).alias("cum_xwoba_n"),
    ])

    # Shift(1) para excluir PA actual (anti-leakage)
    cum_cols = ["cum_pa", "cum_k", "cum_bb", "cum_hr",
                "cum_hits", "cum_bip", "cum_tb", "cum_xwoba", "cum_xwoba_n"]
    flags = flags.with_columns([
        pl.col(c).shift(1).over(["batter_id", "pitcher_throws"]).alias(c)
        for c in cum_cols
    ])

    # Derivar raw rates (career-to-date vs ese lado del lanzador)
    def _safe_div(num: pl.Expr, denom: pl.Expr) -> pl.Expr:
        return pl.when(denom > 0).then(num / denom).otherwise(None)

    flags = flags.with_columns([
        _safe_div(pl.col("cum_k"),  pl.col("cum_pa")).alias("k_raw"),
        _safe_div(pl.col("cum_bb"), pl.col("cum_pa")).alias("bb_raw"),
        _safe_div(pl.col("cum_hr"), pl.col("cum_pa")).alias("hr_raw"),
        # BABIP = (hits - hr) / bip
        _safe_div(pl.col("cum_hits") - pl.col("cum_hr"),
                  pl.col("cum_bip")).alias("babip_raw"),
        # wOBA approximated from xwoba mean
        _safe_div(pl.col("cum_xwoba"), pl.col("cum_xwoba_n")).alias("woba_raw"),
        # ISO = (TB - H) / AB ≈ (extra bases) / PA  (rough but consistent)
        _safe_div(pl.col("cum_tb") - pl.col("cum_hits"),
                  pl.col("cum_pa")).alias("iso_raw"),
    ])

    # James-Stein shrinkage
    mu_woba  = _LEAGUE["woba"]
    mu_k     = _LEAGUE["k_rate"]
    mu_bb    = _LEAGUE["bb_rate"]
    mu_babip = _LEAGUE["babip"]
    mu_iso   = _LEAGUE["iso"]

    t_woba  = _STAB_T["woba"]
    t_k     = _STAB_T["k_rate"]
    t_bb    = _STAB_T["bb_rate"]
    t_babip = _STAB_T["babip"]
    t_iso   = _STAB_T["iso"]

    def _stab(raw_col: str, mu: float, T: float, pa_col: str = "cum_pa") -> pl.Expr:
        """θ_stab = μ + (1-B)*(θ_raw - μ),  B = T/(T + PA)"""
        pa   = pl.col(pa_col).fill_null(0)
        raw  = pl.col(raw_col).fill_null(mu)
        b    = T / (T + pa)
        return mu + (1 - b) * (raw - mu)

    def _b(T: float, pa_col: str = "cum_pa") -> pl.Expr:
        pa = pl.col(pa_col).fill_null(0)
        return T / (T + pa)

    flags = flags.with_columns([
        _stab("woba_raw",  mu_woba,  t_woba) .cast(pl.Float32).alias("woba_stabilized"),
        _b(t_woba)                           .cast(pl.Float32).alias("woba_shrinkage_b"),
        _stab("k_raw",    mu_k,    t_k)     .cast(pl.Float32).alias("k_rate_stabilized"),
        _b(t_k)                              .cast(pl.Float32).alias("k_rate_shrinkage_b"),
        _stab("bb_raw",   mu_bb,   t_bb)    .cast(pl.Float32).alias("bb_rate_stabilized"),
        _b(t_bb)                             .cast(pl.Float32).alias("bb_rate_shrinkage_b"),
        _stab("babip_raw", mu_babip, t_babip).cast(pl.Float32).alias("babip_stabilized"),
        _b(t_babip)                          .cast(pl.Float32).alias("babip_shrinkage_b"),
        _stab("iso_raw",  mu_iso,  t_iso)   .cast(pl.Float32).alias("iso_stabilized"),
        _b(t_iso)                            .cast(pl.Float32).alias("iso_shrinkage_b"),
    ])

    keep = [
        "batter_id", "pitcher_throws", "game_date",
        "woba_stabilized", "woba_shrinkage_b",
        "k_rate_stabilized", "k_rate_shrinkage_b",
        "bb_rate_stabilized", "bb_rate_shrinkage_b",
        "babip_stabilized", "babip_shrinkage_b",
        "iso_stabilized", "iso_shrinkage_b",
    ]
    # Una fila por (batter, mano, fecha): el primer registro del día = stats al
    # inicio del juego. Sin esto el join PA-level era m:n y duplicaba filas.
    return (
        flags.select(keep)
        .group_by(["batter_id", "pitcher_throws", "game_date"], maintain_order=True)
        .first()
    )


# ---------------------------------------------------------------------------
# pa_outcome_idx mapping (same as pa_outcome_int but renamed for consistency)
# ---------------------------------------------------------------------------

def _add_outcome_idx(silver: pl.DataFrame) -> pl.DataFrame:
    """Añade pa_outcome_idx al Silver (mapeado desde pa_outcome_int)."""
    if "pa_outcome_int" in silver.columns:
        return silver.with_columns(
            pl.col("pa_outcome_int").cast(pl.Int32).alias("pa_outcome_idx")
        )
    # Fallback: map from pa_result
    outcome_map = {
        "strikeout": 1, "strikeout_double_play": 1,
        "walk": 2, "intent_walk": 2, "hit_by_pitch": 2,
        "single": 3, "double": 4, "triple": 5, "home_run": 6,
    }
    return silver.with_columns(
        pl.col("pa_result")
        .map_elements(lambda e: outcome_map.get(e, 0), return_dtype=pl.Int32)
        .alias("pa_outcome_idx")
    )


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_gold(output_path: Path) -> None:
    print("=" * 70)
    print("  DEPRECATED: build_gold.py (v2) — usa scripts/build_gold_v3.py")
    print("  El modelo en producción se entrena con features_train_v3.parquet.")
    print("=" * 70, flush=True)
    print("\n[1/6] Cargando Silver...", flush=True)
    silver = load_silver()

    print("\n[2/6] Agregando al grano diario...", flush=True)
    t0 = time.time()
    daily = _build_daily(silver)
    print(f"  {len(daily):,} filas diarias  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[3/6] Calculando rolling features (7d / 15d / 30d)...", flush=True)
    t0 = time.time()
    r7d  = _rolling_shift(daily, 7,  3,  "7d")
    r15d = _rolling_shift(daily, 15, 3,  "15d")
    r30d = _rolling_shift(daily, 30, 10, "30d")
    ewma = _ewma_shift(daily)
    print(f"  Rolling listo ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4/6] Calculando platoon + shrinkage James-Stein...", flush=True)
    t0 = time.time()
    platoon = _build_platoon_stabilized(silver)
    print(f"  Platoon listo ({time.time()-t0:.0f}s)", flush=True)

    print("\n[5/6] Ensamblando Gold...", flush=True)
    t0 = time.time()

    # Silver base con outcome label
    base = _add_outcome_idx(silver).select([
        "batter_id", "pitcher_throws", "game_date", "season",
        "pa_outcome_idx",
        "xwoba", "launch_speed", "launch_angle",  # per-PA leakers (excluidos en entrenamiento)
    ])

    jk_bd = ["batter_id", "game_date"]
    jk_bp = ["batter_id", "pitcher_throws", "game_date"]

    gold = (
        base
        .join(r7d,     on=jk_bd, how="left")
        .join(r15d,    on=jk_bd, how="left")
        .join(r30d,    on=jk_bd, how="left")
        .join(ewma,    on=jk_bd, how="left")
        .join(platoon, on=jk_bp, how="left")
    )

    # Eliminar columnas de join que no son features
    gold = gold.drop([c for c in ("batter_id", "pitcher_throws") if c in gold.columns])

    # Cast Float32 para consistencia con Gold existente
    float_cols = [c for c in gold.columns
                  if gold[c].dtype in (pl.Float64,) and c not in ("batter_id", "pitcher_id")]
    gold = gold.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])

    # Rellenar nulos con 0 (igual que en entrenamiento original)
    gold = gold.with_columns([
        pl.col(c).fill_null(0.0)
        for c in gold.columns if gold[c].dtype == pl.Float32
    ])

    print(f"  Gold ensamblado: {len(gold):,} filas, {len(gold.columns)} columnas ({time.time()-t0:.0f}s)", flush=True)

    print("\n[6/6] Guardando Gold...", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gold.write_parquet(output_path)
    print(f"  Guardado: {output_path}  ({output_path.stat().st_size / 1e6:.0f} MB)", flush=True)

    # Resumen
    print("\nResumen del Gold generado:")
    print(f"  Filas:       {len(gold):,}")
    print(f"  Columnas:    {len(gold.columns)}")
    print(f"  Temporadas:  {sorted(gold['season'].unique().to_list())}")
    print(f"  pa_outcome_idx distribucion:")
    for v, c in sorted(gold["pa_outcome_idx"].value_counts().iter_rows()):
        labels = ["OUT", "K", "BB/HBP", "1B", "2B", "3B", "HR"]
        print(f"    {v} ({labels[v]:6s}): {c:>9,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruye el Gold parquet de entrenamiento")
    parser.add_argument("--output", default=str(DEFAULT_OUT), metavar="PATH",
                        help=f"Ruta de salida (default: {DEFAULT_OUT})")
    parser.add_argument("--val-check", action="store_true",
                        help="Solo muestra stats del archivo si ya existe")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    out = Path(args.output)

    if args.val_check:
        if not out.exists():
            print(f"No existe: {out}")
            return
        df = pl.read_parquet(out)
        print(f"Filas: {len(df):,}  Cols: {len(df.columns)}  Temporadas: {sorted(df['season'].unique().to_list())}")
        for v, c in sorted(df["pa_outcome_idx"].value_counts().iter_rows()):
            print(f"  {v}: {c:,}")
        return

    t_start = time.time()
    build_gold(out)
    print(f"\nTiempo total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
