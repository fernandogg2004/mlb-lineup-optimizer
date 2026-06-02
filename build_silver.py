"""
build_silver.py
===============
Descarga datos Statcast de pybaseball para temporadas 2015-2020 y los
transforma al formato Silver local (mismo schema que los parquets 2021-2024).

Uso:
    python build_silver.py                    # descarga 2015-2020 (solo los que faltan)
    python build_silver.py --years 2018 2019  # solo esos años
    python build_silver.py --force            # reescribe aunque ya existan

Tiempo estimado: 5-15 min por temporada dependiendo de la velocidad de pybaseball.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).parent
SILVER_BASE = ROOT / "data" / "silver" / "plate_appearances"

# Eventos que constituyen una PA terminada (excluye robos, pickoffs, etc.)
_PA_EVENTS = {
    "strikeout", "strikeout_double_play",
    "walk", "intent_walk", "hit_by_pitch",
    "single", "double", "triple", "home_run",
    "field_out", "grounded_into_double_play", "double_play", "triple_play",
    "force_out", "field_error", "fielders_choice", "fielders_choice_out",
    "sac_fly", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play",
    "other_out", "catcher_interf",
}

# Mapeo de events → pa_outcome_int (7 clases)
_OUTCOME_MAP: dict[str, int] = {
    "strikeout":              1,
    "strikeout_double_play":  1,
    "walk":                   2,
    "intent_walk":            2,
    "hit_by_pitch":           2,
    "single":                 3,
    "double":                 4,
    "triple":                 5,
    "home_run":               6,
}


def _outcome_int(event: str | None) -> int:
    if event is None:
        return 0
    return _OUTCOME_MAP.get(event, 0)


def _download_season(year: int) -> pl.DataFrame:
    """Descarga una temporada completa de Statcast via pybaseball.

    Intenta el año completo primero; si falla (error CSV común en años viejos)
    cae back a descarga mes a mes.
    Para el año en curso usa la fecha de hoy como límite superior.
    """
    from datetime import date as _date
    from pybaseball import statcast  # noqa: PLC0415

    today = _date.today()
    # Opening Day históricamente cae entre el 20 y 28 de marzo
    start = f"{year}-03-20"
    # Para el año en curso, no intentar más allá de hoy
    end_candidate = _date(year, 11, 5)
    end = str(min(end_candidate, today))

    print(f"  Descargando {year} ({start} → {end})...", flush=True)
    t0 = time.time()
    try:
        pdf = statcast(start_dt=start, end_dt=end, verbose=False)
        elapsed = time.time() - t0
        print(f"  Descargado: {len(pdf):,} filas pitch-level en {elapsed:.0f}s", flush=True)
        return pl.from_pandas(pdf)
    except Exception as exc:
        print(f"  Descarga año completo fallida ({exc}). Intentando mes a mes...", flush=True)

    # Fallback: descarga mes a mes
    months = [
        (f"{year}-03-28", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-11-05"),
    ]
    parts = []
    for s, e in months:
        print(f"    {s}...", flush=True)
        try:
            pdf = statcast(start_dt=s, end_dt=e, verbose=False)
            parts.append(pl.from_pandas(pdf))
            print(f"    {len(pdf):,} pitches", flush=True)
        except Exception as exc2:
            print(f"    ERROR {s}: {exc2}", flush=True)

    if not parts:
        raise RuntimeError(f"No se pudo descargar ningún mes de {year}")

    result = pl.concat(parts, how="diagonal_relaxed")
    elapsed = time.time() - t0
    print(f"  Descargado mes a mes: {len(result):,} filas en {elapsed:.0f}s", flush=True)
    return result


def _to_silver(raw: pl.DataFrame, year: int) -> pl.DataFrame:
    """Convierte DataFrame pitch-level de pybaseball al schema Silver."""
    # Columnas que necesitamos del raw Statcast
    needed = {
        "batter", "pitcher", "game_date", "stand", "p_throws",
        "pitch_type", "events", "bb_type",
        "estimated_woba_using_speedangle", "launch_speed", "launch_angle",
    }
    available = set(raw.columns)
    missing = needed - available
    if missing:
        print(f"  ADVERTENCIA: columnas faltantes en {year}: {missing}", flush=True)

    # Filtrar solo pitches que terminan una PA
    df = raw.filter(
        pl.col("events").is_not_null()
        & pl.col("events").is_in(list(_PA_EVENTS))
    )
    print(f"  PA-level rows (eventos validos): {len(df):,}", flush=True)

    # Construir Silver
    outcome_map_pl = pl.Series("events_k", list(_OUTCOME_MAP.keys()))
    outcome_map_v  = pl.Series("events_v", list(_OUTCOME_MAP.values()))

    silver = (
        df
        .with_columns([
            pl.col("batter").cast(pl.Int64).alias("batter_id"),
            pl.col("pitcher").cast(pl.Int64).alias("pitcher_id"),
            pl.col("game_date").cast(pl.Utf8).alias("game_date"),
            pl.lit(year).cast(pl.Int32).alias("season"),
            pl.col("stand").alias("batter_stand"),
            pl.col("p_throws").alias("pitcher_throws"),
            pl.col("pitch_type").alias("last_pitch_type"),
            pl.col("events").alias("pa_result"),
            # hit_type: usa events para identificar tipo de hit (single/double/triple/hr)
            pl.col("events").alias("hit_type"),
            # xwoba
            (pl.col("estimated_woba_using_speedangle")
             if "estimated_woba_using_speedangle" in available
             else pl.lit(None).cast(pl.Float32))
            .cast(pl.Float32).alias("xwoba"),
            pl.col("launch_speed").cast(pl.Float32) if "launch_speed" in available
            else pl.lit(None).cast(pl.Float32).alias("launch_speed"),
            pl.col("launch_angle").cast(pl.Float32) if "launch_angle" in available
            else pl.lit(None).cast(pl.Float32).alias("launch_angle"),
        ])
        .select([
            "batter_id", "pitcher_id", "game_date", "season",
            "batter_stand", "pitcher_throws", "last_pitch_type",
            "pa_result", "hit_type", "xwoba", "launch_speed", "launch_angle",
        ])
        # pa_outcome_int: mapeado en Python (es pequeño con map_elements)
        .with_columns(
            pl.col("pa_result")
            .map_elements(lambda e: _OUTCOME_MAP.get(e, 0), return_dtype=pl.Int32)
            .alias("pa_outcome_int")
        )
        .drop_nulls(subset=["batter_id", "pitcher_id", "game_date"])
    )

    return silver


def build_season(year: int, force: bool = False) -> int:
    out_dir  = SILVER_BASE / f"season={year}"
    out_path = out_dir / "data.parquet"

    if out_path.exists() and not force:
        existing = pl.read_parquet(out_path, hive_partitioning=False)
        print(f"  {year}: ya existe ({len(existing):,} PAs) — omitido (usa --force para reescribir)")
        return len(existing)

    out_dir.mkdir(parents=True, exist_ok=True)

    raw    = _download_season(year)
    silver = _to_silver(raw, year)
    silver.write_parquet(out_path)
    print(f"  {year}: guardado {len(silver):,} PAs → {out_path}")
    return len(silver)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga temporadas Statcast a Silver Parquet")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2015, 2021)),
                        help="Temporadas a descargar (default: 2015-2020)")
    parser.add_argument("--force", action="store_true",
                        help="Reescribir aunque ya existan")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    print(f"\nDescargando temporadas: {args.years}")
    print(f"Destino: {SILVER_BASE}\n")

    total_pas = 0
    for year in sorted(args.years):
        print(f"\n[{year}]")
        try:
            n = build_season(year, force=args.force)
            total_pas += n
        except Exception as exc:
            print(f"  ERROR en {year}: {exc}", file=sys.stderr)

    print(f"\nTotal PAs descargados/existentes: {total_pas:,}")
    print("Listo.")


if __name__ == "__main__":
    main()
