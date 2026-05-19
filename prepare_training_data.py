"""
prepare_training_data.py
========================
Descarga datos Statcast vía pybaseball, mapea al schema Silver del proyecto,
calcula las features rolling y platoon usando los engines existentes,
y genera el Gold Parquet listo para entrenar el modelo.

Uso:
    python prepare_training_data.py --start-year 2021 --end-year 2024
    python prepare_training_data.py --start-year 2023 --end-year 2024   # prueba rápida
    python prepare_training_data.py --start-year 2021 --end-year 2024 --output models/features.parquet

Salida:
    data/gold/features_train.parquet   ← Gold feature matrix lista para el modelo
    data/silver/plate_appearances/     ← Silver Parquet por temporada (cacheado)
    data/raw/                          ← Raw Statcast descargado (cacheado)

Una vez generado el Parquet, entrenar con:
    python -m src.models.model_at_bat train \\
        --features-path data/gold/features_train.parquet \\
        --val-seasons 2024 \\
        --output-dir models/
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import structlog

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Mapas de outcome  (Statcast events → label 0-6)
# ---------------------------------------------------------------------------

_OUTCOME_MAP: dict[str, int] = {
    # 0 — OUT_IN_PLAY
    "field_out": 0, "grounded_into_double_play": 0, "double_play": 0,
    "triple_play": 0, "fielders_choice": 0, "fielders_choice_out": 0,
    "force_out": 0, "field_error": 0, "other_out": 0,
    "sac_fly": 0, "sac_bunt": 0, "sac_fly_double_play": 0,
    "catcher_interf": 0, "fan_interference": 0,
    # 1 — STRIKEOUT
    "strikeout": 1, "strikeout_double_play": 1,
    # 2 — WALK / HBP
    "walk": 2, "intent_walk": 2, "hit_by_pitch": 2,
    # 3-6 — HITS
    "single": 3, "double": 4, "triple": 5, "home_run": 6,
}

# pa_result que los engines esperan (valores exactos para los filtros internos)
_PA_RESULT_PASSTHROUGH = {
    "strikeout", "strikeout_double_play",
    "walk", "intent_walk", "hit_by_pitch",
    "sac_fly", "sac_bunt",
}
_HIT_EVENTS = {"single", "double", "triple", "home_run"}


# ---------------------------------------------------------------------------
# Paso 1: Descarga Statcast (cacheada por año)
# ---------------------------------------------------------------------------

def _download_season(year: int, raw_dir: Path) -> pl.DataFrame:
    """Descarga una temporada de Statcast mes a mes, con caché local."""
    import pybaseball

    cache_file = raw_dir / f"statcast_{year}.parquet"
    if cache_file.exists():
        log.info("cache_hit", year=year, path=str(cache_file))
        return pl.read_parquet(cache_file)

    pybaseball.cache.enable()
    log.info("downloading_season", year=year)

    # Temporada regular: abril-septiembre; playoffs: octubre
    month_ranges = [
        (date(year, 4, 1),  date(year, 4, 30)),
        (date(year, 5, 1),  date(year, 5, 31)),
        (date(year, 6, 1),  date(year, 6, 30)),
        (date(year, 7, 1),  date(year, 7, 31)),
        (date(year, 8, 1),  date(year, 8, 31)),
        (date(year, 9, 1),  date(year, 9, 30)),
        (date(year, 10, 1), date(year, 10, 31)),
    ]

    _NEEDED_COLS = [
        "game_date", "batter", "pitcher",
        "stand", "p_throws",
        "events", "pitch_type",
        "launch_speed", "launch_angle",
        "estimated_woba_using_speedangle",
    ]

    chunks: list[pl.DataFrame] = []
    for start, end in month_ranges:
        log.info("downloading_chunk", year=year, month=start.month)
        try:
            df_pd = pybaseball.statcast(
                start_dt=start.isoformat(),
                end_dt=end.isoformat(),
                verbose=False,
            )
            if df_pd is not None and len(df_pd) > 0:
                available = [c for c in _NEEDED_COLS if c in df_pd.columns]
                chunks.append(
                    pl.from_pandas(df_pd[available].reset_index(drop=True))
                )
                log.info("chunk_ok", pitches=len(df_pd), month=start.month)
        except Exception as exc:
            log.warning("chunk_failed", month=start.month, error=str(exc))

    if not chunks:
        raise RuntimeError(f"No se descargó ningún dato para la temporada {year}")

    raw = pl.concat(chunks, how="diagonal_relaxed")
    raw.write_parquet(cache_file)
    log.info("season_cached", year=year, rows=len(raw), path=str(cache_file))
    return raw


# ---------------------------------------------------------------------------
# Paso 2: Mapear Statcast → schema Silver
# ---------------------------------------------------------------------------

def _map_to_silver(raw: pl.DataFrame) -> pl.DataFrame:
    """
    Filtra a filas que terminan una PA (events != null) y mapea columnas
    al schema Silver que los feature engines esperan.

    Schema de salida:
        batter_id, pitcher_id, game_date, season,
        batter_stand, pitcher_throws, last_pitch_type,
        pa_result, hit_type, xwoba, launch_speed, launch_angle,
        pa_outcome_int
    """
    valid_events = list(_OUTCOME_MAP.keys())

    df = (
        raw
        .filter(
            pl.col("events").is_not_null()
            & pl.col("events").is_in(valid_events)
        )
        .with_columns([
            pl.col("batter").cast(pl.Int64).alias("batter_id"),
            pl.col("pitcher").cast(pl.Int64).alias("pitcher_id"),
            # game_date llega como datetime[ns] desde pybaseball → convertir a "YYYY-MM-DD"
            pl.col("game_date").dt.strftime("%Y-%m-%d").alias("game_date"),
            pl.col("stand").alias("batter_stand"),
            pl.col("p_throws").alias("pitcher_throws"),
            pl.col("pitch_type").alias("last_pitch_type"),
            pl.col("estimated_woba_using_speedangle").cast(pl.Float32).alias("xwoba"),
            pl.col("launch_speed").cast(pl.Float32).alias("launch_speed"),
            pl.col("launch_angle").cast(pl.Float32).alias("launch_angle"),

            # pa_result: valores exactos para los filtros del rolling y platoon engine
            pl.when(pl.col("events").is_in(list(_PA_RESULT_PASSTHROUGH)))
              .then(pl.col("events"))
              .otherwise(pl.lit("field_out"))
              .alias("pa_result"),

            # hit_type: solo para hits (single/double/triple/home_run)
            pl.when(pl.col("events").is_in(list(_HIT_EVENTS)))
              .then(pl.col("events"))
              .otherwise(None)
              .alias("hit_type"),

            # pa_outcome_int: label 0-6 para el modelo
            pl.col("events")
              .map_elements(lambda e: _OUTCOME_MAP.get(e, 0), return_dtype=pl.Int32)
              .alias("pa_outcome_int"),

            # season: extraer año directamente del datetime original
            pl.col("game_date").dt.year().cast(pl.Int32).alias("season"),
        ])
        .select([
            "batter_id", "pitcher_id", "game_date", "season",
            "batter_stand", "pitcher_throws", "last_pitch_type",
            "pa_result", "hit_type",
            "xwoba", "launch_speed", "launch_angle",
            "pa_outcome_int",
        ])
        .filter(
            pl.col("batter_stand").is_in(["L", "R", "S"])
            & pl.col("pitcher_throws").is_in(["L", "R"])
        )
    )

    log.info(
        "silver_mapped",
        rows=len(df),
        unique_batters=df["batter_id"].n_unique(),
        label_counts=df["pa_outcome_int"].value_counts().sort("pa_outcome_int").to_dicts(),
    )
    return df


# ---------------------------------------------------------------------------
# Paso 3: Rolling features para TODAS las fechas (en memoria)
# ---------------------------------------------------------------------------

def _compute_rolling_all_dates(silver: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula rolling features para cada (batter_id, game_date) de la historia
    completa usando las funciones internas de features_rolling.py.

    A diferencia de RollingFeaturesEngine.run() que devuelve solo el snapshot
    más reciente, aquí devolvemos UNA FILA POR (batter_id, game_date) para
    poder unirla al nivel de PA en el entrenamiento.

    Anti-leakage garantizado: shift(1) antes de cada rolling.
    """
    from src.features.features_rolling import (
        _add_pa_event_flags,
        _aggregate_to_daily,
        _compute_shift_rolling,
        _compute_ewma,
    )

    log.info("rolling_features_start", total_pa_rows=len(silver))

    # Añadir flags de evento (k_flag, bb_flag, hr_flag, hit_flag, etc.)
    lf_flags = _add_pa_event_flags(silver.lazy())

    # Agregar a grain diario: 1 fila por (batter_id, game_date)
    daily_df = _aggregate_to_daily(lf_flags).collect()
    log.info("daily_aggregation_done", rows=len(daily_df), batters=daily_df["batter_id"].n_unique())

    # Rolling windows por conteo de juegos (anti-leakage via shift)
    roll_7d  = _compute_shift_rolling(daily_df, window_games=7,  min_pa=3,  suffix="7d")
    roll_15d = _compute_shift_rolling(daily_df, window_games=15, min_pa=3,  suffix="15d")
    roll_30d = _compute_shift_rolling(daily_df, window_games=30, min_pa=10, suffix="30d")

    # EWMA lento (α=0.2) y rápido (α=0.5) sobre xwOBA
    daily_ewma = _compute_ewma(daily_df, alpha=0.2, col="xwoba_mean", output_col="xwoba_ewma_alpha02")
    daily_ewma = _compute_ewma(daily_ewma, alpha=0.5, col="xwoba_mean", output_col="xwoba_ewma_alpha05")

    # Seleccionar solo columnas nuevas de cada ventana (evitar duplicados al unir)
    join_key = ["batter_id", "game_date"]
    new_7d   = [c for c in roll_7d.columns  if c.endswith("_7d")]
    new_15d  = [c for c in roll_15d.columns if c.endswith("_15d")]
    new_30d  = [c for c in roll_30d.columns if c.endswith("_30d")]

    result = (
        daily_ewma.select(join_key + ["xwoba_ewma_alpha02", "xwoba_ewma_alpha05"])
        .join(roll_7d.select(join_key + new_7d),   on=join_key, how="left", coalesce=True)
        .join(roll_15d.select(join_key + new_15d), on=join_key, how="left", coalesce=True)
        .join(roll_30d.select(join_key + new_30d), on=join_key, how="left", coalesce=True)
    )

    log.info(
        "rolling_features_done",
        rows=len(result),
        feature_columns=len(result.columns) - 2,
    )
    return result


# ---------------------------------------------------------------------------
# Paso 4: Platoon features (en memoria, sin I/O de ficheros)
# ---------------------------------------------------------------------------

def _compute_platoon_inline(silver: pl.DataFrame, season: int) -> pl.DataFrame:
    """
    Calcula los splits platoon con Empirical Bayes para una temporada,
    usando los métodos internos de PlatoonSplitsEngine sin tocar el disco.
    """
    from src.features.features_platoon import (
        PlatoonSplitsEngine, SplitComputationConfig, WOBAWeights,
    )

    season_silver = silver.filter(pl.col("season") == season)
    if season_silver.is_empty():
        log.warning("no_silver_data_for_season", season=season)
        return pl.DataFrame()

    weights = WOBAWeights.for_season(season)
    config  = SplitComputationConfig(weights=weights)
    engine  = PlatoonSplitsEngine(silver_path="", output_path="", config=config)

    # Mapear eventos a indicadores binarios (singles, doubles, etc.)
    lf_events = engine._map_pa_results_to_events(season_silver.lazy())

    # Agregar por (batter_id, batter_stand, pitcher_throws)
    group_cols = ["batter_id", "batter_stand", "pitcher_throws"]
    agg        = engine._aggregate_by_handedness(lf_events, group_cols)
    metrics_df = engine._compute_raw_metrics(agg).collect()

    if metrics_df.is_empty():
        return metrics_df

    # Prior de la liga y Empirical Bayes shrinkage
    prior_df = engine._compute_league_priors(
        metrics_df, group_cols=["batter_stand", "pitcher_throws"]
    )
    stabilized = engine._stabilize(
        metrics_df, prior_df, join_cols=["batter_stand", "pitcher_throws"]
    )

    return stabilized.with_columns(pl.lit(season).alias("season"))


def _compute_platoon_all_seasons(silver: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """Ejecuta _compute_platoon_inline para cada temporada y concatena."""
    parts = []
    for season in seasons:
        log.info("computing_platoon_splits", season=season)
        splits = _compute_platoon_inline(silver, season)
        if not splits.is_empty():
            parts.append(splits)

    if not parts:
        raise ValueError("No se calcularon platoon splits en ninguna temporada")

    combined = pl.concat(parts, how="diagonal_relaxed")
    log.info("platoon_splits_done", rows=len(combined), seasons=seasons)
    return combined


# ---------------------------------------------------------------------------
# Paso 5: Unir todo en la matriz Gold de entrenamiento
# ---------------------------------------------------------------------------

def _build_gold(
    silver: pl.DataFrame,
    rolling: pl.DataFrame,
    platoon: pl.DataFrame,
) -> pl.DataFrame:
    """
    Une Silver + rolling features + platoon splits en el Gold Parquet final.

    Schema de salida:
        pa_outcome_idx (label 0-6)
        season, game_date
        <feature columns: Float32>
    """
    log.info("building_gold_matrix", silver_rows=len(silver))

    # Unir rolling features (1 fila por (batter_id, game_date))
    gold = silver.join(rolling, on=["batter_id", "game_date"], how="left", coalesce=True)

    # Seleccionar columnas estabilizadas del platoon engine
    platoon_feature_cols = [
        c for c in platoon.columns
        if c.endswith("_stabilized") or c.endswith("_shrinkage_b")
    ]
    platoon_join_cols = ["batter_id", "season", "batter_stand", "pitcher_throws"]
    platoon_sel = platoon.select(
        [c for c in platoon_join_cols if c in platoon.columns]
        + platoon_feature_cols
    )

    gold = gold.join(platoon_sel, on=platoon_join_cols, how="left", coalesce=True)

    # Identificar columnas de features (numéricas, excluir IDs y label)
    exclude = {
        "batter_id", "pitcher_id", "game_date", "season",
        "batter_stand", "pitcher_throws", "last_pitch_type",
        "pa_result", "hit_type", "pa_outcome_int",
    }
    feature_cols = sorted([
        c for c in gold.columns
        if c not in exclude
        and gold[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int8)
    ])

    # Rellenar nulos con 0.0 (ventanas rolling vacías para jugadores nuevos)
    gold = gold.with_columns(
        [pl.col(c).fill_null(0.0).cast(pl.Float32) for c in feature_cols]
    )

    # Gold final: label + claves temporales + features
    gold_final = gold.select(
        ["pa_outcome_int", "season", "game_date"] + feature_cols
    ).rename({"pa_outcome_int": "pa_outcome_idx"})

    # Estadísticas de distribución de labels
    label_dist = (
        gold_final["pa_outcome_idx"]
        .value_counts()
        .sort("pa_outcome_idx")
        .with_columns(
            (pl.col("count") / pl.col("count").sum() * 100).round(1).alias("pct")
        )
    )
    log.info(
        "gold_matrix_built",
        rows=len(gold_final),
        features=len(feature_cols),
        label_distribution=label_dist.to_dicts(),
    )
    return gold_final


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(
    start_year: int,
    end_year: int,
    output_path: Path,
    data_dir: Path,
) -> None:
    raw_dir    = data_dir / "raw"
    silver_dir = data_dir / "silver" / "plate_appearances"

    for d in [raw_dir, silver_dir, output_path.parent]:
        d.mkdir(parents=True, exist_ok=True)

    seasons = list(range(start_year, end_year + 1))
    log.info("pipeline_start", seasons=seasons, output=str(output_path))

    # --- Paso 1 + 2: Descarga y mapeo Silver ---
    silver_parts: list[pl.DataFrame] = []
    for year in seasons:
        raw    = _download_season(year, raw_dir)
        silver = _map_to_silver(raw)

        # Guardar Silver como Parquet particionado por temporada
        silver_out = silver_dir / f"season={year}" / "data.parquet"
        silver_out.parent.mkdir(parents=True, exist_ok=True)
        silver.write_parquet(silver_out)
        log.info("silver_saved", year=year, rows=len(silver))

        silver_parts.append(silver)

    all_silver = pl.concat(silver_parts, how="diagonal_relaxed")
    log.info("silver_combined", total_rows=len(all_silver))

    # --- Paso 3: Rolling features (todas las fechas de una vez) ---
    rolling = _compute_rolling_all_dates(all_silver)

    # --- Paso 4: Platoon features (en memoria) ---
    platoon = _compute_platoon_all_seasons(all_silver, seasons)

    # --- Paso 5: Construir Gold Parquet ---
    gold = _build_gold(all_silver, rolling, platoon)

    gold.write_parquet(output_path)
    log.info("gold_saved", path=str(output_path), rows=len(gold))

    print("\n" + "=" * 60)
    print(f"  Gold Parquet listo:  {output_path}")
    print(f"  Filas totales:       {len(gold):,}")
    print(f"  Features:            {len(gold.columns) - 3}")
    print(f"  Temporadas:          {start_year} – {end_year}")
    print("=" * 60)
    print("\nSiguiente paso — entrenar el modelo:")
    print(f"  python -m src.models.model_at_bat train \\")
    print(f"      --features-path {output_path} \\")
    print(f"      --val-seasons {end_year} \\")
    print(f"      --output-dir models/")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera el Gold Parquet de entrenamiento a partir de Statcast.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--start-year", type=int, default=2021, dest="start_year",
        help="Primera temporada a descargar."
    )
    p.add_argument(
        "--end-year", type=int, default=2024, dest="end_year",
        help="Última temporada a descargar (usada como validación)."
    )
    p.add_argument(
        "--output", default="data/gold/features_train.parquet", dest="output",
        help="Ruta del Gold Parquet de salida."
    )
    p.add_argument(
        "--data-dir", default="data", dest="data_dir",
        help="Directorio raíz para raw/, silver/ y gold/."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run(
        start_year=args.start_year,
        end_year=args.end_year,
        output_path=Path(args.output),
        data_dir=Path(args.data_dir),
    )


if __name__ == "__main__":
    main()
