#!/usr/bin/env python
"""
build_gold_v3.py — Rebuild Gold training features v3
=====================================================

Builds ``data/gold/features_train_v3.parquet`` (one row per PA) from:
  - Silver plate_appearances (2015-2024)  — PA labels, batter stats
  - Raw Statcast (2021-2024)             — GIDP events + pitch_count_in_pa

Improvements over v2:
  1. 8-class label  (class 7 = DOUBLE_PLAY, including GIDP and strikeout_dp)
  2. pitch_count_in_pa feature (median-imputed 4.0 for 2015-2020)
  3. Era encoding   (era_shift_ban, era_universal_dh, era_first_year_shift_ban)
  4. batter_id / pitcher_id retained in output
  5. Pitcher FIP features: pitcher_fip, pitcher_k_rate, pitcher_bb_rate, pitcher_hr_rate
     (rolling 30-game window, anti-leakage shift)

Usage:
    .venv/Scripts/python scripts/build_gold_v3.py [--output-path ...]

Runtime: ~8-15 min on a modern laptop (rolling features are the bottleneck).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

# ─── Paths (relative to repo root) ────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
SILVER_ROOT  = REPO_ROOT / "data" / "silver" / "plate_appearances"
RAW_ROOT     = REPO_ROOT / "data" / "raw"
OUTPUT_PATH  = REPO_ROOT / "data" / "gold" / "features_train_v3.parquet"

# Ensure repo root is on sys.path so that `src.*` imports resolve correctly
# regardless of the working directory the script is invoked from.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ─── Label constants ───────────────────────────────────────────────────────────
GIDP_EVENTS = {"grounded_into_double_play", "strikeout_double_play", "double_play"}

# Seasons where raw Statcast is available (for GIDP + pitch_count)
RAW_SEASONS = {2021, 2022, 2023, 2024}

# Pitch count imputation para Silver legacy sin pitch_count_in_pa exacto.
# (El schema nuevo de build_silver.py trae el conteo exacto por PA; esta
# constante solo aplica al fallback. La feature está excluida del modelo por
# leakage intra-PA — ver _LEAKING en train_v3.py.)
PITCH_COUNT_IMPUTE: float = 4.0        # mediana global MLB


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Load Silver
# ──────────────────────────────────────────────────────────────────────────────

def load_silver_all() -> pl.DataFrame:
    """Load all available Silver plate_appearances into one DataFrame."""
    t0 = time.time()
    frames = []
    # Auto-detect available seasons from directory structure
    available_seasons = sorted(
        int(p.name.replace("season=", ""))
        for p in SILVER_ROOT.glob("season=*")
        if (p / "data.parquet").exists()
    )
    print(f"  Seasons found: {available_seasons}")
    for season in available_seasons:
        path = SILVER_ROOT / f"season={season}" / "data.parquet"
        df = pl.read_parquet(path, hive_partitioning=False)
        # Ensure season column is present
        if "season" not in df.columns:
            df = df.with_columns(pl.lit(season).cast(pl.Int32).alias("season"))
        frames.append(df)
        print(f"  Silver {season}: {len(df):,} rows")

    silver = pl.concat(frames, how="diagonal")

    # Normalise game_date to string "YYYY-MM-DD"
    if silver["game_date"].dtype != pl.Utf8:
        silver = silver.with_columns(
            pl.col("game_date").cast(pl.Utf8).str.slice(0, 10).alias("game_date")
        )
    else:
        silver = silver.with_columns(
            pl.col("game_date").str.slice(0, 10).alias("game_date")
        )

    print(f"Silver total: {len(silver):,} rows  [{time.time()-t0:.1f}s]")
    return silver


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Build PA-level augmentation table from raw Statcast
# ──────────────────────────────────────────────────────────────────────────────

def build_raw_pa_table(season: int) -> pl.DataFrame:
    """Return one row per PA from raw Statcast for a given season.

    Columns: batter, pitcher, game_date_str, pa_seq (0-based within group),
             is_gidp (bool), pitch_count (int).
    """
    path = RAW_ROOT / f"statcast_{season}.parquet"
    if not path.exists():
        return pl.DataFrame(schema={
            "batter": pl.Int64, "pitcher": pl.Int64, "game_date_str": pl.Utf8,
            "pa_seq": pl.Int32, "is_gidp": pl.Boolean, "pitch_count": pl.Int32,
        })

    raw = pl.read_parquet(path)

    # Normalise game_date to string
    raw = raw.with_columns(
        pl.col("game_date").cast(pl.Utf8).str.slice(0, 10).alias("game_date_str")
    )

    # Sort within each (batter, pitcher, game_date) to get consistent pitch order.
    # Raw Statcast is delivered in reverse chronological order within a game,
    # so we rely on the existing file order (already reflects pitch sequence).
    raw = raw.with_row_index("_row_idx")

    # Within each group, assign PA sequence using cumulative count of
    # preceding PA endings (events.is_not_null() shifted by 1).
    raw = raw.sort(["batter", "pitcher", "game_date_str", "_row_idx"])

    raw = raw.with_columns([
        (
            pl.col("events").is_not_null()
             .cast(pl.Int32)
             .shift(1, fill_value=0)
             .cum_sum()
             .over(["batter", "pitcher", "game_date_str"])
        ).alias("pa_seq"),
    ])

    # Keep only PA-ending rows (events != null) — these define the PA outcome
    pa_ends = raw.filter(pl.col("events").is_not_null())

    # Build one row per (batter, pitcher, game_date_str, pa_seq)
    # Count pitches per PA group
    pitch_counts = (
        raw.group_by(["batter", "pitcher", "game_date_str", "pa_seq"])
        .agg(pl.len().alias("pitch_count"))
    )

    result = (
        pa_ends.select([
            "batter", "pitcher", "game_date_str", "pa_seq", "events"
        ])
        .join(pitch_counts, on=["batter", "pitcher", "game_date_str", "pa_seq"],
              how="left", coalesce=True)
        .with_columns([
            pl.col("events").is_in(list(GIDP_EVENTS)).alias("is_gidp_raw"),
            pl.col("pitch_count").fill_null(1).cast(pl.Int32).alias("pitch_count_raw"),
            pl.col("pa_seq").cast(pl.Int32),
        ])
        .drop(["events", "pitch_count"])
    )

    print(f"  Raw {season}: {len(result):,} PA rows  "
          f"(GIDP: {result['is_gidp_raw'].sum():,})")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Join raw augmentation into Silver
# ──────────────────────────────────────────────────────────────────────────────

def compute_pitch_count_imputes(silver: pl.DataFrame) -> None:
    """Diagnoses pitch count data availability in the raw Statcast files.

    If raw Statcast parquets are PA-level aggregates (1 row/PA) rather than
    pitch-level sequences, the median pitch_count will be ~1 and a warning
    is printed. In that case PITCH_COUNT_IMPUTE=4.0 (global median) is used
    unchanged for all seasons without raw data.

    This function is informational only — it does NOT modify PITCH_COUNT_IMPUTE
    unless real pitch-level data (median > 2.0) is detected.
    """
    global PITCH_COUNT_IMPUTE

    if "pitch_count_in_pa" not in silver.columns or "season" not in silver.columns:
        return

    raw_rows = silver.filter(pl.col("season").is_in(list(RAW_SEASONS)))
    if raw_rows.is_empty():
        return

    median_pc = raw_rows["pitch_count_in_pa"].drop_nulls().median()
    if median_pc is None:
        return

    if median_pc < 2.0:
        print(f"  [INFO] Raw Statcast parquets son PA-level (median pitch_count={median_pc:.1f}). "
              f"Usando PITCH_COUNT_IMPUTE={PITCH_COUNT_IMPUTE} para temporadas sin raw data.")
    else:
        # Real pitch-sequence data available — update global impute
        PITCH_COUNT_IMPUTE = round(float(median_pc), 2)
        print(f"  Pitch count impute actualizado desde datos reales: {PITCH_COUNT_IMPUTE}")


def augment_silver(silver: pl.DataFrame) -> pl.DataFrame:
    """Add is_gidp and pitch_count_in_pa to the Silver DataFrame.

    Preferred path (new Silver schema with game_pk/at_bat_number): is_gidp
    comes directly from pa_result and pitch_count_in_pa is already exact —
    no fragile pa_seq row-order join needed.

    Legacy path (old Silver): raw Statcast pa_seq join for 2021-2024,
    PITCH_COUNT_IMPUTE for the rest.
    """
    t0 = time.time()

    # ── Preferred: new Silver schema (deterministic, no pa_seq join) ─────────
    if "pitch_count_in_pa" in silver.columns and "game_pk" in silver.columns:
        silver = silver.with_columns([
            pl.col("pa_result").is_in(list(GIDP_EVENTS)).alias("is_gidp"),
            pl.col("pitch_count_in_pa")
              .fill_null(PITCH_COUNT_IMPUTE).cast(pl.Float32)
              .alias("pitch_count_in_pa"),
        ])
        print(f"Augmentation (schema nuevo, sin pa_seq)  "
              f"GIDP rows: {silver['is_gidp'].sum():,}  [{time.time()-t0:.1f}s]")
        return silver

    # Add pa_seq (0-based) within each (batter_id, pitcher_id, game_date) group.
    # We use row_index + group-min subtraction — robust across all Polars versions.
    silver = silver.sort(["batter_id", "pitcher_id", "game_date"]).with_row_index("_gidx")
    silver = silver.with_columns(
        (pl.col("_gidx") - pl.col("_gidx").min().over(["batter_id", "pitcher_id", "game_date"]))
        .cast(pl.Int32).alias("pa_seq")
    ).drop("_gidx")

    # Collect raw augmentation for all RAW_SEASONS into a single lookup table
    raw_frames = []
    for season in sorted(RAW_SEASONS):
        raw_pa = build_raw_pa_table(season)
        if not raw_pa.is_empty():
            raw_pa = raw_pa.with_columns(pl.lit(season).cast(pl.Int32).alias("season"))
            raw_frames.append(raw_pa)

    if raw_frames:
        raw_all = pl.concat(raw_frames, how="diagonal")
        # Join Silver with raw by (batter_id, pitcher_id, game_date, pa_seq, season)
        silver = silver.join(
            raw_all.select([
                pl.col("batter").alias("batter_id"),
                pl.col("pitcher").alias("pitcher_id"),
                pl.col("game_date_str").alias("game_date"),
                "pa_seq", "season",
                "is_gidp_raw", "pitch_count_raw",
            ]),
            on=["batter_id", "pitcher_id", "game_date", "pa_seq", "season"],
            how="left",
            coalesce=True,
        )
    else:
        # No raw data at all — add empty columns
        silver = silver.with_columns([
            pl.lit(None).cast(pl.Boolean).alias("is_gidp_raw"),
            pl.lit(None).cast(pl.Int32).alias("pitch_count_raw"),
        ])

    # Materialise is_gidp: raw Statcast join takes priority; fall back to
    # pa_result string for seasons without raw Statcast (2015-2020, 2025-2026).
    gidp_events_set = list(GIDP_EVENTS)
    silver = silver.with_columns([
        pl.when(pl.col("is_gidp_raw").is_not_null())
            .then(pl.col("is_gidp_raw"))
            .otherwise(
                pl.col("pa_result").is_in(gidp_events_set)
            )
            .alias("is_gidp"),
        # pitch_count season-specific: usa _PC_IMPUTE_BY_SEASON si disponible,
        # cae al global PITCH_COUNT_IMPUTE (calculado o fallback 4.0) si no.
        pl.col("pitch_count_raw").fill_null(PITCH_COUNT_IMPUTE).cast(pl.Float32).alias("pitch_count_in_pa"),
    ]).drop(["is_gidp_raw", "pitch_count_raw"])

    print(f"Augmentation done  GIDP rows: {silver['is_gidp'].sum():,}  [{time.time()-t0:.1f}s]")
    return silver


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Remap labels to 8 classes
# ──────────────────────────────────────────────────────────────────────────────

def remap_labels(silver: pl.DataFrame) -> pl.DataFrame:
    """Build pa_outcome_idx (0-7) from Silver pa_outcome_int + GIDP flags.

    Remapping rules:
      pa_outcome_int → pa_outcome_idx (default: identity 0-6)
      0 (field_out)  → 7  when is_gidp=True
      1 (strikeout)  → 7  when pa_result == 'strikeout_double_play'
    """
    silver = silver.with_columns([
        pl.when(
            (pl.col("pa_outcome_int") == 0) & pl.col("is_gidp")
        )
        .then(pl.lit(7))
        .when(
            pl.col("pa_result") == "strikeout_double_play"
        )
        .then(pl.lit(7))
        .otherwise(pl.col("pa_outcome_int"))
        .cast(pl.Int32)
        .alias("pa_outcome_idx"),
    ])

    # Print label distribution
    dist = (
        silver.group_by("pa_outcome_idx")
        .agg(pl.len().alias("count"))
        .sort("pa_outcome_idx")
    )
    print("Label distribution (8 classes):")
    print(dist)
    return silver


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Batter rolling features (7d / 15d / 30d + EWMA)
# Implementación ÚNICA en src/features/shared_features.py (contrato anti-skew
# con el serving — predict_tonight usa exactamente las mismas funciones).
# ──────────────────────────────────────────────────────────────────────────────

from src.features.shared_features import (  # noqa: E402
    PITCHER_FEATURE_COLS,
    PLATOON_FEATURE_COLS,
    ROLLING_FEATURE_COLS,
    STAB_FEATURE_COLS,
    add_event_flags,
    add_season_stabilized,
    add_shift_rolling,
    daily_grain,
    pitcher_rolling_table,
    platoon_career_table,
)


def compute_rolling_features(silver: pl.DataFrame) -> pl.DataFrame:
    """Compute per-(batter_id, game_date) rolling features and join back to Silver."""
    t0 = time.time()
    print("Computing rolling features...")

    daily = add_shift_rolling(daily_grain(add_event_flags(silver)))
    print(f"  Daily rolling: {len(daily):,} (batter, date) rows")

    rolling = daily.select(["batter_id", "game_date"] + ROLLING_FEATURE_COLS)

    # Join back to Silver (many-to-one: all PAs in same game get same rolling feats)
    result = silver.join(rolling, on=["batter_id", "game_date"], how="left")
    print(f"Rolling features joined [{time.time()-t0:.1f}s]")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — Season-level stabilized statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_stabilized_stats(silver: pl.DataFrame) -> pl.DataFrame:
    """Season-to-date stabilized stats (James-Stein) — shared_features.

    Anti-leakage: cum_sum con shift(1) dentro de (batter_id, season); ISO ahora
    es (TB - H) / PA, idéntico en training y serving.
    """
    t0 = time.time()
    print("Computing stabilized stats...")
    daily = add_season_stabilized(daily_grain(add_event_flags(silver)))
    stabilized = daily.select(["batter_id", "game_date"] + STAB_FEATURE_COLS)
    print(f"  Stabilized: {len(stabilized):,} rows [{time.time()-t0:.1f}s]")
    return stabilized


def compute_platoon_stats(silver: pl.DataFrame) -> pl.DataFrame:
    """Career-to-date vs mano del lanzador, James-Stein — shared_features.

    Una fila por (batter_id, pitcher_throws, game_date): clave única, el join
    de vuelta a PA-level no multiplica filas.
    """
    t0 = time.time()
    print("Computing platoon (vs hand) stats...")
    platoon = platoon_career_table(add_event_flags(silver))
    print(f"  Platoon: {len(platoon):,} rows [{time.time()-t0:.1f}s]")
    return platoon


# ──────────────────────────────────────────────────────────────────────────────
# Step 6b — Pitcher rolling FIP features (shared_features, shrinkage gradual)
# ──────────────────────────────────────────────────────────────────────────────

def compute_pitcher_fip_features(silver: pl.DataFrame, n_games: int = 30) -> pl.DataFrame:
    """Rolling FIP y rates por (pitcher_id, game_date) — shared_features.

    Anti-leakage shift(1); shrinkage gradual hacia liga (w = PA/FIP_MIN_PA)
    en vez del acantilado binario anterior. Misma función en serving.
    """
    t0 = time.time()
    print("Computing pitcher FIP features...")
    result = pitcher_rolling_table(silver, n_games=n_games)
    print(f"  Pitcher FIP: {len(result):,} (pitcher, date) rows  [{time.time()-t0:.1f}s]")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Step 7 — Era encoding + categorical encoding
# ──────────────────────────────────────────────────────────────────────────────

# Fixed pitch-type encoding (stable across all seasons)
_PITCH_TYPE_MAP: dict[str, int] = {
    "FF": 1, "SI": 2, "FC": 3, "SL": 4, "CH": 5,
    "CU": 6, "KC": 7, "FS": 8, "CS": 9, "ST": 10,
    "SV": 11, "EP": 12, "FO": 13, "IN": 14, "PO": 15,
    "SC": 16, "KN": 17, "FA": 18, "UN": 19,
}


def add_era_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("season") >= 2023).cast(pl.Int8).alias("era_shift_ban"),
        (pl.col("season") >= 2020).cast(pl.Int8).alias("era_universal_dh"),
        (pl.col("season") == 2023).cast(pl.Int8).alias("era_first_year_shift_ban"),
    ])


def add_park_features(df: pl.DataFrame) -> pl.DataFrame:
    """Park factors + home/away como features de ENTRENAMIENTO.

    El modelo aprende las interacciones (p.ej. Coors × fly-ball hitter) en vez
    de recibir un multiplicador post-hoc no calibrado en serving.

    Silver sin contexto (schema viejo): neutral hr/xb=1.0, is_home=0.5.
    """
    from src.constants import PARK_FACTORS_BY_TEAM

    if "home_team" in df.columns:
        hr_map = {t: f["hr"] for t, f in PARK_FACTORS_BY_TEAM.items()}
        xb_map = {t: f["xb"] for t, f in PARK_FACTORS_BY_TEAM.items()}
        # .replace con return_dtype + default (API Polars 0.20.x)
        df = df.with_columns([
            pl.col("home_team")
              .replace(hr_map, default=1.0, return_dtype=pl.Float32)
              .alias("park_factor_hr"),
            pl.col("home_team")
              .replace(xb_map, default=1.0, return_dtype=pl.Float32)
              .alias("park_factor_xb"),
        ])
    else:
        df = df.with_columns([
            pl.lit(1.0).cast(pl.Float32).alias("park_factor_hr"),
            pl.lit(1.0).cast(pl.Float32).alias("park_factor_xb"),
        ])

    if "is_home" in df.columns:
        df = df.with_columns(
            pl.col("is_home").cast(pl.Float32).fill_null(0.5).alias("is_home")
        )
    else:
        df = df.with_columns(pl.lit(0.5).cast(pl.Float32).alias("is_home"))
    return df


def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    """Encode categorical string columns to integers for LightGBM compatibility.

    batter_stand  : L→0, R→1  (Int8, unknown→0)
    pitcher_throws: L→0, R→1  (Int8, unknown→0)
    last_pitch_type: fixed vocabulary → 1..19, unknown→0  (Int16)
    """
    # Batter stand: L=0, R=1
    df = df.with_columns(
        pl.col("batter_stand")
        .map_elements(lambda x: 1 if x == "R" else 0, return_dtype=pl.Int8)
        .alias("batter_stand_r")
    )

    # Pitcher throws: L=0, R=1
    df = df.with_columns(
        pl.col("pitcher_throws")
        .map_elements(lambda x: 1 if x == "R" else 0, return_dtype=pl.Int8)
        .alias("pitcher_throws_r")
    )

    # Last pitch type: fixed map
    df = df.with_columns(
        pl.col("last_pitch_type")
        .map_elements(lambda x: _PITCH_TYPE_MAP.get(x, 0) if x else 0,
                      return_dtype=pl.Int16)
        .alias("last_pitch_type_enc")
    )

    # Drop original string columns, keep encoded versions
    df = df.drop(["batter_stand", "pitcher_throws", "last_pitch_type"])
    df = df.rename({
        "batter_stand_r": "batter_stand",
        "pitcher_throws_r": "pitcher_throws",
        "last_pitch_type_enc": "last_pitch_type",
    })
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build(output_path: Path) -> None:
    t_total = time.time()

    # ── 1. Load Silver ──────────────────────────────────────────────────────
    print("\n=== Step 1: Load Silver ===")
    silver = load_silver_all()

    # ── 2/3. Augment with raw (GIDP + pitch_count) ──────────────────────────
    print("\n=== Step 2: Augment with raw Statcast ===")
    silver = augment_silver(silver)
    compute_pitch_count_imputes(silver)

    # ── 4. Remap labels to 8 classes ────────────────────────────────────────
    print("\n=== Step 3: Remap labels (8 classes) ===")
    silver = remap_labels(silver)

    # ── 5. Batter rolling features ──────────────────────────────────────────
    print("\n=== Step 4: Rolling features ===")
    silver = compute_rolling_features(silver)

    # ── 6. Stabilized season-to-date stats ─────────────────────────────────
    print("\n=== Step 5: Stabilized stats ===")
    stab = compute_stabilized_stats(silver)
    silver = silver.join(stab, on=["batter_id", "game_date"], how="left")

    # ── 6a. Platoon career-to-date vs mano del lanzador ─────────────────────
    print("\n=== Step 5b: Platoon (vs hand) stats ===")
    platoon = compute_platoon_stats(silver)
    silver = silver.join(
        platoon, on=["batter_id", "pitcher_throws", "game_date"], how="left"
    )

    # ── 6b. Pitcher FIP features ────────────────────────────────────────────
    print("\n=== Step 6b: Pitcher FIP features ===")
    from src.constants import FIP_NEUTRAL, LEAGUE_AVG
    pitcher_fip_df = compute_pitcher_fip_features(silver)
    silver = silver.join(pitcher_fip_df, on=["pitcher_id", "game_date"], how="left")
    # Fill nulls for pitchers with no prior Silver history (new pitchers, first game)
    silver = silver.with_columns([
        pl.col("pitcher_fip").fill_null(FIP_NEUTRAL).cast(pl.Float32),
        pl.col("pitcher_k_rate").fill_null(LEAGUE_AVG["k_rate"]).cast(pl.Float32),
        pl.col("pitcher_bb_rate").fill_null(LEAGUE_AVG["bb_rate"]).cast(pl.Float32),
        pl.col("pitcher_hr_rate").fill_null(0.033).cast(pl.Float32),
    ])

    # ── 7. Era + park features + categorical encoding ───────────────────────
    print("\n=== Step 6: Era + park features + categorical encoding ===")
    silver = add_era_features(silver)
    silver = add_park_features(silver)
    silver = encode_categoricals(silver)

    # ── 8. Select final columns ─────────────────────────────────────────────
    print("\n=== Step 7: Finalise output ===")

    # Core identifiers + label + raw PA stats
    id_cols  = ["batter_id", "pitcher_id", "game_date", "season"]
    raw_cols = ["xwoba", "launch_speed", "launch_angle",
                "batter_stand", "pitcher_throws", "last_pitch_type",
                "pitch_count_in_pa"]
    label_col = ["pa_outcome_idx"]
    era_cols  = ["era_shift_ban", "era_universal_dh", "era_first_year_shift_ban"]

    rolling_cols = [c for c in silver.columns if any(
        c.endswith(s) for s in ["_7d", "_15d", "_30d"]
    ) or c.startswith("xwoba_ewma")]

    stab_cols = [c for c in silver.columns if
                 "_stabilized" in c or "_shrinkage_b" in c]

    platoon_extra = [c for c in ("pa_vs_hand",) if c in silver.columns]
    park_cols = ["park_factor_hr", "park_factor_xb", "is_home"]
    fip_cols  = ["pitcher_fip", "pitcher_k_rate", "pitcher_bb_rate", "pitcher_hr_rate"]

    final_cols = (id_cols + label_col + raw_cols + rolling_cols + stab_cols
                  + platoon_extra + park_cols + era_cols + fip_cols)

    # Keep only columns that actually exist in the DataFrame
    final_cols = [c for c in final_cols if c in silver.columns]

    gold = silver.select(final_cols)

    # Cast floating features to Float32 to save space
    float_cols = [c for c in gold.columns
                  if gold[c].dtype in (pl.Float64,) and c not in id_cols + label_col]
    if float_cols:
        gold = gold.with_columns(
            [pl.col(c).cast(pl.Float32) for c in float_cols]
        )

    # Fill remaining nulls in float columns with 0.0 (LightGBM can handle nulls
    # natively, but 0 is a safe fallback for new features)
    # Note: keep nulls for rolling features since LightGBM handles them well.

    print(f"\nFinal shape: {gold.shape}")
    print(f"Columns: {gold.columns}")
    print(f"\nLabel distribution:")
    print(gold.group_by("pa_outcome_idx").agg(pl.len().alias("count")).sort("pa_outcome_idx"))

    # ── 9. Save ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gold.write_parquet(str(output_path), compression="zstd")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Total time: {time.time()-t_total:.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Build Gold v3 training features")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Output parquet file path",
    )
    args = parser.parse_args()

    build(output_path=args.output_path)
