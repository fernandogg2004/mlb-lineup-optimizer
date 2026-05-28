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

# ─── Label constants ───────────────────────────────────────────────────────────
GIDP_EVENTS = {"grounded_into_double_play", "strikeout_double_play", "double_play"}

# Seasons where raw Statcast is available (for GIDP + pitch_count)
RAW_SEASONS = {2021, 2022, 2023, 2024}

# Default pitch count imputation for seasons without raw data
PITCH_COUNT_IMPUTE = 4.0

# Shrinkage (Marcel-style) stabilization thresholds (PA at 50% signal)
STABILIZE_T = {
    "woba":  200,
    "k":      60,
    "bb":    120,
    "babip": 500,
    "iso":   160,
}

# League-average priors (FanGraphs 2024 reference)
LEAGUE_PRIORS = {
    "woba":  0.318,
    "k":     0.229,
    "bb":    0.087,
    "babip": 0.295,
    "iso":   0.166,
}


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Load Silver
# ──────────────────────────────────────────────────────────────────────────────

def load_silver_all() -> pl.DataFrame:
    """Load all Silver plate_appearances (2015-2024) into one DataFrame."""
    t0 = time.time()
    frames = []
    for season in range(2015, 2025):
        path = SILVER_ROOT / f"season={season}" / "data.parquet"
        if not path.exists():
            print(f"  [WARN] Silver {season} not found, skipping")
            continue
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

def augment_silver(silver: pl.DataFrame) -> pl.DataFrame:
    """Add is_gidp and pitch_count_in_pa to the Silver DataFrame.

    For 2021-2024: uses raw Statcast (exact PA-sequence join).
    For 2015-2020: is_gidp = False, pitch_count = PITCH_COUNT_IMPUTE.
    """
    t0 = time.time()

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

    # Materialise is_gidp and pitch_count_in_pa with defaults for missing rows
    silver = silver.with_columns([
        pl.col("is_gidp_raw").fill_null(False).alias("is_gidp"),
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
# ──────────────────────────────────────────────────────────────────────────────

def _add_event_flags(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.lit(1).cast(pl.Int8).alias("pa_flag"),
        pl.col("pa_result").is_in(["strikeout", "strikeout_double_play"]).cast(pl.Int8).alias("k_flag"),
        pl.col("pa_result").is_in(["walk", "intent_walk"]).cast(pl.Int8).alias("bb_flag"),
        (pl.col("hit_type") == "home_run").cast(pl.Int8).alias("hr_flag"),
        pl.col("hit_type").is_in(["single", "double", "triple", "home_run"]).cast(pl.Int8).alias("hit_flag"),
        (pl.col("launch_speed").is_not_null() & (pl.col("launch_speed") >= 95.0)).cast(pl.Int8).alias("hh_flag"),
        (~pl.col("pa_result").is_in([
            "walk", "intent_walk", "hit_by_pitch", "strikeout",
            "strikeout_double_play", "sac_fly", "sac_bunt",
        ])).cast(pl.Int8).alias("bip_flag"),
    ])


def _daily_grain(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-PA data to (batter_id, game_date) daily grain."""
    return (
        df.group_by(["batter_id", "game_date", "season"])
        .agg([
            pl.col("pa_flag").sum().alias("pa"),
            pl.col("k_flag").sum().alias("k"),
            pl.col("bb_flag").sum().alias("bb"),
            pl.col("hr_flag").sum().alias("hr"),
            pl.col("hit_flag").sum().alias("hits"),
            pl.col("hh_flag").sum().alias("hh"),
            pl.col("bip_flag").sum().alias("bip"),
            pl.col("xwoba").drop_nulls().mean().alias("xwoba_mean"),
            pl.col("launch_speed").drop_nulls().mean().alias("ls_mean"),
        ])
        .sort(["batter_id", "game_date"])
    )


def _shift_rolling(daily: pl.DataFrame, window: int, min_pa: int, suffix: str) -> pl.DataFrame:
    """Game-count rolling window with anti-leakage shift(1)."""
    cols_to_shift = ["xwoba_mean", "ls_mean", "pa", "k", "bb", "hr", "hits", "hh", "bip"]

    daily = daily.sort(["batter_id", "game_date"])

    # Shift all columns by 1 within each batter group
    shifts = [
        pl.col(c).shift(1).over("batter_id").alias(f"_{c}_s")
        for c in cols_to_shift
    ]
    daily = daily.with_columns(shifts)

    # Rolling sum/mean over shifted values
    min_p = max(1, min_pa)
    daily = daily.with_columns([
        pl.col("_pa_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"pa_{suffix}"),
        pl.col("_k_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"_k_{suffix}"),
        pl.col("_bb_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"_bb_{suffix}"),
        pl.col("_hr_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"_hr_{suffix}"),
        pl.col("_hits_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"_hits_{suffix}"),
        pl.col("_hh_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"_hh_{suffix}"),
        pl.col("_bip_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"_bip_{suffix}"),
        pl.col("_xwoba_mean_s").rolling_mean(window, min_periods=1).over("batter_id").alias(f"_xwoba_raw_{suffix}"),
        pl.col("_ls_mean_s").rolling_mean(window, min_periods=1).over("batter_id").alias(f"_ls_raw_{suffix}"),
    ])

    # Apply PA gate + compute rates
    gate = pl.col(f"pa_{suffix}") >= min_p
    daily = daily.with_columns([
        pl.when(gate).then(pl.col(f"_xwoba_raw_{suffix}")).otherwise(None).alias(f"xwoba_{suffix}"),
        pl.when(gate).then(pl.col(f"_ls_raw_{suffix}")).otherwise(None).alias(f"launch_speed_{suffix}"),
        pl.when(gate).then(
            pl.col(f"_k_{suffix}") / pl.col(f"pa_{suffix}").clip(lower_bound=1)
        ).otherwise(None).alias(f"k_rate_{suffix}"),
        pl.when(gate).then(
            pl.col(f"_bb_{suffix}") / pl.col(f"pa_{suffix}").clip(lower_bound=1)
        ).otherwise(None).alias(f"bb_rate_{suffix}"),
        pl.when(gate).then(
            pl.col(f"_hr_{suffix}") / pl.col(f"pa_{suffix}").clip(lower_bound=1)
        ).otherwise(None).alias(f"hr_rate_{suffix}"),
        pl.when(gate).then(
            pl.col(f"_hh_{suffix}") / pl.col(f"_bip_{suffix}").clip(lower_bound=1)
        ).otherwise(None).alias(f"hard_hit_rate_{suffix}"),
    ])

    # Drop temp columns
    drop_cols = [f"_{c}_s" for c in cols_to_shift] + [
        f"_k_{suffix}", f"_bb_{suffix}", f"_hr_{suffix}",
        f"_hits_{suffix}", f"_hh_{suffix}", f"_bip_{suffix}",
        f"_xwoba_raw_{suffix}", f"_ls_raw_{suffix}",
    ]
    return daily.drop([c for c in drop_cols if c in daily.columns])


def compute_rolling_features(silver: pl.DataFrame) -> pl.DataFrame:
    """Compute per-(batter_id, game_date) rolling features and join back to Silver."""
    t0 = time.time()
    print("Computing rolling features...")

    flagged = _add_event_flags(silver)
    daily = _daily_grain(flagged)

    # 7d window (≈ 7 games, short window)
    daily = _shift_rolling(daily, window=7,  min_pa=3, suffix="7d")
    # 15d window
    daily = _shift_rolling(daily, window=15, min_pa=3, suffix="15d")
    # 30d window
    daily = _shift_rolling(daily, window=30, min_pa=10, suffix="30d")

    # EWMA (slow α=0.2, fast α=0.5) on xwoba
    daily = daily.with_columns([
        pl.col("xwoba_mean").shift(1).ewm_mean(alpha=0.2, min_periods=1, adjust=True)
            .over("batter_id").alias("xwoba_ewma_alpha02"),
        pl.col("xwoba_mean").shift(1).ewm_mean(alpha=0.5, min_periods=1, adjust=True)
            .over("batter_id").alias("xwoba_ewma_alpha05"),
    ])

    print(f"  Daily rolling: {len(daily):,} (batter, date) rows")

    # Select only the rolling feature columns for the join
    rolling_cols = [
        "pa_7d",  "k_rate_7d",  "bb_rate_7d",  "hr_rate_7d",  "hard_hit_rate_7d",  "xwoba_7d",  "launch_speed_7d",
        "pa_15d", "k_rate_15d", "bb_rate_15d", "hr_rate_15d", "hard_hit_rate_15d", "xwoba_15d", "launch_speed_15d",
        "pa_30d", "k_rate_30d", "bb_rate_30d", "hr_rate_30d", "hard_hit_rate_30d", "xwoba_30d", "launch_speed_30d",
        "xwoba_ewma_alpha02", "xwoba_ewma_alpha05",
    ]
    rolling = daily.select(["batter_id", "game_date"] + rolling_cols)

    # Join back to Silver (many-to-one: all PAs in same game get same rolling feats)
    result = silver.join(rolling, on=["batter_id", "game_date"], how="left")
    print(f"Rolling features joined [{time.time()-t0:.1f}s]")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — Season-level stabilized statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_stabilized_stats(silver_with_flags: pl.DataFrame) -> pl.DataFrame:
    """Compute season-to-date stabilized stats using Empirical Bayes shrinkage.

    For each batter on each game_date, we use all PAs in the current season
    strictly before game_date to compute cumulative raw rate, then shrink
    toward the league average using the Marcel stabilization threshold.

    stat_stabilized = prior + (1 - B) × (raw - prior)
    B = T / (T + PA_to_date)

    This is intentionally conservative (uses season-to-date, not career)
    to avoid using future data.
    """
    t0 = time.time()
    print("Computing stabilized stats...")

    # We need to add event flags if not already present
    if "k_flag" not in silver_with_flags.columns:
        df = _add_event_flags(silver_with_flags)
    else:
        df = silver_with_flags

    daily = _daily_grain(df)

    # Season-to-date cumulative counts (anti-leakage: shifted by 1)
    daily = daily.sort(["batter_id", "season", "game_date"])

    daily = daily.with_columns([
        pl.col("pa").shift(1, fill_value=0).cum_sum().over(["batter_id", "season"]).alias("pa_std"),
        pl.col("k").shift(1, fill_value=0).cum_sum().over(["batter_id", "season"]).alias("k_std"),
        pl.col("bb").shift(1, fill_value=0).cum_sum().over(["batter_id", "season"]).alias("bb_std"),
        pl.col("hr").shift(1, fill_value=0).cum_sum().over(["batter_id", "season"]).alias("hr_std"),
        pl.col("hits").shift(1, fill_value=0).cum_sum().over(["batter_id", "season"]).alias("hits_std"),
        pl.col("bip").shift(1, fill_value=0).cum_sum().over(["batter_id", "season"]).alias("bip_std"),
        pl.col("xwoba_mean").shift(1, fill_value=None)
            .rolling_mean(window_size=200, min_periods=1)
            .over(["batter_id", "season"]).alias("woba_raw"),
    ])

    # Compute stabilized stats and shrinkage B values
    def _stabilize(raw_col: str, pa_col: str, T: float, prior: float, out: str) -> list:
        B = pl.lit(T) / (pl.lit(T) + pl.col(pa_col).clip(lower_bound=0))
        return [
            B.alias(f"{out}_shrinkage_b"),
            (
                pl.lit(prior) + (1 - B) * (pl.col(raw_col) - pl.lit(prior))
            ).alias(f"{out}_stabilized"),
        ]

    rate_cols = []

    # wOBA stabilized
    rate_cols += _stabilize("woba_raw", "pa_std", STABILIZE_T["woba"],
                            LEAGUE_PRIORS["woba"], "woba")

    # K rate stabilized
    daily = daily.with_columns(
        (pl.col("k_std") / pl.col("pa_std").clip(lower_bound=1)).alias("k_rate_raw")
    )
    rate_cols += _stabilize("k_rate_raw", "pa_std", STABILIZE_T["k"],
                            LEAGUE_PRIORS["k"], "k_rate")

    # BB rate stabilized
    daily = daily.with_columns(
        (pl.col("bb_std") / pl.col("pa_std").clip(lower_bound=1)).alias("bb_rate_raw")
    )
    rate_cols += _stabilize("bb_rate_raw", "pa_std", STABILIZE_T["bb"],
                            LEAGUE_PRIORS["bb"], "bb_rate")

    # BABIP stabilized
    daily = daily.with_columns(
        ((pl.col("hits_std") - pl.col("hr_std")) / pl.col("bip_std").clip(lower_bound=1)).alias("babip_raw")
    )
    rate_cols += _stabilize("babip_raw", "pa_std", STABILIZE_T["babip"],
                            LEAGUE_PRIORS["babip"], "babip")

    # ISO stabilized (approximated from HR rate)
    daily = daily.with_columns(
        (pl.col("hr_std") * 1.5 / pl.col("pa_std").clip(lower_bound=1)).alias("iso_raw")
    )
    rate_cols += _stabilize("iso_raw", "pa_std", STABILIZE_T["iso"],
                            LEAGUE_PRIORS["iso"], "iso")

    daily = daily.with_columns(rate_cols)

    stab_cols = [
        "woba_stabilized", "woba_shrinkage_b",
        "k_rate_stabilized", "k_rate_shrinkage_b",
        "bb_rate_stabilized", "bb_rate_shrinkage_b",
        "babip_stabilized", "babip_shrinkage_b",
        "iso_stabilized", "iso_shrinkage_b",
    ]
    stabilized = daily.select(["batter_id", "game_date"] + stab_cols)
    print(f"  Stabilized: {len(stabilized):,} rows [{time.time()-t0:.1f}s]")
    return stabilized


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

    # ── 7. Era features + categorical encoding ──────────────────────────────
    print("\n=== Step 6: Era features + categorical encoding ===")
    silver = add_era_features(silver)
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

    final_cols = id_cols + label_col + raw_cols + rolling_cols + stab_cols + era_cols

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
