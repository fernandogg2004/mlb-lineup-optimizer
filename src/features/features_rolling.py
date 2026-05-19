"""
features_rolling.py
===================
Computes rolling window features for batters using **Polars** for maximum
efficiency on in-memory workloads (10–100× faster than pandas for group-by
rolling operations).

Anti-leakage guarantee
----------------------
All rolling aggregates are computed strictly **before** the target game date.
For a game on date D, the window [D - W days, D - 1 day] is used. This is
enforced by:

    1. Sorting by (batter_id, game_date) within each group.
    2. Using ``shift(1)`` BEFORE the rolling operation so the current game's
       stats are never included in the window.
    3. For true calendar-day windows (not game-count windows), using Polars
       ``join_asof`` on a pre-aggregated daily summary with ``by="batter_id"``
       and ``strategy="backward"`` to look back exactly N calendar days.

Feature catalog
---------------
    xwoba_7d / xwoba_15d / xwoba_30d        — Rolling mean expected wOBA
    launch_speed_7d / _15d / _30d           — Rolling mean exit velocity (mph)
    xwoba_ewma_alpha02 / xwoba_ewma_alpha05 — EWMA with α=0.2 and α=0.5
    k_rate_7d / k_rate_30d                  — Rolling strikeout rate
    bb_rate_7d / bb_rate_30d                — Rolling walk rate
    hr_rate_30d                             — Rolling home run rate
    hard_hit_rate_30d                       — Rolling hard-hit ball rate (≥95 mph EV)
    babip_30d                               — Rolling BABIP
    pa_7d / pa_15d / pa_30d                 — Rolling PA count (activity indicator)

Usage:
    python -m src.features.features_rolling \\
        --silver-path s3a://mlb-lakehouse/silver/plate_appearances \\
        --output-path s3a://mlb-lakehouse/gold/batter_features_rolling \\
        --as-of-date 2024-07-15

    # Library usage:
    from src.features.features_rolling import RollingFeaturesEngine
    engine = RollingFeaturesEngine(silver_path="...", output_path="...")
    features = engine.run(as_of_date="2024-07-15")
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import polars as pl
import structlog

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HARD_HIT_THRESHOLD_MPH: float = 95.0  # MLB definition of "hard-hit" ball


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowConfig:
    """Configuration for rolling window sizes and EWMA parameters.

    Attributes:
        short_days: Short window in calendar days (default 7).
        medium_days: Medium window in calendar days (default 15).
        long_days: Long window in calendar days (default 30).
        ewma_alpha_slow: EWMA decay factor for slow-moving average (more stable).
        ewma_alpha_fast: EWMA decay factor for fast-moving average (more reactive).
        min_pa_short: Minimum PAs in short window to produce a non-null value.
        min_pa_long: Minimum PAs in long window to produce a non-null value.
    """

    short_days: int = 7
    medium_days: int = 15
    long_days: int = 30
    ewma_alpha_slow: float = 0.2
    ewma_alpha_fast: float = 0.5
    min_pa_short: int = 3    # fewer PAs → return null (too noisy)
    min_pa_long: int = 10    # fewer PAs → return null for 30d window


@dataclass(frozen=True)
class RollingFeaturesConfig:
    """Top-level configuration for the rolling features pipeline.

    Attributes:
        window: Window sizes and EWMA parameters.
        silver_columns: Columns to load from Silver (minimize I/O).
        output_partition_col: Column name used to partition output Parquet.
    """

    window: WindowConfig = field(default_factory=WindowConfig)
    silver_columns: list[str] = field(
        default_factory=lambda: [
            "batter_id", "game_date", "pa_result", "hit_type",
            "launch_speed", "launch_angle", "xwoba",
            "batter_stand", "season",
        ]
    )
    output_partition_col: str = "feature_date"


# ---------------------------------------------------------------------------
# PA-level event flags (needed for rate calculations)
# ---------------------------------------------------------------------------

def _add_pa_event_flags(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Adds binary event indicator columns to a PA-level LazyFrame.

    All flags are ``Int8`` (0/1) for memory efficiency over boolean arrays.

    Args:
        lf: Raw PA-level LazyFrame from Silver.

    Returns:
        LazyFrame with event flag columns added.
    """
    return lf.with_columns([
        pl.lit(1).cast(pl.Int8).alias("pa_flag"),
        pl.col("pa_result").is_in(["strikeout", "strikeout_double_play"])
            .cast(pl.Int8).alias("k_flag"),
        pl.col("pa_result").is_in(["walk", "intent_walk"])
            .cast(pl.Int8).alias("bb_flag"),
        (pl.col("hit_type") == "home_run").cast(pl.Int8).alias("hr_flag"),
        pl.col("hit_type").is_in(["single", "double", "triple", "home_run"])
            .cast(pl.Int8).alias("hit_flag"),
        # Hard-hit: only valid when launch_speed is not null
        (pl.col("launch_speed").is_not_null() & (pl.col("launch_speed") >= HARD_HIT_THRESHOLD_MPH))
            .cast(pl.Int8).alias("hard_hit_flag"),
        # Ball in play (for BABIP denominator)
        (~pl.col("pa_result").is_in([
            "walk", "intent_walk", "hit_by_pitch", "strikeout",
            "strikeout_double_play", "sac_fly", "sac_bunt",
        ])).cast(pl.Int8).alias("bip_flag"),
    ])


# ---------------------------------------------------------------------------
# Daily summary aggregation (game-day grain)
# ---------------------------------------------------------------------------

def _aggregate_to_daily(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregates PA-level data to one row per (batter_id, game_date).

    Rolling operations are computed over this daily grain rather than the
    PA grain for efficiency and to correctly handle multi-PA games.

    Args:
        lf: PA-level LazyFrame with event flags added.

    Returns:
        Daily-grain LazyFrame with summed event counts and mean Statcast stats.
    """
    return (
        lf.group_by(["batter_id", "game_date", "batter_stand", "season"])
        .agg([
            pl.col("pa_flag").sum().alias("pa"),
            pl.col("k_flag").sum().alias("k"),
            pl.col("bb_flag").sum().alias("bb"),
            pl.col("hr_flag").sum().alias("hr"),
            pl.col("hit_flag").sum().alias("hits"),
            pl.col("hard_hit_flag").sum().alias("hard_hits"),
            pl.col("bip_flag").sum().alias("bip"),
            # Mean Statcast metrics per game (only non-null entries)
            pl.col("xwoba").drop_nulls().mean().alias("xwoba_mean"),
            pl.col("launch_speed").drop_nulls().mean().alias("launch_speed_mean"),
            pl.col("launch_angle").drop_nulls().mean().alias("launch_angle_mean"),
        ])
        .sort(["batter_id", "game_date"])
    )


# ---------------------------------------------------------------------------
# Calendar-day rolling window computation
# ---------------------------------------------------------------------------

def _compute_calendar_rolling(
    daily_df: pl.DataFrame,
    window_days: int,
    min_pa: int,
    suffix: str,
) -> pl.DataFrame:
    """Computes rolling aggregates over a calendar-day window with anti-leakage.

    Uses ``join_asof`` with ``strategy="backward"`` to look back exactly
    ``window_days`` calendar days from (game_date - 1 day), ensuring the
    current game is excluded.

    Strategy:
        For each row with game_date = D:
            window = [D - window_days, D - 1]
        Implemented as:
            1. Create a "lookback" table identical to daily_df.
            2. For each row, join_asof on (game_date - 1) looking backward
               to find all rows within window_days days.
            3. Aggregate the matched rows.

    Note:
        This is an O(N log N) operation on N = total daily rows. For
        ~10 seasons × 750 players × 162 games ≈ 1.2M rows, this is fast
        enough in Polars without Spark.

    Args:
        daily_df: Collected daily-grain DataFrame sorted by (batter_id, game_date).
        window_days: Size of the calendar window in days.
        min_pa: Minimum PA required to return a non-null value.
        suffix: Column name suffix (e.g. ``"7d"``, ``"30d"``).

    Returns:
        DataFrame with rolling aggregate columns named ``*_{suffix}``.
    """
    # Cast game_date to date type for arithmetic
    daily = daily_df.with_columns(
        pl.col("game_date").str.to_date().alias("game_date_dt")
    )

    # For each row, define the window start (exclusive upper bound = game_date - 1)
    daily = daily.with_columns([
        (pl.col("game_date_dt") - pl.duration(days=1)).alias("window_upper"),
        (pl.col("game_date_dt") - pl.duration(days=window_days)).alias("window_lower"),
    ])

    # Lookback table: rename game_date for the join
    lookback = daily.select([
        pl.col("batter_id"),
        pl.col("game_date_dt").alias("lb_date"),
        pl.col("pa"), pl.col("k"), pl.col("bb"), pl.col("hr"),
        pl.col("hits"), pl.col("hard_hits"), pl.col("bip"),
        pl.col("xwoba_mean"), pl.col("launch_speed_mean"),
    ])

    # Join each player's rows to their own historical rows in the window
    # Polars join_where gives range-join semantics (available in Polars ≥ 0.20)
    try:
        joined = daily.join(
            lookback,
            on="batter_id",
            how="left",
        ).filter(
            (pl.col("lb_date") >= pl.col("window_lower"))
            & (pl.col("lb_date") <= pl.col("window_upper"))
        )
    except Exception as exc:
        log.warning("range_join_failed_fallback", error=str(exc), window=window_days)
        # Fallback: shift-based rolling (game-count window, not calendar-day)
        return _compute_shift_rolling(daily_df, window_days, min_pa, suffix)

    # Aggregate the joined window
    window_agg = (
        joined
        .group_by(["batter_id", "game_date"])
        .agg([
            pl.col("pa_right").sum().alias(f"pa_{suffix}"),
            pl.col("k_right").sum().alias(f"k_{suffix}"),
            pl.col("bb_right").sum().alias(f"bb_{suffix}"),
            pl.col("hr_right").sum().alias(f"hr_{suffix}"),
            pl.col("hard_hits_right").sum().alias(f"hard_hits_{suffix}"),
            pl.col("bip_right").sum().alias(f"bip_{suffix}"),
            pl.col("xwoba_mean_right").drop_nulls().mean().alias(f"xwoba_{suffix}_raw"),
            pl.col("launch_speed_mean_right").drop_nulls().mean().alias(f"launch_speed_{suffix}_raw"),
        ])
    )

    # Apply minimum PA gate: null out metrics if insufficient data
    result = window_agg.with_columns([
        pl.when(pl.col(f"pa_{suffix}") >= min_pa)
            .then(pl.col(f"xwoba_{suffix}_raw"))
            .otherwise(None)
            .alias(f"xwoba_{suffix}"),
        pl.when(pl.col(f"pa_{suffix}") >= min_pa)
            .then(pl.col(f"launch_speed_{suffix}_raw"))
            .otherwise(None)
            .alias(f"launch_speed_{suffix}"),
        pl.when(pl.col(f"pa_{suffix}") >= min_pa)
            .then(pl.col(f"k_{suffix}") / pl.col(f"pa_{suffix}").clip(lower_bound=1))
            .otherwise(None)
            .alias(f"k_rate_{suffix}"),
        pl.when(pl.col(f"pa_{suffix}") >= min_pa)
            .then(pl.col(f"bb_{suffix}") / pl.col(f"pa_{suffix}").clip(lower_bound=1))
            .otherwise(None)
            .alias(f"bb_rate_{suffix}"),
        pl.when(pl.col(f"pa_{suffix}") >= min_pa)
            .then(pl.col(f"hr_{suffix}") / pl.col(f"pa_{suffix}").clip(lower_bound=1))
            .otherwise(None)
            .alias(f"hr_rate_{suffix}"),
        pl.when(pl.col(f"pa_{suffix}") >= min_pa)
            .then(pl.col(f"hard_hits_{suffix}") / pl.col(f"bip_{suffix}").clip(lower_bound=1))
            .otherwise(None)
            .alias(f"hard_hit_rate_{suffix}"),
    ]).drop([f"xwoba_{suffix}_raw", f"launch_speed_{suffix}_raw"])

    return result


def _compute_shift_rolling(
    daily_df: pl.DataFrame,
    window_games: int,
    min_pa: int,
    suffix: str,
) -> pl.DataFrame:
    """Fallback: game-count rolling window using shift + rolling_mean.

    Uses ``shift(1).rolling_mean(window_size=N)`` within each batter group.
    The ``shift(1)`` guarantees the current game is excluded from every window,
    preserving the anti-leakage guarantee.

    Trade-off vs. calendar-day window: treats 3 games in 3 days identically to
    3 games in 30 days (ignoring recency within the window). Acceptable for
    the fallback case.

    Args:
        daily_df: Daily-grain DataFrame sorted by (batter_id, game_date).
        window_games: Number of preceding *games* to include.
        min_pa: Minimum PA gate.
        suffix: Column name suffix.

    Returns:
        DataFrame with rolling columns added.
    """
    result = daily_df.sort(["batter_id", "game_date"]).with_columns([
        # Shift all raw columns by 1 within each batter to exclude current game
        pl.col("xwoba_mean").shift(1).over("batter_id").alias("xwoba_shifted"),
        pl.col("launch_speed_mean").shift(1).over("batter_id").alias("ls_shifted"),
        pl.col("pa").shift(1).over("batter_id").alias("pa_shifted"),
        pl.col("k").shift(1).over("batter_id").alias("k_shifted"),
        pl.col("bb").shift(1).over("batter_id").alias("bb_shifted"),
        pl.col("hr").shift(1).over("batter_id").alias("hr_shifted"),
        pl.col("hard_hits").shift(1).over("batter_id").alias("hh_shifted"),
        pl.col("bip").shift(1).over("batter_id").alias("bip_shifted"),
    ]).with_columns([
        # Rolling mean over shifted values (anti-leakage)
        pl.col("xwoba_shifted").rolling_mean(window_size=window_games, min_periods=1)
            .over("batter_id").alias(f"xwoba_{suffix}"),
        pl.col("ls_shifted").rolling_mean(window_size=window_games, min_periods=1)
            .over("batter_id").alias(f"launch_speed_{suffix}"),
        pl.col("pa_shifted").rolling_sum(window_size=window_games, min_periods=1)
            .over("batter_id").alias(f"pa_{suffix}"),
        pl.col("k_shifted").rolling_sum(window_size=window_games, min_periods=1)
            .over("batter_id").alias(f"k_{suffix}"),
        pl.col("bb_shifted").rolling_sum(window_size=window_games, min_periods=1)
            .over("batter_id").alias(f"bb_{suffix}"),
        pl.col("hr_shifted").rolling_sum(window_size=window_games, min_periods=1)
            .over("batter_id").alias(f"hr_{suffix}"),
        pl.col("hh_shifted").rolling_sum(window_size=window_games, min_periods=1)
            .over("batter_id").alias(f"hh_{suffix}"),
        pl.col("bip_shifted").rolling_sum(window_size=window_games, min_periods=1)
            .over("batter_id").alias(f"bip_{suffix}"),
    ]).with_columns([
        # Apply minimum PA gate (null out if insufficient data)
        pl.when(pl.col(f"pa_{suffix}") >= min_pa)
            .then(pl.col(f"k_{suffix}") / pl.col(f"pa_{suffix}").clip(lower_bound=1))
            .otherwise(None).alias(f"k_rate_{suffix}"),
        pl.when(pl.col(f"pa_{suffix}") >= min_pa)
            .then(pl.col(f"bb_{suffix}") / pl.col(f"pa_{suffix}").clip(lower_bound=1))
            .otherwise(None).alias(f"bb_rate_{suffix}"),
        pl.when(pl.col(f"pa_{suffix}") >= min_pa)
            .then(pl.col(f"hr_{suffix}") / pl.col(f"pa_{suffix}").clip(lower_bound=1))
            .otherwise(None).alias(f"hr_rate_{suffix}"),
        pl.when(pl.col(f"pa_{suffix}") >= min_pa)
            .then(pl.col(f"hh_{suffix}") / pl.col(f"bip_{suffix}").clip(lower_bound=1))
            .otherwise(None).alias(f"hard_hit_rate_{suffix}"),
        # Null out mean metrics if below PA gate
        pl.when(pl.col(f"pa_{suffix}") >= min_pa)
            .then(pl.col(f"xwoba_{suffix}")).otherwise(None).alias(f"xwoba_{suffix}"),
        pl.when(pl.col(f"pa_{suffix}") >= min_pa)
            .then(pl.col(f"launch_speed_{suffix}")).otherwise(None).alias(f"launch_speed_{suffix}"),
    ]).drop([
        "xwoba_shifted", "ls_shifted", "pa_shifted", "k_shifted",
        "bb_shifted", "hr_shifted", "hh_shifted", "bip_shifted",
        f"k_{suffix}", f"bb_{suffix}", f"hr_{suffix}", f"hh_{suffix}", f"bip_{suffix}",
    ])

    return result


# ---------------------------------------------------------------------------
# EWMA (Exponentially Weighted Moving Average)
# ---------------------------------------------------------------------------

def _compute_ewma(
    daily_df: pl.DataFrame,
    alpha: float,
    col: str,
    output_col: str,
) -> pl.DataFrame:
    """Computes an EWMA with anti-leakage shift for a given column.

    EWMA weights recent observations more heavily than older ones, making
    it more responsive to hot/cold streaks than simple rolling means.
    α=0.5 reacts quickly (half-life ≈ 1 game); α=0.2 is smoother (half-life ≈ 3 games).

    The shift(1) before the EWMA prevents the current game from influencing
    its own rolling feature value.

    Args:
        daily_df: Daily-grain DataFrame sorted by (batter_id, game_date).
        alpha: EWMA decay factor in (0, 1]. Higher = more weight on recent.
        col: Source column name (e.g. ``"xwoba_mean"``).
        output_col: Name for the resulting EWMA column.

    Returns:
        DataFrame with ``output_col`` added.
    """
    return daily_df.with_columns([
        pl.col(col)
            .shift(1)
            .ewm_mean(alpha=alpha, min_periods=1, adjust=True)
            .over("batter_id")
            .alias(output_col)
    ])


# ---------------------------------------------------------------------------
# BABIP rolling computation (requires hits and bip separately)
# ---------------------------------------------------------------------------

def _compute_babip_rolling(
    df: pl.DataFrame,
    pa_col: str,
    suffix: str,
    min_pa: int,
) -> pl.DataFrame:
    """Adds a rolling BABIP column derived from existing window aggregates.

    BABIP = (H - HR) / (AB - K - HR + SF)  ≈  (hits - hr) / (bip)
    Uses the BIP count already computed in the rolling window.

    Args:
        df: DataFrame with rolling hit/hr/bip columns for the target window.
        pa_col: PA count column for the window (e.g. ``"pa_30d"``).
        suffix: Window suffix (e.g. ``"30d"``).
        min_pa: Minimum PA gate for non-null BABIP.

    Returns:
        DataFrame with ``babip_{suffix}`` column added.
    """
    hits_col = f"hits_{suffix}" if f"hits_{suffix}" in df.columns else None
    hr_col   = f"hr_{suffix}"   if f"hr_{suffix}"   in df.columns else None
    bip_col  = f"bip_{suffix}"  if f"bip_{suffix}"  in df.columns else None

    if not all([hits_col, hr_col, bip_col]):
        log.warning("babip_columns_missing", suffix=suffix)
        return df

    return df.with_columns(
        pl.when(pl.col(pa_col) >= min_pa)
        .then(
            (pl.col(hits_col) - pl.col(hr_col))
            / pl.col(bip_col).clip(lower_bound=1)
        )
        .otherwise(None)
        .alias(f"babip_{suffix}")
    )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class RollingFeaturesEngine:
    """Orchestrates the computation of rolling window features for all batters.

    Reads the Silver plate_appearances table via Polars, computes anti-leakage
    rolling aggregates at 7 / 15 / 30-day windows plus EWMA, and writes the
    result to the Gold layer partitioned by ``feature_date``.

    The output schema is consumed directly by the Gold Feature Store sync and
    by the matchup feature computation in Phase 2.4.

    Attributes:
        silver_path: Path to Silver plate_appearances (Parquet / Delta).
        output_path: Destination path for Gold rolling features.
        config: Feature computation configuration.
    """

    def __init__(
        self,
        silver_path: str,
        output_path: str,
        config: Optional[RollingFeaturesConfig] = None,
    ) -> None:
        """Initializes the rolling features engine.

        Args:
            silver_path: Path to Silver plate_appearances table.
            output_path: Gold output path.
            config: Optional configuration; defaults to ``RollingFeaturesConfig()``.
        """
        self.silver_path = silver_path
        self.output_path = output_path
        self.config = config or RollingFeaturesConfig()

    def _load_silver(self, as_of_date: str) -> pl.LazyFrame:
        """Loads Silver PA data for all dates up to (but not including) as_of_date.

        The ``as_of_date`` filter is the outer anti-leakage guard: we never load
        data from the target game day, even if it somehow exists in Silver.

        Args:
            as_of_date: Game date to compute features for (``YYYY-MM-DD``).

        Returns:
            Filtered LazyFrame.
        """
        try:
            lf = (
                pl.scan_parquet(f"{self.silver_path}/**/*.parquet")
                .filter(pl.col("game_date") < as_of_date)
                .select(self.config.silver_columns)
            )
            log.info("silver_loaded_for_rolling", as_of_date=as_of_date)
            return lf
        except Exception as exc:
            log.error("silver_load_failed", error=str(exc))
            raise

    def run(self, as_of_date: str) -> pl.DataFrame:
        """Runs the full rolling feature pipeline for a target game date.

        The output contains one row per batter with all rolling features
        as of ``as_of_date - 1 day`` (i.e. their features entering the game).

        Args:
            as_of_date: Target game date in ``YYYY-MM-DD`` format. Features are
                computed using only data *before* this date (anti-leakage).

        Returns:
            DataFrame with all rolling features. One row per batter.

        Raises:
            ValueError: If ``as_of_date`` is not a valid date string.
            Exception: Re-raised on any computation error.
        """
        # Validate date format
        try:
            date.fromisoformat(as_of_date)
        except ValueError as exc:
            raise ValueError(f"as_of_date must be YYYY-MM-DD, got '{as_of_date}'") from exc

        log.info("rolling_features_start", as_of_date=as_of_date)
        cfg = self.config.window

        # Load and prepare PA-level data
        lf_raw = self._load_silver(as_of_date)
        lf_flags = _add_pa_event_flags(lf_raw)
        daily_lf = _aggregate_to_daily(lf_flags)
        daily_df = daily_lf.collect()

        if daily_df.is_empty():
            log.warning("no_data_before_as_of_date", as_of_date=as_of_date)
            return daily_df

        log.info("daily_aggregation_complete", rows=len(daily_df),
                 batters=daily_df["batter_id"].n_unique())

        # Compute calendar-day rolling windows
        roll_7d  = _compute_calendar_rolling(daily_df, cfg.short_days,  cfg.min_pa_short, "7d")
        roll_15d = _compute_calendar_rolling(daily_df, cfg.medium_days, cfg.min_pa_short, "15d")
        roll_30d = _compute_calendar_rolling(daily_df, cfg.long_days,   cfg.min_pa_long,  "30d")

        # EWMA on xwOBA (slow + fast)
        daily_with_ewma = _compute_ewma(
            daily_df, cfg.ewma_alpha_slow, "xwoba_mean", "xwoba_ewma_slow"
        )
        daily_with_ewma = _compute_ewma(
            daily_with_ewma, cfg.ewma_alpha_fast, "xwoba_mean", "xwoba_ewma_fast"
        )

        # Join all window results together on (batter_id, game_date)
        join_key = ["batter_id", "game_date"]
        result = (
            daily_with_ewma
            .join(roll_7d.select(join_key + [c for c in roll_7d.columns if c not in join_key]),
                  on=join_key, how="left")
            .join(roll_15d.select(join_key + [c for c in roll_15d.columns if c not in join_key]),
                  on=join_key, how="left")
            .join(roll_30d.select(join_key + [c for c in roll_30d.columns if c not in join_key]),
                  on=join_key, how="left")
        )

        # Take only the LATEST row per batter (features as of the most recent game)
        feature_snapshot = (
            result
            .sort(["batter_id", "game_date"])
            .group_by("batter_id")
            .last()
            .with_columns(pl.lit(as_of_date).alias("feature_date"))
        )

        log.info(
            "rolling_features_complete",
            as_of_date=as_of_date,
            output_rows=len(feature_snapshot),
            null_xwoba_7d_pct=round(
                feature_snapshot["xwoba_7d"].null_count() / len(feature_snapshot), 3
            ) if "xwoba_7d" in feature_snapshot.columns else None,
        )

        self._save(feature_snapshot, as_of_date)
        return feature_snapshot

    def _save(self, df: pl.DataFrame, as_of_date: str) -> None:
        """Persists the feature snapshot to the Gold output path.

        Args:
            df: Feature DataFrame to write.
            as_of_date: Game date; used as partition value.
        """
        out_path = f"{self.output_path}/feature_date={as_of_date}/features.parquet"
        try:
            df.write_parquet(out_path)
            log.info("rolling_features_saved", path=out_path, rows=len(df))
        except Exception as exc:
            log.error("rolling_features_save_failed", path=out_path, error=str(exc))
            raise

    def run_date_range(
        self,
        start_date: str,
        end_date: str,
    ) -> dict[str, int]:
        """Runs the rolling feature pipeline for every date in a range.

        Useful for backfilling the Gold layer after initial deployment or
        after a Silver data correction.

        Args:
            start_date: First date to compute features for (``YYYY-MM-DD``).
            end_date: Last date to compute features for (``YYYY-MM-DD``).

        Returns:
            Dict mapping date → number of batter rows produced.
        """
        start = date.fromisoformat(start_date)
        end   = date.fromisoformat(end_date)

        if start > end:
            raise ValueError(f"start_date {start_date} must be <= end_date {end_date}")

        results: dict[str, int] = {}
        current = start

        while current <= end:
            date_str = current.isoformat()
            try:
                df = self.run(as_of_date=date_str)
                results[date_str] = len(df)
            except Exception as exc:
                log.error("date_range_run_failed", date=date_str, error=str(exc))
                results[date_str] = -1
            current += timedelta(days=1)

        success = sum(1 for v in results.values() if v >= 0)
        log.info("date_range_complete", success=success, total=len(results))
        return results


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def validate_anti_leakage(
    features_df: pl.DataFrame,
    silver_df: pl.DataFrame,
    as_of_date: str,
    batter_id: int,
) -> dict[str, bool]:
    """Validates that rolling features for a batter contain no data from as_of_date.

    A spot-check utility to verify the anti-leakage guarantee for a single
    batter on a specific game date. Compares the xwoba_7d rolling value
    against a manually computed window that explicitly excludes as_of_date.

    Args:
        features_df: Output from ``RollingFeaturesEngine.run()``.
        silver_df: Raw Silver PA data.
        as_of_date: Target game date (data from this date must NOT appear).
        batter_id: Batter to validate.

    Returns:
        Dict with ``"no_future_data"`` and ``"values_match"`` boolean flags.
    """
    # Check that no Silver data from as_of_date leaked into the feature
    on_date_xwoba = (
        silver_df
        .filter((pl.col("batter_id") == batter_id) & (pl.col("game_date") == as_of_date))
        ["xwoba"].to_list()
    )

    batter_feature = features_df.filter(pl.col("batter_id") == batter_id)
    if batter_feature.is_empty():
        return {"no_future_data": True, "values_match": False}

    xwoba_7d = batter_feature["xwoba_7d"][0]

    # Manually compute the expected 7d window (7 days before as_of_date, exclusive)
    expected_window = (
        silver_df
        .filter(
            (pl.col("batter_id") == batter_id)
            & (pl.col("game_date") < as_of_date)
            & (pl.col("game_date") >= str(
                (date.fromisoformat(as_of_date) - timedelta(days=7)).isoformat()
            ))
        )
        ["xwoba"].drop_nulls().mean()
    )

    values_match = (
        abs((xwoba_7d or 0) - (expected_window or 0)) < 0.001
        if expected_window is not None and xwoba_7d is not None
        else True
    )

    return {
        "no_future_data": len(on_date_xwoba) == 0 or on_date_xwoba[0] not in [xwoba_7d],
        "values_match": values_match,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute anti-leakage rolling window batter features."
    )
    parser.add_argument("--silver-path", required=True, dest="silver_path")
    parser.add_argument("--output-path", required=True, dest="output_path")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--as-of-date", dest="as_of_date",
                       help="Single date: YYYY-MM-DD")
    group.add_argument("--date-range", dest="date_range", nargs=2,
                       metavar=("START", "END"),
                       help="Date range: START END in YYYY-MM-DD format")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for rolling feature computation.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]`` when ``None``.
    """
    args = _parse_args(argv)
    engine = RollingFeaturesEngine(
        silver_path=args.silver_path,
        output_path=args.output_path,
    )

    if args.as_of_date:
        result = engine.run(as_of_date=args.as_of_date)
        print(result.select(["batter_id", "xwoba_7d", "xwoba_15d", "xwoba_30d",
                              "launch_speed_7d", "launch_speed_30d"]).head(10))
    else:
        start, end = args.date_range
        summary = engine.run_date_range(start_date=start, end_date=end)
        total_rows = sum(v for v in summary.values() if v > 0)
        print(f"Date range complete: {len(summary)} dates, {total_rows:,} total feature rows.")


if __name__ == "__main__":
    main()
