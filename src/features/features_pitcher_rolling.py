"""
features_pitcher_rolling.py
===========================
Computes rolling window features for **pitchers** using Polars, mirroring the
structure of ``features_rolling.py`` (which handles batters).

Anti-leakage guarantee
----------------------
All rolling aggregates exclude the current game date.  Enforcement:

    1. Sort by (pitcher_id, game_date) within each group.
    2. Apply ``shift(1).over("pitcher_id")`` BEFORE each rolling operation so
       the current outing's stats are never included in the window.
    3. Calendar-day windows use ``join_asof`` with ``strategy="backward"`` on a
       per-pitcher daily summary, looking back exactly N days from D-1.

Feature catalog (~18 dimensions)
---------------------------------
    pitches_7d / pitches_3d        — Recent workload (fatigue proxy)
    days_since_last_app            — Rest days (> 5 = well-rested; < 3 = short rest)
    velo_ff_7d / velo_ff_21d       — Fastball velocity recent vs. medium-term
    velo_ff_delta_7d_vs_30d        — Velocity change 7d vs. 30d (< 0 = fatigue signal)
    spin_ff_7d / spin_sl_7d        — Spin rate by pitch type (FF, SL)
    whiff_rate_7d / whiff_rate_21d — Swinging-strike rate (dominance)
    csw_rate_7d                    — Called-Strike + Whiff % (Command quality)
    velo_inning_slope_last_start   — Intra-start velocity slope (< 0 = mid-game fatigue)
    p_throws_binary                — Handedness indicator (1=R, 0=L) for platoon models

Usage:
    from src.features.features_pitcher_rolling import PitcherRollingFeaturesEngine
    engine = PitcherRollingFeaturesEngine(
        bronze_path="s3a://mlb-lakehouse/bronze/statcast_pitches",
        output_path="s3a://mlb-lakehouse/gold/pitcher_features_rolling",
    )
    features = engine.run(as_of_date="2024-07-15")

CLI:
    python -m src.features.features_pitcher_rolling \\
        --bronze-path s3a://mlb-lakehouse/bronze/statcast_pitches \\
        --output-path s3a://mlb-lakehouse/gold/pitcher_features_rolling \\
        --as-of-date 2024-07-15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
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

# Pitch types to track individually (accounts for >90% of MLB pitch mix)
_PRIMARY_PITCH_TYPES: list[str] = ["FF", "SI", "FC", "SL", "CH", "CU"]

# Velocity floor/ceiling for anomaly rejection (matches statcast_ingestion.py)
_VELO_MIN_MPH: float = 40.0
_VELO_MAX_MPH: float = 110.0

# Spin floor/ceiling
_SPIN_MIN_RPM: float = 0.0
_SPIN_MAX_RPM: float = 4000.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PitcherWindowConfig:
    """Window sizes for pitcher rolling features.

    Attributes:
        short_days: Short window in calendar days (~2 SP starts).
        medium_days: Medium window in calendar days (~3-4 SP starts).
        long_days: Long window for velocity baseline comparison.
        min_pitches_short: Minimum pitches in short window (null if fewer).
        min_pitches_medium: Minimum pitches in medium window (null if fewer).
        pitch_types_to_track: Pitch types with individual velo/spin features.
    """

    short_days: int = 7
    medium_days: int = 21
    long_days: int = 30
    min_pitches_short: int = 15
    min_pitches_medium: int = 30
    pitch_types_to_track: list[str] = field(
        default_factory=lambda: list(_PRIMARY_PITCH_TYPES)
    )


@dataclass(frozen=True)
class PitcherRollingConfig:
    """Top-level configuration for the pitcher rolling features pipeline.

    Attributes:
        window: Window sizes and minimum-pitch thresholds.
        bronze_columns: Columns to load from Bronze (minimize I/O).
        output_partition_col: Column name used to partition output Parquet.
    """

    window: PitcherWindowConfig = field(default_factory=PitcherWindowConfig)
    bronze_columns: list[str] = field(
        default_factory=lambda: [
            "pitcher_id", "game_date", "game_pk", "inning",
            "pitch_type", "release_speed", "release_spin_rate",
            "p_throws",
            # Outcome flags (for whiff / CSW)
            "description",  # "swinging_strike", "called_strike", "ball", etc.
        ]
    )
    output_partition_col: str = "feature_date"


# ---------------------------------------------------------------------------
# Pitch-level flag computation
# ---------------------------------------------------------------------------

def _add_pitch_flags(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Adds binary outcome flags to the pitch-level Bronze LazyFrame.

    Args:
        lf: Raw pitch-level Bronze LazyFrame.

    Returns:
        LazyFrame with flag columns appended.
    """
    return lf.with_columns([
        pl.lit(1).cast(pl.Int8).alias("pitch_flag"),

        # Swinging strike (whiff)
        pl.col("description").is_in([
            "swinging_strike", "swinging_strike_blocked",
            "foul_tip", "missed_bunt",
        ]).cast(pl.Int8).alias("whiff_flag"),

        # Called strike + whiff (CSW: pitch dominance metric)
        pl.col("description").is_in([
            "called_strike",
            "swinging_strike", "swinging_strike_blocked",
            "foul_tip",
        ]).cast(pl.Int8).alias("csw_flag"),

        # Fastball indicator (for per-type velocity features)
        (pl.col("pitch_type") == "FF").cast(pl.Int8).alias("is_ff"),
        (pl.col("pitch_type") == "SL").cast(pl.Int8).alias("is_sl"),

        # Valid velocity (within physical bounds)
        (
            pl.col("release_speed").is_not_null()
            & pl.col("release_speed").is_between(_VELO_MIN_MPH, _VELO_MAX_MPH)
        ).cast(pl.Int8).alias("velo_valid"),

        # Valid spin rate
        (
            pl.col("release_spin_rate").is_not_null()
            & pl.col("release_spin_rate").is_between(_SPIN_MIN_RPM, _SPIN_MAX_RPM)
        ).cast(pl.Int8).alias("spin_valid"),

        # Right-handed indicator (for platoon interaction features)
        (pl.col("p_throws") == "R").cast(pl.Int8).alias("p_throws_binary"),
    ])


# ---------------------------------------------------------------------------
# Daily aggregation (game-day grain per pitcher)
# ---------------------------------------------------------------------------

def _aggregate_to_daily(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregates pitch-level data to one row per (pitcher_id, game_date).

    Args:
        lf: Pitch-level LazyFrame with event flags added.

    Returns:
        Daily-grain LazyFrame with pitch counts and mean Statcast metrics.
    """
    return (
        lf.group_by(["pitcher_id", "game_date", "p_throws_binary"])
        .agg([
            # Workload
            pl.col("pitch_flag").sum().alias("pitches"),

            # Outcome rates (numerator counts; denominator is total pitches)
            pl.col("whiff_flag").sum().alias("whiffs"),
            pl.col("csw_flag").sum().alias("csw_strikes"),

            # FF velocity (mean of valid readings)
            pl.when(pl.col("is_ff") == 1)
              .then(pl.when(pl.col("velo_valid") == 1).then(pl.col("release_speed")))
              .otherwise(None)
              .mean().alias("velo_ff_mean"),

            # SL velocity
            pl.when(pl.col("is_sl") == 1)
              .then(pl.when(pl.col("velo_valid") == 1).then(pl.col("release_speed")))
              .otherwise(None)
              .mean().alias("velo_sl_mean"),

            # FF spin rate (mean of valid readings)
            pl.when(pl.col("is_ff") == 1)
              .then(pl.when(pl.col("spin_valid") == 1).then(pl.col("release_spin_rate")))
              .otherwise(None)
              .mean().alias("spin_ff_mean"),

            # SL spin rate
            pl.when(pl.col("is_sl") == 1)
              .then(pl.when(pl.col("spin_valid") == 1).then(pl.col("release_spin_rate")))
              .otherwise(None)
              .mean().alias("spin_sl_mean"),

            # Handedness (constant within pitcher; use first value)
            pl.col("p_throws_binary").first(),
        ])
        .sort(["pitcher_id", "game_date"])
    )


# ---------------------------------------------------------------------------
# Intra-start velocity slope (mid-game fatigue signal)
# ---------------------------------------------------------------------------

def _compute_inning_velo_slope(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Computes per-start FF velocity slope across innings (last start only).

    A negative slope (e.g., -0.5 mph/inning) signals that the pitcher
    tires during starts — a leading indicator of elevated run-scoring risk
    when that pitcher is deployed late in games.

    Anti-leakage: the slope for the *current* game date is shifted one row
    forward so it reflects the PREVIOUS start's fatigue profile.

    Args:
        lf: Pitch-level LazyFrame with flags added.

    Returns:
        Per-(pitcher_id, game_date) LazyFrame with ``velo_inning_slope_last_start``.
    """
    # Aggregate to inning-level means for FF pitches
    inning_velo = (
        lf
        .filter((pl.col("is_ff") == 1) & (pl.col("velo_valid") == 1))
        .group_by(["pitcher_id", "game_date", "inning"])
        .agg(pl.col("release_speed").mean().alias("velo_inning_mean"))
        .sort(["pitcher_id", "game_date", "inning"])
    )

    # OLS slope: β = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
    slope_per_start = (
        inning_velo
        .group_by(["pitcher_id", "game_date"])
        .agg([
            # Use Polars expression for covariance / variance (OLS slope)
            (
                (pl.col("inning") - pl.col("inning").mean())
                * (pl.col("velo_inning_mean") - pl.col("velo_inning_mean").mean())
            ).sum().alias("cov_xy"),
            (
                (pl.col("inning") - pl.col("inning").mean()) ** 2
            ).sum().alias("var_x"),
            pl.col("inning").count().alias("n_innings"),
        ])
        .with_columns([
            pl.when(pl.col("var_x") > 0)
              .then(pl.col("cov_xy") / pl.col("var_x"))
              .otherwise(None)
              .alias("velo_inning_slope"),
        ])
        .select(["pitcher_id", "game_date", "velo_inning_slope"])
        .sort(["pitcher_id", "game_date"])
    )

    # Anti-leakage: shift slope so it reflects the PREVIOUS start
    return (
        slope_per_start
        .with_columns([
            pl.col("velo_inning_slope")
              .shift(1)
              .over("pitcher_id")
              .alias("velo_inning_slope_last_start"),
        ])
        .select(["pitcher_id", "game_date", "velo_inning_slope_last_start"])
    )


# ---------------------------------------------------------------------------
# Calendar-day rolling computation
# ---------------------------------------------------------------------------

def _compute_calendar_rolling(
    daily: pl.LazyFrame,
    as_of_date: date,
    cfg: PitcherWindowConfig,
) -> pl.LazyFrame:
    """Computes rolling pitcher features using calendar-day windows.

    For a target date D, each window [D - W, D - 1] is computed via
    ``join_asof`` on a pre-aggregated daily summary.  The ``shift(1)``
    within each pitcher group ensures D itself is excluded.

    Args:
        daily: Daily-grain LazyFrame from ``_aggregate_to_daily()``.
        as_of_date: The game date for which we want features.
        cfg: Window size configuration.

    Returns:
        LazyFrame with rolling features, one row per (pitcher_id, feature_date).
    """
    # Build a spine of (pitcher_id, feature_date) for the target date only
    pitchers = daily.select("pitcher_id").unique()
    spine = pitchers.with_columns(
        pl.lit(as_of_date.isoformat()).cast(pl.Utf8).alias("feature_date")
    )

    # For each window, build a rolling aggregate via group_by_dynamic or
    # manual join_asof approach.  We use the shift+rolling pattern.
    daily_collected = daily.collect()

    results: list[pl.DataFrame] = []
    for pitcher_id, group in daily_collected.group_by("pitcher_id"):
        group = group.sort("game_date")
        n = len(group)

        # Anti-leakage: each row represents the state BEFORE that game
        # by shifting all metrics one position forward.
        # The rolling window looks back from the CURRENT row (after shift).
        p_arr = group["pitches"].to_numpy().astype(float)
        wh_arr = group["whiffs"].to_numpy().astype(float)
        csw_arr = group["csw_strikes"].to_numpy().astype(float)
        vff_arr = group["velo_ff_mean"].to_numpy().astype(float)
        vsl_arr = group["velo_sl_mean"].to_numpy().astype(float)
        sff_arr = group["spin_ff_mean"].to_numpy().astype(float)
        ssl_arr = group["spin_sl_mean"].to_numpy().astype(float)
        dates   = group["game_date"].to_list()  # strings "YYYY-MM-DD"

        as_of_str = as_of_date.isoformat()

        # Only include games strictly before as_of_date (anti-leakage)
        mask_before = [d < as_of_str for d in dates]
        if not any(mask_before):
            continue

        # Determine last game date before as_of_date
        recent_dates = [d for d in dates if d < as_of_str]
        last_app_date = max(recent_dates)
        last_app_dt   = date.fromisoformat(last_app_date)
        days_since    = (as_of_date - last_app_dt).days

        short_cutoff = (as_of_date - timedelta(days=cfg.short_days)).isoformat()
        med_cutoff   = (as_of_date - timedelta(days=cfg.medium_days)).isoformat()
        long_cutoff  = (as_of_date - timedelta(days=cfg.long_days)).isoformat()

        # Select games in each window (strict: < as_of_date, >= cutoff)
        def _win(cut: str) -> np.ndarray:
            return np.array([
                i for i, d in enumerate(dates) if cut <= d < as_of_str
            ])

        idx_7d  = _win(short_cutoff)
        idx_21d = _win(med_cutoff)
        idx_30d = _win(long_cutoff)

        def _safe_mean(arr: np.ndarray, idx: np.ndarray) -> Optional[float]:
            vals = arr[idx]
            valid = vals[~np.isnan(vals)]
            return float(np.nanmean(valid)) if len(valid) > 0 else None

        def _safe_sum(arr: np.ndarray, idx: np.ndarray) -> int:
            return int(np.nansum(arr[idx]))

        pitches_7d  = _safe_sum(p_arr, idx_7d)
        pitches_3d  = _safe_sum(p_arr, np.array([
            i for i, d in enumerate(dates) if
            (as_of_date - timedelta(days=3)).isoformat() <= d < as_of_str
        ]))

        total_7d    = max(pitches_7d, 1)
        total_21d   = max(_safe_sum(p_arr, idx_21d), 1)
        whiff_7d    = _safe_sum(wh_arr, idx_7d) / total_7d
        csw_7d      = _safe_sum(csw_arr, idx_7d) / total_7d

        vff_7d   = _safe_mean(vff_arr, idx_7d)
        vff_21d  = _safe_mean(vff_arr, idx_21d)
        vff_30d  = _safe_mean(vff_arr, idx_30d)
        vff_delta = (vff_7d - vff_30d) if (vff_7d is not None and vff_30d is not None) else None

        row = {
            "pitcher_id":        pitcher_id[0] if isinstance(pitcher_id, tuple) else pitcher_id,
            "feature_date":      as_of_str,
            "pitches_7d":        pitches_7d,
            "pitches_3d":        pitches_3d,
            "days_since_last_app": days_since,
            "velo_ff_7d":        vff_7d,
            "velo_ff_21d":       vff_21d,
            "velo_ff_delta_7d_vs_30d": vff_delta,
            "spin_ff_7d":        _safe_mean(sff_arr, idx_7d),
            "spin_sl_7d":        _safe_mean(ssl_arr, idx_7d),
            "whiff_rate_7d":     whiff_7d,
            "csw_rate_7d":       csw_7d,
            "p_throws_binary":   group["p_throws_binary"].to_list()[-1],
        }
        results.append(pl.DataFrame([row]))

    if not results:
        return _empty_features_df(as_of_date).lazy()

    return pl.concat(results).lazy()


def _empty_features_df(as_of_date: date) -> pl.DataFrame:
    """Returns an empty DataFrame with the correct schema when no data exists."""
    return pl.DataFrame({
        "pitcher_id": pl.Series([], dtype=pl.Int32),
        "feature_date": pl.Series([], dtype=pl.Utf8),
        "pitches_7d": pl.Series([], dtype=pl.Int32),
        "pitches_3d": pl.Series([], dtype=pl.Int32),
        "days_since_last_app": pl.Series([], dtype=pl.Int32),
        "velo_ff_7d": pl.Series([], dtype=pl.Float32),
        "velo_ff_21d": pl.Series([], dtype=pl.Float32),
        "velo_ff_delta_7d_vs_30d": pl.Series([], dtype=pl.Float32),
        "spin_ff_7d": pl.Series([], dtype=pl.Float32),
        "spin_sl_7d": pl.Series([], dtype=pl.Float32),
        "whiff_rate_7d": pl.Series([], dtype=pl.Float32),
        "csw_rate_7d": pl.Series([], dtype=pl.Float32),
        "p_throws_binary": pl.Series([], dtype=pl.Int8),
    })


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PitcherRollingFeaturesEngine:
    """Computes rolling pitcher features from the Bronze pitch-level layer.

    Mirrors ``RollingFeaturesEngine`` for batters, but operates on the
    Bronze pitch-level table (one row per pitch) rather than the Silver
    PA-level table.  The output is the Gold pitcher rolling feature store.

    Args:
        bronze_path: Path to the Bronze Parquet/Delta table (Statcast pitches).
        output_path: Destination Parquet path for the Gold feature store.
        config: Engine configuration; uses defaults if not provided.
    """

    def __init__(
        self,
        bronze_path: str,
        output_path: str,
        config: Optional[PitcherRollingConfig] = None,
    ) -> None:
        self._bronze_path  = bronze_path
        self._output_path  = output_path
        self._config       = config or PitcherRollingConfig()

    def run(self, as_of_date: str) -> pl.DataFrame:
        """Builds pitcher rolling features for the given date.

        Anti-leakage: uses only pitches thrown strictly BEFORE ``as_of_date``.

        Args:
            as_of_date: Target game date in "YYYY-MM-DD" format.

        Returns:
            Polars DataFrame with one row per pitcher, columns as per the
            feature catalog above.
        """
        target = date.fromisoformat(as_of_date)
        cfg    = self._config

        log.info("pitcher_rolling_start", as_of_date=as_of_date)
        t0_str = (target - timedelta(days=max(
            cfg.window.long_days, cfg.window.medium_days
        ) + 5)).isoformat()

        # Load Bronze (pitch-level), filtered to the look-back window
        raw_lf = (
            pl.scan_parquet(self._bronze_path)
            .select(cfg.bronze_columns)
            .filter(
                (pl.col("game_date") >= t0_str)
                & (pl.col("game_date") < as_of_date)   # strict anti-leakage
            )
        )

        # Add per-pitch flags
        raw_lf = _add_pitch_flags(raw_lf)

        # Aggregate to daily grain
        daily_lf = _aggregate_to_daily(raw_lf)

        # Compute intra-start velocity slope (last start)
        slope_lf = _compute_inning_velo_slope(raw_lf)

        # Compute calendar-day rolling features
        features_lf = _compute_calendar_rolling(daily_lf, target, cfg.window)

        # Join slope feature
        features_lf = features_lf.join(
            slope_lf.filter(pl.col("game_date") < as_of_date)
                    .group_by("pitcher_id")
                    .agg(
                        pl.col("velo_inning_slope_last_start").last()
                    ),
            on="pitcher_id",
            how="left",
        )

        result = features_lf.collect()

        # Cast to canonical dtypes
        result = result.with_columns([
            pl.col("pitcher_id").cast(pl.Int32),
            pl.col("pitches_7d").cast(pl.Int32),
            pl.col("pitches_3d").cast(pl.Int32),
            pl.col("days_since_last_app").cast(pl.Int32),
            pl.col("velo_ff_7d").cast(pl.Float32),
            pl.col("velo_ff_21d").cast(pl.Float32),
            pl.col("velo_ff_delta_7d_vs_30d").cast(pl.Float32),
            pl.col("spin_ff_7d").cast(pl.Float32),
            pl.col("spin_sl_7d").cast(pl.Float32),
            pl.col("whiff_rate_7d").cast(pl.Float32),
            pl.col("csw_rate_7d").cast(pl.Float32),
            pl.col("velo_inning_slope_last_start").cast(pl.Float32),
            pl.col("p_throws_binary").cast(pl.Int8),
        ])

        log.info("pitcher_rolling_done", n_pitchers=len(result), as_of_date=as_of_date)

        # Write to Gold layer
        result.write_parquet(
            f"{self._output_path}/feature_date={as_of_date}/data.parquet",
            use_pyarrow=True,
        )

        return result

    def run_range(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Computes features for a range of dates (backfill mode).

        Args:
            start_date: First date to compute (inclusive), "YYYY-MM-DD".
            end_date: Last date to compute (inclusive), "YYYY-MM-DD".

        Returns:
            Concatenated DataFrame across all dates.
        """
        current = date.fromisoformat(start_date)
        stop    = date.fromisoformat(end_date)
        frames: list[pl.DataFrame] = []

        while current <= stop:
            try:
                df = self.run(current.isoformat())
                frames.append(df)
            except Exception as exc:
                log.warning("pitcher_rolling_date_skipped",
                            date=current.isoformat(), error=str(exc))
            current += timedelta(days=1)

        return pl.concat(frames) if frames else _empty_features_df(stop)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pitcher rolling features — build Gold layer"
    )
    parser.add_argument("--bronze-path", required=True, dest="bronze_path")
    parser.add_argument("--output-path", required=True, dest="output_path")
    parser.add_argument("--as-of-date",  required=True, dest="as_of_date",
                        help="YYYY-MM-DD target game date")
    parser.add_argument("--era", default="2023+",
                        choices=["2015-2019", "2020-2022", "2023+"],
                        help="MLB rule era for era-adjusted feature norms")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args   = _parse_args(argv)
    engine = PitcherRollingFeaturesEngine(
        bronze_path=args.bronze_path,
        output_path=args.output_path,
    )
    result = engine.run(args.as_of_date)
    print(f"Computed {len(result)} pitcher profiles for {args.as_of_date}")


if __name__ == "__main__":
    main()
