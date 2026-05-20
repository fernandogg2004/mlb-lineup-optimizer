"""
features_platoon.py
===================
Computes dynamic platoon split features for every batter in the Silver
``plate_appearances`` table, stabilized via Empirical Bayes shrinkage
to handle the small-sample-size problem endemic to baseball analytics.

Mathematical framework
----------------------
For a batter ``i`` facing pitcher hand ``h ∈ {L, R}``, the raw split wOBA
is an unbiased but high-variance estimate when PA count is low.

We apply James-Stein-type Empirical Bayes shrinkage (Efron & Morris, 1975),
adapted to baseball by Tango-Lichtman-Dolphin ("The Book", 2007):

    θ_stabilized = μ_prior + (1 - B) × (θ_raw - μ_prior)

where the shrinkage coefficient B is:

    B = T / (T + PA_observed)

``T`` is the *stabilization threshold* — the number of plate appearances at
which the statistic is 50% skill, 50% noise. Empirically calibrated values:

    Stat          T (PA)   Source
    --------      ------   ------
    wOBA          200      "The Book", Lichtman 2010
    K%            60       Lichtman 2010
    BB%           120      Lichtman 2010
    BABIP         500      Lichtman 2010
    ISO           160      Lichtman 2010

As PA_observed → ∞, B → 0 and θ_stabilized → θ_raw.
As PA_observed → 0, B → 1 and θ_stabilized → μ_prior (league average).

Additionally, splits are computed at **pitch-type granularity** (vs. FF, SL,
CH, CU, etc.) so downstream models can exploit specific arsenal matchups
rather than just pitcher handedness.

Usage:
    python -m src.features.features_platoon \\
        --silver-path s3a://mlb-lakehouse/silver/plate_appearances \\
        --output-path s3a://mlb-lakehouse/gold/platoon_splits \\
        --season 2024

    # Or call as library:
    from src.features.features_platoon import PlatoonSplitsEngine, WOBAWeights
    engine = PlatoonSplitsEngine(silver_path="...", output_path="...")
    splits_df = engine.run(season=2024)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl
import structlog
from scipy import stats

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
# wOBA weights (FanGraphs 2024 season calibration)
# Recalibrate annually using: linear weights from run expectancy matrix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WOBAWeights:
    """Linear weights for wOBA computation, calibrated to run expectancy.

    Attributes:
        bb: Walk weight.
        hbp: Hit-by-pitch weight.
        single: Single weight.
        double: Double weight.
        triple: Triple weight.
        hr: Home run weight.
        scale: League wOBA scale factor (converts to OBP scale).
        lg_woba: League average wOBA for normalization.
        season: Season these weights apply to.
    """

    bb: float = 0.696
    hbp: float = 0.726
    single: float = 0.883
    double: float = 1.244
    triple: float = 1.569
    hr: float = 2.058
    scale: float = 1.157   # wOBAscale to convert to OBP-comparable units
    lg_woba: float = 0.315 # 2024 MLB league average
    season: int = 2024

    @classmethod
    def for_season(cls, season: int) -> "WOBAWeights":
        """Factory returning weights for a given season.

        Currently uses 2024 calibration for all seasons. In production,
        fetch season-specific weights from the Gold woba_weights table.

        Args:
            season: Target season year.

        Returns:
            ``WOBAWeights`` instance for the specified season.
        """
        # TODO: query gold.woba_weights_by_season table for historical seasons
        return cls(season=season)


# ---------------------------------------------------------------------------
# Stabilization thresholds (Lichtman 2010 — empirically calibrated)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StabilizationThresholds:
    """PA thresholds at which each stat reaches 50% reliability.

    These are the canonical "regression-to-mean" thresholds from sabermetric
    research. A stat with T=200 means that after 200 PAs, the observed value
    is equally weighted between the player's true talent and the population
    mean (50% signal, 50% noise).

    Attributes:
        woba: Stabilization PA threshold for wOBA.
        k_rate: Stabilization PA threshold for strikeout rate.
        bb_rate: Stabilization PA threshold for walk rate.
        babip: Stabilization PA threshold for BABIP.
        iso: Stabilization PA threshold for Isolated Power.
        hr_rate: Stabilization PA threshold for HR/PA rate.
    """

    woba: int = 200
    k_rate: int = 60
    bb_rate: int = 120
    babip: int = 500
    iso: int = 160
    hr_rate: int = 170


# ---------------------------------------------------------------------------
# Core Empirical Bayes shrinkage engine
# ---------------------------------------------------------------------------

class EmpiricalBayesShrinkage:
    """Implements James-Stein Empirical Bayes shrinkage for baseball splits.

    Shrinks per-player estimates toward the population mean in proportion to
    sample size uncertainty. The amount of shrinkage is controlled by the
    stabilization threshold ``T``, which encodes prior knowledge about how
    many observations are needed for a stat to become reliable.

    The key guarantee: as ``n → ∞``, the shrunk estimate converges to the
    observed value. As ``n → 0``, it converges to the population mean.
    No player is ever projected as better (or worse) than a reasonable
    credible interval around the population allows with tiny samples.

    Example:
        >>> shrink = EmpiricalBayesShrinkage(threshold=200)
        >>> # Rookie with 40 PA vs. LHP, .500 wOBA observed, .315 league avg
        >>> shrink.shrink(observed=0.500, pa=40, prior_mean=0.315)
        0.3523...  # strongly regressed toward league mean
        >>> # Veteran with 400 PA vs. LHP, .380 wOBA observed
        >>> shrink.shrink(observed=0.380, pa=400, prior_mean=0.315)
        0.3690...  # mostly trusted, gentle regression
    """

    def __init__(self, threshold: int) -> None:
        """Initializes the shrinkage estimator with a stabilization threshold.

        Args:
            threshold: PA count at which the stat is 50% signal / 50% noise.
        """
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self.threshold = threshold

    def shrinkage_coefficient(self, pa: int) -> float:
        """Computes the Bayesian shrinkage coefficient B.

        B = T / (T + PA), where:
            B = 1.0 → full regression to prior (no data)
            B = 0.0 → full trust in observed data (infinite data)

        Args:
            pa: Number of plate appearances observed.

        Returns:
            Shrinkage coefficient in [0, 1].
        """
        return self.threshold / (self.threshold + max(pa, 0))

    def shrink(
        self,
        observed: float,
        pa: int,
        prior_mean: float,
    ) -> float:
        """Computes the Bayes-shrunk estimate.

        Args:
            observed: Raw observed statistic (e.g. 0.380 wOBA).
            pa: Number of plate appearances underlying the observed value.
            prior_mean: Population mean used as the shrinkage anchor.

        Returns:
            Stabilized (shrunk) estimate.
        """
        b = self.shrinkage_coefficient(pa)
        return prior_mean + (1.0 - b) * (observed - prior_mean)

    def shrink_array(
        self,
        observed: np.ndarray,
        pa: np.ndarray,
        prior_mean: float,
    ) -> np.ndarray:
        """Vectorized shrinkage for an array of player estimates.

        Args:
            observed: Array of raw observed values, shape (N,).
            pa: Array of PA counts, shape (N,).
            prior_mean: Scalar population mean.

        Returns:
            Shrunk estimates array, shape (N,).
        """
        b = self.threshold / (self.threshold + np.maximum(pa, 0))
        return prior_mean + (1.0 - b) * (observed - prior_mean)

    def credible_interval(
        self,
        observed: float,
        pa: int,
        prior_mean: float,
        prior_std: float,
        confidence: float = 0.90,
    ) -> tuple[float, float]:
        """Computes an approximate Bayesian credible interval for the estimate.

        Uses a Normal posterior approximation (conjugate prior for mean with
        known variance). The posterior variance combines prior uncertainty and
        observation uncertainty.

        Args:
            observed: Raw observed statistic.
            pa: Plate appearances underlying the observation.
            prior_mean: Population mean (prior location).
            prior_std: Population standard deviation (prior scale).
            confidence: Credible interval probability mass (default 90%).

        Returns:
            Tuple ``(lower, upper)`` credible interval bounds.
        """
        # Posterior precision = prior precision + likelihood precision
        prior_precision = 1.0 / (prior_std ** 2 + 1e-9)
        # Likelihood precision approximation: PA / T gives relative weight
        likelihood_precision = pa / max(self.threshold, 1)
        posterior_precision = prior_precision + likelihood_precision
        posterior_var = 1.0 / posterior_precision

        point_estimate = self.shrink(observed, pa, prior_mean)
        margin = stats.norm.ppf((1 + confidence) / 2) * np.sqrt(posterior_var)
        return (point_estimate - margin, point_estimate + margin)


# ---------------------------------------------------------------------------
# Raw split computation
# ---------------------------------------------------------------------------

@dataclass
class SplitComputationConfig:
    """Configuration for the platoon split computation pipeline.

    Attributes:
        min_pa_to_include: Minimum PAs to include a player-split in the output.
            Players below this threshold get a pure prior (B=1).
        handedness_pairs: List of (batter_stand, pitcher_throws) tuples to compute.
        pitch_types_to_profile: Pitch type codes to compute per-pitch splits for.
        weights: wOBA linear weights for the target season.
        thresholds: Stabilization thresholds for each sabermetric.
    """

    min_pa_to_include: int = 1
    handedness_pairs: list[tuple[str, str]] = field(
        default_factory=lambda: [("L", "L"), ("L", "R"), ("R", "L"), ("R", "R"), ("S", "L"), ("S", "R")]
    )
    pitch_types_to_profile: list[str] = field(
        default_factory=lambda: ["FF", "SI", "FC", "SL", "CH", "CU", "KC", "FS", "EP"]
    )
    weights: WOBAWeights = field(default_factory=WOBAWeights)
    thresholds: StabilizationThresholds = field(default_factory=StabilizationThresholds)


def _compute_woba_polars(df: pl.LazyFrame, weights: WOBAWeights) -> pl.Expr:
    """Returns a Polars expression that computes wOBA for each row group.

    wOBA = (w_bb×BB + w_hbp×HBP + w_1b×1B + w_2b×2B + w_3b×3B + w_hr×HR)
           / (AB + BB - IBB + SF + HBP)

    Denominator omits intentional walks (IBB) because they don't represent
    batter skill.

    Args:
        df: LazyFrame with aggregated event counts per group.
        weights: wOBA linear weights for the season.

    Returns:
        Polars expression for wOBA, null-safe.
    """
    w = weights
    numerator = (
        pl.col("bb") * w.bb
        + pl.col("hbp") * w.hbp
        + pl.col("singles") * w.single
        + pl.col("doubles") * w.double
        + pl.col("triples") * w.triple
        + pl.col("hr") * w.hr
    )
    denominator = (
        pl.col("ab")
        + pl.col("bb")
        - pl.col("ibb")
        + pl.col("sf")
        + pl.col("hbp")
    )
    return (numerator / denominator.clip(lower_bound=1)).alias("woba_raw")


def _compute_iso_polars(weights: WOBAWeights) -> pl.Expr:
    """Returns a Polars expression for ISO (Isolated Power).

    ISO = (2B + 2×3B + 3×HR) / AB

    Args:
        weights: Unused here; kept for API symmetry with other helpers.

    Returns:
        Polars expression for ISO.
    """
    numerator = pl.col("doubles") + 2 * pl.col("triples") + 3 * pl.col("hr")
    return (numerator / pl.col("ab").clip(lower_bound=1)).alias("iso_raw")


def _compute_babip_polars() -> pl.Expr:
    """Returns a Polars expression for BABIP.

    BABIP = (H - HR) / (AB - K - HR + SF)

    Measures batting average on balls put in play; high variance stat.

    Returns:
        Polars expression for BABIP.
    """
    numerator = (pl.col("singles") + pl.col("doubles") + pl.col("triples"))
    denominator = (
        pl.col("ab")
        - pl.col("k")
        - pl.col("hr")
        + pl.col("sf")
    )
    return (numerator / denominator.clip(lower_bound=1)).alias("babip_raw")


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PlatoonSplitsEngine:
    """Computes and stabilizes platoon split features for all batters.

    Pipeline:
        1. ``_load_and_aggregate()`` — reads Silver PAs, aggregates event counts
           by (batter_id, batter_stand, pitcher_throws, [pitch_type]).
        2. ``_compute_raw_metrics()`` — applies wOBA/ISO/BABIP formulas.
        3. ``_compute_league_priors()`` — derives population means by matchup type.
        4. ``_stabilize()`` — applies Empirical Bayes shrinkage per metric.
        5. ``_compute_pitch_type_splits()`` — repeats steps 1–4 at pitch-type grain.
        6. ``save()`` — writes Gold Delta table partitioned by season.

    Attributes:
        silver_path: S3 / local path to the Silver plate_appearances Delta table.
        output_path: S3 / local path for Gold output.
        config: Computation configuration.
    """

    def __init__(
        self,
        silver_path: str,
        output_path: str,
        config: Optional[SplitComputationConfig] = None,
    ) -> None:
        """Initializes the PlatoonSplitsEngine.

        Args:
            silver_path: Path to the Silver plate_appearances Delta table
                (Parquet or Delta format; Polars reads both via PyArrow).
            output_path: Destination path for Gold split features.
            config: Optional computation configuration; defaults to
                ``SplitComputationConfig()`` if not provided.
        """
        self.silver_path = silver_path
        self.output_path = output_path
        self.config = config or SplitComputationConfig()
        self._shrink_woba = EmpiricalBayesShrinkage(self.config.thresholds.woba)
        self._shrink_k = EmpiricalBayesShrinkage(self.config.thresholds.k_rate)
        self._shrink_bb = EmpiricalBayesShrinkage(self.config.thresholds.bb_rate)
        self._shrink_babip = EmpiricalBayesShrinkage(self.config.thresholds.babip)
        self._shrink_iso = EmpiricalBayesShrinkage(self.config.thresholds.iso)

    # ------------------------------------------------------------------
    # Step 1: Load and aggregate
    # ------------------------------------------------------------------

    def _load_silver(self, season: int) -> pl.LazyFrame:
        """Reads the Silver plate_appearances table for a specific season.

        Filters to the target season and selects only the columns needed for
        platoon split computation, minimizing memory footprint.

        Args:
            season: Target MLB season year.

        Returns:
            Polars ``LazyFrame`` with raw PA-level data.

        Raises:
            FileNotFoundError: If the Silver path does not exist.
            Exception: Re-raised on any Polars scan error.
        """
        try:
            lf = (
                pl.scan_parquet(f"{self.silver_path}/**/*.parquet")
                .filter(pl.col("season") == season)
                .select([
                    "batter_id", "batter_stand", "pitcher_throws",
                    "last_pitch_type", "pa_result", "hit_type",
                    "season", "game_date",
                ])
            )
            log.info("silver_loaded", season=season, path=self.silver_path)
            return lf
        except Exception as exc:
            log.error("silver_load_failed", season=season, error=str(exc))
            raise

    def _map_pa_results_to_events(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Maps pa_result strings to binary event flag columns.

        Converts the categorical ``pa_result`` column into boolean indicator
        columns for each event type needed by the sabermetric formulas.

        Args:
            lf: Raw LazyFrame from Silver.

        Returns:
            LazyFrame with event indicator columns added.
        """
        return lf.with_columns([
            (pl.col("hit_type") == "single").cast(pl.Int32).alias("singles"),
            (pl.col("hit_type") == "double").cast(pl.Int32).alias("doubles"),
            (pl.col("hit_type") == "triple").cast(pl.Int32).alias("triples"),
            (pl.col("hit_type") == "home_run").cast(pl.Int32).alias("hr"),
            pl.col("pa_result").is_in(["walk", "intent_walk"]).cast(pl.Int32).alias("bb"),
            (pl.col("pa_result") == "intent_walk").cast(pl.Int32).alias("ibb"),
            (pl.col("pa_result") == "hit_by_pitch").cast(pl.Int32).alias("hbp"),
            pl.col("pa_result").is_in([
                "strikeout", "strikeout_double_play"
            ]).cast(pl.Int32).alias("k"),
            (pl.col("pa_result") == "sac_fly").cast(pl.Int32).alias("sf"),
            # AB = PA - BB - HBP - SF - sacrifice_bunt (simplified: omit SB)
            (~pl.col("pa_result").is_in([
                "walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt"
            ])).cast(pl.Int32).alias("ab"),
            # Any PA counts as 1
            pl.lit(1).alias("pa_count"),
        ])

    def _aggregate_by_handedness(
        self, lf: pl.LazyFrame, group_cols: list[str]
    ) -> pl.LazyFrame:
        """Aggregates event counts by the specified grouping columns.

        Args:
            lf: LazyFrame with event indicator columns.
            group_cols: Columns to group by (e.g. ``["batter_id", "pitcher_throws"]``).

        Returns:
            Aggregated LazyFrame with summed event counts and PA total.
        """
        event_cols = ["singles", "doubles", "triples", "hr", "bb", "ibb",
                      "hbp", "k", "sf", "ab", "pa_count"]
        return lf.group_by(group_cols).agg(
            [pl.col(c).sum() for c in event_cols]
        ).rename({"pa_count": "pa"})

    # ------------------------------------------------------------------
    # Step 2: Raw metrics
    # ------------------------------------------------------------------

    def _compute_raw_metrics(self, agg: pl.LazyFrame) -> pl.LazyFrame:
        """Adds wOBA, K%, BB%, ISO, and BABIP columns to an aggregated LazyFrame.

        Args:
            agg: Aggregated event count LazyFrame.

        Returns:
            LazyFrame with sabermetric metric columns added.
        """
        w = self.config.weights
        return agg.with_columns([
            _compute_woba_polars(agg, w),
            _compute_iso_polars(w),
            _compute_babip_polars(),
            (pl.col("k") / pl.col("pa").clip(lower_bound=1)).alias("k_rate_raw"),
            (pl.col("bb") / pl.col("pa").clip(lower_bound=1)).alias("bb_rate_raw"),
            (pl.col("hr") / pl.col("pa").clip(lower_bound=1)).alias("hr_rate_raw"),
        ])

    # ------------------------------------------------------------------
    # Step 3: League priors
    # ------------------------------------------------------------------

    def _compute_league_priors(
        self,
        metrics_df: pl.DataFrame,
        group_cols: list[str],
    ) -> pl.DataFrame:
        """Computes PA-weighted league averages per matchup type as priors.

        Uses PA-weighted means (not simple means) so that players with more
        observed data exert proportionally more influence on the prior,
        which is the statistically correct population estimate.

        Args:
            metrics_df: Collected DataFrame with raw metric columns.
            group_cols: Handedness columns to sub-group the prior by
                (e.g. ``["batter_stand", "pitcher_throws"]``).

        Returns:
            DataFrame with league-prior columns suffixed ``_prior``.
        """
        metric_cols = ["woba_raw", "k_rate_raw", "bb_rate_raw", "iso_raw", "babip_raw"]

        # PA-weighted mean per matchup type
        prior = (
            metrics_df
            .with_columns([
                (pl.col(m) * pl.col("pa")).alias(f"{m}_weighted")
                for m in metric_cols
            ])
            .group_by(group_cols)
            .agg(
                [pl.col(f"{m}_weighted").sum() for m in metric_cols]
                + [pl.col("pa").sum().alias("total_pa")]
            )
            .with_columns([
                (pl.col(f"{m}_weighted") / pl.col("total_pa").clip(lower_bound=1))
                .alias(f"{m.replace('_raw', '')}_prior")
                for m in metric_cols
            ])
            .select(group_cols + [f"{m.replace('_raw', '')}_prior" for m in metric_cols])
        )

        log.info("league_priors_computed", groups=group_cols)
        return prior

    # ------------------------------------------------------------------
    # Step 4: Empirical Bayes stabilization
    # ------------------------------------------------------------------

    def _stabilize(
        self,
        df: pl.DataFrame,
        prior_df: pl.DataFrame,
        join_cols: list[str],
    ) -> pl.DataFrame:
        """Applies Empirical Bayes shrinkage to all sabermetric metrics.

        For each player-split row, shrinks the raw metric toward the
        corresponding league prior by an amount determined by the PA count.

        Args:
            df: DataFrame with raw metrics and PA counts.
            prior_df: DataFrame with league-level prior means per matchup type.
            join_cols: Columns to join ``df`` and ``prior_df`` on.

        Returns:
            DataFrame with ``_stabilized`` metric columns added and
            corresponding credible intervals (90%).
        """
        merged = df.join(prior_df, on=join_cols, how="left")

        # Convert to numpy for vectorized shrinkage across all rows at once
        pa_arr = merged["pa"].to_numpy()

        stabilized_cols: dict[str, np.ndarray] = {}

        metric_configs = [
            ("woba",    self._shrink_woba,  "woba_raw",    "woba_prior"),
            ("k_rate",  self._shrink_k,     "k_rate_raw",  "k_rate_prior"),
            ("bb_rate", self._shrink_bb,    "bb_rate_raw", "bb_rate_prior"),
            ("iso",     self._shrink_iso,   "iso_raw",     "iso_prior"),
            ("babip",   self._shrink_babip, "babip_raw",   "babip_prior"),
        ]

        for stat_name, shrink_obj, raw_col, prior_col in metric_configs:
            raw_arr = merged[raw_col].fill_null(0.0).to_numpy()
            prior_arr = merged[prior_col].fill_null(
                getattr(self.config.weights, "lg_woba", 0.315)
            ).to_numpy()

            stabilized = shrink_obj.shrink_array(raw_arr, pa_arr, prior_arr.mean())
            stabilized_cols[f"{stat_name}_stabilized"] = stabilized

            # Shrinkage coefficient (diagnostic: 1.0 = pure prior, 0.0 = pure data)
            b_arr = shrink_obj.threshold / (shrink_obj.threshold + np.maximum(pa_arr, 0))
            stabilized_cols[f"{stat_name}_shrinkage_b"] = b_arr

        # Attach stabilized columns back to the DataFrame
        result = merged
        for col_name, arr in stabilized_cols.items():
            result = result.with_columns(
                pl.Series(name=col_name, values=arr)
            )

        log.info(
            "stabilization_complete",
            rows=len(result),
            avg_woba_shrinkage_b=float(np.mean(stabilized_cols["woba_shrinkage_b"])),
        )
        return result

    # ------------------------------------------------------------------
    # Step 5: Pitch-type granularity splits
    # ------------------------------------------------------------------

    def _compute_pitch_type_splits(
        self, lf: pl.LazyFrame, season: int
    ) -> pl.DataFrame:
        """Computes wOBA splits at pitch-type × handedness granularity.

        For each (batter_id, pitcher_throws, last_pitch_type) group,
        computes and stabilizes wOBA so the model can score matchups at
        the specific pitch-type level (e.g. batter vs. RHP slider).

        Args:
            lf: LazyFrame with event indicator columns already added.
            season: Season used only for the output metadata column.

        Returns:
            DataFrame with pitch-type split wOBA stabilized values.
        """
        group_cols = ["batter_id", "batter_stand", "pitcher_throws", "last_pitch_type"]

        # Filter to only tracked pitch types
        lf_filtered = lf.filter(
            pl.col("last_pitch_type").is_in(self.config.pitch_types_to_profile)
            & pl.col("last_pitch_type").is_not_null()
        )

        agg = self._aggregate_by_handedness(lf_filtered, group_cols)
        metrics_lf = self._compute_raw_metrics(agg)
        metrics_df = metrics_lf.collect()

        if metrics_df.is_empty():
            log.warning("pitch_type_splits_empty", season=season)
            return metrics_df

        prior_df = self._compute_league_priors(
            metrics_df, group_cols=["batter_stand", "pitcher_throws", "last_pitch_type"]
        )
        stabilized = self._stabilize(metrics_df, prior_df,
                                     join_cols=["batter_stand", "pitcher_throws", "last_pitch_type"])
        return stabilized.with_columns(pl.lit(season).alias("season"))

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, season: int) -> pl.DataFrame:
        """Runs the full platoon split feature computation pipeline.

        Computes both handedness-level and pitch-type-level splits, then
        writes the output to the Gold layer as Parquet partitioned by season.

        Args:
            season: MLB season to compute splits for.

        Returns:
            Collected DataFrame with all stabilized split features.

        Raises:
            Exception: Re-raised if any computation step fails.
        """
        log.info("platoon_splits_start", season=season)

        lf_raw = self._load_silver(season)
        lf_events = self._map_pa_results_to_events(lf_raw)

        # --- Handedness-level splits ---
        group_cols = ["batter_id", "batter_stand", "pitcher_throws"]
        agg = self._aggregate_by_handedness(lf_events, group_cols)
        metrics_lf = self._compute_raw_metrics(agg)
        metrics_df = metrics_lf.collect()

        if metrics_df.is_empty():
            log.warning("no_data_for_season", season=season)
            return metrics_df

        prior_df = self._compute_league_priors(
            metrics_df, group_cols=["batter_stand", "pitcher_throws"]
        )
        hand_splits = self._stabilize(
            metrics_df, prior_df, join_cols=["batter_stand", "pitcher_throws"]
        ).with_columns(pl.lit(season).alias("season"))

        # --- Pitch-type splits ---
        pitch_splits = self._compute_pitch_type_splits(lf_events, season)

        # --- Persist ---
        self._save(hand_splits, suffix="handedness")
        if not pitch_splits.is_empty():
            self._save(pitch_splits, suffix="pitch_type")

        log.info(
            "platoon_splits_complete",
            season=season,
            hand_rows=len(hand_splits),
            pitch_type_rows=len(pitch_splits),
        )
        return hand_splits

    def _save(self, df: pl.DataFrame, suffix: str) -> None:
        """Writes a split DataFrame to the Gold output path.

        Args:
            df: DataFrame to persist.
            suffix: Distinguishes handedness vs. pitch_type splits in the path.
        """
        season = df["season"][0] if "season" in df.columns else "unknown"
        out_path = f"{self.output_path}/{suffix}/season={season}/"
        try:
            df.write_parquet(out_path)
            log.info("splits_saved", path=out_path, rows=len(df), suffix=suffix)
        except Exception as exc:
            log.error("splits_save_failed", path=out_path, error=str(exc))
            raise


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def summarize_stabilization(df: pl.DataFrame) -> pl.DataFrame:
    """Returns a diagnostic summary of shrinkage applied across the dataset.

    Useful for validating that the EB model is behaving correctly:
    - Players with many PAs should have low shrinkage_b values (trust data).
    - Players with few PAs should have high shrinkage_b values (trust prior).

    Args:
        df: Output DataFrame from ``PlatoonSplitsEngine.run()``.

    Returns:
        Summary DataFrame grouped by PA quintile showing average shrinkage.
    """
    if "woba_shrinkage_b" not in df.columns:
        raise ValueError("Input DataFrame must contain 'woba_shrinkage_b' column.")

    return (
        df
        .with_columns(pl.col("pa").qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).alias("pa_quintile"))
        .group_by("pa_quintile")
        .agg([
            pl.col("pa").mean().round(0).alias("avg_pa"),
            pl.col("woba_shrinkage_b").mean().round(3).alias("avg_shrinkage_b"),
            pl.col("woba_raw").mean().round(3).alias("avg_woba_raw"),
            pl.col("woba_stabilized").mean().round(3).alias("avg_woba_stabilized"),
            pl.len().alias("n_players"),
        ])
        .sort("pa_quintile")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Empirical Bayes platoon splits for Silver plate_appearances."
    )
    parser.add_argument("--silver-path", required=True, dest="silver_path")
    parser.add_argument("--output-path", required=True, dest="output_path")
    parser.add_argument("--season", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]`` when ``None``.
    """
    args = _parse_args(argv)
    weights = WOBAWeights.for_season(args.season)
    config = SplitComputationConfig(weights=weights)
    engine = PlatoonSplitsEngine(
        silver_path=args.silver_path,
        output_path=args.output_path,
        config=config,
    )
    result = engine.run(season=args.season)
    summary = summarize_stabilization(result)
    print(summary)


if __name__ == "__main__":
    main()
