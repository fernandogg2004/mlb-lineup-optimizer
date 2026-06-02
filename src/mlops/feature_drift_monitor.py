"""
feature_drift_monitor.py
========================
Population Stability Index (PSI) monitor for the 40-feature PA prediction model.

PSI measures how much a feature's distribution has shifted between a reference
period (training data) and a recent window (last N days of Silver data).

    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

Interpretation (standard industry thresholds):
    PSI < 0.10  → No significant drift  (green)
    PSI < 0.25  → Moderate drift        (yellow — monitor closely)
    PSI ≥ 0.25  → Significant drift     (red — consider retraining)

Usage (standalone):
    python -m src.mlops.feature_drift_monitor \\
        --silver-dir data/silver/plate_appearances \\
        --reference-seasons 2022 2023 \\
        --recent-days 30 \\
        --output reports/drift/drift_$(date +%F).json

Usage (from morning.py):
    from src.mlops.feature_drift_monitor import DriftMonitor
    monitor = DriftMonitor(silver_dir="data/silver/plate_appearances")
    report  = monitor.run(reference_seasons=[2022, 2023], recent_days=30)
    monitor.print_summary(report)
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

# PSI thresholds
_PSI_GREEN  = 0.10
_PSI_YELLOW = 0.25
_N_BINS     = 10       # equal-frequency bins for PSI computation
_EPS        = 1e-9     # prevents log(0)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FeatureDriftResult:
    """PSI result for one feature.

    Attributes:
        feature:    Feature name.
        psi:        Population Stability Index value.
        status:     "ok" | "warning" | "critical".
        ref_mean:   Mean in reference window.
        recent_mean: Mean in recent window.
        delta_mean: Absolute difference of means (quick sanity check).
    """

    feature:     str
    psi:         float
    status:      str
    ref_mean:    float
    recent_mean: float
    delta_mean:  float


@dataclass
class DriftReport:
    """Full drift report across all features.

    Attributes:
        run_date:       Date the monitor was executed.
        reference_seasons: Seasons used as baseline.
        recent_days:    Rolling window (days) for the recent distribution.
        recent_cutoff:  Earliest date included in the recent window.
        features:       Per-feature PSI results sorted by PSI descending.
        n_features_ok:      Count of features with PSI < 0.10.
        n_features_warning: Count of features with 0.10 ≤ PSI < 0.25.
        n_features_critical: Count of features with PSI ≥ 0.25.
        recommendation: Human-readable action recommendation.
    """

    run_date:            str
    reference_seasons:   list[int]
    recent_days:         int
    recent_cutoff:       str
    features:            list[FeatureDriftResult] = field(default_factory=list)
    n_features_ok:       int = 0
    n_features_warning:  int = 0
    n_features_critical: int = 0
    recommendation:      str = ""


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def _compute_psi(ref: np.ndarray, recent: np.ndarray, n_bins: int = _N_BINS) -> float:
    """Computes PSI using equal-frequency binning on the reference distribution.

    Equal-frequency (quantile) bins are preferred over equal-width bins
    because they produce non-empty buckets even for skewed distributions,
    avoiding log(0) errors in the rare-event tails.

    Args:
        ref:    Reference period values, shape (N,).
        recent: Recent period values, shape (M,).
        n_bins: Number of quantile bins.

    Returns:
        Scalar PSI value ≥ 0.
    """
    if len(ref) < n_bins or len(recent) < 5:
        return 0.0

    # Bin edges from reference quantiles (equal-frequency)
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(ref, quantiles)
    # Deduplicate edges (constant features produce repeated quantile values)
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0

    ref_counts,    _ = np.histogram(ref,    bins=edges)
    recent_counts, _ = np.histogram(recent, bins=edges)

    ref_pct    = ref_counts    / (ref_counts.sum()    + _EPS)
    recent_pct = recent_counts / (recent_counts.sum() + _EPS)

    # Add epsilon to avoid log(0)
    ref_pct    = np.clip(ref_pct,    _EPS, None)
    recent_pct = np.clip(recent_pct, _EPS, None)

    psi = float(np.sum((recent_pct - ref_pct) * np.log(recent_pct / ref_pct)))
    return max(psi, 0.0)


def _status(psi: float) -> str:
    if psi < _PSI_GREEN:
        return "ok"
    if psi < _PSI_YELLOW:
        return "warning"
    return "critical"


# ---------------------------------------------------------------------------
# Main monitor class
# ---------------------------------------------------------------------------

class DriftMonitor:
    """Computes PSI drift for all numeric features in the Silver PA dataset.

    Args:
        silver_dir: Path to the Silver plate_appearances directory.
                    Expected structure: silver_dir/season=YYYY/data.parquet
    """

    # Features that are purely context / encoding — not subject to distribution
    # drift in the same way as batter stats (they're deterministic from the date).
    _EXCLUDE = {
        "batter_id", "pitcher_id", "game_date", "season",
        "batter_stand", "pitcher_throws", "last_pitch_type",
        "era_shift_ban", "era_universal_dh", "era_first_year_shift_ban",
        "pa_outcome_int",
    }

    def __init__(self, silver_dir: str = "data/silver/plate_appearances") -> None:
        self.silver_dir = Path(silver_dir)

    def _load_season(self, season: int) -> pl.DataFrame:
        pq = self.silver_dir / f"season={season}" / "data.parquet"
        if not pq.exists():
            log.warning("Season parquet not found: %s", pq)
            return pl.DataFrame()
        return pl.read_parquet(str(pq), hive_partitioning=False)

    def _load_recent(self, cutoff: date) -> pl.DataFrame:
        """Loads all Silver rows with game_date >= cutoff."""
        parts = []
        for sd in sorted(self.silver_dir.glob("season=*")):
            pq = sd / "data.parquet"
            if not pq.exists():
                continue
            df = pl.read_parquet(str(pq), hive_partitioning=False)
            if "game_date" in df.columns:
                df = df.filter(pl.col("game_date") >= pl.lit(str(cutoff)))
                if not df.is_empty():
                    parts.append(df)
        return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()

    def run(
        self,
        reference_seasons: list[int] | None = None,
        recent_days: int = 30,
    ) -> DriftReport:
        """Computes PSI for all numeric features.

        Args:
            reference_seasons: Seasons to use as reference baseline.
                               Defaults to [2022, 2023].
            recent_days:       Days back from today to define the recent window.

        Returns:
            DriftReport with per-feature PSI values and overall recommendation.
        """
        if reference_seasons is None:
            reference_seasons = [2022, 2023]

        cutoff = date.today() - timedelta(days=recent_days)
        report = DriftReport(
            run_date=str(date.today()),
            reference_seasons=reference_seasons,
            recent_days=recent_days,
            recent_cutoff=str(cutoff),
        )

        # Load reference data
        ref_parts = [self._load_season(s) for s in reference_seasons]
        ref_parts = [df for df in ref_parts if not df.is_empty()]
        if not ref_parts:
            log.error("No reference data found for seasons %s", reference_seasons)
            report.recommendation = "ERROR: No reference data available."
            return report
        ref_df = pl.concat(ref_parts, how="diagonal_relaxed")

        # Load recent data
        recent_df = self._load_recent(cutoff)
        if recent_df.is_empty():
            log.warning("No recent data found after %s", cutoff)
            report.recommendation = (
                f"Sin datos recientes posteriores a {cutoff}. "
                "Actualiza los Silver con statcast_ingestion.py."
            )
            return report

        log.info(
            "Drift monitor: ref=%d rows (%s), recent=%d rows (>=%s)",
            len(ref_df), reference_seasons, len(recent_df), cutoff,
        )

        # Identify numeric columns to monitor
        numeric_cols = [
            c for c in ref_df.columns
            if c not in self._EXCLUDE
            and ref_df[c].dtype in (
                pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt8,
            )
            and c in recent_df.columns
        ]

        results: list[FeatureDriftResult] = []
        for col in numeric_cols:
            ref_vals    = ref_df[col].drop_nulls().to_numpy().astype(np.float64)
            recent_vals = recent_df[col].drop_nulls().to_numpy().astype(np.float64)
            if len(ref_vals) < 10 or len(recent_vals) < 5:
                continue
            psi = _compute_psi(ref_vals, recent_vals)
            results.append(FeatureDriftResult(
                feature=col,
                psi=round(psi, 5),
                status=_status(psi),
                ref_mean=round(float(ref_vals.mean()), 5),
                recent_mean=round(float(recent_vals.mean()), 5),
                delta_mean=round(abs(float(ref_vals.mean()) - float(recent_vals.mean())), 5),
            ))

        results.sort(key=lambda x: x.psi, reverse=True)
        report.features            = results
        report.n_features_ok       = sum(1 for r in results if r.status == "ok")
        report.n_features_warning  = sum(1 for r in results if r.status == "warning")
        report.n_features_critical = sum(1 for r in results if r.status == "critical")

        if report.n_features_critical > 0:
            top_crit = [r.feature for r in results if r.status == "critical"][:3]
            report.recommendation = (
                f"ACCION REQUERIDA: {report.n_features_critical} features con drift "
                f"critico (PSI >= 0.25): {top_crit}. Ejecutar retrain del modelo."
            )
        elif report.n_features_warning > 0:
            report.recommendation = (
                f"MONITOREAR: {report.n_features_warning} features con drift moderado "
                f"(0.10 <= PSI < 0.25). Revisar en la proxima semana."
            )
        else:
            report.recommendation = (
                f"OK: Todas las features estables (PSI < 0.10, n={len(results)})."
            )

        return report

    def print_summary(self, report: DriftReport) -> None:
        """Prints a formatted drift summary to stdout."""
        print(f"\n{'='*60}")
        print(f"  Feature Drift Report — {report.run_date}")
        print(f"  Referencia: {report.reference_seasons}  |  "
              f"Reciente: ultimos {report.recent_days}d (>= {report.recent_cutoff})")
        print(f"{'='*60}")
        print(f"  OK:       {report.n_features_ok:>3} features (PSI < 0.10)")
        print(f"  WARNING:  {report.n_features_warning:>3} features (0.10 <= PSI < 0.25)")
        print(f"  CRITICAL: {report.n_features_critical:>3} features (PSI >= 0.25)")
        print(f"\n  {report.recommendation}\n")

        if report.features:
            print(f"  {'Feature':<30} {'PSI':>8}  {'Status':<10}  "
                  f"{'Ref mean':>10}  {'Now mean':>10}")
            print(f"  {'-'*30} {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
            for r in report.features[:15]:
                flag = {"ok": " ", "warning": "~", "critical": "!"}[r.status]
                print(f"  {flag}{r.feature:<29} {r.psi:>8.4f}  {r.status:<10}  "
                      f"{r.ref_mean:>10.4f}  {r.recent_mean:>10.4f}")
        print()

    def to_json(self, report: DriftReport) -> dict:
        """Serializes the DriftReport to a JSON-compatible dict."""
        return {
            "run_date":             report.run_date,
            "reference_seasons":    report.reference_seasons,
            "recent_days":          report.recent_days,
            "recent_cutoff":        report.recent_cutoff,
            "summary": {
                "n_ok":       report.n_features_ok,
                "n_warning":  report.n_features_warning,
                "n_critical": report.n_features_critical,
            },
            "recommendation": report.recommendation,
            "features": [
                {
                    "feature":      r.feature,
                    "psi":          r.psi,
                    "status":       r.status,
                    "ref_mean":     r.ref_mean,
                    "recent_mean":  r.recent_mean,
                    "delta_mean":   r.delta_mean,
                }
                for r in report.features
            ],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature drift monitor (PSI)")
    p.add_argument("--silver-dir",          default="data/silver/plate_appearances")
    p.add_argument("--reference-seasons",   nargs="+", type=int, default=[2022, 2023])
    p.add_argument("--recent-days",         type=int, default=30)
    p.add_argument("--output",              default=None,
                   help="Save JSON report to this path (optional)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    monitor = DriftMonitor(silver_dir=args.silver_dir)
    report  = monitor.run(
        reference_seasons=args.reference_seasons,
        recent_days=args.recent_days,
    )
    monitor.print_summary(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(monitor.to_json(report), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"  Reporte guardado: {out}\n")


if __name__ == "__main__":
    main()
