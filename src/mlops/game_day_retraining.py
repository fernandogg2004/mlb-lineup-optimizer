"""
Daily Game-Day Retraining Pipeline.

Runs every morning after the previous night's MLB games are processed by
the Statcast ingestion pipeline (Phase 1) and the feature engineering layer
(Phase 2). Designed to be triggered by Argo Workflows, Apache Airflow, or
APScheduler in local/dev environments.

Pipeline steps (sequential, each can fail independently):
    1. Download last-day Statcast data via pybaseball → Bronze Delta
    2. Rebuild rolling features (RollingFeaturesEngine) up to today
    3. Load full Gold feature matrix (train + holdout split by date)
    4. Train AtBatPredictor on the training window
    5. Evaluate on the holdout set → check log-loss ≤ 0.84
    6. Compare challenger vs. current Production champion (MLflow)
    7. If both gates pass: register as Production_Candidate + set staging alias
    8. If either gate fails: emit a structured alert and abort

Production gate:
    Log-Loss threshold = 0.84 (configurable via RetrainingConfig).
    If the newly trained model's log-loss exceeds this threshold on the
    holdout set, the pipeline aborts and sends an alert instead of
    registering the model. The Production alias is NOT updated.

Alert mechanism:
    In local/dev: structlog ERROR + write to alert log file.
    In production: wire to AWS SNS via boto3 (publish_alert_sns stub below).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
import structlog
from sklearn.metrics import log_loss

from src.mlops.mlflow_registry import (
    MLflowRegistry,
    MLflowRegistryConfig,
    ModelPerformanceMetrics,
)
from src.models.model_at_bat import AtBatModelConfig, AtBatPredictor

logger = structlog.get_logger(__name__)

_LOG_LOSS_THRESHOLD = 0.84  # Hard gate: abort if new model exceeds this
_ECE_THRESHOLD = 0.045       # Soft gate: warning only (ECE target slightly relaxed for daily retrain)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RetrainingConfig:
    # Feature store paths (Gold Parquet layer)
    gold_plate_appearances_path: Path = Path("data/gold/plate_appearances")
    gold_rolling_features_path: Path = Path("data/gold/rolling_features")
    gold_platoon_features_path: Path = Path("data/gold/platoon_features")

    # Holdout window
    holdout_days: int = 14
    min_training_samples: int = 10_000
    min_holdout_samples: int = 1_000

    # Model quality gate
    log_loss_threshold: float = _LOG_LOSS_THRESHOLD
    ece_threshold: float = _ECE_THRESHOLD

    # Model training
    model_config: AtBatModelConfig = field(default_factory=AtBatModelConfig)
    model_output_path: Path = Path("models/")

    # MLflow
    mlflow_config: MLflowRegistryConfig = field(default_factory=MLflowRegistryConfig)
    run_name_prefix: str = "daily-retrain"

    # Alert log (structlog writes here; SNS in production)
    alert_log_path: Path = Path("logs/retraining_alerts.jsonl")

    # Whether to auto-promote if challenger beats champion
    auto_promote_on_improvement: bool = False  # require human sign-off in production


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------


@dataclass
class RetrainingResult:
    success: bool
    game_date: str
    log_loss: float
    ece_overall: float
    model_version: Optional[str]
    mlflow_run_id: Optional[str]
    training_samples: int
    validation_samples: int
    elapsed_seconds: float
    alert_message: Optional[str] = None
    challenger_beats_champion: bool = False


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------


class FeatureMatrixBuilder:
    """Assembles the (X, y, time_index) triplet from Gold Parquet files.

    Joins: rolling features + platoon splits + raw PA results.
    Feature columns are sorted lexicographically to guarantee schema stability
    across retraining runs.
    """

    # Feature column prefixes written by Phases 2/3 feature pipelines
    _ROLLING_PREFIXES = ("xwoba_", "ev_mean_", "k_rate_", "bb_rate_", "hr_rate_", "babip_")
    _PLATOON_PREFIXES = ("woba_vs_", "iso_vs_", "k_rate_vs_", "bb_rate_vs_")

    def __init__(self, config: RetrainingConfig) -> None:
        self._config = config

    def load(
        self, as_of_date: date
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and join all Gold features up to as_of_date.

        Returns:
            X: float32 feature matrix (N, D)
            y: int32 label array (N,) — PAOutcome integer values
            time_idx: int64 array (N,) — game_date as YYYYMMDD for temporal CV
        """
        logger.info("loading_features", as_of_date=as_of_date.isoformat())

        # Load plate appearances (labels + join keys)
        pa_df = pl.read_parquet(
            self._config.gold_plate_appearances_path,
            hive_partitioning=True,
        ).filter(pl.col("game_date") <= as_of_date.isoformat())

        if pa_df.is_empty():
            raise ValueError(f"No plate appearances found up to {as_of_date}")

        # Load rolling features (one row per batter_id × game_date)
        rolling_df = pl.read_parquet(self._config.gold_rolling_features_path)

        # Load platoon splits (one row per batter_id × season × matchup_type)
        platoon_df = pl.read_parquet(self._config.gold_platoon_features_path)

        # Join rolling features (latest available as of each game_date)
        combined = pa_df.join(
            rolling_df,
            on=["batter_id", "game_date"],
            how="left",
        ).join(
            platoon_df.select(
                ["batter_id", "season"]
                + [c for c in platoon_df.columns if c.startswith(self._PLATOON_PREFIXES)]
            ),
            on=["batter_id", "season"],
            how="left",
        )

        # Select feature columns (all numeric non-key columns)
        label_col = "pa_outcome_int"
        key_cols = {"batter_id", "game_date", "season", label_col, "pa_id", "pitcher_id"}
        feature_cols = sorted(
            c for c in combined.columns
            if c not in key_cols and combined[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
        )

        combined = combined.with_columns(
            [pl.col(c).fill_null(0.0).cast(pl.Float32) for c in feature_cols]
        )

        X = combined.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)
        y = combined[label_col].to_numpy().astype(np.int32)
        time_idx = (
            combined["game_date"]
            .str.replace("-", "")
            .cast(pl.Int64)
            .to_numpy()
            .astype(np.int64)
        )

        logger.info(
            "feature_matrix_built",
            n_rows=X.shape[0],
            n_features=X.shape[1],
            as_of_date=as_of_date.isoformat(),
        )
        return X, y, time_idx


# ---------------------------------------------------------------------------
# Statcast downloader (thin wrapper over Phase 1 ingestion pipeline)
# ---------------------------------------------------------------------------


def _download_gameday_statcast(game_date: date) -> int:
    """Download Statcast data for game_date into the Bronze layer.

    Wraps the Phase 1 ingestion pipeline. Returns number of pitches ingested.
    Idempotent: re-running on the same date is safe (Delta merge semantics).
    """
    from src.ingestion.statcast_ingestion import StatcastIngestionPipeline

    logger.info("downloading_statcast", game_date=game_date.isoformat())
    try:
        pipeline = StatcastIngestionPipeline.from_env()
        n_pitches = pipeline.ingest_date_range(
            start_date=game_date,
            end_date=game_date,
        )
        logger.info("statcast_download_complete", game_date=game_date.isoformat(), pitches=n_pitches)
        return n_pitches
    except Exception as exc:
        logger.error("statcast_download_failed", game_date=game_date.isoformat(), error=str(exc))
        raise


def _update_rolling_features(as_of_date: date) -> None:
    """Rebuild rolling feature store up to as_of_date."""
    from src.features.features_rolling import RollingFeaturesEngine

    logger.info("updating_rolling_features", as_of_date=as_of_date.isoformat())
    engine = RollingFeaturesEngine.from_env()
    engine.run(as_of_date=as_of_date)
    logger.info("rolling_features_updated", as_of_date=as_of_date.isoformat())


# ---------------------------------------------------------------------------
# Alert mechanism
# ---------------------------------------------------------------------------


def _send_alert(
    reason: str,
    metrics: dict[str, Any],
    alert_log_path: Path,
) -> None:
    """Emit a structured retraining failure alert.

    Writes to a JSONL alert log (for local/dev). In production, replace the
    boto3 SNS stub below with a real publish call.
    """
    alert_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": "CRITICAL",
        "pipeline": "game-day-retraining",
        "reason": reason,
        "metrics": metrics,
    }
    # Structured log — captured by CloudWatch / Loki in production
    logger.error("RETRAINING_ALERT", **alert_payload)

    # Append to local alert JSONL file
    alert_log_path.parent.mkdir(parents=True, exist_ok=True)
    with alert_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(alert_payload) + "\n")

    # Production SNS stub (uncomment and configure ARN):
    # import boto3
    # sns = boto3.client("sns", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    # sns.publish(
    #     TopicArn=os.environ["RETRAINING_ALERT_SNS_ARN"],
    #     Subject="[MLB AI] Retraining gate failure",
    #     Message=json.dumps(alert_payload, indent=2),
    # )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class GameDayRetrainingPipeline:
    """Orchestrates the full daily retraining loop.

    Designed to be called by Argo Workflows, Apache Airflow (as a PythonOperator),
    or APScheduler for local development.

    Usage:
        pipeline = GameDayRetrainingPipeline.from_env()
        result = pipeline.run(game_date=date.today() - timedelta(days=1))
        if not result.success:
            raise RuntimeError(result.alert_message)
    """

    def __init__(
        self,
        config: RetrainingConfig | None = None,
    ) -> None:
        self._config = config or RetrainingConfig()
        self._registry = MLflowRegistry(self._config.mlflow_config)
        self._feature_builder = FeatureMatrixBuilder(self._config)
        self._config.model_output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, config: RetrainingConfig | None = None) -> "GameDayRetrainingPipeline":
        if config is None:
            config = RetrainingConfig(
                mlflow_config=MLflowRegistryConfig(
                    tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                    experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "at-bat-predictor"),
                    registered_model_name=os.environ.get("MLFLOW_MODEL_NAME", "at-bat-predictor"),
                ),
            )
        return cls(config)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _build_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time_idx: np.ndarray,
        as_of_date: date,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Temporal train / holdout split with strict future-leak prevention."""
        cutoff = int(
            (as_of_date - timedelta(days=self._config.holdout_days))
            .isoformat()
            .replace("-", "")
        )
        train_mask = time_idx < cutoff
        val_mask = time_idx >= cutoff

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        if len(X_train) < self._config.min_training_samples:
            raise ValueError(
                f"Insufficient training samples: {len(X_train)} < "
                f"{self._config.min_training_samples}"
            )
        if len(X_val) < self._config.min_holdout_samples:
            raise ValueError(
                f"Insufficient holdout samples: {len(X_val)} < "
                f"{self._config.min_holdout_samples}"
            )

        logger.info(
            "dataset_split",
            training_samples=len(X_train),
            holdout_samples=len(X_val),
            cutoff_date=cutoff,
        )
        return X_train, y_train, X_val, y_val

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> AtBatPredictor:
        """Train a fresh AtBatPredictor on the full training window."""
        predictor = AtBatPredictor(config=self._config.model_config)
        predictor.fit(X_train, y_train, X_val, y_val)
        return predictor

    def _evaluate_model(
        self,
        predictor: AtBatPredictor,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ModelPerformanceMetrics:
        eval_result = predictor.evaluate_calibration(X_val, y_val)
        metrics = ModelPerformanceMetrics.from_evaluate_calibration_result(eval_result)
        metrics.validation_samples = len(X_val)
        return metrics

    def _check_quality_gate(
        self,
        metrics: ModelPerformanceMetrics,
        game_date: date,
        alert_context: dict[str, Any],
    ) -> bool:
        """Returns True if the model passes all quality gates."""
        if metrics.log_loss > self._config.log_loss_threshold:
            reason = (
                f"Log-loss {metrics.log_loss:.5f} exceeds threshold "
                f"{self._config.log_loss_threshold:.2f} — aborting deployment"
            )
            _send_alert(
                reason=reason,
                metrics=alert_context,
                alert_log_path=self._config.alert_log_path,
            )
            return False

        if metrics.ece_overall > self._config.ece_threshold:
            logger.warning(
                "ece_above_soft_threshold",
                ece=metrics.ece_overall,
                threshold=self._config.ece_threshold,
            )
            # ECE is a soft gate: log warning but do NOT abort

        return True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, game_date: date | None = None) -> RetrainingResult:
        """Execute the full daily retraining pipeline for game_date.

        Args:
            game_date: Date of the games to incorporate. Defaults to yesterday.

        Returns:
            RetrainingResult with outcome, metrics, and optional alert message.
        """
        if game_date is None:
            game_date = date.today() - timedelta(days=1)

        as_of_date = game_date + timedelta(days=1)  # features available next morning
        run_name = f"{self._config.run_name_prefix}-{game_date.isoformat()}"
        start_ts = time.perf_counter()

        logger.info("retraining_pipeline_start", game_date=game_date.isoformat(), run_name=run_name)

        # Step 1: Download new Statcast data
        try:
            _download_gameday_statcast(game_date)
        except Exception as exc:
            logger.warning("statcast_download_skipped", error=str(exc))

        # Step 2: Update rolling features
        try:
            _update_rolling_features(as_of_date)
        except Exception as exc:
            logger.warning("rolling_feature_update_skipped", error=str(exc))

        # Steps 3–7: train, evaluate, register — all inside a single MLflow run
        model_version: str | None = None
        mlflow_run_id: str | None = None
        alert_message: str | None = None
        challenger_won = False

        try:
            # Step 3: Load features
            X, y, time_idx = self._feature_builder.load(as_of_date)
            X_train, y_train, X_val, y_val = self._build_splits(X, y, time_idx, as_of_date)

            with self._registry.managed_run(
                run_name=run_name,
                tags={"game_date": game_date.isoformat(), "trigger": "daily_retrain"},
            ) as active_run:
                mlflow_run_id = active_run.info.run_id

                # Step 4: Log config
                self._registry.log_model_config(self._config.model_config)

                # Step 5: Train
                logger.info("training_model", run_id=mlflow_run_id)
                predictor = self._train_model(X_train, y_train, X_val, y_val)

                # Step 6: Evaluate
                metrics = self._evaluate_model(predictor, X_val, y_val)
                metrics.training_samples = len(X_train)
                metrics.validation_samples = len(X_val)
                self._registry.log_performance_metrics(metrics)

                alert_context = {
                    "game_date": game_date.isoformat(),
                    "run_id": mlflow_run_id,
                    "log_loss": metrics.log_loss,
                    "ece_overall": metrics.ece_overall,
                    "training_samples": metrics.training_samples,
                    "holdout_samples": metrics.validation_samples,
                    "log_loss_threshold": self._config.log_loss_threshold,
                }

                # Step 7: Quality gate
                if not self._check_quality_gate(metrics, game_date, alert_context):
                    elapsed = time.perf_counter() - start_ts
                    alert_message = (
                        f"Log-Loss {metrics.log_loss:.5f} > "
                        f"threshold {self._config.log_loss_threshold:.2f}"
                    )
                    logger.error("retraining_aborted_quality_gate", **alert_context)
                    return RetrainingResult(
                        success=False,
                        game_date=game_date.isoformat(),
                        log_loss=metrics.log_loss,
                        ece_overall=metrics.ece_overall,
                        model_version=None,
                        mlflow_run_id=mlflow_run_id,
                        training_samples=len(X_train),
                        validation_samples=len(X_val),
                        elapsed_seconds=elapsed,
                        alert_message=alert_message,
                    )

                # Step 8: Champion / Challenger comparison
                challenger_won = self._registry.challenger_beats_champion(metrics)

                # Step 9: Log artifact + register
                self._registry.log_model_artifact(predictor, mlflow_run_id)
                model_version = self._registry.register_production_candidate(
                    run_id=mlflow_run_id,
                    description=(
                        f"Daily retrain {game_date.isoformat()} — "
                        f"LogLoss={metrics.log_loss:.5f}, ECE={metrics.ece_overall:.5f}"
                    ),
                )

                # Auto-promote if configured AND challenger won
                if self._config.auto_promote_on_improvement and challenger_won:
                    self._registry.promote_to_production(model_version)
                    logger.info(
                        "model_auto_promoted",
                        version=model_version,
                        log_loss=metrics.log_loss,
                    )

        except Exception as exc:
            elapsed = time.perf_counter() - start_ts
            alert_message = f"Retraining pipeline crashed: {exc}"
            _send_alert(
                reason=alert_message,
                metrics={"game_date": game_date.isoformat(), "error": str(exc)},
                alert_log_path=self._config.alert_log_path,
            )
            logger.exception("retraining_pipeline_crash", game_date=game_date.isoformat())
            return RetrainingResult(
                success=False,
                game_date=game_date.isoformat(),
                log_loss=float("inf"),
                ece_overall=float("inf"),
                model_version=None,
                mlflow_run_id=mlflow_run_id,
                training_samples=0,
                validation_samples=0,
                elapsed_seconds=elapsed,
                alert_message=alert_message,
            )

        elapsed = time.perf_counter() - start_ts
        logger.info(
            "retraining_pipeline_success",
            game_date=game_date.isoformat(),
            model_version=model_version,
            log_loss=metrics.log_loss,
            ece_overall=metrics.ece_overall,
            elapsed_seconds=round(elapsed, 1),
            challenger_beats_champion=challenger_won,
        )
        return RetrainingResult(
            success=True,
            game_date=game_date.isoformat(),
            log_loss=metrics.log_loss,
            ece_overall=metrics.ece_overall,
            model_version=model_version,
            mlflow_run_id=mlflow_run_id,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            elapsed_seconds=elapsed,
            challenger_beats_champion=challenger_won,
        )


# ---------------------------------------------------------------------------
# Scheduler entry point (APScheduler for local dev; replace with Argo/Airflow in prod)
# ---------------------------------------------------------------------------


def schedule_daily_retraining() -> None:
    """Start the APScheduler cron job for daily retraining at 06:00 ET.

    In production, this function is NOT called directly. The Argo Workflow CronWorkflow
    or Airflow DAG triggers ``GameDayRetrainingPipeline.run()`` via a Kubernetes Job.
    This entry point exists for local development and integration testing only.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="America/New_York")
    pipeline = GameDayRetrainingPipeline.from_env()

    def _daily_job() -> None:
        game_date = date.today() - timedelta(days=1)
        result = pipeline.run(game_date=game_date)
        if not result.success:
            logger.error("scheduled_retraining_failed", alert=result.alert_message)

    scheduler.add_job(
        _daily_job,
        trigger=CronTrigger(hour=6, minute=0),
        id="daily_retraining",
        replace_existing=True,
    )
    logger.info("scheduler_started", job="daily_retraining", schedule="06:00 ET")
    scheduler.start()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target_date = date.fromisoformat(sys.argv[1])
    else:
        target_date = date.today() - timedelta(days=1)

    result = GameDayRetrainingPipeline.from_env().run(game_date=target_date)
    print(f"Success: {result.success}")
    print(f"Log-Loss: {result.log_loss:.5f}")
    print(f"ECE:      {result.ece_overall:.5f}")
    print(f"Version:  {result.model_version}")
    print(f"Elapsed:  {result.elapsed_seconds:.1f}s")
    if not result.success:
        sys.exit(1)
