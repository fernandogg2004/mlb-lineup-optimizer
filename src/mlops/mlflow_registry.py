"""
MLflow Experiment Tracking and Model Registry interface.

Responsibilities:
  - Open/close MLflow runs with structured provenance tags
  - Log AtBatModelConfig hyperparameters as flat param dict
  - Log offline performance metrics: Log-Loss, classwise ECE, per-fold CV stats
  - Persist the pickle artifact and register the version in the Model Registry
  - Tag approved versions as "Production_Candidate" for human sign-off
  - Champion / Challenger comparison: new model only promotes if it strictly
    improves log-loss over the current Production alias

MLflow 2.x API note:
  Stage transitions ("Staging" → "Production") are deprecated.
  This module uses registered model *aliases* instead:
      pc.set_registered_model_alias(name, "production", version)
      pc.get_model_version_by_alias(name, "production")
  Tags are still set for human-readable filtering in the UI.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

import mlflow
import structlog
from mlflow.tracking import MlflowClient

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MLflowRegistryConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "at-bat-predictor"
    registered_model_name: str = "at-bat-predictor"
    artifact_path: str = "model"          # sub-path inside the MLflow run artifacts
    production_alias: str = "production"
    staging_alias: str = "staging"
    production_candidate_tag_key: str = "status"
    production_candidate_tag_value: str = "Production_Candidate"


# ---------------------------------------------------------------------------
# Metric containers
# ---------------------------------------------------------------------------


@dataclass
class ModelPerformanceMetrics:
    """Offline evaluation metrics produced by AtBatPredictor.evaluate_calibration()."""

    log_loss: float
    ece_overall: float
    per_class_ece: dict[str, float]           # {"OUT_IN_PLAY": 0.024, "HOME_RUN": 0.031, ...}
    per_class_logloss: dict[str, float] = field(default_factory=dict)
    cv_folds: list[dict[str, float]] = field(default_factory=list)  # walk-forward fold results
    training_samples: int = 0
    validation_samples: int = 0
    calibration_method: str = "isotonic"
    ece_target_met: bool = False

    @classmethod
    def from_evaluate_calibration_result(cls, result: dict[str, Any]) -> "ModelPerformanceMetrics":
        """Build from the dict returned by AtBatPredictor.evaluate_calibration()."""
        return cls(
            log_loss=result.get("overall_logloss", float("inf")),
            ece_overall=result.get("overall_ece", float("inf")),
            per_class_ece=result.get("per_class_ece", {}),
            per_class_logloss=result.get("per_class_logloss", {}),
            ece_target_met=result.get("ece_target_met", False),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class MLflowRegistry:
    """Centralised MLflow interaction layer.

    Usage (full training run):
        registry = MLflowRegistry.from_env()
        with registry.managed_run("daily-retrain-2026-05-18") as run:
            registry.log_model_config(config)
            registry.log_performance_metrics(metrics)
            registry.log_model_artifact(predictor, run.info.run_id)
        version = registry.register_production_candidate(run.info.run_id)

    Usage (champion / challenger gate):
        if registry.challenger_beats_champion(new_metrics):
            registry.promote_to_production(version)
    """

    def __init__(self, config: MLflowRegistryConfig | None = None) -> None:
        self._config = config or MLflowRegistryConfig()
        mlflow.set_tracking_uri(self._config.tracking_uri)
        self._client = MlflowClient(tracking_uri=self._config.tracking_uri)
        self._ensure_experiment()
        logger.info(
            "mlflow_registry_ready",
            tracking_uri=self._config.tracking_uri,
            experiment=self._config.experiment_name,
            model_name=self._config.registered_model_name,
        )

    @classmethod
    def from_env(cls, config: MLflowRegistryConfig | None = None) -> "MLflowRegistry":
        cfg = config or MLflowRegistryConfig(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "at-bat-predictor"),
            registered_model_name=os.environ.get("MLFLOW_MODEL_NAME", "at-bat-predictor"),
        )
        return cls(cfg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_experiment(self) -> None:
        if mlflow.get_experiment_by_name(self._config.experiment_name) is None:
            mlflow.create_experiment(
                name=self._config.experiment_name,
                artifact_location=f"s3://{os.environ.get('MLB_S3_BUCKET', 'mlb-lakehouse')}"
                                   f"/mlflow/{self._config.experiment_name}",
                tags={"project": "mlb-lineup-optimizer", "component": "at-bat-predictor"},
            )
            logger.info("mlflow_experiment_created", name=self._config.experiment_name)

    def _get_production_run_metrics(self) -> dict[str, float]:
        """Fetch metrics of the model currently tagged as Production alias."""
        try:
            mv = self._client.get_model_version_by_alias(
                name=self._config.registered_model_name,
                alias=self._config.production_alias,
            )
            run = self._client.get_run(mv.run_id)
            return run.data.metrics
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Context manager for runs
    # ------------------------------------------------------------------

    @contextmanager
    def managed_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """Context manager that wraps an MLflow run with standard tags."""
        mlflow.set_experiment(self._config.experiment_name)
        base_tags = {
            "pipeline": "at-bat-predictor",
            "environment": os.environ.get("MLB_ENVIRONMENT", "development"),
            "git_sha": os.environ.get("GIT_SHA", "unknown"),
        }
        if tags:
            base_tags.update(tags)
        with mlflow.start_run(run_name=run_name, tags=base_tags) as active_run:
            logger.info("mlflow_run_started", run_id=active_run.info.run_id, run_name=run_name)
            try:
                yield active_run
            except Exception:
                logger.exception("mlflow_run_failed", run_id=active_run.info.run_id)
                raise
            else:
                logger.info("mlflow_run_finished", run_id=active_run.info.run_id)

    # ------------------------------------------------------------------
    # Logging helpers (call these inside managed_run context)
    # ------------------------------------------------------------------

    def log_model_config(self, config: Any) -> None:
        """Log AtBatModelConfig dataclass fields as MLflow params."""
        params: dict[str, Any] = {}
        # Support both dataclass (asdict) and plain dict
        try:
            from dataclasses import asdict as _asdict
            params = _asdict(config)
        except TypeError:
            params = dict(config) if hasattr(config, "__iter__") else vars(config)

        # MLflow params must be str-valued
        mlflow.log_params({k: str(v) for k, v in params.items()})
        logger.debug("model_config_logged", n_params=len(params))

    def log_performance_metrics(self, metrics: ModelPerformanceMetrics) -> None:
        """Log all offline evaluation metrics to the active MLflow run."""
        flat: dict[str, float] = {
            "log_loss": metrics.log_loss,
            "ece_overall": metrics.ece_overall,
            "ece_target_met": float(metrics.ece_target_met),
            "training_samples": float(metrics.training_samples),
            "validation_samples": float(metrics.validation_samples),
        }
        # Per-class ECE
        for cls_name, ece_val in metrics.per_class_ece.items():
            flat[f"ece_{cls_name.lower()}"] = ece_val
        # Per-class log-loss
        for cls_name, ll_val in metrics.per_class_logloss.items():
            flat[f"logloss_{cls_name.lower()}"] = ll_val
        # Walk-forward CV summary
        if metrics.cv_folds:
            cv_log_losses = [f.get("overall_logloss", 0) for f in metrics.cv_folds]
            cv_eces = [f.get("overall_ece", 0) for f in metrics.cv_folds]
            import numpy as _np
            flat["cv_logloss_mean"] = float(_np.mean(cv_log_losses))
            flat["cv_logloss_std"] = float(_np.std(cv_log_losses))
            flat["cv_ece_mean"] = float(_np.mean(cv_eces))

        mlflow.log_metrics(flat)
        logger.info(
            "performance_metrics_logged",
            log_loss=metrics.log_loss,
            ece_overall=metrics.ece_overall,
            ece_target_met=metrics.ece_target_met,
        )

    def log_model_artifact(self, predictor: Any, run_id: str) -> str:
        """Save predictor to a temp file and log as MLflow artifact.

        Returns the artifact URI string (e.g. ``runs:/<run_id>/model/model.pkl``).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.pkl"
            predictor.save(str(model_path))
            mlflow.log_artifact(str(model_path), artifact_path=self._config.artifact_path)
            artifact_uri = f"runs:/{run_id}/{self._config.artifact_path}/model.pkl"
            logger.info("model_artifact_logged", uri=artifact_uri)
            return artifact_uri

    # ------------------------------------------------------------------
    # Model Registry
    # ------------------------------------------------------------------

    def register_production_candidate(
        self,
        run_id: str,
        description: str = "",
    ) -> str:
        """Register a trained model version as a Production Candidate.

        Args:
            run_id: MLflow run ID that produced the model artifact.
            description: Human-readable description for the Model Registry UI.

        Returns:
            Registered model version string (e.g. ``"7"``).
        """
        artifact_uri = f"runs:/{run_id}/{self._config.artifact_path}/model.pkl"
        mv = mlflow.register_model(
            model_uri=artifact_uri,
            name=self._config.registered_model_name,
            tags={
                self._config.production_candidate_tag_key: self._config.production_candidate_tag_value,
                "run_id": run_id,
            },
        )
        if description:
            self._client.update_model_version(
                name=self._config.registered_model_name,
                version=mv.version,
                description=description,
            )
        # Set staging alias for human review before production promotion
        self._client.set_registered_model_alias(
            name=self._config.registered_model_name,
            alias=self._config.staging_alias,
            version=mv.version,
        )
        logger.info(
            "model_version_registered",
            model_name=self._config.registered_model_name,
            version=mv.version,
            alias=self._config.staging_alias,
            tag=self._config.production_candidate_tag_value,
        )
        return str(mv.version)

    def challenger_beats_champion(self, challenger_metrics: ModelPerformanceMetrics) -> bool:
        """Champion / Challenger gate: returns True only if challenger improves log-loss.

        If no Production model exists, challenger wins by default (first deployment).
        """
        champion_metrics = self._get_production_run_metrics()
        if not champion_metrics:
            logger.info("no_champion_found_challenger_wins_by_default")
            return True

        champion_ll = float(champion_metrics.get("log_loss", float("inf")))
        challenger_ll = challenger_metrics.log_loss
        is_better = challenger_ll < champion_ll

        logger.info(
            "champion_challenger_comparison",
            champion_log_loss=champion_ll,
            challenger_log_loss=challenger_ll,
            challenger_wins=is_better,
        )
        return is_better

    def promote_to_production(self, version: str) -> None:
        """Assign the production alias to the specified model version.

        This is the final step in the deployment gate: after human sign-off
        on the Production_Candidate, this call makes the model live.
        """
        self._client.set_registered_model_alias(
            name=self._config.registered_model_name,
            alias=self._config.production_alias,
            version=version,
        )
        self._client.set_model_version_tag(
            name=self._config.registered_model_name,
            version=version,
            key=self._config.production_candidate_tag_key,
            value="Production",
        )
        logger.info(
            "model_promoted_to_production",
            model_name=self._config.registered_model_name,
            version=version,
            alias=self._config.production_alias,
        )

    def get_production_model_uri(self) -> str:
        """Returns the MLflow URI for the current Production model artifact."""
        mv = self._client.get_model_version_by_alias(
            name=self._config.registered_model_name,
            alias=self._config.production_alias,
        )
        return f"runs:/{mv.run_id}/{self._config.artifact_path}/model.pkl"

    def get_latest_versions(self, n: int = 5) -> list[dict[str, Any]]:
        """Returns metadata for the N most recent registered versions."""
        versions = self._client.search_model_versions(
            filter_string=f"name='{self._config.registered_model_name}'",
            max_results=n,
            order_by=["version_number DESC"],
        )
        return [
            {
                "version": mv.version,
                "run_id": mv.run_id,
                "tags": dict(mv.tags),
                "description": mv.description,
                "creation_timestamp": mv.creation_timestamp,
            }
            for mv in versions
        ]
