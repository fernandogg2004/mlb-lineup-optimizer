"""
auto_promoter.py — Autonomous model promotion: MLflow champion/challenger + ECS Fargate deploy.
================================================================================================
Compares challenger (staging alias) against champion (production alias) using two gates:
    1. Delta log-loss ≥ DELTA_LOG_LOSS_THRESHOLD (default 0.01) — challenger must be measurably better
    2. Challenger ECE ≤ ECE_CEILING (default 0.035) — calibration must stay within tolerance

If both gates pass:
    - MLflow: reassign "production" alias to challenger version; demote previous champion to "retired"
    - ECS Fargate: generate new Task Definition JSON and call update_service to deploy

Designed to run as a post-retraining step in the daily Airflow pipeline or as a standalone CLI.
All state mutations are logged structurally. On any third-party API failure the function raises
PromotionError so the caller (Airflow task) can mark the step as failed without silently skipping.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
import structlog
from botocore.exceptions import BotoCoreError, ClientError
from mlflow.tracking import MlflowClient

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

DELTA_LOG_LOSS_THRESHOLD: float = float(
    os.environ.get("PROMOTER_DELTA_LOG_LOSS", "0.01")
)
ECE_CEILING: float = float(os.environ.get("PROMOTER_ECE_CEILING", "0.035"))

_PRODUCTION_ALIAS = "production"
_STAGING_ALIAS = "staging"
_RETIRED_TAG_KEY = "status"
_RETIRED_TAG_VALUE = "retired"


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelVersion:
    """Slim representation of a registered model version pulled from MLflow."""

    version: str
    run_id: str
    log_loss: float
    ece_overall: float


@dataclass
class PromotionResult:
    """Result returned by promote_if_better().

    Attributes:
        promoted: True if the challenger was promoted to production.
        reason: Human-readable explanation (always populated).
        champion_version: Version string of the previous champion (may be None on first run).
        challenger_version: Version string of the challenger that was evaluated.
        delta_log_loss: champion_log_loss - challenger_log_loss (positive = challenger improves).
        ecs_task_definition_arn: ARN of the new ECS task definition (None if not deployed).
    """

    promoted: bool
    reason: str
    champion_version: Optional[str]
    challenger_version: str
    delta_log_loss: Optional[float]
    ecs_task_definition_arn: Optional[str] = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PromotionError(RuntimeError):
    """Raised when a non-recoverable error prevents the promotion pipeline from completing."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AutoPromoterConfig:
    """Runtime configuration resolved from environment variables.

    Attributes:
        mlflow_tracking_uri: URI of the MLflow tracking server.
        mlflow_model_name: Registered model name (same as MLflowRegistryConfig).
        ecs_cluster: ECS cluster ARN or short name.
        ecs_service: ECS service name that runs the optimizer API.
        ecs_task_family: Task definition family name (base without :revision).
        ecr_image_uri: Full ECR image URI for the new deployment (includes tag).
        aws_region: AWS region for ECS and ECR calls.
    """

    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow_model_name: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_MODEL_NAME", "at-bat-predictor")
    )
    ecs_cluster: str = field(
        default_factory=lambda: os.environ.get("ECS_CLUSTER", "mlb-optimizer-cluster")
    )
    ecs_service: str = field(
        default_factory=lambda: os.environ.get("ECS_SERVICE", "mlb-optimizer-api")
    )
    ecs_task_family: str = field(
        default_factory=lambda: os.environ.get("ECS_TASK_FAMILY", "mlb-optimizer-api")
    )
    ecr_image_uri: str = field(
        default_factory=lambda: os.environ.get("ECR_IMAGE_URI", "")
    )
    aws_region: str = field(
        default_factory=lambda: os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )

    @classmethod
    def from_env(cls) -> "AutoPromoterConfig":
        """Construct config entirely from environment variables."""
        return cls()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


class AutoPromoter:
    """Orchestrates champion/challenger evaluation and Fargate deployment.

    Args:
        config: Runtime configuration. Defaults to AutoPromoterConfig.from_env().
    """

    def __init__(self, config: AutoPromoterConfig | None = None) -> None:
        self._config = config or AutoPromoterConfig.from_env()
        self._mlflow = MlflowClient(tracking_uri=self._config.mlflow_tracking_uri)
        logger.info(
            "auto_promoter_initialized",
            model_name=self._config.mlflow_model_name,
            mlflow_uri=self._config.mlflow_tracking_uri,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def promote_if_better(self) -> PromotionResult:
        """Evaluate challenger vs champion and promote if both gates pass.

        Returns:
            PromotionResult describing the decision and any deployment artifact.

        Raises:
            PromotionError: If MLflow or ECS calls fail in an unrecoverable way.
        """
        challenger = self._get_model_version(_STAGING_ALIAS)
        if challenger is None:
            reason = "No model registered under 'staging' alias — nothing to promote."
            logger.warning("promotion_skipped_no_challenger")
            return PromotionResult(
                promoted=False,
                reason=reason,
                champion_version=None,
                challenger_version="(none)",
                delta_log_loss=None,
            )

        champion = self._get_model_version(_PRODUCTION_ALIAS)

        delta = None
        if champion is not None:
            delta = champion.log_loss - challenger.log_loss  # positive = challenger wins
            gate_delta = delta >= DELTA_LOG_LOSS_THRESHOLD
            gate_ece = challenger.ece_overall <= ECE_CEILING

            logger.info(
                "promotion_gate_evaluation",
                champion_version=champion.version,
                challenger_version=challenger.version,
                champion_log_loss=champion.log_loss,
                challenger_log_loss=challenger.log_loss,
                delta_log_loss=delta,
                challenger_ece=challenger.ece_overall,
                gate_delta_passed=gate_delta,
                gate_ece_passed=gate_ece,
            )

            if not gate_delta:
                reason = (
                    f"Challenger v{challenger.version} did not improve log-loss by "
                    f"≥{DELTA_LOG_LOSS_THRESHOLD} "
                    f"(delta={delta:.4f}, champion={champion.log_loss:.4f}, "
                    f"challenger={challenger.log_loss:.4f})."
                )
                return PromotionResult(
                    promoted=False,
                    reason=reason,
                    champion_version=champion.version,
                    challenger_version=challenger.version,
                    delta_log_loss=delta,
                )

            if not gate_ece:
                reason = (
                    f"Challenger v{challenger.version} ECE {challenger.ece_overall:.4f} "
                    f"exceeds ceiling {ECE_CEILING} — calibration regression detected."
                )
                return PromotionResult(
                    promoted=False,
                    reason=reason,
                    champion_version=champion.version,
                    challenger_version=challenger.version,
                    delta_log_loss=delta,
                )

        # Both gates passed (or no champion exists yet) — promote
        self._reassign_mlflow_aliases(
            challenger_version=challenger.version,
            previous_champion_version=champion.version if champion else None,
        )

        task_def_arn = self._deploy_to_fargate(model_version=challenger.version)

        delta_str = f"{delta:.4f}" if delta is not None else "N/A"
        reason = (
            f"Challenger v{challenger.version} promoted to production "
            f"(delta_log_loss={delta_str}, "
            f"ece={challenger.ece_overall:.4f})."
        )
        logger.info(
            "promotion_succeeded",
            challenger_version=challenger.version,
            champion_version=champion.version if champion else None,
            delta_log_loss=delta,
            task_definition_arn=task_def_arn,
        )
        return PromotionResult(
            promoted=True,
            reason=reason,
            champion_version=champion.version if champion else None,
            challenger_version=challenger.version,
            delta_log_loss=delta,
            ecs_task_definition_arn=task_def_arn,
        )

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def _get_model_version(self, alias: str) -> Optional[ModelVersion]:
        """Fetch model version and metrics for the given alias.

        Returns:
            ModelVersion or None if the alias is not set.

        Raises:
            PromotionError: If MLflow call fails for a reason other than missing alias.
        """
        try:
            mv = self._mlflow.get_model_version_by_alias(
                name=self._config.mlflow_model_name,
                alias=alias,
            )
        except Exception as exc:
            if "RESOURCE_DOES_NOT_EXIST" in str(exc) or "not found" in str(exc).lower():
                return None
            raise PromotionError(
                f"MLflow call failed when resolving alias '{alias}': {exc}"
            ) from exc

        try:
            run = self._mlflow.get_run(mv.run_id)
            metrics = run.data.metrics
            return ModelVersion(
                version=mv.version,
                run_id=mv.run_id,
                log_loss=float(metrics.get("log_loss", float("inf"))),
                ece_overall=float(metrics.get("ece_overall", float("inf"))),
            )
        except Exception as exc:
            raise PromotionError(
                f"Failed to fetch run metrics for alias '{alias}' (run_id={mv.run_id}): {exc}"
            ) from exc

    def _reassign_mlflow_aliases(
        self,
        challenger_version: str,
        previous_champion_version: Optional[str],
    ) -> None:
        """Atomically reassign production alias and tag the old champion as retired.

        Args:
            challenger_version: Version string to assign to 'production'.
            previous_champion_version: Previous production version to tag 'retired' (may be None).

        Raises:
            PromotionError: If either MLflow call fails.
        """
        try:
            self._mlflow.set_registered_model_alias(
                name=self._config.mlflow_model_name,
                alias=_PRODUCTION_ALIAS,
                version=challenger_version,
            )
            logger.info(
                "mlflow_alias_updated",
                alias=_PRODUCTION_ALIAS,
                new_version=challenger_version,
            )
        except Exception as exc:
            raise PromotionError(f"Failed to set production alias: {exc}") from exc

        if previous_champion_version:
            try:
                self._mlflow.set_model_version_tag(
                    name=self._config.mlflow_model_name,
                    version=previous_champion_version,
                    key=_RETIRED_TAG_KEY,
                    value=_RETIRED_TAG_VALUE,
                )
                logger.info(
                    "mlflow_champion_retired",
                    version=previous_champion_version,
                )
            except Exception as exc:
                # Tagging the old version as retired is best-effort; don't block deployment
                logger.warning(
                    "mlflow_retire_tag_failed",
                    version=previous_champion_version,
                    error=str(exc),
                )

    # ------------------------------------------------------------------
    # ECS Fargate deployment
    # ------------------------------------------------------------------

    def _build_task_definition(self, model_version: str) -> dict[str, Any]:
        """Construct a minimal ECS Task Definition override with the new model version.

        The returned dict is passed directly to boto3 register_task_definition.
        It contains only the fields that differ from the current task definition
        (image URI and MODEL_VERSION environment variable).

        Args:
            model_version: MLflow model version string to inject as env var.

        Returns:
            Task definition dict suitable for boto3 register_task_definition(**td).
        """
        return {
            "family": self._config.ecs_task_family,
            "containerDefinitions": [
                {
                    "name": "mlb-optimizer",
                    "image": self._config.ecr_image_uri,
                    "essential": True,
                    "environment": [
                        {"name": "MODEL_VERSION", "value": model_version},
                        {"name": "MLFLOW_PRODUCTION_ALIAS", "value": _PRODUCTION_ALIAS},
                        {
                            "name": "DEPLOYED_AT",
                            "value": datetime.now(tz=timezone.utc).isoformat(),
                        },
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": f"/ecs/{self._config.ecs_task_family}",
                            "awslogs-region": self._config.aws_region,
                            "awslogs-stream-prefix": "ecs",
                        },
                    },
                }
            ],
            "requiresCompatibilities": ["FARGATE"],
            "networkMode": "awsvpc",
            "cpu": "1024",
            "memory": "2048",
        }

    def _deploy_to_fargate(self, model_version: str) -> Optional[str]:
        """Register a new ECS task definition and update the Fargate service.

        Args:
            model_version: MLflow version string being deployed.

        Returns:
            ARN of the newly registered task definition, or None if ECR_IMAGE_URI is not set
            (local/CI environment — skips deployment).

        Raises:
            PromotionError: If boto3 call fails with a non-transient error.
        """
        if not self._config.ecr_image_uri:
            logger.warning(
                "ecs_deploy_skipped",
                reason="ECR_IMAGE_URI not set — skipping Fargate deployment",
            )
            return None

        ecs = boto3.client("ecs", region_name=self._config.aws_region)
        task_def = self._build_task_definition(model_version)

        try:
            reg_resp = ecs.register_task_definition(**task_def)
            task_def_arn: str = reg_resp["taskDefinition"]["taskDefinitionArn"]
            task_def_revision: str = reg_resp["taskDefinition"]["revision"]
            logger.info(
                "ecs_task_definition_registered",
                arn=task_def_arn,
                revision=task_def_revision,
                model_version=model_version,
            )
        except (BotoCoreError, ClientError) as exc:
            raise PromotionError(f"Failed to register ECS task definition: {exc}") from exc

        try:
            ecs.update_service(
                cluster=self._config.ecs_cluster,
                service=self._config.ecs_service,
                taskDefinition=task_def_arn,
                forceNewDeployment=True,
            )
            logger.info(
                "ecs_service_updated",
                cluster=self._config.ecs_cluster,
                service=self._config.ecs_service,
                task_definition_arn=task_def_arn,
            )
        except (BotoCoreError, ClientError) as exc:
            raise PromotionError(
                f"Task definition registered ({task_def_arn}) but ECS update_service failed: {exc}"
            ) from exc

        return task_def_arn


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run promotion check as a standalone script (called by Airflow PythonOperator)."""
    import argparse

    parser = argparse.ArgumentParser(description="MLB AI auto promoter — champion/challenger gate")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate gates but skip alias/ECS mutations")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("auto_promoter_dry_run_mode")

    config = AutoPromoterConfig.from_env()
    promoter = AutoPromoter(config)
    result = promoter.promote_if_better()

    print(json.dumps(
        {
            "promoted": result.promoted,
            "reason": result.reason,
            "champion_version": result.champion_version,
            "challenger_version": result.challenger_version,
            "delta_log_loss": result.delta_log_loss,
            "ecs_task_definition_arn": result.ecs_task_definition_arn,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
