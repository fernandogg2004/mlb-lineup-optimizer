"""
Tests for AutoPromoter — MLflow champion/challenger gate and ECS Fargate deployment.

All MLflow and AWS calls are mocked; no running MLflow server or AWS credentials needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from src.mlops.auto_promoter import (
    DELTA_LOG_LOSS_THRESHOLD,
    ECE_CEILING,
    AutoPromoter,
    AutoPromoterConfig,
    PromotionError,
)

MODEL_NAME = "test-model"
TASK_DEF_ARN = "arn:aws:ecs:us-east-1:123456789:task-definition/test-task:42"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(ecr_image_uri: str = "") -> AutoPromoterConfig:
    return AutoPromoterConfig(
        mlflow_tracking_uri="http://mock:5000",
        mlflow_model_name=MODEL_NAME,
        ecs_cluster="cluster",
        ecs_service="service",
        ecs_task_family="family",
        ecr_image_uri=ecr_image_uri,
        aws_region="us-east-1",
    )


def _mv(version: str, run_id: str) -> MagicMock:
    """Minimal ModelVersion mock for get_model_version_by_alias."""
    mv = MagicMock()
    mv.version = version
    mv.run_id = run_id
    return mv


def _run(log_loss: float, ece: float) -> MagicMock:
    """Minimal Run mock for get_run."""
    run = MagicMock()
    run.data.metrics = {"log_loss": log_loss, "ece_overall": ece}
    return run


def _make_promoter(mock_mlflow: MagicMock, ecr_image_uri: str = "") -> AutoPromoter:
    with patch("src.mlops.auto_promoter.MlflowClient", return_value=mock_mlflow):
        return AutoPromoter(_config(ecr_image_uri=ecr_image_uri))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mlflow() -> MagicMock:
    return MagicMock()


@pytest.fixture
def promoter(mock_mlflow: MagicMock) -> AutoPromoter:
    return _make_promoter(mock_mlflow)


# ---------------------------------------------------------------------------
# No challenger
# ---------------------------------------------------------------------------


def test_no_challenger_returns_not_promoted(promoter: AutoPromoter, mock_mlflow: MagicMock):
    mock_mlflow.get_model_version_by_alias.side_effect = Exception(
        "RESOURCE_DOES_NOT_EXIST: No alias 'staging'"
    )
    result = promoter.promote_if_better()
    assert result.promoted is False
    assert result.challenger_version == "(none)"
    assert result.delta_log_loss is None


def test_no_challenger_does_not_call_ecs(promoter: AutoPromoter, mock_mlflow: MagicMock):
    mock_mlflow.get_model_version_by_alias.side_effect = Exception("RESOURCE_DOES_NOT_EXIST")
    with patch("src.mlops.auto_promoter.boto3") as mock_boto3:
        promoter.promote_if_better()
    mock_boto3.client.assert_not_called()


# ---------------------------------------------------------------------------
# Gate: delta log-loss
# ---------------------------------------------------------------------------


def _setup_champion_challenger(
    mock_mlflow: MagicMock,
    champ_log_loss: float,
    champ_ece: float,
    chal_log_loss: float,
    chal_ece: float,
) -> None:
    champ_mv = _mv("1", "run-1")
    chal_mv = _mv("2", "run-2")
    champ_run = _run(champ_log_loss, champ_ece)
    chal_run = _run(chal_log_loss, chal_ece)

    def alias_side_effect(name, alias):
        return {"staging": chal_mv, "production": champ_mv}[alias]

    def run_side_effect(run_id):
        return chal_run if run_id == "run-2" else champ_run

    mock_mlflow.get_model_version_by_alias.side_effect = alias_side_effect
    mock_mlflow.get_run.side_effect = run_side_effect


def test_delta_below_threshold_blocks_promotion(promoter: AutoPromoter, mock_mlflow: MagicMock):
    # delta = 0.80 - 0.795 = 0.005, below DELTA_LOG_LOSS_THRESHOLD (0.01)
    _setup_champion_challenger(mock_mlflow, 0.80, 0.030, 0.795, 0.030)
    result = promoter.promote_if_better()
    assert result.promoted is False
    assert pytest.approx(result.delta_log_loss, rel=1e-4) == 0.005


def test_delta_exactly_at_threshold_promotes(promoter: AutoPromoter, mock_mlflow: MagicMock):
    # delta = 0.80 - 0.79 = 0.01 == DELTA_LOG_LOSS_THRESHOLD
    _setup_champion_challenger(mock_mlflow, 0.80, 0.030, 0.79, 0.030)
    result = promoter.promote_if_better()
    assert result.promoted is True


def test_delta_above_threshold_promotes(promoter: AutoPromoter, mock_mlflow: MagicMock):
    _setup_champion_challenger(mock_mlflow, 0.85, 0.030, 0.82, 0.028)
    result = promoter.promote_if_better()
    assert result.promoted is True


# ---------------------------------------------------------------------------
# Gate: ECE ceiling
# ---------------------------------------------------------------------------


def test_ece_above_ceiling_blocks_promotion(promoter: AutoPromoter, mock_mlflow: MagicMock):
    # delta OK (0.04) but challenger ECE 0.040 > ECE_CEILING (0.035)
    _setup_champion_challenger(mock_mlflow, 0.84, 0.030, 0.80, 0.040)
    result = promoter.promote_if_better()
    assert result.promoted is False
    assert "ECE" in result.reason or "ece" in result.reason.lower()


def test_ece_exactly_at_ceiling_promotes(promoter: AutoPromoter, mock_mlflow: MagicMock):
    _setup_champion_challenger(mock_mlflow, 0.84, 0.030, 0.82, ECE_CEILING)
    result = promoter.promote_if_better()
    assert result.promoted is True


# ---------------------------------------------------------------------------
# First deploy (no champion)
# ---------------------------------------------------------------------------


def test_first_deploy_no_champion_promotes_immediately(mock_mlflow: MagicMock):
    chal_mv = _mv("1", "run-1")
    chal_run = _run(log_loss=0.80, ece=0.028)

    def alias_side_effect(name, alias):
        if alias == "production":
            raise Exception("RESOURCE_DOES_NOT_EXIST: no production alias")
        return chal_mv

    mock_mlflow.get_model_version_by_alias.side_effect = alias_side_effect
    mock_mlflow.get_run.return_value = chal_run

    p = _make_promoter(mock_mlflow)
    result = p.promote_if_better()

    assert result.promoted is True
    assert result.champion_version is None
    assert result.delta_log_loss is None


# ---------------------------------------------------------------------------
# Delta value correctness
# ---------------------------------------------------------------------------


def test_delta_log_loss_is_champion_minus_challenger(promoter: AutoPromoter, mock_mlflow: MagicMock):
    _setup_champion_challenger(mock_mlflow, 0.90, 0.030, 0.86, 0.028)
    result = promoter.promote_if_better()
    assert pytest.approx(result.delta_log_loss, rel=1e-4) == 0.04


# ---------------------------------------------------------------------------
# MLflow alias mutations
# ---------------------------------------------------------------------------


def test_production_alias_set_to_challenger_version(promoter: AutoPromoter, mock_mlflow: MagicMock):
    _setup_champion_challenger(mock_mlflow, 0.85, 0.030, 0.83, 0.028)
    promoter.promote_if_better()
    mock_mlflow.set_registered_model_alias.assert_called_once_with(
        name=MODEL_NAME,
        alias="production",
        version="2",
    )


def test_old_champion_tagged_retired_after_promotion(promoter: AutoPromoter, mock_mlflow: MagicMock):
    _setup_champion_challenger(mock_mlflow, 0.85, 0.030, 0.83, 0.028)
    promoter.promote_if_better()
    mock_mlflow.set_model_version_tag.assert_called_once_with(
        name=MODEL_NAME,
        version="1",
        key="status",
        value="retired",
    )


def test_retire_tag_failure_is_non_fatal(promoter: AutoPromoter, mock_mlflow: MagicMock):
    """Failing to tag old champion as retired should log a warning but not raise."""
    _setup_champion_challenger(mock_mlflow, 0.85, 0.030, 0.83, 0.028)
    mock_mlflow.set_model_version_tag.side_effect = Exception("tag write failed")
    result = promoter.promote_if_better()
    assert result.promoted is True


def test_mlflow_alias_update_failure_raises_promotion_error(
    promoter: AutoPromoter, mock_mlflow: MagicMock
):
    _setup_champion_challenger(mock_mlflow, 0.85, 0.030, 0.83, 0.028)
    mock_mlflow.set_registered_model_alias.side_effect = Exception("MLflow connection refused")
    with pytest.raises(PromotionError, match="production alias"):
        promoter.promote_if_better()


# ---------------------------------------------------------------------------
# ECS deployment
# ---------------------------------------------------------------------------


def test_ecs_deploy_skipped_when_ecr_uri_empty(mock_mlflow: MagicMock):
    chal_mv = _mv("1", "run-1")

    def _alias_side_effect(name, alias):
        if alias == "production":
            raise Exception("RESOURCE_DOES_NOT_EXIST: no production alias")
        return chal_mv

    mock_mlflow.get_model_version_by_alias.side_effect = _alias_side_effect
    mock_mlflow.get_run.return_value = _run(0.80, 0.028)

    p = _make_promoter(mock_mlflow, ecr_image_uri="")
    result = p.promote_if_better()

    assert result.ecs_task_definition_arn is None


def test_ecs_deploy_called_when_ecr_uri_set(mock_mlflow: MagicMock):
    _setup_champion_challenger(mock_mlflow, 0.85, 0.030, 0.83, 0.028)

    mock_ecs = MagicMock()
    mock_ecs.register_task_definition.return_value = {
        "taskDefinition": {"taskDefinitionArn": TASK_DEF_ARN, "revision": 42}
    }

    with (
        patch("src.mlops.auto_promoter.MlflowClient", return_value=mock_mlflow),
        patch("src.mlops.auto_promoter.boto3") as mock_boto3,
    ):
        mock_boto3.client.return_value = mock_ecs
        p = AutoPromoter(_config(ecr_image_uri="123.dkr.ecr.us-east-1.amazonaws.com/mlb:v1"))
        result = p.promote_if_better()

    assert result.promoted is True
    assert result.ecs_task_definition_arn == TASK_DEF_ARN
    mock_ecs.update_service.assert_called_once()


def test_ecs_register_failure_raises_promotion_error(mock_mlflow: MagicMock):
    _setup_champion_challenger(mock_mlflow, 0.85, 0.030, 0.83, 0.028)

    mock_ecs = MagicMock()
    mock_ecs.register_task_definition.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
        "RegisterTaskDefinition",
    )

    with (
        patch("src.mlops.auto_promoter.MlflowClient", return_value=mock_mlflow),
        patch("src.mlops.auto_promoter.boto3") as mock_boto3,
    ):
        mock_boto3.client.return_value = mock_ecs
        p = AutoPromoter(_config(ecr_image_uri="123.dkr.ecr.us-east-1.amazonaws.com/mlb:v1"))

    with pytest.raises(PromotionError, match="task definition"):
        p.promote_if_better()


def test_ecs_update_service_failure_raises_promotion_error(mock_mlflow: MagicMock):
    """Task definition registered but update_service fails → PromotionError."""
    _setup_champion_challenger(mock_mlflow, 0.85, 0.030, 0.83, 0.028)

    mock_ecs = MagicMock()
    mock_ecs.register_task_definition.return_value = {
        "taskDefinition": {"taskDefinitionArn": TASK_DEF_ARN, "revision": 1}
    }
    mock_ecs.update_service.side_effect = ClientError(
        {"Error": {"Code": "ServiceNotFoundException", "Message": "Service not found"}},
        "UpdateService",
    )

    with (
        patch("src.mlops.auto_promoter.MlflowClient", return_value=mock_mlflow),
        patch("src.mlops.auto_promoter.boto3") as mock_boto3,
    ):
        mock_boto3.client.return_value = mock_ecs
        p = AutoPromoter(_config(ecr_image_uri="123.dkr.ecr.us-east-1.amazonaws.com/mlb:v1"))
        with pytest.raises(PromotionError, match="update_service"):
            p.promote_if_better()
