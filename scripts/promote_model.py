"""
scripts/promote_model.py
========================
Promueve un modelo challenger a producción (champion) de forma controlada.

Flujo de promoción:
  1. Carga champion (producción actual) y challenger (nuevo modelo)
  2. Evalúa ambos en el holdout set más reciente (temporada de validación)
  3. Compara ECE (Expected Calibration Error) y log-loss
  4. Si challenger ≥ champion en ambas métricas → promueve en MLflow
  5. Actualiza variable de entorno en ECS Task Definition
  6. Notifica al equipo vía Slack

Uso:
  python scripts/promote_model.py \\
    --challenger-version 5 \\
    --holdout-season 2024 \\
    --features-path data/gold/features_train_v2.parquet

  # Dry-run (no modifica nada, solo evalúa):
  python scripts/promote_model.py --challenger-version 5 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("model_promoter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME    = os.getenv("MLB_MODEL_NAME", "mlb-at-bat-predictor")
AWS_REGION    = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ECS_CLUSTER   = os.getenv("ECS_CLUSTER_NAME", "mlb-ai-cluster")
ECS_SERVICE   = os.getenv("ECS_SERVICE_NAME", "mlb-api-service")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")

_LEAKING = {"xwoba", "launch_speed", "launch_angle"}


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def _load_predictor_from_mlflow(client: MlflowClient, version: int) -> Any:
    model_uri = f"models:/{MODEL_NAME}/{version}"
    log.info("Cargando modelo v%d desde MLflow...", version)

    import src.models.model_at_bat as _mat
    for _n in dir(_mat):
        setattr(sys.modules["__main__"], _n, getattr(_mat, _n))

    return mlflow.sklearn.load_model(model_uri)


def _load_predictor_from_pkl(path: str) -> Any:
    import src.models.model_at_bat as _mat
    for _n in dir(_mat):
        setattr(sys.modules["__main__"], _n, getattr(_mat, _n))
    with open(path, "rb") as f:
        payload = pickle.load(f)
    predictor = _mat.AtBatPredictor(config=payload["config"])
    predictor._calibrated_model = payload["calibrated_model"]
    predictor._base_model       = payload["base_model"]
    predictor._feature_names    = payload["feature_names"]
    predictor._is_fitted        = True
    return predictor


def evaluate_predictor(predictor: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import log_loss
    from src.models.model_at_bat import ECEComputer

    probs  = predictor.predict_proba(X)
    ll     = log_loss(y, probs)
    ece_computer = ECEComputer(n_bins=10)
    ece    = ece_computer.compute(probs, y)["ece"]
    acc    = float((probs.argmax(axis=1) == y).mean())

    return {"log_loss": round(ll, 4), "ece": round(ece, 4), "accuracy": round(acc, 4)}


def load_holdout(features_path: str, holdout_season: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pl.read_parquet(features_path)
    holdout = df.filter(pl.col("season") == holdout_season)

    feature_cols = [
        c for c in df.columns
        if c not in {"pa_outcome_idx", "season", "game_date"} | _LEAKING
    ]
    X = holdout.select(feature_cols).to_numpy().astype(np.float32)
    y = holdout["pa_outcome_idx"].to_numpy().astype(np.int32)
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Promoción en MLflow
# ---------------------------------------------------------------------------
def promote_in_mlflow(client: MlflowClient, challenger_version: int) -> None:
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(challenger_version),
        stage="Production",
        archive_existing_versions=True,  # archiva el champion anterior
    )
    log.info("Modelo v%d → Production en MLflow.", challenger_version)


# ---------------------------------------------------------------------------
# Actualización ECS Task Definition
# ---------------------------------------------------------------------------
def update_ecs_model_version(challenger_version: int, dry_run: bool = False) -> None:
    if dry_run:
        log.info("[DRY-RUN] Actualizaría ECS service con model_version=%d.", challenger_version)
        return
    try:
        import boto3
        ecs = boto3.client("ecs", region_name=AWS_REGION)

        # Obtiene task definition actual del servicio
        service = ecs.describe_services(cluster=ECS_CLUSTER, services=[ECS_SERVICE])
        task_def_arn = service["services"][0]["taskDefinition"]
        task_def = ecs.describe_task_definition(taskDefinition=task_def_arn)["taskDefinition"]

        # Actualiza la variable MODEL_VERSION en el container environment
        containers = task_def["containerDefinitions"]
        for c in containers:
            env = c.get("environment", [])
            env = [e for e in env if e["name"] != "MODEL_VERSION"]
            env.append({"name": "MODEL_VERSION", "value": str(challenger_version)})
            c["environment"] = env

        # Registra nueva task definition
        new_td = ecs.register_task_definition(
            family=task_def["family"],
            containerDefinitions=containers,
            taskRoleArn=task_def.get("taskRoleArn", ""),
            executionRoleArn=task_def.get("executionRoleArn", ""),
            networkMode=task_def.get("networkMode", "awsvpc"),
            cpu=task_def.get("cpu", "1024"),
            memory=task_def.get("memory", "2048"),
        )
        new_arn = new_td["taskDefinition"]["taskDefinitionArn"]

        # Despliega en el servicio (rolling update)
        ecs.update_service(
            cluster=ECS_CLUSTER,
            service=ECS_SERVICE,
            taskDefinition=new_arn,
            forceNewDeployment=True,
        )
        log.info("ECS service actualizado con nueva task definition: %s", new_arn)
    except Exception as exc:
        log.error("Error actualizando ECS: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Notificación Slack
# ---------------------------------------------------------------------------
def _notify_slack(result: dict, dry_run: bool) -> None:
    if not SLACK_WEBHOOK:
        return
    import requests
    prefix = "[DRY-RUN] " if dry_run else ""
    icon   = ":white_check_mark:" if result["promoted"] else ":x:"
    msg = {
        "text": (
            f"{icon} {prefix}*MLB AI Model Promotion* — v{result['challenger_version']}\n"
            f">Status: *{'PROMOTED' if result['promoted'] else 'REJECTED'}*\n"
            f">Champion: log_loss={result['champion_metrics']['log_loss']}  "
            f"ECE={result['champion_metrics']['ece']}\n"
            f">Challenger: log_loss={result['challenger_metrics']['log_loss']}  "
            f"ECE={result['challenger_metrics']['ece']}\n"
            f">Reason: {result['reason']}"
        )
    }
    try:
        requests.post(SLACK_WEBHOOK, json=msg, timeout=10)
    except Exception as exc:
        log.warning("Slack notify falló: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MLB AI model promotion")
    parser.add_argument("--challenger-version", type=int, required=True)
    parser.add_argument("--holdout-season",     type=int, default=2024)
    parser.add_argument("--features-path",
                        default="data/gold/features_train_v2.parquet")
    parser.add_argument("--champion-pkl",
                        default="models/at_bat_predictor.pkl",
                        help="Path al pkl del champion actual")
    parser.add_argument("--challenger-pkl",
                        default="models/pa_predictor_v1.pkl",
                        help="Path al pkl del challenger")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evalúa pero no modifica nada")
    args = parser.parse_args()

    log.info("=== MLB AI Model Promotion ===")
    log.info("Champion:   %s", args.champion_pkl)
    log.info("Challenger: %s (v%d)", args.challenger_pkl, args.challenger_version)
    log.info("Holdout:    season=%d", args.holdout_season)

    X, y, _ = load_holdout(args.features_path, args.holdout_season)
    log.info("Holdout set: %d samples", len(X))

    champion   = _load_predictor_from_pkl(args.champion_pkl)
    challenger = _load_predictor_from_pkl(args.challenger_pkl)

    champ_metrics = evaluate_predictor(champion,   X, y)
    chall_metrics = evaluate_predictor(challenger, X, y)

    log.info("Champion   metrics: %s", champ_metrics)
    log.info("Challenger metrics: %s", chall_metrics)

    # Criterio: challenger debe mejorar log_loss Y ECE
    ll_better  = chall_metrics["log_loss"] < champ_metrics["log_loss"]
    ece_better = chall_metrics["ece"]      < champ_metrics["ece"]
    promote    = ll_better and ece_better

    if not promote:
        reason = (
            f"Challenger no supera al champion. "
            f"ll_better={ll_better}, ece_better={ece_better}"
        )
        log.warning("RECHAZADO: %s", reason)
    else:
        reason = (
            f"Challenger supera al champion en log_loss y ECE. "
            f"Delta_ll={champ_metrics['log_loss']-chall_metrics['log_loss']:.4f}, "
            f"Delta_ece={champ_metrics['ece']-chall_metrics['ece']:.4f}"
        )
        log.info("PROMOVIDO: %s", reason)

        if not args.dry_run:
            update_ecs_model_version(args.challenger_version, dry_run=False)
            log.info("ECS actualizado con modelo v%d.", args.challenger_version)

    result = {
        "promoted":           promote,
        "challenger_version": args.challenger_version,
        "champion_metrics":   champ_metrics,
        "challenger_metrics": chall_metrics,
        "reason":             reason,
        "dry_run":            args.dry_run,
    }
    print(json.dumps(result, indent=2))
    _notify_slack(result, dry_run=args.dry_run)

    sys.exit(0 if promote else 2)  # exit 2 = challenger no supera champion


if __name__ == "__main__":
    main()
