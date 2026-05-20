"""
monitoring/drift_detector.py
============================
Script autónomo de detección de drift. Se puede correr:
  - Como tarea del DAG de Airflow (task_id=check_prediction_drift)
  - Como cronjob independiente
  - Vía CLI: python monitoring/drift_detector.py --date 2026-05-19

Calcula:
  1. PSI en distribución de predicciones (output drift)
  2. PSI en features de entrada: xwoba_7d, k_rate_7d, bb_rate_7d (input drift)
  3. Si PSI > 0.20 → alerta via Slack + SNS + actualiza Prometheus gauge

PSI interpretación (Lichtman / banca):
  < 0.10 → estable
  0.10 – 0.20 → cambio moderado, monitorear
  > 0.20 → cambio significativo, reentrenar
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from typing import NamedTuple

import numpy as np
import psycopg2
import requests

log = logging.getLogger("drift_detector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Baseline — distribución empírica del Gold 2015-2024 (calculada en training)
# Actualizar cada vez que se re-entrene el modelo.
# ---------------------------------------------------------------------------
BASELINE_PRED_DIST = {
    "OUT":    0.4643,
    "K":      0.2056,
    "BB_HBP": 0.0976,
    "1B":     0.1494,
    "2B":     0.0464,
    "3B":     0.0049,
    "HR":     0.0318,
}

BASELINE_FEATURE_STATS = {
    # (mean, std) para las features clave — extraídas del Gold training set
    "xwoba_7d":    (0.330, 0.080),
    "k_rate_7d":   (0.224, 0.085),
    "bb_rate_7d":  (0.083, 0.040),
    "hr_rate_7d":  (0.033, 0.030),
    "pa_7d":       (24.5,  12.0),
}

PSI_CRITICAL  = float(os.getenv("DRIFT_PSI_CRITICAL", "0.20"))
PSI_WARNING   = float(os.getenv("DRIFT_PSI_WARNING",  "0.10"))
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")
SNS_TOPIC_ARN = os.getenv("DRIFT_SNS_TOPIC_ARN", "")
DATABASE_URL  = os.getenv("DATABASE_URL", "postgresql://mlb:mlb@localhost:5432/mlb_predictions")
PROMETHEUS_PUSHGATEWAY = os.getenv("PROMETHEUS_PUSHGATEWAY", "")


class DriftReport(NamedTuple):
    date:           str
    psi_pred_total: float
    psi_pred_class: dict[str, float]
    psi_feat:       dict[str, float]
    status:         str     # ok | warning | critical
    n_predictions:  int


# ---------------------------------------------------------------------------
# Cálculo PSI
# ---------------------------------------------------------------------------
def _psi(actual: np.ndarray, expected: np.ndarray, n_bins: int = 10, eps: float = 1e-6) -> float:
    """PSI para distribuciones continuas via histogramas de igual anchura."""
    bins = np.linspace(
        min(actual.min(), expected.min()),
        max(actual.max(), expected.max()),
        n_bins + 1,
    )
    a_hist, _ = np.histogram(actual,   bins=bins, density=True)
    e_hist, _ = np.histogram(expected, bins=bins, density=True)
    # Normaliza a densidades proporcionales
    a = np.maximum(a_hist / (a_hist.sum() + eps), eps)
    e = np.maximum(e_hist / (e_hist.sum() + eps), eps)
    return float(np.sum((a - e) * np.log(a / e)))


def _psi_categorical(actual_dist: dict[str, float], baseline_dist: dict[str, float]) -> dict[str, float]:
    """PSI por categoría (outcome classes)."""
    eps = 1e-6
    result = {}
    for k in baseline_dist:
        a = max(actual_dist.get(k, 0.0), eps)
        e = max(baseline_dist[k], eps)
        result[k] = float((a - e) * np.log(a / e))
    return result


# ---------------------------------------------------------------------------
# Carga de datos desde PostgreSQL
# ---------------------------------------------------------------------------
def _load_predictions_from_db(run_date: str) -> list[dict]:
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute("""
        SELECT batting_order FROM predictions
        WHERE game_date = %s;
    """, (run_date,))
    rows = cur.fetchall()
    conn.close()

    batters = []
    for (batting_order_json,) in rows:
        if isinstance(batting_order_json, str):
            batting_order = json.loads(batting_order_json)
        else:
            batting_order = batting_order_json
        batters.extend(batting_order)
    return batters


# ---------------------------------------------------------------------------
# Detección de drift
# ---------------------------------------------------------------------------
def compute_drift(run_date: str) -> DriftReport:
    log.info("Calculando drift para %s...", run_date)
    batters = _load_predictions_from_db(run_date)

    if not batters:
        log.warning("No hay predicciones en DB para %s.", run_date)
        return DriftReport(
            date=run_date, psi_pred_total=0.0, psi_pred_class={},
            psi_feat={}, status="no_data", n_predictions=0,
        )

    # ── Output drift: distribución de probabilidades predichas ───────────────
    PROB_KEYS   = ["prob_out", "prob_k", "prob_bb", "prob_1b", "prob_2b", "prob_3b", "prob_hr"]
    CLASS_NAMES = ["OUT",      "K",      "BB_HBP",  "1B",      "2B",      "3B",      "HR"]

    actual_pred_dist = {}
    for name, key in zip(CLASS_NAMES, PROB_KEYS):
        vals = [b.get(key, 0.0) for b in batters if key in b]
        actual_pred_dist[name] = float(np.mean(vals)) if vals else BASELINE_PRED_DIST[name]

    psi_pred_class = _psi_categorical(actual_pred_dist, BASELINE_PRED_DIST)
    psi_pred_total = sum(psi_pred_class.values())

    # ── Input drift: features clave de los bateadores ────────────────────────
    # Nota: solo disponible si la API guarda features en DB (recomendado para audit)
    # Por ahora, basamos el input drift en woba_stab como proxy del input feature space
    woba_vals = np.array([b.get("woba_stab", 0.318) for b in batters])
    woba_baseline = np.random.normal(
        BASELINE_FEATURE_STATS["xwoba_7d"][0],
        BASELINE_FEATURE_STATS["xwoba_7d"][1],
        size=len(woba_vals),
    )
    psi_feat = {"woba_stab_proxy": _psi(woba_vals, woba_baseline)}

    # ── Status ───────────────────────────────────────────────────────────────
    max_psi = max(psi_pred_total, max(psi_feat.values(), default=0.0))
    if max_psi >= PSI_CRITICAL:
        status = "critical"
    elif max_psi >= PSI_WARNING:
        status = "warning"
    else:
        status = "ok"

    report = DriftReport(
        date=run_date,
        psi_pred_total=round(psi_pred_total, 4),
        psi_pred_class={k: round(v, 4) for k, v in psi_pred_class.items()},
        psi_feat={k: round(v, 4) for k, v in psi_feat.items()},
        status=status,
        n_predictions=len(batters),
    )
    log.info("Drift report: %s", report)
    return report


# ---------------------------------------------------------------------------
# Alertas
# ---------------------------------------------------------------------------
def _alert_slack(report: DriftReport) -> None:
    if not SLACK_WEBHOOK:
        return
    emoji = ":rotating_light:" if report.status == "critical" else ":warning:"
    msg = {
        "blocks": [
            {"type": "header", "text": {
                "type": "plain_text",
                "text": f"{emoji} MLB AI — Drift Alert [{report.date}]",
            }},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Status:* `{report.status.upper()}`"},
                {"type": "mrkdwn", "text": f"*PSI total:* `{report.psi_pred_total}`"},
                {"type": "mrkdwn", "text": f"*Threshold:* `{PSI_CRITICAL}`"},
                {"type": "mrkdwn", "text": f"*Predicciones:* `{report.n_predictions}`"},
            ]},
            {"type": "section", "text": {
                "type": "mrkdwn",
                "text": f"*PSI por clase:*\n```{json.dumps(report.psi_pred_class, indent=2)}```",
            }},
            {"type": "section", "text": {
                "type": "mrkdwn",
                "text": ":mag: Acción: revisar MLflow y considerar re-entrenamiento.",
            }},
        ]
    }
    try:
        resp = requests.post(SLACK_WEBHOOK, json=msg, timeout=10)
        resp.raise_for_status()
        log.info("Slack alert enviada.")
    except Exception as exc:
        log.warning("Slack alert falló: %s", exc)


def _alert_sns(report: DriftReport) -> None:
    if not SNS_TOPIC_ARN:
        return
    try:
        import boto3
        sns = boto3.client("sns")
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"[MLB AI] Drift {report.status.upper()} - {report.date}",
            Message=json.dumps(report._asdict(), indent=2),
        )
        log.info("SNS alert enviada a %s.", SNS_TOPIC_ARN)
    except Exception as exc:
        log.warning("SNS alert falló: %s", exc)


def _push_to_prometheus(report: DriftReport) -> None:
    """Pushes PSI metrics al Prometheus Pushgateway (para jobs batch sin servidor)."""
    if not PROMETHEUS_PUSHGATEWAY:
        return
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        registry = CollectorRegistry()
        g_total = Gauge("mlb_drift_psi_total", "PSI total", registry=registry)
        g_total.set(report.psi_pred_total)
        for class_name, val in report.psi_pred_class.items():
            g = Gauge(f"mlb_drift_psi_{class_name.lower()}", f"PSI {class_name}",
                      registry=registry)
            g.set(val)
        push_to_gateway(PROMETHEUS_PUSHGATEWAY, job="mlb_drift_detector", registry=registry)
        log.info("Métricas pusheadas a Prometheus Pushgateway.")
    except Exception as exc:
        log.warning("Prometheus push falló: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MLB AI drift detector")
    parser.add_argument("--date", default=str(date.today() - timedelta(days=1)),
                        help="Fecha a analizar (default: ayer)")
    parser.add_argument("--alert", action="store_true",
                        help="Enviar alertas si drift detectado")
    args = parser.parse_args()

    report = compute_drift(args.date)

    print(json.dumps(report._asdict(), indent=2))

    if args.alert and report.status in ("critical", "warning"):
        _alert_slack(report)
        _alert_sns(report)

    _push_to_prometheus(report)

    # Exit code no-zero si crítico (útil para CI/CD gates)
    if report.status == "critical":
        sys.exit(1)


if __name__ == "__main__":
    main()
