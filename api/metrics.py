"""
api/metrics.py
==============
Registro centralizado de métricas Prometheus para la API de inferencia.

Métricas expuestas:
  - mlb_inference_latency_seconds   → histograma de latencia por endpoint
  - mlb_requests_total              → contador de requests por endpoint + status
  - mlb_inference_errors_total      → contador de errores por tipo
  - mlb_prediction_prob_*           → histograma de distribución de probabilidades
  - mlb_model_info                  → gauge con metadata del modelo activo
  - mlb_drift_psi                   → gauge con el PSI de drift calculado por el DAG
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

# ---------------------------------------------------------------------------
# Latencia de inferencia
# ---------------------------------------------------------------------------
INFERENCE_LATENCY = Histogram(
    "mlb_inference_latency_seconds",
    "Tiempo de respuesta de la API por endpoint",
    labelnames=["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# ---------------------------------------------------------------------------
# Contadores de requests y errores
# ---------------------------------------------------------------------------
REQUEST_COUNTER = Counter(
    "mlb_requests_total",
    "Total de requests HTTP procesados",
    labelnames=["endpoint", "status"],
)

INFERENCE_ERRORS = Counter(
    "mlb_inference_errors_total",
    "Total de errores durante inferencia, clasificados por tipo",
    labelnames=["error_type"],
    # error_type: auth_failure | mlb_api_error | inference_error | model_load_error
)

# ---------------------------------------------------------------------------
# Distribución de predicciones por clase (detección de drift en tiempo real)
# ---------------------------------------------------------------------------
_OUTCOME_LABELS = ["OUT", "K", "BB_HBP", "1B", "2B", "3B", "HR"]
_PROB_FIELDS    = ["prob_out", "prob_k", "prob_bb", "prob_1b", "prob_2b", "prob_3b", "prob_hr"]

PREDICTIONS_DIST = Histogram(
    "mlb_prediction_probability",
    "Distribución de probabilidades predichas por clase de outcome",
    labelnames=["outcome_class"],
    buckets=(0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0),
)

# ---------------------------------------------------------------------------
# Métricas de modelo activo
# ---------------------------------------------------------------------------
MODEL_INFO = Info(
    "mlb_model",
    "Metadatos del modelo de inferencia activo",
)

DRIFT_PSI = Gauge(
    "mlb_drift_psi_total",
    "Population Stability Index calculado en el último ciclo diario",
)

DRIFT_PSI_BY_CLASS = Gauge(
    "mlb_drift_psi_by_class",
    "PSI por clase de outcome",
    labelnames=["outcome_class"],
)

ACTIVE_BATTERS_GAUGE = Gauge(
    "mlb_active_batters_today",
    "Número de bateadores procesados en la última llamada /predict/all",
)

SILVER_ROWS_GAUGE = Gauge(
    "mlb_silver_rows_loaded",
    "Filas de Silver cargadas en memoria al iniciar la API",
)


# ---------------------------------------------------------------------------
# Helper: registra la distribución de una respuesta completa
# ---------------------------------------------------------------------------
def record_prediction_distribution(batting_order: list[dict]) -> None:
    """Alimenta los histogramas de distribución con las predicciones de un equipo."""
    for batter in batting_order:
        for label, field in zip(_OUTCOME_LABELS, _PROB_FIELDS):
            prob = batter.get(field)
            if prob is not None:
                PREDICTIONS_DIST.labels(outcome_class=label).observe(float(prob))
    ACTIVE_BATTERS_GAUGE.inc(len(batting_order))


def update_drift_metrics(psi_total: float, psi_per_class: dict[str, float]) -> None:
    """Actualiza los gauges de drift (llamado desde el DAG o el drift_detector)."""
    DRIFT_PSI.set(psi_total)
    for class_name, psi_val in psi_per_class.items():
        safe_name = class_name.replace("/", "_")
        DRIFT_PSI_BY_CLASS.labels(outcome_class=safe_name).set(psi_val)
