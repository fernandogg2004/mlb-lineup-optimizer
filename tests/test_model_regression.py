"""
Model Regression Tests — Transversal Roadmap Item

Guards against silent degradation of calibration metrics when code changes.
Run before every deployment: `python -m pytest tests/test_model_regression.py`

Criteria:
  - Log-Loss must stay below a known upper bound (derived from baseline)
  - Brier Score must stay below baseline
  - The model must NOT be systematically over- or under-confident
    (ECE < 0.10 is required; alert if > 0.05)

Design — sólo califica predicciones del modelo ACTUAL:
  El backtest agrega TODAS las predicciones guardadas en results/, incluidas
  las generadas por modelos anteriores. Calificar un modelo nuevo con métricas
  contaminadas por predicciones de un modelo retirado (o al revés) es ruido.
  Por eso estos tests filtran los juegos a `game_date >= fecha de despliegue`
  del modelo en producción y RECALCULAN las métricas sobre ese subconjunto.
  Si aún no hay suficientes juegos post-despliegue, se hace SKIP (no fail):
  el gate real y libre de leakage es el de prior-drift en train_v3.py, que
  corre a nivel PA en cada retrain.
"""

import datetime as _dt
import json
import math
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
BACKTEST_FILE = ROOT / "reports" / "backtest" / "backtest_results.json"
MODEL_FILE = ROOT / "models" / "at_bat_predictor.pkl"

# ── Thresholds (adjust after first validated backtest run) ───────────────────
# Upper bounds — if the model exceeds these, something degraded.
LOG_LOSS_MAX = 0.450          # Naive coin-flip = 0.693; Elo-style baseline ≈ 0.431
BRIER_SCORE_MAX = 0.250       # Random guess = 0.25; should beat this convincingly
ECE_ALERT_THRESHOLD = 0.050   # > 5% ECE → calibration drift alert
ECE_FAIL_THRESHOLD = 0.100    # > 10% ECE → hard fail

# Minimum sample size for a valid test
MIN_GAMES = 30

# Reusar las MISMAS métricas que produce el backtest, para que el gate y el
# reporte hablen exactamente el mismo idioma (idéntica convención de win_prob).
from backtest import (  # noqa: E402
    auc_roc as _auc_roc,
    brier_score as _brier_score,
    calibration_bins as _calibration_bins,
    ece as _ece,
    log_loss as _log_loss,
)


def _model_deploy_date() -> _dt.date | None:
    """Fecha a partir de la cual una predicción la generó el modelo actual.

    Fuente de verdad: campo ``trained_at`` del artefacto (lo escribe save()).
    Fallback: mtime del .pkl de producción. Leemos ``trained_at`` sin
    deserializar el modelo completo (evita arrastrar lightgbm/mlflow en el
    test) — un fallo en la lectura cae al mtime, que falla-cerrado a SKIP
    porque un checkout de git deja el mtime reciente.
    """
    if not MODEL_FILE.exists():
        return None
    try:
        import pickle

        class _StubUnpickler(pickle.Unpickler):
            # No reconstruimos las clases del modelo: cualquier objeto no-builtin
            # se vuelve un stub. Sólo nos interesa la string trained_at del dict.
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except Exception:
                    return type(name, (), {"__init__": lambda self, *a, **k: None})

        with open(MODEL_FILE, "rb") as f:
            payload = _StubUnpickler(f).load()
        ts = payload.get("trained_at") if isinstance(payload, dict) else None
        if ts:
            return _dt.datetime.fromisoformat(ts).date()
    except Exception:
        pass
    return _dt.date.fromtimestamp(os.path.getmtime(MODEL_FILE))


@pytest.fixture(scope="module")
def backtest_data():
    """Load backtest results. Skip tests if file not found (not a failure)."""
    if not BACKTEST_FILE.exists():
        pytest.skip(
            f"Backtest file not found at {BACKTEST_FILE}. "
            "Run 'python backtest.py' first to generate it."
        )
    with open(BACKTEST_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def current_model_games(backtest_data):
    """Juegos del backtest generados por el modelo en producción actual.

    Skip (no fail) si el modelo no existe o si hay menos de MIN_GAMES juegos
    posteriores a su despliegue — el gate de calidad no puede medirse aún.
    """
    deploy_date = _model_deploy_date()
    if deploy_date is None:
        pytest.skip(f"Modelo de producción no encontrado en {MODEL_FILE}.")

    games = [
        g for g in backtest_data.get("games", [])
        if g.get("win_probability") is not None
        and g.get("home_won") is not None
        and g.get("game_date", "") >= deploy_date.isoformat()
    ]
    if len(games) < MIN_GAMES:
        pytest.skip(
            f"Sólo {len(games)} juegos con game_date >= {deploy_date.isoformat()} "
            f"(despliegue del modelo); se necesitan {MIN_GAMES}. El gate de "
            "regresión a nivel juego se activará al acumular más predicciones "
            "del modelo actual. El gate de prior-drift de train_v3.py ya cubre "
            "el retrain a nivel PA."
        )
    return games


@pytest.fixture(scope="module")
def current_model_predictions(current_model_games):
    """(win_probability, home_won) sólo de juegos del modelo actual."""
    return [
        (float(g["win_probability"]), int(g["home_won"]))
        for g in current_model_games
    ]


def test_backtest_file_has_minimum_games(backtest_data):
    """Ensure the backtest covers enough games to be statistically meaningful."""
    n_games = backtest_data.get("n_games", 0)
    assert n_games >= MIN_GAMES, (
        f"Backtest has only {n_games} games — need at least {MIN_GAMES} for reliable metrics."
    )


def test_log_loss_below_upper_bound(current_model_predictions):
    """Log-Loss del modelo actual no debe exceder la cota superior."""
    log_loss = _log_loss(current_model_predictions)
    assert log_loss < LOG_LOSS_MAX, (
        f"Log-Loss regression: {log_loss:.4f} >= threshold {LOG_LOSS_MAX}. "
        "Model may have degraded. Check recent code changes."
    )


def test_brier_score_beats_random(current_model_predictions):
    """Brier Score del modelo actual debe ganarle al azar (0.25)."""
    brier = _brier_score(current_model_predictions)
    assert brier < BRIER_SCORE_MAX, (
        f"Brier Score regression: {brier:.4f} >= threshold {BRIER_SCORE_MAX}. "
        "Model may have degraded."
    )


def test_ece_calibration_within_alert_threshold(current_model_predictions):
    """ECE del modelo actual debe estar bajo el umbral de fallo (10%)."""
    ece_val = _ece(_calibration_bins(current_model_predictions))
    assert ece_val < ECE_FAIL_THRESHOLD, (
        f"ECE HARD FAIL: {ece_val:.4f} >= {ECE_FAIL_THRESHOLD}. Calibration severely degraded."
    )
    if ece_val > ECE_ALERT_THRESHOLD:
        import warnings
        warnings.warn(
            f"ECE drift warning: {ece_val:.4f} > {ECE_ALERT_THRESHOLD}. "
            "Monitor calibration — may need recalibration.",
            UserWarning,
        )


def test_auc_beats_coin_flip(current_model_predictions):
    """AUC ROC del modelo actual debe superar 0.5 (mejor que el azar)."""
    auc = _auc_roc(current_model_predictions)
    assert auc > 0.50, f"AUC ROC {auc:.4f} <= 0.50 — model is no better than random."
    assert auc > 0.52, (
        f"AUC ROC {auc:.4f} is barely above random. Minimal predictive power."
    )


def test_predictions_are_not_degenerate(current_model_games):
    """
    Verify predictions don't cluster near 0.5 (which would be safe but useless)
    or near 0.0/1.0 (overconfident).

    A useful model has variance in its predictions.
    """
    probs = [g["win_probability"] for g in current_model_games]
    mean_prob = sum(probs) / len(probs)
    variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
    std_dev = math.sqrt(variance)

    # If std_dev < 0.02, the model is basically always predicting ~50% (useless)
    assert std_dev > 0.02, (
        f"Prediction std_dev {std_dev:.4f} is too low — model is degenerate (outputs near 0.5 always)."
    )

    # If std_dev > 0.45, the model is overconfident (many near 0 or 1)
    assert std_dev < 0.45, (
        f"Prediction std_dev {std_dev:.4f} is too high — model may be overconfident."
    )


def test_no_predictions_outside_unit_interval(backtest_data):
    """All win probabilities must be in (0, 1)."""
    games = backtest_data.get("games", [])
    bad = [
        g["game_pk"]
        for g in games
        if g.get("win_probability") is not None
        and not (0.0 < g["win_probability"] < 1.0)
    ]
    assert not bad, (
        f"{len(bad)} predictions outside (0, 1): {bad[:5]}{'...' if len(bad) > 5 else ''}"
    )


# ── Unit tests for economic layer ─────────────────────────────────────────────

def test_ev_positive_when_model_has_edge():
    """EV must be positive when model win prob > book implied prob."""
    from api.economic_layer import compute_ev
    result = compute_ev(model_win_prob=0.60, odds_american=-110)
    assert result["has_value"] is True
    assert result["ev"] > 0
    assert result["edge_pct"] > 0


def test_ev_negative_when_model_has_no_edge():
    """EV must be negative when model prob equals book implied."""
    from api.economic_layer import compute_ev
    # American -110 implies ~52.4%; we say 50% → no edge
    result = compute_ev(model_win_prob=0.50, odds_american=-110)
    assert result["has_value"] is False
    assert result["ev"] < 0


def test_clv_positive_when_market_moves_our_way():
    """CLV must be positive when the line moves in our direction."""
    from api.economic_layer import compute_clv
    # Bet at -110 open; line moves to -130 (market agrees more)
    result = compute_clv(open_odds_american=-110, closing_odds_american=-130)
    assert result["clv_pct"] > 0
    assert result["verdict"] == "value"


def test_american_to_decimal_conversion():
    """Verify odds conversion accuracy."""
    from api.economic_layer import american_to_decimal
    assert abs(american_to_decimal(-110) - 1.9091) < 0.001
    assert abs(american_to_decimal(+150) - 2.5) < 0.001
    assert abs(american_to_decimal(-200) - 1.5) < 0.001


# ── Unit tests for fatigue model ──────────────────────────────────────────────

def test_fatigue_fresh_player_returns_1():
    """A well-rested player with typical workload should return factor ~1.0."""
    from api.fatigue_model import FatigueSignals, compute_fatigue_factor
    signals = FatigueSignals(
        consecutive_games_played=2,
        rest_days_last_7=3,
        pa_last_7_days=25.0,
        timezone_changes_last_3_days=0,
    )
    factor = compute_fatigue_factor(signals)
    assert factor == 1.0


def test_fatigue_heavy_load_returns_lower_factor():
    """A heavily loaded player should return factor < 1.0."""
    from api.fatigue_model import FatigueSignals, compute_fatigue_factor
    signals = FatigueSignals(
        consecutive_games_played=9,
        rest_days_last_7=0,
        pa_last_7_days=45.0,
        timezone_changes_last_3_days=2,
    )
    factor = compute_fatigue_factor(signals)
    assert factor < 0.90


def test_fatigue_factor_within_bounds():
    """Factor must always be in [FACTOR_MIN, FACTOR_MAX]."""
    from api.fatigue_model import FatigueSignals, compute_fatigue_factor
    extreme = FatigueSignals(
        consecutive_games_played=20,
        rest_days_last_7=0,
        pa_last_7_days=70.0,
        timezone_changes_last_3_days=5,
    )
    factor = compute_fatigue_factor(extreme)
    assert 0.82 <= factor <= 1.00
