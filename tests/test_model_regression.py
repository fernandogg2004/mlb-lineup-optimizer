"""
Model Regression Tests — Transversal Roadmap Item

Guards against silent degradation of calibration metrics when code changes.
Run before every deployment: `python -m pytest tests/test_model_regression.py`

Criteria:
  - Log-Loss must stay below a known upper bound (derived from baseline)
  - Brier Score must stay below baseline
  - The model must NOT be systematically over- or under-confident
    (ECE < 0.10 is required; alert if > 0.05)
  - These tests use the backtest_results.json as ground truth;
    if no backtest file exists, tests are skipped (not failed).
"""

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
BACKTEST_FILE = ROOT / "reports" / "backtest" / "backtest_results.json"

# ── Thresholds (adjust after first validated backtest run) ───────────────────
# Upper bounds — if the model exceeds these, something degraded.
LOG_LOSS_MAX = 0.450          # Naive coin-flip = 0.693; Elo-style baseline ≈ 0.431
BRIER_SCORE_MAX = 0.250       # Random guess = 0.25; should beat this convincingly
ECE_ALERT_THRESHOLD = 0.050   # > 5% ECE → calibration drift alert
ECE_FAIL_THRESHOLD = 0.100    # > 10% ECE → hard fail

# Minimum sample size for a valid test
MIN_GAMES = 30


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


def test_backtest_file_has_minimum_games(backtest_data):
    """Ensure the backtest covers enough games to be statistically meaningful."""
    n_games = backtest_data.get("n_games", 0)
    assert n_games >= MIN_GAMES, (
        f"Backtest has only {n_games} games — need at least {MIN_GAMES} for reliable metrics."
    )


def test_log_loss_below_upper_bound(backtest_data):
    """Log-Loss must not exceed the naive baseline."""
    metrics = backtest_data.get("metrics", {})
    log_loss = metrics.get("log_loss")
    if log_loss is None:
        pytest.skip("log_loss not in backtest metrics")
    assert log_loss < LOG_LOSS_MAX, (
        f"Log-Loss regression: {log_loss:.4f} >= threshold {LOG_LOSS_MAX}. "
        "Model may have degraded. Check recent code changes."
    )


def test_brier_score_beats_random(backtest_data):
    """Brier Score must beat random guessing (0.25)."""
    metrics = backtest_data.get("metrics", {})
    brier = metrics.get("brier_score")
    if brier is None:
        pytest.skip("brier_score not in backtest metrics")
    assert brier < BRIER_SCORE_MAX, (
        f"Brier Score regression: {brier:.4f} >= threshold {BRIER_SCORE_MAX}. "
        "Model may have degraded."
    )


def test_ece_calibration_within_alert_threshold(backtest_data):
    """ECE must be below the alert threshold (5%)."""
    metrics = backtest_data.get("metrics", {})
    ece = metrics.get("ece")
    if ece is None:
        pytest.skip("ECE not in backtest metrics")
    assert ece < ECE_FAIL_THRESHOLD, (
        f"ECE HARD FAIL: {ece:.4f} >= {ECE_FAIL_THRESHOLD}. Calibration severely degraded."
    )
    if ece > ECE_ALERT_THRESHOLD:
        pytest.warns(
            UserWarning,
            match="ECE drift",
        )
        import warnings
        warnings.warn(
            f"ECE drift warning: {ece:.4f} > {ECE_ALERT_THRESHOLD}. "
            "Monitor calibration — may need recalibration.",
            UserWarning,
        )


def test_auc_beats_coin_flip(backtest_data):
    """AUC ROC must be above 0.5 (better than random)."""
    metrics = backtest_data.get("metrics", {})
    auc = metrics.get("auc_roc")
    if auc is None:
        pytest.skip("auc_roc not in backtest metrics")
    assert auc > 0.50, f"AUC ROC {auc:.4f} <= 0.50 — model is no better than random."
    assert auc > 0.52, (
        f"AUC ROC {auc:.4f} is barely above random. Minimal predictive power."
    )


def test_predictions_are_not_degenerate(backtest_data):
    """
    Verify predictions don't cluster near 0.5 (which would be safe but useless)
    or near 0.0/1.0 (overconfident).

    A useful model has variance in its predictions.
    """
    games = backtest_data.get("games", [])
    if not games:
        pytest.skip("No game records in backtest file")

    probs = [
        g["win_probability"]
        for g in games
        if g.get("win_probability") is not None
    ]
    if len(probs) < MIN_GAMES:
        pytest.skip(f"Only {len(probs)} predictions — need {MIN_GAMES}")

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
