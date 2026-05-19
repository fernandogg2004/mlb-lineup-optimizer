"""
Unit tests for _build_interpretation_es().

Pure function — parametrized to cover all code paths:
  1. sample_too_small=True  → "Muestra insuficiente" + correct needed count
  2. None values            → "Datos incompletos"
  3. manager > 0.05         → positive advantage phrasing
  4. manager < -0.05        → negative advantage phrasing
  5. |manager| < 0.05       → neutral / "equivalentes"
  6. large positive          → 162-game season projection included
"""
import pytest

from src.experiment.experiment_store import _MIN_SAMPLE_SIGNIFICANT, _build_interpretation_es


@pytest.mark.parametrize(
    "total, avg_ai, avg_actual, advantage, sample_too_small, lookback, expected_fragment",
    [
        # sample_too_small — shows how many games are missing
        (14, 4.5, 5.0, 0.5, True, 30, "Muestra insuficiente"),
        (14, 4.5, 5.0, 0.5, True, 30, "Se necesitan 1 más"),

        # None values — incomplete data message
        (20, 4.5, None, None, False, 30, "Datos incompletos"),

        # Positive advantage — manager beats AI expectation
        (20, 4.5, 5.0, 0.5, False, 30, "supera la expectativa"),

        # Negative advantage — AI expectation higher than what manager achieved
        (20, 5.0, 4.0, -1.0, False, 30, "habría producido"),

        # Near-zero advantage — statistically equivalent
        (20, 4.5, 4.52, 0.02, False, 30, "equivalentes"),

        # Large advantage triggers 162-game season projection
        (20, 3.5, 5.5, 2.0, False, 30, "162 partidos"),

        # Lookback window mentioned in interpretation text
        (20, 4.5, 5.0, 0.5, False, 90, "90 días"),
    ],
)
def test_interpretation_fragment(
    total, avg_ai, avg_actual, advantage, sample_too_small, lookback, expected_fragment
):
    result = _build_interpretation_es(
        total=total,
        avg_ai=avg_ai,
        avg_actual=avg_actual,
        manager_advantage=advantage,
        sample_too_small=sample_too_small,
        lookback_days=lookback,
    )
    assert expected_fragment in result, f"Expected '{expected_fragment}' in:\n{result!r}"


def test_small_sample_shows_exact_needed_count():
    """needed = max(0, 15 - total) → for total=12 should say 'Se necesitan 3 más'."""
    result = _build_interpretation_es(
        total=12,
        avg_ai=4.5,
        avg_actual=5.0,
        manager_advantage=0.5,
        sample_too_small=True,
        lookback_days=30,
    )
    needed = _MIN_SAMPLE_SIGNIFICANT - 12
    assert f"Se necesitan {needed} más" in result


def test_small_sample_shows_game_count():
    """The interpretation includes how many games are currently tracked."""
    result = _build_interpretation_es(
        total=7,
        avg_ai=4.5,
        avg_actual=5.0,
        manager_advantage=0.5,
        sample_too_small=True,
        lookback_days=30,
    )
    assert "7 partido" in result


def test_162_game_projection_math():
    """Season projection = |advantage| * 162, rounded to 1 decimal."""
    advantage = 0.5
    expected_proj = round(advantage * 162, 1)  # 81.0

    result = _build_interpretation_es(
        total=20,
        avg_ai=4.0,
        avg_actual=4.5,
        manager_advantage=advantage,
        sample_too_small=False,
        lookback_days=30,
    )
    assert str(expected_proj) in result


def test_interpretation_includes_mean_values():
    """Both avg_ai and avg_actual should appear in the interpretation text."""
    result = _build_interpretation_es(
        total=20,
        avg_ai=4.50,
        avg_actual=5.00,
        manager_advantage=0.50,
        sample_too_small=False,
        lookback_days=30,
    )
    assert "4.50" in result
    assert "5.00" in result
