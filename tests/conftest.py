"""
Shared pytest fixtures for experiment A/B tests.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from src.experiment.experiment_store import ExperimentStore, ExperimentStoreConfig

# ---------------------------------------------------------------------------
# Nine realistic MLB player IDs (used consistently across all tests)
# ---------------------------------------------------------------------------
AI_LINEUP = [592450, 545361, 660271, 665742, 518934, 621566, 571448, 669242, 608369]

# Same 9 players, batting positions 0 and 1 swapped → order_only divergence
MANAGER_LINEUP_ORDER_ONLY = [545361, 592450, 660271, 665742, 518934, 621566, 571448, 669242, 608369]

# Player at slot 4 replaced by an unknown batter → player_swap divergence
MANAGER_LINEUP_PLAYER_SWAP = [592450, 545361, 660271, 665742, 999001, 621566, 571448, 669242, 608369]

# Three players replaced → full_override divergence
MANAGER_LINEUP_FULL_OVERRIDE = [592450, 545361, 660271, 888001, 518934, 777002, 571448, 666003, 608369]

AI_EXPECTED_RUNS = 4.50


# ---------------------------------------------------------------------------
# Store fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ExperimentStore:
    """Fresh ExperimentStore backed by a temporary directory."""
    return ExperimentStore(ExperimentStoreConfig(store_path=tmp_path / "ab_experiment"))


# ---------------------------------------------------------------------------
# Helper — create a fully closed observation in one call
# ---------------------------------------------------------------------------


def make_closed_game(
    store: ExperimentStore,
    index: int,
    manager_lineup: list[int],
    actual_runs: int,
    ai_er: float = AI_EXPECTED_RUNS,
    date_offset: int | None = None,
) -> str:
    """Record a divergence and immediately close it with actual_runs.

    Args:
        date_offset: Days back for the game_date partition. Defaults to index.
            Use this to keep dates within the record_outcome search window (14 days)
            while still having unique game IDs.
    """
    day = date_offset if date_offset is not None else index
    game_date = (date.today() - timedelta(days=day)).isoformat()
    game_id = f"test-game-{index:04d}"
    obs_id = f"testobs{index:09d}"  # exactly 16 chars

    store.record_divergence(
        game_id=game_id,
        game_date=game_date,
        ai_lineup_ids=AI_LINEUP,
        manager_lineup_ids=manager_lineup,
        ai_expected_runs=ai_er,
        manager_rationale="",
        observation_id=obs_id,
    )
    store.record_outcome(
        game_id=game_id,
        actual_runs_scored=actual_runs,
        innings_played=9,
    )
    return game_id


# ---------------------------------------------------------------------------
# Pre-built stores with N closed observations
# ---------------------------------------------------------------------------


@pytest.fixture
def store_with_14_closed(store: ExperimentStore) -> ExperimentStore:
    """14 closed observations — just below the significance threshold."""
    for i in range(14):
        make_closed_game(store, i, MANAGER_LINEUP_ORDER_ONLY, actual_runs=5)
    return store


@pytest.fixture
def store_with_15_closed(store: ExperimentStore) -> ExperimentStore:
    """15 closed observations — exactly at the significance threshold.

    ai_expected_runs = 4.50 for all games.
    actual_runs_scored = 5 for all games.
    Expected manager_advantage_runs = 5.0 - 4.5 = +0.50.
    """
    for i in range(15):
        make_closed_game(store, i, MANAGER_LINEUP_ORDER_ONLY, actual_runs=5)
    return store


@pytest.fixture
def store_with_20_mixed(store: ExperimentStore) -> ExperimentStore:
    """20 closed observations with mixed divergence types and varying actual runs.

    All game dates are mapped to the last 13 days (date_offset = i % 13) so that
    record_outcome's 14-day search window covers every observation.

    Distribution: 7 order_only | 7 player_swap | 6 full_override
    """
    for i in range(7):
        make_closed_game(store, i, MANAGER_LINEUP_ORDER_ONLY, actual_runs=5, date_offset=i % 13)
    for i in range(7, 14):
        make_closed_game(store, i, MANAGER_LINEUP_PLAYER_SWAP, actual_runs=3, date_offset=i % 13)
    for i in range(14, 20):
        make_closed_game(store, i, MANAGER_LINEUP_FULL_OVERRIDE, actual_runs=4, date_offset=i % 13)
    return store
