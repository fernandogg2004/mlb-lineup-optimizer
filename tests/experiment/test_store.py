"""
Integration tests for ExperimentStore.

Uses pytest's tmp_path fixture so each test gets a clean filesystem. Covers
all three public methods: record_divergence, record_outcome, get_summary.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.experiment.experiment_store import (
    ExperimentStore,
    ExperimentStoreConfig,
    _MIN_SAMPLE_SIGNIFICANT,
)
from tests.conftest import (
    AI_EXPECTED_RUNS,
    AI_LINEUP,
    MANAGER_LINEUP_FULL_OVERRIDE,
    MANAGER_LINEUP_ORDER_ONLY,
    MANAGER_LINEUP_PLAYER_SWAP,
    make_closed_game,
)

TODAY = date.today().isoformat()
GAME_ID = "store-test-2026-NYY-BOS"
OBS_ID = "storetest000000a"  # exactly 16 chars


# ---------------------------------------------------------------------------
# record_divergence
# ---------------------------------------------------------------------------


def test_record_divergence_creates_parquet_file(store: ExperimentStore):
    store.record_divergence(
        game_id=GAME_ID,
        game_date=TODAY,
        ai_lineup_ids=AI_LINEUP,
        manager_lineup_ids=MANAGER_LINEUP_ORDER_ONLY,
        ai_expected_runs=AI_EXPECTED_RUNS,
        manager_rationale="Swing decision",
        observation_id=OBS_ID,
    )
    expected_file = store._config.store_path / f"game_date={TODAY}" / f"{OBS_ID}.parquet"
    assert expected_file.exists()


def test_record_divergence_parquet_has_correct_schema(store: ExperimentStore):
    store.record_divergence(
        game_id=GAME_ID,
        game_date=TODAY,
        ai_lineup_ids=AI_LINEUP,
        manager_lineup_ids=MANAGER_LINEUP_ORDER_ONLY,
        ai_expected_runs=AI_EXPECTED_RUNS,
        manager_rationale="",
        observation_id=OBS_ID,
    )
    df = pl.read_parquet(
        store._config.store_path / f"game_date={TODAY}" / f"{OBS_ID}.parquet",
        hive_partitioning=False,
    )
    assert df["status"][0] == "open"
    assert df["game_id"][0] == GAME_ID
    assert abs(float(df["ai_expected_runs"][0]) - AI_EXPECTED_RUNS) < 1e-6
    assert df["actual_runs_scored"][0] is None  # null until closed


@pytest.mark.parametrize("manager_lineup,expected_type,min_players", [
    (MANAGER_LINEUP_ORDER_ONLY, "order_only", 0),
    (MANAGER_LINEUP_PLAYER_SWAP, "player_swap", 1),
    (MANAGER_LINEUP_FULL_OVERRIDE, "full_override", 2),
])
def test_record_divergence_returns_correct_classification(
    store: ExperimentStore,
    manager_lineup: list[int],
    expected_type: str,
    min_players: int,
):
    dtype, slots, players = store.record_divergence(
        game_id=GAME_ID,
        game_date=TODAY,
        ai_lineup_ids=AI_LINEUP,
        manager_lineup_ids=manager_lineup,
        ai_expected_runs=AI_EXPECTED_RUNS,
        manager_rationale="",
        observation_id=OBS_ID,
    )
    assert dtype == expected_type
    assert players >= min_players


def test_record_divergence_identical_lineup_is_order_only_zero_slots(store: ExperimentStore):
    dtype, slots, players = store.record_divergence(
        game_id=GAME_ID,
        game_date=TODAY,
        ai_lineup_ids=AI_LINEUP,
        manager_lineup_ids=list(AI_LINEUP),  # exact copy
        ai_expected_runs=AI_EXPECTED_RUNS,
        manager_rationale="",
        observation_id=OBS_ID,
    )
    assert dtype == "order_only"
    assert slots == 0
    assert players == 0


# ---------------------------------------------------------------------------
# record_outcome
# ---------------------------------------------------------------------------


def test_record_outcome_closes_open_observation(store: ExperimentStore):
    store.record_divergence(
        game_id=GAME_ID,
        game_date=TODAY,
        ai_lineup_ids=AI_LINEUP,
        manager_lineup_ids=MANAGER_LINEUP_ORDER_ONLY,
        ai_expected_runs=AI_EXPECTED_RUNS,
        manager_rationale="",
        observation_id=OBS_ID,
    )
    n_updated, _ = store.record_outcome(game_id=GAME_ID, actual_runs_scored=6)

    assert n_updated == 1
    df = pl.read_parquet(
        store._config.store_path / f"game_date={TODAY}" / f"{OBS_ID}.parquet",
        hive_partitioning=False,
    )
    assert df["status"][0] == "closed"
    assert int(df["actual_runs_scored"][0]) == 6


def test_record_outcome_returns_ai_expected_runs(store: ExperimentStore):
    store.record_divergence(
        game_id=GAME_ID,
        game_date=TODAY,
        ai_lineup_ids=AI_LINEUP,
        manager_lineup_ids=MANAGER_LINEUP_ORDER_ONLY,
        ai_expected_runs=AI_EXPECTED_RUNS,
        manager_rationale="",
        observation_id=OBS_ID,
    )
    n_updated, ai_er = store.record_outcome(game_id=GAME_ID, actual_runs_scored=5)

    assert n_updated == 1
    assert pytest.approx(ai_er, rel=1e-4) == AI_EXPECTED_RUNS


def test_record_outcome_unknown_game_id_returns_zero(store: ExperimentStore):
    n_updated, ai_er = store.record_outcome(game_id="no-such-game", actual_runs_scored=3)
    assert n_updated == 0
    assert ai_er == 0.0


def test_record_outcome_idempotent_on_closed_observation(store: ExperimentStore):
    store.record_divergence(
        game_id=GAME_ID,
        game_date=TODAY,
        ai_lineup_ids=AI_LINEUP,
        manager_lineup_ids=MANAGER_LINEUP_ORDER_ONLY,
        ai_expected_runs=AI_EXPECTED_RUNS,
        manager_rationale="",
        observation_id=OBS_ID,
    )
    store.record_outcome(game_id=GAME_ID, actual_runs_scored=5)  # first close
    n_updated, _ = store.record_outcome(game_id=GAME_ID, actual_runs_scored=5)  # second call

    assert n_updated == 0  # already closed, nothing to update


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------


def test_get_summary_empty_store(store: ExperimentStore):
    summary = store.get_summary(lookback_days=90)
    assert summary["total_closed_games"] == 0
    assert summary["sample_too_small"] is True
    assert summary["avg_ai_expected_runs"] is None
    assert summary["manager_advantage_runs"] is None
    assert "Sin datos" in summary["interpretation_es"]


def test_get_summary_14_games_is_below_threshold(store_with_14_closed: ExperimentStore):
    summary = store_with_14_closed.get_summary(lookback_days=90)
    assert summary["total_closed_games"] == 14
    assert summary["sample_too_small"] is True
    assert "Muestra insuficiente" in summary["interpretation_es"]


def test_get_summary_15_games_meets_threshold(store_with_15_closed: ExperimentStore):
    summary = store_with_15_closed.get_summary(lookback_days=90)
    assert summary["total_closed_games"] == 15
    assert summary["sample_too_small"] is False


def test_get_summary_computes_correct_manager_advantage(store_with_15_closed: ExperimentStore):
    """ai_expected=4.50, actual=5 for all 15 games → advantage = +0.50."""
    summary = store_with_15_closed.get_summary(lookback_days=90)
    assert pytest.approx(summary["avg_ai_expected_runs"], rel=1e-3) == 4.50
    assert pytest.approx(summary["avg_actual_runs_scored"], rel=1e-3) == 5.0
    assert pytest.approx(summary["manager_advantage_runs"], rel=1e-3) == 0.50


def test_get_summary_by_divergence_type_breakdown(store_with_20_mixed: ExperimentStore):
    """20 games: 7 order_only, 7 player_swap, 6 full_override."""
    summary = store_with_20_mixed.get_summary(lookback_days=90)
    assert summary["total_closed_games"] == 20
    assert summary["by_divergence_type"]["order_only"] == 7
    assert summary["by_divergence_type"]["player_swap"] == 7
    assert summary["by_divergence_type"]["full_override"] == 6


def test_get_summary_negative_advantage_when_manager_underperforms(store: ExperimentStore):
    """Manager scores fewer runs than AI expected → negative advantage."""
    for i in range(15):
        make_closed_game(store, i, MANAGER_LINEUP_ORDER_ONLY, actual_runs=3, ai_er=5.0)

    summary = store.get_summary(lookback_days=90)
    assert summary["manager_advantage_runs"] < 0
    assert "habría producido" in summary["interpretation_es"]


def test_get_summary_respects_lookback_window(store: ExperimentStore):
    """Observations older than lookback_days are excluded from the summary."""
    # Create 5 games from 100+ days ago — outside lookback window
    for i in range(5):
        days_back = 100 + i
        game_date = (date.today() - timedelta(days=days_back)).isoformat()
        game_id = f"old-game-{i:04d}"
        obs_id = f"oldobs{i:010d}"
        store.record_divergence(
            game_id=game_id,
            game_date=game_date,
            ai_lineup_ids=AI_LINEUP,
            manager_lineup_ids=MANAGER_LINEUP_ORDER_ONLY,
            ai_expected_runs=AI_EXPECTED_RUNS,
            manager_rationale="",
            observation_id=obs_id,
        )
        # record_outcome won't find these (outside 14-day search window),
        # so we simulate a closed observation by writing directly
        partition = store._config.store_path / f"game_date={game_date}" / f"{obs_id}.parquet"
        df = pl.read_parquet(partition, hive_partitioning=False)
        df = df.with_columns([
            pl.lit(5).cast(pl.Int64).alias("actual_runs_scored"),
            pl.lit(9).cast(pl.Int64).alias("innings_played"),
            pl.lit("closed").alias("status"),
            pl.lit("2026-01-01T00:00:00+00:00").alias("closed_at"),
        ])
        df.write_parquet(partition)

    summary = store.get_summary(lookback_days=90)
    assert summary["total_closed_games"] == 0  # all are beyond 90 days
