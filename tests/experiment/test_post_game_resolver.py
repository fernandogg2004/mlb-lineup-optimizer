"""
Tests for PostGameResolver — MLB schedule parsing and A/B experiment observation closure.

Unit tests mock the HTTP layer. The integration tests use a real ExperimentStore
on a tmp_path to verify the full open→closed lifecycle.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.experiment.experiment_store import ExperimentStore, ExperimentStoreConfig
from src.experiment.post_game_resolver import (
    MLBScheduleClient,
    PostGameResolver,
    ResolverConfig,
)
from tests.conftest import AI_EXPECTED_RUNS, AI_LINEUP, MANAGER_LINEUP_ORDER_ONLY

TODAY = date.today().isoformat()

# game_id that matches the resolver's _GAME_ID_PATTERN
GAME_ID = f"{TODAY}-BOS-NYY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_schedule_game(game_pk: int, away_abbrev: str, home_abbrev: str) -> dict:
    return {
        "gamePk": game_pk,
        "teams": {
            "home": {"team": {"abbreviation": home_abbrev}},
            "away": {"team": {"abbreviation": away_abbrev}},
        },
    }


def _fake_boxscore_response(home_runs: int, away_runs: int) -> dict:
    return {
        "teams": {
            "home": {"teamStats": {"batting": {"runs": home_runs}}},
            "away": {"teamStats": {"batting": {"runs": away_runs}}},
        }
    }


@pytest.fixture
def api_client() -> MLBScheduleClient:
    return MLBScheduleClient(api_base="http://mock-mlb")


@pytest.fixture
def exp_store(tmp_path: Path) -> ExperimentStore:
    return ExperimentStore(ExperimentStoreConfig(store_path=tmp_path / "ab_experiment"))


@pytest.fixture
def resolver(tmp_path: Path) -> PostGameResolver:
    config = ResolverConfig(
        api_base="http://mock-mlb",
        store_path=tmp_path / "ab_experiment",
        lookback_days=0,  # only today — keeps integration tests deterministic
        home_team_id=147,
    )
    r = PostGameResolver(config)
    r._api = MagicMock()  # replace real HTTP client with mock
    return r


# ---------------------------------------------------------------------------
# MLBScheduleClient.build_team_abbrev_map
# ---------------------------------------------------------------------------


def test_build_team_abbrev_map_extracts_home_and_away(api_client: MLBScheduleClient):
    games = [_fake_schedule_game(12345, "BOS", "NYY")]
    abbrev_map = api_client.build_team_abbrev_map(games)
    assert abbrev_map["NYY"] == 12345
    assert abbrev_map["BOS"] == 12345


def test_build_team_abbrev_map_empty_games_returns_empty(api_client: MLBScheduleClient):
    assert api_client.build_team_abbrev_map([]) == {}


def test_build_team_abbrev_map_skips_games_without_game_pk(api_client: MLBScheduleClient):
    games = [{"teams": {"home": {"team": {"abbreviation": "NYY"}}, "away": {"team": {"abbreviation": "BOS"}}}}]
    abbrev_map = api_client.build_team_abbrev_map(games)
    assert abbrev_map == {}


def test_build_team_abbrev_map_multiple_games(api_client: MLBScheduleClient):
    games = [
        _fake_schedule_game(100, "BOS", "NYY"),
        _fake_schedule_game(101, "LAD", "SF"),
    ]
    abbrev_map = api_client.build_team_abbrev_map(games)
    assert abbrev_map["NYY"] == 100
    assert abbrev_map["LAD"] == 101
    assert abbrev_map["SF"] == 101


# ---------------------------------------------------------------------------
# MLBScheduleClient.get_boxscore_runs
# ---------------------------------------------------------------------------


def test_get_boxscore_runs_returns_home_and_away(api_client: MLBScheduleClient):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _fake_boxscore_response(home_runs=5, away_runs=3)

    with patch.object(api_client._session, "get", return_value=mock_resp):
        runs = api_client.get_boxscore_runs(game_pk=12345)

    assert runs == {"home": 5, "away": 3}


def test_get_boxscore_runs_returns_none_when_runs_missing(api_client: MLBScheduleClient):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"teams": {"home": {}, "away": {}}}

    with patch.object(api_client._session, "get", return_value=mock_resp):
        runs = api_client.get_boxscore_runs(game_pk=12345)

    assert runs is None


# ---------------------------------------------------------------------------
# PostGameResolver._close_single_game — unit
# ---------------------------------------------------------------------------


def test_close_single_game_unparseable_id_returns_no_match(resolver: PostGameResolver):
    result = resolver._close_single_game("invalid-game-id", abbrev_map={"NYY": 123})
    assert result == "no_match"


def test_close_single_game_team_not_in_abbrev_map_returns_no_match(resolver: PostGameResolver):
    # GAME_ID = TODAY-BOS-NYY but abbrev_map has no NYY or BOS
    result = resolver._close_single_game(GAME_ID, abbrev_map={"LAD": 999})
    assert result == "no_match"


def test_close_single_game_no_boxscore_returns_no_boxscore(resolver: PostGameResolver):
    resolver._api.get_boxscore_runs.return_value = None
    result = resolver._close_single_game(GAME_ID, abbrev_map={"NYY": 12345, "BOS": 12345})
    assert result == "no_boxscore"


def test_close_single_game_calls_record_outcome_with_home_runs(resolver: PostGameResolver):
    resolver._api.get_boxscore_runs.return_value = {"home": 7, "away": 2}
    resolver._store.record_outcome = MagicMock(return_value=(1, AI_EXPECTED_RUNS))

    resolver._close_single_game(GAME_ID, abbrev_map={"NYY": 12345, "BOS": 12345})

    resolver._store.record_outcome.assert_called_once_with(
        game_id=GAME_ID,
        actual_runs_scored=7,  # home runs, not away
    )


# ---------------------------------------------------------------------------
# Integration tests — real ExperimentStore + mocked HTTP
# ---------------------------------------------------------------------------


def _create_open_observation(exp_store: ExperimentStore, game_id: str) -> str:
    obs_id = "integobs000000a"
    exp_store.record_divergence(
        game_id=game_id,
        game_date=TODAY,
        ai_lineup_ids=AI_LINEUP,
        manager_lineup_ids=MANAGER_LINEUP_ORDER_ONLY,
        ai_expected_runs=AI_EXPECTED_RUNS,
        manager_rationale="Test",
        observation_id=obs_id,
    )
    return obs_id


def _read_obs(store_path: Path, obs_id: str) -> pl.DataFrame:
    parquet = store_path / f"game_date={TODAY}" / f"{obs_id}.parquet"
    return pl.read_parquet(parquet, hive_partitioning=False)


def test_integration_open_observation_is_closed_after_resolve(
    resolver: PostGameResolver,
    exp_store: ExperimentStore,
    tmp_path: Path,
):
    obs_id = _create_open_observation(exp_store, GAME_ID)

    resolver._api.get_schedule.return_value = [_fake_schedule_game(9999, "BOS", "NYY")]
    resolver._api.build_team_abbrev_map.return_value = {"NYY": 9999, "BOS": 9999}
    resolver._api.get_boxscore_runs.return_value = {"home": 5, "away": 3}

    summaries = resolver.resolve_pending_outcomes(target_date=TODAY)

    assert sum(s.resolved for s in summaries) == 1

    df = _read_obs(tmp_path / "ab_experiment", obs_id)
    assert df["status"][0] == "closed"
    assert int(df["actual_runs_scored"][0]) == 5


def test_integration_er_differential_recorded_correctly(
    resolver: PostGameResolver,
    exp_store: ExperimentStore,
    tmp_path: Path,
):
    """actual_runs (6) - ai_expected (4.50) = +1.50 should be recoverable from the store."""
    obs_id = _create_open_observation(exp_store, GAME_ID)

    resolver._api.get_schedule.return_value = [_fake_schedule_game(9999, "BOS", "NYY")]
    resolver._api.build_team_abbrev_map.return_value = {"NYY": 9999, "BOS": 9999}
    resolver._api.get_boxscore_runs.return_value = {"home": 6, "away": 2}

    resolver.resolve_pending_outcomes(target_date=TODAY)

    df = _read_obs(tmp_path / "ab_experiment", obs_id)
    actual = int(df["actual_runs_scored"][0])
    ai_er = float(df["ai_expected_runs"][0])
    assert pytest.approx(actual - ai_er, rel=1e-3) == 1.50


def test_integration_no_open_observations_returns_zero_resolved(
    resolver: PostGameResolver,
):
    summaries = resolver.resolve_pending_outcomes(target_date=TODAY)
    assert all(s.resolved == 0 for s in summaries)
    assert all(s.open_found == 0 for s in summaries)


def test_integration_already_closed_observation_not_double_closed(
    resolver: PostGameResolver,
    exp_store: ExperimentStore,
    tmp_path: Path,
):
    obs_id = _create_open_observation(exp_store, GAME_ID)
    # Close it first via the store directly
    exp_store.record_outcome(game_id=GAME_ID, actual_runs_scored=4)

    resolver._api.get_schedule.return_value = [_fake_schedule_game(9999, "BOS", "NYY")]
    resolver._api.build_team_abbrev_map.return_value = {"NYY": 9999, "BOS": 9999}
    resolver._api.get_boxscore_runs.return_value = {"home": 5, "away": 3}

    summaries = resolver.resolve_pending_outcomes(target_date=TODAY)

    # record_outcome returns (0, 0.0) for already-closed — resolver counts it as "resolved"
    # (it ran without error) but the stored value stays as the first close (4 runs)
    df = _read_obs(tmp_path / "ab_experiment", obs_id)
    assert int(df["actual_runs_scored"][0]) == 4  # original value preserved


def test_integration_schedule_api_error_increments_errors(
    resolver: PostGameResolver,
    exp_store: ExperimentStore,
):
    from requests.exceptions import ConnectionError as ReqConnectionError

    _create_open_observation(exp_store, GAME_ID)
    resolver._api.get_schedule.side_effect = ReqConnectionError("Connection refused")

    summaries = resolver.resolve_pending_outcomes(target_date=TODAY)

    assert sum(s.errors for s in summaries) >= 1
    assert sum(s.resolved for s in summaries) == 0
