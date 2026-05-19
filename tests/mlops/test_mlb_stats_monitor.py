"""
Tests for MLBStatsMonitor — roster parsing, change detection, and poll orchestration.

Database, Redis, and HTTP calls are mocked throughout.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest
from requests.exceptions import HTTPError

from src.mlops.mlb_stats_monitor import (
    CacheInvalidator,
    MLBStatsAPIClient,
    MLBStatsMonitor,
    MonitorConfig,
    PlayerAvailability,
    PlayerAvailabilityStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_roster_entry(
    player_id: int,
    player_name: str,
    status: str = "Active",
    note: str | None = None,
) -> dict:
    return {
        "person": {"id": player_id, "fullName": player_name},
        "status": {"description": status},
        "note": note,
    }


def _make_mock_conn(fetchall_rows: list) -> MagicMock:
    """Build a psycopg2-like mock connection supporting with-statement usage."""
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_rows
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.fixture
def api_client() -> MLBStatsAPIClient:
    return MLBStatsAPIClient(api_base="http://mock-mlb")


@pytest.fixture
def monitor() -> MLBStatsMonitor:
    config = MonitorConfig(
        api_base="http://mock-mlb",
        team_id=147,
        database_url="postgresql://test",
        redis_url="redis://test",
        poll_seconds=900,
    )
    with patch("src.mlops.mlb_stats_monitor.redis.from_url"):
        m = MLBStatsMonitor(config)
    m._api = MagicMock(spec=MLBStatsAPIClient)
    m._store = MagicMock(spec=PlayerAvailabilityStore)
    m._cache = MagicMock(spec=CacheInvalidator)
    return m


# ---------------------------------------------------------------------------
# MLBStatsAPIClient.parse_roster — status mapping
# ---------------------------------------------------------------------------


def test_parse_roster_active_maps_to_active(api_client: MLBStatsAPIClient):
    roster = [_raw_roster_entry(592450, "Aaron Judge", "Active")]
    records = api_client.parse_roster(roster, team_id=147)
    assert records[0].status == "active"


def test_parse_roster_ten_day_il_maps_correctly(api_client: MLBStatsAPIClient):
    roster = [_raw_roster_entry(592450, "Aaron Judge", "10-Day IL")]
    records = api_client.parse_roster(roster, team_id=147)
    assert records[0].status == "10day_il"


def test_parse_roster_sixty_day_il_maps_correctly(api_client: MLBStatsAPIClient):
    roster = [_raw_roster_entry(592450, "Aaron Judge", "60-Day IL")]
    records = api_client.parse_roster(roster, team_id=147)
    assert records[0].status == "60day_il"


def test_parse_roster_unknown_status_maps_to_unknown(api_client: MLBStatsAPIClient):
    roster = [_raw_roster_entry(592450, "Aaron Judge", "Some Weird Status")]
    records = api_client.parse_roster(roster, team_id=147)
    assert records[0].status == "unknown"


def test_parse_roster_extracts_player_id_and_name(api_client: MLBStatsAPIClient):
    roster = [_raw_roster_entry(592450, "Aaron Judge", "Active")]
    records = api_client.parse_roster(roster, team_id=147)
    assert records[0].player_id == 592450
    assert records[0].player_name == "Aaron Judge"
    assert records[0].team_id == 147


def test_parse_roster_includes_injury_note(api_client: MLBStatsAPIClient):
    roster = [_raw_roster_entry(592450, "Aaron Judge", "10-Day IL", note="Right knee strain")]
    records = api_client.parse_roster(roster, team_id=147)
    assert records[0].injury_note == "Right knee strain"


def test_parse_roster_empty_returns_empty_list(api_client: MLBStatsAPIClient):
    assert api_client.parse_roster([], team_id=147) == []


# ---------------------------------------------------------------------------
# PlayerAvailabilityStore.get_changed_player_ids
# ---------------------------------------------------------------------------


def test_get_changed_player_ids_detects_status_change():
    store = PlayerAvailabilityStore(database_url="postgresql://test")
    incoming = [
        PlayerAvailability(592450, "Aaron Judge", 147, "10day_il", "10-Day IL", None),
    ]
    mock_conn = _make_mock_conn(fetchall_rows=[(592450, "active")])

    with patch.object(store, "_connect", return_value=mock_conn):
        changed = store.get_changed_player_ids(incoming)

    assert 592450 in changed


def test_get_changed_player_ids_same_status_not_in_result():
    store = PlayerAvailabilityStore(database_url="postgresql://test")
    incoming = [
        PlayerAvailability(592450, "Aaron Judge", 147, "active", "Active", None),
    ]
    mock_conn = _make_mock_conn(fetchall_rows=[(592450, "active")])

    with patch.object(store, "_connect", return_value=mock_conn):
        changed = store.get_changed_player_ids(incoming)

    assert 592450 not in changed


def test_get_changed_player_ids_new_player_not_in_db_is_changed():
    """Player absent from DB (new addition to roster) counts as changed."""
    store = PlayerAvailabilityStore(database_url="postgresql://test")
    incoming = [
        PlayerAvailability(999999, "New Player", 147, "active", "Active", None),
    ]
    mock_conn = _make_mock_conn(fetchall_rows=[])  # no rows returned — player not in DB

    with patch.object(store, "_connect", return_value=mock_conn):
        changed = store.get_changed_player_ids(incoming)

    assert 999999 in changed


def test_get_changed_player_ids_empty_input_returns_empty():
    store = PlayerAvailabilityStore(database_url="postgresql://test")
    changed = store.get_changed_player_ids([])
    assert changed == []


# ---------------------------------------------------------------------------
# MLBStatsMonitor.poll_once — orchestration
# ---------------------------------------------------------------------------


def _make_records(n: int = 3) -> list[PlayerAvailability]:
    return [
        PlayerAvailability(600000 + i, f"Player {i}", 147, "active", "Active", None)
        for i in range(n)
    ]


def test_poll_once_calls_get_roster(monitor: MLBStatsMonitor):
    records = _make_records(3)
    monitor._api.get_roster.return_value = []
    monitor._api.parse_roster.return_value = records
    monitor._store.get_changed_player_ids.return_value = []
    monitor._store.upsert_batch.return_value = 3

    monitor.poll_once()

    monitor._api.get_roster.assert_called_once_with(147)


def test_poll_once_upserts_all_fetched_players(monitor: MLBStatsMonitor):
    records = _make_records(5)
    monitor._api.get_roster.return_value = []
    monitor._api.parse_roster.return_value = records
    monitor._store.get_changed_player_ids.return_value = []
    monitor._store.upsert_batch.return_value = 5

    stats = monitor.poll_once()

    monitor._store.upsert_batch.assert_called_once_with(records)
    assert stats["upserted"] == 5


def test_poll_once_invalidates_cache_for_changed_players(monitor: MLBStatsMonitor):
    records = _make_records(3)
    changed_ids = [600001, 600002]
    monitor._api.get_roster.return_value = []
    monitor._api.parse_roster.return_value = records
    monitor._store.get_changed_player_ids.return_value = changed_ids
    monitor._store.upsert_batch.return_value = 3

    monitor.poll_once()

    monitor._cache.invalidate.assert_called_once_with(changed_ids)


def test_poll_once_does_not_invalidate_when_no_changes(monitor: MLBStatsMonitor):
    records = _make_records(3)
    monitor._api.get_roster.return_value = []
    monitor._api.parse_roster.return_value = records
    monitor._store.get_changed_player_ids.return_value = []
    monitor._store.upsert_batch.return_value = 3

    monitor.poll_once()

    monitor._cache.invalidate.assert_not_called()


def test_poll_once_returns_correct_stats(monitor: MLBStatsMonitor):
    records = _make_records(4)
    monitor._api.get_roster.return_value = []
    monitor._api.parse_roster.return_value = records
    monitor._store.get_changed_player_ids.return_value = [600000, 600001]
    monitor._store.upsert_batch.return_value = 4

    stats = monitor.poll_once()

    assert stats["fetched"] == 4
    assert stats["changed"] == 2
    assert stats["upserted"] == 4


def test_poll_once_raises_on_api_request_exception(monitor: MLBStatsMonitor):
    from requests.exceptions import ConnectionError as ReqConnectionError
    monitor._api.get_roster.side_effect = ReqConnectionError("Connection refused")

    with pytest.raises(ReqConnectionError):
        monitor.poll_once()


def test_poll_once_cache_failure_is_non_fatal(monitor: MLBStatsMonitor):
    """Redis failure during invalidation should not prevent a successful poll."""
    records = _make_records(2)
    monitor._api.get_roster.return_value = []
    monitor._api.parse_roster.return_value = records
    monitor._store.get_changed_player_ids.return_value = [600000]
    monitor._store.upsert_batch.return_value = 2
    monitor._cache.invalidate.side_effect = Exception("Redis connection refused")

    stats = monitor.poll_once()  # must not raise

    assert stats["upserted"] == 2
