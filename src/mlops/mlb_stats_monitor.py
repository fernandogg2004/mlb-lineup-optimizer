"""
mlb_stats_monitor.py — Real-time roster and injury monitor via MLB StatsAPI.
=============================================================================
Polls the MLB StatsAPI Roster and Transactions endpoints every 15 minutes.
On any change it:
    1. Updates the PostgreSQL `player_availability` table (upsert).
    2. Publishes an invalidation message to Redis so the optimizer cache
       drops stale lineup scores for affected players.

Designed to run as a background APScheduler job launched inside the FastAPI
lifespan, or as a standalone daemon (``python -m src.mlops.mlb_stats_monitor``).

PostgreSQL table (must be pre-created — see CREATE TABLE below):
    player_availability(
        player_id       INTEGER PRIMARY KEY,
        player_name     TEXT    NOT NULL,
        team_id         INTEGER,
        status          TEXT    NOT NULL,   -- "active" | "10day_il" | "60day_il" | "dfa" | "unknown"
        status_detail   TEXT,
        injury_note     TEXT,
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

Required environment variables:
    MLB_STATS_API_BASE   — default: https://statsapi.mlb.com/api/v1
    MLB_TEAM_ID          — integer team ID to monitor (e.g. 147 for NYY)
    DATABASE_URL         — PostgreSQL DSN (asyncpg or psycopg2 format)
    REDIS_URL            — Redis connection string (default: redis://localhost:6379)
    MONITOR_POLL_SEC     — poll interval in seconds (default: 900 = 15 min)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import psycopg2
import psycopg2.extras
import redis
import requests
import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from requests.exceptions import RequestException

logger = structlog.get_logger(__name__)

_DEFAULT_API_BASE = "https://statsapi.mlb.com/api/v1"
_DEFAULT_POLL_SEC = 900  # 15 minutes
_REQUEST_TIMEOUT = 10    # seconds per HTTP call
_CACHE_INVALIDATION_CHANNEL = "optimizer:cache:invalidate"

# Mapping from MLB roster status codes to our canonical status strings
_STATUS_MAP: dict[str, str] = {
    "Active": "active",
    "10-Day IL": "10day_il",
    "60-Day IL": "60day_il",
    "Designated for Assignment": "dfa",
    "Restricted List": "restricted",
    "Paternity List": "paternity",
    "Bereavement/Family Medical Emergency List": "bereavement",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MonitorConfig:
    """Runtime configuration sourced from environment variables.

    Attributes:
        api_base: Base URL for MLB StatsAPI.
        team_id: MLB team ID to monitor.
        database_url: PostgreSQL connection string.
        redis_url: Redis connection string.
        poll_seconds: Polling interval in seconds.
    """

    api_base: str = field(default_factory=lambda: os.environ.get("MLB_STATS_API_BASE", _DEFAULT_API_BASE))
    team_id: int = field(default_factory=lambda: int(os.environ.get("MLB_TEAM_ID", "147")))
    database_url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", ""))
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379"))
    poll_seconds: int = field(default_factory=lambda: int(os.environ.get("MONITOR_POLL_SEC", str(_DEFAULT_POLL_SEC))))

    @classmethod
    def from_env(cls) -> "MonitorConfig":
        return cls()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PlayerAvailability:
    """Canonical availability record for a single player.

    Attributes:
        player_id: MLB player ID (integer).
        player_name: Full player name.
        team_id: MLB team ID.
        status: Canonical status string from _STATUS_MAP.
        status_detail: Raw detail string from the API response.
        injury_note: Free-text injury description if available.
        updated_at: UTC timestamp of this record.
    """

    player_id: int
    player_name: str
    team_id: int
    status: str
    status_detail: Optional[str]
    injury_note: Optional[str]
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ---------------------------------------------------------------------------
# MLB StatsAPI client
# ---------------------------------------------------------------------------


class MLBStatsAPIClient:
    """Thin wrapper around the MLB StatsAPI Roster and Transactions endpoints.

    Args:
        api_base: Base URL, e.g. ``https://statsapi.mlb.com/api/v1``.
    """

    def __init__(self, api_base: str = _DEFAULT_API_BASE) -> None:
        self._base = api_base.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def get_roster(self, team_id: int) -> list[dict[str, Any]]:
        """Fetch the 40-man roster for the given team.

        Args:
            team_id: MLB team ID.

        Returns:
            List of raw roster entry dicts from the API.

        Raises:
            RequestException: On HTTP or network error.
        """
        url = f"{self._base}/teams/{team_id}/roster/40Man"
        resp = self._session.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("roster", [])

    def get_transactions(self, team_id: int) -> list[dict[str, Any]]:
        """Fetch recent transactions (last 7 days) for the given team.

        Args:
            team_id: MLB team ID.

        Returns:
            List of raw transaction dicts from the API.

        Raises:
            RequestException: On HTTP or network error.
        """
        url = f"{self._base}/transactions"
        params = {
            "teamId": team_id,
            "limit": 50,
        }
        resp = self._session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("transactions", [])

    def parse_roster(self, roster: list[dict[str, Any]], team_id: int) -> list[PlayerAvailability]:
        """Convert raw roster entries to PlayerAvailability records.

        Args:
            roster: Raw roster list from get_roster().
            team_id: Team ID to attach to each record.

        Returns:
            List of PlayerAvailability instances.
        """
        result: list[PlayerAvailability] = []
        for entry in roster:
            person = entry.get("person", {})
            status_obj = entry.get("status", {})
            raw_status = status_obj.get("description", "Unknown")
            canonical = _STATUS_MAP.get(raw_status, "unknown")

            result.append(PlayerAvailability(
                player_id=int(person.get("id", 0)),
                player_name=person.get("fullName", ""),
                team_id=team_id,
                status=canonical,
                status_detail=raw_status,
                injury_note=entry.get("note"),
            ))
        return result


# ---------------------------------------------------------------------------
# PostgreSQL persistence
# ---------------------------------------------------------------------------


class PlayerAvailabilityStore:
    """Upserts PlayerAvailability records into PostgreSQL.

    Args:
        database_url: PostgreSQL DSN.
    """

    def __init__(self, database_url: str) -> None:
        self._dsn = database_url

    def _connect(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(self._dsn)

    def upsert_batch(self, records: list[PlayerAvailability]) -> int:
        """Upsert a batch of availability records.

        Args:
            records: PlayerAvailability instances to persist.

        Returns:
            Number of rows affected (inserted + updated).

        Raises:
            psycopg2.Error: On database error.
        """
        if not records:
            return 0

        sql = """
            INSERT INTO player_availability
                (player_id, player_name, team_id, status, status_detail, injury_note, updated_at)
            VALUES %s
            ON CONFLICT (player_id) DO UPDATE SET
                player_name   = EXCLUDED.player_name,
                team_id       = EXCLUDED.team_id,
                status        = EXCLUDED.status,
                status_detail = EXCLUDED.status_detail,
                injury_note   = EXCLUDED.injury_note,
                updated_at    = EXCLUDED.updated_at
        """
        values = [
            (
                r.player_id,
                r.player_name,
                r.team_id,
                r.status,
                r.status_detail,
                r.injury_note,
                r.updated_at,
            )
            for r in records
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, sql, values)
            conn.commit()

        logger.info("player_availability_upserted", n_rows=len(records))
        return len(records)

    def get_changed_player_ids(
        self,
        incoming: list[PlayerAvailability],
    ) -> list[int]:
        """Return player IDs whose status has changed since the last upsert.

        Args:
            incoming: New availability snapshot to compare against the DB.

        Returns:
            List of player_id ints with a different status than stored.

        Raises:
            psycopg2.Error: On database error.
        """
        if not incoming:
            return []

        player_ids = [r.player_id for r in incoming]
        sql = "SELECT player_id, status FROM player_availability WHERE player_id = ANY(%s)"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (player_ids,))
                stored: dict[int, str] = {row[0]: row[1] for row in cur.fetchall()}

        changed = [
            r.player_id
            for r in incoming
            if stored.get(r.player_id) != r.status
        ]
        return changed


# ---------------------------------------------------------------------------
# Redis cache invalidation
# ---------------------------------------------------------------------------


class CacheInvalidator:
    """Publishes player ID lists to Redis so the optimizer drops stale cache entries.

    Args:
        redis_url: Redis connection string.
    """

    def __init__(self, redis_url: str) -> None:
        self._client = redis.from_url(redis_url, decode_responses=True)

    def invalidate(self, player_ids: list[int]) -> None:
        """Publish player IDs that need cache eviction.

        Args:
            player_ids: List of player IDs whose cached scores are stale.
        """
        if not player_ids:
            return
        import json as _json
        payload = _json.dumps({"player_ids": player_ids, "ts": time.time()})
        self._client.publish(_CACHE_INVALIDATION_CHANNEL, payload)
        logger.info(
            "cache_invalidation_published",
            n_players=len(player_ids),
            channel=_CACHE_INVALIDATION_CHANNEL,
        )


# ---------------------------------------------------------------------------
# Monitor orchestrator
# ---------------------------------------------------------------------------


class MLBStatsMonitor:
    """Polls MLB StatsAPI, detects roster changes, and propagates updates downstream.

    Args:
        config: MonitorConfig instance. Defaults to from_env().
    """

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self._config = config or MonitorConfig.from_env()
        self._api = MLBStatsAPIClient(api_base=self._config.api_base)
        self._store = PlayerAvailabilityStore(database_url=self._config.database_url)
        self._cache = CacheInvalidator(redis_url=self._config.redis_url)
        logger.info(
            "mlb_stats_monitor_initialized",
            team_id=self._config.team_id,
            poll_seconds=self._config.poll_seconds,
        )

    def poll_once(self) -> dict[str, int]:
        """Execute one full poll cycle: fetch → diff → upsert → invalidate.

        Returns:
            Dict with keys: ``fetched``, ``changed``, ``upserted``.

        Raises:
            RequestException: If the StatsAPI call fails.
            psycopg2.Error: If the database write fails.
        """
        try:
            raw_roster = self._api.get_roster(self._config.team_id)
        except RequestException as exc:
            logger.error(
                "roster_fetch_failed",
                team_id=self._config.team_id,
                error=str(exc),
            )
            raise

        records = self._api.parse_roster(raw_roster, self._config.team_id)

        try:
            changed_ids = self._store.get_changed_player_ids(records)
        except Exception as exc:
            logger.warning(
                "change_detection_failed",
                error=str(exc),
                fallback="treating all players as changed",
            )
            changed_ids = [r.player_id for r in records]

        try:
            n_upserted = self._store.upsert_batch(records)
        except Exception as exc:
            logger.error("db_upsert_failed", error=str(exc))
            raise

        if changed_ids:
            try:
                self._cache.invalidate(changed_ids)
            except Exception as exc:
                # Cache invalidation failure is non-fatal; optimizer will self-heal on next TTL
                logger.warning("cache_invalidation_failed", error=str(exc))

        stats = {"fetched": len(records), "changed": len(changed_ids), "upserted": n_upserted}
        logger.info("poll_cycle_complete", **stats)
        return stats

    def start_scheduler(self) -> BackgroundScheduler:
        """Start a background APScheduler that calls poll_once every poll_seconds.

        Returns:
            Running BackgroundScheduler instance (caller is responsible for shutdown).
        """
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            self._safe_poll,
            trigger="interval",
            seconds=self._config.poll_seconds,
            id="mlb_stats_monitor",
            replace_existing=True,
        )
        scheduler.start()
        logger.info(
            "monitor_scheduler_started",
            interval_seconds=self._config.poll_seconds,
        )
        return scheduler

    def _safe_poll(self) -> None:
        """Wrapper that catches all exceptions to prevent scheduler from dying on transient errors."""
        try:
            self.poll_once()
        except Exception as exc:
            logger.error("poll_cycle_error", error=str(exc), exc_info=True)


# ---------------------------------------------------------------------------
# Standalone daemon entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the monitor as a long-lived daemon process."""
    monitor = MLBStatsMonitor()
    scheduler = monitor.start_scheduler()
    logger.info("mlb_stats_monitor_daemon_running")
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("mlb_stats_monitor_shutdown")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
