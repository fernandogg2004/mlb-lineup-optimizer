"""
post_game_resolver.py — Nightly Airflow job: close open A/B experiment observations.
=====================================================================================
Runs after all MLB games on a given date have ended (typically 02:00 ET next day).
For each date that has open observations in the ExperimentStore it:
    1. Fetches the MLB StatsAPI schedule to get gamePks for that date.
    2. Matches each open observation's game_id (format: YYYY-MM-DD-{away}-{home})
       to a gamePk by parsing the away/home team abbreviations.
    3. Fetches the boxscore for each matched gamePk to get the actual runs scored
       by the home team (our lineup).
    4. Calls ExperimentStore.record_outcome() to close the observation.

The Airflow DAG should call resolve_pending_outcomes() as a PythonOperator task
with execution_date as the target date.

Required environment variables:
    MLB_STATS_API_BASE     — default: https://statsapi.mlb.com/api/v1
    EXPERIMENT_STORE_PATH  — path to the ExperimentStore Parquet directory
    RESOLVER_LOOKBACK_DAYS — how many days back to scan for open observations (default: 3)
    RESOLVER_HOME_TEAM_ID  — MLB team ID of the team whose runs we record (e.g. 147 for NYY)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import requests
import structlog
from requests.exceptions import RequestException

from src.experiment.experiment_store import ExperimentStore, ExperimentStoreConfig

logger = structlog.get_logger(__name__)

_DEFAULT_API_BASE = "https://statsapi.mlb.com/api/v1"
_DEFAULT_LOOKBACK = 3
_REQUEST_TIMEOUT = 10

# game_id format: YYYY-MM-DD-{away_abbrev}-{home_abbrev}
_GAME_ID_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<away>[A-Z]{2,4})-(?P<home>[A-Z]{2,4})$"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ResolverConfig:
    """Runtime configuration for the post-game resolver.

    Attributes:
        api_base: MLB StatsAPI base URL.
        store_path: ExperimentStore root directory.
        lookback_days: Number of days back to scan for open observations.
        home_team_id: MLB team ID whose runs are recorded as actual_runs_scored.
    """

    api_base: str = field(
        default_factory=lambda: os.environ.get("MLB_STATS_API_BASE", _DEFAULT_API_BASE)
    )
    store_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("EXPERIMENT_STORE_PATH", "data/gold/ab_experiment")
        )
    )
    lookback_days: int = field(
        default_factory=lambda: int(os.environ.get("RESOLVER_LOOKBACK_DAYS", str(_DEFAULT_LOOKBACK)))
    )
    home_team_id: int = field(
        default_factory=lambda: int(os.environ.get("RESOLVER_HOME_TEAM_ID", "147"))
    )

    @classmethod
    def from_env(cls) -> "ResolverConfig":
        return cls()


# ---------------------------------------------------------------------------
# MLB StatsAPI helpers
# ---------------------------------------------------------------------------


class MLBScheduleClient:
    """Fetches MLB schedule and boxscore data from StatsAPI.

    Args:
        api_base: Base URL for the MLB StatsAPI.
    """

    def __init__(self, api_base: str = _DEFAULT_API_BASE) -> None:
        self._base = api_base.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def get_schedule(self, game_date: str) -> list[dict[str, Any]]:
        """Fetch all games scheduled on a given date.

        Args:
            game_date: Date in YYYY-MM-DD format.

        Returns:
            List of game dicts, each containing ``gamePk``, ``teams``, etc.

        Raises:
            RequestException: On network or HTTP error.
        """
        url = f"{self._base}/schedule"
        params = {
            "sportId": 1,
            "date": game_date,
            "hydrate": "team",
        }
        resp = self._session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        games: list[dict[str, Any]] = []
        for day in data.get("dates", []):
            games.extend(day.get("games", []))
        return games

    def get_boxscore_runs(self, game_pk: int) -> Optional[dict[str, int]]:
        """Fetch runs scored by each team from a game boxscore.

        Args:
            game_pk: MLB integer game primary key.

        Returns:
            Dict with keys ``home`` and ``away`` (int runs scored), or None if
            the game has not yet started or data is unavailable.

        Raises:
            RequestException: On network or HTTP error.
        """
        url = f"{self._base}/game/{game_pk}/boxscore"
        resp = self._session.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        teams = data.get("teams", {})
        home_runs = teams.get("home", {}).get("teamStats", {}).get("batting", {}).get("runs")
        away_runs = teams.get("away", {}).get("teamStats", {}).get("batting", {}).get("runs")

        if home_runs is None or away_runs is None:
            return None

        return {"home": int(home_runs), "away": int(away_runs)}

    def build_team_abbrev_map(self, games: list[dict[str, Any]]) -> dict[str, int]:
        """Build a mapping from team abbreviation → gamePk for a list of games.

        Args:
            games: Raw game list from get_schedule().

        Returns:
            Dict mapping team abbreviation (upper-case) to its gamePk for that date.
            Each gamePk appears twice (once for home, once for away team).
        """
        abbrev_to_pk: dict[str, int] = {}
        for game in games:
            game_pk = game.get("gamePk")
            if game_pk is None:
                continue
            teams = game.get("teams", {})
            for side in ("home", "away"):
                abbrev = (
                    teams.get(side, {})
                    .get("team", {})
                    .get("abbreviation", "")
                    .upper()
                )
                if abbrev:
                    abbrev_to_pk[abbrev] = int(game_pk)
        return abbrev_to_pk


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------


@dataclass
class ResolutionSummary:
    """Statistics for a single resolver run.

    Attributes:
        target_date: The calendar date processed.
        open_found: Number of open observations found.
        resolved: Number successfully closed.
        skipped_no_match: Observations where no matching gamePk was found.
        skipped_no_boxscore: Observations where boxscore data was unavailable.
        errors: Number of observations that raised an exception during processing.
    """

    target_date: str
    open_found: int = 0
    resolved: int = 0
    skipped_no_match: int = 0
    skipped_no_boxscore: int = 0
    errors: int = 0


class PostGameResolver:
    """Closes open ExperimentStore observations using MLB StatsAPI boxscore data.

    Args:
        config: ResolverConfig. Defaults to from_env().
    """

    def __init__(self, config: ResolverConfig | None = None) -> None:
        self._config = config or ResolverConfig.from_env()
        self._store = ExperimentStore(
            ExperimentStoreConfig(store_path=self._config.store_path)
        )
        self._api = MLBScheduleClient(api_base=self._config.api_base)
        logger.info(
            "post_game_resolver_initialized",
            home_team_id=self._config.home_team_id,
            lookback_days=self._config.lookback_days,
        )

    def resolve_pending_outcomes(
        self,
        target_date: Optional[str] = None,
    ) -> list[ResolutionSummary]:
        """Close all open observations for dates in the lookback window.

        Args:
            target_date: ISO date string (YYYY-MM-DD) to use as "today".
                Defaults to today's date. Used by Airflow for backfill.

        Returns:
            List of ResolutionSummary, one per calendar date processed.
        """
        today = date.fromisoformat(target_date) if target_date else date.today()
        summaries: list[ResolutionSummary] = []

        for days_back in range(self._config.lookback_days + 1):
            check_date = today - timedelta(days=days_back)
            summary = self._resolve_date(check_date.isoformat())
            summaries.append(summary)

        total_resolved = sum(s.resolved for s in summaries)
        total_open = sum(s.open_found for s in summaries)
        logger.info(
            "resolve_run_complete",
            dates_checked=len(summaries),
            total_open_found=total_open,
            total_resolved=total_resolved,
        )
        return summaries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_date(self, game_date: str) -> ResolutionSummary:
        """Attempt to close all open observations for a single calendar date.

        Args:
            game_date: ISO date string (YYYY-MM-DD).

        Returns:
            ResolutionSummary for this date.
        """
        summary = ResolutionSummary(target_date=game_date)
        open_game_ids = self._find_open_game_ids(game_date)
        summary.open_found = len(open_game_ids)

        if not open_game_ids:
            logger.debug("no_open_observations", game_date=game_date)
            return summary

        logger.info(
            "resolving_date",
            game_date=game_date,
            n_open=len(open_game_ids),
        )

        # Fetch MLB schedule and build abbreviation → gamePk map
        try:
            games = self._api.get_schedule(game_date)
            abbrev_map = self._api.build_team_abbrev_map(games)
        except RequestException as exc:
            logger.error(
                "schedule_fetch_failed",
                game_date=game_date,
                error=str(exc),
            )
            summary.errors += len(open_game_ids)
            return summary

        for game_id in open_game_ids:
            try:
                result = self._close_single_game(game_id, abbrev_map)
                if result == "resolved":
                    summary.resolved += 1
                elif result == "no_match":
                    summary.skipped_no_match += 1
                elif result == "no_boxscore":
                    summary.skipped_no_boxscore += 1
            except Exception as exc:
                logger.error(
                    "game_resolution_error",
                    game_id=game_id,
                    error=str(exc),
                    exc_info=True,
                )
                summary.errors += 1

        return summary

    def _find_open_game_ids(self, game_date: str) -> list[str]:
        """Return game_ids of open observations for the given date.

        Reads the partition directory directly so we never scan the full store.

        Args:
            game_date: ISO date string.

        Returns:
            List of game_id strings with at least one open observation.
        """
        partition_dir = self._config.store_path / f"game_date={game_date}"
        if not partition_dir.exists():
            return []

        import polars as pl

        game_ids: set[str] = set()
        for parquet_file in partition_dir.glob("*.parquet"):
            try:
                df = pl.read_parquet(parquet_file, hive_partitioning=False)
                open_df = df.filter(pl.col("status") == "open")
                for gid in open_df["game_id"].to_list():
                    game_ids.add(gid)
            except Exception as exc:
                logger.warning(
                    "parquet_read_error",
                    file=str(parquet_file),
                    error=str(exc),
                )
        return list(game_ids)

    def _close_single_game(
        self,
        game_id: str,
        abbrev_map: dict[str, int],
    ) -> str:
        """Match a game_id to a gamePk and close its observations.

        Args:
            game_id: Internal game ID string (YYYY-MM-DD-AWAY-HOME).
            abbrev_map: Team abbreviation → gamePk for the schedule date.

        Returns:
            One of: ``"resolved"``, ``"no_match"``, ``"no_boxscore"``.

        Raises:
            RequestException: If the boxscore API call fails.
        """
        match = _GAME_ID_PATTERN.match(game_id)
        if not match:
            logger.warning("unparseable_game_id", game_id=game_id)
            return "no_match"

        home_abbrev = match.group("home").upper()
        away_abbrev = match.group("away").upper()

        # Try to find gamePk by home team first, then away
        game_pk = abbrev_map.get(home_abbrev) or abbrev_map.get(away_abbrev)
        if game_pk is None:
            logger.warning(
                "game_pk_not_found",
                game_id=game_id,
                home=home_abbrev,
                away=away_abbrev,
                available_teams=list(abbrev_map.keys()),
            )
            return "no_match"

        try:
            runs = self._api.get_boxscore_runs(game_pk)
        except RequestException as exc:
            logger.error("boxscore_fetch_failed", game_pk=game_pk, error=str(exc))
            raise

        if runs is None:
            logger.warning("boxscore_runs_unavailable", game_pk=game_pk, game_id=game_id)
            return "no_boxscore"

        # Record runs for the home team (our monitored team)
        actual_runs = runs["home"]
        n_updated, ai_er = self._store.record_outcome(
            game_id=game_id,
            actual_runs_scored=actual_runs,
        )

        if n_updated > 0:
            logger.info(
                "observation_closed",
                game_id=game_id,
                game_pk=game_pk,
                actual_runs=actual_runs,
                ai_expected_runs=ai_er,
                er_differential=actual_runs - ai_er,
            )
        else:
            logger.debug("observation_already_closed_or_not_found", game_id=game_id)

        return "resolved"


# ---------------------------------------------------------------------------
# Airflow-compatible callable
# ---------------------------------------------------------------------------


def resolve_pending_outcomes(
    execution_date: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, int]:
    """Airflow PythonOperator callable.

    Args:
        execution_date: Airflow execution date string (YYYY-MM-DD). Defaults to today.
        **kwargs: Unused Airflow context kwargs.

    Returns:
        Aggregated stats dict for XCom: resolved, open_found, errors.
    """
    resolver = PostGameResolver()
    summaries = resolver.resolve_pending_outcomes(target_date=execution_date)

    totals = {
        "resolved": sum(s.resolved for s in summaries),
        "open_found": sum(s.open_found for s in summaries),
        "skipped_no_match": sum(s.skipped_no_match for s in summaries),
        "skipped_no_boxscore": sum(s.skipped_no_boxscore for s in summaries),
        "errors": sum(s.errors for s in summaries),
    }
    logger.info("airflow_task_complete", **totals)
    return totals


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the resolver once for the given date (or today)."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="MLB post-game A/B experiment resolver")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    result = resolve_pending_outcomes(execution_date=args.date)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
