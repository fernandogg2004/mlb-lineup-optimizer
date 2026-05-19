"""
what_if_engine.py
=================
Stateless business-logic layer for the What-If scenario analysis feature.

Responsibilities:
  - Hold the baseline optimal lineup as a reference scenario
  - Call the FastAPI /v1/optimize/lineup endpoint with modified lineups
  - Compute ΔE[R] and ΔP(W) relative to baseline
  - Classify the impact magnitude (HIGH / MEDIUM / LOW / NEUTRAL)
  - Provide colour-coding metadata consumed by the Streamlit UI

Completely separated from Streamlit so it can be unit-tested with httpx mocking.

Usage:
    engine = WhatIfEngine(api_base_url="http://localhost:8000")
    engine.set_baseline(baseline_scenario)
    modified = engine.build_scenario(new_player_list, ...)
    result  = engine.fetch_scenario(modified, pitcher_hand="R")
    report  = engine.delta(result)
    print(report.summary_line())   # "ΔE[R] = +0.18 ↑  ΔP(W) = +3.2% ↑  [HIGH IMPACT]"
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import httpx
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# League-average PA outcome probs used as opponent baseline when not supplied
_LEAGUE_AVG_PROBS: list[list[float]] = [
    [0.284, 0.222, 0.082, 0.148, 0.048, 0.009, 0.028]
] * 9

# Impact thresholds (absolute ΔE[R])
_HIGH_THRESHOLD = 0.15
_MEDIUM_THRESHOLD = 0.05
_NEUTRAL_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class PlayerProfile:
    """Minimal player record needed by the optimizer API."""

    player_id: int
    player_name: str
    obp: float
    woba: float
    iso: float
    batter_stand: str                # "L", "R", or "S"
    prob_vector: list[float]         # 7-element PA outcome probability vector


@dataclass
class LineupScenario:
    """Full representation of one lineup scenario for comparison."""

    players: list[PlayerProfile]     # order = batting positions 1–9
    expected_runs: float
    win_probability: float
    optimization_mode: str = "fast"
    total_evaluations: int = 0
    elapsed_seconds: float = 0.0
    scenario_name: str = "baseline"
    api_error: Optional[str] = None  # set when the API call failed

    @property
    def player_ids(self) -> list[int]:
        return [p.player_id for p in self.players]

    @property
    def player_names(self) -> list[str]:
        return [p.player_name for p in self.players]


@dataclass
class DeltaReport:
    """Comparison between a what-if scenario and the current baseline."""

    delta_er: float
    delta_pw: float
    baseline_er: float
    baseline_pw: float
    scenario_er: float
    scenario_pw: float
    is_improvement: bool
    impact_level: str               # "HIGH" | "MEDIUM" | "LOW" | "NEUTRAL"
    color_er: str                   # "#4caf50" (green) or "#f44336" (red) or "#9e9e9e"
    color_pw: str

    def summary_line(self) -> str:
        arrow_er = "↑" if self.delta_er > 0 else ("↓" if self.delta_er < 0 else "→")
        arrow_pw = "↑" if self.delta_pw > 0 else ("↓" if self.delta_pw < 0 else "→")
        return (
            f"ΔE[R] = {self.delta_er:+.3f} {arrow_er}  |  "
            f"ΔP(W) = {self.delta_pw:+.1%} {arrow_pw}  "
            f"[{self.impact_level} IMPACT]"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class WhatIfEngine:
    """Manages scenario evaluation against the FastAPI backend.

    Instantiated once per Streamlit session and stored in st.session_state.
    Thread-safe for Streamlit's single-threaded execution model.
    """

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        timeout: float = 90.0,
    ) -> None:
        self._api_base_url = api_base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self.baseline: LineupScenario | None = None
        logger.info("what_if_engine_ready", api=self._api_base_url)

    @classmethod
    def from_env(cls) -> "WhatIfEngine":
        return cls(
            api_base_url=os.environ.get("API_BASE_URL", "http://localhost:8000"),
        )

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def set_baseline(self, scenario: LineupScenario) -> None:
        self.baseline = scenario
        logger.info(
            "baseline_set",
            er=scenario.expected_runs,
            pw=scenario.win_probability,
            players=scenario.player_names,
        )

    def has_baseline(self) -> bool:
        return self.baseline is not None

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    @staticmethod
    def build_request_payload(
        players: list[PlayerProfile],
        pitcher_id: int,
        pitcher_name: str,
        pitcher_hand: str,
        park_factor_hr: float,
        park_factor_xb: float,
        opp_lineup_probs: list[list[float]] | None,
        fast_mode: bool,
    ) -> dict:
        return {
            "roster": [
                {
                    "player_id": p.player_id,
                    "player_name": p.player_name,
                    "obp": p.obp,
                    "woba": p.woba,
                    "iso": p.iso,
                    "batter_stand": p.batter_stand,
                    "prob_vector": p.prob_vector,
                }
                for p in players
            ],
            "rival_pitcher": {
                "pitcher_id": pitcher_id,
                "pitcher_name": pitcher_name,
                "hand": pitcher_hand,
                "era": 4.00,           # placeholder when not available
                "pitch_mix": {"FF": 0.40, "SL": 0.25, "CH": 0.20, "CU": 0.15},
            },
            "opp_lineup_probs": opp_lineup_probs or _LEAGUE_AVG_PROBS,
            "park_factor_hr": park_factor_hr,
            "park_factor_xb": park_factor_xb,
            "fast_mode": fast_mode,
        }

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def fetch_scenario(
        self,
        players: list[PlayerProfile],
        pitcher_id: int,
        pitcher_name: str,
        pitcher_hand: str,
        park_factor_hr: float = 1.0,
        park_factor_xb: float = 1.0,
        opp_lineup_probs: list[list[float]] | None = None,
        fast_mode: bool = True,
        scenario_name: str = "what-if",
    ) -> LineupScenario:
        """Call /v1/optimize/lineup and return a LineupScenario.

        On API failure, returns a LineupScenario with api_error set so the UI
        can display a graceful error state instead of crashing.
        """
        payload = self.build_request_payload(
            players=players,
            pitcher_id=pitcher_id,
            pitcher_name=pitcher_name,
            pitcher_hand=pitcher_hand,
            park_factor_hr=park_factor_hr,
            park_factor_xb=park_factor_xb,
            opp_lineup_probs=opp_lineup_probs,
            fast_mode=fast_mode,
        )
        try:
            response = self._client.post(
                f"{self._api_base_url}/v1/optimize/lineup",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Reconstruct ordered PlayerProfile list from API response
            id_to_profile = {p.player_id: p for p in players}
            ordered_players = [
                id_to_profile.get(pid, players[i])
                for i, pid in enumerate(data["lineup_player_ids"])
            ]

            result = LineupScenario(
                players=ordered_players,
                expected_runs=data["expected_runs"],
                win_probability=data["win_probability"],
                optimization_mode=data["optimization_mode"],
                total_evaluations=data["total_evaluations"],
                elapsed_seconds=data["elapsed_seconds"],
                scenario_name=scenario_name,
            )
            logger.info(
                "what_if_scenario_fetched",
                scenario=scenario_name,
                er=result.expected_runs,
                pw=result.win_probability,
            )
            return result

        except Exception as exc:
            logger.error("what_if_api_error", error=str(exc))
            return LineupScenario(
                players=players,
                expected_runs=0.0,
                win_probability=0.0,
                scenario_name=scenario_name,
                api_error=str(exc),
            )

    # ------------------------------------------------------------------
    # Delta computation
    # ------------------------------------------------------------------

    def delta(self, scenario: LineupScenario) -> DeltaReport:
        """Compute ΔE[R] and ΔP(W) relative to the stored baseline."""
        if self.baseline is None:
            raise ValueError("Call set_baseline() before computing delta.")
        if scenario.api_error:
            raise RuntimeError(f"Scenario has API error: {scenario.api_error}")

        d_er = scenario.expected_runs - self.baseline.expected_runs
        d_pw = scenario.win_probability - self.baseline.win_probability

        abs_d_er = abs(d_er)
        if abs_d_er < _NEUTRAL_THRESHOLD:
            impact = "NEUTRAL"
        elif abs_d_er < _MEDIUM_THRESHOLD:
            impact = "LOW"
        elif abs_d_er < _HIGH_THRESHOLD:
            impact = "MEDIUM"
        else:
            impact = "HIGH"

        _green = "#4caf50"
        _red = "#f44336"
        _grey = "#9e9e9e"

        return DeltaReport(
            delta_er=d_er,
            delta_pw=d_pw,
            baseline_er=self.baseline.expected_runs,
            baseline_pw=self.baseline.win_probability,
            scenario_er=scenario.expected_runs,
            scenario_pw=scenario.win_probability,
            is_improvement=d_er > _NEUTRAL_THRESHOLD,
            impact_level=impact,
            color_er=_green if d_er > _NEUTRAL_THRESHOLD else (_red if d_er < -_NEUTRAL_THRESHOLD else _grey),
            color_pw=_green if d_pw > 0.002 else (_red if d_pw < -0.002 else _grey),
        )


# ---------------------------------------------------------------------------
# Zone data generation (synthetic — replace with real Statcast zone splits)
# ---------------------------------------------------------------------------


def synthetic_zone_xwoba(player_id: int, xwoba: float, batter_stand: str) -> np.ndarray:
    """Generate a 5×5 strike-zone xwOBA matrix seeded by player_id.

    Row 0 = high, Row 4 = low. Col 0 = inside, Col 4 = outside (pitcher POV).
    The 3×3 inner block (rows 1–3, cols 1–3) represents the true strike zone.

    Returns:
        float32 array shape (5, 5), values in [0.15, 0.65].
    """
    rng = np.random.default_rng(player_id % 100_000)
    grid = rng.normal(xwoba, 0.038, (5, 5)).astype(np.float32)

    # Strike-zone centre boost
    grid[1:4, 1:4] += 0.025

    # Platoon tendency: LHB stronger on outer half, RHB on inner half
    if batter_stand == "L":
        grid[:, 3:] += 0.018
    else:
        grid[:, :2] += 0.018

    # High pitches are generally harder to control
    grid[0, :] -= 0.012

    return np.clip(grid, 0.15, 0.65)
