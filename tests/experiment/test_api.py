"""
FastAPI endpoint integration tests for the A/B Experiment Tracker.

Covers all three endpoints:
    POST /v1/experiment/record-divergence
    POST /v1/experiment/record-outcome
    GET  /v1/experiment/summary

Approach:
    - TestClient WITHOUT context manager → lifespan does not run (avoids heavy model loading).
    - _state.experiment_store is injected manually via the `client` fixture.
    - State is cleaned up after each test.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from src.experiment.experiment_store import ExperimentStore, ExperimentStoreConfig
from tests.conftest import (
    AI_EXPECTED_RUNS,
    AI_LINEUP,
    MANAGER_LINEUP_ORDER_ONLY,
    MANAGER_LINEUP_PLAYER_SWAP,
    make_closed_game,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TODAY = datetime.date.today().isoformat()
_GAME_ID = f"api-test-{_TODAY}-NYY-BOS"

_DIVERGENCE_PAYLOAD: dict = {
    "game_id": _GAME_ID,
    "game_date": _TODAY,
    "ai_lineup_ids": AI_LINEUP,
    "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
    "ai_expected_runs": AI_EXPECTED_RUNS,
    "manager_rationale": "El lanzador rival tiene debilidad contra el primer bateador actual",
}

_OUTCOME_PAYLOAD: dict = {
    "game_id": _GAME_ID,
    "actual_runs_scored": 5,
    "innings_played": 9,
}


@pytest.fixture
def exp_store(tmp_path: Path) -> ExperimentStore:
    return ExperimentStore(ExperimentStoreConfig(store_path=tmp_path / "ab"))


@pytest.fixture
def client(exp_store: ExperimentStore):
    """TestClient with experiment_store injected. Lifespan is NOT started."""
    original = main_module._state.experiment_store
    main_module._state.experiment_store = exp_store

    # No context manager → lifespan (heavy model loading) does not run.
    test_client = TestClient(main_module.app, raise_server_exceptions=True)
    yield test_client

    main_module._state.experiment_store = original


# ---------------------------------------------------------------------------
# POST /v1/experiment/record-divergence — happy path
# ---------------------------------------------------------------------------


def test_record_divergence_returns_200(client: TestClient):
    resp = client.post("/v1/experiment/record-divergence", json=_DIVERGENCE_PAYLOAD)
    assert resp.status_code == 200


def test_record_divergence_response_has_observation_id(client: TestClient):
    resp = client.post("/v1/experiment/record-divergence", json=_DIVERGENCE_PAYLOAD)
    body = resp.json()
    assert "observation_id" in body
    assert len(body["observation_id"]) > 0


def test_record_divergence_classifies_order_only(client: TestClient):
    resp = client.post("/v1/experiment/record-divergence", json=_DIVERGENCE_PAYLOAD)
    body = resp.json()
    assert body["divergence_type"] == "order_only"
    assert body["slots_changed"] == 2  # first two slots swapped
    assert body["players_changed"] == 0


def test_record_divergence_classifies_player_swap(client: TestClient):
    payload = {**_DIVERGENCE_PAYLOAD, "manager_lineup_ids": MANAGER_LINEUP_PLAYER_SWAP}
    resp = client.post("/v1/experiment/record-divergence", json=payload)
    assert resp.status_code == 200
    assert resp.json()["divergence_type"] == "player_swap"


def test_record_divergence_message_es_in_response(client: TestClient):
    resp = client.post("/v1/experiment/record-divergence", json=_DIVERGENCE_PAYLOAD)
    body = resp.json()
    assert "message_es" in body
    assert len(body["message_es"]) > 0


# ---------------------------------------------------------------------------
# POST /v1/experiment/record-divergence — validation errors
# ---------------------------------------------------------------------------


def test_record_divergence_invalid_game_date_format_returns_422(client: TestClient):
    payload = {**_DIVERGENCE_PAYLOAD, "game_date": "19/05/2026"}  # wrong format
    resp = client.post("/v1/experiment/record-divergence", json=payload)
    assert resp.status_code == 422


def test_record_divergence_short_ai_lineup_returns_422(client: TestClient):
    payload = {**_DIVERGENCE_PAYLOAD, "ai_lineup_ids": AI_LINEUP[:8]}  # only 8 players
    resp = client.post("/v1/experiment/record-divergence", json=payload)
    assert resp.status_code == 422


def test_record_divergence_negative_ai_expected_runs_returns_422(client: TestClient):
    payload = {**_DIVERGENCE_PAYLOAD, "ai_expected_runs": -1.0}
    resp = client.post("/v1/experiment/record-divergence", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /v1/experiment/record-outcome — happy path
# ---------------------------------------------------------------------------


def test_record_outcome_returns_200_after_divergence(client: TestClient):
    client.post("/v1/experiment/record-divergence", json=_DIVERGENCE_PAYLOAD)
    resp = client.post("/v1/experiment/record-outcome", json=_OUTCOME_PAYLOAD)
    assert resp.status_code == 200


def test_record_outcome_computes_er_differential(client: TestClient):
    client.post("/v1/experiment/record-divergence", json=_DIVERGENCE_PAYLOAD)
    resp = client.post("/v1/experiment/record-outcome", json=_OUTCOME_PAYLOAD)
    body = resp.json()

    expected_diff = round(5 - AI_EXPECTED_RUNS, 3)  # 5 - 4.50 = 0.50
    assert pytest.approx(body["er_differential"], abs=1e-3) == expected_diff
    assert body["observations_updated"] == 1


def test_record_outcome_unknown_game_id_returns_404(client: TestClient):
    resp = client.post(
        "/v1/experiment/record-outcome",
        json={"game_id": "no-such-game-xyz", "actual_runs_scored": 3},
    )
    assert resp.status_code == 404


def test_record_outcome_second_call_returns_404(client: TestClient):
    """Recording the outcome twice for the same game returns 404 on the second call."""
    client.post("/v1/experiment/record-divergence", json=_DIVERGENCE_PAYLOAD)
    client.post("/v1/experiment/record-outcome", json=_OUTCOME_PAYLOAD)  # closes it
    resp = client.post("/v1/experiment/record-outcome", json=_OUTCOME_PAYLOAD)  # no open obs
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/experiment/summary
# ---------------------------------------------------------------------------


def test_summary_empty_store_returns_200(client: TestClient):
    resp = client.get("/v1/experiment/summary")
    assert resp.status_code == 200


def test_summary_empty_store_shows_zero_games(client: TestClient):
    body = client.get("/v1/experiment/summary").json()
    assert body["total_closed_games"] == 0
    assert body["sample_too_small"] is True


def test_summary_lookback_zero_returns_422(client: TestClient):
    resp = client.get("/v1/experiment/summary?lookback_days=0")
    assert resp.status_code == 422


def test_summary_lookback_too_large_returns_422(client: TestClient):
    resp = client.get("/v1/experiment/summary?lookback_days=400")
    assert resp.status_code == 422


def test_summary_custom_lookback_accepted(client: TestClient):
    resp = client.get("/v1/experiment/summary?lookback_days=30")
    assert resp.status_code == 200
    assert resp.json()["lookback_days"] == 30


# ---------------------------------------------------------------------------
# Full E2E flow — divergence → outcome → summary
# ---------------------------------------------------------------------------


def test_full_e2e_single_game(client: TestClient, exp_store: ExperimentStore):
    """
    Record one divergence, close it with an outcome, then verify the summary
    reflects exactly that one game with the correct statistics.
    """
    # Step 1: record divergence
    div_resp = client.post("/v1/experiment/record-divergence", json=_DIVERGENCE_PAYLOAD)
    assert div_resp.status_code == 200

    # Step 2: record outcome (manager scored 5 runs, AI expected 4.50)
    out_resp = client.post("/v1/experiment/record-outcome", json=_OUTCOME_PAYLOAD)
    assert out_resp.status_code == 200
    assert out_resp.json()["er_differential"] == pytest.approx(0.50, abs=1e-3)

    # Step 3: verify summary
    summary = client.get("/v1/experiment/summary").json()
    assert summary["total_closed_games"] == 1
    assert summary["sample_too_small"] is True  # 1 < 15
    assert pytest.approx(summary["avg_ai_expected_runs"], abs=1e-3) == AI_EXPECTED_RUNS
    assert pytest.approx(summary["avg_actual_runs_scored"], abs=1e-3) == 5.0
    assert pytest.approx(summary["manager_advantage_runs"], abs=1e-3) == 0.50
    assert "interpretation_es" in summary


def test_e2e_response_includes_all_schema_fields(client: TestClient):
    """Verify the response bodies conform to the documented schema fields."""
    div_resp = client.post("/v1/experiment/record-divergence", json=_DIVERGENCE_PAYLOAD)
    div_body = div_resp.json()
    for field in ("observation_id", "game_id", "divergence_type", "slots_changed",
                  "players_changed", "message_es"):
        assert field in div_body, f"Missing field '{field}' in divergence response"

    client.post("/v1/experiment/record-outcome", json=_OUTCOME_PAYLOAD)
    summary_body = client.get("/v1/experiment/summary").json()
    for field in ("lookback_days", "total_closed_games", "by_divergence_type",
                  "avg_ai_expected_runs", "avg_actual_runs_scored",
                  "manager_advantage_runs", "sample_too_small", "interpretation_es"):
        assert field in summary_body, f"Missing field '{field}' in summary response"
