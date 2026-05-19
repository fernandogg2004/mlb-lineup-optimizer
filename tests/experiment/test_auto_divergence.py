"""
Tests for the automatic divergence recording via POST /v1/lineup/confirm.

Uses FastAPI TestClient with _state patched so no real models are needed.
All ExperimentStore calls use a real store on tmp_path to verify persistence.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import _AppState, _CachedRecommendation, _LINEUP_CACHE_TTL_SECONDS, app
from src.experiment.experiment_store import ExperimentStore, ExperimentStoreConfig
from tests.conftest import AI_EXPECTED_RUNS, AI_LINEUP, MANAGER_LINEUP_ORDER_ONLY, MANAGER_LINEUP_PLAYER_SWAP

TODAY = "2026-05-19"
GAME_ID = f"{TODAY}-NYY-BOS"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs_id(game_id: str) -> str:
    return hashlib.sha256(game_id.encode()).hexdigest()[:16]


def _make_state(tmp_path: Path, cached: _CachedRecommendation | None = None) -> _AppState:
    store = ExperimentStore(ExperimentStoreConfig(store_path=tmp_path / "ab"))
    state = _AppState(experiment_store=store)
    if cached is not None:
        state.lineup_cache[cached.game_id] = cached
    return state


def _fresh_cached(game_id: str = GAME_ID) -> _CachedRecommendation:
    return _CachedRecommendation(
        game_id=game_id,
        ai_lineup_ids=AI_LINEUP,
        ai_expected_runs=AI_EXPECTED_RUNS,
    )


# ---------------------------------------------------------------------------
# _CachedRecommendation unit tests
# ---------------------------------------------------------------------------


def test_cached_recommendation_not_expired_when_fresh():
    rec = _CachedRecommendation(game_id=GAME_ID, ai_lineup_ids=AI_LINEUP, ai_expected_runs=4.5)
    assert rec.is_expired() is False


def test_cached_recommendation_expired_after_ttl():
    rec = _CachedRecommendation(
        game_id=GAME_ID,
        ai_lineup_ids=AI_LINEUP,
        ai_expected_runs=4.5,
        cached_at=time.time() - _LINEUP_CACHE_TTL_SECONDS - 1,
    )
    assert rec.is_expired() is True


def test_cached_recommendation_not_confirmed_by_default():
    rec = _CachedRecommendation(game_id=GAME_ID, ai_lineup_ids=AI_LINEUP, ai_expected_runs=4.5)
    assert rec.already_confirmed is False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _patched_client(state: _AppState):
    """Return context manager that yields a TestClient with _state and lifespan patched."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with patch("app.main._state", state), patch("app.main._load_models_sync"):
            with TestClient(app) as c:
                yield c

    return _ctx()


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """TestClient with a fresh _AppState wired to a real ExperimentStore."""
    state = _make_state(tmp_path)
    with _patched_client(state) as c:
        yield c


@pytest.fixture
def client_with_cache(tmp_path: Path) -> TestClient:
    """TestClient that already has a cached AI recommendation for GAME_ID."""
    state = _make_state(tmp_path, cached=_fresh_cached())
    with _patched_client(state) as c:
        yield c


# ---------------------------------------------------------------------------
# 404 — no cache entry
# ---------------------------------------------------------------------------


def test_confirm_without_prior_optimize_returns_404(client: TestClient):
    resp = client.post("/v1/lineup/confirm", json={
        "game_id": GAME_ID,
        "game_date": TODAY,
        "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
    })
    assert resp.status_code == 404


def test_confirm_unknown_game_id_returns_404(client: TestClient):
    resp = client.post("/v1/lineup/confirm", json={
        "game_id": "non-existent-game",
        "game_date": TODAY,
        "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
    })
    assert resp.status_code == 404


def test_confirm_expired_cache_returns_404(tmp_path: Path):
    expired = _CachedRecommendation(
        game_id=GAME_ID,
        ai_lineup_ids=AI_LINEUP,
        ai_expected_runs=AI_EXPECTED_RUNS,
        cached_at=time.time() - _LINEUP_CACHE_TTL_SECONDS - 1,
    )
    state = _make_state(tmp_path, cached=expired)
    with _patched_client(state) as c:
        resp = c.post("/v1/lineup/confirm", json={
            "game_id": GAME_ID,
            "game_date": TODAY,
            "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
        })
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# No divergence — identical lineup
# ---------------------------------------------------------------------------


def test_confirm_identical_lineup_returns_no_divergence(client_with_cache: TestClient):
    resp = client_with_cache.post("/v1/lineup/confirm", json={
        "game_id": GAME_ID,
        "game_date": TODAY,
        "manager_lineup_ids": AI_LINEUP,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["divergence_detected"] is False
    assert body["observation_id"] is None
    assert body["slots_changed"] == 0
    assert body["players_changed"] == 0


def test_confirm_identical_lineup_does_not_create_observation(tmp_path: Path):
    state = _make_state(tmp_path, cached=_fresh_cached())
    with _patched_client(state) as c:
        c.post("/v1/lineup/confirm", json={
            "game_id": GAME_ID,
            "game_date": TODAY,
            "manager_lineup_ids": AI_LINEUP,
        })
    # No parquet partition should have been written
    partition_dir = tmp_path / "ab" / f"game_date={TODAY}"
    assert not partition_dir.exists() or not any(partition_dir.glob("*.parquet"))


# ---------------------------------------------------------------------------
# Divergence detected — observation created
# ---------------------------------------------------------------------------


def test_confirm_order_only_divergence_detected(client_with_cache: TestClient):
    resp = client_with_cache.post("/v1/lineup/confirm", json={
        "game_id": GAME_ID,
        "game_date": TODAY,
        "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["divergence_detected"] is True
    assert body["divergence_type"] == "order_only"
    assert body["slots_changed"] == 2
    assert body["players_changed"] == 0


def test_confirm_player_swap_divergence_type(tmp_path: Path):
    state = _make_state(tmp_path, cached=_fresh_cached())
    with _patched_client(state) as c:
        resp = c.post("/v1/lineup/confirm", json={
            "game_id": GAME_ID,
            "game_date": TODAY,
            "manager_lineup_ids": MANAGER_LINEUP_PLAYER_SWAP,
        })
    body = resp.json()
    assert body["divergence_type"] == "player_swap"
    assert body["players_changed"] == 1


def test_confirm_divergence_observation_id_is_deterministic(client_with_cache: TestClient):
    resp = client_with_cache.post("/v1/lineup/confirm", json={
        "game_id": GAME_ID,
        "game_date": TODAY,
        "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
    })
    assert resp.json()["observation_id"] == _obs_id(GAME_ID)


def test_confirm_divergence_returns_ai_expected_runs(client_with_cache: TestClient):
    resp = client_with_cache.post("/v1/lineup/confirm", json={
        "game_id": GAME_ID,
        "game_date": TODAY,
        "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
    })
    assert resp.json()["ai_expected_runs"] == AI_EXPECTED_RUNS


# ---------------------------------------------------------------------------
# Persistence — observation actually written to store
# ---------------------------------------------------------------------------


def test_confirm_divergence_observation_persisted(tmp_path: Path):
    state = _make_state(tmp_path, cached=_fresh_cached())
    with _patched_client(state) as c:
        c.post("/v1/lineup/confirm", json={
            "game_id": GAME_ID,
            "game_date": TODAY,
            "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
        })

    import polars as pl
    parquet = tmp_path / "ab" / f"game_date={TODAY}" / f"{_obs_id(GAME_ID)}.parquet"
    assert parquet.exists()
    df = pl.read_parquet(parquet, hive_partitioning=False)
    assert df["status"][0] == "open"
    assert df["game_id"][0] == GAME_ID


# ---------------------------------------------------------------------------
# Idempotency — double confirm does not create duplicate observation
# ---------------------------------------------------------------------------


def test_double_confirm_same_game_id_is_idempotent(tmp_path: Path):
    state = _make_state(tmp_path, cached=_fresh_cached())
    with _patched_client(state) as c:
        resp1 = c.post("/v1/lineup/confirm", json={
            "game_id": GAME_ID,
            "game_date": TODAY,
            "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
        })
        resp2 = c.post("/v1/lineup/confirm", json={
            "game_id": GAME_ID,
            "game_date": TODAY,
            "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
        })

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["observation_id"] == resp2.json()["observation_id"]

    # Only one parquet file should exist
    import polars as pl
    parquet = tmp_path / "ab" / f"game_date={TODAY}" / f"{_obs_id(GAME_ID)}.parquet"
    df = pl.read_parquet(parquet, hive_partitioning=False)
    assert len(df) == 1


def test_already_confirmed_flag_set_after_first_confirm(tmp_path: Path):
    state = _make_state(tmp_path, cached=_fresh_cached())
    with _patched_client(state) as c:
        c.post("/v1/lineup/confirm", json={
            "game_id": GAME_ID,
            "game_date": TODAY,
            "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
        })
    assert state.lineup_cache[GAME_ID].already_confirmed is True


# ---------------------------------------------------------------------------
# 503 — experiment store not initialized
# ---------------------------------------------------------------------------


def test_confirm_without_experiment_store_returns_503(tmp_path: Path):
    state = _AppState()  # experiment_store = None
    state.lineup_cache[GAME_ID] = _fresh_cached()
    with _patched_client(state) as c:
        resp = c.post("/v1/lineup/confirm", json={
            "game_id": GAME_ID,
            "game_date": TODAY,
            "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
        })
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# message_es content
# ---------------------------------------------------------------------------


def test_confirm_divergence_message_es_contains_tipo_label(client_with_cache: TestClient):
    resp = client_with_cache.post("/v1/lineup/confirm", json={
        "game_id": GAME_ID,
        "game_date": TODAY,
        "manager_lineup_ids": MANAGER_LINEUP_ORDER_ONLY,
    })
    assert "orden" in resp.json()["message_es"].lower()


def test_confirm_identical_message_es_mentions_coincide(client_with_cache: TestClient):
    resp = client_with_cache.post("/v1/lineup/confirm", json={
        "game_id": GAME_ID,
        "game_date": TODAY,
        "manager_lineup_ids": AI_LINEUP,
    })
    msg = resp.json()["message_es"].lower()
    assert "coincide" in msg or "no se registró" in msg
