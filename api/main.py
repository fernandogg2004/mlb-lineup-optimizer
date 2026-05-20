"""
api/main.py
===========
FastAPI — Serving de inferencia para el modelo MLB AtBatPredictor.

Endpoints:
  POST /v1/predict/game/{game_pk}   → predicción de un partido
  POST /v1/predict/all              → todos los partidos de una fecha
  GET  /v1/predict/history          → consulta RDS de predicciones pasadas
  GET  /health                      → healthcheck (usado por ECS/ALB)
  GET  /metrics                     → Prometheus scrape endpoint

Seguridad: Bearer token via cabecera Authorization.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import polars as pl
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from api.metrics import (
    INFERENCE_ERRORS,
    INFERENCE_LATENCY,
    REQUEST_COUNTER,
    record_prediction_distribution,
)
from api.shadow import ShadowPredictor

log = logging.getLogger("mlb_api")

# ---------------------------------------------------------------------------
# Configuración de entorno
# ---------------------------------------------------------------------------
API_TOKEN        = os.environ["MLB_API_TOKEN"]          # requerido — falla si no existe
MODEL_PATH       = os.getenv("MODEL_PATH", "models/at_bat_predictor.pkl")
SHADOW_MODEL_PATH = os.getenv("SHADOW_MODEL_PATH", "")  # vacío = shadow desactivado
SILVER_DIR       = os.getenv("SILVER_DIR", "data/silver/plate_appearances")
MAX_WORKERS      = int(os.getenv("MAX_WORKERS", "4"))
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# ---------------------------------------------------------------------------
# Estado global (cargado una sola vez en startup)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga modelo y Silver una sola vez al arrancar el servidor."""
    import pickle
    import sys
    from pathlib import Path

    log.info("Cargando modelo desde %s...", MODEL_PATH)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import src.models.model_at_bat as _mat
    for _n in dir(_mat):
        setattr(sys.modules["__main__"], _n, getattr(_mat, _n))

    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    predictor = _mat.AtBatPredictor(config=payload["config"])
    predictor._calibrated_model = payload["calibrated_model"]
    predictor._base_model       = payload["base_model"]
    predictor._feature_names    = payload["feature_names"]
    predictor._is_fitted        = True
    _state["predictor"]         = predictor
    _state["feature_names"]     = payload["feature_names"]
    _state["model_version"]     = getattr(payload.get("config"), "mlflow_experiment", "unknown")

    if SHADOW_MODEL_PATH:
        _state["shadow"] = ShadowPredictor(SHADOW_MODEL_PATH)
        log.info("Shadow model cargado desde %s.", SHADOW_MODEL_PATH)

    log.info("Cargando Silver histórico desde %s...", SILVER_DIR)
    _state["silver"] = _load_silver(SILVER_DIR)
    log.info("Silver listo: %d PAs.", len(_state["silver"]))

    log.info("API lista.")
    yield
    # Cleanup
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MLB AI Inference API",
    version="3.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Middleware: latencia y contadores Prometheus
# ---------------------------------------------------------------------------
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    INFERENCE_LATENCY.labels(endpoint=path).observe(elapsed)
    REQUEST_COUNTER.labels(endpoint=path, status=response.status_code).inc()
    return response


# ---------------------------------------------------------------------------
# Autenticación
# ---------------------------------------------------------------------------
_bearer = HTTPBearer(auto_error=True)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(_bearer)) -> str:
    if credentials.credentials != API_TOKEN:
        INFERENCE_ERRORS.labels(error_type="auth_failure").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------
class BatterPrediction(BaseModel):
    slot:      int
    id:        int
    name:      str
    ev_per_pa: float
    prob_out:  float
    prob_k:    float
    prob_bb:   float
    prob_1b:   float
    prob_2b:   float
    prob_3b:   float
    prob_hr:   float
    woba_stab: float
    obp_est:   float


class TeamPrediction(BaseModel):
    game_date:              str
    game_pk:                int
    matchup:                str
    team:                   str
    team_abbr:              str
    side:                   str
    opp_pitcher:            str
    batting_order:          list[BatterPrediction]
    expected_runs_per_game: float
    model_version:          str


class AllGamesRequest(BaseModel):
    game_date:    str
    gold_s3_key:  str | None = None   # opcional — si se omite usa Silver local


class AllGamesResponse(BaseModel):
    game_date:     str
    teams:         list[TeamPrediction]
    model_version: str
    total_teams:   int


class HealthResponse(BaseModel):
    status:        str
    model_version: str
    silver_rows:   int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return {
        "status":        "ok",
        "model_version": _state.get("model_version", "unknown"),
        "silver_rows":   len(_state.get("silver", [])),
    }


@app.get("/metrics", tags=["ops"])
async def metrics():
    """Endpoint para scraping de Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/v1/predict/game/{game_pk}",
    response_model=list[TeamPrediction],
    tags=["inference"],
)
async def predict_game(
    game_pk:    int,
    game_date:  str,
    _token:     str = Depends(verify_token),
) -> list[TeamPrediction]:
    """Predicción para ambos equipos de un partido específico."""
    import requests as _req
    from predict_tonight import _predict_one_side

    try:
        resp = _req.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "gamePk": game_pk, "hydrate": "lineups,team,probablePitcher"},
            timeout=10,
        )
        resp.raise_for_status()
        games = [g for d in resp.json().get("dates", []) for g in d.get("games", [])]
        if not games:
            raise HTTPException(status_code=404, detail=f"gamePk {game_pk} no encontrado.")
        game = games[0]
    except HTTPException:
        raise
    except Exception as exc:
        INFERENCE_ERRORS.labels(error_type="mlb_api_error").inc()
        raise HTTPException(status_code=502, detail=f"MLB API error: {exc}")

    results = []
    for side in ("away", "home"):
        try:
            r = _predict_one_side(
                game, side, _state["silver"], _state["predictor"], game_date, verbose=False
            )
            if r:
                r["model_version"] = _state["model_version"]
                results.append(r)
                record_prediction_distribution(r["batting_order"])
        except Exception as exc:
            INFERENCE_ERRORS.labels(error_type="inference_error").inc()
            log.exception("Inferencia falló para gamePk=%d side=%s: %s", game_pk, side, exc)

    if not results:
        raise HTTPException(status_code=422, detail="No se pudo procesar ningún equipo.")
    return results


@app.post(
    "/v1/predict/all",
    response_model=AllGamesResponse,
    tags=["inference"],
)
async def predict_all(
    req:    AllGamesRequest,
    _token: str = Depends(verify_token),
) -> AllGamesResponse:
    """Predicción para todos los partidos de una fecha. Usado por el DAG de Airflow."""
    from predict_tonight import fetch_games, _predict_one_side

    try:
        games = fetch_games(req.game_date)
    except Exception as exc:
        INFERENCE_ERRORS.labels(error_type="mlb_api_error").inc()
        raise HTTPException(status_code=502, detail=f"MLB API error: {exc}")

    teams: list[dict] = []
    for game in games:
        for side in ("away", "home"):
            try:
                r = _predict_one_side(
                    game, side, _state["silver"], _state["predictor"],
                    req.game_date, verbose=False,
                )
                if r:
                    r["model_version"] = _state["model_version"]
                    teams.append(r)
                    record_prediction_distribution(r["batting_order"])

                    # Shadow mode: corre challenger en paralelo si está configurado
                    if "shadow" in _state:
                        _state["shadow"].run_async(game, side, _state["silver"], req.game_date)

            except Exception as exc:
                INFERENCE_ERRORS.labels(error_type="inference_error").inc()
                log.exception("Error en gamePk=%s side=%s: %s", game.get("gamePk"), side, exc)

    return {
        "game_date":     req.game_date,
        "teams":         teams,
        "model_version": _state.get("model_version", "unknown"),
        "total_teams":   len(teams),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_silver(silver_dir: str) -> pl.DataFrame:
    from pathlib import Path
    parts = []
    base = Path(silver_dir)
    if base.exists():
        for sd in sorted(base.glob("season=*")):
            pq = sd / "data.parquet"
            if pq.exists():
                parts.append(pl.read_parquet(pq, hive_partitioning=False))
    if not parts:
        log.warning("No Silver parquets encontrados en %s. Usando DataFrame vacío.", silver_dir)
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed").sort(["batter_id", "game_date"])


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False, workers=MAX_WORKERS)
