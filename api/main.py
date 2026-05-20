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


# ---------------------------------------------------------------------------
# PostgreSQL + MLB Stats API helpers (dashboard read endpoints)
# ---------------------------------------------------------------------------
PG_HOST = os.getenv("POSTGRES_HOST", "")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB   = os.getenv("POSTGRES_DB", "")
PG_USER = os.getenv("POSTGRES_USER", "")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "")

_STATS_BASE = "https://statsapi.mlb.com/api/v1"
_STATS_V11  = "https://statsapi.mlb.com/api/v1.1"


def _pg_conn():
    import psycopg2
    return psycopg2.connect(
        host=PG_HOST, port=int(PG_PORT),
        dbname=PG_DB, user=PG_USER, password=PG_PASS,
        connect_timeout=5,
    )


def _stats_get(path: str, base: str = _STATS_BASE, params: dict | None = None) -> dict:
    import requests as _r
    for attempt in range(3):
        try:
            resp = _r.get(f"{base}{path}", params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"exhausted retries: {base}{path}")


def _players_bulk(player_ids: list[int]) -> dict[int, dict]:
    """Career hitting stats + primary position for multiple player IDs (one MLB API call)."""
    valid = [p for p in player_ids if p and p > 0]
    if not valid:
        return {}
    try:
        data = _stats_get(
            "/people",
            params={
                "personIds": ",".join(str(p) for p in valid),
                "hydrate": "stats(group=[hitting],type=[career])",
            },
        )
    except Exception:
        return {}

    out: dict[int, dict] = {}
    for p in data.get("people", []):
        pid = p.get("id")
        if not pid:
            continue
        name = p.get("fullName", f"Player {pid}")
        hand = p.get("batSide", {}).get("code", "R")
        pos  = p.get("primaryPosition", {}).get("abbreviation", "—")
        pitch_hand = p.get("pitchHand", {}).get("code", "R")

        cs: dict = {}
        for s in p.get("stats", []):
            if s.get("type", {}).get("displayName") == "career":
                splits = s.get("splits", [])
                if splits:
                    cs = splits[0].get("stat", {})
                    break

        avg = float(cs.get("avg", 0.248) or 0.248)
        obp = float(cs.get("obp", 0.318) or 0.318)
        slg = float(cs.get("slg", 0.398) or 0.398)
        out[pid] = {
            "player_id": pid, "name": name, "pos": pos,
            "hand": hand, "pitch_hand": pitch_hand,
            "avg":  round(avg, 3),
            "ops":  round(obp + slg, 3),
            "woba": round((obp * 1.2 + slg * 0.7) / 2, 3),
            "obp":  round(obp, 3),
            "iso":  round(max(slg - avg, 0.0), 3),
        }
    return out


# Optional auth: accepts a valid token but also allows unauthenticated calls.
_optional_bearer = HTTPBearer(auto_error=False)


def _optional_token(
    credentials: HTTPAuthorizationCredentials | None = Security(_optional_bearer),
) -> None:
    if credentials and credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido.")


# ---------------------------------------------------------------------------
# Dashboard endpoints  (Streamlit ← FastAPI ← PostgreSQL / MLB Stats API)
# ---------------------------------------------------------------------------


@app.get("/v1/games/today", tags=["dashboard"])
async def games_today(_: None = Depends(_optional_token)) -> dict:
    """Partidos MLB programados para hoy con probable pitchers."""
    import pytz
    from datetime import date as _date, datetime as _dt

    today = _date.today().isoformat()
    try:
        data = _stats_get("/schedule", params={
            "sportId": 1, "date": today,
            "hydrate": "team,venue,probablePitcher,status",
        })
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MLB API error: {exc}")

    raw_games: list[dict] = []
    pitcher_ids: list[int] = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            game_pk = g.get("gamePk")
            if not game_pk:
                continue
            home   = g.get("teams", {}).get("home", {})
            away   = g.get("teams", {}).get("away", {})
            hp     = home.get("probablePitcher", {})
            ap     = away.get("probablePitcher", {})

            game_time_et = "TBD"
            gd_str = g.get("gameDate", "")
            if gd_str:
                try:
                    dt_utc = _dt.fromisoformat(gd_str.replace("Z", "+00:00"))
                    dt_et  = dt_utc.astimezone(pytz.timezone("America/New_York"))
                    game_time_et = dt_et.strftime("%-I:%M ET")
                except Exception:
                    game_time_et = gd_str[11:16] + " UTC"

            if hp.get("id"): pitcher_ids.append(int(hp["id"]))
            if ap.get("id"): pitcher_ids.append(int(ap["id"]))

            raw_games.append({
                "game_pk":           int(game_pk),
                "home_team":         home.get("team", {}).get("abbreviation", ""),
                "away_team":         away.get("team", {}).get("abbreviation", ""),
                "home_name":         home.get("team", {}).get("name", ""),
                "away_name":         away.get("team", {}).get("name", ""),
                "game_time":         game_time_et,
                "venue":             g.get("venue", {}).get("name", ""),
                "game_date":         today,
                "home_pitcher":      hp.get("fullName", "TBD"),
                "away_pitcher":      ap.get("fullName", "TBD"),
                "_home_pitcher_id":  hp.get("id"),
                "_away_pitcher_id":  ap.get("id"),
            })

    pitcher_info = _players_bulk(pitcher_ids)

    games = []
    for rg in raw_games:
        hp_id = rg.pop("_home_pitcher_id")
        ap_id = rg.pop("_away_pitcher_id")
        rg["home_pitcher_hand"] = pitcher_info.get(hp_id, {}).get("pitch_hand", "R") if hp_id else "R"
        rg["away_pitcher_hand"] = pitcher_info.get(ap_id, {}).get("pitch_hand", "R") if ap_id else "R"
        games.append(rg)

    return {"games": games, "date": today, "total": len(games)}


@app.get("/v1/optimize/{game_pk}", tags=["dashboard"])
async def get_optimize(game_pk: int, _: None = Depends(_optional_token)) -> dict:
    """Lineup óptimo del modelo para un partido (leído de PostgreSQL, generado por Airflow)."""
    if not PG_HOST:
        raise HTTPException(status_code=503, detail="PostgreSQL no configurado.")

    try:
        conn = _pg_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT ai_recommended_lineup, ai_expected_runs, win_probability,
                   optimization_mode, model_version, predicted_at,
                   home_batting_order, home_team_name, away_team_name, lineup_source
            FROM gameday_predictions
            WHERE game_pk = %s
        """, (game_pk,))
        row = cur.fetchone()
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB error: {exc}")

    if not row:
        raise HTTPException(status_code=404, detail=f"Sin predicción para game_pk={game_pk}")

    (ai_lineup, ai_er, win_prob, opt_mode, model_ver, predicted_at,
     home_order, home_name, away_name, lineup_source) = row

    ordered_ids  = (ai_lineup or home_order or [])[:9]
    all_ids      = list(dict.fromkeys(ordered_ids + (home_order or [])[:12]))
    player_info  = _players_bulk(all_ids)

    lineup = []
    for i, pid in enumerate(ordered_ids, 1):
        p = player_info.get(pid, {"player_id": pid, "name": f"Player {pid}", "pos": "—",
                                  "hand": "R", "avg": .248, "ops": .716,
                                  "woba": .315, "obp": .318, "iso": .150})
        lineup.append({"order": i, **{k: v for k, v in p.items() if k != "pitch_hand"}})

    bench_ids = [pid for pid in (home_order or []) if pid not in set(ordered_ids)][:3]
    bench = []
    for pid in bench_ids:
        p = player_info.get(pid, {"player_id": pid, "name": f"Player {pid}", "pos": "—",
                                  "hand": "R", "avg": .248, "ops": .716,
                                  "woba": .315, "obp": .318, "iso": .150})
        bench.append({k: v for k, v in p.items() if k != "pitch_hand"})

    rag = (
        f"## Lineup Óptimo — {away_name} @ {home_name}\n\n"
        f"Lineup generado por el motor Monte Carlo (Airflow). Fuente: `{lineup_source}`.\n\n"
        f"E[R] proyectado: **{ai_er:.2f}** · P(Victoria): **{(win_prob or 0.5)*100:.1f}%**\n\n"
        f"*(Explicación RAG completa disponible en los logs de Airflow del DAG "
        f"`mlb_gameday_orchestrator`.)*"
    )

    return {
        "game_pk":           game_pk,
        "expected_runs":     round(ai_er or 0.0, 3),
        "win_probability":   round(win_prob or 0.5, 4),
        "model_confidence":  0.80,
        "optimization_mode": opt_mode or "fast",
        "model_version":     model_ver or "unknown",
        "total_simulations": 10_000,
        "elapsed_seconds":   0.0,
        "lineup":            lineup,
        "bench":             bench,
        "rag_explanation":   rag,
    }


@app.get("/v1/games/history", tags=["dashboard"])
async def games_history(date: str, _: None = Depends(_optional_token)) -> dict:
    """Partidos históricos con resultados reales para una fecha dada."""
    rows: list = []
    if PG_HOST:
        try:
            conn = _pg_conn()
            cur  = conn.cursor()
            cur.execute("""
                SELECT game_pk, home_team_name, away_team_name,
                       actual_home_runs, actual_away_runs,
                       home_starting_pitcher_id, away_starting_pitcher_id
                FROM gameday_predictions
                WHERE game_date = %s
                ORDER BY game_pk
            """, (date,))
            rows = cur.fetchall()
            conn.close()
        except Exception as exc:
            log.warning("games_history DB error: %s", exc)

    games: list[dict] = []

    if rows:
        pitcher_ids = []
        for r in rows:
            if r[5]: pitcher_ids.append(r[5])
            if r[6]: pitcher_ids.append(r[6])
        pitcher_info = _players_bulk(pitcher_ids)

        for row in rows:
            gpk, home_name, away_name, home_runs, away_runs, hp_id, ap_id = row
            final = (
                f"{home_runs}-{away_runs}"
                if home_runs is not None and away_runs is not None
                else "N/A"
            )
            games.append({
                "game_pk":         gpk,
                "home_name":       home_name or "",
                "away_name":       away_name or "",
                "home_team":       "",
                "away_team":       "",
                "game_date":       date,
                "final_score":     final,
                "home_runs_actual": home_runs or 0,
                "away_runs_actual": away_runs or 0,
                "home_pitcher":    pitcher_info.get(hp_id, {}).get("name", "") if hp_id else "",
                "away_pitcher":    pitcher_info.get(ap_id, {}).get("name", "") if ap_id else "",
            })
    else:
        # Fallback: query MLB Stats API (linescore hydration)
        try:
            data = _stats_get("/schedule", params={
                "sportId": 1, "date": date,
                "hydrate": "team,linescore,status",
            })
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"MLB API error: {exc}")

        for date_entry in data.get("dates", []):
            for g in date_entry.get("games", []):
                gp = g.get("gamePk")
                if not gp:
                    continue
                home = g.get("teams", {}).get("home", {}).get("team", {})
                away = g.get("teams", {}).get("away", {}).get("team", {})
                ls   = g.get("linescore", {}).get("teams", {})
                hr   = ls.get("home", {}).get("runs", 0) or 0
                ar   = ls.get("away", {}).get("runs", 0) or 0
                st   = g.get("status", {}).get("abstractGameState", "")
                games.append({
                    "game_pk":          int(gp),
                    "home_name":        home.get("name", ""),
                    "away_name":        away.get("name", ""),
                    "home_team":        home.get("abbreviation", ""),
                    "away_team":        away.get("abbreviation", ""),
                    "game_date":        date,
                    "final_score":      f"{hr}-{ar}" if st == "Final" else "N/A",
                    "home_runs_actual": hr,
                    "away_runs_actual": ar,
                    "home_pitcher":     "",
                    "away_pitcher":     "",
                })

    return {"games": games, "date": date, "total": len(games)}


@app.get("/v1/report/{game_pk}", tags=["dashboard"])
async def get_report(game_pk: int, date: str | None = None,
                     _: None = Depends(_optional_token)) -> dict:
    """Reporte post-partido completo: lineups, E[R] delta, log-loss y análisis."""
    import math

    if not PG_HOST:
        raise HTTPException(status_code=503, detail="PostgreSQL no configurado.")

    try:
        conn = _pg_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT game_pk, game_date, home_team_name, away_team_name,
                   ai_recommended_lineup, home_batting_order,
                   ai_expected_runs, win_probability, model_version,
                   actual_home_runs, actual_away_runs, delta_er, game_state
            FROM gameday_predictions
            WHERE game_pk = %s
        """, (game_pk,))
        row = cur.fetchone()
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB error: {exc}")

    if not row:
        raise HTTPException(status_code=404, detail=f"Sin datos para game_pk={game_pk}")

    (gpk, game_date, home_name, away_name,
     ai_lineup, home_order,
     ai_er, win_prob, model_ver,
     actual_home_runs, actual_away_runs, _delta_er, game_state) = row

    # Fetch all player names in one call
    all_ids = list(dict.fromkeys((ai_lineup or []) + (home_order or [])))
    player_info = _players_bulk([p for p in all_ids if p])

    # Proposed lineup (model's AI recommendation)
    proposed_lineup = []
    for i, pid in enumerate((ai_lineup or [])[:9], 1):
        p = player_info.get(pid, {})
        proposed_lineup.append({
            "order": i,
            "name":  p.get("name", f"Player {pid}"),
            "pos":   p.get("pos", "—"),
        })

    # Actual lineup: try to get results from live-feed boxscore
    actual_lineup = []
    try:
        feed  = _stats_get(f"/game/{game_pk}/feed/live", base=_STATS_V11)
        teams = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
        home  = teams.get("home", {})
        bs_order   = home.get("battingOrder", []) or home_order or []
        bs_players = home.get("players", {})

        for i, pid in enumerate(bs_order[:9], 1):
            key  = f"ID{pid}"
            pbox = bs_players.get(key, {})
            st   = pbox.get("stats", {}).get("batting", {})
            h, ab = st.get("hits", 0) or 0, st.get("atBats", 0) or 0
            hr, bb = st.get("homeRuns", 0) or 0, st.get("baseOnBalls", 0) or 0
            rbi, so = st.get("rbi", 0) or 0, st.get("strikeOuts", 0) or 0

            parts = [f"{h}-for-{ab}"]
            if hr:              parts.append(f"{hr} HR")
            elif rbi:           parts.append(f"{rbi} RBI")
            if bb:              parts.append(f"{bb} BB")
            if so and not h:    parts.append(f"{so}K")
            result_str = ", ".join(parts)

            actual_lineup.append({
                "order":  i,
                "name":   pbox.get("person", {}).get("fullName", player_info.get(pid, {}).get("name", f"Player {pid}")),
                "pos":    pbox.get("position", {}).get("abbreviation", "—"),
                "result": result_str,
            })
    except Exception:
        for i, pid in enumerate((home_order or [])[:9], 1):
            p = player_info.get(pid, {})
            actual_lineup.append({
                "order":  i,
                "name":   p.get("name", f"Player {pid}"),
                "pos":    p.get("pos", "—"),
                "result": "—",
            })

    # Metrics
    hr_actual  = actual_home_runs or 0
    ar_actual  = actual_away_runs or 0
    game_result = hr_actual > ar_actual
    prob        = max(0.001, min(0.999, win_prob or 0.5))
    log_loss    = round(-math.log(prob if game_result else (1.0 - prob)), 3)
    delta       = hr_actual - (ai_er or 0.0)
    sign        = "+" if delta >= 0 else ""

    matchup = f"{away_name} @ {home_name}  ·  {hr_actual}–{ar_actual}"

    report_md = (
        f"## Post-Game Analysis — {home_name} {hr_actual}, {away_name} {ar_actual}\n"
        f"**Fecha:** {game_date} &nbsp;|&nbsp; "
        f"**Modelo:** {model_ver or 'unknown'} &nbsp;|&nbsp; "
        f"**Log-Loss:** {log_loss:.3f}\n\n"
        f"---\n\n"
        f"### Evaluación del Modelo\n\n"
        f"| Métrica | Valor |\n"
        f"|---|---|\n"
        f"| E[R] proyectado | {ai_er or 0:.2f} |\n"
        f"| Carreras reales | **{hr_actual}** |\n"
        f"| Δ E[R] | {sign}{delta:.2f} {'✅' if delta >= 0 else '⚠️'} |\n"
        f"| Win Prob. proyectada | {prob*100:.1f}% |\n"
        f"| Log-Loss del partido | {log_loss:.3f} {'✅ bueno' if log_loss < 0.5 else '⚠️ revisar'} |\n\n"
        f"### Estado del partido\n"
        f"**{game_state or 'N/A'}** — "
        f"{'✅ Victoria' if game_result else '❌ Derrota'} del equipo local.\n"
    )

    return {
        "game_pk":                  gpk,
        "game_date":                str(game_date),
        "matchup":                  matchup,
        "game_result":              game_result,
        "proposed_lineup":          proposed_lineup,
        "actual_lineup":            actual_lineup,
        "projected_runs":           round(ai_er or 0.0, 2),
        "actual_home_runs":         hr_actual,
        "actual_away_runs":         ar_actual,
        "win_probability_projected": round(prob, 4),
        "model_log_loss":           log_loss,
        "model_version":            model_ver or "unknown",
        "report_markdown":          report_md,
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False, workers=MAX_WORKERS)
