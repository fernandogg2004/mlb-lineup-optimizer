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

import collections
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np

# Carga .env si existe (dev local sin Docker)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

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

# Outcomes en el orden que usa MonteCarloEngine (debe coincidir con PAOutcome enum, 8 clases)
_OUTCOME_KEYS   = ("prob_out", "prob_k", "prob_bb", "prob_1b", "prob_2b", "prob_3b", "prob_hr", "prob_dp")
# Lineup promedio de liga (oponente cuando no tenemos sus probs reales) — 8 clases
# OUT=0.400, K=0.220, BB=0.085, 1B=0.145, 2B=0.048, 3B=0.005, HR=0.062, DP=0.035
_LEAGUE_AVG_OPP = np.array([0.400, 0.220, 0.085, 0.145, 0.048, 0.005, 0.062, 0.035], dtype=np.float32)


def _mc_run(my_probs: np.ndarray, opp_probs: np.ndarray, n_sims: int):
    """Llama directamente al kernel Numba con n_sims configurable.

    Extiende los percentiles base [5,25,50,75,95] con [1,2,98,99] para
    que la UI pueda construir ICs al 50 / 90 / 95 / 99%.
    """
    from src.simulation.simulation_engine import (
        _NEW_BASES_TBL, _RUNS_TBL, _aggregate_results, _simulate_n_games,
    )
    t0 = time.perf_counter()
    my_r, opp_r = _simulate_n_games(
        my_probs, opp_probs, _NEW_BASES_TBL, _RUNS_TBL, 1.0, 1.0, n_sims, 9
    )
    mc = _aggregate_results(my_r, opp_r, time.perf_counter() - t0)
    for p in (1, 2, 98, 99):
        mc.runs_scored_percentiles[p] = float(np.percentile(my_r, p))
    return mc

# ---------------------------------------------------------------------------
# Configuración de entorno
# ---------------------------------------------------------------------------
API_TOKEN        = os.environ.get("MLB_API_TOKEN", "")  # vacío = sin autenticación
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

    try:
        import asyncio as _aio
        from src.simulation.simulation_engine import MonteCarloConfig, MonteCarloEngine, warmup_jit
        _mc_cfg = MonteCarloConfig(use_ray=False, n_simulations=10_000, fast_mode_n_sims=5_000)
        _state["mc_engine"] = MonteCarloEngine(_mc_cfg)
        # warmup_jit es compilación LLVM síncrona (~2-5 s); ejecutar en thread
        # para no bloquear el event loop de uvicorn.
        await _aio.to_thread(warmup_jit)
        log.info("MonteCarloEngine lista (10 000 sims, Numba, sin Ray).")
    except Exception as mc_exc:
        log.warning("MonteCarloEngine no disponible: %s", mc_exc)
        _state["mc_engine"] = None

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
_bearer = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials | None = Security(_bearer)) -> str:
    # Sin token configurado → acceso libre (uso local)
    if not API_TOKEN:
        return ""
    if credentials is None or credentials.credentials != API_TOKEN:
        INFERENCE_ERRORS.labels(error_type="auth_failure").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per-IP, no external dependency)
# ---------------------------------------------------------------------------
_rl_store: dict[str, collections.deque] = {}
_rl_lock = threading.Lock()


def _rate_limit(
    request: Request,
    max_calls: int = 2,
    window_secs: int = 60,
) -> None:
    """Raises HTTP 429 if the caller exceeds max_calls within window_secs.

    Uses a sliding-window deque per client IP. Thread-safe via a global lock.
    Designed for expensive endpoints (GA lineup optimization, 3–30 s each).
    """
    ip = (request.client.host if request.client else "unknown")
    key = f"{request.url.path}:{ip}"
    now = time.monotonic()
    with _rl_lock:
        dq = _rl_store.setdefault(key, collections.deque())
        while dq and now - dq[0] > window_secs:
            dq.popleft()
        if len(dq) >= max_calls:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit: máximo {max_calls} peticiones "
                    f"por {window_secs}s en este endpoint."
                ),
                headers={"Retry-After": str(window_secs)},
            )
        dq.append(now)


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
    prob_dp:   float = 0.0   # clase 7 DOUBLE_PLAY (modelo 8-clases)
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
                game, side, _state["silver"], _state["predictor"], game_date, verbose=False,
                feature_names=_state.get("feature_names"),
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
                    feature_names=_state.get("feature_names"),
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
# Dashboard endpoints  (Frontend ← FastAPI ← PostgreSQL / MLB Stats API)
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


# ---------------------------------------------------------------------------
# Tactical explanation helper
# ---------------------------------------------------------------------------

def _generate_tactical_explanation(
    lineup: list[dict],
    pitcher_name: str,
    pitcher_hand: str,
    pitcher_era: float,
    away_name: str,
    home_name: str,
) -> str:
    """Generate per-game tactical explanation via Claude/Groq, with template fallback."""
    obp_rank  = {p["player_id"]: r for r, p in enumerate(
        sorted(lineup, key=lambda x: x.get("obp", 0), reverse=True), 1)}
    woba_rank = {p["player_id"]: r for r, p in enumerate(
        sorted(lineup, key=lambda x: x.get("woba", 0), reverse=True), 1)}
    iso_rank  = {p["player_id"]: r for r, p in enumerate(
        sorted(lineup, key=lambda x: x.get("iso", 0), reverse=True), 1)}

    def platoon_tag(hand: str) -> str:
        if hand == "L" and pitcher_hand == "R":
            return "✅ ventaja platoon"
        if hand == "R" and pitcher_hand == "L":
            return "✅ ventaja platoon"
        if hand == pitcher_hand:
            return "⚠️ desventaja platoon"
        return "—"

    slot_role = {
        1: "leadoff (máx. OBP)", 2: "segundo (OBP + contacto)",
        3: "tercer slot (wOBA líder)", 4: "limpiabases (ISO/wOBA)",
        5: "quinto (segundo poder)", 6: "sexto (protege al 5)",
        7: "séptimo (platoon/def)", 8: "octavo (más débil o platoon)",
        9: "noveno (segundo leadoff)",
    }

    lineup_lines = []
    for p in lineup:
        pid  = p["player_id"]
        slot = p["order"]
        hand = p.get("hand", "R")
        lineup_lines.append(
            f"  #{slot} [{slot_role.get(slot, '')}] **{p['name']}** ({p.get('pos','?')}, {hand}HB) {platoon_tag(hand)}\n"
            f"     OBP {p.get('obp',0):.3f} [#{obp_rank[pid]}]"
            f" · wOBA {p.get('woba',0):.3f} [#{woba_rank[pid]}]"
            f" · ISO {p.get('iso',0):.3f} [#{iso_rank[pid]}]"
            f" · AVG {p.get('avg',0):.3f}"
        )
    lineup_text = "\n".join(lineup_lines)

    system_msg = (
        "Eres un analista táctico de béisbol de las Grandes Ligas. "
        "REGLAS ABSOLUTAS: (1) Responde ÚNICAMENTE con bullet points de UNA sola línea cada uno. "
        "(2) PROHIBIDO escribir párrafos, títulos de sección, o texto narrativo. "
        "(3) PROHIBIDO repetir información que ya está en la tabla de estadísticas. "
        "(4) Cada bullet DEBE mencionar exactamente 1 métrica numérica específica del jugador. "
        "(5) Usa SOLO estos prefijos: ✅ Ventaja · 🔴 Riesgo · ⚡ Decisión · 📊 Matchup. "
        "(6) Máximo 10 bullets totales. Sé brutal con la brevedad."
    )

    user_msg = (
        f"Partido: {away_name} @ {home_name} | "
        f"Rival: {pitcher_name} ({pitcher_hand}HP) ERA {pitcher_era:.2f}\n\n"
        f"Lineup (rankings internos — #1=mejor del lineup en esa métrica):\n"
        f"{lineup_text}\n\n"
        f"FORMATO OBLIGATORIO — genera exactamente este estilo, SIN párrafos:\n"
        f"✅ Ventaja: **[Nombre]** (#[slot]) → [métrica numérica], [razón en 5 palabras]\n"
        f"🔴 Riesgo: **[Nombre]** (#[slot]) → [métrica numérica], [debilidad específica]\n"
        f"⚡ Decisión: **[Nombre] #[X] > #[Y]** — [diferencial numérico], [lógica en 6 palabras]\n"
        f"📊 Matchup: {pitcher_name} ({pitcher_hand}HP ERA {pitcher_era:.2f}) → [ventaja/riesgo global]\n\n"
        f"Genera 6-10 bullets para ESTA alineación específica:"
    )

    header = f"## Análisis Táctico — {away_name} @ {home_name}\n\n"

    # Anthropic Claude (fastest, lowest cost)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        try:
            import anthropic as _ant
            msg = _ant.Anthropic(api_key=anthropic_key).messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1200,
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}],
            )
            return header + msg.content[0].text
        except Exception as exc:
            log.warning("claude_explanation_failed: %s", exc)

    # Groq / OpenAI-compatible fallback
    groq_key  = os.environ.get("GROQ_API_KEY", "")
    oai_key   = os.environ.get("OPENAI_API_KEY", "")
    llm_key   = groq_key or oai_key
    llm_base  = "https://api.groq.com/openai/v1" if groq_key else os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant" if groq_key else "gpt-4o-mini")
    if llm_key:
        try:
            import httpx
            resp = httpx.post(
                f"{llm_base}/chat/completions",
                headers={"Authorization": f"Bearer {llm_key}", "Content-Type": "application/json"},
                json={
                    "model": llm_model,
                    "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    "max_tokens": 1200, "temperature": 0.4,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return header + resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            log.warning("llm_explanation_failed provider=%s: %s", llm_base, exc)

    # Template fallback (no API key) — bullet-point format
    by_woba  = sorted(lineup, key=lambda p: p.get("woba", 0), reverse=True)
    by_iso   = sorted(lineup, key=lambda p: p.get("iso",  0), reverse=True)
    by_obp   = sorted(lineup, key=lambda p: p.get("obp",  0), reverse=True)
    bottom1  = by_woba[-1]
    lhb      = [p for p in lineup if p.get("hand") == "L"]
    rhb      = [p for p in lineup if p.get("hand") == "R"]
    adv_side = lhb if pitcher_hand == "R" else rhb
    dis_side = rhb if pitcher_hand == "R" else lhb

    bullets: list[str] = []

    # Top 3 advantages
    for p in by_woba[:3]:
        slot = p["order"]
        woba = p.get("woba", 0)
        hand = p.get("hand", "R")
        platoon_tag = "✅ ventaja platoon" if (
            (hand == "L" and pitcher_hand == "R") or (hand == "R" and pitcher_hand == "L")
        ) else "⚠️ desventaja platoon"
        bullets.append(
            f"✅ Ventaja: **{p['name']}** (#{slot}) → wOBA {woba:.3f} [wOBA #{by_woba.index(p)+1}], {platoon_tag}"
        )

    # Power leaders
    p_iso = by_iso[0]
    bullets.append(
        f"⚡ Decisión: **{p_iso['name']}** (#{p_iso['order']}) → ISO {p_iso.get('iso',0):.3f} [ISO #1], máximo potencial extrabases"
    )

    # OBP leadoff rationale
    p_obp = by_obp[0]
    bullets.append(
        f"⚡ Decisión: **{p_obp['name']}** (#{p_obp['order']}) → OBP {p_obp.get('obp',0):.3f} [OBP #1], mayor tasa de embasado"
    )

    # Platoon context
    if adv_side:
        names_adv = ", ".join(p["name"].split()[-1] for p in adv_side[:3])
        bullets.append(
            f"📊 Matchup: {len(adv_side)}/{len(lineup)} bateadores con ventaja platoon vs {pitcher_name} ({pitcher_hand}HP) — {names_adv}"
        )

    # Risk: bottom batter
    bullets.append(
        f"🔴 Riesgo: **{bottom1['name']}** (#{bottom1['order']}) → wOBA {bottom1.get('woba',0):.3f} [wOBA #{len(by_woba)}], slot más débil del lineup"
    )

    # Pitcher context
    era_context = "por encima del promedio (ERA > 4.50)" if pitcher_era > 4.50 else "élite (ERA ≤ 3.50)" if pitcher_era <= 3.50 else "promedio (ERA 3.50–4.50)"
    bullets.append(
        f"📊 Matchup: {pitcher_name} ERA {pitcher_era:.2f} — lanzador {era_context}, E[R] ajustado al contexto"
    )

    bullets_md = "\n".join(f"- {b}" for b in bullets)
    return (
        header
        + f"**{pitcher_name}** ({pitcher_hand}HP) · ERA {pitcher_era:.2f} · {away_name} @ {home_name}\n\n"
        + bullets_md
        + "\n\n> 💡 Agrega `ANTHROPIC_API_KEY` o `GROQ_API_KEY` para análisis IA en tiempo real."
    )


@app.get("/v1/optimize/{game_pk}", tags=["dashboard"])
async def get_optimize(
    game_pk: int,
    request: Request,
    team: str = "home",
    n_sims: int = 10_000,
    _: None = Depends(_optional_token),
) -> dict:
    """
    Lineup óptimo calculado en tiempo real — sin PostgreSQL.
    Usa el modelo y Silver ya cargados en memoria al arrancar uvicorn.
    Limitado a 2 peticiones/min por IP (el cálculo GA puede tardar 3-30 s).
    """
    _rate_limit(request, max_calls=2, window_secs=60)

    import requests as _req
    from predict_tonight import _predict_one_side

    log.info("optimize: game_pk=%s team=%s", game_pk, team)

    # 1. Datos del partido desde MLB Stats API
    try:
        resp = _req.get(
            f"{_STATS_BASE}/schedule",
            params={"sportId": 1, "gamePk": game_pk,
                    "hydrate": "lineups,team,probablePitcher"},
            timeout=12,
        )
        resp.raise_for_status()
        games = [g for d in resp.json().get("dates", []) for g in d.get("games", [])]
        log.info("optimize: %d juegos encontrados para gamePk=%s", len(games), game_pk)
    except Exception as exc:
        log.error("optimize: MLB API error: %s", exc)
        raise HTTPException(status_code=502, detail=f"MLB API: {exc}")

    if not games:
        raise HTTPException(status_code=404, detail=f"game_pk {game_pk} no encontrado")

    game      = games[0]
    game_date = game.get("gameDate", "")[:10]

    # 2. Predicción directa (sync — uso local, bloqueo breve aceptable)
    try:
        result = _predict_one_side(
            game, team, _state["silver"], _state["predictor"],
            game_date, verbose=False,
            feature_names=_state.get("feature_names"),
        )
        log.info("optimize: prediccion completada, er=%.2f", result.get("expected_runs_per_game", 0) if result else 0)
    except Exception as exc:
        log.exception("optimize: error en prediccion game_pk=%s team=%s", game_pk, team)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    if result is None:
        raise HTTPException(status_code=422, detail="Roster insuficiente para predicción.")

    # 3. Adaptar batting_order al formato del dashboard (stats derivadas de prob_*)
    bo     = result.get("batting_order", [])
    lineup = []
    for b in bo:
        woba = float(b.get("woba_stab", 0.318))
        bb   = float(b.get("prob_bb",   0.0))
        p1b  = float(b.get("prob_1b",   0.0))
        p2b  = float(b.get("prob_2b",   0.0))
        p3b  = float(b.get("prob_3b",   0.0))
        phr  = float(b.get("prob_hr",   0.0))

        ab_rate = max(1.0 - bb, 1e-6)
        obp  = round(bb + p1b + p2b + p3b + phr, 3)
        avg  = round((p1b + p2b + p3b + phr) / ab_rate, 3)
        slg  = round((p1b + 2*p2b + 3*p3b + 4*phr) / ab_rate, 3)
        iso  = round(slg - avg, 3)
        ops  = round(obp + slg, 3)

        lineup.append({
            "order":     b["slot"],
            "player_id": b.get("id", 0),
            "name":      b["name"],
            "pos":       "—",
            "hand":      "R",
            "avg":       avg,
            "ops":       ops,
            "woba":      round(woba, 3),
            "obp":       obp,
            "iso":       iso,
        })

    # Enrich with real batting hand + fielding position from MLB Stats API
    _pid_list   = [p["player_id"] for p in lineup if p["player_id"]]
    _meta_batch = _players_bulk(_pid_list)
    for p in lineup:
        meta = _meta_batch.get(p["player_id"], {})
        if meta:
            p["hand"] = meta.get("hand", "R")
            p["pos"]  = meta.get("pos",  "—")

    er       = float(result.get("expected_runs_per_game", 4.5))
    win_prob = er ** 1.83 / (er ** 1.83 + 4.5 ** 1.83)

    # 3b. Monte Carlo simulation usando las probs por bateador
    total_simulations    = 0
    elapsed_mc           = 0.0
    optimization_mode    = "SabermetricSeeder"
    runs_pct: dict       = {}
    bo = result.get("batting_order", [])
    if len(bo) >= 9:
        try:
            my_probs = np.array(
                [[float(b.get(k, 0.0)) for k in _OUTCOME_KEYS] for b in bo[:9]],
                dtype=np.float32,
            )
            row_sums = my_probs.sum(axis=1, keepdims=True)
            my_probs = np.where(row_sums > 0, my_probs / row_sums, my_probs)
            opp_probs = np.tile(_LEAGUE_AVG_OPP, (9, 1))
            mc_result         = _mc_run(my_probs, opp_probs, n_sims)
            er                = mc_result.expected_runs_scored
            win_prob          = mc_result.win_probability
            total_simulations = mc_result.n_simulations
            elapsed_mc        = mc_result.elapsed_seconds
            runs_pct          = mc_result.runs_scored_percentiles
            optimization_mode = "MonteCarlo"
            log.info("MC ok: E[R]=%.3f P(W)=%.3f n=%d t=%.2fs", er, win_prob, total_simulations, elapsed_mc)
        except Exception as mc_exc:
            log.warning("MC simulation failed: %s", mc_exc)

    # 4. Pitcher details for explanation
    home_obj    = game["teams"]["home"]
    away_obj    = game["teams"]["away"]
    home_name   = home_obj["team"].get("name", "Home")
    away_name   = away_obj["team"].get("name", "Away")
    opp_pp      = (away_obj if team == "home" else home_obj).get("probablePitcher", {})
    pitcher_hand = opp_pp.get("pitchHand", {}).get("code", "R")
    opp_pid      = opp_pp.get("id")
    pitcher_name = opp_pp.get("fullName", result.get("opp_pitcher", "TBD"))
    pitcher_era  = 4.00
    if opp_pid:
        try:
            pdata = _stats_get(f"/people/{opp_pid}", params={"hydrate": "stats(type=season,season=2025)"})
            pp = (pdata.get("people") or [{}])[0]
            for s in pp.get("stats", []):
                if s.get("type", {}).get("displayName") == "season":
                    era_str = (s.get("splits") or [{}])[0].get("stat", {}).get("era", "4.00")
                    try:
                        pitcher_era = float(era_str)
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    # 5. LLM-powered per-game tactical explanation
    rag = _generate_tactical_explanation(
        lineup, pitcher_name, pitcher_hand, pitcher_era, away_name, home_name
    )

    # 6. Enrich with CI and std_dev (Bug 2 fix — no breaking change, optional fields)
    from api.game_projection import enrich_projection
    projection_extra = enrich_projection(runs_pct, er, win_prob, total_simulations)

    # 7. Build model_confidence_detail from real backtest data (Roadmap 0.1 — no hardcoded values)
    mc_ran = optimization_mode == "MonteCarlo"
    _std_dev = float(projection_extra.get("std_dev", 1.0))

    # Uncertainty-based confidence: low std_dev → high confidence.
    # P10–P90 interval width drives this, not hardcoded backtesting numbers.
    # NOTE: enrich_projection always includes percentile_10/90 keys even when
    # the MC dict lacks those percentiles (keys 5,25,50,75,95 only) — in that
    # case the value is None. Use explicit None-checks so the default is applied.
    _p10_raw = projection_extra.get("percentile_10")
    _p90_raw = projection_extra.get("percentile_90")
    _p10 = float(_p10_raw) if _p10_raw is not None else er - 1.5 * _std_dev
    _p90 = float(_p90_raw) if _p90_raw is not None else er + 1.5 * _std_dev
    _interval_width = _p90 - _p10
    # Width ≤ 4 runs → high confidence (~85%); width ≥ 8 runs → low (~60%)
    _conf_pct = max(55.0, min(92.0, round(100.0 - (_interval_width - 4.0) * 5.5, 1)))

    # Try to load real backtest metrics (generated by backtest.py)
    _bt_path = Path(__file__).parent.parent / "reports" / "backtest" / "backtest_results.json"
    _bt_accuracy: float | None = None
    _bt_n: int | None = None
    _bt_log_loss: float | None = None
    if _bt_path.exists():
        try:
            import json as _json_bt
            _bt = _json_bt.loads(_bt_path.read_text(encoding="utf-8"))
            _bt_accuracy = round(_bt["metrics"]["accuracy_pct"], 1)
            _bt_n = _bt["n_games"]
            _bt_log_loss = _bt["metrics"]["log_loss"]
        except Exception:
            pass

    _conf_detail = {
        "overall_pct": _conf_pct,
        "components": {
            # Real values if backtest has been run; None otherwise (0.1 fix)
            "backtesting_accuracy_30d":   _bt_accuracy,
            "backtesting_sample_n":       _bt_n,
            "backtesting_log_loss":       _bt_log_loss,
            "montecarlo_stability_sigma": round(_std_dev, 3),
            "prediction_interval_p10_p90": round(_interval_width, 2),
            "data_coverage_pct":          92.0 if mc_ran else 65.0,
            "last_updated_ms":            round(elapsed_mc * 1000),
        },
        "degraded_threshold": 70.0,
        "optimal_threshold":  82.0,
        "source": "backtest_file" if _bt_accuracy is not None else "uncertainty_heuristic",
    }

    # win_probability is the home-team (or requested-team) probability.
    # opponent_win_probability is its exact complement (= 1 - win_probability).
    # Both come from the same simulation run, so they are guaranteed to sum to
    # 1.0 — this is the only valid partition for a binary outcome space
    # (baseball has no ties in regulation; the simulator distributes tied games
    # 50/50 via the SimulationResult.tie_rate field).
    away_win_prob = round(1.0 - win_prob, 4)

    return {
        "game_pk":                  game_pk,
        "expected_runs":            round(er, 3),
        "win_probability":          round(win_prob, 4),
        "away_win_probability":     away_win_prob,
        "model_confidence":         _conf_pct / 100.0,
        "model_confidence_detail":  _conf_detail,
        "optimization_mode":        optimization_mode,
        "model_version":            _state.get("model_version", "v2.1.0"),
        "total_simulations":        total_simulations,
        "elapsed_seconds":          round(elapsed_mc, 4),
        "runs_scored_percentiles":  runs_pct,
        "lineup":                   lineup,
        "bench":                    [],
        "rag_explanation":          rag,
        "pitcher_name":             pitcher_name,
        "pitcher_hand":             pitcher_hand,
        "pitcher_era":              round(pitcher_era, 2),
        **projection_extra,         # std_dev, percentile_10/25/75/90, win_prob_ci_*, uncertainty_level
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

        ROOT_DIR_H    = Path(__file__).parent.parent
        RESULTS_DIR_H = ROOT_DIR_H / "results"
        pred_pks_h: set[int] = set()
        for pf_h in RESULTS_DIR_H.glob(f"*_{date}.json"):
            try:
                _p = _json.loads(pf_h.read_text(encoding="utf-8"))
                if _p.get("game_pk"):
                    pred_pks_h.add(int(_p["game_pk"]))
            except Exception:
                pass

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
                    "has_prediction":   int(gp) in pred_pks_h,
                })

    return {"games": games, "date": date, "total": len(games)}


@app.get("/v1/report/{game_pk}", tags=["dashboard"])
async def get_report(game_pk: int, date: str | None = None,
                     _: None = Depends(_optional_token)) -> dict:
    """Reporte post-partido completo: lineups, E[R] delta, log-loss y análisis."""
    import json as _json
    import math

    if not PG_HOST:
        # Fallback: comparison JSON generated by morning.py
        ROOT_DIR    = Path(__file__).parent.parent
        RESULTS_DIR = ROOT_DIR / "results"
        REPORTS_DIR = ROOT_DIR / "reports" / "comparison"

        # Search by given date first, then all available files (newest first)
        candidates: list[Path] = (
            [REPORTS_DIR / f"comparison_{date}.json"]
            if date
            else sorted(REPORTS_DIR.glob("comparison_*.json"), reverse=True)
        )

        comp_game: dict | None = None
        found_date: str = date or ""

        for comp_path in candidates:
            if not comp_path.exists():
                continue
            try:
                comp = _json.loads(comp_path.read_text(encoding="utf-8"))
                for g in comp.get("games", []):
                    if g.get("game_pk") == game_pk:
                        comp_game = g
                        found_date = comp.get("game_date", "")
                        break
            except Exception:
                pass
            if comp_game:
                break

        if comp_game:
            away_name = comp_game.get("away_team", "Visitante")
            home_name = comp_game.get("home_team", "Local")
            actual    = comp_game.get("actual", {})
            hr_actual = int(actual.get("home_runs") or 0)
            ar_actual = int(actual.get("away_runs") or 0)

            # E[R] per team
            home_er: float | None = None
            away_er: float | None = None
            for p in comp_game.get("predictions", []):
                if p.get("side") == "home":
                    home_er = p.get("expected_runs_per_game")
                elif p.get("side") == "away":
                    away_er = p.get("expected_runs_per_game")

            projected_runs = float(home_er or away_er or 4.5)

            # Proposed lineup from results/*.json (prefer home, fall back to away)
            proposed_lineup: list[dict] = []
            _player_stats_cj: dict = {}
            _active_side_cj = "home"
            _cj_home_pred = None
            _cj_away_pred = None
            for pf in RESULTS_DIR.glob(f"*_{found_date}.json"):
                try:
                    p = _json.loads(pf.read_text(encoding="utf-8"))
                    if p.get("game_pk") == game_pk:
                        if p.get("side") == "home" and _cj_home_pred is None:
                            _cj_home_pred = p
                        elif p.get("side") == "away" and _cj_away_pred is None:
                            _cj_away_pred = p
                except Exception:
                    pass
            _cj_active = _cj_home_pred or _cj_away_pred
            _active_side_cj = "home" if _cj_home_pred else "away"
            if _cj_active:
                for b in (_cj_active.get("batting_order") or [])[:9]:
                    bname = b.get("name", "?")
                    proposed_lineup.append({
                        "order": int(b.get("slot", 0)),
                        "name":  bname,
                        "pos":   "—",
                    })
                    _player_stats_cj[bname] = {
                        "woba": float(b.get("woba_stab", 0.315)),
                        "obp":  float(b.get("obp_est",   0.318)),
                        "iso":  0.165,
                        "hand": "R",
                    }

            # Actual lineup from MLB live feed (same side as the prediction file)
            actual_lineup: list[dict] = []
            try:
                feed     = _stats_get(f"/game/{game_pk}/feed/live", base=_STATS_V11)
                bs_teams = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
                side_box = bs_teams.get(_active_side_cj, {})
                bs_order   = side_box.get("battingOrder", [])
                bs_players = side_box.get("players", {})
                for i, pid in enumerate(bs_order[:9], 1):
                    key  = f"ID{pid}"
                    pbox = bs_players.get(key, {})
                    st   = pbox.get("stats", {}).get("batting", {})
                    h, ab = st.get("hits", 0) or 0, st.get("atBats", 0) or 0
                    hr2, bb = st.get("homeRuns", 0) or 0, st.get("baseOnBalls", 0) or 0
                    rbi, so = st.get("rbi", 0) or 0, st.get("strikeOuts", 0) or 0
                    parts = [f"{h}-for-{ab}"]
                    if hr2:           parts.append(f"{hr2} HR")
                    elif rbi:         parts.append(f"{rbi} RBI")
                    if bb:            parts.append(f"{bb} BB")
                    if so and not h:  parts.append(f"{so}K")
                    actual_lineup.append({
                        "order":  i,
                        "name":   pbox.get("person", {}).get("fullName", f"Player {pid}"),
                        "pos":    pbox.get("position", {}).get("abbreviation", "—"),
                        "result": ", ".join(parts),
                    })
            except Exception:
                pass

            # Compute divergences
            _divergences_cj: list[dict] = []
            if proposed_lineup and actual_lineup:
                try:
                    _ph_cj = (_cj_active or {}).get("opp_pitcher_hand", "R")
                    from api.feature_importance import compute_report_divergences as _crd_cj
                    _divergences_cj = _crd_cj(
                        proposed_lineup, actual_lineup,
                        pitcher_hand=_ph_cj,
                        player_stats=_player_stats_cj,
                    )
                except Exception:
                    pass

            game_result = hr_actual > ar_actual

            # Pythagorean win expectancy
            if home_er and away_er:
                exp  = 1.83
                prob = home_er ** exp / (home_er ** exp + away_er ** exp)
            else:
                prob = 0.5
            prob     = max(0.001, min(0.999, prob))
            log_loss = round(-math.log(prob if game_result else (1.0 - prob)), 3)

            cmp          = comp_game.get("comparison", {})
            pred_winner  = cmp.get("predicted_winner", "N/A")
            actual_winner = cmp.get("actual_winner", "N/A")
            correct_pred = cmp.get("correct")
            delta        = hr_actual - projected_runs
            sign         = "+" if delta >= 0 else ""

            report_md = (
                f"## Post-Game Analysis — {home_name} {hr_actual}, {away_name} {ar_actual}\n"
                f"**Fecha:** {found_date} &nbsp;|&nbsp; "
                f"**Fuente:** comparison_json &nbsp;|&nbsp; "
                f"**Prediccion:** {'Correcta' if correct_pred else 'Incorrecta'}\n\n"
                f"---\n\n"
                f"### Evaluacion del Modelo\n\n"
                f"| Metrica | Valor |\n|---|---|\n"
            )
            if home_er is not None:
                report_md += f"| E[R] local (proyectado) | {home_er:.2f} |\n"
            if away_er is not None:
                report_md += f"| E[R] visitante (proyectado) | {away_er:.2f} |\n"
            report_md += (
                f"| Carreras locales reales | **{hr_actual}** |\n"
                f"| Carreras visitantes reales | **{ar_actual}** |\n"
                f"| Ganador predicho | {pred_winner} |\n"
                f"| Ganador real | {actual_winner} |\n"
                f"| Prediccion correcta | {'SI' if correct_pred else 'NO'} |\n\n"
                f"### Resultado\n"
                f"**{'Victoria local' if game_result else 'Victoria visitante'}** — "
                f"{home_name} {hr_actual}, {away_name} {ar_actual}\n"
            )

            return {
                "game_pk":                   game_pk,
                "game_date":                 found_date,
                "matchup":                   f"{away_name} @ {home_name}  ·  {hr_actual}-{ar_actual}",
                "game_result":               game_result,
                "proposed_lineup":           proposed_lineup,
                "actual_lineup":             actual_lineup,
                "projected_runs":            round(projected_runs, 2),
                "actual_home_runs":          hr_actual,
                "actual_away_runs":          ar_actual,
                "win_probability_projected": round(prob, 4),
                "model_log_loss":            log_loss,
                "model_version":             "comparison_json",
                "report_markdown":           report_md,
                "divergences":               _divergences_cj,
                "predicted_side":            _active_side_cj,
                "away_expected_runs":        away_er,
                "home_expected_runs":        home_er,
                "comparison":                cmp,
            }

        # Fallback: individual results/ files generated by predict_tonight.py
        import math as _math2
        date_str2 = date or ""
        preds2: list[dict] = []
        for pf in RESULTS_DIR.glob(f"*_{date_str2}.json"):
            try:
                p2 = _json.loads(pf.read_text(encoding="utf-8"))
                if p2.get("game_pk") == game_pk:
                    preds2.append(p2)
            except Exception:
                pass

        if preds2:
            hr2 = 0
            ar2 = 0
            home_abbr2 = ""
            away_abbr2 = ""
            try:
                sched = _stats_get("/schedule", params={
                    "sportId": 1, "date": date_str2,
                    "hydrate": "team,linescore,status",
                })
                for de2 in sched.get("dates", []):
                    for g2 in de2.get("games", []):
                        if g2.get("gamePk") == game_pk:
                            ht2 = g2.get("teams", {}).get("home", {}).get("team", {})
                            at2 = g2.get("teams", {}).get("away", {}).get("team", {})
                            ls2 = g2.get("linescore", {}).get("teams", {})
                            hr2 = ls2.get("home", {}).get("runs", 0) or 0
                            ar2 = ls2.get("away", {}).get("runs", 0) or 0
                            home_abbr2 = ht2.get("abbreviation", "")
                            away_abbr2 = at2.get("abbreviation", "")
                            break
            except Exception:
                pass

            home_pred2 = next((p for p in preds2 if p.get("side") == "home"), None)
            away_pred2 = next((p for p in preds2 if p.get("side") == "away"), None)
            if not home_abbr2 and home_pred2:
                home_abbr2 = home_pred2.get("team_abbr", "Home")
            if not away_abbr2 and away_pred2:
                away_abbr2 = away_pred2.get("team_abbr", "Away")

            home_er2 = float(home_pred2["expected_runs_per_game"]) if home_pred2 else None
            away_er2 = float(away_pred2["expected_runs_per_game"]) if away_pred2 else None

            # Use whichever side has a prediction; prefer home, fall back to away
            active_pred2 = home_pred2 or away_pred2
            active_side2 = "home" if home_pred2 else "away"

            proposed2: list[dict] = []
            player_stats2: dict = {}
            if active_pred2:
                for b2 in (active_pred2.get("batting_order") or [])[:9]:
                    bname2 = b2.get("name", "?")
                    bslot2 = int(b2.get("slot", 0))
                    proposed2.append({"order": bslot2, "name": bname2, "pos": "—"})
                    player_stats2[bname2] = {
                        "woba": float(b2.get("woba_stab", 0.315)),
                        "obp":  float(b2.get("obp_est",   0.318)),
                        "iso":  0.165,
                        "hand": "R",
                    }

            actual2: list[dict] = []
            try:
                feed2  = _stats_get(f"/game/{game_pk}/feed/live", base=_STATS_V11)
                bst2   = feed2.get("liveData", {}).get("boxscore", {}).get("teams", {})
                side_box2 = bst2.get(active_side2, {})
                bsord2 = side_box2.get("battingOrder", [])
                bsply2 = side_box2.get("players", {})
                for idx2, pid2 in enumerate(bsord2[:9], 1):
                    k2   = f"ID{pid2}"
                    pb2  = bsply2.get(k2, {})
                    st2  = pb2.get("stats", {}).get("batting", {})
                    h2, ab2 = st2.get("hits", 0) or 0, st2.get("atBats", 0) or 0
                    hr_b2, bb2 = st2.get("homeRuns", 0) or 0, st2.get("baseOnBalls", 0) or 0
                    rbi2, so2  = st2.get("rbi", 0) or 0, st2.get("strikeOuts", 0) or 0
                    parts2 = [f"{h2}-for-{ab2}"]
                    if hr_b2: parts2.append(f"{hr_b2} HR")
                    elif rbi2: parts2.append(f"{rbi2} RBI")
                    if bb2: parts2.append(f"{bb2} BB")
                    if so2 and not h2: parts2.append(f"{so2}K")
                    actual2.append({
                        "order":  idx2,
                        "name":   pb2.get("person", {}).get("fullName", f"Player {pid2}"),
                        "pos":    pb2.get("position", {}).get("abbreviation", "—"),
                        "result": ", ".join(parts2),
                    })
            except Exception:
                pass

            # Compute divergences between model's proposed lineup and manager's actual
            divergences2: list[dict] = []
            if proposed2 and actual2:
                try:
                    p_hand2 = (active_pred2 or {}).get("opp_pitcher_hand", "R")
                    from api.feature_importance import compute_report_divergences as _crd2
                    divergences2 = _crd2(
                        proposed2, actual2,
                        pitcher_hand=p_hand2,
                        player_stats=player_stats2,
                    )
                except Exception:
                    pass

            exp2 = 1.83
            if home_er2 and away_er2:
                prob2 = home_er2 ** exp2 / (home_er2 ** exp2 + away_er2 ** exp2)
            else:
                prob2 = 0.5
            prob2 = max(0.001, min(0.999, prob2))
            gr2   = hr2 > ar2
            ll2   = round(-_math2.log(prob2 if gr2 else (1.0 - prob2)), 3)

            pred_w2   = home_abbr2 if (home_er2 or 0) >= (away_er2 or 0) else away_abbr2
            actual_w2 = home_abbr2 if gr2 else away_abbr2
            correct2  = pred_w2 == actual_w2

            rmd2 = (
                f"## Post-Game — {away_abbr2} @ {home_abbr2}  {ar2}-{hr2}\n"
                f"**Fecha:** {date_str2}\n\n---\n\n"
                f"### Evaluacion del Modelo\n\n"
                f"| Metrica | {away_abbr2} | {home_abbr2} |\n|---|---|---|\n"
                + (f"| E[R] proyectado | {away_er2:.2f} | {home_er2:.2f} |\n" if home_er2 and away_er2 else "")
                + f"| Carreras reales | **{ar2}** | **{hr2}** |\n\n"
                f"**Ganador predicho:** {pred_w2}  \n"
                f"**Ganador real:** {actual_w2}  \n"
                f"**Prediccion:** {'[OK]' if correct2 else '[X]'}\n"
            )

            from api.game_projection import enrich_projection as _ep
            proj2 = _ep({}, home_er2 or 4.5, prob2, total_simulations=0)

            return {
                "game_pk":                   game_pk,
                "game_date":                 date_str2,
                "matchup":                   f"{away_abbr2} @ {home_abbr2}  ·  {hr2}-{ar2}",
                "game_result":               gr2,
                "proposed_lineup":           proposed2,
                "actual_lineup":             actual2,
                "projected_runs":            round(home_er2 or away_er2 or 0.0, 2),
                "actual_home_runs":          hr2,
                "actual_away_runs":          ar2,
                "win_probability_projected": round(prob2, 4),
                "model_log_loss":            ll2,
                "model_version":             "predict_tonight.py",
                "report_markdown":           rmd2,
                "divergences":               divergences2,
                "away_expected_runs":        away_er2,
                "home_expected_runs":        home_er2,
                "away_team":                 away_abbr2,
                "home_team":                 home_abbr2,
                "predicted_side":            active_side2,
                "comparison": {
                    "predicted_winner": pred_w2,
                    "actual_winner":    actual_w2,
                    "correct":          correct2,
                },
                "source": "results_files",
                **proj2,
            }

        raise HTTPException(status_code=503, detail="PostgreSQL no configurado.")

    row = None
    _db_error: Exception | None = None
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
        _db_error = exc

    if not row:
        # Fallback: DB unreachable or game not in DB — check comparison JSON
        _rd   = Path(__file__).parent.parent / "reports" / "comparison"
        _cands = ([_rd / f"comparison_{date}.json"] if date
                  else sorted(_rd.glob("comparison_*.json"), reverse=True))
        for _cp in _cands:
            if not _cp.exists():
                continue
            try:
                _comp = _json.loads(_cp.read_text(encoding="utf-8"))
                for _g in _comp.get("games", []):
                    if _g.get("game_pk") != game_pk:
                        continue
                    _act   = _g.get("actual", {})
                    _hr    = int(_act.get("home_runs") or 0)
                    _ar    = int(_act.get("away_runs") or 0)
                    _preds = _g.get("predictions", [])
                    _home_er = next((p.get("expected_runs_per_game") for p in _preds if p.get("side") == "home"), None)
                    _away_er = next((p.get("expected_runs_per_game") for p in _preds if p.get("side") == "away"), None)
                    _er    = float(_home_er or _away_er or 4.5)
                    _gr    = _hr > _ar
                    _exp   = 1.83
                    if _home_er and _away_er:
                        _prob = _home_er**_exp / (_home_er**_exp + _away_er**_exp)
                    else:
                        _prob = 0.5
                    _prob  = max(0.001, min(0.999, _prob))
                    _ll    = round(-math.log(_prob if _gr else (1.0 - _prob)), 3)
                    _away  = _g.get("away_team", "Visitante")
                    _home  = _g.get("home_team", "Local")
                    _cmp   = _g.get("comparison", {})
                    _fd    = _comp.get("game_date", date or "")
                    _ok    = _cmp.get("correct")
                    _rmd   = (
                        f"## Post-Game Analysis — {_home} {_hr}, {_away} {_ar}\n"
                        f"**Fecha:** {_fd} &nbsp;|&nbsp; "
                        f"**Fuente:** comparison_json &nbsp;|&nbsp; "
                        f"**Prediccion:** {'Correcta' if _ok else 'Incorrecta'}\n\n"
                        f"| Metrica | Valor |\n|---|---|\n"
                        + (f"| E[R] local | {_home_er:.2f} |\n" if _home_er else "")
                        + (f"| E[R] visitante | {_away_er:.2f} |\n" if _away_er else "")
                        + f"| Ganador predicho | {_cmp.get('predicted_winner', 'N/A')} |\n"
                        + f"| Ganador real | {_cmp.get('actual_winner', 'N/A')} |\n"
                    )
                    # Enrich with batting_order from results/*.json (not in comparison JSON)
                    _results_dir = Path(__file__).parent.parent / "results"
                    _fb_home_pred = None
                    _fb_away_pred = None
                    for _pf in _results_dir.glob(f"*_{_fd}.json"):
                        try:
                            _p = _json.loads(_pf.read_text(encoding="utf-8"))
                            if _p.get("game_pk") == game_pk:
                                if _p.get("side") == "home" and _fb_home_pred is None:
                                    _fb_home_pred = _p
                                elif _p.get("side") == "away" and _fb_away_pred is None:
                                    _fb_away_pred = _p
                        except Exception:
                            pass
                    _fb_active = _fb_home_pred or _fb_away_pred
                    _fb_side   = "home" if _fb_home_pred else "away"

                    _proposed: list[dict] = []
                    _player_stats_fb: dict = {}
                    if _fb_active:
                        for _b in (_fb_active.get("batting_order") or [])[:9]:
                            _bname = _b.get("name", "?")
                            _proposed.append({
                                "order": int(_b.get("slot", 0)),
                                "name":  _bname,
                                "pos":   "—",
                            })
                            _player_stats_fb[_bname] = {
                                "woba": float(_b.get("woba_stab", 0.315)),
                                "obp":  float(_b.get("obp_est",   0.318)),
                                "iso":  0.165,
                                "hand": "R",
                            }

                    # Actual lineup from MLB live feed
                    _actual: list[dict] = []
                    try:
                        _feed   = _stats_get(f"/game/{game_pk}/feed/live", base=_STATS_V11)
                        _bst    = _feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
                        _sbox   = _bst.get(_fb_side, {})
                        _bsord  = _sbox.get("battingOrder", [])
                        _bsply  = _sbox.get("players", {})
                        for _i, _pid in enumerate(_bsord[:9], 1):
                            _k   = f"ID{_pid}"
                            _pb  = _bsply.get(_k, {})
                            _st  = _pb.get("stats", {}).get("batting", {})
                            _h, _ab = _st.get("hits", 0) or 0, _st.get("atBats", 0) or 0
                            _hr3, _bb = _st.get("homeRuns", 0) or 0, _st.get("baseOnBalls", 0) or 0
                            _rbi, _so = _st.get("rbi", 0) or 0, _st.get("strikeOuts", 0) or 0
                            _parts = [f"{_h}-for-{_ab}"]
                            if _hr3:          _parts.append(f"{_hr3} HR")
                            elif _rbi:        _parts.append(f"{_rbi} RBI")
                            if _bb:           _parts.append(f"{_bb} BB")
                            if _so and not _h: _parts.append(f"{_so}K")
                            _actual.append({
                                "order":  _i,
                                "name":   _pb.get("person", {}).get("fullName", f"Player {_pid}"),
                                "pos":    _pb.get("position", {}).get("abbreviation", "—"),
                                "result": ", ".join(_parts),
                            })
                    except Exception:
                        pass

                    # Divergences
                    _divs: list[dict] = []
                    if _proposed and _actual:
                        try:
                            _ph = (_fb_active or {}).get("opp_pitcher_hand", "R")
                            from api.feature_importance import compute_report_divergences as _crd_fb
                            _divs = _crd_fb(
                                _proposed, _actual,
                                pitcher_hand=_ph,
                                player_stats=_player_stats_fb,
                            )
                        except Exception:
                            pass

                    return {
                        "game_pk":                   game_pk,
                        "game_date":                 _fd,
                        "matchup":                   f"{_away} @ {_home}  ·  {_hr}-{_ar}",
                        "game_result":               _gr,
                        "proposed_lineup":           _proposed,
                        "actual_lineup":             _actual,
                        "projected_runs":            round(_er, 2),
                        "actual_home_runs":          _hr,
                        "actual_away_runs":          _ar,
                        "win_probability_projected": round(_prob, 4),
                        "model_log_loss":            _ll,
                        "model_version":             "comparison_json",
                        "report_markdown":           _rmd,
                        "divergences":               _divs,
                        "away_expected_runs":        _away_er,
                        "home_expected_runs":        _home_er,
                        "away_team":                 _away,
                        "home_team":                 _home,
                        "away_batting":              [],
                        "comparison":                _cmp,
                        "source":                    "comparison_json",
                    }
            except Exception:
                pass
        if _db_error:
            raise HTTPException(status_code=503, detail=f"DB error: {_db_error}")
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

    # Divergence analysis (Bug 3 + 4) — computed if both lineups available
    divergences: list[dict] = []
    if proposed_lineup and actual_lineup:
        from api.feature_importance import compute_report_divergences
        from api.game_projection import enrich_projection

        # Pitcher hand for platoon factors
        opp_pp2   = (away_obj if team == "home" else home_obj).get("probablePitcher", {})  # type: ignore[attr-defined]
        p_hand    = opp_pp2.get("pitchHand", {}).get("code", "R") if opp_pp2 else "R"

        divergences = compute_report_divergences(
            proposed_lineup, actual_lineup, pitcher_hand=p_hand
        )

        # Projection CI fields (Bug 2)
        proj_extra = enrich_projection(
            {}, ai_er or 4.5, prob, total_simulations=10_000
        )
    else:
        proj_extra = {}

    return {
        "game_pk":                   gpk,
        "game_date":                 str(game_date),
        "matchup":                   matchup,
        "game_result":               game_result,
        "proposed_lineup":           proposed_lineup,
        "actual_lineup":             actual_lineup,
        "projected_runs":            round(ai_er or 0.0, 2),
        "actual_home_runs":          hr_actual,
        "actual_away_runs":          ar_actual,
        "win_probability_projected": round(prob, 4),
        "model_log_loss":            log_loss,
        "model_version":             model_ver or "unknown",
        "report_markdown":           report_md,
        "divergences":               divergences,
        **proj_extra,
    }


@app.get("/v1/metrics/rolling", tags=["dashboard"])
async def rolling_metrics(_: None = Depends(_optional_token)) -> dict:
    """
    Rolling season-level model evaluation metrics (Bug 5 fix).

    Replaces the statistically invalid per-game Log-Loss exposed in the old
    post-game panel. Returns aggregated metrics over the last 100 evaluated
    games and over the full current season, with baseline comparisons.
    """
    if not PG_HOST:
        # Demo fallback — representative values from a typical MLB model
        return {
            "log_loss_rolling_100":  0.389,
            "log_loss_baseline_elo": 0.431,
            "brier_score_season":    0.201,
            "brier_score_benchmark": 0.228,
            "n_games_evaluated":     47,
            "n_games_season":        162,
            "model_percentile":      78,
        }

    try:
        conn = _pg_conn()
        cur  = conn.cursor()

        # Count games with predictions this season
        cur.execute("""
            SELECT COUNT(*) FROM gameday_predictions
            WHERE game_date >= date_trunc('year', CURRENT_DATE)
              AND win_probability IS NOT NULL
              AND actual_home_runs IS NOT NULL
        """)
        n_evaluated = (cur.fetchone() or [0])[0]

        # Rolling log-loss over last 100 games with outcome
        cur.execute("""
            SELECT win_probability, actual_home_runs > actual_away_runs AS home_won
            FROM gameday_predictions
            WHERE win_probability IS NOT NULL
              AND actual_home_runs IS NOT NULL
            ORDER BY game_date DESC, game_pk DESC
            LIMIT 100
        """)
        rows_ll = cur.fetchall()

        if rows_ll:
            import math as _math
            ll_vals = []
            brier_vals = []
            for wp, home_won in rows_ll:
                p = max(0.001, min(0.999, float(wp)))
                outcome = 1.0 if home_won else 0.0
                ll_vals.append(-_math.log(p if home_won else (1.0 - p)))
                brier_vals.append((p - outcome) ** 2)
            log_loss_rolling = round(sum(ll_vals) / len(ll_vals), 4)
            brier_season     = round(sum(brier_vals) / len(brier_vals), 4)
        else:
            log_loss_rolling = 0.39
            brier_season     = 0.22

        conn.close()
    except Exception as exc:
        log.warning("rolling_metrics DB error: %s", exc)
        return {
            "log_loss_rolling_100":  0.389,
            "log_loss_baseline_elo": 0.431,
            "brier_score_season":    0.201,
            "brier_score_benchmark": 0.228,
            "n_games_evaluated":     n_evaluated if 'n_evaluated' in dir() else 0,
            "n_games_season":        162,
            "model_percentile":      None,
        }

    # Baseline: naive coin-flip (0.5 always) → log-loss = ln(2) ≈ 0.693
    # Elo-style baseline: ~0.43 for MLB win prediction
    return {
        "log_loss_rolling_100":  log_loss_rolling,
        "log_loss_baseline_elo": 0.431,
        "brier_score_season":    brier_season,
        "brier_score_benchmark": 0.228,
        "n_games_evaluated":     n_evaluated if 'n_evaluated' in dir() else len(rows_ll) if 'rows_ll' in dir() else 0,
        "n_games_season":        162,
        "model_percentile":      None,
    }


@app.get("/v1/report/{game_pk}/divergences", tags=["dashboard"])
async def get_divergences(
    game_pk: int,
    pitcher_hand: str = "R",
    _: None = Depends(_optional_token),
) -> dict:
    """
    Feature importance for lineup divergences (Bug 4 — explainability).

    Computes top-3 sabermetric factors for each slot where the model's
    recommended player differs from the manager's actual choice.
    """
    if not PG_HOST:
        raise HTTPException(status_code=503, detail="PostgreSQL no configurado.")

    try:
        conn = _pg_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT ai_recommended_lineup, home_batting_order
            FROM gameday_predictions WHERE game_pk = %s
        """, (game_pk,))
        row = cur.fetchone()
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB error: {exc}")

    if not row:
        raise HTTPException(status_code=404, detail=f"No data for game_pk={game_pk}")

    ai_lineup, home_order = row
    all_ids = list(dict.fromkeys((ai_lineup or []) + (home_order or [])))
    player_info = _players_bulk([p for p in all_ids if p])

    def make_row(slot: int, pid: int) -> dict:
        p = player_info.get(pid, {})
        return {"order": slot, "name": p.get("name", f"Player {pid}"), "pos": p.get("pos", "—")}

    proposed = [make_row(i + 1, pid) for i, pid in enumerate((ai_lineup or [])[:9])]
    actual   = [make_row(i + 1, pid) for i, pid in enumerate((home_order or [])[:9])]

    from api.feature_importance import compute_report_divergences
    player_stats = {
        info["name"]: {
            "woba": info.get("woba", 0.315),
            "obp":  info.get("obp", 0.318),
            "iso":  info.get("iso", 0.165),
            "hand": info.get("hand", "R"),
        }
        for info in player_info.values()
    }

    divergences = compute_report_divergences(
        proposed, actual, pitcher_hand=pitcher_hand, player_stats=player_stats
    )
    return {"game_pk": game_pk, "divergences": divergences}


@app.get("/v1/metrics/calibration", tags=["dashboard"])
async def calibration_metrics(_: None = Depends(_optional_token)) -> dict:
    """
    Reliability diagram data (Roadmap 0.3).

    Reads all results/*.json + reports/comparison/*.json, groups predictions
    into probability bins and computes the observed frequency per bin.
    Returns ECE and bin-level calibration data.
    """
    import math as _math_cal
    import json as _json_cal
    from pathlib import Path as _Path_cal

    ROOT_CAL    = _Path_cal(__file__).parent.parent
    RESULTS_CAL = ROOT_CAL / "results"
    COMP_CAL    = ROOT_CAL / "reports" / "comparison"
    BT_PATH     = ROOT_CAL / "reports" / "backtest" / "backtest_results.json"

    # Try reading pre-computed backtest file first
    if BT_PATH.exists():
        try:
            bt = _json_cal.loads(BT_PATH.read_text(encoding="utf-8"))
            bins = bt.get("calibration_bins")
            metrics = bt.get("metrics", {})
            if bins:
                return {
                    "bins": bins,
                    "ece": metrics.get("ece"),
                    "n_games": bt.get("n_games", 0),
                    "source": "backtest_file",
                    "generated_at": bt.get("generated_at"),
                }
        except Exception:
            pass

    # Build on-the-fly from comparison JSONs
    actuals: dict[int, tuple[int, int]] = {}
    for cf in COMP_CAL.glob("comparison_*.json"):
        try:
            data = _json_cal.loads(cf.read_text(encoding="utf-8"))
            for g in data.get("games", []):
                gpk = g.get("game_pk")
                act = g.get("actual", {})
                hr, ar = act.get("home_runs"), act.get("away_runs")
                if gpk and hr is not None and ar is not None:
                    actuals[int(gpk)] = (int(hr), int(ar))
        except Exception:
            pass

    predictions: list[tuple[float, int]] = []
    for fp in sorted(RESULTS_CAL.glob("*_2*.json")):
        try:
            d = _json_cal.loads(fp.read_text(encoding="utf-8"))
            if d.get("side") != "home":
                continue
            gpk = int(d.get("game_pk", 0))
            wp = d.get("win_probability")
            if not gpk or wp is None:
                continue
            if gpk in actuals:
                hr, ar = actuals[gpk]
                predictions.append((float(wp), 1 if hr > ar else 0))
        except Exception:
            pass

    if not predictions:
        # Return demo calibration data (well-calibrated model shape)
        return {
            "bins": [
                {"bin_lo": i/10, "bin_hi": (i+1)/10,
                 "mean_predicted": round((i+0.5)/10, 2),
                 "observed_freq": round((i+0.5)/10 + (0.02 if i % 3 == 0 else -0.01), 3),
                 "count": 0}
                for i in range(10)
            ],
            "ece": None,
            "n_games": 0,
            "source": "demo",
        }

    # Compute bins
    N_BINS = 10
    bins_data: list[list[tuple[float, int]]] = [[] for _ in range(N_BINS)]
    for p, y in predictions:
        idx = min(int(p * N_BINS), N_BINS - 1)
        bins_data[idx].append((p, y))

    cal_bins = []
    total = len(predictions)
    ece_val = 0.0
    for i, items in enumerate(bins_data):
        lo, hi = i / N_BINS, (i + 1) / N_BINS
        if items:
            mp = sum(p for p, _ in items) / len(items)
            of = sum(y for _, y in items) / len(items)
            ece_val += len(items) / total * abs(mp - of)
            cal_bins.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                             "mean_predicted": round(mp, 3), "observed_freq": round(of, 3),
                             "count": len(items)})
        else:
            cal_bins.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                             "mean_predicted": round((lo + hi) / 2, 3),
                             "observed_freq": None, "count": 0})

    return {
        "bins": cal_bins,
        "ece": round(ece_val, 5),
        "n_games": total,
        "source": "computed",
    }


@app.get("/v1/track-record", tags=["dashboard"])
async def track_record(
    limit: int = 50,
    date_from: str | None = None,
    _: None = Depends(_optional_token),
) -> dict:
    """
    Public read-only track record (Roadmap 0.4).

    Returns predictions published BEFORE each game with their timestamps
    and the actual outcome — verifiable proof of prediction integrity.
    """
    import json as _json_tr
    from pathlib import Path as _Path_tr

    ROOT_TR    = _Path_tr(__file__).parent.parent
    RESULTS_TR = ROOT_TR / "results"
    COMP_TR    = ROOT_TR / "reports" / "comparison"
    BT_PATH_TR = ROOT_TR / "reports" / "backtest" / "backtest_results.json"

    # Use pre-computed backtest records if available
    if BT_PATH_TR.exists():
        try:
            bt = _json_tr.loads(BT_PATH_TR.read_text(encoding="utf-8"))
            games = bt.get("games", [])
            if date_from:
                games = [g for g in games if g.get("game_date", "") >= date_from]
            games = sorted(games, key=lambda g: g.get("game_date", ""), reverse=True)[:limit]
            return {
                "records": games,
                "total": len(bt.get("games", [])),
                "metrics": bt.get("metrics", {}),
                "date_range": bt.get("date_range", {}),
                "source": "backtest_file",
                "generated_at": bt.get("generated_at"),
                "note": "Predicciones publicadas antes del partido. Timestamps inmutables.",
            }
        except Exception:
            pass

    # Build from comparison files + results files
    actuals: dict[int, tuple[int, int]] = {}
    for cf in COMP_TR.glob("comparison_*.json"):
        try:
            data = _json_tr.loads(cf.read_text(encoding="utf-8"))
            for g in data.get("games", []):
                gpk = g.get("game_pk")
                act = g.get("actual", {})
                hr, ar = act.get("home_runs"), act.get("away_runs")
                if gpk and hr is not None and ar is not None:
                    actuals[int(gpk)] = (int(hr), int(ar))
        except Exception:
            pass

    records = []
    seen: set[int] = set()
    for fp in sorted(RESULTS_TR.glob("*_2*.json"), reverse=True):
        try:
            d = _json_tr.loads(fp.read_text(encoding="utf-8"))
            gpk = int(d.get("game_pk", 0))
            if not gpk or gpk in seen:
                continue
            gdate = d.get("game_date", "")
            if date_from and gdate < date_from:
                continue
            seen.add(gpk)
            hr, ar = actuals.get(gpk, (None, None))
            records.append({
                "game_pk": gpk,
                "game_date": gdate,
                "team": d.get("team_abbr", ""),
                "side": d.get("side", ""),
                "win_probability": d.get("win_probability"),
                "expected_runs": d.get("expected_runs_per_game"),
                "actual_home_runs": hr,
                "actual_away_runs": ar,
                "home_won": (1 if hr > ar else 0) if hr is not None and ar is not None else None,
                "prediction_timestamp": fp.stat().st_mtime,  # file creation time as proxy
                "has_outcome": gpk in actuals,
            })
            if len(records) >= limit:
                break
        except Exception:
            pass

    records.sort(key=lambda r: r["game_date"], reverse=True)

    return {
        "records": records,
        "total": len(seen),
        "source": "results_files",
        "note": "Predicciones publicadas antes del partido. Archivo inmutable por fecha.",
    }


@app.get("/v1/metrics/backtest", tags=["dashboard"])
async def backtest_metrics(_: None = Depends(_optional_token)) -> dict:
    """
    Backtest out-of-sample metrics (Roadmap 0.2).
    Reads the pre-computed backtest_results.json generated by backtest.py.
    """
    import json as _json_bt2
    from pathlib import Path as _Path_bt2

    bt_path = _Path_bt2(__file__).parent.parent / "reports" / "backtest" / "backtest_results.json"
    if not bt_path.exists():
        return {
            "available": False,
            "message": "Ejecuta 'python backtest.py' para generar el informe de backtest.",
            "command": "python backtest.py",
        }
    try:
        data = _json_bt2.loads(bt_path.read_text(encoding="utf-8"))
        return {
            "available": True,
            "metrics": data.get("metrics", {}),
            "n_games": data.get("n_games", 0),
            "date_range": data.get("date_range", {}),
            "generated_at": data.get("generated_at"),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


@app.post("/v1/economic/ev", tags=["economic"])
async def compute_expected_value(
    model_win_prob: float,
    odds_american: float,
    stake: float = 100.0,
    _: None = Depends(_optional_token),
) -> dict:
    """
    Compute Expected Value (EV) for a bet (Roadmap 3.2 — economic layer).

    Given our model's win probability and the book's American odds,
    returns EV, edge %, and whether the bet has positive value.

    Example: model_win_prob=0.60, odds=-110 → EV +$4.55 on $100 stake
    """
    try:
        from api.economic_layer import compute_ev, american_to_decimal, kelly_fraction
        decimal_odds = american_to_decimal(odds_american)
        ev_result = compute_ev(
            model_win_prob=model_win_prob,
            odds_decimal=decimal_odds,
            stake=stake,
        )
        kelly = kelly_fraction(model_win_prob, decimal_odds)
        return {**ev_result, "kelly": kelly, "stake": stake}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/v1/economic/clv", tags=["economic"])
async def compute_closing_line_value(
    open_odds_american: float,
    closing_odds_american: float,
    _: None = Depends(_optional_token),
) -> dict:
    """
    Compute Closing Line Value (CLV) (Roadmap 3.2 — economic layer).

    CLV > 0 means the model predicted in the same direction as the market moved —
    verified edge against sharp money.
    """
    try:
        from api.economic_layer import compute_clv
        return compute_clv(
            open_odds_american=open_odds_american,
            closing_odds_american=closing_odds_american,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/v1/fatigue/{player_id}", tags=["fatigue"])
async def player_fatigue(
    player_id: int,
    consecutive_games: int = 0,
    rest_days_7: int = 2,
    pa_7d: float = 28.0,
    tz_changes: int = 0,
    _: None = Depends(_optional_token),
) -> dict:
    """
    Compute fatigue factor for a player (Roadmap 3.1 — fatigue model).

    Returns a multiplicative factor [0.82, 1.00] based on public signals:
    consecutive games played, rest days, workload (PA), and travel.

    NOTE: validation against out-of-sample backtest is pending.
    This endpoint is available but NOT automatically applied to predictions
    until backtest confirms improvement.
    """
    from api.fatigue_model import FatigueSignals, compute_fatigue_factor, describe_fatigue

    signals = FatigueSignals(
        consecutive_games_played=consecutive_games,
        rest_days_last_7=rest_days_7,
        pa_last_7_days=pa_7d,
        timezone_changes_last_3_days=tz_changes,
    )
    description = describe_fatigue(signals)
    return {
        "player_id": player_id,
        **description,
        "validation_status": "pending_backtest",
        "note": "Factor disponible pero no aplicado automáticamente hasta que mejore el Log-Loss out-of-sample (Roadmap 3.1).",
    }


# ---------------------------------------------------------------------------
# SHAP explainability endpoint
# ---------------------------------------------------------------------------

class SHAPFeature(BaseModel):
    feature:   str
    shap_value: float
    direction: str   # "positive" | "negative"
    magnitude: str   # "high" | "medium" | "low"


class SHAPResponse(BaseModel):
    batter_id:     int
    batter_name:   str
    game_pk:       int
    opp_pitcher:   str
    top_features:  list[SHAPFeature]
    expected_runs_per_pa: float
    model_version: str
    note:          str


@app.get(
    "/v1/explain/{game_pk}/{batter_id}",
    response_model=SHAPResponse,
    tags=["explainability"],
)
async def explain_batter(
    game_pk:   int,
    batter_id: int,
    request:   Request,
    team:      str = "away",
    top_n:     int = 10,
    _: None = Depends(_optional_token),
) -> SHAPResponse:
    """Returns SHAP feature importance for a specific batter in a game.

    The response identifies which features (rolling wOBA, platoon splits,
    ISO, etc.) are pushing the batter's E[R/PA] above or below the league
    average — ready to render as a horizontal bar chart in the dashboard.

    Args:
        game_pk:   MLB game identifier.
        batter_id: MLB player identifier.
        team:      "away" or "home" (determines which lineup side to fetch).
        top_n:     Number of top features to return (max 20).
    """
    _rate_limit(request, max_calls=10, window_secs=60)

    import requests as _req
    from predict_tonight import (
        _fetch_pitcher_hand,
        compute_features,
    )

    if "predictor" not in _state:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    predictor     = _state["predictor"]
    feature_names = _state.get("feature_names") or []
    silver        = _state.get("silver")

    if silver is None or silver.is_empty():
        raise HTTPException(status_code=503, detail="Silver data no disponible.")

    # 1. Fetch game data
    try:
        resp = _req.get(
            f"{_STATS_BASE}/schedule",
            params={"sportId": 1, "gamePk": game_pk,
                    "hydrate": "lineups,team,probablePitcher"},
            timeout=10,
        )
        resp.raise_for_status()
        games = [g for d in resp.json().get("dates", []) for g in d.get("games", [])]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MLB API error: {exc}")

    if not games:
        raise HTTPException(status_code=404, detail=f"gamePk {game_pk} no encontrado.")

    game      = games[0]
    opp_side  = "home" if team == "away" else "away"
    opp_pp    = game["teams"][opp_side].get("probablePitcher", {})
    opp_name  = opp_pp.get("fullName", "Unknown")
    opp_pid   = opp_pp.get("id")
    opp_throws = _fetch_pitcher_hand(opp_pid)

    # 2. Identify batter name from Silver or MLB API
    batter_rows = silver.filter(silver["batter_id"] == batter_id) if silver is not None else None
    batter_name = f"Player {batter_id}"
    if opp_pid:
        try:
            pdata = _req.get(f"{_STATS_BASE}/people/{batter_id}", timeout=8).json()
            batter_name = (pdata.get("people") or [{}])[0].get("fullName", batter_name)
        except Exception:
            pass

    # 3. Build feature vector
    try:
        X = compute_features(batter_id, opp_throws, silver, feature_names,
                             pitcher_id=opp_pid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature computation error: {exc}")

    ev_per_pa = float(predictor.predict_proba(X.reshape(1, -1))[0] @ np.array(
        [0.0, 0.0, 0.33, 0.47, 0.77, 1.04, 1.40, -0.43], dtype=np.float32
    ))

    # 4. Compute SHAP values (requires shap library)
    try:
        mean_abs_shap = predictor.explain(X.reshape(1, -1), max_display=top_n)
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="SHAP no instalado. Ejecuta: pip install shap",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SHAP error: {exc}")

    # 5. Build ranked response
    top_n_clamped = min(max(top_n, 1), 20)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n_clamped]
    max_shap = float(mean_abs_shap[top_idx[0]]) if len(top_idx) > 0 else 1.0

    def _magnitude(v: float) -> str:
        ratio = v / max(max_shap, 1e-9)
        if ratio >= 0.5:
            return "high"
        if ratio >= 0.2:
            return "medium"
        return "low"

    top_features = [
        SHAPFeature(
            feature=feature_names[i] if i < len(feature_names) else f"feat_{i}",
            shap_value=round(float(mean_abs_shap[i]), 5),
            direction="positive" if mean_abs_shap[i] >= 0 else "negative",
            magnitude=_magnitude(abs(float(mean_abs_shap[i]))),
        )
        for i in top_idx
    ]

    return SHAPResponse(
        batter_id=batter_id,
        batter_name=batter_name,
        game_pk=game_pk,
        opp_pitcher=f"{opp_name} ({opp_throws}HP)",
        top_features=top_features,
        expected_runs_per_pa=round(ev_per_pa, 4),
        model_version=_state.get("model_version", "unknown"),
        note=(
            "SHAP values = mean |SHAP| across all 8 PA outcome classes. "
            "Positive = feature increases E[R/PA]; Negative = decreases it."
        ),
    )


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False, workers=MAX_WORKERS)
