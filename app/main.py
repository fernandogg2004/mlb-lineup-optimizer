"""
MLB Lineup Optimizer — FastAPI Inference Service.

Endpoints:
    POST /v1/predict/at-bat    — single PA outcome prediction, P99 < 50ms
    POST /v1/optimize/lineup   — full GA lineup optimization (async, 3–30s)
    GET  /health               — Kubernetes liveness + readiness probe
    GET  /metrics              — Prometheus text exposition format

Architecture:
    - Models are loaded ONCE at startup via the lifespan context manager.
    - Numba JIT functions are pre-compiled during warmup to eliminate cold-start
      latency from the first request hitting /optimize/lineup.
    - CPU-intensive lineup optimization runs in a ThreadPoolExecutor to avoid
      blocking the asyncio event loop.
    - Prometheus middleware intercepts every request to record latency and
      request counts without touching endpoint handler code.

P99 < 50ms target for /predict/at-bat:
    The hot path is: JSON parse → numpy array → calibrated LightGBM forward pass.
    LightGBM inference on a single row is typically < 5ms; total overhead is < 15ms.
    The 50ms budget is tracked by the predict_at_bat_duration_seconds histogram.
    If P99 exceeds 50ms, check for GIL contention in multi-worker mode.

Multi-worker deployment:
    Run with ``--workers N`` (gunicorn + uvicorn workers). Each worker loads
    its own model copy. Set PROMETHEUS_MULTIPROC_DIR to share metrics across
    workers. Model loading is idempotent (read-only pickle deserialization).
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from app.schemas import (
    ActiveOverridesResponse,
    AtBatPredictRequest,
    AtBatPredictResponse,
    ExperimentSummaryResponse,
    FeatureProposal,
    HealthResponse,
    LineupConfirmRequest,
    LineupConfirmResponse,
    LineupOptimizeRequest,
    LineupOptimizeResponse,
    QualitativeFeedbackRequest,
    QualitativeFeedbackResponse,
    RecordDivergenceRequest,
    RecordDivergenceResponse,
    RecordOutcomeRequest,
    RecordOutcomeResponse,
)
from src.trust.confidence_scoring import build_lineup_scorecard

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_REQUEST_COUNT = Counter(
    "mlb_api_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "http_status"],
)
_PREDICT_LATENCY = Histogram(
    "mlb_predict_at_bat_duration_seconds",
    "Latency of /v1/predict/at-bat endpoint (P99 SLA ≤ 50ms)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.050, 0.075, 0.10, 0.25, 0.50, 1.0],
)
_OPTIMIZE_LATENCY = Histogram(
    "mlb_optimize_lineup_duration_seconds",
    "Latency of /v1/optimize/lineup endpoint",
    buckets=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0],
)
_MODEL_LOAD_SUCCESS = Counter(
    "mlb_model_load_total",
    "Model load events",
    ["model", "status"],
)
_FEEDBACK_SUBMISSIONS = Counter(
    "mlb_feedback_submissions_total",
    "Qualitative feedback submissions",
    ["submission_type", "flag_count"],
)
_OVERRIDES_APPLIED = Counter(
    "mlb_feedback_overrides_applied_total",
    "Qualitative overrides applied to optimization requests",
)
_AB_DIVERGENCES = Counter(
    "mlb_ab_divergences_total",
    "A/B experiment divergences recorded",
    ["divergence_type"],
)
_AB_OUTCOMES = Counter(
    "mlb_ab_outcomes_total",
    "A/B experiment outcomes recorded (post-game)",
)

# ---------------------------------------------------------------------------
# App state — loaded once at startup, shared across all requests
# ---------------------------------------------------------------------------

_OUTCOME_NAMES = [
    "OUT_IN_PLAY", "STRIKEOUT", "WALK_HBP", "SINGLE", "DOUBLE", "TRIPLE", "HOME_RUN"
]
_RUN_VALUES = np.array([0.00, 0.00, 0.33, 0.47, 0.77, 1.04, 1.40], dtype=np.float32)

# League-average 9×7 prob matrix used when opp_lineup_probs is not supplied
_LEAGUE_AVG_PA_PROBS = np.array(
    [[0.28, 0.22, 0.08, 0.15, 0.05, 0.01, 0.03]] * 9,
    dtype=np.float32,
)
# Renormalize to sum-to-1
_LEAGUE_AVG_PA_PROBS /= _LEAGUE_AVG_PA_PROBS.sum(axis=1, keepdims=True)


_LINEUP_CACHE_TTL_SECONDS: float = 86_400  # 24 hours


@dataclass
class _CachedRecommendation:
    """AI recommendation cached after /v1/optimize/lineup for auto-divergence detection."""

    game_id: str
    ai_lineup_ids: list[int]
    ai_expected_runs: float
    cached_at: float = field(default_factory=time.time)
    already_confirmed: bool = False

    def is_expired(self) -> bool:
        return (time.time() - self.cached_at) > _LINEUP_CACHE_TTL_SECONDS


@dataclass
class _AppState:
    at_bat_model: Any = None          # AtBatPredictor
    mc_engine: Any = None             # MonteCarloEngine
    feedback_store: Any = None        # FeedbackStore (lazy — inicializado en lifespan)
    experiment_store: Any = None      # ExperimentStore (lazy — inicializado en lifespan)
    model_version: str = "unknown"
    startup_time: float = field(default_factory=time.time)
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=int(os.environ.get("OPTIMIZER_THREADS", "4")),
            thread_name_prefix="lineup-optimizer",
        )
    )
    # In-memory cache for auto-divergence detection (single-worker; use Redis for multi-worker)
    lineup_cache: dict[str, _CachedRecommendation] = field(default_factory=dict)


_state = _AppState()


# ---------------------------------------------------------------------------
# Startup / shutdown (lifespan)
# ---------------------------------------------------------------------------


def _load_models_sync() -> None:
    """Load all heavy models synchronously (called from executor at startup)."""
    from src.feedback.feedback_store import FeedbackStore
    from src.models.model_at_bat import AtBatPredictor
    from src.simulation.simulation_engine import MonteCarloEngine, warmup_jit

    _state.feedback_store = FeedbackStore.from_env()
    logger.info("feedback_store_ready")

    from src.experiment.experiment_store import ExperimentStore
    _state.experiment_store = ExperimentStore.from_env()
    logger.info("experiment_store_ready")

    model_path = os.environ.get("AT_BAT_MODEL_PATH", "models/at_bat_predictor.pkl")
    if Path(model_path).exists():
        try:
            _state.at_bat_model = AtBatPredictor.load(model_path)
            _state.model_version = os.environ.get("MODEL_VERSION", "local")
            _MODEL_LOAD_SUCCESS.labels(model="at-bat-predictor", status="success").inc()
            logger.info("at_bat_model_loaded", path=model_path, version=_state.model_version)
        except Exception as exc:
            _MODEL_LOAD_SUCCESS.labels(model="at-bat-predictor", status="error").inc()
            logger.error("at_bat_model_load_failed", path=model_path, error=str(exc))
    else:
        logger.warning(
            "at_bat_model_not_found",
            path=model_path,
            hint="Set AT_BAT_MODEL_PATH or run training pipeline first",
        )

    _state.mc_engine = MonteCarloEngine()

    # Pre-compile Numba JIT to eliminate first-request latency spike
    logger.info("numba_jit_warmup_started")
    warmup_jit()
    logger.info("numba_jit_warmup_complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_models_sync)
    logger.info(
        "api_startup_complete",
        model_loaded=_state.at_bat_model is not None,
        model_version=_state.model_version,
    )
    yield
    _state.executor.shutdown(wait=False)
    logger.info("api_shutdown")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MLB Lineup Optimizer API",
    description=(
        "Probabilistic MLB lineup optimization engine. "
        "Monte Carlo simulation + Genetic Algorithm + RAG Tactical Briefing."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Prometheus middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def _prometheus_middleware(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = time.perf_counter() - start

    endpoint = request.url.path
    _REQUEST_COUNT.labels(
        endpoint=endpoint,
        method=request.method,
        http_status=str(response.status_code),
    ).inc()

    # Record per-endpoint latency histograms
    if endpoint == "/v1/predict/at-bat":
        _PREDICT_LATENCY.observe(elapsed)
    elif endpoint == "/v1/optimize/lineup":
        _OPTIMIZE_LATENCY.observe(elapsed)

    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus text exposition endpoint. Scraped by the Prometheus server."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Kubernetes liveness and readiness probe.

    Returns HTTP 200 when ready to serve, HTTP 503 when model is not loaded.
    """
    model_loaded = _state.at_bat_model is not None
    uptime = time.time() - _state.startup_time
    status_str = "healthy" if model_loaded else "degraded"

    if not model_loaded:
        # Kubernetes readiness probe interprets non-2xx as "not ready"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="At-bat model not loaded. Service not ready.",
        )

    return HealthResponse(
        status=status_str,
        at_bat_model_loaded=model_loaded,
        model_version=_state.model_version,
        uptime_seconds=round(uptime, 1),
        details={
            "monte_carlo_engine_ready": _state.mc_engine is not None,
            "optimizer_threads": _state.executor._max_workers,
        },
    )


@app.post("/v1/predict/at-bat", response_model=AtBatPredictResponse, tags=["inference"])
async def predict_at_bat(request: AtBatPredictRequest) -> AtBatPredictResponse:
    """Single plate appearance outcome prediction.

    Hot path: JSON deserialization → numpy array → calibrated LightGBM → JSON.
    Target: P99 latency < 50ms (tracked in mlb_predict_at_bat_duration_seconds).

    Args:
        request: AtBatPredictRequest with player_id, pitcher_id, and feature vector.

    Returns:
        7-class probability distribution + expected run value + server latency.
    """
    if _state.at_bat_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="At-bat model is not loaded.",
        )

    t0 = time.perf_counter()

    features = np.asarray(request.features, dtype=np.float32).reshape(1, -1)
    probs: np.ndarray = _state.at_bat_model.predict_proba(features)[0]  # shape (7,)
    erv = float(np.dot(probs, _RUN_VALUES))
    inference_ms = (time.perf_counter() - t0) * 1_000

    logger.debug(
        "at_bat_prediction",
        player_id=request.player_id,
        pitcher_id=request.pitcher_id,
        erv=round(erv, 4),
        inference_ms=round(inference_ms, 2),
    )

    return AtBatPredictResponse(
        player_id=request.player_id,
        pitcher_id=request.pitcher_id,
        outcome_probabilities={
            name: round(float(p), 6) for name, p in zip(_OUTCOME_NAMES, probs)
        },
        expected_run_value=round(erv, 5),
        model_version=_state.model_version,
        inference_time_ms=round(inference_ms, 3),
    )


def _run_optimization_sync(request: LineupOptimizeRequest) -> LineupOptimizeResponse:
    """Synchronous GA optimization — runs in ThreadPoolExecutor."""
    from src.optimizer.lineup_optimizer import (
        GAConfig,
        GeneticLineupOptimizer,
        PlayerStats,
    )

    start = time.perf_counter()

    # ---------------------------------------------------------------------------
    # Capa 1 — Overrides cualitativos del bench coach
    # ---------------------------------------------------------------------------
    effective_roster = list(request.roster)

    if request.apply_feedback_overrides and _state.feedback_store is not None:
        from src.feedback.override_engine import apply_qualitative_overrides

        game_date = (request.game_id or "")[:10] or __import__("datetime").date.today().isoformat()
        player_ids = [p.player_id for p in request.roster]
        try:
            active_flags = _state.feedback_store.get_active_flags(
                game_date=game_date,
                player_ids=player_ids,
            )
            if active_flags:
                effective_roster = apply_qualitative_overrides(effective_roster, active_flags)
                _OVERRIDES_APPLIED.inc()
                logger.info(
                    "qualitative_overrides_applied",
                    game_date=game_date,
                    n_flags=len(active_flags),
                    affected_players=[
                        f.player_id for f in active_flags
                    ],
                )
        except Exception as exc:
            # Override no-crítico: no bloquear la optimización si falla el store
            logger.warning("qualitative_override_load_failed", error=str(exc))

    players = [
        PlayerStats(
            player_id=p.player_id,
            player_name=p.player_name,
            obp=p.obp,
            woba=p.woba,
            iso=p.iso,
            batter_stand=p.batter_stand,
            prob_vector=np.asarray(p.prob_vector, dtype=np.float32),
        )
        for p in effective_roster
    ]

    lineup_probs = np.stack([p.prob_vector for p in players]).astype(np.float32)

    if request.opp_lineup_probs is not None:
        opp_probs = np.asarray(request.opp_lineup_probs, dtype=np.float32)
    else:
        opp_probs = _LEAGUE_AVG_PA_PROBS.copy()

    ga_config = GAConfig()
    if request.fast_mode:
        ga_config.n_generations = 50
        ga_config.population_size = 100

    optimizer = GeneticLineupOptimizer(
        players=players,
        lineup_probs=lineup_probs,
        simulation_engine=_state.mc_engine,
        opp_lineup_probs=opp_probs,
        config=ga_config,
    )

    result = optimizer.run(
        pitcher_hand=request.rival_pitcher.hand,
        park_factor_hr=request.park_factor_hr,
        park_factor_xb=request.park_factor_xb,
    )

    elapsed = time.perf_counter() - start
    mode = "fast" if request.fast_mode else "full"

    logger.info(
        "lineup_optimization_complete",
        mode=mode,
        expected_runs=round(result.best_expected_runs, 3),
        win_probability=round(result.best_win_probability, 4),
        total_evaluations=result.n_fitness_evaluations,
        elapsed_seconds=round(elapsed, 1),
    )

    game_id = request.game_id or f"game-{time.strftime('%Y%m%d-%H%M%S')}"

    # ---------------------------------------------------------------------------
    # Crisis detection (< 1ms, sin simulación adicional)
    # ---------------------------------------------------------------------------
    crisis_triggered = False
    defense_report = None

    if not request.fast_mode:
        from src.crisis.crisis_detector import CrisisDetector

        try:
            detector = CrisisDetector(
                players=list(request.roster),
                pitcher_hand=request.rival_pitcher.hand,
            )
            trigger_result = detector.detect(result.best_lineup_indices)
            crisis_triggered = trigger_result.triggered

            if crisis_triggered:
                logger.warning(
                    "crisis_triggers_detected",
                    game_id=game_id,
                    trigger_codes=trigger_result.trigger_codes,
                    expected_runs=round(result.best_expected_runs, 3),
                )
        except Exception as exc:
            logger.warning("crisis_detection_failed", error=str(exc))
            trigger_result = None  # type: ignore[assignment]

        # ---------------------------------------------------------------------------
        # Defense report (RAG + Claude, ~10-15s) — solo si hay trigger
        # ---------------------------------------------------------------------------
        if crisis_triggered and trigger_result is not None:
            from src.crisis.defense_report import CrisisReportGenerator

            try:
                generator = CrisisReportGenerator.from_env()
                defense_report = generator.generate(
                    game_id=game_id,
                    players=list(request.roster),
                    best_lineup_indices=result.best_lineup_indices,
                    canonical_lineup_indices=detector.canonical_order(),
                    trigger_result=trigger_result,
                    pitcher=request.rival_pitcher,
                    best_er=result.best_expected_runs,
                    refinement_scores=result.refinement_scores,
                )
            except Exception as exc:
                # No-crítico: el reporte de defensa nunca bloquea la respuesta principal
                logger.warning(
                    "defense_report_generation_failed",
                    game_id=game_id,
                    error=str(exc),
                )

    # ---------------------------------------------------------------------------
    # Scorecard de confianza: solo en modo full
    # ---------------------------------------------------------------------------
    scorecard = None
    if not request.fast_mode and result.refinement_scores:
        try:
            scorecard = build_lineup_scorecard(
                game_id=game_id,
                players=request.roster,
                best_lineup_indices=result.best_lineup_indices,
                best_expected_runs=result.best_expected_runs,
                best_win_probability=result.best_win_probability,
                refinement_scores=result.refinement_scores,
                pitcher=request.rival_pitcher,
                park_factor_hr=request.park_factor_hr,
                park_factor_xb=request.park_factor_xb,
                data_freshness_hours=request.data_freshness_hours,
                crisis_triggered=crisis_triggered,
            )
        except Exception as exc:
            logger.warning("scorecard_build_failed", error=str(exc), game_id=game_id)

    # Cache recommendation for automatic divergence detection on /v1/lineup/confirm
    _state.lineup_cache[game_id] = _CachedRecommendation(
        game_id=game_id,
        ai_lineup_ids=result.best_lineup_ids,
        ai_expected_runs=round(result.best_expected_runs, 4),
    )

    return LineupOptimizeResponse(
        lineup_player_ids=result.best_lineup_ids,
        lineup_player_names=result.best_lineup_names,
        expected_runs=round(result.best_expected_runs, 4),
        win_probability=round(result.best_win_probability, 4),
        optimization_mode=mode,
        total_evaluations=result.n_fitness_evaluations,
        elapsed_seconds=round(elapsed, 2),
        model_version=_state.model_version,
        scorecard=scorecard,
        defense_report=defense_report,
    )


@app.post("/v1/optimize/lineup", response_model=LineupOptimizeResponse, tags=["optimization"])
async def optimize_lineup(request: LineupOptimizeRequest) -> LineupOptimizeResponse:
    """GA-based lineup optimization.

    Runs the full 3-layer optimizer (sabermetric seeding → Genetic Algorithm →
    100k-simulation refinement) or a fast 5k-simulation proxy depending on
    the ``fast_mode`` flag.

    CPU-intensive work runs in a ThreadPoolExecutor to avoid blocking the
    asyncio event loop and starving concurrent /predict/at-bat requests.

    Fast mode (fast_mode=True):  ~3s  — suitable for real-time UI polling
    Full mode (fast_mode=False): ~25s — recommended for final pre-game decision
    """
    if _state.mc_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monte Carlo engine is not initialized.",
        )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _state.executor,
            _run_optimization_sync,
            request,
        )
    except Exception as exc:
        logger.exception("optimization_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {exc}",
        ) from exc

    return result


# ---------------------------------------------------------------------------
# Lineup confirm — auto-divergence detection
# ---------------------------------------------------------------------------


@app.post(
    "/v1/lineup/confirm",
    response_model=LineupConfirmResponse,
    tags=["experiment"],
    summary="Confirmar alineación real y registrar divergencia automáticamente",
)
async def confirm_lineup(
    request: LineupConfirmRequest,
) -> LineupConfirmResponse:
    """Recibe la alineación real del mánager y la compara con la recomendación AI cacheada.

    Si difieren, registra la observación A/B automáticamente sin llamada adicional a
    /v1/experiment/record-divergence. Si son idénticas, no se crea ninguna observación.

    La recomendación AI se cachea en memoria durante 24 horas desde la última llamada
    a /v1/optimize/lineup para ese game_id. En despliegues multi-worker se recomienda
    externalizar el cache a Redis.

    Args:
        request: game_id, game_date, manager_lineup_ids y rationale opcional.

    Returns:
        LineupConfirmResponse indicando si se detectó y registró una divergencia.

    Raises:
        HTTP 404 si no existe una recomendación cacheada para game_id.
        HTTP 503 si el experiment store no está inicializado.
    """
    from src.experiment.experiment_store import classify_divergence

    if _state.experiment_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Experiment store no inicializado.",
        )

    cached = _state.lineup_cache.get(request.game_id)
    if cached is None or cached.is_expired():
        if cached is not None:
            del _state.lineup_cache[request.game_id]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No se encontró una recomendación AI cacheada para game_id='{request.game_id}'. "
                "Llama primero a POST /v1/optimize/lineup con ese game_id."
            ),
        )

    dtype, slots_changed, players_changed = classify_divergence(
        ai_ids=cached.ai_lineup_ids,
        manager_ids=request.manager_lineup_ids,
    )

    if slots_changed == 0 and players_changed == 0:
        logger.info(
            "lineup_confirm_no_divergence",
            game_id=request.game_id,
            already_confirmed=cached.already_confirmed,
        )
        cached.already_confirmed = True
        return LineupConfirmResponse(
            game_id=request.game_id,
            divergence_detected=False,
            observation_id=None,
            divergence_type=None,
            slots_changed=0,
            players_changed=0,
            ai_expected_runs=cached.ai_expected_runs,
            message_es=(
                "La alineación del mánager coincide con la recomendación AI. "
                "No se registró divergencia."
            ),
        )

    # Idempotency: return same observation_id on repeated confirms
    observation_id = hashlib.sha256(request.game_id.encode()).hexdigest()[:16]

    if not cached.already_confirmed:
        try:
            _state.experiment_store.record_divergence(
                game_id=request.game_id,
                game_date=request.game_date,
                ai_lineup_ids=cached.ai_lineup_ids,
                manager_lineup_ids=request.manager_lineup_ids,
                ai_expected_runs=cached.ai_expected_runs,
                manager_rationale=request.manager_rationale or "",
                observation_id=observation_id,
            )
        except Exception as exc:
            logger.error(
                "auto_divergence_record_failed",
                game_id=request.game_id,
                error=str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al registrar la divergencia automática: {exc}",
            ) from exc

        cached.already_confirmed = True
        _AB_DIVERGENCES.labels(divergence_type=dtype).inc()
        logger.info(
            "auto_divergence_recorded",
            game_id=request.game_id,
            observation_id=observation_id,
            divergence_type=dtype,
            slots_changed=slots_changed,
            players_changed=players_changed,
        )

    tipo_map = {
        "order_only": "cambio de orden de bateo",
        "player_swap": "sustitución de jugador",
        "full_override": "alineación diferente",
    }
    tipo_label = tipo_map.get(dtype, dtype)
    mensaje = (
        f"Divergencia registrada automáticamente ({tipo_label}): "
        f"{slots_changed} posición(es) modificada(s). "
        f"Registra el resultado post-partido con POST /v1/experiment/record-outcome."
    )

    return LineupConfirmResponse(
        game_id=request.game_id,
        divergence_detected=True,
        observation_id=observation_id,
        divergence_type=dtype,
        slots_changed=slots_changed,
        players_changed=players_changed,
        ai_expected_runs=cached.ai_expected_runs,
        message_es=mensaje,
    )


# ---------------------------------------------------------------------------
# Feedback endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/v1/feedback/game",
    response_model=QualitativeFeedbackResponse,
    tags=["feedback"],
    summary="Registrar feedback cualitativo del bench coach",
)
async def submit_game_feedback(
    request: QualitativeFeedbackRequest,
) -> QualitativeFeedbackResponse:
    """Recibe y persiste los flags cualitativos del bench coach para un partido.

    Admite dos tipos de envío:
    - **pre_game**: Flags que afectarán la próxima llamada a /v1/optimize/lineup
      cuando se use apply_feedback_overrides=true. Deben enviarse antes del primer pitch.
    - **post_game**: Evaluación de la decisión real + contexto oculto.
      Se usará en la re-ponderación de etiquetas del reentrenamiento semanal.

    Args:
        request: Payload con game_id, game_date, submitted_by, player_flags
                 y campos opcionales de evaluación post-partido.

    Returns:
        Confirmación con submission_id y conteo de flags registrados.
    """
    import uuid

    if _state.feedback_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback store no inicializado.",
        )

    submission_id = str(uuid.uuid4())[:16]

    try:
        n_saved = _state.feedback_store.save(request, submission_id)
    except Exception as exc:
        logger.error(
            "feedback_save_failed",
            submission_id=submission_id,
            game_id=request.game_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al guardar el feedback: {exc}",
        ) from exc

    _FEEDBACK_SUBMISSIONS.labels(
        submission_type=request.submission_type.value,
        flag_count=str(len(request.player_flags)),
    ).inc()

    logger.info(
        "feedback_submission_accepted",
        submission_id=submission_id,
        game_id=request.game_id,
        submitted_by=request.submitted_by,
        submission_type=request.submission_type.value,
        n_flags=n_saved,
    )

    tipo_label = "pre-partido" if request.submission_type.value == "pre_game" else "post-partido"
    mensaje = (
        f"Feedback {tipo_label} registrado correctamente. "
        f"{n_saved} flag(s) guardados para el partido {request.game_id}."
    )
    if request.submission_type.value == "pre_game" and n_saved > 0:
        mensaje += (
            " Los ajustes se aplicarán automáticamente en la próxima optimización "
            "si se usa apply_feedback_overrides=true."
        )

    return QualitativeFeedbackResponse(
        submission_id=submission_id,
        game_id=request.game_id,
        flags_recorded=n_saved,
        submission_type=request.submission_type.value,
        message_es=mensaje,
    )


@app.get(
    "/v1/feedback/overrides",
    response_model=ActiveOverridesResponse,
    tags=["feedback"],
    summary="Consultar overrides cualitativos activos para un partido",
)
async def get_active_overrides(
    game_date: str,
    player_ids: Optional[str] = None,
) -> ActiveOverridesResponse:
    """Devuelve los overrides cualitativos activos (pre_game) para un game_date.

    Útil para que el dashboard muestre al mánager qué jugadores tienen flags
    antes de confirmar la alineación. El resultado incluye el scale_factor
    que se aplicaría a cada jugador si se usa apply_feedback_overrides=true.

    Args:
        game_date: Fecha del partido YYYY-MM-DD.
        player_ids: IDs de jugadores separados por coma para filtrar
                    (opcional, ej. "592450,545361,660271").

    Returns:
        Lista de ActiveOverrideEntry con flag_types, effective_weight y
        performance_impact_pct por jugador.
    """
    import re
    from datetime import datetime as _dt

    from src.feedback.override_engine import compute_override_summary

    if _state.feedback_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback store no inicializado.",
        )

    if not re.match(r"^\d{4}-\d{2}-\d{2}$", game_date):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="game_date debe tener formato YYYY-MM-DD.",
        )

    pid_list: Optional[list[int]] = None
    if player_ids:
        try:
            pid_list = [int(p.strip()) for p in player_ids.split(",") if p.strip()]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="player_ids debe ser una lista de enteros separados por coma.",
            )

    try:
        flags = _state.feedback_store.get_active_flags(
            game_date=game_date,
            player_ids=pid_list,
        )
    except Exception as exc:
        logger.error("get_active_overrides_failed", game_date=game_date, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al cargar los overrides: {exc}",
        ) from exc

    summaries = compute_override_summary(roster=[], flags=flags)

    from app.schemas import ActiveOverrideEntry

    entries = [
        ActiveOverrideEntry(
            player_id=s["player_id"],
            player_name=s["player_name"],
            flag_types=s["flag_types"],
            severity=s["severity"],
            aggregated_weight=s["aggregated_weight"],
            scale_factor=s["scale_factor"],
            performance_impact_pct=s["performance_impact_pct"],
            notes=s["notes"],
        )
        for s in summaries
    ]

    return ActiveOverridesResponse(
        game_date=game_date,
        player_count=len(entries),
        entries=entries,
        generated_at=_dt.now(tz=__import__("datetime").timezone.utc).isoformat(),
    )


@app.get(
    "/v1/feedback/feature-proposals",
    response_model=list[FeatureProposal],
    tags=["feedback"],
    summary="Propuestas de features derivadas de flags recurrentes (Layer 3)",
)
async def get_feature_proposals(
    lookback_days: int = 30,
    threshold: int = 3,
) -> list[FeatureProposal]:
    """Analiza flags recurrentes y devuelve propuestas de features para revisión.

    Genera una propuesta cuando el mismo tipo de flag aparece ≥ threshold veces
    para el mismo jugador en los últimos lookback_days días. Las propuestas
    requieren aprobación humana — no se integran automáticamente al pipeline.

    Args:
        lookback_days: Ventana de análisis (default 30).
        threshold: Mínimo de ocurrencias para generar propuesta (default 3).

    Returns:
        Lista de FeatureProposal ordenada por número de ocurrencias descendente.
    """
    if _state.feedback_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback store no inicializado.",
        )

    try:
        proposals = _state.feedback_store.analyze_recurring_flags(
            lookback_days=lookback_days,
            threshold=threshold,
        )
    except Exception as exc:
        logger.error("feature_proposals_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al analizar flags recurrentes: {exc}",
        ) from exc

    return proposals


# ---------------------------------------------------------------------------
# A/B Experiment endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/v1/experiment/record-divergence",
    response_model=RecordDivergenceResponse,
    tags=["experiment"],
    summary="Registrar divergencia AI vs. decisión real del mánager",
)
async def record_divergence(
    request: RecordDivergenceRequest,
) -> RecordDivergenceResponse:
    """Persiste una observación de divergencia pre-partido.

    Llamar después de recibir la respuesta de /v1/optimize/lineup y antes del
    primer pitch, cuando el mánager decide usar una alineación diferente a la
    recomendada por el sistema.

    La observación queda en status='open' hasta que se registra el resultado
    con POST /v1/experiment/record-outcome.

    Args:
        request: Payload con game_id, ambas alineaciones y E[R] del AI.

    Returns:
        Confirmación con observation_id y clasificación de la divergencia.
    """
    import uuid

    if _state.experiment_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Experiment store no inicializado.",
        )

    observation_id = str(uuid.uuid4())[:16]

    try:
        dtype, slots_changed, players_changed = _state.experiment_store.record_divergence(
            game_id=request.game_id,
            game_date=request.game_date,
            ai_lineup_ids=request.ai_lineup_ids,
            manager_lineup_ids=request.manager_lineup_ids,
            ai_expected_runs=request.ai_expected_runs,
            manager_rationale=request.manager_rationale or "",
            observation_id=observation_id,
        )
    except Exception as exc:
        logger.error(
            "record_divergence_failed",
            game_id=request.game_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al registrar la divergencia: {exc}",
        ) from exc

    _AB_DIVERGENCES.labels(divergence_type=dtype).inc()

    logger.info(
        "divergence_accepted",
        observation_id=observation_id,
        game_id=request.game_id,
        divergence_type=dtype,
        slots_changed=slots_changed,
        players_changed=players_changed,
    )

    tipo_map = {
        "order_only": "cambio de orden de bateo",
        "player_swap": "sustitución de jugador",
        "full_override": "alineación diferente",
    }
    tipo_label = tipo_map.get(dtype, dtype)
    mensaje = (
        f"Divergencia registrada ({tipo_label}): {slots_changed} posición(es) modificada(s). "
        f"Registra el resultado post-partido con POST /v1/experiment/record-outcome."
    )

    return RecordDivergenceResponse(
        observation_id=observation_id,
        game_id=request.game_id,
        divergence_type=dtype,
        slots_changed=slots_changed,
        players_changed=players_changed,
        message_es=mensaje,
    )


@app.post(
    "/v1/experiment/record-outcome",
    response_model=RecordOutcomeResponse,
    tags=["experiment"],
    summary="Registrar el resultado real post-partido para cerrar la observación A/B",
)
async def record_outcome(
    request: RecordOutcomeRequest,
) -> RecordOutcomeResponse:
    """Cierra una observación A/B con el resultado real del partido.

    Debe llamarse después del partido, con las carreras reales anotadas
    por el equipo con la alineación que eligió el mánager.

    El sistema busca automáticamente las observaciones abiertas de los últimos
    14 días que coincidan con el game_id.

    Args:
        request: game_id, actual_runs_scored e innings_played.

    Returns:
        Confirmación con el diferencial E[R] AI vs. resultado real.
    """
    if _state.experiment_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Experiment store no inicializado.",
        )

    try:
        observations_updated, ai_expected_runs = _state.experiment_store.record_outcome(
            game_id=request.game_id,
            actual_runs_scored=request.actual_runs_scored,
            innings_played=request.innings_played,
        )
    except Exception as exc:
        logger.error(
            "record_outcome_failed",
            game_id=request.game_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al registrar el resultado: {exc}",
        ) from exc

    if observations_updated == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No se encontraron observaciones abiertas para game_id='{request.game_id}'. "
                "Asegúrate de haber llamado a POST /v1/experiment/record-divergence antes del partido."
            ),
        )

    _AB_OUTCOMES.inc()

    er_differential = round(request.actual_runs_scored - ai_expected_runs, 3)
    signo = "+" if er_differential >= 0 else ""
    mensaje = (
        f"Resultado registrado: {request.actual_runs_scored} carrera(s). "
        f"El AI esperaba {round(ai_expected_runs, 2)} E[C] → diferencial: {signo}{er_differential}."
    )

    logger.info(
        "outcome_accepted",
        game_id=request.game_id,
        actual_runs_scored=request.actual_runs_scored,
        ai_expected_runs=round(ai_expected_runs, 3),
        er_differential=er_differential,
        observations_updated=observations_updated,
    )

    return RecordOutcomeResponse(
        game_id=request.game_id,
        observations_updated=observations_updated,
        ai_expected_runs=round(ai_expected_runs, 3),
        actual_runs_scored=request.actual_runs_scored,
        er_differential=er_differential,
        message_es=mensaje,
    )


@app.get(
    "/v1/experiment/summary",
    response_model=ExperimentSummaryResponse,
    tags=["experiment"],
    summary="Estadísticas acumuladas del experimento A/B Manager vs. AI",
)
async def experiment_summary(
    lookback_days: int = 90,
) -> ExperimentSummaryResponse:
    """Devuelve las estadísticas del experimento A/B sobre partidos cerrados.

    Agrega todas las observaciones con resultado registrado en los últimos
    lookback_days días y calcula si el mánager tiende a superar o quedar por
    debajo de la expectativa del sistema.

    El análisis es significativo cuando total_closed_games >= 15.

    Args:
        lookback_days: Ventana de análisis en días (default 90).

    Returns:
        ExperimentSummaryResponse con estadísticas e interpretación en español.
    """
    if _state.experiment_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Experiment store no inicializado.",
        )

    if lookback_days < 1 or lookback_days > 365:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="lookback_days debe estar entre 1 y 365.",
        )

    try:
        summary = _state.experiment_store.get_summary(lookback_days=lookback_days)
    except Exception as exc:
        logger.error("experiment_summary_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al calcular las estadísticas: {exc}",
        ) from exc

    return ExperimentSummaryResponse(
        lookback_days=lookback_days,
        total_closed_games=summary["total_closed_games"],
        by_divergence_type=summary["by_divergence_type"],
        avg_ai_expected_runs=summary["avg_ai_expected_runs"],
        avg_actual_runs_scored=summary["avg_actual_runs_scored"],
        manager_advantage_runs=summary["manager_advantage_runs"],
        sample_too_small=summary["sample_too_small"],
        interpretation_es=summary["interpretation_es"],
    )


# ---------------------------------------------------------------------------
# Dashboard endpoints  (Streamlit ← FastAPI ← PostgreSQL / MLB Stats API)
# ---------------------------------------------------------------------------

_PG_HOST = os.getenv("POSTGRES_HOST", "")
_PG_PORT = os.getenv("POSTGRES_PORT", "5432")
_PG_DB   = os.getenv("POSTGRES_DB", "")
_PG_USER = os.getenv("POSTGRES_USER", "")
_PG_PASS = os.getenv("POSTGRES_PASSWORD", "")

_MLB_STATS  = "https://statsapi.mlb.com/api/v1"
_MLB_V11    = "https://statsapi.mlb.com/api/v1.1"


def _pg_conn():
    import psycopg2
    return psycopg2.connect(
        host=_PG_HOST, port=int(_PG_PORT),
        dbname=_PG_DB, user=_PG_USER, password=_PG_PASS,
        connect_timeout=5,
    )


def _stats_get(path: str, base: str = _MLB_STATS, params: dict | None = None) -> dict:
    import requests as _r
    for attempt in range(3):
        try:
            r = _r.get(f"{base}{path}", params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"MLB API exhausted retries: {base}{path}")


def _players_bulk(player_ids: list[int]) -> dict[int, dict]:
    """Career stats + position for multiple player IDs in a single MLB API call."""
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
        name       = p.get("fullName", f"Player {pid}")
        hand       = p.get("batSide", {}).get("code", "R")
        pos        = p.get("primaryPosition", {}).get("abbreviation", "—")
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
            "player_id":  pid,  "name": name, "pos": pos,
            "hand":       hand, "pitch_hand": pitch_hand,
            "avg":        round(avg, 3),
            "ops":        round(obp + slg, 3),
            "woba":       round((obp * 1.2 + slg * 0.7) / 2, 3),
            "obp":        round(obp, 3),
            "iso":        round(max(slg - avg, 0.0), 3),
        }
    return out


def _approx_prob_vector(avg: float, obp: float, slg: float) -> list[float]:
    """7-class PA outcome prob vector from career slash line (ported from DAG)."""
    pa_adj = 0.97
    p_bb  = max(round((obp - avg) * pa_adj, 4), 0.04)
    p_hr  = max(round((slg - avg) / 4.2, 4), 0.01)
    p_3b  = 0.005
    p_2b  = max(round((slg - avg - p_hr * 3 - p_3b * 2) * 0.35, 4), 0.02)
    p_1b  = max(round(avg - p_hr - p_3b - p_2b, 4), 0.08)
    p_k   = max(round(0.22 - (obp - 0.310) * 0.6, 4), 0.08)
    p_out = max(round(1.0 - p_bb - p_hr - p_3b - p_2b - p_1b - p_k, 4), 0.20)
    raw   = [p_out, p_k, p_bb, p_1b, p_2b, p_3b, p_hr]
    total = sum(raw)
    return [round(x / total, 4) for x in raw]


def _generate_lineup_explanation(
    lineup: list[dict],
    pitcher_name: str,
    pitcher_hand: str,
    pitcher_era: float,
    away_name: str,
    home_name: str,
    optimization_mode: str = "on_demand_fast",
) -> str:
    """Generate tactical lineup explanation via Claude, with rich template fallback."""
    import os

    lineup_text = "\n".join(
        f"  #{p['order']} {p['name']} ({p.get('pos','?')}, {p.get('hand','?')}HB) "
        f"— OBP {p.get('obp',0):.3f} · wOBA {p.get('woba',0):.3f} · ISO {p.get('iso',0):.3f}"
        for p in lineup
    )
    source_note = (
        "Predicción on-demand (fast mode, 5k sims). La predicción definitiva llega a T-2h via Airflow."
        if optimization_mode == "on_demand_fast"
        else "Predicción generada por Airflow (lineup oficial confirmado)."
    )

    system_prompt = (
        f"Eres el analista táctico del equipo {home_name}. "
        f"El oponente de hoy es {away_name} con el lanzador abridor "
        f"{pitcher_name} ({pitcher_hand}HP, ERA {pitcher_era:.2f}).\n\n"
        f"El motor Monte Carlo generó el siguiente lineup óptimo:\n{lineup_text}\n\n"
        f"Redacta un briefing táctico pre-partido en Markdown (máx 350 palabras) que explique:\n"
        f"1. Por qué cada slot del orden de bateo está asignado a ese jugador\n"
        f"2. Cómo el perfil de {pitcher_name} afecta las decisiones (mano, ERA, pitch mix estimado)\n"
        f"3. Una alerta de platoon o sustitución de banca si aplica\n"
        f"Sé conciso, usa negrita para nombres de jugadores y métricas clave."
    )

    # Anthropic Claude
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                messages=[{"role": "user", "content": system_prompt}],
            )
            return f"## Análisis Táctico — {away_name} @ {home_name}\n\n{msg.content[0].text}\n\n---\n*{source_note}*"
        except Exception as exc:
            logger.warning("claude_explanation_failed", error=str(exc))

    # Groq / cualquier API compatible con OpenAI (gratis)
    groq_key  = os.environ.get("GROQ_API_KEY", "")
    oai_key   = os.environ.get("OPENAI_API_KEY", "")
    oai_base  = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_key   = groq_key or oai_key
    llm_base  = "https://api.groq.com/openai/v1" if groq_key else oai_base
    llm_model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant" if groq_key else "gpt-4o-mini")
    if llm_key:
        try:
            import httpx
            resp = httpx.post(
                f"{llm_base}/chat/completions",
                headers={"Authorization": f"Bearer {llm_key}", "Content-Type": "application/json"},
                json={"model": llm_model, "messages": [{"role": "user", "content": system_prompt}],
                      "max_tokens": 600, "temperature": 0.7},
                timeout=30.0,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return f"## Análisis Táctico — {away_name} @ {home_name}\n\n{text}\n\n---\n*{source_note}*"
        except Exception as exc:
            logger.warning("llm_explanation_failed", provider=llm_base, error=str(exc))

    # Rich template fallback (no API key)
    top3 = sorted(lineup, key=lambda p: p.get("obp", 0), reverse=True)[:3]
    power = sorted(lineup, key=lambda p: p.get("iso", 0), reverse=True)[:2]
    lhb   = [p["name"].split()[-1] for p in lineup if p.get("hand") == "L"]
    rhb   = [p["name"].split()[-1] for p in lineup if p.get("hand") == "R"]

    top3_str  = ", ".join(f"**{p['name']}** ({p.get('obp', 0):.3f})" for p in top3)
    power_str = ", ".join(f"**{p['name']}** (ISO {p.get('iso', 0):.3f})" for p in power)

    platoon = ""
    if pitcher_hand == "R" and lhb:
        platoon = f"\n\n> ⚔️ **Ventaja de platoon:** {', '.join(lhb[:3])} (zurdos) tienen ventaja estadística vs {pitcher_hand}HP."
    elif pitcher_hand == "L" and rhb:
        platoon = f"\n\n> ⚔️ **Ventaja de platoon:** {', '.join(rhb[:3])} (diestros) tienen ventaja estadística vs {pitcher_hand}HP."

    return (
        f"## Análisis Táctico — {away_name} @ {home_name}\n\n"
        f"**Lanzador rival:** {pitcher_name} ({pitcher_hand}HP) · ERA {pitcher_era:.2f}\n\n"
        f"### Orden de Bateo — Lógica del Modelo\n\n"
        f"El modelo optimizó el orden minimizando outs consecutivos y maximizando "
        f"situaciones de runners en base para los bateadores de mayor potencia.\n\n"
        f"**Generadores de base (OBP líderes):** {top3_str}\n\n"
        f"**Bateadores de poder (ISO líderes):** {power_str}"
        f"{platoon}\n\n"
        f"---\n*{source_note}*\n\n"
        f"> 💡 Agrega `GROQ_API_KEY` (gratis en console.groq.com) para análisis táctico generado por IA."
    )


def _on_demand_optimize(game_pk: int, team: str = "home") -> dict:
    """Fast on-demand optimization when Airflow hasn't written to DB yet.

    `team` controls which side to optimize: "home" (default) or "away".
    The rival pitcher is always the opponent's probable starter.
    """
    from app.schemas import LineupOptimizeRequest, PlayerRosterEntry, RivalPitcherRequest

    # 1. Fetch game from MLB API
    data = _stats_get("/schedule", params={
        "sportId": 1, "gamePk": game_pk,
        "hydrate": "team,probablePitcher,status",
    })
    game_info: dict | None = None
    for de in data.get("dates", []):
        for g in de.get("games", []):
            if g.get("gamePk") == game_pk:
                game_info = g
                break

    if not game_info:
        raise HTTPException(status_code=404, detail=f"game_pk={game_pk} not found in MLB schedule")

    home         = game_info["teams"]["home"]
    away         = game_info["teams"]["away"]
    home_team_id = home["team"]["id"]
    home_name    = home["team"]["name"]
    away_team_id = away["team"]["id"]
    away_name    = away["team"]["name"]

    # Select batting team and opponent pitcher based on `team` param
    if team == "away":
        batting_team_id = away_team_id
        opp_pp  = home.get("probablePitcher", {})
    else:
        batting_team_id = home_team_id
        opp_pp  = away.get("probablePitcher", {})
    opp_pid = opp_pp.get("id")

    # 2. Build projected batting order from 40-Man roster
    pos_priority = {"DH": 0, "RF": 1, "LF": 2, "CF": 3,
                    "1B": 4, "3B": 5, "SS": 6, "2B": 7, "C": 8}
    roster_raw = _stats_get(
        f"/teams/{batting_team_id}/roster/40Man",
        params={"hydrate": "person(stats(type=career,group=hitting))"},
    ).get("roster", [])

    batters: list[dict] = []
    for entry in roster_raw:
        pos = entry.get("position", {}).get("abbreviation", "")
        if pos in ("SP", "RP", "P"):
            continue
        person = entry.get("person", {})
        pid = person.get("id")
        if not pid:
            continue
        avg, obp, slg = 0.248, 0.318, 0.398
        for s in person.get("stats", []):
            if s.get("type", {}).get("displayName") == "career":
                sp = (s.get("splits") or [{}])[0].get("stat", {})
                avg = float(sp.get("avg", avg) or avg)
                obp = float(sp.get("obp", obp) or obp)
                slg = float(sp.get("slg", slg) or slg)
                break
        iso  = round(max(slg - avg, 0.0), 3)
        woba = round((obp * 1.2 + slg * 0.7) / 2, 3)
        batters.append({
            "player_id":   int(pid),
            "player_name": person.get("fullName", f"Player {pid}"),
            "batter_stand": person.get("batSide", {}).get("code", "R"),
            "obp": round(obp, 3), "woba": woba, "iso": iso,
            "prob_vector": _approx_prob_vector(avg, obp, slg),
            "_priority": pos_priority.get(pos, 9),
        })
    batters.sort(key=lambda x: x["_priority"])

    if len(batters) < 9:
        raise HTTPException(status_code=503, detail="Insufficient roster data for on-demand optimization")

    roster_entries = [{k: v for k, v in b.items() if not k.startswith("_")}
                      for b in batters[:9]]

    # 3. Pitcher info
    pitcher_name = opp_pp.get("fullName", f"Pitcher {opp_pid}")
    pitcher_hand, pitcher_era = "R", 4.00
    if opp_pid:
        try:
            pdata = _stats_get(f"/people/{opp_pid}",
                               params={"hydrate": "stats(group=[pitching],type=[season])"})
            pp = (pdata.get("people") or [{}])[0]
            pitcher_name = pp.get("fullName", pitcher_name)
            pitcher_hand = pp.get("pitchHand", {}).get("code", "R")
            for s in pp.get("stats", []):
                if s.get("type", {}).get("displayName") == "season":
                    era_str = (s.get("splits") or [{}])[0].get("stat", {}).get("era", "4.00")
                    try:
                        pitcher_era = float(era_str)
                    except (ValueError, TypeError):
                        pass
                    break
        except Exception:
            pass

    # 4. Optimize (fast mode — 5k sims)
    req = LineupOptimizeRequest(
        roster=[PlayerRosterEntry(**p) for p in roster_entries],
        rival_pitcher=RivalPitcherRequest(
            pitcher_id=int(opp_pid) if opp_pid else 0,
            pitcher_name=pitcher_name,
            hand=pitcher_hand,
            era=pitcher_era,
            pitch_mix={"FF": 0.55, "SL": 0.20, "CH": 0.15, "CB": 0.10},
        ),
        fast_mode=True,
        apply_feedback_overrides=False,
    )
    result = _run_optimization_sync(req)

    # 5. Enrich lineup with display stats from bulk lookup
    ordered_ids  = [p["player_id"] for p in roster_entries]
    bench_ids    = [b["player_id"] for b in batters[9:12]]
    pinfo        = _players_bulk(ordered_ids + bench_ids)
    _def         = lambda pid: {"player_id": pid, "name": f"Player {pid}", "pos": "—",
                                "hand": "R", "avg": .248, "ops": .716,
                                "woba": .315, "obp": .318, "iso": .150}
    lineup = [{"order": i, **{k: v for k, v in pinfo.get(pid, _def(pid)).items()
                              if k != "pitch_hand"}}
              for i, pid in enumerate(ordered_ids, 1)]
    bench  = [{k: v for k, v in pinfo.get(pid, _def(pid)).items() if k != "pitch_hand"}
              for pid in bench_ids if pid]

    explanation = _generate_lineup_explanation(
        lineup, pitcher_name, pitcher_hand, pitcher_era,
        away_name, home_name, optimization_mode="on_demand_fast",
    )

    sim_count = 100_000 if result.optimization_mode == "full" else 5_000
    return {
        "game_pk":           game_pk,
        "team":              team,
        "expected_runs":     round(result.expected_runs, 3),
        "win_probability":   round(result.win_probability, 4),
        "model_confidence":  0.65,
        "optimization_mode": "on_demand_fast",
        "model_version":     _state.model_version,
        "total_simulations": sim_count,
        "elapsed_seconds":   round(result.elapsed_seconds, 1),
        "lineup":            lineup,
        "bench":             bench,
        "rag_explanation":   explanation,
    }


@app.get("/v1/games/today", tags=["dashboard"])
async def games_today() -> dict:
    """Partidos MLB programados para hoy con probable pitchers."""
    from datetime import date as _date, datetime as _dt
    from zoneinfo import ZoneInfo

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
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            hp   = home.get("probablePitcher", {})
            ap   = away.get("probablePitcher", {})

            game_time_et = "TBD"
            gd_str = g.get("gameDate", "")
            if gd_str:
                try:
                    dt_utc = _dt.fromisoformat(gd_str.replace("Z", "+00:00"))
                    dt_et  = dt_utc.astimezone(ZoneInfo("America/New_York"))
                    game_time_et = dt_et.strftime("%-I:%M ET")
                except Exception:
                    game_time_et = gd_str[11:16] + " UTC"

            if hp.get("id"): pitcher_ids.append(int(hp["id"]))
            if ap.get("id"): pitcher_ids.append(int(ap["id"]))

            raw_games.append({
                "game_pk":          int(game_pk),
                "home_team":        home.get("team", {}).get("abbreviation", ""),
                "away_team":        away.get("team", {}).get("abbreviation", ""),
                "home_name":        home.get("team", {}).get("name", ""),
                "away_name":        away.get("team", {}).get("name", ""),
                "game_time":        game_time_et,
                "venue":            g.get("venue", {}).get("name", ""),
                "game_date":        today,
                "home_pitcher":     hp.get("fullName", "TBD"),
                "away_pitcher":     ap.get("fullName", "TBD"),
                "_home_pitcher_id": hp.get("id"),
                "_away_pitcher_id": ap.get("id"),
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
async def get_optimize(game_pk: int, team: str = "home") -> dict:
    """Lineup óptimo del modelo para un partido.

    `team` query param selects which side to optimize: "home" (default) or "away".
    The DB only stores home-team predictions from Airflow; away always uses on-demand.
    """
    loop = asyncio.get_event_loop()

    # Away team: DB has no away-side predictions — go straight to on-demand
    if team == "away":
        return await loop.run_in_executor(None, _on_demand_optimize, game_pk, "away")

    if not _PG_HOST:
        raise HTTPException(status_code=503, detail="PostgreSQL no configurado.")

    try:
        conn = _pg_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT ai_recommended_lineup, ai_expected_runs, win_probability,
                   optimization_mode, model_version, predicted_at,
                   home_batting_order, home_team_name, away_team_name, lineup_source
            FROM gameday_predictions WHERE game_pk = %s
        """, (game_pk,))
        row = cur.fetchone()
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB error: {exc}")

    if not row:
        # Airflow hasn't written a prediction yet — run fast on-demand optimization
        return await loop.run_in_executor(None, _on_demand_optimize, game_pk, "home")

    (ai_lineup, ai_er, win_prob, opt_mode, model_ver, predicted_at,
     home_order, home_name, away_name, lineup_source) = row

    ordered_ids = (ai_lineup or home_order or [])[:9]
    all_ids     = list(dict.fromkeys(ordered_ids + (home_order or [])[:12]))
    pinfo       = _players_bulk([p for p in all_ids if p])

    _default = lambda pid: {
        "player_id": pid, "name": f"Player {pid}", "pos": "—",
        "hand": "R", "avg": .248, "ops": .716, "woba": .315, "obp": .318, "iso": .150,
    }

    lineup = []
    for i, pid in enumerate(ordered_ids, 1):
        p = pinfo.get(pid, _default(pid))
        lineup.append({"order": i, **{k: v for k, v in p.items() if k != "pitch_hand"}})

    bench_ids = [pid for pid in (home_order or []) if pid not in set(ordered_ids)][:3]
    bench = [{k: v for k, v in pinfo.get(pid, _default(pid)).items() if k != "pitch_hand"}
             for pid in bench_ids if pid]

    rag = _generate_lineup_explanation(
        lineup, "Lanzador rival", "R", 4.00,
        away_name, home_name, optimization_mode=lineup_source or "airflow",
    )

    return {
        "game_pk":           game_pk,
        "expected_runs":     round(ai_er or 0.0, 3),
        "win_probability":   round(win_prob or 0.5, 4),
        "model_confidence":  0.80,
        "optimization_mode": opt_mode or "fast",
        "model_version":     model_ver or _state.model_version,
        "total_simulations": 10_000,
        "elapsed_seconds":   0.0,
        "lineup":            lineup,
        "bench":             bench,
        "rag_explanation":   rag,
    }


@app.get("/v1/games/history", tags=["dashboard"])
async def games_history(date: str) -> dict:
    """Partidos históricos con resultados para una fecha dada (PostgreSQL o MLB API)."""
    # Always fetch the full game list from MLB API so all games appear regardless
    # of how many DB predictions Airflow has written for that date.
    try:
        data = _stats_get("/schedule", params={
            "sportId": 1, "date": date, "hydrate": "team,linescore,status",
        })
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MLB API error: {exc}")

    games: list[dict] = []
    for de in data.get("dates", []):
        for g in de.get("games", []):
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

    # Enrich with DB pitcher data where available without restricting the list.
    # Previously the endpoint returned ONLY DB rows when any existed, causing
    # all games without a DB prediction to disappear from the UI.
    if _PG_HOST and games:
        try:
            conn = _pg_conn()
            cur  = conn.cursor()
            cur.execute("""
                SELECT game_pk, home_team_name, away_team_name,
                       actual_home_runs, actual_away_runs,
                       home_starting_pitcher_id, away_starting_pitcher_id
                FROM gameday_predictions WHERE game_date = %s
            """, (date,))
            db_rows = {r[0]: r for r in cur.fetchall()}
            conn.close()

            pitcher_ids = [r[5] for r in db_rows.values() if r[5]] + \
                          [r[6] for r in db_rows.values() if r[6]]
            pinfo = _players_bulk(pitcher_ids) if pitcher_ids else {}

            for game in games:
                row = db_rows.get(game["game_pk"])
                if row:
                    _, home_name, away_name, db_hr, db_ar, hp_id, ap_id = row
                    # Prefer DB run totals when Airflow stored them
                    if db_hr is not None:
                        game["home_runs_actual"] = db_hr
                        game["away_runs_actual"] = db_ar or 0
                        game["final_score"] = f"{db_hr}-{db_ar or 0}"
                    game["home_pitcher"] = pinfo.get(hp_id, {}).get("name", "") if hp_id else ""
                    game["away_pitcher"] = pinfo.get(ap_id, {}).get("name", "") if ap_id else ""
        except Exception as exc:
            logger.warning("games_history_db_error", error=str(exc))

    return {"games": games, "date": date, "total": len(games)}


@app.get("/v1/report/{game_pk}", tags=["dashboard"])
async def get_report(game_pk: int, date: str | None = None) -> dict:
    """Reporte post-partido completo: lineups, métricas y análisis Markdown."""
    import math

    if not _PG_HOST:
        raise HTTPException(status_code=503, detail="PostgreSQL no configurado.")

    try:
        conn = _pg_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT game_pk, game_date, home_team_name, away_team_name,
                   ai_recommended_lineup, home_batting_order,
                   ai_expected_runs, win_probability, model_version,
                   actual_home_runs, actual_away_runs, delta_er, game_state
            FROM gameday_predictions WHERE game_pk = %s
        """, (game_pk,))
        row = cur.fetchone()
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB error: {exc}")

    if not row:
        # No DB prediction — build a basic report from MLB live feed
        try:
            feed      = _stats_get(f"/game/{game_pk}/feed/live", base=_MLB_V11)
            gdata     = feed.get("gameData", {})
            home_info = gdata.get("teams", {}).get("home", {})
            away_info = gdata.get("teams", {}).get("away", {})
            home_name = home_info.get("name", "Home")
            away_name = away_info.get("name", "Away")
            gdate     = gdata.get("datetime", {}).get("officialDate", date or "")
            ldata     = feed.get("liveData", {})
            ls_teams  = ldata.get("linescore", {}).get("teams", {})
            hr_actual = ls_teams.get("home", {}).get("runs", 0) or 0
            ar_actual = ls_teams.get("away", {}).get("runs", 0) or 0
            home_box  = ldata.get("boxscore", {}).get("teams", {}).get("home", {})
            bs_order  = home_box.get("battingOrder", [])
            bs_players = home_box.get("players", {})
            actual_lineup = []
            for i, pid in enumerate(bs_order[:9], 1):
                pbox = bs_players.get(f"ID{pid}", {})
                st   = pbox.get("stats", {}).get("batting", {})
                h, ab  = st.get("hits", 0) or 0, st.get("atBats", 0) or 0
                hr_s, bb = st.get("homeRuns", 0) or 0, st.get("baseOnBalls", 0) or 0
                rbi, so  = st.get("rbi", 0) or 0, st.get("strikeOuts", 0) or 0
                parts = [f"{h}-for-{ab}"]
                if hr_s:          parts.append(f"{hr_s} HR")
                elif rbi:         parts.append(f"{rbi} RBI")
                if bb:            parts.append(f"{bb} BB")
                if so and not h:  parts.append(f"{so}K")
                actual_lineup.append({
                    "order":  i,
                    "name":   pbox.get("person", {}).get("fullName", f"Player {pid}"),
                    "pos":    pbox.get("position", {}).get("abbreviation", "—"),
                    "result": ", ".join(parts),
                })
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Sin datos para game_pk={game_pk}: {exc}")

        game_result = hr_actual > ar_actual
        report_md = (
            f"## Post-Game Analysis — {home_name} {hr_actual}, {away_name} {ar_actual}\n"
            f"**Fecha:** {gdate}\n\n"
            f"> ℹ️ Este partido no fue procesado por Airflow — no hay predicción de modelo disponible.\n\n"
            f"---\n\n"
            f"### Resultado\n"
            f"{'✅ Victoria' if game_result else '❌ Derrota'} del equipo local "
            f"({home_name} **{hr_actual}**, {away_name} **{ar_actual}**)\n"
        )
        return {
            "game_pk":                   game_pk,
            "game_date":                 gdate,
            "matchup":                   f"{away_name} @ {home_name}  ·  {hr_actual}–{ar_actual}",
            "game_result":               game_result,
            "proposed_lineup":           [],
            "actual_lineup":             actual_lineup,
            "projected_runs":            0.0,
            "actual_home_runs":          hr_actual,
            "actual_away_runs":          ar_actual,
            "win_probability_projected": 0.5,
            "model_log_loss":            0.0,
            "model_version":             "N/A",
            "report_markdown":           report_md,
        }

    (gpk, game_date, home_name, away_name,
     ai_lineup, home_order,
     ai_er, win_prob, model_ver,
     actual_home_runs, actual_away_runs, _delta_er, game_state) = row

    all_ids     = list(dict.fromkeys((ai_lineup or []) + (home_order or [])))
    player_info = _players_bulk([p for p in all_ids if p])

    # Proposed lineup (model recommendation)
    proposed_lineup = [
        {"order": i, "name": player_info.get(pid, {}).get("name", f"Player {pid}"),
         "pos": player_info.get(pid, {}).get("pos", "—")}
        for i, pid in enumerate((ai_lineup or [])[:9], 1)
    ]

    # Actual lineup: try live-feed boxscore, fall back to stored batting order
    actual_lineup = []
    try:
        feed     = _stats_get(f"/game/{game_pk}/feed/live", base=_MLB_V11)
        home_box = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get("home", {})
        bs_order  = home_box.get("battingOrder", []) or home_order or []
        bs_players = home_box.get("players", {})

        for i, pid in enumerate(bs_order[:9], 1):
            pbox = bs_players.get(f"ID{pid}", {})
            st   = pbox.get("stats", {}).get("batting", {})
            h, ab = st.get("hits", 0) or 0, st.get("atBats", 0) or 0
            hr, bb = st.get("homeRuns", 0) or 0, st.get("baseOnBalls", 0) or 0
            rbi, so = st.get("rbi", 0) or 0, st.get("strikeOuts", 0) or 0
            parts = [f"{h}-for-{ab}"]
            if hr:           parts.append(f"{hr} HR")
            elif rbi:        parts.append(f"{rbi} RBI")
            if bb:           parts.append(f"{bb} BB")
            if so and not h: parts.append(f"{so}K")
            actual_lineup.append({
                "order":  i,
                "name":   pbox.get("person", {}).get("fullName", player_info.get(pid, {}).get("name", f"Player {pid}")),
                "pos":    pbox.get("position", {}).get("abbreviation", "—"),
                "result": ", ".join(parts),
            })
    except Exception:
        actual_lineup = [
            {"order": i, "name": player_info.get(pid, {}).get("name", f"Player {pid}"),
             "pos": player_info.get(pid, {}).get("pos", "—"), "result": "—"}
            for i, pid in enumerate((home_order or [])[:9], 1)
        ]

    hr_actual   = actual_home_runs or 0
    ar_actual   = actual_away_runs or 0
    game_result = hr_actual > ar_actual
    prob        = max(0.001, min(0.999, win_prob or 0.5))
    log_loss    = round(-math.log(prob if game_result else (1.0 - prob)), 3)
    delta       = hr_actual - (ai_er or 0.0)
    sign        = "+" if delta >= 0 else ""

    report_md = (
        f"## Post-Game Analysis — {home_name} {hr_actual}, {away_name} {ar_actual}\n"
        f"**Fecha:** {game_date} &nbsp;|&nbsp; "
        f"**Modelo:** {model_ver or 'unknown'} &nbsp;|&nbsp; "
        f"**Log-Loss:** {log_loss:.3f}\n\n---\n\n"
        f"### Evaluación del Modelo\n\n"
        f"| Métrica | Valor |\n|---|---|\n"
        f"| E[R] proyectado | {ai_er or 0:.2f} |\n"
        f"| Carreras reales | **{hr_actual}** |\n"
        f"| Δ E[R] | {sign}{delta:.2f} {'✅' if delta >= 0 else '⚠️'} |\n"
        f"| Win Prob. proyectada | {prob*100:.1f}% |\n"
        f"| Log-Loss del partido | {log_loss:.3f} "
        f"{'✅ bueno' if log_loss < 0.5 else '⚠️ revisar'} |\n\n"
        f"### Estado del partido\n"
        f"**{game_state or 'N/A'}** — "
        f"{'✅ Victoria' if game_result else '❌ Derrota'} del equipo local.\n"
    )

    return {
        "game_pk":                   gpk,
        "game_date":                 str(game_date),
        "matchup":                   f"{away_name} @ {home_name}  ·  {hr_actual}–{ar_actual}",
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
    }


# ---------------------------------------------------------------------------
# Entry point (local dev only — use gunicorn in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
        loop="uvloop",
        http="httptools",
        log_level="info",
        reload=False,
    )
