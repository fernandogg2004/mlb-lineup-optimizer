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
