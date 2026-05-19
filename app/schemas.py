"""
Pydantic v2 request / response schemas for the MLB Lineup Optimizer API.

All schemas use strict typing and field-level validation. API contract versioning
is handled via the /v1/ URL prefix defined in main.py.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Feedback Cualitativo — Enums compartidos entre API y src/feedback/
# ---------------------------------------------------------------------------


class FlagType(str, Enum):
    """Tipos de observación cualitativa del bench coach."""

    MENTAL_DISTRACTION = "mental_distraction"    # jugador distraído/sin foco
    PHYSICAL_ILLNESS = "physical_illness"        # malestar físico, gripe, etc.
    MECHANICAL_ISSUE = "mechanical_issue"        # cambio en mecánica de bateo
    FATIGUE_TRAVEL = "fatigue_travel"            # viaje tardío, jet lag
    PERSONAL_SITUATION = "personal_situation"    # situación personal fuera del campo
    HOT_STREAK_FEEL = "hot_streak_feel"          # jugador en racha — subirle en el orden
    COLD_STREAK_FEEL = "cold_streak_feel"        # jugador frío — bajarle en el orden


class FlagSeverity(str, Enum):
    """Nivel de confianza del coach en la observación."""

    SOFT = "soft"   # "puede que..." — incertidumbre moderada
    HARD = "hard"   # "definitivamente" — alta confianza del coach


class SubmissionType(str, Enum):
    PRE_GAME = "pre_game"    # enviado antes del primer pitch (afecta optimización)
    POST_GAME = "post_game"  # enviado post-partido (datos para análisis y reentrenamiento)


# ---------------------------------------------------------------------------
# Feedback Cualitativo — Modelos de request / response
# ---------------------------------------------------------------------------


class PlayerFlagRequest(BaseModel):
    """Flag cualitativo para un jugador específico."""

    player_id: int = Field(description="MLB player ID")
    player_name: str = Field(description="Nombre del jugador (para legibilidad en logs)")
    flag_type: FlagType
    severity: FlagSeverity
    weight_override: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Impacto esperado en el rendimiento ofensivo. "
            "-1.0 = jugador completamente fuera de forma hoy. "
            "+1.0 = jugador en el mejor momento de la temporada. "
            "0.0 = sin ajuste."
        ),
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=280,
        description="Observación textual libre del coach (máx. 280 chars)",
    )


class QualitativeFeedbackRequest(BaseModel):
    """Formulario de retroalimentación cualitativa del bench coach."""

    game_id: str = Field(description="ID del partido (ej. '2026-05-19-NYY-BOS')")
    game_date: str = Field(
        description="Fecha del partido YYYY-MM-DD",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    submitted_by: str = Field(
        description="Rol del remitente (ej. 'bench_coach', 'manager', 'pitching_coach')"
    )
    submission_type: SubmissionType

    player_flags: list[PlayerFlagRequest] = Field(
        default_factory=list,
        description="Lista de flags por jugador. Puede estar vacía si solo se envía evaluación.",
    )

    # Campos de evaluación post-partido (opcionales, submission_type=post_game)
    ai_utility_score: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="¿Qué tan útil fue la recomendación AI? (1=nada útil, 5=muy útil)",
    )
    lineup_decision_quality: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="¿Qué tan buena fue la decisión real de alineación? (1-5)",
    )
    hidden_context: Optional[str] = Field(
        default=None,
        max_length=280,
        description="Contexto que el mánager tenía y el sistema no vio",
    )
    model_was_right: Optional[bool] = Field(
        default=None,
        description="¿Coincidió el resultado con la predicción del sistema?",
    )
    why_model_was_wrong: Optional[str] = Field(
        default=None,
        max_length=280,
        description="Si model_was_right=False, ¿qué falló?",
    )


class QualitativeFeedbackResponse(BaseModel):
    """Confirmación de recepción del feedback."""

    submission_id: str
    game_id: str
    flags_recorded: int
    submission_type: str
    message_es: str = Field(description="Confirmación en español para el bench coach")


class ActiveOverrideEntry(BaseModel):
    """Resumen del override activo para un jugador específico."""

    player_id: int
    player_name: str
    flag_types: list[str]
    severity: str = Field(description='"soft" | "hard" — peor caso entre los flags activos')
    aggregated_weight: float = Field(
        description="Peso efectivo agregado en [-1.0, 1.0]"
    )
    scale_factor: float = Field(
        description="Factor por el que se escalan los outcomes ofensivos positivos"
    )
    performance_impact_pct: float = Field(
        description="Impacto porcentual en el rendimiento esperado (negativo = peor)"
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Notas del coach para contexto",
    )


class ActiveOverridesResponse(BaseModel):
    """Overrides cualitativos activos para un game_date."""

    game_date: str
    player_count: int = Field(description="Número de jugadores con overrides activos")
    entries: list[ActiveOverrideEntry]
    generated_at: str = Field(description="Timestamp ISO 8601 de generación")


class FeatureProposal(BaseModel):
    """Propuesta de feature generada por flags recurrentes (Layer 3).

    Requiere aprobación humana antes de integrarse al pipeline de features.
    """

    player_id: int
    player_name: str
    flag_type: str
    occurrences: int = Field(description="Veces que se observó el flag en el período")
    lookback_days: int
    first_observed: str = Field(description="Primera fecha de observación YYYY-MM-DD")
    last_observed: str = Field(description="Última fecha de observación YYYY-MM-DD")
    proposed_feature_name: str = Field(
        description="Nombre de la feature propuesta (ej. recurring_physical_illness_last_30d)"
    )
    rationale_es: str = Field(description="Justificación en español para el equipo ML")
    requires_human_review: bool = True


# ---------------------------------------------------------------------------
# Trust / Explicabilidad — LineupScorecard
# ---------------------------------------------------------------------------


class ConfidenceTier(str, Enum):
    """Nivel de confianza semafórico para el mánager.

    HIGH   (>= 0.80): "Confía — desviarte cuesta ~0.3-0.5 runs de expectativa."
    MEDIUM (0.60-0.79): "Revisa — hay 1-2 posiciones donde tu criterio importa."
    LOW    (< 0.60): "Tu experiencia manda — usa esto como punto de partida."
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class UncertaintyFlag(BaseModel):
    """Aviso de incertidumbre específico para el bench coach."""

    flag_code: str = Field(description="Código interno del tipo de aviso")
    message_es: str = Field(description="Mensaje en español para el staff técnico")
    affected_positions: list[int] = Field(
        description="Posiciones del batting order afectadas (1-9). Vacío = lineup completo."
    )
    severity: Literal["INFO", "WARNING", "CRITICAL"] = Field(
        description="INFO = referencia, WARNING = requiere atención, CRITICAL = bloqueo"
    )


class SlotConfidence(BaseModel):
    """Confianza del modelo para un slot específico del batting order."""

    position: int = Field(..., ge=1, le=9, description="Posición en el batting order (1-9)")
    player_id: int = Field(description="MLB player ID del bateador asignado")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza en [0, 1]")
    sample_size_pa: int = Field(
        description="PAs históricas del jugador vs. tipo/mano del lanzador de hoy"
    )
    is_low_sample: bool = Field(
        description="True si sample_size_pa < 30 — señal de advertencia"
    )
    key_reason: str = Field(description="Razón principal de la asignación (1 línea, español)")


class LineupScorecard(BaseModel):
    """Scorecard de confianza completo para presentar al mánager antes del partido.

    Diseñado para responder: ¿Cuánto debo arriesgar discrepando con el sistema hoy?
    Contiene el score global, semáforo, confianza por slot, avisos de incertidumbre,
    razones cuantitativas en español y señal de acción recomendada.
    """

    game_id: str = Field(description="Identificador único del partido")
    generated_at: datetime = Field(description="Timestamp UTC de generación del scorecard")

    # --- Headline --- lo que el mánager ve primero
    confidence_overall: float = Field(
        ..., ge=0.0, le=1.0,
        description="Score de confianza global en [0, 1]",
    )
    confidence_tier: ConfidenceTier
    headline_es: str = Field(description="Frase de resumen en español (1 línea)")

    # --- Detalle por posición ---
    slots: list[SlotConfidence] = Field(
        ...,
        min_length=9,
        max_length=9,
        description="Confianza por slot (9 entradas, posiciones 1-9)",
    )

    # --- Avisos de incertidumbre ---
    uncertainty_flags: list[UncertaintyFlag] = Field(
        default_factory=list,
        description="Lista de avisos cualitativos para el bench coach",
    )

    # --- Top 3 razones cuantitativas ---
    top_drivers: list[str] = Field(
        ...,
        max_length=3,
        description="Top 3 razones de la recomendación en lenguaje natural (español)",
    )

    # --- Rendimiento histórico del sistema en condiciones similares ---
    historical_accuracy_similar_games: Optional[float] = Field(
        default=None,
        description="Precisión histórica del sistema en partidos similares (se popula tras 30+ partidos de A/B)",
    )
    similar_games_sample: Optional[int] = Field(
        default=None,
        description="Número de partidos similares en el histórico de A/B",
    )

    # --- Señal de acción ---
    recommend_manager_review: bool = Field(
        description="True si algún slot requiere atención especial del mánager"
    )
    review_positions: list[int] = Field(
        description="Posiciones (1-9) que merecen revisión por baja confianza o muestra insuficiente"
    )

    # --- Link al reporte de defensa RAG (se popula si es recomendación contra-intuitiva) ---
    rag_report_url: Optional[str] = Field(
        default=None,
        description="URL al reporte RAG de defensa si se detectó una recomendación inusual",
    )


# ---------------------------------------------------------------------------
# /predict/at-bat
# ---------------------------------------------------------------------------


class AtBatPredictRequest(BaseModel):
    """Feature vector for a single batter × pitcher × context matchup.

    ``features`` is a flat float32 array assembled by the upstream feature
    pipeline. Dimensionality must match the training configuration (~340 dims
    in Phase 3 baseline). Callers are responsible for correct feature ordering;
    the API does not perform feature selection or validation beyond shape checks.
    """

    player_id: int = Field(..., description="MLB player ID (batter)")
    pitcher_id: int = Field(..., description="MLB pitcher ID")
    features: list[float] = Field(..., min_length=1, description="Flat feature vector")

    @field_validator("features")
    @classmethod
    def _no_nan_inf(cls, v: list[float]) -> list[float]:
        import math
        if any(math.isnan(x) or math.isinf(x) for x in v):
            raise ValueError("features must not contain NaN or Inf values")
        return v


class AtBatPredictResponse(BaseModel):
    player_id: int
    pitcher_id: int
    outcome_probabilities: dict[str, float] = Field(
        description="7-class probability distribution: OUT_IN_PLAY, STRIKEOUT, "
                    "WALK_HBP, SINGLE, DOUBLE, TRIPLE, HOME_RUN"
    )
    expected_run_value: float = Field(description="E[R] contribution = probs · FanGraphs run values")
    model_version: str
    inference_time_ms: float = Field(description="Server-side inference latency in milliseconds")


# ---------------------------------------------------------------------------
# /optimize/lineup
# ---------------------------------------------------------------------------


class PlayerRosterEntry(BaseModel):
    """Per-player sabermetric profile for lineup optimization input."""

    player_id: int
    player_name: str
    obp: float = Field(..., ge=0.0, le=1.0)
    woba: float = Field(..., ge=0.0, le=1.0)
    iso: float = Field(..., ge=0.0, le=1.0)
    batter_stand: str = Field(..., pattern="^[LRS]$")
    sample_size_pa: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "PAs históricas de este bateador vs. el tipo/mano del lanzador de hoy. "
            "Usado por el scorecard de confianza. Omitir si no está disponible."
        ),
    )
    prob_vector: list[float] = Field(
        ...,
        min_length=7,
        max_length=7,
        description="Pre-computed calibrated PA outcome probability vector (7 classes)",
    )

    @field_validator("prob_vector")
    @classmethod
    def _valid_probability_vector(cls, v: list[float]) -> list[float]:
        total = sum(v)
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"prob_vector must sum to ~1.0 (got {total:.4f}). "
                "Renormalize before submitting."
            )
        if any(x < 0 for x in v):
            raise ValueError("prob_vector elements must be non-negative")
        return v


class RivalPitcherRequest(BaseModel):
    pitcher_id: int
    pitcher_name: str
    hand: str = Field(..., pattern="^[LR]$")
    era: float = Field(..., ge=0.0)
    pitch_mix: dict[str, float] = Field(
        ...,
        description="Pitch type → usage fraction dict (should sum to ~1.0)",
    )


class LineupOptimizeRequest(BaseModel):
    roster: list[PlayerRosterEntry] = Field(
        ...,
        min_length=9,
        max_length=9,
        description="Exactly 9 PlayerRosterEntry objects (today's active lineup)",
    )
    rival_pitcher: RivalPitcherRequest
    opp_lineup_probs: Optional[list[list[float]]] = Field(
        default=None,
        description="Opponent 9×7 probability matrix. Uses league-average defaults if omitted.",
    )
    park_factor_hr: float = Field(default=1.0, ge=0.7, le=1.5)
    park_factor_xb: float = Field(default=1.0, ge=0.7, le=1.5)
    game_id: Optional[str] = Field(
        default=None,
        description=(
            "Identificador único del partido (ej. '2026-05-19-NYY-BOS'). "
            "Se usa como key en el scorecard y el A/B experiment tracker. "
            "Si se omite, el servidor genera uno desde el timestamp."
        ),
    )
    data_freshness_hours: float = Field(
        default=2.0,
        ge=0.0,
        le=24.0,
        description=(
            "Horas desde la última actualización del feature más antiguo "
            "(Statcast, clima, platoon splits). Afecta el score de confianza "
            "del scorecard. Valor bajo = datos frescos = mayor confianza."
        ),
    )
    apply_feedback_overrides: bool = Field(
        default=False,
        description=(
            "Si True, el servidor carga los flags pre-partido activos del FeedbackStore "
            "para game_id/game_date y ajusta los prob_vectors antes de optimizar. "
            "Requiere que game_id esté especificado y que existan flags para ese partido."
        ),
    )
    fast_mode: bool = Field(
        default=False,
        description="True = 5k-simulation fast proxy (< 3s). False = full 100k-sim GA (~ 20–30s).",
    )

    @model_validator(mode="after")
    def _validate_opp_probs_shape(self) -> "LineupOptimizeRequest":
        if self.opp_lineup_probs is not None:
            if len(self.opp_lineup_probs) != 9:
                raise ValueError("opp_lineup_probs must have exactly 9 rows")
            for row in self.opp_lineup_probs:
                if len(row) != 7:
                    raise ValueError("Each opp_lineup_probs row must have exactly 7 elements")
        return self


class LineupOptimizeResponse(BaseModel):
    lineup_player_ids: list[int]
    lineup_player_names: list[str]
    expected_runs: float
    win_probability: float
    optimization_mode: str = Field(description='"fast" (5k sims) or "full" (100k sims)')
    total_evaluations: int
    elapsed_seconds: float
    model_version: str
    scorecard: Optional[LineupScorecard] = Field(
        default=None,
        description=(
            "Scorecard de confianza y explicabilidad para el staff técnico. "
            "Se incluye siempre en modo full. En fast_mode puede ser None."
        ),
    )
    defense_report: Optional[CrisisDefenseReport] = Field(
        default=None,
        description=(
            "Reporte de defensa RAG generado proactivamente cuando el sistema "
            "detecta una recomendación contra-intuitiva (STAR_PLAYER_MAJOR_DROP, "
            "LEADOFF_OBP_SUBOPTIMAL o POWER_HITTER_BURIED). None si la "
            "recomendación es convencional o si se usó fast_mode."
        ),
    )


# ---------------------------------------------------------------------------
# Crisis / Fallback — CrisisDefenseReport
# ---------------------------------------------------------------------------


class CrisisDefenseReport(BaseModel):
    """Reporte de defensa proactivo generado por el RAG cuando se detecta
    una recomendación contra-intuitiva.

    Se genera antes de que el mánager pregunte "¿qué pasó con...?".
    Incluye las 3 razones cuantitativas, contexto de scouts, las condiciones
    para ignorar la recomendación y una alineación alternativa (canónica).
    """

    report_id: str = Field(description="ID único del reporte (16 chars hex)")
    game_id: str
    generated_at: datetime

    # Qué trigger activó el reporte
    trigger_codes: list[str] = Field(
        description="Códigos de los triggers activados: STAR_PLAYER_MAJOR_DROP, "
                    "LEADOFF_OBP_SUBOPTIMAL, POWER_HITTER_BURIED"
    )
    trigger_descriptions_es: list[str] = Field(
        description="Descripción legible en español de cada trigger"
    )
    unusual_move_summary_es: str = Field(
        description="Resumen de una línea del movimiento inusual (español)"
    )

    # Contenido generado por el LLM
    markdown_content: str = Field(
        description="Reporte completo en Markdown con las 5 secciones obligatorias"
    )

    # Alineación alternativa (orden canónico sabermétrco)
    fallback_lineup_player_ids: list[int] = Field(
        description="IDs de los jugadores en el orden canónico alternativo"
    )
    fallback_lineup_player_names: list[str] = Field(
        description="Nombres en el orden canónico alternativo (slot 1 primero)"
    )
    fallback_er_estimate: float = Field(
        description="E[R] estimado del orden canónico (límite inferior conservador)"
    )
    er_cost_of_fallback: float = Field(
        description="Carreras/partido que se ceden al usar la alternativa vs. AI"
    )

    # Jugadores que impulsaron la recuperación RAG
    controversial_player_ids: list[int] = Field(
        description="IDs de los jugadores con mayor divergencia posicional vs. canónico"
    )

    # Proveniencia
    rag_chunks_used: int
    input_tokens: int
    output_tokens: int
    model_version: str


# ---------------------------------------------------------------------------
# A/B Experiment Tracker — Protocolo de Experimento Manager vs. AI
# ---------------------------------------------------------------------------


class DivergenceType(str, Enum):
    """Tipo de divergencia entre la recomendación AI y la decisión del mánager.

    ORDER_ONLY    — Los mismos 9 jugadores, distinto orden de bateo.
    PLAYER_SWAP   — Un jugador del AI fue sustituido por otro.
    FULL_OVERRIDE — Dos o más jugadores diferentes respecto al AI.
    """

    ORDER_ONLY = "order_only"
    PLAYER_SWAP = "player_swap"
    FULL_OVERRIDE = "full_override"


class RecordDivergenceRequest(BaseModel):
    """Registra una divergencia: la alineación del AI vs. la decisión real del mánager.

    Debe llamarse en el período pre-partido, después de recibir la respuesta de
    /v1/optimize/lineup y antes del primer pitch.
    """

    game_id: str = Field(description="ID del partido (ej. '2026-05-19-NYY-BOS')")
    game_date: str = Field(
        description="Fecha del partido YYYY-MM-DD",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    ai_lineup_ids: list[int] = Field(
        ...,
        min_length=9,
        max_length=9,
        description="IDs de los 9 jugadores en el orden de bateo recomendado por el AI (del campo lineup_player_ids de /v1/optimize/lineup)",
    )
    manager_lineup_ids: list[int] = Field(
        ...,
        min_length=9,
        max_length=9,
        description="IDs de los 9 jugadores en el orden de bateo real que presentó el mánager",
    )
    ai_expected_runs: float = Field(
        ...,
        ge=0.0,
        description="E[R] estimado por el optimizador para su lineup (del campo expected_runs de /v1/optimize/lineup)",
    )
    manager_rationale: Optional[str] = Field(
        default=None,
        max_length=280,
        description="Razón del mánager para divergir (máx. 280 chars)",
    )


class RecordDivergenceResponse(BaseModel):
    """Confirmación del registro de la divergencia."""

    observation_id: str = Field(description="ID único de la observación (16 chars hex)")
    game_id: str
    divergence_type: DivergenceType
    slots_changed: int = Field(
        description="Posiciones del batting order donde el mánager difiere del AI"
    )
    players_changed: int = Field(
        description="Jugadores del AI que el mánager no incluyó"
    )
    message_es: str = Field(description="Confirmación en español")


class RecordOutcomeRequest(BaseModel):
    """Registra el resultado real post-partido para cerrar la observación A/B."""

    game_id: str = Field(description="ID del partido (debe coincidir con el usado en record-divergence)")
    actual_runs_scored: int = Field(
        ...,
        ge=0,
        description="Carreras anotadas por el equipo con la alineación real del mánager",
    )
    innings_played: int = Field(
        default=9,
        ge=1,
        le=18,
        description="Entradas jugadas (9 en partidos normales, <9 en suspended games)",
    )


class RecordOutcomeResponse(BaseModel):
    """Resultado del cierre de la observación A/B."""

    game_id: str
    observations_updated: int
    ai_expected_runs: float = Field(
        description="E[R] que el AI había estimado para su lineup en este partido"
    )
    actual_runs_scored: int
    er_differential: float = Field(
        description=(
            "actual_runs_scored − ai_expected_runs. "
            "Positivo = el mánager superó la expectativa del AI. "
            "Negativo = el mánager quedó por debajo de la expectativa del AI."
        )
    )
    message_es: str


class ExperimentSummaryResponse(BaseModel):
    """Estadísticas acumuladas del experimento A/B Manager vs. AI.

    Se activa con significancia cuando total_closed_games >= 15.
    """

    lookback_days: int
    total_closed_games: int = Field(
        description="Partidos con resultado registrado en el período analizado"
    )
    by_divergence_type: dict[str, int] = Field(
        description="Distribución de observaciones por tipo: order_only, player_swap, full_override"
    )
    avg_ai_expected_runs: Optional[float] = Field(
        default=None,
        description="Media de E[R] del AI en los partidos donde hubo divergencia"
    )
    avg_actual_runs_scored: Optional[float] = Field(
        default=None,
        description="Media de carreras reales anotadas con la alineación del mánager"
    )
    manager_advantage_runs: Optional[float] = Field(
        default=None,
        description=(
            "avg_actual − avg_ai_expected. "
            "Positivo = el mánager tiende a superar la expectativa del AI. "
            "Negativo = el AI tiende a tener expectativas más altas que lo que se concreta."
        ),
    )
    sample_too_small: bool = Field(
        description="True si total_closed_games < 15 — estadísticas no concluyentes"
    )
    interpretation_es: str = Field(
        description="Interpretación automática en español (2-3 frases)"
    )


# ---------------------------------------------------------------------------
# /lineup/confirm — Registro automático de divergencias
# ---------------------------------------------------------------------------


class LineupConfirmRequest(BaseModel):
    """Payload enviado por la app del mánager al confirmar la alineación real.

    El servidor busca la recomendación AI cacheada para game_id y, si difiere,
    registra la divergencia automáticamente sin intervención humana adicional.
    """

    game_id: str = Field(
        description=(
            "ID del partido. Debe coincidir exactamente con el game_id devuelto "
            "por la llamada previa a /v1/optimize/lineup."
        )
    )
    game_date: str = Field(
        description="Fecha del partido YYYY-MM-DD",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    manager_lineup_ids: list[int] = Field(
        ...,
        min_length=9,
        max_length=9,
        description="IDs de los 9 jugadores en el orden de bateo real del mánager (slot 1 primero)",
    )
    manager_rationale: Optional[str] = Field(
        default=None,
        max_length=280,
        description="Razón del mánager para divergir (máx. 280 chars, opcional)",
    )


class LineupConfirmResponse(BaseModel):
    """Resultado del confirm: informa si se detectó y registró una divergencia."""

    game_id: str
    divergence_detected: bool = Field(
        description="True si el lineup del mánager difiere del AI y se registró la observación"
    )
    observation_id: Optional[str] = Field(
        default=None,
        description="ID de la observación registrada (16 chars hex). None si no hubo divergencia.",
    )
    divergence_type: Optional[str] = Field(
        default=None,
        description='"order_only" | "player_swap" | "full_override". None si no hubo divergencia.',
    )
    slots_changed: int = Field(
        description="Posiciones del batting order donde el mánager difiere del AI"
    )
    players_changed: int = Field(
        description="Jugadores del AI que el mánager no incluyó"
    )
    ai_expected_runs: float = Field(
        description="E[R] que el AI había estimado para su lineup"
    )
    message_es: str = Field(description="Mensaje de confirmación en español")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = Field(description='"healthy" | "degraded" | "unhealthy"')
    at_bat_model_loaded: bool
    model_version: str
    uptime_seconds: float
    details: dict[str, Any] = Field(default_factory=dict)
