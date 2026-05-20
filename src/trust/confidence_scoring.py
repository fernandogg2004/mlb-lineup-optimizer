"""
confidence_scoring.py
=====================
Builds the LineupScorecard — capa de confianza/explicabilidad para el staff técnico.

El scorecard responde una pregunta para el mánager:
"¿Cuánto debo arriesgar discrepando con el sistema hoy?"

La confianza se deriva de tres señales independientes:

    1. Estabilidad de la optimización (weight=0.55):
       Spread de E[R] entre los top-K candidatos del refinamiento GA.
       Un gap claro entre el mejor y el segundo mejor lineup → señal fuerte.
       Un landscape plano (todos similares) → el orden importa menos hoy.

    2. Frescura de los datos (weight=0.25):
       Antigüedad de los features de entrada (Statcast, clima, etc.).
       Datos > 6h antes del primer pitch degradan la confianza linealmente.

    3. Cobertura muestral (weight=0.20):
       Fracción de slots con >= 100 PAs históricas vs. el tipo/mano
       del lanzador de hoy. Matchups sin datos estables = predicción frágil.

Los pesos (0.55/0.25/0.20) fueron calibrados contra criterio experto en
40 partidos de pre-temporada. La estabilidad de optimización es el driver
principal porque mide directamente la confianza del modelo en el ranking.

Architecture note:
    Este módulo importa de app.schemas (dirección src/ → app/) porque
    confidence_scoring.py es una capa de integración que produce outputs
    listos para la API. No contiene lógica de ML — solo composición de señales.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import structlog

from app.schemas import (
    ConfidenceTier,
    LineupScorecard,
    PlayerRosterEntry,
    RivalPitcherRequest,
    SlotConfidence,
    UncertaintyFlag,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

_HIGH_CONFIDENCE_THRESHOLD: float = 0.80
_MEDIUM_CONFIDENCE_THRESHOLD: float = 0.60

# Optimization spread: 0.25 carreras de gap entre 1º y último candidato top-K = score=1.0
_SPREAD_SATURATION_RUNS: float = 0.25

# Freshness: confianza plena hasta 2h, decae a 0 a las 8h
_FRESHNESS_GRACE_HOURS: float = 2.0
_FRESHNESS_DECAY_HOURS: float = 6.0  # horas adicionales hasta score=0

# Sample size: 100 PAs vs. pitcher tipo/mano → score=1.0 para ese slot
_SAMPLE_SATURATION_PA: int = 100
_SAMPLE_LOW_THRESHOLD: int = 30


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------


def _stability_score(refinement_scores: list[float]) -> float:
    """Score de estabilidad: spread E[R] entre los top-K candidatos del GA.

    Un spread alto significa que el optimizador encontró un ganador claro.
    Un spread bajo significa que el landscape es plano — el orden importa menos.

    Args:
        refinement_scores: Lista de E[R] para los top-K candidatos del
            refinamiento de alta fidelidad (100k sims). Al menos 2 valores.

    Returns:
        Stability score en [0, 1].
    """
    if len(refinement_scores) < 2:
        return 0.50
    spread = max(refinement_scores) - min(refinement_scores)
    return round(min(1.0, spread / _SPREAD_SATURATION_RUNS), 4)


def _freshness_score(data_freshness_hours: float) -> float:
    """Score de frescura: degrada linealmente desde las 2h hasta las 8h de antigüedad.

    Args:
        data_freshness_hours: Horas desde la última actualización del feature
            más antiguo relevante (Statcast, clima, platoon splits).

    Returns:
        Freshness score en [0, 1].
    """
    age_past_grace = max(0.0, data_freshness_hours - _FRESHNESS_GRACE_HOURS)
    return round(max(0.0, 1.0 - age_past_grace / _FRESHNESS_DECAY_HOURS), 4)


def _sample_score(sample_sizes: list[int]) -> float:
    """Score muestral: fracción promedio de cobertura de PAs por slot.

    Cada slot contribuye min(1.0, sample_size_pa / SATURATION) al promedio.

    Args:
        sample_sizes: Lista de PAs históricas por posición del batting order
            (9 valores, uno por slot).

    Returns:
        Sample score en [0, 1].
    """
    if not sample_sizes:
        return 0.50
    scores = [min(1.0, pa / _SAMPLE_SATURATION_PA) for pa in sample_sizes]
    return round(float(np.mean(scores)), 4)


def _slot_confidence_value(sample_size_pa: int, prob_vector: list[float]) -> float:
    """Confianza por slot: combina cobertura muestral y definición del perfil.

    La definición del perfil (CV del prob_vector) mide cuán extremo/identificable
    es el jugador vs. la media de liga. Un perfil extremo (ej. alto HR%, alto K%)
    da una señal más clara al modelo que un perfil genérico.

    Args:
        sample_size_pa: PAs históricas del bateador vs. el tipo de lanzador de hoy.
        prob_vector: Vector de probabilidades de 7 outcomes calibradas.

    Returns:
        Slot confidence en [0, 1].
    """
    sample_component = min(1.0, sample_size_pa / _SAMPLE_SATURATION_PA)

    pv = np.array(prob_vector, dtype=np.float64)
    cv = pv.std() / (pv.mean() + 1e-9)
    profile_component = min(1.0, cv / 2.0)  # CV~2 = perfil muy definido

    return round(0.65 * sample_component + 0.35 * profile_component, 3)


# ---------------------------------------------------------------------------
# Natural language generators (Spanish)
# ---------------------------------------------------------------------------


def _tier_from_score(score: float) -> ConfidenceTier:
    if score >= _HIGH_CONFIDENCE_THRESHOLD:
        return ConfidenceTier.HIGH
    if score >= _MEDIUM_CONFIDENCE_THRESHOLD:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.LOW


def _headline_es(
    tier: ConfidenceTier, n_flags: int, crisis_triggered: bool = False
) -> str:
    if crisis_triggered:
        return (
            "RECOMENDACION INUSUAL — ver Reporte de Defensa adjunto antes de decidir."
        )
    if tier == ConfidenceTier.HIGH:
        if n_flags == 0:
            return "Sistema con alta confianza. Datos sólidos y optimización estable."
        return f"Sistema con alta confianza. {n_flags} punto(s) de atención a revisar."
    if tier == ConfidenceTier.MEDIUM:
        return f"Confianza moderada. Revisar {n_flags} posición(es) marcadas en amarillo."
    return "Datos insuficientes o alta varianza. Usa esto como punto de partida, no como respuesta."


def _key_reason_es(
    player: PlayerRosterEntry,
    pitcher: RivalPitcherRequest,
    batting_position: int,
) -> str:
    """Justificación de una línea en español para la asignación del slot."""
    hand_label = "zurdo" if pitcher.hand == "L" else "derecho"
    has_platoon_adv = (
        (player.batter_stand == "L" and pitcher.hand == "R")
        or (player.batter_stand == "R" and pitcher.hand == "L")
    )

    if batting_position <= 2:
        return (
            f"OBP {player.obp:.3f} — "
            f"alta tasa de embase para posición {batting_position}"
        )
    if batting_position <= 5:
        platoon_note = " + ventaja de platoon" if has_platoon_adv else ""
        return (
            f"wOBA {player.woba:.3f} vs. lanzador {hand_label}{platoon_note}"
        )
    return f"wOBA {player.woba:.3f} — posición {batting_position} en la alineación"


def _build_uncertainty_flags(
    players: list[PlayerRosterEntry],
    best_lineup_indices: list[int],
    pitcher: RivalPitcherRequest,
    refinement_scores: list[float],
    data_freshness_hours: float,
) -> list[UncertaintyFlag]:
    """Genera avisos de incertidumbre legibles para el bench coach."""
    flags: list[UncertaintyFlag] = []
    ordered = [players[i] for i in best_lineup_indices]

    # Aviso 1: Slots con pocas PAs históricas vs. este tipo de lanzador
    low_sample_positions = [
        pos + 1
        for pos, p in enumerate(ordered)
        if (p.sample_size_pa or 0) < _SAMPLE_LOW_THRESHOLD
    ]
    if low_sample_positions:
        flags.append(UncertaintyFlag(
            flag_code="LOW_SAMPLE_MATCHUP",
            message_es=(
                f"Pocas PAs históricas (<{_SAMPLE_LOW_THRESHOLD}) vs. este tipo de lanzador "
                f"en posición(es) {low_sample_positions}. Predicción menos estable."
            ),
            affected_positions=low_sample_positions,
            severity="WARNING",
        ))

    # Aviso 2: Landscape de optimización plano — el orden importa poco hoy
    if len(refinement_scores) >= 2:
        spread = max(refinement_scores) - min(refinement_scores)
        if spread < 0.10:
            flags.append(UncertaintyFlag(
                flag_code="FLAT_OPTIMIZATION_LANDSCAPE",
                message_es=(
                    f"Las 10 mejores alineaciones difieren solo {spread:.2f} carreras. "
                    "El orden específico importa menos hoy — múltiples variantes son viables."
                ),
                affected_positions=list(range(1, 10)),
                severity="INFO",
            ))

    # Aviso 3: Bateadores del corazón (3-5) sin ventaja de platoon
    core_no_platoon = [
        pos + 3
        for pos, p in enumerate(ordered[2:5])
        if p.batter_stand != "S"
        and not (
            (p.batter_stand == "L" and pitcher.hand == "R")
            or (p.batter_stand == "R" and pitcher.hand == "L")
        )
    ]
    if core_no_platoon:
        hand_label = "zurdo" if pitcher.hand == "L" else "derecho"
        flags.append(UncertaintyFlag(
            flag_code="NO_PLATOON_ADVANTAGE_CORE",
            message_es=(
                f"Posición(es) {core_no_platoon} sin ventaja de platoon "
                f"vs. lanzador {hand_label}. El modelo ajusta via wOBA vs. mano — "
                "verifica que los splits estén actualizados."
            ),
            affected_positions=core_no_platoon,
            severity="INFO",
        ))

    # Aviso 4: Datos desactualizados
    if data_freshness_hours > 4.0:
        severity = "WARNING" if data_freshness_hours > 6.0 else "INFO"
        flags.append(UncertaintyFlag(
            flag_code="STALE_INPUT_DATA",
            message_es=(
                f"Datos con {data_freshness_hours:.1f}h de antigüedad. "
                "Verificar si hay actualizaciones de Statcast o clima disponibles."
            ),
            affected_positions=[],
            severity=severity,
        ))

    return flags


def _build_top_drivers_es(
    players: list[PlayerRosterEntry],
    best_lineup_indices: list[int],
    pitcher: RivalPitcherRequest,
    park_factor_hr: float,
    park_factor_xb: float,
    expected_runs: float,
) -> list[str]:
    """Genera las 3 razones cuantitativas principales en español."""
    ordered = [players[i] for i in best_lineup_indices]
    drivers: list[str] = []

    # Driver 1: leadoff OBP
    leadoff = ordered[0]
    drivers.append(
        f"{leadoff.player_name} lidera con OBP {leadoff.obp:.3f} — "
        f"máxima creación de oportunidades desde el slot 1."
    )

    # Driver 2: ventaja de platoon en el lineup
    platoon_count = sum(
        1 for p in ordered
        if (p.batter_stand == "L" and pitcher.hand == "R")
        or (p.batter_stand == "R" and pitcher.hand == "L")
    )
    if platoon_count >= 5:
        hand_label = "zurdo" if pitcher.hand == "L" else "derecho"
        drivers.append(
            f"{platoon_count}/9 bateadores con ventaja de platoon "
            f"vs. lanzador {hand_label} — lineup construido para este matchup."
        )
    else:
        top_core = max(ordered[2:5], key=lambda p: p.woba)
        drivers.append(
            f"{top_core.player_name} (wOBA {top_core.woba:.3f}) ancla el corazón "
            f"del orden — mayor concentración de valor ofensivo en slots 3-5."
        )

    # Driver 3: park factor o contexto de E[R]
    if park_factor_hr > 1.10:
        drivers.append(
            f"Factor de parque HR = {park_factor_hr:.2f} — "
            f"mayor tasa de cuadrangulares eleva E[R] proyectado = {expected_runs:.2f}."
        )
    elif park_factor_xb > 1.10:
        drivers.append(
            f"Factor de parque extrabase = {park_factor_xb:.2f} — "
            f"potencia los hits de extra base del orden."
        )
    else:
        drivers.append(
            f"E[R] proyectado = {expected_runs:.2f} carreras/partido "
            f"sobre 100,000 simulaciones Monte Carlo."
        )

    return drivers[:3]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_confidence_score(
    refinement_scores: list[float],
    data_freshness_hours: float,
    sample_sizes: list[int],
) -> float:
    """Calcula el score de confianza global de la alineación recomendada.

    Combina tres señales independientes con pesos calibrados:
        0.55 × estabilidad_optimización
        0.25 × frescura_datos
        0.20 × cobertura_muestral

    Args:
        refinement_scores: E[R] de los top-K candidatos del refinamiento GA
            (100k sims). Al menos 2 valores requeridos para el cálculo de spread.
        data_freshness_hours: Horas desde la última actualización del feature
            más antiguo (Statcast, clima, platoon splits).
        sample_sizes: PAs históricas por slot del batting order (9 valores).
            Representan cobertura vs. el tipo/mano del lanzador de hoy.

    Returns:
        Score de confianza en [0, 1].
        Umbrales operativos: HIGH >= 0.80, MEDIUM >= 0.60, LOW < 0.60.
    """
    stability = _stability_score(refinement_scores)
    freshness = _freshness_score(data_freshness_hours)
    sample = _sample_score(sample_sizes)

    score = 0.55 * stability + 0.25 * freshness + 0.20 * sample
    return round(min(1.0, max(0.0, score)), 3)


def build_lineup_scorecard(
    game_id: str,
    players: list[PlayerRosterEntry],
    best_lineup_indices: list[int],
    best_expected_runs: float,
    best_win_probability: float,
    refinement_scores: list[float],
    pitcher: RivalPitcherRequest,
    park_factor_hr: float,
    park_factor_xb: float,
    data_freshness_hours: float = 2.0,
    crisis_triggered: bool = False,
) -> LineupScorecard:
    """Construye el LineupScorecard completo para el staff técnico.

    Orquesta todas las señales de confianza en un único objeto serializable
    listo para incluir en la respuesta de /v1/optimize/lineup.

    Args:
        game_id: Identificador único del partido (ej. "2026-05-19-NYY-BOS").
        players: Los 9 PlayerRosterEntry del request de optimización.
            Incluir sample_size_pa para confianza por slot precisa.
        best_lineup_indices: Permutación óptima como índices sobre ``players``.
        best_expected_runs: E[R] del refinamiento de alta fidelidad (100k sims).
        best_win_probability: P(W) del refinamiento de alta fidelidad.
        refinement_scores: E[R] por cada candidato top-K del GA refinement.
        pitcher: Datos del lanzador rival.
        park_factor_hr: Factor de parque de cuadrangulares para el venue de hoy.
        park_factor_xb: Factor de parque de hits de extrabase.
        data_freshness_hours: Horas desde la actualización del input más antiguo.

    Returns:
        LineupScorecard populado, listo para serialización JSON en la API.
    """
    sample_sizes = [p.sample_size_pa or 0 for p in players]

    confidence_overall = compute_confidence_score(
        refinement_scores, data_freshness_hours, sample_sizes
    )
    tier = _tier_from_score(confidence_overall)

    ordered_players = [players[i] for i in best_lineup_indices]
    slots = [
        SlotConfidence(
            position=pos,
            player_id=p.player_id,
            confidence=_slot_confidence_value(p.sample_size_pa or 0, p.prob_vector),
            sample_size_pa=p.sample_size_pa or 0,
            is_low_sample=(p.sample_size_pa or 0) < _SAMPLE_LOW_THRESHOLD,
            key_reason=_key_reason_es(p, pitcher, pos),
        )
        for pos, p in enumerate(ordered_players, start=1)
    ]

    flags = _build_uncertainty_flags(
        players, best_lineup_indices, pitcher, refinement_scores, data_freshness_hours
    )
    top_drivers = _build_top_drivers_es(
        players, best_lineup_indices, pitcher,
        park_factor_hr, park_factor_xb, best_expected_runs,
    )

    review_positions = [
        s.position for s in slots
        if s.is_low_sample or s.confidence < 0.50
    ]

    scorecard = LineupScorecard(
        game_id=game_id,
        generated_at=datetime.now(tz=timezone.utc),
        confidence_overall=confidence_overall,
        confidence_tier=tier,
        headline_es=_headline_es(tier, len(flags), crisis_triggered=crisis_triggered),
        slots=slots,
        uncertainty_flags=flags,
        top_drivers=top_drivers,
        historical_accuracy_similar_games=None,
        similar_games_sample=None,
        recommend_manager_review=(tier != ConfidenceTier.HIGH or bool(review_positions)),
        review_positions=review_positions,
        rag_report_url=None,
    )

    logger.info(
        "lineup_scorecard_built",
        game_id=game_id,
        confidence=confidence_overall,
        tier=tier.value,
        n_flags=len(flags),
        n_review_positions=len(review_positions),
        stability_component=_stability_score(refinement_scores),
        freshness_component=_freshness_score(data_freshness_hours),
        sample_component=_sample_score(sample_sizes),
    )
    return scorecard
