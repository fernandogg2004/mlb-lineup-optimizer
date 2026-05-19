"""
override_engine.py  —  Capa 1 del sistema de retroalimentación cualitativa.
============================================================================
Aplica los flags del bench coach como modificadores del prob_vector de cada
jugador antes de que el optimizador genétoco calcule E[R].

Mecanismo:
----------
El prob_vector de 7 outcomes se divide en dos grupos:
    Negativos (índices 0-1): OUT_IN_PLAY, STRIKEOUT
    Positivos  (índices 2-6): WALK_HBP, SINGLE, DOUBLE, TRIPLE, HOME_RUN

Un flag con weight_override = -0.6 (jugador con molestia física):
    factor = 1.0 + (-0.6) = 0.40
    → Las probabilidades de outcomes positivos se multiplican por 0.40.
    → Se renormaliza el vector a suma 1.0.
    → El jugador pasa a parecerse ofensivamente a un bateador débil.

Un flag con weight_override = +0.4 (jugador en racha caliente):
    factor = 1.0 + 0.40 = 1.40
    → Las probabilidades de outcomes positivos se multiplican por 1.40.
    → El optimizador lo valorará más alto en slots de alto apalancamiento.

Reglas de severidad:
    SOFT → factor = 1.0 + weight_override × 0.5   (señal incierta del coach)
    HARD → factor = 1.0 + weight_override × 1.0   (coach tiene alta confianza)

Agregación de múltiples flags para el mismo jugador:
    Se suman los pesos efectivos y se recortan a [-1.0, 1.0].
    Ejemplo: HARD physical_illness(-0.7) + SOFT fatigue_travel(-0.3×0.5=-0.15)
             → total = -0.85 → factor = 0.15 → jugador casi sin valor ofensivo.

Invariantes:
    - El vector resultado siempre suma 1.0 (se renormaliza).
    - factor mínimo = 0.05 → nunca probabilidad totalmente colapsada.
    - El roster resultante tiene exactamente 9 jugadores (misma cantidad).
    - Los player_id no cambian — solo los prob_vector.
"""
from __future__ import annotations

import numpy as np
import structlog

from app.schemas import FlagSeverity, PlayerFlagRequest, PlayerRosterEntry

logger = structlog.get_logger(__name__)

# Índices del prob_vector que representan outcomes ofensivos positivos
_POSITIVE_INDICES: list[int] = [2, 3, 4, 5, 6]  # WALK_HBP, SINGLE, DOUBLE, TRIPLE, HR

# Factor de escala por severidad
_SEVERITY_SCALE: dict[str, float] = {
    FlagSeverity.SOFT.value: 0.5,
    FlagSeverity.HARD.value: 1.0,
}

# Límites del factor de escalado para evitar vectores degenerados
_FACTOR_MIN: float = 0.05   # probabilidad mínima relativa
_FACTOR_MAX: float = 3.0    # máximo boost (150% por encima de baseline)


def _aggregate_weight(flags: list[PlayerFlagRequest]) -> float:
    """Agrega múltiples flags del mismo jugador en un peso efectivo único.

    Suma pesos efectivos (weight × severity_scale) y recorta a [-1.0, 1.0].

    Args:
        flags: Lista de flags del mismo jugador en el mismo partido.

    Returns:
        Peso efectivo agregado en [-1.0, 1.0].
    """
    total = sum(
        f.weight_override * _SEVERITY_SCALE.get(f.severity.value, 1.0)
        for f in flags
    )
    return max(-1.0, min(1.0, total))


def _scale_prob_vector(prob_vector: list[float], effective_weight: float) -> list[float]:
    """Escala outcomes positivos por el factor derivado del peso efectivo.

    Args:
        prob_vector: Vector de probabilidades de 7 outcomes (suma ~1.0).
        effective_weight: Peso efectivo en [-1.0, 1.0]. Negativo = peor
            rendimiento esperado. Positivo = mejor rendimiento esperado.

    Returns:
        Nuevo vector escalado y renormalizado de 7 floats (suma exactamente 1.0).
    """
    factor = max(_FACTOR_MIN, min(_FACTOR_MAX, 1.0 + effective_weight))
    pv = np.array(prob_vector, dtype=np.float64)
    pv[_POSITIVE_INDICES] *= factor

    total = pv.sum()
    if total <= 0.0:
        # Guardia de seguridad: devolver distribución uniforme si colapsa
        return [1.0 / 7] * 7

    normalized = (pv / total).tolist()
    return normalized


def apply_qualitative_overrides(
    roster: list[PlayerRosterEntry],
    flags: list[PlayerFlagRequest],
) -> list[PlayerRosterEntry]:
    """Aplica flags del bench coach como modificadores del prob_vector.

    Devuelve una NUEVA lista de PlayerRosterEntry con los prob_vectors ajustados
    para los jugadores flaggeados. Los jugadores sin flags se devuelven sin cambios.

    Args:
        roster: Los 9 PlayerRosterEntry del request de optimización.
        flags: Flags activos del día (pre-partido), cargados desde el FeedbackStore
            o pasados directamente desde el payload del request.

    Returns:
        Nueva lista de 9 PlayerRosterEntry. Los prob_vectors de jugadores flaggeados
        reflejan el ajuste cualitativo del coach.
    """
    if not flags:
        return roster

    # Agrupar flags por player_id
    by_player: dict[int, list[PlayerFlagRequest]] = {}
    for flag in flags:
        by_player.setdefault(flag.player_id, []).append(flag)

    new_roster: list[PlayerRosterEntry] = []
    for player in roster:
        player_flags = by_player.get(player.player_id)
        if not player_flags:
            new_roster.append(player)
            continue

        eff_weight = _aggregate_weight(player_flags)
        new_pv = _scale_prob_vector(player.prob_vector, eff_weight)

        flag_types = [f.flag_type.value for f in player_flags]
        logger.info(
            "qualitative_override_applied",
            player_id=player.player_id,
            player_name=player.player_name,
            flag_types=flag_types,
            effective_weight=round(eff_weight, 3),
            scale_factor=round(max(_FACTOR_MIN, min(_FACTOR_MAX, 1.0 + eff_weight)), 3),
        )

        new_roster.append(player.model_copy(update={"prob_vector": new_pv}))

    return new_roster


def compute_override_summary(
    roster: list[PlayerRosterEntry],
    flags: list[PlayerFlagRequest],
) -> list[dict]:
    """Devuelve un resumen legible de qué jugadores serán ajustados y cómo.

    Útil para el endpoint GET /v1/feedback/overrides sin ejecutar la optimización.

    Args:
        roster: Los 9 PlayerRosterEntry del lineup.
        flags: Flags activos del día.

    Returns:
        Lista de dicts con player_id, player_name, effective_weight, scale_factor,
        flag_types, y performance_impact_pct para cada jugador ajustado.
    """
    by_player: dict[int, list[PlayerFlagRequest]] = {}
    for flag in flags:
        by_player.setdefault(flag.player_id, []).append(flag)

    summaries: list[dict] = []
    player_names = {p.player_id: p.player_name for p in roster}

    for player_id, player_flags in by_player.items():
        eff_weight = _aggregate_weight(player_flags)
        scale_factor = max(_FACTOR_MIN, min(_FACTOR_MAX, 1.0 + eff_weight))
        summaries.append({
            "player_id": player_id,
            "player_name": player_names.get(player_id, f"id:{player_id}"),
            "flag_types": [f.flag_type.value for f in player_flags],
            "severity": max(f.severity.value for f in player_flags),
            "aggregated_weight": round(eff_weight, 3),
            "scale_factor": round(scale_factor, 3),
            "performance_impact_pct": round((scale_factor - 1.0) * 100, 1),
            "notes": [f.notes for f in player_flags if f.notes],
        })

    return summaries
