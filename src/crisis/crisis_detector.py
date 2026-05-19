"""
crisis_detector.py
==================
Detecta recomendaciones de alineación contra-intuitivas que requieren
un reporte de defensa proactivo antes de que el mánager pregunte.

Una recomendación es "contra-intuitiva" cuando los cambios estructurales
del optimizador divergen significativamente de la línea base sabermétrca
canónica. Tres triggers cubren los casos más visibles para el staff:

    STAR_PLAYER_MAJOR_DROP
        Un bateador top-2 por wOBA cae ≥3 posiciones vs. el orden canónico.
        Ejemplo: el mejor bateador va del slot 3 al slot 7.
        Por qué sorprende: el valor ofensivo concentrado en slots 3-5
        es uno de los principios más arraigados del béisbol moderno.

    LEADOFF_OBP_SUBOPTIMAL
        El bateador inicial tiene ≥0.040 de OBP menos que el máximo del roster.
        Ejemplo: mejor OBP de la plantilla = .420; AI batea al .370 primero.
        Por qué sorprende: el leadoff de mayor OBP es casi un axioma;
        cualquier desviación necesita explicación explícita.

    POWER_HITTER_BURIED
        El jugador con mayor ISO queda en slots 7-9 cuando el canónico
        lo colocaría en 3-6.
        Ejemplo: slugger con ISO .280 bateando 8vo.
        Por qué sorprende: entierra el mayor potencial RISP en zona de
        mínimo apalancamiento.

Principios de diseño:
    - Puramente estructural: sin simulación adicional → latencia < 1ms.
    - Replica la lógica de SabermetricSeeder.canonical_seed() localmente
      para evitar un import circular con el módulo de optimización.
    - Retorna el primer trigger activado como "trigger primario" para el
      titular del reporte; los demás se incluyen como contexto adicional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import structlog

from app.schemas import PlayerRosterEntry

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Umbrales de activación
# ---------------------------------------------------------------------------

_STAR_DROP_THRESHOLD: int = 3
_LEADOFF_OBP_GAP: float = 0.040

# Slots de "alto apalancamiento" donde debe vivir el jugador de más poder
_POWER_HIGH_LEVERAGE_SLOTS: set[int] = {3, 4, 5, 6}
_POWER_BURIED_SLOTS: set[int] = {7, 8, 9}


# ---------------------------------------------------------------------------
# Tipos de output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleTrigger:
    """Un trigger de crisis activado."""

    code: str
    description_es: str
    affected_player_name: str
    canonical_position: int
    ai_position: int


@dataclass
class TriggerResult:
    """Resultado de la detección de crisis para una alineación."""

    triggered: bool
    triggers: list[SingleTrigger] = field(default_factory=list)

    @property
    def trigger_codes(self) -> list[str]:
        return [t.code for t in self.triggers]

    @property
    def descriptions_es(self) -> list[str]:
        return [t.description_es for t in self.triggers]

    def primary_trigger(self) -> Optional[SingleTrigger]:
        """Devuelve el trigger de mayor impacto comunicativo (primero en la lista)."""
        return self.triggers[0] if self.triggers else None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CrisisDetector:
    """Compara la alineación AI vs. la línea base sabermétrca para detectar triggers.

    Args:
        players: Los 9 PlayerRosterEntry del request de optimización.
        pitcher_hand: Mano del lanzador rival ("L" o "R") para el bonus platoon
            en el cálculo del orden canónico.
    """

    def __init__(
        self,
        players: list[PlayerRosterEntry],
        pitcher_hand: str = "R",
    ) -> None:
        if len(players) != 9:
            raise ValueError(f"Se esperaban 9 jugadores, se recibieron {len(players)}")
        self._players = players
        self._pitcher_hand = pitcher_hand
        self._canonical = self._compute_canonical_order()

    def _compute_canonical_order(self) -> list[int]:
        """Replica SabermetricSeeder.canonical_seed() sin importar lineup_optimizer.

        Slots 1-2: mayor OBP (con bonus platoon LHB vs. RHP).
        Slots 3-9: jugadores restantes por wOBA descendente.

        Returns:
            Lista de 9 índices de jugador; canonical[i] = índice en players para slot i+1.
        """
        def _key(idx: int, stat: str) -> tuple:
            p = self._players[idx]
            bonus = 0.005 if (self._pitcher_hand == "R" and p.batter_stand == "L") else 0.0
            return (-(getattr(p, stat) + bonus), idx)

        remaining = list(range(9))
        order: list[int] = []

        # Slots 1-2: OBP
        remaining.sort(key=lambda i: _key(i, "obp"))
        order.append(remaining.pop(0))
        remaining.sort(key=lambda i: _key(i, "obp"))
        order.append(remaining.pop(0))

        # Slots 3-9: wOBA
        remaining.sort(key=lambda i: _key(i, "woba"))
        order.extend(remaining)

        return order

    def canonical_order(self) -> list[int]:
        """Devuelve el orden canónico como lista de índices de jugador (slot 1 primero)."""
        return list(self._canonical)

    def canonical_lineup_names(self) -> list[str]:
        """Devuelve los nombres del orden canónico (slot 1 primero)."""
        return [self._players[i].player_name for i in self._canonical]

    def detect(self, best_lineup_indices: list[int]) -> TriggerResult:
        """Evalúa los tres triggers sobre la alineación AI recomendada.

        Args:
            best_lineup_indices: Orden óptimo del GA como índices sobre ``players``.

        Returns:
            TriggerResult con triggered=True si al menos un trigger se activa.
        """
        canonical_slot: dict[int, int] = {
            pidx: slot + 1 for slot, pidx in enumerate(self._canonical)
        }
        ai_slot: dict[int, int] = {
            pidx: slot + 1 for slot, pidx in enumerate(best_lineup_indices)
        }

        triggers: list[SingleTrigger] = []

        t = self._check_star_drop(canonical_slot, ai_slot)
        if t:
            triggers.append(t)

        t = self._check_leadoff_obp(best_lineup_indices)
        if t:
            triggers.append(t)

        t = self._check_power_buried(canonical_slot, ai_slot)
        if t:
            triggers.append(t)

        result = TriggerResult(triggered=bool(triggers), triggers=triggers)
        logger.info(
            "crisis_detection_complete",
            triggered=result.triggered,
            trigger_codes=result.trigger_codes,
        )
        return result

    # ------------------------------------------------------------------
    # Trigger checkers
    # ------------------------------------------------------------------

    def _check_star_drop(
        self,
        canonical_slot: dict[int, int],
        ai_slot: dict[int, int],
    ) -> Optional[SingleTrigger]:
        """Activa si un bateador top-2 por wOBA cae ≥_STAR_DROP_THRESHOLD slots."""
        top2 = sorted(range(9), key=lambda i: self._players[i].woba, reverse=True)[:2]
        for idx in top2:
            c = canonical_slot[idx]
            a = ai_slot[idx]
            drop = a - c  # positivo = baja en el orden
            if drop >= _STAR_DROP_THRESHOLD:
                p = self._players[idx]
                return SingleTrigger(
                    code="STAR_PLAYER_MAJOR_DROP",
                    description_es=(
                        f"{p.player_name} (wOBA {p.woba:.3f}) baja {drop} posiciones: "
                        f"slot {c} → slot {a} vs. el orden canónico."
                    ),
                    affected_player_name=p.player_name,
                    canonical_position=c,
                    ai_position=a,
                )
        return None

    def _check_leadoff_obp(
        self, best_lineup_indices: list[int]
    ) -> Optional[SingleTrigger]:
        """Activa si el leadoff AI tiene ≥_LEADOFF_OBP_GAP de OBP menos que el máximo del roster."""
        max_obp = max(p.obp for p in self._players)
        leadoff_idx = best_lineup_indices[0]
        leadoff_obp = self._players[leadoff_idx].obp
        gap = max_obp - leadoff_obp

        if gap >= _LEADOFF_OBP_GAP:
            best_obp_idx = max(range(9), key=lambda i: self._players[i].obp)
            p_leadoff = self._players[leadoff_idx]
            p_best = self._players[best_obp_idx]
            canonical_pos = self._canonical.index(leadoff_idx) + 1
            return SingleTrigger(
                code="LEADOFF_OBP_SUBOPTIMAL",
                description_es=(
                    f"Bateador inicial {p_leadoff.player_name} (OBP {leadoff_obp:.3f}) "
                    f"tiene OBP {gap:.3f} inferior a {p_best.player_name} "
                    f"({max_obp:.3f}), el mayor del roster."
                ),
                affected_player_name=p_leadoff.player_name,
                canonical_position=canonical_pos,
                ai_position=1,
            )
        return None

    def _check_power_buried(
        self,
        canonical_slot: dict[int, int],
        ai_slot: dict[int, int],
    ) -> Optional[SingleTrigger]:
        """Activa si el mayor ISO del roster va a slots 7-9 cuando el canónico lo ubica en 3-6."""
        power_idx = max(range(9), key=lambda i: self._players[i].iso)
        c = canonical_slot[power_idx]
        a = ai_slot[power_idx]

        if c in _POWER_HIGH_LEVERAGE_SLOTS and a in _POWER_BURIED_SLOTS:
            p = self._players[power_idx]
            return SingleTrigger(
                code="POWER_HITTER_BURIED",
                description_es=(
                    f"{p.player_name} (ISO {p.iso:.3f}) asignado al slot {a} cuando "
                    f"el orden canónico lo ubica en slot {c}. "
                    "Máximo poder ofensivo en zona de bajo apalancamiento."
                ),
                affected_player_name=p.player_name,
                canonical_position=c,
                ai_position=a,
            )
        return None
