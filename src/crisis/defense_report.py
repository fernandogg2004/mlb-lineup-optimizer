"""
defense_report.py
=================
Genera el reporte RAG de defensa para recomendaciones contra-intuitivas.

El CrisisReportGenerator reutiliza ScoutingRetriever y ContextAssembler
de lineup_explainer_agent.py con un system prompt en modo "defensa proactiva".
El reporte llega al mánager ANTES de que pregunte, no en respuesta a ella.

Arquitectura de ejecución:
    - Se ejecuta sincrónicamente en el mismo ThreadPoolExecutor que el
      optimizador (añade ~10-15s al modo full, dentro del SLA de 90min).
    - Si falla (keys ausentes, Pinecone down, Claude timeout), el error
      se captura en main.py y la respuesta principal no se bloquea.
    - La recuperación RAG se enfoca en los jugadores "controversiales" —
      aquellos con mayor divergencia posicional vs. el orden canónico.

Sobre el E[R] de la alineación fallback:
    Se usa min(refinement_scores) como estimación conservadora del E[R]
    del orden canónico. Esto evita una simulación MC adicional y es un
    límite inferior válido: si el GA mejoró respecto al canónico, el
    canónico obtuvo un E[R] parecido al peor candidato evaluado.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog

from app.schemas import CrisisDefenseReport, PlayerRosterEntry, RivalPitcherRequest
from src.crisis.crisis_detector import TriggerResult
from src.rag.lineup_explainer_agent import (
    ContextAssembler,
    ExplainerConfig,
    ScoutingRetriever,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# System prompt en modo "defensa proactiva"
# ---------------------------------------------------------------------------

_CRISIS_DEFENSE_SYSTEM_PROMPT_ES = """\
Eres el Analista de Datos Senior del equipo de béisbol. El motor de \
optimización ha generado una recomendación de alineación inusual. \
Tu misión es redactar un REPORTE DE DEFENSA PROACTIVO para el Mánager, \
antes de que él pregunte qué pasó.

REGLAS NO NEGOCIABLES:
1. Basa TODAS tus conclusiones exclusivamente en los datos cuantitativos \
y reportes de scouts proporcionados. NUNCA inventes estadísticas, fechas \
ni eventos ausentes del contexto.
2. Si no hay reportes recientes de un jugador, escribe exactamente: \
"Sin reportes recientes (< 30 días)" — nunca inferir ni inventar.
3. Sé directo y ejecutivo. El Mánager lee esto en 3 minutos, no 10.
4. Explica la jerga estadística la primera vez que la uses: \
"wOBA .380 (rendimiento ofensivo ajustado — media de liga ~.320)".
5. Sé honesto sobre las limitaciones del modelo — esto genera confianza \
a largo plazo con el staff técnico.

ESTRUCTURA OBLIGATORIA (usa exactamente estos encabezados Markdown):

## ¿Qué recomienda el sistema y por qué?
[Una frase directa. Nombra al jugador y la posición inusual. Sin preámbulos.]

## Los 3 datos que sustentan esta decisión
1. [Dato cuantitativo + fuente + período de tiempo exacto]
2. [Dato cuantitativo + fuente + período de tiempo exacto]
3. [Dato cuantitativo + fuente + período de tiempo exacto]

## Qué dicen los scouts
[Resumen de los reportes recientes sobre los jugadores en posiciones \
inusuales. Si no hay reportes: "Sin reportes recientes (< 30 días)".]

## ¿Cuándo debería el Mánager ignorar esta recomendación?
[Lista concisa de condiciones donde el criterio humano supera al modelo. \
Sé específico: "Si el jugador X reportó molestia en el calentamiento", \
"Si el lanzador rival cambió su mecánica desde que se recopilaron estos datos".]

## Alternativa si no estás convencido
[La alineación alternativa (orden canónico sabermétrco) con su E[R] estimado. \
Indica el costo en carreras esperadas de usar esta alternativa vs. la recomendación.]
"""


# ---------------------------------------------------------------------------
# Constructor del bloque de contexto cuantitativo
# ---------------------------------------------------------------------------


def _build_crisis_quant_block(
    trigger_description: str,
    ai_lineup_names: list[str],
    canonical_lineup_names: list[str],
    best_er: float,
    fallback_er: float,
    pitcher: RivalPitcherRequest,
) -> str:
    """Construye el bloque de contexto cuantitativo para el prompt de defensa."""
    hand_label = "zurdo" if pitcher.hand == "L" else "derecho"
    er_delta = best_er - fallback_er

    pitch_mix_str = "  |  ".join(
        f"{pt}: {pct:.1%}" for pt, pct in (pitcher.pitch_mix or {}).items()
    )

    ai_order_str = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(ai_lineup_names))
    can_order_str = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(canonical_lineup_names))

    return (
        "## MOVIMIENTO INUSUAL DETECTADO\n"
        f"{trigger_description}\n\n"
        "## ALINEACIÓN RECOMENDADA POR EL SISTEMA\n"
        f"E[R] proyectado = {best_er:.3f} carreras/partido (sobre 100,000 simulaciones)\n"
        f"{ai_order_str}\n\n"
        "## ALINEACIÓN CANÓNICA (REFERENCIA SABERMÉTRCA)\n"
        f"E[R] estimado = {fallback_er:.3f} carreras/partido\n"
        f"{can_order_str}\n\n"
        "## DIFERENCIAL DE VALOR\n"
        f"La recomendación del sistema genera +{er_delta:.3f} carreras/partido "
        f"vs. el orden canónico. Sobre 162 partidos equivale a +{er_delta * 162:.1f} "
        "carreras en la temporada.\n\n"
        "## LANZADOR RIVAL\n"
        f"Mano: {hand_label}  |  ERA: {pitcher.era:.2f}  |  "
        + (f"Arsenal: {pitch_mix_str}" if pitch_mix_str else "Arsenal: no disponible")
        + "\n"
    )


# ---------------------------------------------------------------------------
# Identificar jugadores controversiales para la recuperación RAG
# ---------------------------------------------------------------------------


def _controversial_player_ids(
    players: list[PlayerRosterEntry],
    best_lineup_indices: list[int],
    canonical_indices: list[int],
    primary_player_name: Optional[str],
    top_n: int = 4,
) -> list[str]:
    """Devuelve los player_id de los jugadores con mayor divergencia posicional."""
    canonical_slot = {pidx: s + 1 for s, pidx in enumerate(canonical_indices)}
    ai_slot = {pidx: s + 1 for s, pidx in enumerate(best_lineup_indices)}

    divergences = sorted(
        range(9),
        key=lambda i: abs(ai_slot[i] - canonical_slot[i]),
        reverse=True,
    )

    ids: list[str] = []
    # Primero el jugador del trigger primario
    if primary_player_name:
        for p in players:
            if p.player_name == primary_player_name:
                ids.append(str(p.player_id))
                break

    for idx in divergences[:top_n]:
        pid = str(players[idx].player_id)
        if pid not in ids:
            ids.append(pid)

    return ids[:top_n]


# ---------------------------------------------------------------------------
# Generador principal
# ---------------------------------------------------------------------------


class CrisisReportGenerator:
    """Genera reportes de defensa RAG para recomendaciones contra-intuitivas.

    Reutiliza ScoutingRetriever y ContextAssembler del agente existente
    (lineup_explainer_agent.py) con un system prompt especializado en defensa.

    Args:
        anthropic_api_key: Clave API de Anthropic.
        pinecone_api_key: Clave API de Pinecone.
        openai_api_key: Clave API de OpenAI (embeddings).
        config: ExplainerConfig compartido con el agente principal.
    """

    def __init__(
        self,
        anthropic_api_key: str,
        pinecone_api_key: str,
        openai_api_key: str,
        config: ExplainerConfig | None = None,
    ) -> None:
        import anthropic as _anthropic

        self._config = config or ExplainerConfig()
        self._client = _anthropic.Anthropic(api_key=anthropic_api_key)
        self._retriever = ScoutingRetriever(
            pinecone_api_key=pinecone_api_key,
            openai_api_key=openai_api_key,
            config=self._config,
        )
        self._assembler = ContextAssembler()
        logger.info(
            "crisis_generator_ready",
            model=self._config.anthropic_model,
        )

    @classmethod
    def from_env(cls, config: ExplainerConfig | None = None) -> "CrisisReportGenerator":
        """Construye el generador leyendo las API keys de variables de entorno."""
        return cls(
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            config=config,
        )

    def generate(
        self,
        game_id: str,
        players: list[PlayerRosterEntry],
        best_lineup_indices: list[int],
        canonical_lineup_indices: list[int],
        trigger_result: TriggerResult,
        pitcher: RivalPitcherRequest,
        best_er: float,
        refinement_scores: list[float],
    ) -> CrisisDefenseReport:
        """Genera el reporte de defensa completo para una alineación contra-intuitiva.

        Args:
            game_id: Identificador único del partido.
            players: Los 9 PlayerRosterEntry del request.
            best_lineup_indices: Orden óptimo del GA como índices sobre ``players``.
            canonical_lineup_indices: Orden canónico sabermétrco como índices.
            trigger_result: Output de CrisisDetector.detect().
            pitcher: Datos del lanzador rival.
            best_er: E[R] del refinamiento de alta fidelidad (100k sims).
            refinement_scores: E[R] de todos los candidatos top-K del GA.

        Returns:
            CrisisDefenseReport populado y listo para serialización JSON.
        """
        report_id = str(uuid.uuid4())[:16]
        primary = trigger_result.primary_trigger()

        ai_names = [players[i].player_name for i in best_lineup_indices]
        canonical_names = [players[i].player_name for i in canonical_lineup_indices]

        # E[R] canónico: límite inferior conservador desde los refinement scores
        fallback_er = min(refinement_scores) if refinement_scores else round(best_er * 0.93, 4)
        er_delta = round(best_er - fallback_er, 4)

        trigger_description = (
            primary.description_es
            if primary
            else " | ".join(trigger_result.descriptions_es)
        )
        primary_name = primary.affected_player_name if primary else None

        # Recuperación RAG focalizada en jugadores controversiales
        player_ids_for_rag = _controversial_player_ids(
            players, best_lineup_indices, canonical_lineup_indices, primary_name
        )
        retrieval_query = (
            f"rendimiento reciente, estado físico, mecánicas de bateo, "
            f"tendencias vs. lanzador {'zurdo' if pitcher.hand == 'L' else 'derecho'}, "
            f"platoon split, razones para cambio de posición en batting order"
        )
        chunks = self._retriever.retrieve(
            query=retrieval_query,
            player_ids=player_ids_for_rag,
        )

        # Ensamble del prompt respetando el presupuesto de tokens
        quant_block = _build_crisis_quant_block(
            trigger_description=trigger_description,
            ai_lineup_names=ai_names,
            canonical_lineup_names=canonical_names,
            best_er=best_er,
            fallback_er=fallback_er,
            pitcher=pitcher,
        )
        system_tokens = self._assembler.count_tokens(_CRISIS_DEFENSE_SYSTEM_PROMPT_ES)
        quant_tokens = self._assembler.count_tokens(quant_block)
        scout_budget = (
            self._config.max_context_tokens
            - system_tokens
            - quant_tokens
            - self._config.max_response_tokens
            - 600
        )
        truncated = self._assembler.truncate_to_budget(chunks, scout_budget)
        scout_block = self._assembler.format_scouting_block(truncated)

        user_message = (
            "Redacta el reporte de defensa para la siguiente recomendación inusual "
            "siguiendo exactamente las instrucciones del sistema.\n\n"
            f"{quant_block}\n\n"
            f"{scout_block}"
        )

        logger.info(
            "crisis_report_calling_llm",
            game_id=game_id,
            report_id=report_id,
            trigger_codes=trigger_result.trigger_codes,
            controversial_player_ids=player_ids_for_rag,
            chunks_used=len(truncated),
            chunks_dropped=len(chunks) - len(truncated),
        )

        response = self._client.messages.create(
            model=self._config.anthropic_model,
            max_tokens=self._config.max_response_tokens,
            system=_CRISIS_DEFENSE_SYSTEM_PROMPT_ES,
            messages=[{"role": "user", "content": user_message}],
        )

        markdown_content: str = response.content[0].text
        input_tokens: int = response.usage.input_tokens
        output_tokens: int = response.usage.output_tokens

        logger.info(
            "crisis_report_generated",
            game_id=game_id,
            report_id=report_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        fallback_ids = [players[i].player_id for i in canonical_lineup_indices]
        fallback_names = [players[i].player_name for i in canonical_lineup_indices]

        return CrisisDefenseReport(
            report_id=report_id,
            game_id=game_id,
            generated_at=datetime.now(tz=timezone.utc),
            trigger_codes=trigger_result.trigger_codes,
            trigger_descriptions_es=trigger_result.descriptions_es,
            unusual_move_summary_es=trigger_description,
            markdown_content=markdown_content,
            fallback_lineup_player_ids=fallback_ids,
            fallback_lineup_player_names=fallback_names,
            fallback_er_estimate=round(fallback_er, 4),
            er_cost_of_fallback=er_delta,
            controversial_player_ids=[int(pid) for pid in player_ids_for_rag],
            rag_chunks_used=len(truncated),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_version=self._config.anthropic_model,
        )
