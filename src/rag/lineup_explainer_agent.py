"""
Lineup Explainer Agent — RAG-powered Tactical Briefing generator.

Bridges the quantitative output of the Phase 3 optimizer (Monte Carlo + GA)
with qualitative scouting context retrieved from Pinecone to produce a
structured pre-game Tactical Briefing for the Manager.

Flow:
    1. Accept top-3 LineupOptions + RivalPitcherContext + WeatherContext
    2. Extract unique player_ids from all three lineups
    3. Query Pinecone with metadata filters:
           player_id IN [...] AND report_date_ts >= (today - 30 days)
    4. Assemble context within a strict token budget
    5. Call Claude (claude-sonnet-4-6) with Bench Coach system prompt
    6. Return TacticalBriefing dataclass with full provenance
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import anthropic
import structlog
import tiktoken
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

logger = structlog.get_logger(__name__)

_TIKTOKEN_ENCODING = "cl100k_base"


# ---------------------------------------------------------------------------
# Input contracts (bridge from Phase 3 OptimizerResult)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineupOption:
    """Single lineup candidate produced by the GA optimizer."""

    rank: int                                     # 1 = best by E[R]
    player_ids: list[str]                         # index 0 = lead-off, index 8 = 9-hole
    player_names: list[str]
    expected_runs: float
    win_probability: float
    batting_order_notes: dict[int, str] = field(default_factory=dict)  # batting_pos → note


@dataclass(frozen=True)
class RivalPitcherContext:
    pitcher_id: str
    pitcher_name: str
    hand: str                      # "L" or "R"
    primary_pitch: str             # e.g. "SL", "FF"
    pitch_mix: dict[str, float]    # pitch_type → usage fraction (sums to ~1.0)
    era: float
    whip: float
    fip: float
    k_per_9: float
    bb_per_9: float


@dataclass(frozen=True)
class WeatherContext:
    venue_name: str
    temperature_f: float
    humidity_pct: float
    wind_speed_mph: float
    wind_direction_deg: float
    park_factor_hr: float
    park_factor_xb: float
    precipitation_prob: float = 0.0


@dataclass
class RetrievedChunk:
    player_id: str
    player_name: str
    scout_type: str
    report_date: str
    content: str
    relevance_score: float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExplainerConfig:
    anthropic_model: str = "claude-sonnet-4-6"
    max_response_tokens: int = 4_096
    # Conservative context budget: keeps cost predictable on every call
    max_context_tokens: int = 40_000
    retrieval_top_k: int = 12         # Pinecone chunks retrieved per query
    lookback_days: int = 30           # metadata filter window
    pinecone_index_name: str = "mlb-scouting"
    pinecone_namespace: str = "scouting"
    embedding_model: str = "text-embedding-3-large"
    briefing_language: str = "es"     # "es" → Spanish (default), "en" → English


# ---------------------------------------------------------------------------
# System prompts (Bench Coach persona)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_ES = """\
Eres el Bench Coach de las Grandes Ligas con más de 20 años de experiencia \
integrando análisis saberométrico avanzado con estrategia táctica en campo. \
Tu responsabilidad es redactar el Tactical Briefing pre-partido para el Mánager, \
combinando los resultados del motor cuantitativo de optimización de lineup \
(Monte Carlo + Algoritmo Genético Evolutivo) con los reportes cualitativos más \
recientes de los scouts.

REGLAS DE RESPUESTA — NO NEGOCIABLES:
1. Basa tus conclusiones EXCLUSIVAMENTE en los datos cuantitativos y el contexto \
de scouts proporcionados en este mensaje. NUNCA inventes estadísticas, nombres, \
fechas ni eventos que no estén presentes en el contexto suministrado.
2. Si existe contradicción entre un reporte médico/scout y la recomendación \
cuantitativa (ej. molestia leve vs. alta proyección ofensiva), reconócela \
explícitamente en la sección "Alertas de Scout" y justifica la decisión final \
con métricas concretas: wOBA esperado, E[R], P(W).
3. Estructura tu respuesta en Markdown con las secciones obligatorias:
   ## Resumen Ejecutivo          (máx. 5 líneas)
   ## Lineup Recomendado         (tabla: pos | nombre | wOBA proy. | ventaja platoon)
   ## Análisis del Lanzador Rival (vulnerabilidades y matchups clave)
   ## Condiciones del Parque y Clima
   ## Alertas de Scout           (nivel de riesgo: BAJO / MEDIO / ALTO por jugador)
   ## Escenarios Alternativos    (Lineup #2 y #3 con criterios de activación)
4. Para cada jugador del lineup recomendado menciona explícitamente su ventaja \
o desventaja de platoon frente al lanzador rival.
5. Tono profesional y ejecutivo. El Mánager debe poder tomar la decisión en \
menos de 5 minutos de lectura.
6. Si el contexto de scouts de un jugador está ausente o es insuficiente, \
indica "Sin reportes recientes (< 30 días)" — NUNCA inferir ni inventar.\
"""

_SYSTEM_PROMPT_EN = """\
You are a Major League Baseball Bench Coach with over 20 years of experience \
integrating advanced sabermetric analysis with on-field tactical strategy. \
Your responsibility is to draft the pre-game Tactical Briefing for the Manager, \
combining quantitative lineup optimization output (Monte Carlo + Genetic Algorithm) \
with the most recent qualitative scouting reports.

RESPONSE RULES — NON-NEGOTIABLE:
1. Base conclusions EXCLUSIVELY on the quantitative data and scouting context \
provided. NEVER invent statistics, names, dates or events absent from the context.
2. If a contradiction exists between a medical/scout report and the quantitative \
recommendation, acknowledge it explicitly and justify the decision with concrete \
metrics: projected wOBA, E[R], P(W).
3. Structure response in Markdown with required sections: Executive Summary, \
Recommended Lineup, Rival Pitcher Analysis, Park & Weather Conditions, \
Scout Alerts (risk level per player), Alternative Scenarios.
4. For each player in the recommended lineup state their platoon advantage \
or disadvantage against the rival pitcher.
5. Professional and executive tone. Manager decision in under 5 minutes.
6. If scouting context for a player is absent, state "No recent reports (< 30 days)" \
— never infer or fabricate information.\
"""


# ---------------------------------------------------------------------------
# Retrieval layer (LlamaIndex + Pinecone, exact metadata filters)
# ---------------------------------------------------------------------------


class ScoutingRetriever:
    """Filtered vector retrieval: player_id ∈ [...] AND report_date_ts ≥ cutoff."""

    def __init__(
        self,
        pinecone_api_key: str,
        openai_api_key: str,
        config: ExplainerConfig,
    ) -> None:
        # Configure LlamaIndex global embed model to match the ingest pipeline
        Settings.embed_model = OpenAIEmbedding(
            model=config.embedding_model,
            api_key=openai_api_key,
            dimensions=3_072,
        )
        pc = Pinecone(api_key=pinecone_api_key)
        pinecone_index = pc.Index(config.pinecone_index_name)
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            namespace=config.pinecone_namespace,
        )
        # Load existing index — no document ingestion here
        self._llama_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        self._config = config
        logger.info(
            "retriever_initialized",
            index=config.pinecone_index_name,
            namespace=config.pinecone_namespace,
        )

    def retrieve(
        self,
        query: str,
        player_ids: list[str],
        cutoff_date: date | None = None,
    ) -> list[RetrievedChunk]:
        if cutoff_date is None:
            cutoff_date = date.today() - timedelta(days=self._config.lookback_days)

        cutoff_ts = int(
            datetime(
                cutoff_date.year, cutoff_date.month, cutoff_date.day, tzinfo=timezone.utc
            ).timestamp()
        )

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="player_id",
                    operator=FilterOperator.IN,
                    value=player_ids,
                ),
                MetadataFilter(
                    key="report_date_ts",
                    operator=FilterOperator.GTE,
                    value=cutoff_ts,
                ),
            ],
            condition=FilterCondition.AND,
        )
        retriever = self._llama_index.as_retriever(
            similarity_top_k=self._config.retrieval_top_k,
            filters=filters,
        )
        nodes = retriever.retrieve(query)

        chunks: list[RetrievedChunk] = []
        for node in nodes:
            meta: dict[str, Any] = node.metadata or {}
            chunks.append(
                RetrievedChunk(
                    player_id=meta.get("player_id", ""),
                    player_name=meta.get("player_name", ""),
                    scout_type=meta.get("scout_type", ""),
                    report_date=meta.get("report_date", ""),
                    content=meta.get("text", node.text or ""),
                    relevance_score=float(node.score or 0.0),
                )
            )

        logger.info(
            "retrieval_complete",
            n_player_ids=len(player_ids),
            n_chunks_returned=len(chunks),
            cutoff_date=cutoff_date.isoformat(),
        )
        return chunks  # already ordered by descending cosine similarity


# ---------------------------------------------------------------------------
# Context assembler and token budget manager
# ---------------------------------------------------------------------------


class ContextAssembler:
    """Builds the LLM prompt blocks and enforces a hard token budget."""

    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding(_TIKTOKEN_ENCODING)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def truncate_to_budget(
        self, chunks: list[RetrievedChunk], token_budget: int
    ) -> list[RetrievedChunk]:
        """Keep highest-relevance chunks that fit within token_budget."""
        selected: list[RetrievedChunk] = []
        used = 0
        for chunk in chunks:
            tokens = self.count_tokens(chunk.content)
            if used + tokens > token_budget:
                logger.warning(
                    "context_budget_hit",
                    selected_chunks=len(selected),
                    dropped_chunks=len(chunks) - len(selected),
                    tokens_used=used,
                    token_budget=token_budget,
                )
                break
            selected.append(chunk)
            used += tokens
        return selected

    def format_quantitative_block(
        self,
        lineups: list[LineupOption],
        pitcher: RivalPitcherContext,
        weather: WeatherContext,
    ) -> str:
        lines: list[str] = ["## DATOS CUANTITATIVOS DEL OPTIMIZADOR\n"]

        for lineup in lineups:
            lines.append(
                f"### Lineup #{lineup.rank}  —  "
                f"E[R] = {lineup.expected_runs:.3f}  |  "
                f"P(W) = {lineup.win_probability:.2%}"
            )
            for pos, (pid, pname) in enumerate(
                zip(lineup.player_ids, lineup.player_names), start=1
            ):
                note = lineup.batting_order_notes.get(pos, "")
                lines.append(f"  {pos}. {pname}  [{pid}]" + (f"  — {note}" if note else ""))
            lines.append("")

        pitch_mix_str = "  |  ".join(
            f"{pt}: {pct:.1%}" for pt, pct in pitcher.pitch_mix.items()
        )
        lines += [
            "## LANZADOR RIVAL",
            f"Nombre: {pitcher.pitcher_name}  |  Mano: {pitcher.hand}  |  "
            f"ERA: {pitcher.era:.2f}  |  WHIP: {pitcher.whip:.2f}  |  "
            f"FIP: {pitcher.fip:.2f}  |  K/9: {pitcher.k_per_9:.1f}  |  "
            f"BB/9: {pitcher.bb_per_9:.1f}",
            f"Pitch Principal: {pitcher.primary_pitch}  |  Arsenal: {pitch_mix_str}",
            "",
            "## CONDICIONES DEL PARQUE",
            f"Estadio: {weather.venue_name}  |  "
            f"Temperatura: {weather.temperature_f:.0f}°F  |  "
            f"Humedad: {weather.humidity_pct:.0f}%",
            f"Viento: {weather.wind_speed_mph:.1f} mph @ {weather.wind_direction_deg:.0f}°  |  "
            f"P(lluvia): {weather.precipitation_prob:.1%}",
            f"Park Factor HR: {weather.park_factor_hr:.3f}  |  "
            f"Park Factor XB: {weather.park_factor_xb:.3f}",
        ]
        return "\n".join(lines)

    def format_scouting_block(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return (
                "## REPORTES DE SCOUTS\n"
                "_No se encontraron reportes recientes (< 30 días) para los jugadores seleccionados._"
            )

        # Group by player for readability
        by_player: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            by_player.setdefault(chunk.player_id, []).append(chunk)

        lines: list[str] = ["## REPORTES DE SCOUTS (ventana: últimos 30 días)\n"]
        for player_id, player_chunks in by_player.items():
            pname = player_chunks[0].player_name or player_id
            lines.append(f"### {pname}  [{player_id}]")
            for chunk in player_chunks:
                lines.append(
                    f"**[{chunk.scout_type.upper()}  |  {chunk.report_date}]**  "
                    f"(score de relevancia: {chunk.relevance_score:.3f})"
                )
                lines.append(chunk.content)
                lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


@dataclass
class TacticalBriefing:
    markdown_content: str
    recommended_lineup: LineupOption
    alternative_lineups: list[LineupOption]
    retrieved_chunks: list[RetrievedChunk]
    input_tokens: int
    output_tokens: int
    model_version: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def display(self) -> None:
        print(self.markdown_content)
        print(
            f"\n---\n"
            f"Modelo: {self.model_version}  |  "
            f"Tokens: {self.input_tokens} entrada / {self.output_tokens} salida  |  "
            f"Chunks recuperados: {len(self.retrieved_chunks)}  |  "
            f"Generado: {self.generated_at}"
        )


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------


class LineupExplainerAgent:
    """RAG agent: retrieves scouting context and calls Claude to write the briefing.

    Usage:
        agent = LineupExplainerAgent.from_env()
        briefing = agent.generate_briefing(top_lineups, pitcher_ctx, weather_ctx)
        briefing.display()
    """

    def __init__(
        self,
        anthropic_api_key: str,
        pinecone_api_key: str,
        openai_api_key: str,
        config: ExplainerConfig | None = None,
    ) -> None:
        self._config = config or ExplainerConfig()
        self._anthropic = anthropic.Anthropic(api_key=anthropic_api_key)
        self._retriever = ScoutingRetriever(
            pinecone_api_key=pinecone_api_key,
            openai_api_key=openai_api_key,
            config=self._config,
        )
        self._assembler = ContextAssembler()
        logger.info(
            "explainer_agent_ready",
            model=self._config.anthropic_model,
            language=self._config.briefing_language,
        )

    @classmethod
    def from_env(cls, config: ExplainerConfig | None = None) -> "LineupExplainerAgent":
        return cls(
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            config=config,
        )

    def _system_prompt(self) -> str:
        return _SYSTEM_PROMPT_ES if self._config.briefing_language == "es" else _SYSTEM_PROMPT_EN

    @staticmethod
    def _unique_player_ids(lineups: list[LineupOption]) -> list[str]:
        seen: set[str] = set()
        ids: list[str] = []
        for lineup in lineups:
            for pid in lineup.player_ids:
                if pid not in seen:
                    seen.add(pid)
                    ids.append(pid)
        return ids

    def _build_retrieval_query(
        self, pitcher: RivalPitcherContext, weather: WeatherContext
    ) -> str:
        return (
            f"rendimiento ofensivo contra lanzador {pitcher.hand}HP, "
            f"pitch principal {pitcher.primary_pitch}, "
            f"estadio {weather.venue_name}, viento {weather.wind_speed_mph:.0f} mph, "
            f"estado físico, mecánicas, tendencias recientes, platoon split"
        )

    def generate_briefing(
        self,
        lineups: list[LineupOption],
        pitcher: RivalPitcherContext,
        weather: WeatherContext,
        override_cutoff_date: date | None = None,
    ) -> TacticalBriefing:
        """Generate a Tactical Briefing Markdown document for the Manager.

        Args:
            lineups: 1–3 LineupOption objects ranked by E[R] (rank=1 is best).
            pitcher: Rival starter context.
            weather: Park and weather conditions.
            override_cutoff_date: Override the 30-day lookback window if needed.

        Returns:
            TacticalBriefing with full markdown content and provenance metadata.
        """
        if not lineups:
            raise ValueError("At least one LineupOption is required")

        lineups = sorted(lineups, key=lambda x: x.rank)[:3]

        # Step 1: Retrieve scouting context
        player_ids = self._unique_player_ids(lineups)
        query = self._build_retrieval_query(pitcher, weather)
        chunks = self._retriever.retrieve(
            query=query,
            player_ids=player_ids,
            cutoff_date=override_cutoff_date,
        )

        # Step 2: Build quantitative block (always fits — no truncation needed)
        system_prompt = self._system_prompt()
        quant_block = self._assembler.format_quantitative_block(lineups, pitcher, weather)

        # Step 3: Compute remaining token budget for scouting block
        system_tokens = self._assembler.count_tokens(system_prompt)
        quant_tokens = self._assembler.count_tokens(quant_block)
        overhead = 600  # instruction prefix + formatting tokens
        scout_budget = (
            self._config.max_context_tokens
            - system_tokens
            - quant_tokens
            - self._config.max_response_tokens
            - overhead
        )
        truncated_chunks = self._assembler.truncate_to_budget(chunks, scout_budget)
        scout_block = self._assembler.format_scouting_block(truncated_chunks)

        # Step 4: Assemble user message
        user_message = (
            "A continuación encontrarás los datos cuantitativos del optimizador "
            "y los reportes de scouts más recientes. "
            "Redacta el Tactical Briefing siguiendo las instrucciones del sistema.\n\n"
            f"{quant_block}\n\n"
            f"{scout_block}"
        )

        total_estimated_tokens = system_tokens + self._assembler.count_tokens(user_message)
        logger.info(
            "calling_llm",
            model=self._config.anthropic_model,
            estimated_input_tokens=total_estimated_tokens,
            scout_chunks_used=len(truncated_chunks),
            scout_chunks_dropped=len(chunks) - len(truncated_chunks),
            players=len(player_ids),
        )

        # Step 5: Call Claude
        response = self._anthropic.messages.create(
            model=self._config.anthropic_model,
            max_tokens=self._config.max_response_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        markdown_content: str = response.content[0].text
        input_tokens: int = response.usage.input_tokens
        output_tokens: int = response.usage.output_tokens

        logger.info(
            "briefing_generated",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._config.anthropic_model,
        )

        return TacticalBriefing(
            markdown_content=markdown_content,
            recommended_lineup=lineups[0],
            alternative_lineups=lineups[1:],
            retrieved_chunks=truncated_chunks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_version=self._config.anthropic_model,
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def generate_tactical_briefing(
    lineups: list[LineupOption],
    pitcher: RivalPitcherContext,
    weather: WeatherContext,
    config: ExplainerConfig | None = None,
) -> TacticalBriefing:
    """One-call wrapper that reads all API keys from environment variables."""
    agent = LineupExplainerAgent.from_env(config=config)
    return agent.generate_briefing(lineups, pitcher, weather)
