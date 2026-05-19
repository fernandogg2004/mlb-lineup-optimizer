"""
Scout report ingestion pipeline: token-aware chunking, OpenAI embedding, Pinecone upsert.

Processes free-text scouting documents (injury notes, mechanical observations,
situational tendencies, developmental flags) into searchable vector chunks
enriched with a strict metadata contract for filtered retrieval.

Ingest path:
    ScoutReport → TokenAwareChunker → EmbeddingClient → PineconeVectorStore

Metadata guarantee per vector:
    player_id, player_name, team_iso, scout_type, report_date,
    report_date_ts (Unix, for range queries), chunk_index, total_chunks, text
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import structlog
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

ScoutType = Literal["pro_scout", "medical", "developmental", "analytics"]

_TIKTOKEN_ENCODING = "cl100k_base"  # matches text-embedding-3-large tokenization


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


class ChunkMetadata(BaseModel):
    """Strict metadata persisted alongside every Pinecone vector.

    All fields are flat scalars so Pinecone metadata filters can address them
    directly without nested-key workarounds.
    """

    player_id: str
    player_name: str
    team_iso: str          # e.g. "NYY", "LAD" — normalized to uppercase
    scout_type: ScoutType
    report_date: str       # ISO-8601 string: "2024-06-01"
    report_date_ts: int    # Unix timestamp (UTC) for $gte/$lte range filters
    source_path: str
    chunk_index: int
    total_chunks: int
    text: str              # chunk content stored for retrieval (≤2 000 chars)

    @field_validator("report_date")
    @classmethod
    def _validate_iso_date(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v

    @field_validator("team_iso")
    @classmethod
    def _normalize_team(cls, v: str) -> str:
        return v.upper().strip()


@dataclass(frozen=True)
class ScoutReport:
    """Raw scouting document before chunking."""

    player_id: str
    player_name: str
    team_iso: str
    scout_type: ScoutType
    report_date: date
    content: str
    source_path: str = ""


@dataclass
class IngestionConfig:
    pinecone_index_name: str = "mlb-scouting"
    pinecone_namespace: str = "scouting"
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3_072          # native dimension for text-embedding-3-large
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 200
    upsert_batch_size: int = 100
    embedding_batch_size: int = 64      # safe batch size; OpenAI limit is 2 048
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"


# ---------------------------------------------------------------------------
# Token-aware chunker
# ---------------------------------------------------------------------------


class TokenAwareChunker:
    """Splits text into overlapping fixed-size token windows.

    Uses tiktoken for byte-accurate token counting so every chunk stays within
    the embedding model's 8 191-token context limit.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 200) -> None:
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        self._enc = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        """Return a list of decoded chunk strings with token overlap."""
        tokens = self._enc.encode(text)
        if not tokens:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunks.append(self._enc.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += step
        return chunks

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


# ---------------------------------------------------------------------------
# OpenAI embedding client with retry
# ---------------------------------------------------------------------------


class EmbeddingClient:
    """Batched text-embedding-3-large calls with tenacity retry."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-large") -> None:
        self._client = OpenAI(api_key=api_key)
        self.model = model

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self.model)
        # Preserve insertion order — API returns items sorted by index
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed in micro-batches to stay within rate-limit thresholds."""
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)
            logger.debug(
                "embedding_batch_done",
                batch_start=start,
                batch_size=len(batch),
                cumulative=len(all_embeddings),
            )
        return all_embeddings


# ---------------------------------------------------------------------------
# Pinecone vector store
# ---------------------------------------------------------------------------


class PineconeVectorStore:
    """Pinecone Serverless index wrapper with idempotent creation and bulk upsert."""

    def __init__(self, api_key: str, config: IngestionConfig) -> None:
        self._pc = Pinecone(api_key=api_key)
        self._config = config
        self._index = self._ensure_index()

    def _ensure_index(self):
        existing_names = {idx.name for idx in self._pc.list_indexes().indexes}
        if self._config.pinecone_index_name not in existing_names:
            self._pc.create_index(
                name=self._config.pinecone_index_name,
                dimension=self._config.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self._config.pinecone_cloud,
                    region=self._config.pinecone_region,
                ),
            )
            logger.info(
                "pinecone_index_created",
                index=self._config.pinecone_index_name,
                dimension=self._config.embedding_dim,
            )
        return self._pc.Index(self._config.pinecone_index_name)

    def upsert(
        self,
        vectors: list[list[float]],
        ids: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        if not (len(vectors) == len(ids) == len(metadatas)):
            raise ValueError("vectors, ids, and metadatas must all have the same length")

        batch_size = self._config.upsert_batch_size
        total_upserted = 0
        for start in range(0, len(vectors), batch_size):
            end = start + batch_size
            records = [
                {"id": vid, "values": vec, "metadata": meta}
                for vid, vec, meta in zip(
                    ids[start:end], vectors[start:end], metadatas[start:end]
                )
            ]
            self._index.upsert(
                vectors=records,
                namespace=self._config.pinecone_namespace,
            )
            total_upserted += len(records)
            logger.info(
                "pinecone_upsert_batch",
                batch_count=len(records),
                total_upserted=total_upserted,
            )
        return total_upserted


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------


class ScoutingIngestionPipeline:
    """End-to-end pipeline: ScoutReport list → tokenized chunks → embeddings → Pinecone.

    Usage (from environment variables):
        pipeline = ScoutingIngestionPipeline.from_env()
        n = pipeline.ingest_report(report)

    Usage (explicit keys):
        pipeline = ScoutingIngestionPipeline(
            openai_api_key="sk-...",
            pinecone_api_key="pcsk-...",
        )
        results = pipeline.ingest_directory(Path("data/scouting/"), manifest)
    """

    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: str,
        config: IngestionConfig | None = None,
    ) -> None:
        self._config = config or IngestionConfig()
        self._chunker = TokenAwareChunker(
            chunk_size=self._config.chunk_size_tokens,
            overlap=self._config.chunk_overlap_tokens,
        )
        self._embedder = EmbeddingClient(
            api_key=openai_api_key,
            model=self._config.embedding_model,
        )
        self._store = PineconeVectorStore(
            api_key=pinecone_api_key,
            config=self._config,
        )
        logger.info(
            "scouting_pipeline_ready",
            index=self._config.pinecone_index_name,
            chunk_size=self._config.chunk_size_tokens,
            overlap=self._config.chunk_overlap_tokens,
        )

    @classmethod
    def from_env(cls, config: IngestionConfig | None = None) -> "ScoutingIngestionPipeline":
        return cls(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
            config=config,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_id(player_id: str, report_date: date, chunk_idx: int) -> str:
        """Deterministic 24-char hex ID: SHA-256(player_id|date|chunk_idx)."""
        raw = f"{player_id}|{report_date.isoformat()}|{chunk_idx}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    @staticmethod
    def _report_date_ts(d: date) -> int:
        return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())

    def _build_metadata(
        self,
        report: ScoutReport,
        chunk_idx: int,
        total_chunks: int,
        chunk_text: str,
    ) -> ChunkMetadata:
        return ChunkMetadata(
            player_id=report.player_id,
            player_name=report.player_name,
            team_iso=report.team_iso,
            scout_type=report.scout_type,
            report_date=report.report_date.isoformat(),
            report_date_ts=self._report_date_ts(report.report_date),
            source_path=report.source_path,
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            text=chunk_text[:2_000],  # Pinecone metadata values are capped; clip safely
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_report(self, report: ScoutReport) -> int:
        """Chunk, embed and upsert a single ScoutReport. Returns number of chunks upserted."""
        chunks = self._chunker.chunk(report.content)
        if not chunks:
            logger.warning("empty_report_skipped", player_id=report.player_id)
            return 0

        total = len(chunks)
        ids = [self._chunk_id(report.player_id, report.report_date, i) for i in range(total)]
        metadatas = [
            self._build_metadata(report, i, total, text).model_dump()
            for i, text in enumerate(chunks)
        ]
        embeddings = self._embedder.embed_batch(
            chunks, batch_size=self._config.embedding_batch_size
        )
        upserted = self._store.upsert(
            vectors=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info(
            "report_ingested",
            player_id=report.player_id,
            scout_type=report.scout_type,
            report_date=report.report_date.isoformat(),
            total_chunks=total,
            upserted=upserted,
        )
        return upserted

    def ingest_directory(
        self,
        directory: Path,
        manifest: list[ScoutReport],
    ) -> dict[str, int]:
        """Bulk-ingest from a list of ScoutReport objects. Returns {player_id: chunks_upserted}."""
        results: dict[str, int] = {}
        for report in manifest:
            try:
                results[report.player_id] = self.ingest_report(report)
            except Exception as exc:
                logger.error(
                    "ingest_report_failed",
                    player_id=report.player_id,
                    error=str(exc),
                )
                results[report.player_id] = 0
        return results

    def ingest_text_files(self, directory: Path) -> dict[str, int]:
        """Auto-ingest .txt files following naming convention:
        {player_id}_{team_iso}_{scout_type}_{YYYY-MM-DD}.txt
        """
        results: dict[str, int] = {}
        for path in sorted(directory.glob("*.txt")):
            try:
                report = _parse_filename_convention(path)
                results[str(path)] = self.ingest_report(report)
            except Exception as exc:
                logger.error("ingest_file_failed", path=str(path), error=str(exc))
                results[str(path)] = 0
        return results


def _parse_filename_convention(path: Path) -> ScoutReport:
    """Parse filename: {player_id}_{team_iso}_{scout_type}_{YYYY-MM-DD}.txt"""
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"Filename '{stem}' does not follow convention: "
            "player_id_team_iso_scout_type_YYYY-MM-DD"
        )
    player_id, team_iso, scout_type = parts[0], parts[1], parts[2]
    report_date = date.fromisoformat(parts[3])
    return ScoutReport(
        player_id=player_id,
        player_name=player_id,  # enriched externally if player registry is available
        team_iso=team_iso,
        scout_type=scout_type,  # type: ignore[arg-type]
        report_date=report_date,
        content=path.read_text(encoding="utf-8"),
        source_path=str(path),
    )
