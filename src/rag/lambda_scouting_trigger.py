"""
lambda_scouting_trigger.py — AWS Lambda handler: S3 ObjectCreated → Pinecone ingestion.
=========================================================================================
Triggered by ``s3:ObjectCreated:*`` events on the scouting bucket. Downloads the
object (TXT or PDF), extracts the text, builds a ScoutReport from the S3 key
path convention, and ingests it into Pinecone via the existing ScoutingIngestionPipeline.

Expected S3 key convention:
    scouting/{team_iso}/{scout_type}/{player_id}_{player_name}_{YYYY-MM-DD}.{txt|pdf}

Example:
    scouting/NYY/pro_scout/592450_Aaron_Judge_2026-05-18.txt

If the key does not match the convention the Lambda logs a warning and returns
without error so the event is not retried.

PDF text extraction uses pypdf (pure Python, no system dependencies required in Lambda).

Required environment variables:
    OPENAI_API_KEY        — for EmbeddingClient
    PINECONE_API_KEY      — for PineconeVectorStore
    PINECONE_INDEX_NAME   — default: mlb-scouting
    PINECONE_NAMESPACE    — default: scouting

Dead-letter queue: configure an SQS DLQ on the Lambda to capture parse/ingestion
failures without losing the S3 event.
"""
from __future__ import annotations

import io
import os
import re
from datetime import date
from typing import Any

import boto3
import structlog

from src.rag.scouting_rag_ingest import (
    IngestionConfig,
    ScoutReport,
    ScoutingIngestionPipeline,
    ScoutType,
)

logger = structlog.get_logger(__name__)

# S3 key pattern: scouting/{team_iso}/{scout_type}/{player_id}_{slug}_{YYYY-MM-DD}.{ext}
_KEY_PATTERN = re.compile(
    r"^scouting/"
    r"(?P<team_iso>[A-Za-z0-9]+)/"
    r"(?P<scout_type>[a-z_]+)/"
    r"(?P<player_id>\d+)_"
    r"(?P<player_name_slug>[^_]+(?:_[^_]+)*)_"
    r"(?P<report_date>\d{4}-\d{2}-\d{2})"
    r"\.(?P<ext>txt|pdf)$"
)

_VALID_SCOUT_TYPES: set[str] = {"pro_scout", "medical", "developmental", "analytics"}


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def _extract_text_txt(raw_bytes: bytes) -> str:
    """Decode UTF-8 text file, falling back to latin-1 on decode errors.

    Args:
        raw_bytes: Raw bytes downloaded from S3.

    Returns:
        Decoded text string.
    """
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1")


def _extract_text_pdf(raw_bytes: bytes) -> str:
    """Extract plain text from a PDF using pypdf.

    Args:
        raw_bytes: Raw PDF bytes downloaded from S3.

    Returns:
        Concatenated text from all pages.

    Raises:
        ImportError: If pypdf is not installed in the Lambda layer.
        ValueError: If the PDF has no extractable text (scanned/image-only).
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF text extraction in the Lambda layer.") from exc

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    if not pages:
        raise ValueError("PDF contains no extractable text (may be image-only scan).")

    return "\n".join(pages)


def _extract_text(raw_bytes: bytes, extension: str) -> str:
    """Route to the correct extractor based on file extension.

    Args:
        raw_bytes: Raw file bytes.
        extension: File extension without dot (``txt`` or ``pdf``).

    Returns:
        Extracted text content.

    Raises:
        ValueError: For unsupported extensions.
    """
    if extension == "txt":
        return _extract_text_txt(raw_bytes)
    if extension == "pdf":
        return _extract_text_pdf(raw_bytes)
    raise ValueError(f"Unsupported file extension: {extension!r}")


# ---------------------------------------------------------------------------
# S3 key parsing
# ---------------------------------------------------------------------------


def _parse_s3_key(key: str) -> dict[str, str] | None:
    """Parse the S3 object key into its structural components.

    Args:
        key: S3 object key, e.g.
             ``scouting/NYY/pro_scout/592450_Aaron_Judge_2026-05-18.txt``.

    Returns:
        Dict with keys: team_iso, scout_type, player_id, player_name, report_date, ext.
        Returns None if the key does not match the convention.
    """
    match = _KEY_PATTERN.match(key)
    if not match:
        return None

    scout_type = match.group("scout_type")
    if scout_type not in _VALID_SCOUT_TYPES:
        logger.warning("unknown_scout_type", scout_type=scout_type, key=key)
        return None

    player_name = match.group("player_name_slug").replace("_", " ")

    return {
        "team_iso": match.group("team_iso").upper(),
        "scout_type": scout_type,
        "player_id": match.group("player_id"),
        "player_name": player_name,
        "report_date": match.group("report_date"),
        "ext": match.group("ext"),
    }


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entry point.

    Processes every S3 ObjectCreated record in the event. Returns a summary
    dict; Lambda does not use the return value for S3 triggers but it aids
    local testing and CloudWatch logs.

    Args:
        event: Lambda event dict from the S3 trigger.
        context: Lambda runtime context (unused).

    Returns:
        Dict with keys: ``processed`` (int), ``skipped`` (int), ``failed`` (int).
    """
    s3 = boto3.client("s3")
    pipeline = _build_pipeline()

    processed = 0
    skipped = 0
    failed = 0

    records: list[dict[str, Any]] = event.get("Records", [])
    logger.info("lambda_invoked", n_records=len(records))

    for record in records:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        log = logger.bind(bucket=bucket, key=key)

        parsed = _parse_s3_key(key)
        if parsed is None:
            log.warning("s3_key_skipped", reason="key does not match scouting convention")
            skipped += 1
            continue

        try:
            raw_bytes = _download_object(s3, bucket, key)
            text = _extract_text(raw_bytes, parsed["ext"])

            report = ScoutReport(
                player_id=parsed["player_id"],
                player_name=parsed["player_name"],
                team_iso=parsed["team_iso"],
                scout_type=parsed["scout_type"],  # type: ignore[arg-type]
                report_date=date.fromisoformat(parsed["report_date"]),
                content=text,
                source_path=f"s3://{bucket}/{key}",
            )

            n_chunks = pipeline.ingest_report(report)
            log.info(
                "scouting_report_ingested",
                player_id=parsed["player_id"],
                player_name=parsed["player_name"],
                n_chunks=n_chunks,
            )
            processed += 1

        except Exception as exc:
            log.error("scouting_ingestion_failed", error=str(exc), exc_info=True)
            failed += 1
            # Re-raise to trigger DLQ if there is only one record
            if len(records) == 1:
                raise

    result = {"processed": processed, "skipped": skipped, "failed": failed}
    logger.info("lambda_complete", **result)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_object(s3_client: Any, bucket: str, key: str) -> bytes:
    """Download an S3 object and return its raw bytes.

    Args:
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        key: S3 object key.

    Returns:
        Raw bytes of the object body.

    Raises:
        botocore.exceptions.ClientError: On S3 access or not-found errors.
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def _build_pipeline() -> ScoutingIngestionPipeline:
    """Construct the ingestion pipeline from environment variables.

    Returns:
        Configured ScoutingIngestionPipeline ready to ingest reports.

    Raises:
        KeyError: If OPENAI_API_KEY or PINECONE_API_KEY are not set.
    """
    config = IngestionConfig(
        pinecone_index_name=os.environ.get("PINECONE_INDEX_NAME", "mlb-scouting"),
        pinecone_namespace=os.environ.get("PINECONE_NAMESPACE", "scouting"),
    )
    return ScoutingIngestionPipeline.from_env(config=config)
