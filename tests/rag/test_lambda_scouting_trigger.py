"""
Tests for lambda_scouting_trigger — S3 key parsing, text extraction, and handler flow.

All S3 and ScoutingIngestionPipeline calls are mocked.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from src.rag.lambda_scouting_trigger import (
    _extract_text_txt,
    _parse_s3_key,
    handler,
)


# ---------------------------------------------------------------------------
# _parse_s3_key
# ---------------------------------------------------------------------------


def test_parse_valid_txt_key_returns_dict():
    key = "scouting/NYY/pro_scout/592450_Aaron_Judge_2026-05-18.txt"
    result = _parse_s3_key(key)
    assert result is not None
    assert result["team_iso"] == "NYY"
    assert result["scout_type"] == "pro_scout"
    assert result["player_id"] == "592450"
    assert result["report_date"] == "2026-05-18"
    assert result["ext"] == "txt"


def test_parse_valid_pdf_key_returns_dict():
    key = "scouting/LAD/medical/660271_Shohei_Ohtani_2026-05-01.pdf"
    result = _parse_s3_key(key)
    assert result is not None
    assert result["ext"] == "pdf"
    assert result["player_id"] == "660271"


def test_parse_wrong_prefix_returns_none():
    key = "reports/NYY/pro_scout/592450_Aaron_Judge_2026-05-18.txt"
    assert _parse_s3_key(key) is None


def test_parse_wrong_extension_returns_none():
    key = "scouting/NYY/pro_scout/592450_Aaron_Judge_2026-05-18.docx"
    assert _parse_s3_key(key) is None


def test_parse_unknown_scout_type_returns_none():
    key = "scouting/NYY/secret_intel/592450_Aaron_Judge_2026-05-18.txt"
    assert _parse_s3_key(key) is None


def test_parse_team_iso_normalized_to_uppercase():
    key = "scouting/nyy/pro_scout/592450_Aaron_Judge_2026-05-18.txt"
    result = _parse_s3_key(key)
    assert result is not None
    assert result["team_iso"] == "NYY"


def test_parse_player_name_underscores_converted_to_spaces():
    key = "scouting/NYY/pro_scout/592450_Aaron_Judge_2026-05-18.txt"
    result = _parse_s3_key(key)
    assert result is not None
    assert result["player_name"] == "Aaron Judge"


def test_parse_multi_word_name_reconstructed():
    key = "scouting/BOS/analytics/545361_Raphael_Devers_Jr_2026-05-18.txt"
    result = _parse_s3_key(key)
    assert result is not None
    assert result["player_id"] == "545361"


def test_parse_all_valid_scout_types():
    for scout_type in ("pro_scout", "medical", "developmental", "analytics"):
        key = f"scouting/NYY/{scout_type}/123_Test_Player_2026-01-01.txt"
        assert _parse_s3_key(key) is not None, f"Expected {scout_type} to be valid"


# ---------------------------------------------------------------------------
# _extract_text_txt
# ---------------------------------------------------------------------------


def test_extract_text_utf8_returns_decoded_string():
    raw = "Aaron Judge está en forma.\n".encode("utf-8")
    assert "Aaron Judge" in _extract_text_txt(raw)


def test_extract_text_latin1_fallback_on_utf8_error():
    # byte 0xe9 is valid latin-1 but invalid UTF-8
    raw = b"Pitcher f\xe9vrier report"
    result = _extract_text_txt(raw)
    assert "Pitcher" in result
    assert "vrier" in result  # é decoded as latin-1


# ---------------------------------------------------------------------------
# handler — S3 event routing
# ---------------------------------------------------------------------------


def _make_s3_event(bucket: str, key: str) -> dict:
    return {
        "Records": [
            {"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}
        ]
    }


def _make_multi_event(*keys: str, bucket: str = "mlb-scouting") -> dict:
    return {
        "Records": [
            {"s3": {"bucket": {"name": bucket}, "object": {"key": k}}}
            for k in keys
        ]
    }


@pytest.fixture
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.ingest_report.return_value = 3
    return pipeline


def test_handler_valid_txt_event_calls_ingest(mock_pipeline: MagicMock):
    bucket = "mlb-scouting"
    key = "scouting/NYY/pro_scout/592450_Aaron_Judge_2026-05-18.txt"
    content = b"Great swing mechanics today."

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": io.BytesIO(content)}

    with (
        patch("src.rag.lambda_scouting_trigger.boto3") as mock_boto3,
        patch("src.rag.lambda_scouting_trigger._build_pipeline", return_value=mock_pipeline),
    ):
        mock_boto3.client.return_value = mock_s3
        result = handler(_make_s3_event(bucket, key), None)

    mock_pipeline.ingest_report.assert_called_once()
    assert result["processed"] == 1
    assert result["skipped"] == 0
    assert result["failed"] == 0


def test_handler_non_matching_key_is_skipped(mock_pipeline: MagicMock):
    bucket = "mlb-scouting"
    key = "uploads/misc/some_random_file.txt"

    mock_s3 = MagicMock()

    with (
        patch("src.rag.lambda_scouting_trigger.boto3") as mock_boto3,
        patch("src.rag.lambda_scouting_trigger._build_pipeline", return_value=mock_pipeline),
    ):
        mock_boto3.client.return_value = mock_s3
        result = handler(_make_s3_event(bucket, key), None)

    mock_pipeline.ingest_report.assert_not_called()
    assert result["skipped"] == 1
    assert result["processed"] == 0


def test_handler_mixed_batch_counts_correctly(mock_pipeline: MagicMock):
    bucket = "mlb-scouting"
    valid_key = "scouting/NYY/pro_scout/592450_Aaron_Judge_2026-05-18.txt"
    invalid_key = "raw/NYY/dump.txt"

    content = b"Good report."
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": io.BytesIO(content)}

    with (
        patch("src.rag.lambda_scouting_trigger.boto3") as mock_boto3,
        patch("src.rag.lambda_scouting_trigger._build_pipeline", return_value=mock_pipeline),
    ):
        mock_boto3.client.return_value = mock_s3
        result = handler(_make_multi_event(valid_key, invalid_key, bucket=bucket), None)

    assert result["processed"] == 1
    assert result["skipped"] == 1
    assert result["failed"] == 0


def test_handler_single_record_ingestion_failure_reraises(mock_pipeline: MagicMock):
    bucket = "mlb-scouting"
    key = "scouting/NYY/pro_scout/592450_Aaron_Judge_2026-05-18.txt"
    content = b"Report content."

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": io.BytesIO(content)}
    mock_pipeline.ingest_report.side_effect = RuntimeError("Pinecone connection failed")

    with (
        patch("src.rag.lambda_scouting_trigger.boto3") as mock_boto3,
        patch("src.rag.lambda_scouting_trigger._build_pipeline", return_value=mock_pipeline),
    ):
        mock_boto3.client.return_value = mock_s3
        with pytest.raises(RuntimeError, match="Pinecone"):
            handler(_make_s3_event(bucket, key), None)


def test_handler_multi_record_ingestion_failure_does_not_reraise(mock_pipeline: MagicMock):
    """With multiple records, a single failure increments 'failed' but doesn't re-raise."""
    bucket = "mlb-scouting"
    keys = [
        "scouting/NYY/pro_scout/592450_Aaron_Judge_2026-05-18.txt",
        "scouting/BOS/medical/545361_Xander_Bogaerts_2026-05-18.txt",
    ]
    content = b"Report content."
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": io.BytesIO(content)}
    mock_pipeline.ingest_report.side_effect = [3, RuntimeError("Pinecone error")]

    with (
        patch("src.rag.lambda_scouting_trigger.boto3") as mock_boto3,
        patch("src.rag.lambda_scouting_trigger._build_pipeline", return_value=mock_pipeline),
    ):
        mock_boto3.client.return_value = mock_s3
        result = handler(_make_multi_event(*keys, bucket=bucket), None)

    assert result["processed"] == 1
    assert result["failed"] == 1


def test_handler_ingest_receives_correct_player_id(mock_pipeline: MagicMock):
    """ScoutReport passed to ingest_report has the player_id extracted from the key."""
    bucket = "mlb-scouting"
    key = "scouting/NYY/pro_scout/592450_Aaron_Judge_2026-05-18.txt"
    content = b"Judge is crushing it."

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": io.BytesIO(content)}

    with (
        patch("src.rag.lambda_scouting_trigger.boto3") as mock_boto3,
        patch("src.rag.lambda_scouting_trigger._build_pipeline", return_value=mock_pipeline),
    ):
        mock_boto3.client.return_value = mock_s3
        handler(_make_s3_event(bucket, key), None)

    call_args = mock_pipeline.ingest_report.call_args[0][0]
    assert call_args.player_id == "592450"
    assert call_args.team_iso == "NYY"
