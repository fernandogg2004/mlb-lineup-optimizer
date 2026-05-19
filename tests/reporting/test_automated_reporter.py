"""
Tests for AutomatedReporter — chart building, PDF generation, and SES distribution.

kaleido (PDF rendering), pypdf (PDF merging), and AWS SES are mocked throughout
so the suite runs in any environment without heavy optional dependencies.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import plotly.graph_objects as go
import pytest

from src.reporting.automated_reporter import (
    AutomatedReporter,
    ReporterConfig,
    _build_adoption_trend_chart,
    _build_cover_page,
    _build_divergence_type_chart,
    _build_er_accuracy_chart,
    _month_label_es,
    generate_monthly_report,
)

# ---------------------------------------------------------------------------
# Canonical summary dict used across chart tests
# ---------------------------------------------------------------------------

FULL_SUMMARY: dict = {
    "lookback_days": 30,
    "total_closed_games": 20,
    "by_divergence_type": {"order_only": 9, "player_swap": 7, "full_override": 4},
    "avg_ai_expected_runs": 4.50,
    "avg_actual_runs_scored": 5.10,
    "manager_advantage_runs": 0.60,
    "sample_too_small": False,
    "interpretation_es": "El lineup del mánager supera la expectativa del modelo.",
}

EMPTY_SUMMARY: dict = {
    "lookback_days": 30,
    "total_closed_games": 0,
    "by_divergence_type": {"order_only": 0, "player_swap": 0, "full_override": 0},
    "avg_ai_expected_runs": None,
    "avg_actual_runs_scored": None,
    "manager_advantage_runs": None,
    "sample_too_small": True,
    "interpretation_es": "Sin datos suficientes.",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reporter(tmp_path: Path) -> AutomatedReporter:
    config = ReporterConfig(
        store_path=tmp_path / "experiment",
        ses_from_email="from@test.com",
        ses_to_emails=["gm@team.com", "analytics@team.com"],
        aws_region="us-east-1",
        lookback_days=30,
        output_dir=tmp_path / "reports",
    )
    (tmp_path / "experiment").mkdir()
    (tmp_path / "reports").mkdir()
    return AutomatedReporter(config)


# ---------------------------------------------------------------------------
# _month_label_es
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("month,expected", [
    (1, "Enero"),
    (5, "Mayo"),
    (12, "Diciembre"),
    (7, "Julio"),
])
def test_month_label_es_returns_correct_spanish_name(month: int, expected: str):
    d = date(2026, month, 15)
    assert _month_label_es(d).startswith(expected)


def test_month_label_es_includes_year():
    d = date(2026, 5, 19)
    assert "2026" in _month_label_es(d)


# ---------------------------------------------------------------------------
# Chart builders — return type and structure
# ---------------------------------------------------------------------------


def test_build_cover_page_returns_figure():
    fig = _build_cover_page(FULL_SUMMARY, "Mayo 2026")
    assert isinstance(fig, go.Figure)


def test_build_cover_page_empty_summary_does_not_raise():
    fig = _build_cover_page(EMPTY_SUMMARY, "Mayo 2026")
    assert isinstance(fig, go.Figure)


def test_build_adoption_trend_chart_returns_figure():
    fig = _build_adoption_trend_chart(FULL_SUMMARY, lookback_days=30)
    assert isinstance(fig, go.Figure)


def test_build_adoption_trend_chart_has_bar_trace():
    fig = _build_adoption_trend_chart(FULL_SUMMARY, lookback_days=28)
    assert any(isinstance(t, go.Bar) for t in fig.data)


def test_build_divergence_type_chart_returns_figure():
    fig = _build_divergence_type_chart(FULL_SUMMARY)
    assert isinstance(fig, go.Figure)


def test_build_divergence_type_chart_has_three_slices():
    fig = _build_divergence_type_chart(FULL_SUMMARY)
    pie = next(t for t in fig.data if isinstance(t, go.Pie))
    assert len(pie.labels) == 3
    assert len(pie.values) == 3


def test_build_er_accuracy_chart_with_data_has_indicator_trace():
    fig = _build_er_accuracy_chart(FULL_SUMMARY)
    assert any(isinstance(t, go.Indicator) for t in fig.data)


def test_build_er_accuracy_chart_without_data_has_no_indicator():
    fig = _build_er_accuracy_chart(EMPTY_SUMMARY)
    assert not any(isinstance(t, go.Indicator) for t in fig.data)


# ---------------------------------------------------------------------------
# generate_and_send — PDF file creation
# ---------------------------------------------------------------------------


FAKE_PAGE = b"%PDF-1.4 fake page"
FAKE_MERGED = b"%PDF-1.4 merged document"


@pytest.fixture
def patched_pdf(reporter: AutomatedReporter):
    """Patch kaleido and pypdf so generate_and_send works without those libs."""
    reporter._store.get_summary = MagicMock(return_value=FULL_SUMMARY)
    with (
        patch("src.reporting.automated_reporter._fig_to_pdf_bytes", return_value=FAKE_PAGE),
        patch("src.reporting.automated_reporter._merge_pdf_pages", return_value=FAKE_MERGED),
        patch("src.reporting.automated_reporter._send_via_ses"),
    ):
        yield


def test_generate_and_send_writes_pdf_to_output_dir(reporter: AutomatedReporter, patched_pdf):
    output_path = reporter.generate_and_send(period_label="Mayo 2026", send_email=False)
    assert output_path.exists()
    assert output_path.suffix == ".pdf"
    assert output_path.parent == reporter._config.output_dir


def test_generate_and_send_pdf_contains_merged_bytes(reporter: AutomatedReporter, patched_pdf):
    output_path = reporter.generate_and_send(period_label="Mayo 2026", send_email=False)
    assert output_path.read_bytes() == FAKE_MERGED


def test_generate_and_send_no_email_skips_ses(reporter: AutomatedReporter, patched_pdf):
    with patch("src.reporting.automated_reporter._send_via_ses") as mock_ses:
        reporter.generate_and_send(period_label="Mayo 2026", send_email=False)
    mock_ses.assert_not_called()


def test_generate_and_send_with_email_calls_ses(reporter: AutomatedReporter):
    reporter._store.get_summary = MagicMock(return_value=FULL_SUMMARY)
    with (
        patch("src.reporting.automated_reporter._fig_to_pdf_bytes", return_value=FAKE_PAGE),
        patch("src.reporting.automated_reporter._merge_pdf_pages", return_value=FAKE_MERGED),
        patch("src.reporting.automated_reporter._send_via_ses") as mock_ses,
    ):
        reporter.generate_and_send(period_label="Mayo 2026", send_email=True)

    mock_ses.assert_called_once()
    call_kwargs = mock_ses.call_args
    assert call_kwargs[0][1] == FAKE_MERGED  # pdf_bytes arg
    assert "Mayo 2026" in call_kwargs[0][2]  # period_label arg


def test_generate_and_send_calls_get_summary_with_lookback_days(reporter: AutomatedReporter):
    reporter._store.get_summary = MagicMock(return_value=FULL_SUMMARY)
    with (
        patch("src.reporting.automated_reporter._fig_to_pdf_bytes", return_value=FAKE_PAGE),
        patch("src.reporting.automated_reporter._merge_pdf_pages", return_value=FAKE_MERGED),
        patch("src.reporting.automated_reporter._send_via_ses"),
    ):
        reporter.generate_and_send(period_label="Mayo 2026", send_email=False)

    reporter._store.get_summary.assert_called_once_with(lookback_days=30)


def test_generate_and_send_builds_four_figures(reporter: AutomatedReporter):
    """Four charts should be rendered — one per call to _fig_to_pdf_bytes."""
    reporter._store.get_summary = MagicMock(return_value=FULL_SUMMARY)
    with (
        patch("src.reporting.automated_reporter._fig_to_pdf_bytes", return_value=FAKE_PAGE) as mock_render,
        patch("src.reporting.automated_reporter._merge_pdf_pages", return_value=FAKE_MERGED),
        patch("src.reporting.automated_reporter._send_via_ses"),
    ):
        reporter.generate_and_send(period_label="Mayo 2026", send_email=False)

    assert mock_render.call_count == 4


# ---------------------------------------------------------------------------
# generate_monthly_report — Airflow callable
# ---------------------------------------------------------------------------


def test_generate_monthly_report_returns_path_string(tmp_path: Path):
    with (
        patch("src.reporting.automated_reporter.AutomatedReporter") as MockReporter,
    ):
        mock_instance = MagicMock()
        mock_instance.generate_and_send.return_value = tmp_path / "report.pdf"
        MockReporter.return_value = mock_instance

        result = generate_monthly_report(execution_date="2026-05-01", send_email=False)

    assert isinstance(result, str)
    assert "report.pdf" in result
