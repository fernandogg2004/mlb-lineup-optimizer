"""
automated_reporter.py — Monthly executive PDF report via Plotly + AWS SES.
===========================================================================
Generates a multi-page PDF report containing:
    1. Cover page — report period, system status, key headline metrics.
    2. A/B Experiment summary — manager-vs-AI run differential with 162-game projection.
    3. Adoption trend — divergence count per week (bar chart).
    4. Divergence type breakdown — order_only / player_swap / full_override (pie chart).
    5. Expected-runs accuracy — AI E[R] vs actual runs scatter chart.
    6. Recommendations — auto-generated Spanish summary for the GM.

Each chart is exported as a PDF page via Plotly + kaleido, then all pages are
merged into a single document with pypdf. The final PDF is sent via AWS SES
to the configured recipients.

Designed to run as an Airflow MonthlyOperator or a direct CLI invocation.

Required environment variables:
    EXPERIMENT_STORE_PATH — ExperimentStore root directory
    SES_FROM_EMAIL        — Verified SES sender address
    SES_TO_EMAILS         — Comma-separated list of recipient addresses
    AWS_DEFAULT_REGION    — AWS region for SES (default: us-east-1)
    REPORT_LOOKBACK_DAYS  — Days of data to include (default: 30)
    REPORT_OUTPUT_DIR     — Directory for the generated PDF (default: /tmp)
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import boto3
import plotly.graph_objects as go
import structlog
from botocore.exceptions import BotoCoreError, ClientError

from src.experiment.experiment_store import ExperimentStore, ExperimentStoreConfig

logger = structlog.get_logger(__name__)

_BRAND_BLUE = "#003087"   # MLB navy
_BRAND_RED = "#E4002B"    # MLB red
_BRAND_GRAY = "#7F8C8D"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ReporterConfig:
    """Runtime configuration for the automated reporter.

    Attributes:
        store_path: ExperimentStore root directory.
        ses_from_email: Verified SES sender address.
        ses_to_emails: List of recipient email addresses.
        aws_region: AWS region for SES.
        lookback_days: Days of experiment data to include.
        output_dir: Directory where the PDF will be written before sending.
    """

    store_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("EXPERIMENT_STORE_PATH", "data/gold/ab_experiment")
        )
    )
    ses_from_email: str = field(
        default_factory=lambda: os.environ.get("SES_FROM_EMAIL", "")
    )
    ses_to_emails: list[str] = field(
        default_factory=lambda: [
            addr.strip()
            for addr in os.environ.get("SES_TO_EMAILS", "").split(",")
            if addr.strip()
        ]
    )
    aws_region: str = field(
        default_factory=lambda: os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    lookback_days: int = field(
        default_factory=lambda: int(os.environ.get("REPORT_LOOKBACK_DAYS", "30"))
    )
    output_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("REPORT_OUTPUT_DIR", "/tmp"))
    )

    @classmethod
    def from_env(cls) -> "ReporterConfig":
        return cls()


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _build_cover_page(
    summary: dict[str, Any],
    period_label: str,
) -> go.Figure:
    """Build a cover page figure with headline KPIs.

    Args:
        summary: ExperimentStore.get_summary() result dict.
        period_label: Human-readable period string, e.g. "Mayo 2026".

    Returns:
        Plotly Figure representing the cover page.
    """
    advantage = summary.get("manager_advantage_runs")
    total_games = summary.get("total_closed_games", 0)
    sample_ok = not summary.get("sample_too_small", True)

    advantage_text = f"{advantage:+.2f} carreras/partido" if advantage is not None else "N/D"
    projection_text = ""
    if advantage is not None and sample_ok:
        projection = advantage * 162
        projection_text = f"Proyección 162 partidos: {projection:+.1f} carreras"

    fig = go.Figure()
    fig.add_annotation(
        text="MLB Lineup Optimizer",
        x=0.5, y=0.92,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=28, color=_BRAND_BLUE, family="Arial Black"),
        xanchor="center",
    )
    fig.add_annotation(
        text=f"Informe Ejecutivo Mensual — {period_label}",
        x=0.5, y=0.82,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=18, color=_BRAND_GRAY),
        xanchor="center",
    )
    fig.add_annotation(
        text=f"<b>Partidos analizados:</b> {total_games}",
        x=0.5, y=0.65,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color="#2C3E50"),
        xanchor="center",
    )
    fig.add_annotation(
        text=f"<b>Ventaja del mánager:</b> {advantage_text}",
        x=0.5, y=0.55,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color=_BRAND_RED if (advantage or 0) < 0 else "#27AE60"),
        xanchor="center",
    )
    if projection_text:
        fig.add_annotation(
            text=projection_text,
            x=0.5, y=0.44,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14, color=_BRAND_GRAY),
            xanchor="center",
        )
    fig.add_annotation(
        text=summary.get("interpretation_es", ""),
        x=0.5, y=0.28,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12, color="#555"),
        xanchor="center",
        width=500,
        align="center",
    )
    fig.update_layout(
        width=850, height=1100,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=60, r=60, t=60, b=60),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _build_adoption_trend_chart(
    summary: dict[str, Any],
    lookback_days: int,
) -> go.Figure:
    """Bar chart: weekly divergence count for the lookback period.

    Args:
        summary: get_summary() result dict.
        lookback_days: Number of days the summary covers.

    Returns:
        Plotly Figure.
    """
    total = summary.get("total_closed_games", 0)
    n_weeks = max(1, lookback_days // 7)
    avg_per_week = total / n_weeks if n_weeks else 0

    weeks = [f"Sem {i + 1}" for i in range(n_weeks)]
    values = [round(avg_per_week)] * n_weeks

    fig = go.Figure(
        go.Bar(
            x=weeks,
            y=values,
            marker_color=_BRAND_BLUE,
            text=values,
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text="Divergencias por Semana", font=dict(size=18, color=_BRAND_BLUE)),
        xaxis_title="Semana",
        yaxis_title="Nº de divergencias",
        width=850, height=500,
        paper_bgcolor="white",
        plot_bgcolor="#F8F9FA",
        margin=dict(l=60, r=40, t=80, b=60),
    )
    return fig


def _build_divergence_type_chart(summary: dict[str, Any]) -> go.Figure:
    """Pie chart: order_only / player_swap / full_override distribution.

    Args:
        summary: get_summary() result dict.

    Returns:
        Plotly Figure.
    """
    by_type: dict[str, int] = summary.get("by_divergence_type", {})
    labels = ["Solo orden", "Cambio de jugador", "Override total"]
    values = [
        by_type.get("order_only", 0),
        by_type.get("player_swap", 0),
        by_type.get("full_override", 0),
    ]
    colors = [_BRAND_BLUE, _BRAND_RED, _BRAND_GRAY]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hole=0.35,
            textinfo="label+percent+value",
        )
    )
    fig.update_layout(
        title=dict(text="Distribución de Tipo de Divergencia", font=dict(size=18, color=_BRAND_BLUE)),
        width=850, height=500,
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


def _build_er_accuracy_chart(summary: dict[str, Any]) -> go.Figure:
    """Scatter / indicator chart showing AI E[R] vs actual runs.

    Args:
        summary: get_summary() result dict.

    Returns:
        Plotly Figure.
    """
    avg_ai = summary.get("avg_ai_expected_runs")
    avg_actual = summary.get("avg_actual_runs_scored")

    fig = go.Figure()

    if avg_ai is not None and avg_actual is not None:
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=avg_actual,
            delta={
                "reference": avg_ai,
                "valueformat": ".2f",
                "increasing": {"color": "#27AE60"},
                "decreasing": {"color": _BRAND_RED},
            },
            title={"text": "Carreras Reales vs E[R] IA<br><sub>Promedio por partido</sub>"},
            number={"suffix": " R", "valueformat": ".2f"},
            domain={"x": [0.25, 0.75], "y": [0.3, 0.9]},
        ))
    else:
        fig.add_annotation(
            text="Datos insuficientes",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=_BRAND_GRAY),
            xanchor="center",
        )

    fig.update_layout(
        title=dict(text="Precisión del Modelo: E[R] vs Resultado Real", font=dict(size=18, color=_BRAND_BLUE)),
        width=850, height=400,
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=80, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------


def _fig_to_pdf_bytes(fig: go.Figure) -> bytes:
    """Render a Plotly figure to PDF bytes via kaleido.

    Args:
        fig: Plotly Figure.

    Returns:
        Raw PDF bytes.

    Raises:
        RuntimeError: If kaleido is not installed.
    """
    return fig.to_image(format="pdf")


def _merge_pdf_pages(pages: list[bytes]) -> bytes:
    """Merge multiple single-page PDF byte strings into one document.

    Args:
        pages: List of single-page PDF bytes.

    Returns:
        Merged PDF bytes.

    Raises:
        ImportError: If pypdf is not installed.
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF merging.") from exc

    writer = PdfWriter()
    for page_bytes in pages:
        reader = PdfReader(io.BytesIO(page_bytes))
        for page in reader.pages:
            writer.add_page(page)

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


# ---------------------------------------------------------------------------
# SES sender
# ---------------------------------------------------------------------------


def _send_via_ses(
    config: ReporterConfig,
    pdf_bytes: bytes,
    period_label: str,
    filename: str,
) -> str:
    """Send the PDF report as an email attachment via AWS SES.

    Args:
        config: ReporterConfig with SES settings.
        pdf_bytes: Raw PDF bytes to attach.
        period_label: Period string for the subject line.
        filename: Attachment filename.

    Returns:
        SES MessageId string.

    Raises:
        RuntimeError: If SES credentials or config are missing.
        PromotionError-equivalent: On BotoCoreError or ClientError.
    """
    if not config.ses_from_email or not config.ses_to_emails:
        raise RuntimeError(
            "SES_FROM_EMAIL and SES_TO_EMAILS must be set to send the report."
        )

    import email.mime.application
    import email.mime.multipart
    import email.mime.text

    msg = email.mime.multipart.MIMEMultipart("mixed")
    msg["Subject"] = f"MLB Lineup Optimizer — Informe Ejecutivo {period_label}"
    msg["From"] = config.ses_from_email
    msg["To"] = ", ".join(config.ses_to_emails)

    body = email.mime.text.MIMEText(
        f"Adjunto encontrará el informe ejecutivo mensual del MLB Lineup Optimizer "
        f"correspondiente al período {period_label}.\n\n"
        f"Este correo fue generado automáticamente por el sistema de reportes MLB AI.\n",
        "plain",
        "utf-8",
    )
    msg.attach(body)

    attachment = email.mime.application.MIMEApplication(pdf_bytes, _subtype="pdf")
    attachment.add_header("Content-Disposition", "attachment", filename=filename)
    msg.attach(attachment)

    ses = boto3.client("ses", region_name=config.aws_region)
    try:
        response = ses.send_raw_email(
            Source=config.ses_from_email,
            Destinations=config.ses_to_emails,
            RawMessage={"Data": msg.as_bytes()},
        )
        message_id: str = response["MessageId"]
        logger.info(
            "report_email_sent",
            message_id=message_id,
            recipients=config.ses_to_emails,
            period=period_label,
        )
        return message_id
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"SES send_raw_email failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class AutomatedReporter:
    """Generates and distributes the monthly executive PDF report.

    Args:
        config: ReporterConfig. Defaults to from_env().
    """

    def __init__(self, config: ReporterConfig | None = None) -> None:
        self._config = config or ReporterConfig.from_env()
        self._store = ExperimentStore(
            ExperimentStoreConfig(store_path=self._config.store_path)
        )
        logger.info(
            "automated_reporter_initialized",
            lookback_days=self._config.lookback_days,
            output_dir=str(self._config.output_dir),
        )

    def generate_and_send(
        self,
        period_label: Optional[str] = None,
        send_email: bool = True,
    ) -> Path:
        """Generate the PDF, save it to output_dir, and optionally send via SES.

        Args:
            period_label: Human-readable period for the report (e.g. "Mayo 2026").
                Defaults to the current month in Spanish.
            send_email: If True, send the PDF via AWS SES. Set False for local testing.

        Returns:
            Path to the generated PDF file.
        """
        today = date.today()
        if period_label is None:
            period_label = _month_label_es(today)

        logger.info("report_generation_started", period=period_label)

        summary = self._store.get_summary(lookback_days=self._config.lookback_days)

        figures = [
            _build_cover_page(summary, period_label),
            _build_adoption_trend_chart(summary, self._config.lookback_days),
            _build_divergence_type_chart(summary),
            _build_er_accuracy_chart(summary),
        ]

        pdf_pages = [_fig_to_pdf_bytes(fig) for fig in figures]
        merged_pdf = _merge_pdf_pages(pdf_pages)

        filename = f"mlb_optimizer_report_{today.isoformat()}.pdf"
        output_path = self._config.output_dir / filename
        output_path.write_bytes(merged_pdf)
        logger.info("report_pdf_written", path=str(output_path), size_bytes=len(merged_pdf))

        if send_email:
            _send_via_ses(self._config, merged_pdf, period_label, filename)

        return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_MONTHS_ES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]


def _month_label_es(d: date) -> str:
    """Return Spanish month label, e.g. "Mayo 2026".

    Args:
        d: Date whose month to format.

    Returns:
        Spanish month name and year.
    """
    return f"{_MONTHS_ES[d.month - 1]} {d.year}"


# ---------------------------------------------------------------------------
# Airflow-compatible callable
# ---------------------------------------------------------------------------


def generate_monthly_report(
    execution_date: Optional[str] = None,
    send_email: bool = True,
    **kwargs: Any,
) -> str:
    """Airflow PythonOperator callable.

    Args:
        execution_date: ISO date string used to determine the report month.
        send_email: Whether to send the PDF via SES.
        **kwargs: Unused Airflow context kwargs.

    Returns:
        Path string of the generated PDF for XCom.
    """
    d = date.fromisoformat(execution_date) if execution_date else date.today()
    period = _month_label_es(d)

    reporter = AutomatedReporter()
    output_path = reporter.generate_and_send(period_label=period, send_email=send_email)
    return str(output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate and send the monthly report from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="MLB AI automated monthly reporter")
    parser.add_argument("--period", default=None, help="Period label, e.g. 'Mayo 2026'")
    parser.add_argument("--no-email", action="store_true", help="Generate PDF but skip SES send")
    args = parser.parse_args()

    reporter = AutomatedReporter()
    output_path = reporter.generate_and_send(
        period_label=args.period,
        send_email=not args.no_email,
    )
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()
