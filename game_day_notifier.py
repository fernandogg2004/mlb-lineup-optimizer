"""
game_day_notifier.py
====================
Automated pre-game notification pipeline.

Designed to run 4 hours before first pitch via Argo CronWorkflow, Airflow DAG,
or a simple cron job. On each execution:

  1. Fetch the definitive lineup and briefing data from the FastAPI backend
  2. Generate a 2-page executive PDF report using ReportLab
  3. Upload the PDF and send an enriched Block Kit message to the Slack channel
  4. Send a high-priority HTML email with the PDF attached via AWS SES

Environment variables required:
    API_BASE_URL             FastAPI backend (default: http://localhost:8000)
    SLACK_BOT_TOKEN          OAuth bot token (xoxb-...)
    SLACK_CHANNEL_ID         Target channel (e.g. "#war-room" or channel ID)
    SES_SENDER_EMAIL         Verified SES sender address
    SES_RECIPIENT_EMAILS     Comma-separated recipient list
    AWS_DEFAULT_REGION       AWS region for SES (default: us-east-1)
    DEMO_MODE                Set to "1" to run with synthetic data (no APIs)

Run:
    python game_day_notifier.py --game-date 2026-05-18
    python game_day_notifier.py  # defaults to today
"""
from __future__ import annotations

import argparse
import io
import json
import os
import smtplib
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional

import boto3
import httpx
import structlog
from jinja2 import Environment, BaseLoader
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = structlog.get_logger(__name__)

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEMO_MODE: bool = os.environ.get("DEMO_MODE", "0") == "1"

# ---------------------------------------------------------------------------
# PDF colour palette
# ---------------------------------------------------------------------------

_NAVY = colors.HexColor("#0d1528")
_BLUE = colors.HexColor("#1565c0")
_LIGHT_BLUE = colors.HexColor("#90caf9")
_GREEN = colors.HexColor("#2e7d32")
_LIGHT_GREEN = colors.HexColor("#a5d6a7")
_AMBER = colors.HexColor("#f59f00")
_RED = colors.HexColor("#c62828")
_LIGHT_GREY = colors.HexColor("#eceff1")
_MID_GREY = colors.HexColor("#b0bec5")
_WHITE = colors.white

# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@dataclass
class PlayerBriefEntry:
    batting_pos: int
    player_name: str
    batter_stand: str
    woba: float
    obp: float
    iso: float
    platoon_note: str = ""


@dataclass
class BriefingData:
    game_date: str
    rival_team: str
    pitcher_name: str
    pitcher_hand: str
    venue_name: str
    temperature_f: float
    wind_mph: float
    humidity_pct: float
    park_factor_hr: float
    park_factor_xb: float
    expected_runs: float
    win_probability: float
    lineup: list[PlayerBriefEntry]
    alt_lineups: list[dict[str, Any]] = field(default_factory=list)  # [{rank, er, pw}]
    rag_briefing_excerpt: str = ""
    scout_alerts: list[dict[str, str]] = field(default_factory=list)  # [{player, level, note}]
    optimization_mode: str = "full"
    total_evaluations: int = 0
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def _demo_briefing_data(game_date: str) -> BriefingData:
    return BriefingData(
        game_date=game_date,
        rival_team="NYY",
        pitcher_name="Gerrit Cole",
        pitcher_hand="R",
        venue_name="Yankee Stadium",
        temperature_f=72.0,
        wind_mph=8.0,
        humidity_pct=55.0,
        park_factor_hr=1.08,
        park_factor_xb=1.03,
        expected_runs=5.23,
        win_probability=0.614,
        lineup=[
            PlayerBriefEntry(1, "Aaron Judge", "R", 0.421, 0.406, 0.342, "Platoon neutral vs RHP"),
            PlayerBriefEntry(2, "Gleyber Torres", "R", 0.349, 0.341, 0.162, "K-rate ↑ vs breaking balls"),
            PlayerBriefEntry(3, "Giancarlo Stanton", "R", 0.455, 0.378, 0.310, "RISP wOBA .501"),
            PlayerBriefEntry(4, "Anthony Rizzo", "L", 0.329, 0.370, 0.185, "+0.038 platoon vs RHP"),
            PlayerBriefEntry(5, "Anthony Volpe", "R", 0.318, 0.320, 0.148, "EV ↑ último mes"),
            PlayerBriefEntry(6, "DJ LeMahieu", "R", 0.305, 0.349, 0.109, "Contact hitter"),
            PlayerBriefEntry(7, "Oswaldo Peraza", "R", 0.295, 0.305, 0.133, ""),
            PlayerBriefEntry(8, "Jose Bauers", "L", 0.290, 0.360, 0.120, "+0.021 vs RHP"),
            PlayerBriefEntry(9, "Austin Wells", "L", 0.302, 0.332, 0.155, ""),
        ],
        alt_lineups=[
            {"rank": 2, "er": 5.18, "pw": 0.601},
            {"rank": 3, "er": 5.11, "pw": 0.589},
        ],
        rag_briefing_excerpt=(
            "## Resumen Ejecutivo\n"
            "El modelo recomienda alinear a Judge en 1er turno por su OBP élite (.406) "
            "maximizando los PAs en posiciones de impacto. Stanton de 3er bate capitaliza "
            "los RISP generados por el uno-dos. Su wOBA proyectado contra el slider de Cole "
            "es +0.125 superior al promedio del equipo. "
            "Rizzo de 4to aprovecha la ventaja de platón zurdo-derecho contra Cole.\n\n"
            "## Alertas de Scout\n"
            "- **Torres** [BAJO]: Molestia leve en muñeca derecha reportada el 05-16. "
            "El médico del equipo lo aprueba para jugar. Monitorear en inning 6+.\n"
            "- **Stanton** [BAJO]: Mecánica de swing ajustada esta semana. EV promedio "
            "+2.1 mph en batting practice."
        ),
        scout_alerts=[
            {"player": "Gleyber Torres", "level": "BAJO", "note": "Molestia leve muñeca — aprobado para jugar"},
            {"player": "Giancarlo Stanton", "level": "BAJO", "note": "Ajuste de mecánica — EV +2.1 mph BP"},
        ],
        optimization_mode="full",
        total_evaluations=30_000,
    )


def fetch_briefing_data(game_date: str) -> BriefingData:
    """Fetch lineup and briefing data from the FastAPI backend."""
    if DEMO_MODE:
        logger.info("demo_mode_active_using_synthetic_data")
        return _demo_briefing_data(game_date)

    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.get(f"{API_BASE_URL}/v1/briefing/today", params={"game_date": game_date})
            r.raise_for_status()
            raw = r.json()
        logger.info("briefing_data_fetched", game_date=game_date)
        return BriefingData(**raw)
    except Exception as exc:
        logger.warning("api_fetch_failed_using_demo", error=str(exc))
        return _demo_briefing_data(game_date)


# ---------------------------------------------------------------------------
# PDF generation (ReportLab Platypus — 2 pages, letter)
# ---------------------------------------------------------------------------


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontSize=20,
            textColor=_WHITE,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=_LIGHT_BLUE,
            spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "section",
            parent=base["Heading2"],
            fontSize=12,
            textColor=_LIGHT_BLUE,
            spaceBefore=10,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#37474f"),
            spaceAfter=4,
            leading=13,
        ),
        "alert_low": ParagraphStyle(
            "alert_low", parent=base["Normal"], fontSize=9,
            textColor=colors.HexColor("#1b5e20"), spaceAfter=3,
        ),
        "alert_mid": ParagraphStyle(
            "alert_mid", parent=base["Normal"], fontSize=9,
            textColor=colors.HexColor("#e65100"), spaceAfter=3,
        ),
        "alert_high": ParagraphStyle(
            "alert_high", parent=base["Normal"], fontSize=9,
            textColor=colors.HexColor("#b71c1c"), spaceAfter=3,
        ),
    }


def _header_block(data: BriefingData, styles: dict) -> list:
    elements = [
        Table(
            [[
                Paragraph(f"⚾ MLB LINEUP OPTIMIZER", styles["title"]),
                Paragraph(
                    f"<b>TACTICAL BRIEFING</b><br/>{data.game_date}",
                    ParagraphStyle("hdr_r", fontSize=11, textColor=_LIGHT_BLUE,
                                   alignment=2, fontName="Helvetica-Bold"),
                ),
            ]],
            colWidths=[4 * inch, 3.5 * inch],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), _NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (0, -1), 14),
                ("RIGHTPADDING", (-1, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ]),
        ),
        Spacer(1, 6),
        Table(
            [[
                Paragraph(f"vs. <b>{data.rival_team}</b> · {data.pitcher_name} ({data.pitcher_hand}HP)", styles["subtitle"]),
                Paragraph(
                    f"<b>E[R] = {data.expected_runs:.2f}</b>  |  <b>P(W) = {data.win_probability:.1%}</b>",
                    ParagraphStyle("kpi", fontSize=13, textColor=_LIGHT_GREEN,
                                   alignment=2, fontName="Helvetica-Bold"),
                ),
            ]],
            colWidths=[4 * inch, 3.5 * inch],
            style=TableStyle([("BOTTOMPADDING", (0, 0), (-1, -1), 6)]),
        ),
        HRFlowable(width="100%", thickness=1.5, color=_BLUE, spaceAfter=8),
    ]
    return elements


def _lineup_table(data: BriefingData, styles: dict) -> list:
    elements = [Paragraph("LINEUP ÓPTIMO", styles["section"])]

    header = ["#", "Jugador", "Mano", "xwOBA", "OBP", "ISO", "Nota de Platoon"]
    rows = [header]
    for p in data.lineup:
        rows.append([
            str(p.batting_pos),
            p.player_name,
            p.batter_stand,
            f"{p.woba:.3f}",
            f"{p.obp:.3f}",
            f"{p.iso:.3f}",
            p.platoon_note or "—",
        ])

    col_widths = [0.3 * inch, 1.5 * inch, 0.45 * inch, 0.65 * inch,
                  0.65 * inch, 0.55 * inch, 2.5 * inch]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        # Header
        ("BACKGROUND",   (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, 0), _LIGHT_BLUE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 8),
        ("ALIGN",        (0, 0), (-1, 0), "CENTER"),
        # Data rows
        ("FONTSIZE",     (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT_GREY, _WHITE]),
        ("TEXTCOLOR",    (0, 1), (-1, -1), colors.HexColor("#263238")),
        ("ALIGN",        (0, 0), (0, -1), "CENTER"),
        ("ALIGN",        (3, 1), (5, -1), "CENTER"),
        # Borders
        ("GRID",         (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)
    return elements


def _summary_kpi_block(data: BriefingData, styles: dict) -> list:
    elements = [Spacer(1, 10)]
    kpi_data = [
        ["E[R] Proyectado", f"{data.expected_runs:.3f}",
         "P(Victoria)", f"{data.win_probability:.1%}",
         "Modo", data.optimization_mode.upper()],
    ]
    t = Table(kpi_data, colWidths=[1.2*inch]*6)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), _NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, -1), _WHITE),
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica"),
        ("FONTNAME",     (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME",     (2, 0), (2, -1), "Helvetica"),
        ("FONTNAME",     (3, 0), (3, -1), "Helvetica-Bold"),
        ("FONTNAME",     (4, 0), (4, -1), "Helvetica"),
        ("FONTNAME",     (5, 0), (5, -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",    (1, 0), (1, -1), _LIGHT_GREEN),
        ("TEXTCOLOR",    (3, 0), (3, -1), _LIGHT_GREEN),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("BOX",          (0, 0), (-1, -1), 0.5, _BLUE),
        ("INNERGRID",    (0, 0), (-1, -1), 0.3, _BLUE),
    ]))
    elements.append(t)
    return elements


def _conditions_block(data: BriefingData, styles: dict) -> list:
    elements = [Paragraph("CONDICIONES DEL PARQUE Y CLIMA", styles["section"])]
    rows = [
        ["Estadio:", data.venue_name, "Temperatura:", f"{data.temperature_f:.0f}°F"],
        ["Viento:", f"{data.wind_mph:.0f} mph", "Humedad:", f"{data.humidity_pct:.0f}%"],
        ["Park Factor HR:", f"{data.park_factor_hr:.3f}", "Park Factor XB:", f"{data.park_factor_xb:.3f}"],
    ]
    t = Table(rows, colWidths=[1.3*inch, 2.0*inch, 1.3*inch, 2.0*inch])
    t.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#263238")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_LIGHT_GREY, _WHITE, _LIGHT_GREY]),
        ("GRID",      (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
        ("TOPPADDING",(0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    return elements


def _alt_lineups_block(data: BriefingData, styles: dict) -> list:
    if not data.alt_lineups:
        return []
    elements = [Spacer(1, 8), Paragraph("ESCENARIOS ALTERNATIVOS", styles["section"])]
    rows = [["Rank", "E[R]", "P(Victoria)", "Δ vs. Óptimo"]]
    for alt in data.alt_lineups:
        d_er = alt["er"] - data.expected_runs
        rows.append([
            f"#{alt['rank']}",
            f"{alt['er']:.3f}",
            f"{alt['pw']:.1%}",
            f"{d_er:+.3f}",
        ])
    t = Table(rows, colWidths=[0.6*inch, 1.0*inch, 1.1*inch, 1.2*inch], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, 0), _LIGHT_BLUE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT_GREY, _WHITE]),
        ("TEXTCOLOR",    (3, 1), (3, -1), _RED),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("GRID",         (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    return elements


def _er_bar_chart(data: BriefingData) -> Drawing:
    """Horizontal bar chart comparing E[R] across lineup options."""
    all_lineups = [{"rank": 1, "er": data.expected_runs}] + data.alt_lineups
    labels = [f"Lineup #{d['rank']}" for d in all_lineups]
    values = [d["er"] for d in all_lineups]

    drawing = Drawing(280, max(80, 30 * len(all_lineups)))
    bc = HorizontalBarChart()
    bc.x = 70; bc.y = 10
    bc.height = drawing.height - 20
    bc.width = 200
    bc.data = [values]
    bc.categoryAxis.categoryNames = labels
    bc.categoryAxis.labels.fontSize = 8
    bc.categoryAxis.labels.fillColor = colors.HexColor("#37474f")
    bc.valueAxis.valueMin = max(0, min(values) - 0.5)
    bc.valueAxis.valueMax = max(values) + 0.3
    bc.valueAxis.labels.fontSize = 7
    bc.bars[0].fillColor = _BLUE
    bc.bars[0].strokeColor = None
    drawing.add(bc)
    return drawing


def _rag_excerpt_block(data: BriefingData, styles: dict) -> list:
    if not data.rag_briefing_excerpt:
        return []
    elements = [Paragraph("BRIEFING TÁCTICO — RESUMEN RAG", styles["section"])]
    # Render markdown-lite excerpt as plain paragraphs
    for line in data.rag_briefing_excerpt.splitlines():
        line = line.strip()
        if not line:
            elements.append(Spacer(1, 4))
        elif line.startswith("## "):
            elements.append(Paragraph(line[3:], styles["section"]))
        elif line.startswith("- "):
            elements.append(Paragraph(f"• {line[2:]}", styles["body"]))
        else:
            elements.append(Paragraph(line, styles["body"]))
    return elements


def _scout_alerts_block(data: BriefingData, styles: dict) -> list:
    if not data.scout_alerts:
        return []
    elements = [Spacer(1, 8), Paragraph("ALERTAS DE SCOUT", styles["section"])]
    level_map = {"BAJO": styles["alert_low"], "MEDIO": styles["alert_mid"], "ALTO": styles["alert_high"]}
    level_icon = {"BAJO": "🟡", "MEDIO": "🟠", "ALTO": "🔴"}
    for alert in data.scout_alerts:
        lvl = alert.get("level", "BAJO")
        icon = level_icon.get(lvl, "🟡")
        style = level_map.get(lvl, styles["alert_low"])
        elements.append(
            Paragraph(
                f"{icon} <b>{alert['player']}</b> [{lvl}]: {alert['note']}",
                style,
            )
        )
    return elements


def generate_pdf(data: BriefingData) -> bytes:
    """Compile a 2-page executive PDF. Returns raw bytes for upload."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=f"MLB Tactical Briefing — {data.game_date}",
        author="MLB Lineup Optimizer AI",
        subject=f"vs. {data.rival_team}",
    )
    styles = _build_styles()
    story: list = []

    # === PAGE 1 ===
    story += _header_block(data, styles)
    story += _lineup_table(data, styles)
    story += _summary_kpi_block(data, styles)

    story.append(PageBreak())

    # === PAGE 2 ===
    story += _header_block(data, styles)
    story += _conditions_block(data, styles)
    story += _alt_lineups_block(data, styles)
    story.append(Spacer(1, 10))

    # E[R] bar chart
    if data.alt_lineups:
        story.append(Paragraph("E[R] — COMPARATIVA DE LINEUPS", styles["section"]))
        story.append(_er_bar_chart(data))

    story += _rag_excerpt_block(data, styles)
    story += _scout_alerts_block(data, styles)

    # Footer note
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_MID_GREY))
    story.append(
        Paragraph(
            f"Generado: {data.generated_at} · MLB Lineup Optimizer AI · "
            f"Modo: {data.optimization_mode.upper()} · {data.total_evaluations:,} evaluaciones",
            ParagraphStyle("footer", fontSize=7, textColor=_MID_GREY, spaceAfter=0),
        )
    )

    doc.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Slack notification
# ---------------------------------------------------------------------------


def _build_slack_blocks(data: BriefingData) -> list[dict]:
    lineup_lines = "\n".join(
        f"{p.batting_pos}. *{p.player_name}* ({p.batter_stand}HB) — xwOBA `{p.woba:.3f}`"
        for p in data.lineup
    )
    alert_lines = "\n".join(
        f"{'🟡' if a['level']=='BAJO' else '🟠' if a['level']=='MEDIO' else '🔴'} "
        f"*{a['player']}* [{a['level']}]: {a['note']}"
        for a in data.scout_alerts
    ) or "_Sin alertas activas_"

    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"⚾ MLB Tactical Briefing — {data.game_date}"},
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"vs. *{data.rival_team}* · "
                        f"{data.pitcher_name} ({data.pitcher_hand}HP) · "
                        f"{data.venue_name}"
                    ),
                }
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*E[Carreras Proyectadas]:*\n`{data.expected_runs:.3f}`"},
                {"type": "mrkdwn", "text": f"*P(Victoria):*\n`{data.win_probability:.1%}`"},
                {"type": "mrkdwn", "text": f"*Temp:* {data.temperature_f:.0f}°F · Viento: {data.wind_mph:.0f} mph"},
                {"type": "mrkdwn", "text": f"*PF-HR:* {data.park_factor_hr:.3f} · *PF-XB:* {data.park_factor_xb:.3f}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*LINEUP ÓPTIMO*\n{lineup_lines}"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*ALERTAS DE SCOUT*\n{alert_lines}"},
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"📎 PDF adjunto en este hilo · "
                        f"Modo: {data.optimization_mode.upper()} · "
                        f"{data.total_evaluations:,} evaluaciones · "
                        f"Generado: {data.generated_at}"
                    ),
                }
            ],
        },
    ]


def send_slack_notification(
    data: BriefingData,
    pdf_bytes: bytes,
    bot_token: str,
    channel_id: str,
) -> str:
    """Upload PDF and post an enriched Block Kit message. Returns message timestamp."""
    client = WebClient(token=bot_token)

    try:
        # Step 1: Upload PDF as file (v2 API)
        upload_resp = client.files_upload_v2(
            channel=channel_id,
            content=pdf_bytes,
            filename=f"tactical_briefing_{data.game_date}.pdf",
            title=f"Tactical Briefing — {data.game_date} vs. {data.rival_team}",
            initial_comment=(
                f":baseball: *MLB Tactical Briefing — {data.game_date}* "
                f"vs. {data.rival_team} adjunto."
            ),
        )
        file_ts = upload_resp.get("file", {}).get("shares", {})
        logger.info("slack_file_uploaded", channel=channel_id, date=data.game_date)

        # Step 2: Post rich Block Kit message
        msg_resp = client.chat_postMessage(
            channel=channel_id,
            blocks=_build_slack_blocks(data),
            text=f"MLB Tactical Briefing — {data.game_date} vs. {data.rival_team}",
            unfurl_links=False,
        )
        ts = msg_resp["ts"]
        logger.info("slack_message_posted", channel=channel_id, ts=ts)
        return ts

    except SlackApiError as exc:
        logger.error("slack_api_error", error=exc.response["error"])
        raise


# ---------------------------------------------------------------------------
# AWS SES email
# ---------------------------------------------------------------------------

_EMAIL_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<style>
  body  { font-family: 'Helvetica Neue', Arial, sans-serif; background:#f0f4f8; margin:0; padding:0; }
  .wrap { max-width:620px; margin:28px auto; background:#ffffff; border-radius:10px;
          box-shadow:0 2px 12px rgba(0,0,0,.10); overflow:hidden; }
  .hdr  { background:#0d1528; padding:22px 28px; }
  .hdr h1 { color:#90caf9; font-size:20px; margin:0 0 4px; }
  .hdr p  { color:#546e7a; font-size:13px; margin:0; }
  .kpi  { display:flex; gap:0; border-bottom:1px solid #e3e8ef; }
  .kpi-cell { flex:1; padding:18px 0; text-align:center; border-right:1px solid #e3e8ef; }
  .kpi-cell:last-child { border-right:none; }
  .kpi-label { font-size:11px; color:#78909c; text-transform:uppercase; letter-spacing:.06em; }
  .kpi-value { font-size:24px; font-weight:700; color:#1565c0; margin-top:4px; }
  .section  { padding:20px 28px 4px; }
  .section h2 { font-size:13px; color:#1565c0; text-transform:uppercase;
                letter-spacing:.06em; border-bottom:2px solid #e3f2fd; padding-bottom:6px; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th  { background:#0d1528; color:#90caf9; padding:8px 10px; text-align:left; font-weight:600; }
  tr:nth-child(even) td { background:#f5f7fa; }
  td  { padding:7px 10px; border-bottom:1px solid #eceff1; color:#37474f; }
  .alert-low  { color:#2e7d32; } .alert-mid { color:#e65100; } .alert-high { color:#c62828; }
  .footer { background:#f5f7fa; padding:14px 28px; font-size:11px; color:#90a4ae;
            text-align:center; border-top:1px solid #e3e8ef; }
</style>
</head>
<body>
<div class="wrap">
  <div class="hdr">
    <h1>⚾ MLB Tactical Briefing</h1>
    <p>vs. <strong>{{ data.rival_team }}</strong> · {{ data.pitcher_name }} ({{ data.pitcher_hand }}HP)
       · {{ data.venue_name }} · {{ data.game_date }}</p>
  </div>

  <div class="kpi">
    <div class="kpi-cell">
      <div class="kpi-label">E[Carreras]</div>
      <div class="kpi-value">{{ "%.2f"|format(data.expected_runs) }}</div>
    </div>
    <div class="kpi-cell">
      <div class="kpi-label">P(Victoria)</div>
      <div class="kpi-value">{{ "%.1f%%"|format(data.win_probability * 100) }}</div>
    </div>
    <div class="kpi-cell">
      <div class="kpi-label">PF HR</div>
      <div class="kpi-value">{{ "%.3f"|format(data.park_factor_hr) }}</div>
    </div>
    <div class="kpi-cell">
      <div class="kpi-label">Temp</div>
      <div class="kpi-value">{{ data.temperature_f|int }}°F</div>
    </div>
  </div>

  <div class="section">
    <h2>Lineup Óptimo</h2>
    <table>
      <tr><th>#</th><th>Jugador</th><th>Mano</th><th>xwOBA</th><th>OBP</th><th>ISO</th></tr>
      {% for p in data.lineup %}
      <tr>
        <td>{{ p.batting_pos }}</td>
        <td><strong>{{ p.player_name }}</strong></td>
        <td>{{ p.batter_stand }}HB</td>
        <td>{{ "%.3f"|format(p.woba) }}</td>
        <td>{{ "%.3f"|format(p.obp) }}</td>
        <td>{{ "%.3f"|format(p.iso) }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>

  {% if data.scout_alerts %}
  <div class="section">
    <h2>Alertas de Scout</h2>
    {% for alert in data.scout_alerts %}
    <p class="alert-{{ alert.level|lower }}">
      {% if alert.level == 'BAJO' %}🟡{% elif alert.level == 'MEDIO' %}🟠{% else %}🔴{% endif %}
      <strong>{{ alert.player }}</strong> [{{ alert.level }}]: {{ alert.note }}
    </p>
    {% endfor %}
  </div>
  {% endif %}

  <div class="footer">
    Generado: {{ data.generated_at }} · MLB Lineup Optimizer AI ·
    Modo: {{ data.optimization_mode|upper }} · {{ "{:,}".format(data.total_evaluations) }} evaluaciones<br>
    <em>PDF ejecutivo completo adjunto a este correo.</em>
  </div>
</div>
</body>
</html>
"""


def _render_email_html(data: BriefingData) -> str:
    env = Environment(loader=BaseLoader())
    tmpl = env.from_string(_EMAIL_HTML_TEMPLATE)
    return tmpl.render(data=data)


def send_ses_email(
    data: BriefingData,
    pdf_bytes: bytes,
    sender_email: str,
    recipient_emails: list[str],
    region: str = "us-east-1",
) -> str:
    """Send a high-priority HTML email with PDF attachment via AWS SES.

    Returns the SES MessageId.
    """
    html_body = _render_email_html(data)
    subject = (
        f"[ALTA PRIORIDAD] ⚾ MLB Tactical Briefing — {data.game_date} "
        f"vs. {data.rival_team} | E[R]={data.expected_runs:.2f} P(W)={data.win_probability:.1%}"
    )

    # Build MIME message
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipient_emails)
    msg["X-Priority"] = "1"
    msg["X-MSMail-Priority"] = "High"

    alt_part = MIMEMultipart("alternative")
    alt_part.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(alt_part)

    pdf_part = MIMEApplication(pdf_bytes, Name=f"tactical_briefing_{data.game_date}.pdf")
    pdf_part["Content-Disposition"] = (
        f'attachment; filename="tactical_briefing_{data.game_date}.pdf"'
    )
    msg.attach(pdf_part)

    try:
        ses = boto3.client("ses", region_name=region)
        resp = ses.send_raw_email(
            Source=sender_email,
            Destinations=recipient_emails,
            RawMessage={"Data": msg.as_bytes()},
        )
        message_id: str = resp["MessageId"]
        logger.info(
            "ses_email_sent",
            message_id=message_id,
            recipients=recipient_emails,
            game_date=data.game_date,
        )
        return message_id
    except Exception as exc:
        logger.error("ses_email_failed", error=str(exc))
        raise


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_notifier(game_date: str | None = None) -> None:
    """Execute the full notification pipeline for game_date."""
    if game_date is None:
        game_date = date.today().isoformat()

    logger.info("notifier_started", game_date=game_date, demo=DEMO_MODE)

    # Step 1: Fetch data
    data = fetch_briefing_data(game_date)
    logger.info(
        "briefing_data_ready",
        rival=data.rival_team,
        pitcher=data.pitcher_name,
        er=data.expected_runs,
        pw=data.win_probability,
    )

    # Step 2: Generate PDF
    pdf_bytes = generate_pdf(data)
    pdf_size_kb = len(pdf_bytes) / 1024
    logger.info("pdf_generated", size_kb=round(pdf_size_kb, 1))

    # Persist PDF locally for audit
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    pdf_path = out_dir / f"briefing_{game_date}_{data.rival_team}.pdf"
    pdf_path.write_bytes(pdf_bytes)
    logger.info("pdf_saved", path=str(pdf_path))

    if DEMO_MODE:
        logger.info("demo_mode_skipping_external_notifications")
        print(f"\n✅ PDF generado: {pdf_path} ({pdf_size_kb:.1f} KB)")
        print(f"   E[R]={data.expected_runs:.3f} | P(W)={data.win_probability:.1%}")
        print("   [DEMO MODE] Slack + SES notifications skipped.")
        return

    # Step 3: Slack
    slack_token = os.environ.get("SLACK_BOT_TOKEN", "")
    slack_channel = os.environ.get("SLACK_CHANNEL_ID", "")
    if slack_token and slack_channel:
        try:
            ts = send_slack_notification(data, pdf_bytes, slack_token, slack_channel)
            logger.info("slack_notification_sent", ts=ts, channel=slack_channel)
        except Exception as exc:
            logger.error("slack_notification_failed", error=str(exc))
    else:
        logger.warning("slack_env_vars_missing_skipping")

    # Step 4: AWS SES
    sender = os.environ.get("SES_SENDER_EMAIL", "")
    recipients_raw = os.environ.get("SES_RECIPIENT_EMAILS", "")
    recipients = [e.strip() for e in recipients_raw.split(",") if e.strip()]
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    if sender and recipients:
        try:
            msg_id = send_ses_email(data, pdf_bytes, sender, recipients, region)
            logger.info("ses_email_dispatched", message_id=msg_id)
        except Exception as exc:
            logger.error("ses_email_failed", error=str(exc))
    else:
        logger.warning("ses_env_vars_missing_skipping")

    logger.info("notifier_complete", game_date=game_date)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Game-Day Pre-Game Notifier")
    parser.add_argument(
        "--game-date",
        type=str,
        default=None,
        help="Game date in YYYY-MM-DD format (defaults to today)",
    )
    args = parser.parse_args()
    run_notifier(game_date=args.game_date)
