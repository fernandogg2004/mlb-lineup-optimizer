"""
utils/charts.py
===============
Componentes Plotly reutilizables — paleta dark war-room.
"""
from __future__ import annotations

import plotly.graph_objects as go

# ── Color palette ──────────────────────────────────────────────────────────────
C = {
    "bg":      "rgba(0,0,0,0)",
    "surface": "#111827",
    "border":  "#1e2d4a",
    "grid":    "#1a2540",
    "green":   "#2e7d32",
    "green_lt":"#a5d6a7",
    "red":     "#c62828",
    "red_lt":  "#ef9a9a",
    "blue":    "#1565c0",
    "blue_lt": "#90caf9",
    "yellow":  "#f9a825",
    "text":    "#e8eaf6",
    "sub":     "#90a4ae",
}

_BASE = dict(
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["bg"],
    font=dict(color=C["text"], family="Inter, SF Pro, sans-serif"),
)


def er_comparison_chart(projected: float, actual: float, matchup: str = "") -> go.Figure:
    """Barras E[R] proyectado vs carreras reales."""
    delta  = actual - projected
    colors = [C["blue"], C["green"] if delta >= 0 else C["red"]]
    sign   = "+" if delta >= 0 else ""
    ann_c  = C["green_lt"] if delta >= 0 else C["red_lt"]

    fig = go.Figure(go.Bar(
        x=["E[R] Proyectado", "Carreras Reales"],
        y=[projected, actual],
        marker_color=colors,
        marker_line_width=0,
        text=[f"{projected:.2f}", f"{int(actual)}"],
        textposition="outside",
        textfont=dict(color=C["text"], size=16, family="monospace"),
        width=[0.38, 0.38],
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.add_annotation(
        x=1, y=max(projected, actual) + 0.8,
        text=f"Δ = {sign}{delta:.2f}",
        font=dict(color=ann_c, size=16, family="monospace"),
        showarrow=False,
    )
    fig.update_layout(
        **_BASE,
        margin=dict(l=16, r=16, t=44, b=16),
        title=dict(text=matchup, font=dict(color=C["sub"], size=13), x=0),
        xaxis=dict(gridcolor=C["grid"], linecolor=C["border"], tickfont=dict(color=C["text"], size=13)),
        yaxis=dict(
            gridcolor=C["grid"], linecolor=C["border"],
            range=[0, max(projected, actual) + 2.0],
            tickfont=dict(color=C["sub"]),
        ),
        height=320,
        showlegend=False,
    )
    return fig


def win_probability_gauge(probability: float) -> go.Figure:
    """Velocímetro semicircular para win probability."""
    pct = probability * 100
    bar_c = (
        C["green"] if pct >= 60 else
        C["blue"]  if pct >= 50 else
        C["yellow"]if pct >= 40 else
        C["red"]
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(size=36, color=C["text"])),
        title=dict(text="Probabilidad de Victoria", font=dict(size=13, color=C["sub"])),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                tickfont=dict(color=C["sub"], size=10),
                tickcolor=C["border"],
            ),
            bar=dict(color=bar_c, thickness=0.28),
            bgcolor=C["surface"],
            borderwidth=1,
            bordercolor=C["border"],
            steps=[
                {"range": [0,  40], "color": "#1a0d0d"},
                {"range": [40, 60], "color": "#0d1020"},
                {"range": [60,100], "color": "#0d1a0d"},
            ],
            threshold=dict(
                line=dict(color=C["text"], width=2),
                thickness=0.75,
                value=50,
            ),
        ),
    ))
    fig.update_layout(
        **_BASE,
        height=215,
        margin=dict(l=20, r=20, t=30, b=10),  # overrides _BASE intentionally
    )
    return fig


def ops_lineup_chart(lineup: list[dict]) -> go.Figure:
    """Barras horizontales de OPS por posición en el lineup."""
    names  = [f"{p['order']}. {p['name'].split()[-1]}" for p in lineup]
    values = [p["ops"] for p in lineup]

    def _c(v: float) -> str:
        return (
            C["green"]  if v >= 0.900 else
            C["blue"]   if v >= 0.800 else
            C["yellow"] if v >= 0.700 else
            "#546e7a"
        )

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=[_c(v) for v in values],
        marker_line_width=0,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color=C["text"], size=11, family="monospace"),
        hovertemplate="%{y}: OPS %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(
        x=0.750, line_dash="dot", line_color=C["yellow"],
        annotation_text=".750", annotation_font_color=C["sub"],
        annotation_font_size=10,
    )
    fig.update_layout(
        **_BASE,
        title=dict(text="OPS por Posición en el Lineup", font=dict(color=C["sub"], size=12), x=0),
        xaxis=dict(
            gridcolor=C["grid"], linecolor=C["border"],
            range=[0.50, 1.18],
            tickfont=dict(color=C["sub"], size=10),
        ),
        yaxis=dict(
            gridcolor=C["grid"], linecolor=C["border"],
            tickfont=dict(color=C["text"], size=11),
            autorange="reversed",
        ),
        height=310,
        margin=dict(l=16, r=16, t=44, b=16),
        showlegend=False,
    )
    return fig


def delta_bar_chart(baseline_er: float, whatsif_er: float) -> go.Figure:
    """Comparación Baseline vs What-If."""
    delta   = whatsif_er - baseline_er
    wi_c    = C["green"]    if delta >= 0 else C["red"]
    wi_tc   = C["green_lt"] if delta >= 0 else C["red_lt"]
    sign    = "+" if delta >= 0 else ""

    fig = go.Figure(go.Bar(
        x=["Baseline", "What-If"],
        y=[baseline_er, whatsif_er],
        marker_color=[C["blue"], wi_c],
        marker_line_width=0,
        text=[f"{baseline_er:.3f}", f"{whatsif_er:.3f}"],
        textposition="outside",
        textfont=dict(color=C["text"], size=14, family="monospace"),
        width=[0.38, 0.38],
    ))
    fig.add_annotation(
        x=1, y=max(baseline_er, whatsif_er) + 0.45,
        text=f"Δ = {sign}{delta:.3f}",
        font=dict(color=wi_tc, size=16, family="monospace"),
        showarrow=False,
    )
    lo = min(baseline_er, whatsif_er)
    hi = max(baseline_er, whatsif_er)
    fig.update_layout(
        **_BASE,
        xaxis=dict(gridcolor=C["grid"], tickfont=dict(color=C["text"], size=13)),
        yaxis=dict(
            gridcolor=C["grid"],
            range=[max(0, lo - 0.6), hi + 0.9],
            tickfont=dict(color=C["sub"]),
        ),
        height=225,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig
