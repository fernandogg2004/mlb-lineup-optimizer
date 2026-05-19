"""
app_dashboard.py
================
MLB War Room — Streamlit Tactical Dashboard.

Optimised for iPad (1024×768+) with a dark "War Room" theme.
Tabs: Lineup Óptimo · Analytics · What-If Analysis.

Run:
    streamlit run app_dashboard.py --server.port 8501 --server.address 0.0.0.0

Environment variables:
    API_BASE_URL   FastAPI backend URL (default: http://localhost:8000)
    DEMO_MODE      Set to "1" to use synthetic demo data (no API required)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from what_if_engine import (
    DeltaReport,
    LineupScenario,
    PlayerProfile,
    WhatIfEngine,
    synthetic_zone_xwoba,
)

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="⚾ MLB War Room",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "MLB Lineup Optimizer War Room v1.0 — Anthropic Claude"},
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEMO_MODE: bool = os.environ.get("DEMO_MODE", "0") == "1"
API_TIMEOUT: float = 90.0

MLB_TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]

_OUTCOME_NAMES = ["OUT", "K", "BB/HBP", "1B", "2B", "3B", "HR"]

# ---------------------------------------------------------------------------
# Custom CSS — dark war-room theme, iPad-friendly touch targets
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {background: #080d1a;}
[data-testid="stSidebar"] {background: #0d1528; border-right: 1px solid #1e2d4a;}
[data-testid="stHeader"] {background: transparent;}
html, body, [class*="css"] {font-family: 'Inter', 'SF Pro', sans-serif;}

/* ── Headings ── */
h1 {color: #e8eaf6; letter-spacing: -0.5px;}
h2 {color: #c5cae9; font-size: 1.1rem; font-weight: 600;}
h3 {color: #9fa8da; font-size: 0.95rem; font-weight: 500;}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 12px 16px;
}

/* ── Player card ── */
.player-card {
    background: linear-gradient(135deg, #111827, #0d1f3c);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.player-card:hover {border-color: #3d6fd4;}

/* ── Batting position badge ── */
.pos-badge {
    display: inline-block;
    background: #1a3a6b;
    color: #90caf9;
    font-size: 1.4rem;
    font-weight: 800;
    width: 42px; height: 42px;
    line-height: 42px;
    text-align: center;
    border-radius: 50%;
    margin-right: 8px;
}

/* ── Delta alerts ── */
.delta-positive {
    background: #0a2e1f; border: 1px solid #2e7d32;
    border-radius: 10px; padding: 10px 14px; color: #a5d6a7;
}
.delta-negative {
    background: #2e0000; border: 1px solid #c62828;
    border-radius: 10px; padding: 10px 14px; color: #ef9a9a;
}
.delta-neutral {
    background: #1a1f2e; border: 1px solid #37474f;
    border-radius: 10px; padding: 10px 14px; color: #90a4ae;
}

/* ── Buttons — bigger touch targets for iPad ── */
[data-testid="baseButton-primary"] {
    height: 48px; font-size: 1rem; border-radius: 10px;
}

/* ── Sidebar labels ── */
.sidebar-label {color: #90a4ae; font-size: 0.8rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;}

/* ── Tab strip ── */
[data-baseweb="tab-list"] {gap: 8px;}
[data-baseweb="tab"] {border-radius: 8px !important; padding: 8px 18px !important;}

/* ── Scrollbar ── */
::-webkit-scrollbar {width: 6px;}
::-webkit-scrollbar-thumb {background: #1e3a5f; border-radius: 3px;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults: dict = {
        "rival_team": "NYY",
        "pitcher_id": 1000,
        "pitcher_name": "Gerrit Cole",
        "pitcher_hand": "R",
        "temperature_f": 72.0,
        "wind_mph": 8.0,
        "wind_dir": 180.0,
        "humidity": 55.0,
        "park_factor_hr": 1.05,
        "park_factor_xb": 1.02,
        "baseline_scenario": None,
        "what_if_scenario": None,
        "what_if_delta": None,
        "engine": None,
        "loading": False,
        "last_optimized": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state["engine"] is None:
        st.session_state["engine"] = WhatIfEngine(api_base_url=API_BASE_URL)


_init_state()

# ---------------------------------------------------------------------------
# Demo data (used when DEMO_MODE=1 or API unreachable)
# ---------------------------------------------------------------------------


def _demo_players() -> list[PlayerProfile]:
    names = [
        ("Aaron Judge", "R", 0.421, 0.406, 0.342),
        ("Gleyber Torres", "R", 0.349, 0.341, 0.162),
        ("Giancarlo Stanton", "R", 0.455, 0.378, 0.310),
        ("Anthony Rizzo", "L", 0.329, 0.370, 0.185),
        ("Anthony Volpe", "R", 0.318, 0.320, 0.148),
        ("DJ LeMahieu", "R", 0.305, 0.349, 0.109),
        ("Oswaldo Peraza", "R", 0.295, 0.305, 0.133),
        ("Jose Bauers", "L", 0.290, 0.360, 0.120),
        ("Austin Wells", "L", 0.302, 0.332, 0.155),
    ]
    players = []
    for i, (name, stand, woba, obp, iso) in enumerate(names):
        rng = np.random.default_rng(i * 137)
        raw = rng.dirichlet([5, 3, 1, 2, 0.6, 0.08, 0.4]).tolist()
        players.append(
            PlayerProfile(
                player_id=1000 + i,
                player_name=name,
                obp=obp,
                woba=woba,
                iso=iso,
                batter_stand=stand,
                prob_vector=raw,
            )
        )
    return players


def _demo_scenario(players: list[PlayerProfile], name: str = "baseline") -> LineupScenario:
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    return LineupScenario(
        players=players,
        expected_runs=4.80 + rng.uniform(-0.3, 0.3),
        win_probability=0.56 + rng.uniform(-0.05, 0.05),
        scenario_name=name,
        optimization_mode="full",
        total_evaluations=30000,
        elapsed_seconds=22.4,
    )


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _call_optimize_api(
    roster_payload: str,
    pitcher_hand: str,
    park_factor_hr: float,
    park_factor_xb: float,
    fast_mode: bool,
) -> dict | None:
    """Cached API call: re-fetches only when inputs change or cache expires."""
    import json

    payload = json.loads(roster_payload)
    payload["rival_pitcher"]["hand"] = pitcher_hand
    payload["park_factor_hr"] = park_factor_hr
    payload["park_factor_xb"] = park_factor_xb
    payload["fast_mode"] = fast_mode

    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            r = client.post(f"{API_BASE_URL}/v1/optimize/lineup", json=payload)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        st.warning(f"API unreachable — switching to demo mode. ({exc})")
        return None


# ---------------------------------------------------------------------------
# Plotly components
# ---------------------------------------------------------------------------


def _zone_heatmap(player: PlayerProfile) -> go.Figure:
    zone = synthetic_zone_xwoba(player.player_id, player.woba, player.batter_stand)

    zone_labels = [[f"{v:.3f}" for v in row] for row in zone]
    row_labels = ["High", "H-Mid", "Mid", "L-Mid", "Low"]
    col_labels = ["In", "In-M", "Ctr", "Out-M", "Out"]

    fig = go.Figure(
        go.Heatmap(
            z=zone,
            text=zone_labels,
            texttemplate="%{text}",
            textfont={"size": 9, "color": "white"},
            colorscale=[
                [0.00, "#b71c1c"],
                [0.35, "#e53935"],
                [0.50, "#fbc02d"],
                [0.65, "#43a047"],
                [1.00, "#1b5e20"],
            ],
            zmin=0.15,
            zmax=0.65,
            showscale=False,
            x=col_labels,
            y=row_labels,
        )
    )
    # Strike zone boundary
    fig.add_shape(
        type="rect",
        x0=0.5, x1=3.5, y0=0.5, y1=3.5,
        line=dict(color="#e0e0e0", width=2, dash="dash"),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=140,
        xaxis=dict(showgrid=False, tickfont=dict(size=8, color="#90a4ae")),
        yaxis=dict(showgrid=False, tickfont=dict(size=8, color="#90a4ae"), autorange="reversed"),
    )
    return fig


def _outcome_donut(player: PlayerProfile) -> go.Figure:
    colors = ["#455a64", "#e53935", "#1e88e5", "#43a047", "#fb8c00", "#ab47bc", "#e53935"]
    fig = go.Figure(
        go.Pie(
            labels=_OUTCOME_NAMES,
            values=player.prob_vector,
            hole=0.62,
            marker_colors=colors,
            textinfo="none",
            hovertemplate="%{label}: %{value:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=130,
        annotations=[
            dict(
                text=f"wOBA<br><b>{player.woba:.3f}</b>",
                x=0.5, y=0.5,
                font_size=11,
                font_color="#c5cae9",
                showarrow=False,
            )
        ],
    )
    return fig


def _er_gauge(er: float, pw: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=er,
            title={"text": "E[Runs Scored]", "font": {"color": "#c5cae9", "size": 14}},
            number={"font": {"color": "#90caf9", "size": 36}},
            gauge={
                "axis": {"range": [0, 10], "tickcolor": "#546e7a"},
                "bar": {"color": "#1565c0"},
                "bgcolor": "#1a2035",
                "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0, 3.5], "color": "#1a1f2e"},
                    {"range": [3.5, 5.5], "color": "#0d2040"},
                    {"range": [5.5, 10], "color": "#0a2e3f"},
                ],
                "threshold": {
                    "line": {"color": "#43a047", "width": 2},
                    "thickness": 0.75,
                    "value": er,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        height=200,
        font=dict(color="#c5cae9"),
        annotations=[
            dict(
                text=f"P(W) = <b>{pw:.1%}</b>",
                x=0.5, y=-0.05,
                font_size=15,
                font_color="#a5d6a7",
                showarrow=False,
            )
        ],
    )
    return fig


def _delta_bar(baseline_er: float, what_if_er: float) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=["Baseline", "What-If"],
            y=[baseline_er, what_if_er],
            marker_color=["#1565c0", "#43a047" if what_if_er >= baseline_er else "#c62828"],
            text=[f"{baseline_er:.3f}", f"{what_if_er:.3f}"],
            textposition="outside",
            textfont=dict(color="#c5cae9", size=13),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=190,
        yaxis=dict(
            range=[max(0, min(baseline_er, what_if_er) - 0.5), max(baseline_er, what_if_er) + 0.5],
            gridcolor="#1e2d4a",
            tickfont=dict(color="#90a4ae"),
        ),
        xaxis=dict(tickfont=dict(color="#c5cae9", size=13)),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Player card renderer
# ---------------------------------------------------------------------------


def _render_player_card(pos: int, player: PlayerProfile) -> None:
    with st.container():
        st.markdown(
            f"""
            <div class="player-card">
              <span class="pos-badge">{pos}</span>
              <strong style="color:#e8eaf6; font-size:1rem;">{player.player_name}</strong>
              <span style="color:#546e7a; font-size:0.8rem; margin-left:8px;">
                {'🔵' if player.batter_stand=='L' else '🔴' if player.batter_stand=='R' else '🟡'}
                {player.batter_stand}HB
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("xwOBA", f"{player.woba:.3f}")
        m2.metric("OBP", f"{player.obp:.3f}")
        m3.metric("ISO", f"{player.iso:.3f}")

        z1, z2 = st.columns([3, 2])
        with z1:
            st.caption("Strike Zone xwOBA")
            st.plotly_chart(
                _zone_heatmap(player),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"zone_{player.player_id}_{pos}",
            )
        with z2:
            st.caption("PA Outcome Mix")
            st.plotly_chart(
                _outcome_donut(player),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"donut_{player.player_id}_{pos}",
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            "<h1 style='color:#90caf9; font-size:1.4rem; margin-bottom:0'>⚾ MLB War Room</h1>",
            unsafe_allow_html=True,
        )
        st.caption(f"API: `{API_BASE_URL}` {'🟢' if not DEMO_MODE else '🟡 DEMO'}")
        st.divider()

        st.markdown("<p class='sidebar-label'>Configuración del Partido</p>", unsafe_allow_html=True)
        rival = st.selectbox("Equipo Rival", MLB_TEAMS, index=MLB_TEAMS.index("NYY"))
        p_name = st.text_input("Lanzador Abridor Rival", value="Gerrit Cole")
        p_hand = st.radio("Mano del Lanzador", ["R", "L"], horizontal=True)

        st.divider()
        st.markdown("<p class='sidebar-label'>Condiciones Climáticas</p>", unsafe_allow_html=True)
        temp = st.slider("Temperatura (°F)", 40, 105, 72)
        wind = st.slider("Viento (mph)", 0, 35, 8)
        wind_dir = st.select_slider(
            "Dirección del Viento",
            options=["⬇ Hacia plato", "↗ Cruzado", "⬆ Hacia CF"],
            value="↗ Cruzado",
        )
        humidity = st.slider("Humedad (%)", 20, 95, 55)

        st.divider()
        st.markdown("<p class='sidebar-label'>Park Factors</p>", unsafe_allow_html=True)
        pf_hr = st.slider("HR Factor", 0.80, 1.30, 1.05, step=0.01, format="%.2f")
        pf_xb = st.slider("XB Factor", 0.80, 1.20, 1.02, step=0.01, format="%.2f")

        st.divider()
        run_btn = st.button(
            "🚀 Generar Lineup Óptimo",
            type="primary",
            use_container_width=True,
        )

        st.markdown(
            f"""
            <div style='margin-top:20px; padding:10px; background:#0d1528;
                        border-radius:8px; font-size:0.78rem; color:#546e7a;'>
            Temp: {temp}°F | Viento: {wind} mph<br>
            PF-HR: {pf_hr:.2f} | PF-XB: {pf_xb:.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "rival_team": rival,
        "pitcher_name": p_name,
        "pitcher_hand": p_hand,
        "temperature_f": temp,
        "wind_mph": wind,
        "wind_dir": wind_dir,
        "humidity": humidity,
        "park_factor_hr": pf_hr,
        "park_factor_xb": pf_xb,
        "run_clicked": run_btn,
    }


# ---------------------------------------------------------------------------
# Tab 1 — Optimal Lineup
# ---------------------------------------------------------------------------


def _render_lineup_tab(scenario: LineupScenario, config: dict) -> None:
    # Header row
    hcol1, hcol2, hcol3, hcol4 = st.columns([3, 2, 2, 2])
    with hcol1:
        st.markdown(
            f"<h2 style='color:#90caf9'>vs. {config['rival_team']} · "
            f"{config['pitcher_name']} ({config['pitcher_hand']}HP)</h2>",
            unsafe_allow_html=True,
        )
    with hcol2:
        st.plotly_chart(
            _er_gauge(scenario.expected_runs, scenario.win_probability),
            use_container_width=True,
            config={"displayModeBar": False},
            key="er_gauge_main",
        )
    with hcol3:
        st.metric("E[Carreras]", f"{scenario.expected_runs:.3f}")
        st.metric("P(Victoria)", f"{scenario.win_probability:.1%}")
    with hcol4:
        st.metric("Evaluaciones GA", f"{scenario.total_evaluations:,}")
        st.metric("Tiempo optimización", f"{scenario.elapsed_seconds:.1f}s")

    st.divider()

    # Player cards — 3 columns
    col_a, col_b, col_c = st.columns(3)
    targets = [col_a, col_b, col_c, col_a, col_b, col_c, col_a, col_b, col_c]
    for pos, (player, col) in enumerate(zip(scenario.players, targets), start=1):
        with col:
            _render_player_card(pos, player)


# ---------------------------------------------------------------------------
# Tab 2 — Analytics
# ---------------------------------------------------------------------------


def _render_analytics_tab(scenario: LineupScenario) -> None:
    st.subheader("Distribución de Outcomes por Bateador")

    names = [p.player_name.split()[-1] for p in scenario.players]
    probs_matrix = np.array([p.prob_vector for p in scenario.players])

    # Stacked bar chart — outcome distribution per batter
    outcome_colors = [
        "#455a64", "#e53935", "#1e88e5",
        "#43a047", "#fb8c00", "#ab47bc", "#c62828",
    ]
    fig_stack = go.Figure()
    for i, (outcome, color) in enumerate(zip(_OUTCOME_NAMES, outcome_colors)):
        fig_stack.add_trace(
            go.Bar(
                name=outcome,
                x=names,
                y=probs_matrix[:, i],
                marker_color=color,
                hovertemplate=f"{outcome}: %{{y:.1%}}<extra></extra>",
            )
        )
    fig_stack.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.06, font=dict(color="#c5cae9", size=11)),
        xaxis=dict(tickfont=dict(color="#c5cae9"), gridcolor="#1e2d4a"),
        yaxis=dict(
            tickformat=".0%",
            tickfont=dict(color="#c5cae9"),
            gridcolor="#1e2d4a",
        ),
        height=310,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#c5cae9"),
    )
    st.plotly_chart(fig_stack, use_container_width=True, config={"displayModeBar": False})

    st.divider()
    st.subheader("xwOBA por Posición en el Orden")

    woba_vals = [p.woba for p in scenario.players]
    fig_woba = go.Figure(
        go.Scatter(
            x=list(range(1, 10)),
            y=woba_vals,
            mode="lines+markers+text",
            text=[f"{v:.3f}" for v in woba_vals],
            textposition="top center",
            textfont=dict(color="#90caf9", size=11),
            line=dict(color="#1565c0", width=2),
            marker=dict(
                size=14,
                color=woba_vals,
                colorscale=[[0, "#c62828"], [0.5, "#fbc02d"], [1, "#2e7d32"]],
                showscale=False,
            ),
            hovertemplate="Posición %{x}: xwOBA %{y:.3f}<extra></extra>",
        )
    )
    fig_woba.add_hline(
        y=np.mean(woba_vals),
        line_dash="dot",
        line_color="#546e7a",
        annotation_text=f"Media: {np.mean(woba_vals):.3f}",
        annotation_font_color="#90a4ae",
    )
    fig_woba.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickvals=list(range(1, 10)),
            ticktext=[f"{i}°" for i in range(1, 10)],
            tickfont=dict(color="#c5cae9"),
            gridcolor="#1e2d4a",
        ),
        yaxis=dict(
            range=[0.25, 0.50],
            tickformat=".3f",
            tickfont=dict(color="#c5cae9"),
            gridcolor="#1e2d4a",
        ),
        height=250,
        margin=dict(l=10, r=10, t=10, b=30),
    )
    st.plotly_chart(fig_woba, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Tab 3 — What-If Analysis
# ---------------------------------------------------------------------------


def _render_whatif_tab(baseline: LineupScenario, config: dict) -> None:
    engine: WhatIfEngine = st.session_state["engine"]
    if not engine.has_baseline():
        engine.set_baseline(baseline)

    st.subheader("🔬 Modificar Lineup — Análisis What-If")
    st.caption(
        "Ajusta la alineación (lesión, emergente zurdo, etc.) y pulsa **Recalcular** "
        "para ver el impacto en E[R] y P(Victoria) en tiempo real."
    )

    # Editable lineup table
    lineup_df = pd.DataFrame(
        {
            "Pos": list(range(1, 10)),
            "Jugador": [p.player_name for p in baseline.players],
            "Mano": [p.batter_stand for p in baseline.players],
            "xwOBA": [round(p.woba, 3) for p in baseline.players],
            "OBP": [round(p.obp, 3) for p in baseline.players],
            "ISO": [round(p.iso, 3) for p in baseline.players],
            "Activo": [True] * 9,
        }
    )

    st.markdown("**Lineup actual — edita y pulsa Recalcular:**")
    edited_df = st.data_editor(
        lineup_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pos": st.column_config.NumberColumn("Pos", disabled=True, width="small"),
            "Jugador": st.column_config.TextColumn("Jugador", width="medium"),
            "Mano": st.column_config.SelectboxColumn("Mano", options=["L", "R", "S"], width="small"),
            "xwOBA": st.column_config.NumberColumn("xwOBA", format="%.3f", min_value=0.0, max_value=1.0),
            "OBP": st.column_config.NumberColumn("OBP", format="%.3f", min_value=0.0, max_value=1.0),
            "ISO": st.column_config.NumberColumn("ISO", format="%.3f", min_value=0.0, max_value=0.5),
            "Activo": st.column_config.CheckboxColumn("En lineup", width="small"),
        },
        num_rows="fixed",
        key="what_if_editor",
    )

    wcol1, wcol2, _ = st.columns([2, 2, 4])
    with wcol1:
        fast_mode = st.toggle("Modo rápido (5k sims)", value=True)
    with wcol2:
        recalc = st.button("⚡ Recalcular Escenario", type="primary", use_container_width=True)

    if recalc:
        # Build modified player list from edited dataframe
        modified_players: list[PlayerProfile] = []
        original_by_name = {p.player_name: p for p in baseline.players}

        for _, row in edited_df.iterrows():
            if not row["Activo"]:
                continue
            original = original_by_name.get(row["Jugador"])
            if original:
                modified = PlayerProfile(
                    player_id=original.player_id,
                    player_name=row["Jugador"],
                    obp=row["OBP"],
                    woba=row["xwOBA"],
                    iso=row["ISO"],
                    batter_stand=row["Mano"],
                    prob_vector=original.prob_vector,
                )
            else:
                # New player not in original roster — use league-average prob vector
                avg_probs = [0.284, 0.222, 0.082, 0.148, 0.048, 0.009, 0.028]
                modified = PlayerProfile(
                    player_id=hash(row["Jugador"]) % 100_000,
                    player_name=row["Jugador"],
                    obp=row["OBP"],
                    woba=row["xwOBA"],
                    iso=row["ISO"],
                    batter_stand=row["Mano"],
                    prob_vector=avg_probs,
                )
            modified_players.append(modified)

        if len(modified_players) != 9:
            st.error(f"El lineup debe tener exactamente 9 jugadores activos (tiene {len(modified_players)}).")
            return

        with st.spinner("Ejecutando simulación Monte Carlo…"):
            if DEMO_MODE:
                time.sleep(1.5)
                new_scenario = _demo_scenario(modified_players, name="what-if")
            else:
                new_scenario = engine.fetch_scenario(
                    players=modified_players,
                    pitcher_id=st.session_state["pitcher_id"],
                    pitcher_name=config["pitcher_name"],
                    pitcher_hand=config["pitcher_hand"],
                    park_factor_hr=config["park_factor_hr"],
                    park_factor_xb=config["park_factor_xb"],
                    fast_mode=fast_mode,
                    scenario_name="what-if",
                )

        if new_scenario.api_error:
            st.error(f"Error en API: {new_scenario.api_error}")
            return

        st.session_state["what_if_scenario"] = new_scenario
        try:
            st.session_state["what_if_delta"] = engine.delta(new_scenario)
        except Exception as exc:
            st.error(str(exc))

    # Display delta results
    delta: DeltaReport | None = st.session_state.get("what_if_delta")
    if delta is None:
        st.info("Modifica la alineación y pulsa **Recalcular** para ver el impacto.")
        return

    st.divider()
    st.subheader("Impacto del Cambio")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("E[R] Baseline", f"{delta.baseline_er:.3f}")
    r2.metric(
        "E[R] What-If",
        f"{delta.scenario_er:.3f}",
        delta=f"{delta.delta_er:+.3f}",
        delta_color="normal",
    )
    r3.metric("P(W) Baseline", f"{delta.baseline_pw:.1%}")
    r4.metric(
        "P(W) What-If",
        f"{delta.scenario_pw:.1%}",
        delta=f"{delta.delta_pw:+.1%}",
        delta_color="normal",
    )

    # Alert box
    if delta.impact_level == "NEUTRAL":
        css_class, icon, msg = "delta-neutral", "↔", "Cambio mínimo"
    elif delta.is_improvement:
        css_class, icon, msg = "delta-positive", "↑", f"MEJORA [{delta.impact_level}]"
    else:
        css_class, icon, msg = "delta-negative", "↓", f"PÉRDIDA [{delta.impact_level}]"

    st.markdown(
        f"""
        <div class="{css_class}" style="text-align:center; font-size:1.1rem; margin:12px 0;">
            {icon} &nbsp; ΔE[R] = <strong>{delta.delta_er:+.3f}</strong> &nbsp;|&nbsp;
            ΔP(W) = <strong>{delta.delta_pw:+.1%}</strong>
            &nbsp;&nbsp;— {msg}
        </div>
        """,
        unsafe_allow_html=True,
    )

    gc1, gc2 = st.columns(2)
    with gc1:
        st.caption("E[R]: Baseline vs. What-If")
        st.plotly_chart(
            _delta_bar(delta.baseline_er, delta.scenario_er),
            use_container_width=True,
            config={"displayModeBar": False},
            key="delta_bar",
        )
    with gc2:
        new_scenario: LineupScenario | None = st.session_state.get("what_if_scenario")
        if new_scenario:
            st.caption("Nuevo Lineup (What-If)")
            for pos, player in enumerate(new_scenario.players, start=1):
                st.markdown(
                    f"<span style='color:#90a4ae'>{pos}.</span> "
                    f"<span style='color:#e8eaf6'>{player.player_name}</span> "
                    f"<span style='color:#546e7a; font-size:0.85rem;'>"
                    f"xwOBA {player.woba:.3f}</span>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    config = _render_sidebar()

    # Header
    st.markdown(
        "<h1 style='margin-bottom:0'>⚾ MLB War Room — Tactical Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Rival: **{config['rival_team']}** · "
        f"Lanzador: **{config['pitcher_name']} ({config['pitcher_hand']}HP)** · "
        f"Temp: {config['temperature_f']:.0f}°F · Viento: {config['wind_mph']:.0f}mph"
    )

    # Load / refresh optimal lineup
    if config["run_clicked"] or st.session_state["baseline_scenario"] is None:
        with st.spinner("🧬 Ejecutando optimizador genético…"):
            if DEMO_MODE:
                time.sleep(2)
                players = _demo_players()
                scenario = _demo_scenario(players)
            else:
                players = _demo_players()  # In production: load from roster API
                # Build request and call API (returns None on error → fall back to demo)
                import json
                payload_str = json.dumps(
                    WhatIfEngine.build_request_payload(
                        players=players,
                        pitcher_id=st.session_state["pitcher_id"],
                        pitcher_name=config["pitcher_name"],
                        pitcher_hand=config["pitcher_hand"],
                        park_factor_hr=config["park_factor_hr"],
                        park_factor_xb=config["park_factor_xb"],
                        opp_lineup_probs=None,
                        fast_mode=False,
                    )
                )
                api_result = _call_optimize_api(
                    roster_payload=payload_str,
                    pitcher_hand=config["pitcher_hand"],
                    park_factor_hr=config["park_factor_hr"],
                    park_factor_xb=config["park_factor_xb"],
                    fast_mode=False,
                )
                if api_result:
                    # Reorder players per API response
                    id_map = {p.player_id: p for p in players}
                    ordered = [id_map.get(pid, players[i]) for i, pid in enumerate(api_result["lineup_player_ids"])]
                    scenario = LineupScenario(
                        players=ordered,
                        expected_runs=api_result["expected_runs"],
                        win_probability=api_result["win_probability"],
                        optimization_mode=api_result["optimization_mode"],
                        total_evaluations=api_result["total_evaluations"],
                        elapsed_seconds=api_result["elapsed_seconds"],
                    )
                else:
                    scenario = _demo_scenario(players)

        st.session_state["baseline_scenario"] = scenario
        st.session_state["what_if_scenario"] = None
        st.session_state["what_if_delta"] = None
        # Reset engine baseline
        st.session_state["engine"].set_baseline(scenario)

    scenario: LineupScenario = st.session_state["baseline_scenario"]

    # Main tabs
    tab_lineup, tab_analytics, tab_whatif = st.tabs(
        ["📋 Lineup Óptimo", "📊 Analytics", "🔬 What-If"]
    )
    with tab_lineup:
        _render_lineup_tab(scenario, config)
    with tab_analytics:
        _render_analytics_tab(scenario)
    with tab_whatif:
        _render_whatif_tab(scenario, config)


main()
