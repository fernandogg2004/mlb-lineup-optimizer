"""
MLB Lineup Optimizer — War Room Dashboard v2.0
===============================================
Interfaz táctica para el cuerpo técnico.

Vistas:
  • War Room       — Lineup óptimo, RAG, What-If pre-partido
  • Post-Game Review — Análisis histórico, comparativa real vs. proyectado

Arranque local:
    streamlit run app.py

Arranque en servidor (puerto 8501 ya expuesto en docker-compose.vps.yml):
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0

Variables de entorno:
    API_BASE_URL   URL de la FastAPI  (default: http://localhost:8000)
    DEMO_MODE      "1" = mock data    (default: "1")
"""
from __future__ import annotations

import os

import streamlit as st

from views.war_room  import render_war_room
from views.post_game import render_post_game

# ── Page config — debe ser la primera llamada Streamlit ───────────────────────
st.set_page_config(
    page_title="⚾ MLB War Room",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "MLB Lineup Optimizer War Room v2.0 — Powered by Claude"},
)

# ── Global CSS — dark war-room theme ──────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #080d1a; }
[data-testid="stSidebar"]          { background: #0d1528; border-right: 1px solid #1e2d4a; }
[data-testid="stHeader"]           { background: transparent; }
html, body, [class*="css"]         { font-family: 'Inter', 'SF Pro Display', sans-serif; }

/* ── Typography ── */
h1 { color: #e8eaf6; letter-spacing: -0.4px; margin-bottom: 2px !important; }
h2 { color: #c5cae9; font-size: 1.12rem; font-weight: 600; }
h3 { color: #9fa8da; font-size: 0.95rem; font-weight: 500; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 14px 18px;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #3d6fd4; }
[data-testid="stMetricValue"]   { color: #e8eaf6 !important; font-size: 1.9rem !important; font-weight: 700; }
[data-testid="stMetricLabel"]   { color: #90a4ae !important; font-size: 0.70rem !important;
                                   text-transform: uppercase; letter-spacing: 0.07em; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #90caf9 !important; font-weight: 500; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border: 1px solid #1e2d4a; border-radius: 8px; }

/* ── Inputs / Selects ── */
[data-baseweb="select"] > div { background: #111827 !important; border-color: #1e2d4a !important; }
[data-testid="stDateInput"] input { background: #111827 !important; border-color: #1e2d4a !important; }

/* ── Buttons ── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    border: none !important;
    height: 46px;
    font-size: 0.95rem;
    font-weight: 600;
    border-radius: 10px;
    transition: opacity 0.15s;
}
[data-testid="baseButton-primary"]:hover  { opacity: 0.85 !important; }
[data-testid="baseButton-secondary"] {
    border-color: #1e2d4a !important;
    color: #90a4ae !important;
}

/* ── Alert boxes ── */
.alert-positive {
    background: #0a2e1f; border-left: 4px solid #2e7d32;
    border-radius: 8px; padding: 12px 16px; color: #a5d6a7; margin: 8px 0;
}
.alert-negative {
    background: #2e0a0a; border-left: 4px solid #c62828;
    border-radius: 8px; padding: 12px 16px; color: #ef9a9a; margin: 8px 0;
}
.alert-neutral {
    background: #1a1f2e; border-left: 4px solid #37474f;
    border-radius: 8px; padding: 12px 16px; color: #90a4ae; margin: 8px 0;
}

/* ── Divider ── */
hr { border-color: #1e2d4a !important; margin: 16px 0 !important; }

/* ── Tab strip ── */
[data-baseweb="tab-list"]  { gap: 6px; border-bottom: 1px solid #1e2d4a; }
[data-baseweb="tab"]       { border-radius: 6px 6px 0 0 !important; }
[data-baseweb="tab-panel"] { padding-top: 18px !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { color: #90caf9 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* ── Toast ── */
[data-testid="stToast"] { background: #111827 !important; border: 1px solid #1e2d4a !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "war_room"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h1 style='color:#90caf9; font-size:1.45rem; margin-bottom:2px;'>⚾ MLB Optimizer</h1>"
        "<p style='color:#546e7a; font-size:0.76rem; margin-top:0;'>Tactical War Room v2.0</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Vista",
        options=["🏟️  War Room", "📊  Post-Game Review"],
        label_visibility="collapsed",
        key="nav_radio",
    )

    st.divider()

    api_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    demo    = os.environ.get("DEMO_MODE", "1") == "1"
    status  = "🟡 DEMO" if demo else "🟢 LIVE"

    st.markdown(f"""
    <div style='background:#0d1528; border-radius:8px; padding:12px 14px;
                font-size:0.78rem; color:#546e7a; line-height:1.85;'>
        <strong style='color:#4c9eff;'>Modo</strong> &nbsp;{status}<br>
        <code style='color:#78909c; font-size:0.70rem;'>{api_url}</code><br><br>
        <strong style='color:#4c9eff;'>Stack</strong><br>
        FastAPI · PostgreSQL · Airflow<br>
        Monte Carlo · RAG · LLM<br><br>
        <strong style='color:#4c9eff;'>Modelo</strong> v2.1.0<br>
        <strong style='color:#4c9eff;'>Autor</strong> Fernando
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("🔄 Limpiar Caché", use_container_width=True, type="secondary"):
        st.cache_data.clear()
        for k in ["baseline_scenario", "what_if_scenario", "what_if_delta",
                  "engine", "whatsif_result", "wr_game_pk"]:
            st.session_state.pop(k, None)
        st.success("Caché limpiado", icon="✅")

# ── Page routing ──────────────────────────────────────────────────────────────
if "War Room" in page:
    render_war_room()
else:
    render_post_game()
