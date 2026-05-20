"""
views/war_room.py
=================
Vista War Room — estrategia pre-partido.

Secciones:
  1. Selector de partidos del día
  2. KPI cards (E[R], Win%, confianza, sims)
  3. Lineup óptimo + gauge + OPS chart
  4. Análisis RAG expandible
  5. Motor What-If interactivo
"""
from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st

from api.client import get_todays_games, get_game_lineup, simulate_whatsif
from utils.charts import ops_lineup_chart, win_probability_gauge, delta_bar_chart
from what_if_engine import WhatIfEngine

DEMO_MODE: bool = os.environ.get("DEMO_MODE", "1") == "1"


# ── Session state ──────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict = {
        "engine":            None,
        "baseline_scenario": None,
        "what_if_scenario":  None,
        "what_if_delta":     None,
        "whatsif_result":    None,
        "wr_game_pk":        None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state["engine"] is None:
        st.session_state["engine"] = WhatIfEngine(
            api_base_url=os.environ.get("API_BASE_URL", "http://localhost:8000")
        )


# ── Lineup table ───────────────────────────────────────────────────────────────
def _lineup_table(lineup: list[dict]) -> None:
    df = pd.DataFrame(lineup)[["order", "name", "pos", "hand", "avg", "ops", "woba", "obp"]]
    df.columns = ["#", "Jugador", "Pos", "Bat", "AVG", "OPS", "wOBA", "OBP"]

    def _row_bg(row):
        ops = row["OPS"]
        if ops >= 0.900: bg = "#0d2a0d"
        elif ops >= 0.800: bg = "#0d1a2e"
        elif ops >= 0.700: bg = "#1a1a0a"
        else: bg = "#1a1a1a"
        return [f"background-color: {bg}"] * len(row)

    styled = (
        df.style
        .apply(_row_bg, axis=1)
        .format({"AVG": "{:.3f}", "OPS": "{:.3f}", "wOBA": "{:.3f}", "OBP": "{:.3f}"})
        .set_properties(**{"font-family": "monospace", "font-size": "13px"})
        .bar(subset=["OPS"], color="#1565c0", vmin=0.600, vmax=1.100)
    )
    st.dataframe(styled, use_container_width=True, height=370, hide_index=True)


# ── What-If engine ─────────────────────────────────────────────────────────────
def _render_whatsif(game_pk: int, data: dict) -> None:
    st.markdown("### 🔄 Motor What-If — Simulador Interactivo")
    st.caption(
        "Modifica hasta dos posiciones del lineup y recalcula el impacto en "
        "E[R] y P(Victoria) mediante Monte Carlo."
    )

    lineup = data["lineup"]
    bench  = data["bench"]

    slot_opts  = {f"#{p['order']} — {p['name']} ({p['pos']})": p for p in lineup}
    bench_opts = {f"{p['name']} ({p['pos']})": p for p in bench}

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cambio 1**")
        slot1  = st.selectbox("Reemplazar",  list(slot_opts),  key="wi_s1")
        bench1 = st.selectbox("Por (banca)", list(bench_opts), key="wi_b1")
    with c2:
        st.markdown("**Cambio 2 *(opcional)***")
        other  = ["— Sin cambio —"] + [k for k in slot_opts if k != slot1]
        slot2  = st.selectbox("Reemplazar",  other,            key="wi_s2")
        bench2 = (
            st.selectbox("Por (banca)", list(bench_opts), key="wi_b2")
            if slot2 != "— Sin cambio —" else None
        )

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        recalc = st.button("⚡ Recalcular Impacto", type="primary", use_container_width=True)

    if recalc:
        changes: dict[int, dict] = {slot_opts[slot1]["order"]: bench_opts[bench1]}
        if slot2 != "— Sin cambio —" and bench2:
            changes[slot_opts[slot2]["order"]] = bench_opts[bench2]

        with st.spinner("Ejecutando Monte Carlo…"):
            if DEMO_MODE:
                time.sleep(0.6)
            result = simulate_whatsif(game_pk, changes)

        st.session_state["whatsif_result"] = result

    result = st.session_state.get("whatsif_result")
    if result is None:
        st.info(
            "Selecciona los cambios y pulsa **Recalcular Impacto** para ver el análisis.",
            icon="💡",
        )
        return

    # Results
    st.divider()
    st.markdown("#### Resultado de la Simulación")

    delta_er = result["delta_er"]
    delta_wp = result["delta_win_probability"] * 100
    is_pos   = delta_er >= 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("E[R] Original",  f"{result['base_expected_runs']:.3f}")
    m2.metric(
        "E[R] Nuevo",
        f"{result['new_expected_runs']:.3f}",
        delta=f"{'+' if delta_er >= 0 else ''}{delta_er:.3f}",
        delta_color="normal",
    )
    m3.metric("P(W) Original", f"{result['base_win_probability']*100:.1f}%")
    m4.metric(
        "P(W) Nueva",
        f"{result['new_win_probability']*100:.1f}%",
        delta=f"{'+' if delta_wp >= 0 else ''}{delta_wp:.1f}pp",
        delta_color="normal",
    )

    css  = "alert-positive" if is_pos else "alert-negative"
    icon = "✅ MEJORA" if is_pos else "⚠️ PÉRDIDA"
    msg  = (
        f"ΔE[R] = <strong>{'+' if delta_er >= 0 else ''}{delta_er:.3f}</strong>"
        f" &nbsp;|&nbsp; "
        f"ΔP(W) = <strong>{'+' if delta_wp >= 0 else ''}{delta_wp:.1f}pp</strong>"
    )
    st.markdown(
        f'<div class="{css}" style="text-align:center; font-size:1rem;">'
        f'{icon} &nbsp;— {msg}</div>',
        unsafe_allow_html=True,
    )

    bar_col, ci_col = st.columns([3, 2])
    with bar_col:
        st.plotly_chart(
            delta_bar_chart(result["base_expected_runs"], result["new_expected_runs"]),
            width="stretch",
            config={"displayModeBar": False},
            key="wi_delta_bar",
        )
    with ci_col:
        lo, hi = result["confidence_interval"]
        st.markdown(f"""
        <div style='background:#111827; border:1px solid #1e2d4a; border-radius:10px;
                    padding:16px; margin-top:8px; font-size:0.88rem; color:#90a4ae;
                    line-height:1.9;'>
            <strong style='color:#c5cae9;'>IC 95%</strong><br>
            <span style='font-family:monospace; font-size:1.05rem; color:#e8eaf6;'>
                [{lo:.2f}, {hi:.2f}] carreras
            </span><br>
            <strong style='color:#c5cae9;'>Simulaciones</strong><br>
            {result['simulation_n']:,} iter. Monte Carlo
        </div>
        """, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def render_war_room() -> None:
    _init_state()

    st.markdown(
        "<h1>🏟️ War Room"
        "<span style='font-size:0.9rem; color:#546e7a; font-weight:400; margin-left:10px;'>"
        "Estrategia Pre-Partido</span></h1>",
        unsafe_allow_html=True,
    )

    # ── Game selector ──────────────────────────────────────────────────────────
    with st.spinner("Cargando partidos del día…"):
        games = get_todays_games()

    game_map = {
        f"{g['away_name']}  @  {g['home_name']}   ·   {g['game_time']}  —  {g['venue']}": g
        for g in games
    }
    sel_label = st.selectbox(
        "Partido",
        list(game_map),
        key="wr_game_select",
        label_visibility="collapsed",
    )
    sel_game = game_map[sel_label]
    game_pk  = sel_game["game_pk"]

    if st.session_state["wr_game_pk"] != game_pk:
        st.session_state["wr_game_pk"]     = game_pk
        st.session_state["whatsif_result"] = None

    # Pitcher banner
    st.markdown(f"""
    <div style='background:linear-gradient(90deg,#0d1f3c,#111827); border:1px solid #1e3a5f;
                border-radius:10px; padding:10px 18px; color:#b0bec5; font-size:0.88rem;
                margin-bottom:4px;'>
        🏟️ <strong>{sel_game['venue']}</strong>
        &nbsp;|&nbsp; ⚔️ &nbsp;
        <strong>{sel_game['away_pitcher']}</strong> ({sel_game['away_pitcher_hand']}HP · Visitante)
        &nbsp;vs&nbsp;
        <strong>{sel_game['home_pitcher']}</strong> ({sel_game['home_pitcher_hand']}HP · Local)
        &nbsp;|&nbsp; 📅 {sel_game['game_time']}
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Lineup data ────────────────────────────────────────────────────────────
    with st.spinner("Generando lineup óptimo… (primera carga ~50s si Airflow aún no corrió)"):
        data = get_game_lineup(game_pk)

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        er = data["expected_runs"]
        st.metric(
            "E[R] Lineup Óptimo", f"{er:.2f}",
            delta=f"IC 95%: [{er-0.42:.2f}, {er+0.42:.2f}]",
            delta_color="off",
        )
    with k2:
        st.metric(
            "Probabilidad de Victoria", f"{data['win_probability']*100:.1f}%",
            delta=f"Confianza modelo: {data['model_confidence']*100:.0f}%",
            delta_color="off",
        )
    with k3:
        st.metric(
            "Simulaciones Monte Carlo", f"{data['total_simulations']:,}",
            delta=f"Modo: {data['optimization_mode']}",
            delta_color="off",
        )
    with k4:
        st.metric(
            "Versión del Modelo", data["model_version"],
            delta=f"t = {data['elapsed_seconds']:.1f}s",
            delta_color="off",
        )

    st.divider()

    # ── Lineup + Charts ────────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("### Lineup Óptimo Sugerido")
        _lineup_table(data["lineup"])

        with st.expander(
            "🤖 Análisis RAG — Por qué este lineup (LLM + Scouting)",
            expanded=False,
        ):
            st.markdown(data["rag_explanation"])

    with right:
        st.plotly_chart(
            win_probability_gauge(data["win_probability"]),
            width="stretch",
            config={"displayModeBar": False},
            key="wr_gauge",
        )
        st.plotly_chart(
            ops_lineup_chart(data["lineup"]),
            width="stretch",
            config={"displayModeBar": False},
            key="wr_ops",
        )

    st.divider()

    # ── What-If ────────────────────────────────────────────────────────────────
    _render_whatsif(game_pk, data)
