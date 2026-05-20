"""
views/post_game.py
==================
Vista Post-Game Review — análisis histórico.

Secciones:
  1. Selector de fecha + partido
  2. Comparativa lineup propuesto vs. real (divergencias resaltadas)
  3. E[R] proyectado vs. carreras reales
  4. Reporte de rendimiento completo (Markdown)
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from api.client import get_historical_games, get_post_game_report
from utils.charts import er_comparison_chart


# ── Lineup comparison ──────────────────────────────────────────────────────────
def _lineup_comparison(report: dict) -> None:
    proposed     = report["proposed_lineup"]
    actual       = report["actual_lineup"]
    prop_names   = [p["name"] for p in proposed]
    actual_names = [p["name"] for p in actual]
    n_diffs      = sum(1 for p, a in zip(prop_names, actual_names) if p != a)

    col_prop, col_act = st.columns(2, gap="large")

    with col_prop:
        st.markdown(
            "<h3 style='color:#90caf9; margin-bottom:6px;'>🤖 Lineup del Modelo</h3>",
            unsafe_allow_html=True,
        )
        rows = []
        for p in proposed:
            order = p["order"]
            match = order <= len(actual_names) and actual_names[order - 1] == p["name"]
            rows.append({"#": order, "Jugador": p["name"], "Pos": p["pos"],
                         "Estado": "✅ Coincide" if match else "🔀 Diverge"})
        df_p = pd.DataFrame(rows).set_index("#")

        def _sty_prop(row):
            return (
                ["background-color:#1a1030; color:#ce93d8"] * len(row)
                if "Diverge" in str(row["Estado"])
                else ["color:#a5d6a7"] * len(row)
            )
        st.dataframe(df_p.style.apply(_sty_prop, axis=1), use_container_width=True, height=360)

    with col_act:
        st.markdown(
            "<h3 style='color:#ffcc80; margin-bottom:6px;'>👨‍💼 Lineup Real (Manager)</h3>",
            unsafe_allow_html=True,
        )
        rows_a = []
        for p in actual:
            order = p["order"]
            match = order <= len(prop_names) and prop_names[order - 1] == p["name"]
            rows_a.append({"#": order, "Jugador": p["name"], "Pos": p["pos"],
                           "Resultado": p.get("result", "—"),
                           "Est.": "✅" if match else "🔀"})
        df_a = pd.DataFrame(rows_a).set_index("#")

        def _sty_act(row):
            return (
                ["background-color:#2a1008; color:#ffcc80"] * len(row)
                if row["Est."] == "🔀"
                else ["color:#e8eaf6"] * len(row)
            )
        st.dataframe(df_a.style.apply(_sty_act, axis=1), use_container_width=True, height=360)

    if n_diffs == 0:
        st.success("✅ El manager siguió exactamente el lineup propuesto por el modelo.")
    else:
        css = "alert-negative" if n_diffs >= 3 else "alert-neutral"
        st.markdown(
            f'<div class="{css}">🔀 <strong>{n_diffs} divergencia(s)</strong> entre el '
            f'lineup propuesto y el utilizado. Filas marcadas con 🔀 indican posiciones que difieren.</div>',
            unsafe_allow_html=True,
        )


# ── Main ───────────────────────────────────────────────────────────────────────
def render_post_game() -> None:
    st.markdown(
        "<h1>📊 Post-Game Review"
        "<span style='font-size:0.9rem; color:#546e7a; font-weight:400; margin-left:10px;'>"
        "Análisis Histórico</span></h1>",
        unsafe_allow_html=True,
    )

    # ── Filters ────────────────────────────────────────────────────────────────
    f1, f2 = st.columns([1, 2], gap="medium")
    with f1:
        sel_date = st.date_input(
            "Fecha del partido",
            value=date.today() - timedelta(days=1),
            max_value=date.today() - timedelta(days=1),
            min_value=date(2024, 1, 1),
            key="pg_date",
            format="DD/MM/YYYY",
        )

    with st.spinner("Buscando partidos…"):
        games = get_historical_games(sel_date)

    if not games:
        st.warning("No hay partidos registrados para esta fecha.", icon="⚠️")
        return

    with f2:
        game_opts = {
            f"{g['away_name']} @ {g['home_name']}  ·  Final: {g['final_score']}": g
            for g in games
        }
        sel_label = st.selectbox(
            "Partido",
            list(game_opts),
            key="pg_game",
            label_visibility="collapsed",
        )
        sel_game = game_opts[sel_label]

    st.divider()

    # ── Report ─────────────────────────────────────────────────────────────────
    with st.spinner("Cargando reporte post-partido…"):
        report = get_post_game_report(sel_game["game_pk"], sel_date)

    # ── Lineup comparison ──────────────────────────────────────────────────────
    st.markdown("### Lineup Propuesto vs. Lineup Utilizado")
    _lineup_comparison(report)

    st.divider()

    # ── E[R] chart + stats ─────────────────────────────────────────────────────
    st.markdown("### E[R] Proyectado vs. Carreras Reales")

    chart_c, stats_c = st.columns([3, 2], gap="large")

    with chart_c:
        st.plotly_chart(
            er_comparison_chart(
                projected=report["projected_runs"],
                actual=report["actual_home_runs"],
                matchup=report["matchup"],
            ),
            width="stretch",
            config={"displayModeBar": False},
            key="pg_er_chart",
        )

    with stats_c:
        delta_er = report["actual_home_runs"] - report["projected_runs"]
        is_win   = report["game_result"]

        st.markdown("#### Resumen")
        st.metric("E[R] Proyectado",     f"{report['projected_runs']:.2f}")
        st.metric(
            "Carreras Reales",
            str(report["actual_home_runs"]),
            delta=f"{'+' if delta_er >= 0 else ''}{delta_er:.2f} vs proyección",
            delta_color="normal",
        )
        st.metric(
            "Win Prob. Proyectada",
            f"{report['win_probability_projected']*100:.1f}%",
        )
        st.metric(
            "Log-Loss Partido",
            f"{report['model_log_loss']:.3f}",
            delta="bueno (< 0.5)" if report["model_log_loss"] < 0.5 else "revisar (≥ 0.5)",
            delta_color="inverse",
        )

        res_text = "✅ Victoria" if is_win else "❌ Derrota"
        res_css  = "alert-positive" if is_win else "alert-negative"
        st.markdown(
            f'<div class="{res_css}" style="text-align:center; font-size:1.1rem; '
            f'font-weight:600; margin-top:14px;">{res_text}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Full report ─────────────────────────────────────────────────────────────
    st.markdown("### 📋 Reporte de Rendimiento")
    st.markdown(
        f"<div style='background:#0d1528; border:1px solid #1e2d4a; border-radius:10px;"
        f"padding:20px 26px; line-height:1.75;'>{report['report_markdown']}</div>",
        unsafe_allow_html=True,
    )
