"""
api/client.py
=============
Capa de datos de la aplicación War Room.

Modo DEMO_MODE=1 (default): datos sintéticos de alta fidelidad.
Modo DEMO_MODE=0: llamadas reales a la FastAPI del backend.

Cada función documenta el endpoint real que corresponde.
"""
from __future__ import annotations

import os
import random
from datetime import date, timedelta

import streamlit as st

API_BASE:   str  = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEMO_MODE:  bool = os.environ.get("DEMO_MODE", "1") == "1"
_API_TOKEN: str  = os.environ.get("MLB_API_TOKEN", "")


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {_API_TOKEN}"} if _API_TOKEN else {}

# ---------------------------------------------------------------------------
# Roster (mock) — en producción viene de GET /v1/roster/{team_abbr}
# ---------------------------------------------------------------------------

_ROSTER_NYY: list[dict] = [
    {"player_id": 592450, "name": "Aaron Judge",       "pos": "CF", "hand": "R", "avg": .322, "ops": 1.059, "woba": .431, "obp": .404, "iso": .335},
    {"player_id": 660271, "name": "Juan Soto",         "pos": "RF", "hand": "L", "avg": .288, "ops": .972,  "woba": .392, "obp": .415, "iso": .241},
    {"player_id": 608369, "name": "Giancarlo Stanton", "pos": "DH", "hand": "R", "avg": .244, "ops": .833,  "woba": .338, "obp": .312, "iso": .310},
    {"player_id": 543760, "name": "Anthony Rizzo",     "pos": "1B", "hand": "L", "avg": .224, "ops": .739,  "woba": .302, "obp": .332, "iso": .185},
    {"player_id": 605141, "name": "Gleyber Torres",    "pos": "2B", "hand": "R", "avg": .257, "ops": .764,  "woba": .318, "obp": .331, "iso": .162},
    {"player_id": 657557, "name": "Jazz Chisholm Jr.", "pos": "3B", "hand": "L", "avg": .241, "ops": .771,  "woba": .315, "obp": .315, "iso": .209},
    {"player_id": 519203, "name": "Austin Wells",      "pos": "C",  "hand": "L", "avg": .228, "ops": .704,  "woba": .291, "obp": .307, "iso": .155},
    {"player_id": 650402, "name": "Oswaldo Cabrera",   "pos": "LF", "hand": "S", "avg": .212, "ops": .623,  "woba": .265, "obp": .275, "iso": .120},
    {"player_id": 668804, "name": "Anthony Volpe",     "pos": "SS", "hand": "R", "avg": .233, "ops": .698,  "woba": .288, "obp": .305, "iso": .148},
    # Bench
    {"player_id": 666176, "name": "Ben Rice",          "pos": "1B", "hand": "L", "avg": .245, "ops": .751,  "woba": .308, "obp": .338, "iso": .193},
    {"player_id": 683905, "name": "Jasson Dominguez",  "pos": "CF", "hand": "S", "avg": .261, "ops": .792,  "woba": .322, "obp": .341, "iso": .219},
    {"player_id": 677951, "name": "Trent Grisham",     "pos": "CF", "hand": "L", "avg": .201, "ops": .651,  "woba": .270, "obp": .301, "iso": .132},
]

_RAG_NYY_BOS = """
## Análisis Táctico — NYY vs BOS · Brayan Bello (RHP)

**Perfil del lanzador rival:** ERA 3.87 · WHIP 1.21 · K/9 9.4
Bello predomina con fastball de 4 costuras (93-95 mph, zona alta) y slider de alto giro en zona
baja-exterior. **Bateadores zurdos** tienen AVG .218 contra él; diestros, .271.

---

### Por qué este orden de bateo

**#1 — Juan Soto (LHB):** xwOBA .431 vs fastball alto de zona cintura. Disciplina excepcional
(BB% 17.4%) — fuerza a Bello a pitcher cuidadosamente desde el primer inning. El modelo asigna
**89% de prob.** de que Soto llegue a base en el primer inning.

**#2 — Aaron Judge (RHB):** Con Soto como precedente, Judge ve más fastballs en la zona.
Su xwOBA contra fastball 93-95 mph es .485. Monte Carlo sitúa este slot como el de mayor
producción de E[R] dado el rate de on-base de Soto.

**#3 — Stanton (RHB):** HR rate de 1/9.2 AB contra slider dominante. El modelo proyecta
23% de probabilidad de extra-base con hombre en base.

**#4 — Rizzo → Considerar Ben Rice:** Factor platoon zurdo-zurdo penaliza a Rizzo en −.048 wOBA.
> ⚠️ **Alerta platoon:** Ben Rice (OPS .751 vs RHP) puede ser superior si Bello pitchea ≥6 innings.

**#5–9:** El modelo optimiza para maximizar bateadores con OBP >.330 en slots 1-5.
Torres (.331), Chisholm (.315) y Wells (.307) ofrecen continuidad de runners en base.

---

### Señales del Scouting *(RAG — Baseball Savant + Notas del Scout)*
> ✅ **Torres** tiene AVG .341 contra Bello en los últimos 18 meses (zone contact %).
> ⚠️ **Volpe** debe trabajar el count: Bello usa slider zona baja con 2 strikes y runners en base.
> 📊 **Park factor Yankee Stadium (HR):** 1.08 — favorece a Stanton (#3) y Judge (#2).
"""

_RAG_LAD_ATL = """
## Análisis Táctico — LAD vs ATL · Spencer Strider (RHP)

**Perfil del lanzador rival:** ERA 2.61 · K/9 14.1 · Slider dominante (54% uso)
Strider es uno de los lanzadores más difíciles de la NL. Fastball promedio 97.3 mph. Su slider
zona exterior a diestros tiene SwStr% de 42%.

---

### Decisión táctica del modelo

El modelo prioriza bateadores **zurdos** en los primeros 5 slots:
LHB tienen K% de 22% vs. Strider frente al 31% de RHB.

**Ohtani (#1, LHB):** xwOBA .408 contra fastball Strider (44% barrel rate en zona media).
**Freeman (#2, LHB):** Slot #2 para maximizar RBI con Ohtani como leadoff.

> 📊 E[R] proyectado = 3.92 — por debajo del promedio de temporada (4.6) reflejando la
> dificultad de Strider. Win probability al 53.2%.
"""


# ---------------------------------------------------------------------------
# Partidos del día
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_todays_games() -> list[dict]:
    """
    Partidos programados para hoy.

    REAL API CALL:
        import httpx
        r = httpx.get(f"{API_BASE}/v1/games/today", timeout=10)
        r.raise_for_status()
        return r.json()["games"]
    """
    if not DEMO_MODE:
        try:
            import httpx
            r = httpx.get(f"{API_BASE}/v1/games/today", headers=_auth_headers(), timeout=10)
            r.raise_for_status()
            return r.json()["games"]
        except Exception as exc:
            st.toast(f"API no disponible — modo demo ({exc})", icon="⚠️")

    today = date.today().isoformat()
    return [
        {
            "game_pk": 745467,
            "home_team": "NYY", "away_team": "BOS",
            "home_name": "New York Yankees", "away_name": "Boston Red Sox",
            "game_time": "19:05 ET", "venue": "Yankee Stadium", "game_date": today,
            "home_pitcher": "Gerrit Cole",  "home_pitcher_hand": "R",
            "away_pitcher": "Brayan Bello", "away_pitcher_hand": "R",
        },
        {
            "game_pk": 745490,
            "home_team": "LAD", "away_team": "ATL",
            "home_name": "Los Angeles Dodgers", "away_name": "Atlanta Braves",
            "game_time": "22:10 ET", "venue": "Dodger Stadium", "game_date": today,
            "home_pitcher": "Yoshinobu Yamamoto", "home_pitcher_hand": "R",
            "away_pitcher": "Spencer Strider",    "away_pitcher_hand": "R",
        },
        {
            "game_pk": 745512,
            "home_team": "HOU", "away_team": "SD",
            "home_name": "Houston Astros", "away_name": "San Diego Padres",
            "game_time": "20:10 ET", "venue": "Minute Maid Park", "game_date": today,
            "home_pitcher": "Framber Valdez", "home_pitcher_hand": "L",
            "away_pitcher": "Dylan Cease",    "away_pitcher_hand": "R",
        },
    ]


# ---------------------------------------------------------------------------
# Lineup óptimo
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120, show_spinner=False)
def get_game_lineup(game_pk: int) -> dict:
    """
    Lineup óptimo + métricas + explicación RAG.

    REAL API CALL:
        import httpx
        r = httpx.get(f"{API_BASE}/v1/optimize/{game_pk}", timeout=30)
        r.raise_for_status()
        return r.json()
    """
    if not DEMO_MODE:
        try:
            import httpx
            r = httpx.get(f"{API_BASE}/v1/optimize/{game_pk}", headers=_auth_headers(), timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            st.toast(f"API no disponible — modo demo ({exc})", icon="⚠️")

    def _slot(order: int, idx: int) -> dict:
        return {"order": order, **_ROSTER_NYY[idx]}

    mock: dict[int, dict] = {
        745467: {
            "game_pk": 745467,
            "expected_runs": 5.23,
            "win_probability": 0.617,
            "model_confidence": 0.84,
            "optimization_mode": "OPS+",
            "model_version": "v2.1.0",
            "total_simulations": 10_000,
            "elapsed_seconds": 22.4,
            "lineup": [
                _slot(1, 1), _slot(2, 0), _slot(3, 2), _slot(4, 3), _slot(5, 4),
                _slot(6, 5), _slot(7, 6), _slot(8, 7), _slot(9, 8),
            ],
            "bench": [_ROSTER_NYY[9], _ROSTER_NYY[10], _ROSTER_NYY[11]],
            "rag_explanation": _RAG_NYY_BOS,
        },
        745490: {
            "game_pk": 745490,
            "expected_runs": 3.92,
            "win_probability": 0.532,
            "model_confidence": 0.79,
            "optimization_mode": "OPS+",
            "model_version": "v2.1.0",
            "total_simulations": 10_000,
            "elapsed_seconds": 19.1,
            "lineup": [
                {"order": 1, "player_id": 660271, "name": "Shohei Ohtani",    "pos": "DH", "hand": "L", "avg": .310, "ops": .999, "woba": .421, "obp": .388, "iso": .318},
                {"order": 2, "player_id": 592450, "name": "Freddie Freeman",  "pos": "1B", "hand": "L", "avg": .298, "ops": .919, "woba": .382, "obp": .381, "iso": .244},
                {"order": 3, "player_id": 608369, "name": "Mookie Betts",     "pos": "RF", "hand": "R", "avg": .287, "ops": .892, "woba": .368, "obp": .362, "iso": .241},
                {"order": 4, "player_id": 543760, "name": "Teoscar Hernandez","pos": "LF", "hand": "R", "avg": .270, "ops": .829, "woba": .341, "obp": .321, "iso": .220},
                {"order": 5, "player_id": 605141, "name": "Will Smith",       "pos": "C",  "hand": "R", "avg": .261, "ops": .809, "woba": .331, "obp": .340, "iso": .198},
                {"order": 6, "player_id": 657557, "name": "Max Muncy",        "pos": "3B", "hand": "L", "avg": .238, "ops": .791, "woba": .322, "obp": .358, "iso": .213},
                {"order": 7, "player_id": 519203, "name": "Gavin Lux",        "pos": "2B", "hand": "L", "avg": .251, "ops": .751, "woba": .308, "obp": .331, "iso": .158},
                {"order": 8, "player_id": 650402, "name": "Miguel Rojas",     "pos": "SS", "hand": "R", "avg": .240, "ops": .652, "woba": .272, "obp": .289, "iso": .115},
                {"order": 9, "player_id": 668804, "name": "James Outman",     "pos": "CF", "hand": "L", "avg": .224, "ops": .711, "woba": .291, "obp": .311, "iso": .168},
            ],
            "bench": [
                {"player_id": 99901, "name": "Chris Taylor",     "pos": "UT", "hand": "R", "avg": .225, "ops": .681, "woba": .281, "obp": .311, "iso": .143},
                {"player_id": 99902, "name": "Enrique Hernandez","pos": "2B", "hand": "R", "avg": .219, "ops": .664, "woba": .272, "obp": .299, "iso": .128},
            ],
            "rag_explanation": _RAG_LAD_ATL,
        },
    }
    return mock.get(game_pk, mock[745467])


# ---------------------------------------------------------------------------
# Simulación What-If
# ---------------------------------------------------------------------------

def simulate_whatsif(game_pk: int, slot_changes: dict[int, dict]) -> dict:
    """
    Impacto de cambios en el lineup via Monte Carlo.

    REAL API CALL:
        import httpx
        r = httpx.post(
            f"{API_BASE}/v1/optimize/{game_pk}/whatsif",
            json={"changes": {str(k): v for k, v in slot_changes.items()}},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    `slot_changes`: {batting_order_1_9: player_dict_with_woba}
    """
    if not DEMO_MODE:
        try:
            import httpx
            r = httpx.post(
                f"{API_BASE}/v1/optimize/{game_pk}/whatsif",
                headers=_auth_headers(),
                json={"changes": {str(k): v for k, v in slot_changes.items()}},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            st.toast(f"API no disponible — simulación aproximada ({exc})", icon="⚠️")

    base    = get_game_lineup(game_pk)
    base_er = base["expected_runs"]
    base_wp = base["win_probability"]

    # Heurística: ΔwOBA * 4.8 ≈ ΔE[R]  (aprox. lineal; la API usa Monte Carlo real)
    delta_woba = sum(
        new_p["woba"] - base["lineup"][slot - 1]["woba"]
        for slot, new_p in slot_changes.items()
        if 1 <= slot <= len(base["lineup"])
    )
    delta_er = round(delta_woba * 4.8, 3)
    delta_wp = round(delta_er * 0.038, 4)

    new_er = round(base_er + delta_er, 3)
    new_wp = max(0.01, min(0.99, round(base_wp + delta_wp, 4)))

    return {
        "base_expected_runs":    base_er,
        "new_expected_runs":     new_er,
        "delta_er":              delta_er,
        "base_win_probability":  base_wp,
        "new_win_probability":   new_wp,
        "delta_win_probability": round(new_wp - base_wp, 4),
        "simulation_n":          10_000,
        "confidence_interval":   [round(new_er - 0.42, 2), round(new_er + 0.42, 2)],
    }


# ---------------------------------------------------------------------------
# Partidos históricos
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_historical_games(game_date: date) -> list[dict]:
    """
    Partidos jugados en una fecha pasada.

    REAL API CALL:
        import httpx
        r = httpx.get(
            f"{API_BASE}/v1/games/history",
            params={"date": game_date.isoformat()},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["games"]
    """
    if not DEMO_MODE:
        try:
            import httpx
            r = httpx.get(
                f"{API_BASE}/v1/games/history",
                headers=_auth_headers(),
                params={"date": game_date.isoformat()},
                timeout=10,
            )
            r.raise_for_status()
            return r.json()["games"]
        except Exception as exc:
            st.toast(f"API no disponible — modo demo ({exc})", icon="⚠️")

    rng = random.Random(game_date.toordinal())
    hr1, ar1 = rng.randint(3, 8), rng.randint(1, 5)
    hr2, ar2 = rng.randint(4, 9), rng.randint(0, 4)

    return [
        {
            "game_pk": 745400,
            "home_team": "NYY", "away_team": "TOR",
            "home_name": "New York Yankees", "away_name": "Toronto Blue Jays",
            "game_date": game_date.isoformat(),
            "final_score": f"{hr1}-{ar1}",
            "home_runs_actual": hr1, "away_runs_actual": ar1,
            "home_pitcher": "Carlos Rodón",  "away_pitcher": "Kevin Gausman",
        },
        {
            "game_pk": 745401,
            "home_team": "LAD", "away_team": "SFG",
            "home_name": "Los Angeles Dodgers", "away_name": "San Francisco Giants",
            "game_date": game_date.isoformat(),
            "final_score": f"{hr2}-{ar2}",
            "home_runs_actual": hr2, "away_runs_actual": ar2,
            "home_pitcher": "Tyler Glasnow", "away_pitcher": "Logan Webb",
        },
    ]


# ---------------------------------------------------------------------------
# Reporte post-partido
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_post_game_report(game_pk: int, game_date: date | None = None) -> dict:
    """
    Reporte completo: lineups, métricas y análisis Markdown.

    REAL API CALL:
        import httpx
        params = {"date": game_date.isoformat()} if game_date else {}
        r = httpx.get(f"{API_BASE}/v1/report/{game_pk}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    """
    if not DEMO_MODE:
        try:
            import httpx
            params = {"date": game_date.isoformat()} if game_date else {}
            r = httpx.get(
                f"{API_BASE}/v1/report/{game_pk}",
                headers=_auth_headers(),
                params=params,
                timeout=15,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            st.toast(f"API no disponible — modo demo ({exc})", icon="⚠️")

    gd = (game_date or date.today() - timedelta(days=1)).isoformat()

    return {
        "game_pk":    game_pk,
        "game_date":  gd,
        "matchup":    "NYY 5  ·  TOR 3",
        "game_result": True,
        "proposed_lineup": [
            {"order": 1, "name": "Juan Soto",        "pos": "RF"},
            {"order": 2, "name": "Aaron Judge",       "pos": "CF"},
            {"order": 3, "name": "Giancarlo Stanton", "pos": "DH"},
            {"order": 4, "name": "Anthony Rizzo",     "pos": "1B"},
            {"order": 5, "name": "Gleyber Torres",    "pos": "2B"},
            {"order": 6, "name": "Jazz Chisholm Jr.", "pos": "3B"},
            {"order": 7, "name": "Austin Wells",      "pos": "C"},
            {"order": 8, "name": "Oswaldo Cabrera",   "pos": "LF"},
            {"order": 9, "name": "Anthony Volpe",     "pos": "SS"},
        ],
        "actual_lineup": [
            {"order": 1, "name": "Aaron Judge",       "pos": "CF", "result": "2-for-4, HR"},
            {"order": 2, "name": "Juan Soto",         "pos": "RF", "result": "1-for-3, 2 BB"},
            {"order": 3, "name": "Giancarlo Stanton", "pos": "DH", "result": "1-for-4, RBI"},
            {"order": 4, "name": "Anthony Rizzo",     "pos": "1B", "result": "0-for-4"},
            {"order": 5, "name": "Gleyber Torres",    "pos": "2B", "result": "1-for-3, BB"},
            {"order": 6, "name": "Jazz Chisholm Jr.", "pos": "3B", "result": "0-for-4, 2K"},
            {"order": 7, "name": "Austin Wells",      "pos": "C",  "result": "1-for-3"},
            {"order": 8, "name": "Anthony Volpe",     "pos": "SS", "result": "2-for-4"},
            {"order": 9, "name": "Oswaldo Cabrera",   "pos": "LF", "result": "0-for-3"},
        ],
        "projected_runs":            4.87,
        "actual_home_runs":          5,
        "actual_away_runs":          3,
        "win_probability_projected": 0.59,
        "model_log_loss":            0.412,
        "model_version":             "v2.1.0",
        "report_markdown": f"""
## Post-Game Analysis — NYY 5, TOR 3
**Fecha:** {gd} &nbsp;|&nbsp; **Modelo:** v2.1.0 &nbsp;|&nbsp; **Log-Loss:** 0.412

---

### Contraste de Decisiones

El manager optó por invertir los slots #1 y #2 (Judge delante de Soto), divergiendo del modelo
en **−0.08 E[R]** proyectado. El resultado final de **5 carreras reales** superó la proyección de
4.87, validando el ajuste táctico en este contexto.

**Divergencias detectadas (4 de 9 posiciones):**

| Pos Modelo | Jugador (Modelo) | Pos Real | Jugador (Real) | ΔImpacto |
|:---:|---|:---:|---|:---:|
| #1 | Juan Soto | #2 | Juan Soto | Neutro |
| #2 | Aaron Judge | #1 | Aaron Judge | +0.03 E[R] |
| #8 | Oswaldo Cabrera | #9 | Oswaldo Cabrera | Neutro |
| #9 | Anthony Volpe | #8 | Anthony Volpe | +0.01 E[R] |

### Evaluación del Modelo

| Métrica | Valor | Benchmark |
|---|---|---|
| E[R] proyectado | 4.87 | — |
| Carreras reales | **5** | **+0.13 vs proyección ✅** |
| Win Prob. proyectada | 59% | ✅ Acertó dirección |
| Log-Loss del partido | 0.412 | < 0.5 = bueno |
| Calibración acumulada (14d) | 0.389 | En rango aceptable |

### Rendimiento Individual

- **Aaron Judge (#1 real):** 2-for-4, HR en el 4to inning. El modelo infraestimó su HR rate vs fastball elevado en contexto nocturno.
- **Juan Soto (#2 real):** 1-for-3, 2 BB — cumplió el rol de generador de base.
- **Gleyber Torres (#5):** 1-for-3, BB — exactamente en línea con la proyección de OBP.
- **Jazz Chisholm (#6):** 0-for-4, 2K — el scouting previo marcaba debilidad ante curveball zona baja. Confirmado.

### Recomendaciones para Próximo Partido

> El modelo sugiere revisar el coeficiente de ajuste nocturno para Judge (+0.015 xwOBA sugerido).
> Revisar peso del factor platoon para Rizzo en series contra bullpens de mayoría zurda.
""",
    }
