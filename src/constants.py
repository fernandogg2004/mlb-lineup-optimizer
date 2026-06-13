"""
src/constants.py — Fuente única de verdad para constantes compartidas.

Todas las constantes de dominio (promedios de liga, run values, FIP, park
factors) deben vivir aquí y ser importadas desde el resto del codebase.
Centralizar evita skew entre training (build_gold_v3.py) y serving
(predict_tonight.py, api/main.py).

Actualizar comentarios de fuente cuando se recalibren los valores.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Estabilización James-Stein / Marcel
# ---------------------------------------------------------------------------
# PA necesarias para que la estadística tenga 50% de señal vs ruido.
# Fuente: Lichtman 2010 "The Book" + calibración empírica MLB 2015-2024.
# Incluye alias cortos ("k", "bb") y largos ("k_rate", "bb_rate") para
# compatibilidad entre build_gold_v3.py y predict_tonight.py.
STAB_T: dict[str, int] = {
    "woba":   200,
    "k":       60,  "k_rate":   60,   # alias
    "bb":     120,  "bb_rate": 120,   # alias
    "babip":  500,
    "iso":    160,
}

# ---------------------------------------------------------------------------
# Promedios de liga (prior para shrinkage)
# ---------------------------------------------------------------------------
# Fuente: FanGraphs 2023-2025, promedio ponderado temporadas completas.
LEAGUE_AVG: dict[str, float] = {
    "woba":    0.318,
    "k_rate":  0.229,
    "bb_rate": 0.087,
    "babip":   0.295,
    "iso":     0.166,
}

# ---------------------------------------------------------------------------
# Distribución PA de liga — 8 clases (para oponente sin datos)
# ---------------------------------------------------------------------------
# Orden: OUT_IN_PLAY, STRIKEOUT, WALK_HBP, SINGLE, DOUBLE, TRIPLE, HR, DP
# Fuente: MLB Statcast 2023-2025, normalizado.
LEAGUE_AVG_PA_PROBS: np.ndarray = np.array(
    [0.451, 0.223, 0.087, 0.150, 0.048, 0.008, 0.032, 0.010],
    dtype=np.float32,
)
LEAGUE_AVG_PA_PROBS = LEAGUE_AVG_PA_PROBS / LEAGUE_AVG_PA_PROBS.sum()

# Matriz 9×8: mismo vector replicado para un lineup completo de 9 bateadores.
LEAGUE_AVG_LINEUP: np.ndarray = np.tile(LEAGUE_AVG_PA_PROBS, (9, 1))

# ---------------------------------------------------------------------------
# Run values (Linear Weights)
# ---------------------------------------------------------------------------
# Orden idéntico a PAOutcome enum en model_at_bat.py:
#   0=OUT_IN_PLAY  1=STRIKEOUT  2=WALK_HBP  3=SINGLE  4=DOUBLE
#   5=TRIPLE       6=HOME_RUN   7=DOUBLE_PLAY
# Fuente: FanGraphs 2024 linear weights (índices 0-6) +
#         Retrosheet 2015-2024 GIDP penalty (índice 7, ≈ -0.43 wOBA).
RUN_VALUES: np.ndarray = np.array(
    [0.00, 0.00, 0.33, 0.47, 0.77, 1.04, 1.40, -0.43], dtype=np.float32
)

# ---------------------------------------------------------------------------
# FIP (Fielding Independent Pitching)
# ---------------------------------------------------------------------------
# Fuente: Tangotiger FIP formula; constante calibrada 2015-2024 MLB data.
FIP_CONSTANT: float = 3.10   # cFIP = league_ERA - raw_FIP_numerator/IP_total
FIP_NEUTRAL: float  = 4.20   # fallback cuando PA insuficientes (liga avg)
FIP_FLOOR: float    = 1.50   # límite inferior razonable
FIP_CEIL: float     = 9.00   # límite superior razonable
FIP_MIN_PA: int     = 50     # PA mínimas para usar datos reales (shrinkage gradual)

# ---------------------------------------------------------------------------
# Park factors por venue_id MLB
# ---------------------------------------------------------------------------
# Fuente: FanGraphs Park Factors 2025. Solo venues con desviación > 3% del neutro.
# El resto de estadios no listados aquí usan hr=1.0, xb=1.0 (neutral).
# Actualizar anualmente antes del Opening Day.
#
# venue_id → {"hr": float, "xb": float}
#   hr  = factor multiplicador de probabilidad de HR (>1 = favorable a HR)
#   xb  = factor multiplicador de extra bases (2B, 3B)
PARK_FACTORS: dict[str, dict[str, float]] = {
    # Estadios que favorecen la ofensiva
    "3313": {"hr": 1.12, "xb": 1.08},   # Coors Field (COL) — altitud 1609m
    "32": {"hr": 1.09, "xb": 1.03},     # Great American Ball Park (CIN)
    "4705": {"hr": 1.06, "xb": 1.02},   # Globe Life Field (TEX)
    "5325": {"hr": 1.05, "xb": 1.02},   # Yankee Stadium (NYY)
    "2392": {"hr": 1.04, "xb": 1.01},   # Wrigley Field (CHC)
    # Estadios que perjudican la ofensiva
    "4169": {"hr": 0.88, "xb": 0.94},   # Petco Park (SD) — grande, cerca de océano
    "14": {"hr": 0.91, "xb": 0.96},     # Dodger Stadium (LAD)
    "680": {"hr": 0.92, "xb": 0.97},    # Oracle Park (SF) — famoso por suprimir HRs
    "395": {"hr": 0.93, "xb": 0.97},    # Tropicana Field (TB) — techo
    "31": {"hr": 0.94, "xb": 0.98},     # Kauffman Stadium (KC)
}

# Park factors indexados por abreviatura Statcast del HOME team (la columna
# home_team del Silver identifica el estadio). Para training histórico: el
# equipo juega en su estadio habitual; sedes especiales (London Series,
# Field of Dreams, etc.) son <0.5% de los juegos y quedan en su factor base.
# Equipos no listados → neutral (hr=1.0, xb=1.0).
PARK_FACTORS_BY_TEAM: dict[str, dict[str, float]] = {
    "COL": PARK_FACTORS["3313"],
    "CIN": PARK_FACTORS["32"],
    "TEX": PARK_FACTORS["4705"],
    "NYY": PARK_FACTORS["5325"],
    "CHC": PARK_FACTORS["2392"],
    "SD":  PARK_FACTORS["4169"],
    "LAD": PARK_FACTORS["14"],
    "SF":  PARK_FACTORS["680"],
    "TB":  PARK_FACTORS["395"],
    "KC":  PARK_FACTORS["31"],
}
