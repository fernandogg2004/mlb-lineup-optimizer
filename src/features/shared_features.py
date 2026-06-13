"""
shared_features.py
==================
Definición ÚNICA de las features de bateador y pitcher, usada por:

  - Training : scripts/build_gold_v3.py  (batch, todas las filas PA)
  - Serving  : predict_tonight.py / api  (snapshot de un jugador "as of" hoy)

Contrato anti-skew
------------------
El serving NO reimplementa las features: construye el grano diario del jugador,
añade una fila sintética con la fecha del partido (sin estadísticas) y ejecuta
las MISMAS funciones batch. El ``shift(1)`` interno garantiza que la fila
sintética solo ve los juegos completados — exactamente lo que ve cada fila de
training. Paridad por construcción, no por convención.

Semántica de ventanas
---------------------
Los sufijos ``7d/15d/30d`` son ventanas de CONTEO DE JUEGOS (últimos N juegos
con actividad), no días calendario. Se mantienen los nombres por compatibilidad
con el modelo ya desplegado.

Política de nulos
-----------------
Debajo del gate de PA mínimas las features rolling son ``null`` (→ NaN en
numpy). LightGBM las rutea por la rama default del split. PROHIBIDO imputar
0.0 en serving: un ``k_rate_30d=0.0`` significa "élite que nunca se poncha",
no "sin datos".
"""
from __future__ import annotations

import polars as pl

from src.constants import (
    FIP_CEIL,
    FIP_CONSTANT,
    FIP_FLOOR,
    FIP_MIN_PA,
    FIP_NEUTRAL,
    LEAGUE_AVG,
    STAB_T,
)

# ---------------------------------------------------------------------------
# Constantes de dominio
# ---------------------------------------------------------------------------
HARD_HIT_THRESHOLD_MPH: float = 95.0

# (window_games, min_pa, suffix)
ROLLING_WINDOWS: list[tuple[int, int, str]] = [(7, 3, "7d"), (15, 3, "15d"), (30, 10, "30d")]

GIDP_EVENTS = {"grounded_into_double_play", "strikeout_double_play", "double_play"}

_K_EVENTS   = ["strikeout", "strikeout_double_play"]
_BB_EVENTS  = ["walk", "intent_walk"]
_HIT_EVENTS = ["single", "double", "triple", "home_run"]
_NON_BIP    = ["walk", "intent_walk", "hit_by_pitch", "strikeout",
               "strikeout_double_play", "sac_fly", "sac_bunt"]

# Catálogo de nombres producidos (contrato con el modelo)
ROLLING_FEATURE_COLS: list[str] = [
    f"{stat}_{sfx}"
    for _, _, sfx in ROLLING_WINDOWS
    for stat in ("pa", "k_rate", "bb_rate", "hr_rate", "hard_hit_rate",
                 "xwoba", "launch_speed")
] + ["xwoba_ewma_alpha02", "xwoba_ewma_alpha05"]

STAB_FEATURE_COLS: list[str] = [
    f"{stat}_{kind}"
    for stat in ("woba", "k_rate", "bb_rate", "babip", "iso")
    for kind in ("stabilized", "shrinkage_b")
]

PLATOON_FEATURE_COLS: list[str] = [
    "woba_vs_hand_stabilized", "k_rate_vs_hand_stabilized",
    "bb_rate_vs_hand_stabilized", "babip_vs_hand_stabilized",
    "iso_vs_hand_stabilized", "pa_vs_hand",
]

PITCHER_FEATURE_COLS: list[str] = [
    "pitcher_fip", "pitcher_k_rate", "pitcher_bb_rate", "pitcher_hr_rate",
]


# ---------------------------------------------------------------------------
# Flags de evento y grano diario
# ---------------------------------------------------------------------------

def add_event_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Añade flags binarios por PA (Int8) + total bases para ISO."""
    return df.with_columns([
        pl.lit(1).cast(pl.Int8).alias("pa_flag"),
        pl.col("pa_result").is_in(_K_EVENTS).cast(pl.Int8).alias("k_flag"),
        pl.col("pa_result").is_in(_BB_EVENTS).cast(pl.Int8).alias("bb_flag"),
        (pl.col("hit_type") == "home_run").cast(pl.Int8).alias("hr_flag"),
        pl.col("hit_type").is_in(_HIT_EVENTS).cast(pl.Int8).alias("hit_flag"),
        (pl.col("launch_speed").is_not_null()
         & (pl.col("launch_speed") >= HARD_HIT_THRESHOLD_MPH))
        .cast(pl.Int8).alias("hh_flag"),
        (~pl.col("pa_result").is_in(_NON_BIP)).cast(pl.Int8).alias("bip_flag"),
        # Total bases (para ISO ≈ (TB - H) / PA, consistente train/serve)
        (
            (pl.col("pa_result") == "single").cast(pl.Int8) * 1
            + (pl.col("pa_result") == "double").cast(pl.Int8) * 2
            + (pl.col("pa_result") == "triple").cast(pl.Int8) * 3
            + (pl.col("pa_result") == "home_run").cast(pl.Int8) * 4
        ).cast(pl.Int16).alias("tb"),
    ])


def daily_grain(flagged: pl.DataFrame) -> pl.DataFrame:
    """Agrega PAs al grano (batter_id, game_date, season).

    Nota: NO agrupa por batter_stand — los switch hitters tendrían dos filas
    por fecha y el join de vuelta a PA-level duplicaría filas.
    """
    return (
        flagged.group_by(["batter_id", "game_date", "season"])
        .agg([
            pl.col("pa_flag").sum().alias("pa"),
            pl.col("k_flag").sum().alias("k"),
            pl.col("bb_flag").sum().alias("bb"),
            pl.col("hr_flag").sum().alias("hr"),
            pl.col("hit_flag").sum().alias("hits"),
            pl.col("hh_flag").sum().alias("hh"),
            pl.col("bip_flag").sum().alias("bip"),
            pl.col("tb").sum().alias("tb"),
            pl.col("xwoba").drop_nulls().mean().alias("xwoba_mean"),
            pl.col("launch_speed").drop_nulls().mean().alias("ls_mean"),
        ])
        .sort(["batter_id", "game_date"])
    )


# ---------------------------------------------------------------------------
# Rolling (conteo de juegos) con shift(1) anti-leakage
# ---------------------------------------------------------------------------

def add_shift_rolling(daily: pl.DataFrame) -> pl.DataFrame:
    """Añade las ventanas rolling 7d/15d/30d + EWMA xwOBA, con shift(1)."""
    cols_to_shift = ["xwoba_mean", "ls_mean", "pa", "k", "bb", "hr", "hits", "hh", "bip"]
    daily = daily.sort(["batter_id", "game_date"]).with_columns([
        pl.col(c).shift(1).over("batter_id").alias(f"_{c}_s") for c in cols_to_shift
    ])

    for window, min_pa, sfx in ROLLING_WINDOWS:
        daily = daily.with_columns([
            pl.col("_pa_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"pa_{sfx}"),
            pl.col("_k_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"_k_{sfx}"),
            pl.col("_bb_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"_bb_{sfx}"),
            pl.col("_hr_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"_hr_{sfx}"),
            pl.col("_hh_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"_hh_{sfx}"),
            pl.col("_bip_s").rolling_sum(window, min_periods=1).over("batter_id").alias(f"_bip_{sfx}"),
            pl.col("_xwoba_mean_s").rolling_mean(window, min_periods=1).over("batter_id").alias(f"_xw_{sfx}"),
            pl.col("_ls_mean_s").rolling_mean(window, min_periods=1).over("batter_id").alias(f"_ls_{sfx}"),
        ])
        gate = pl.col(f"pa_{sfx}") >= max(1, min_pa)
        daily = daily.with_columns([
            pl.when(gate).then(pl.col(f"_xw_{sfx}")).otherwise(None).alias(f"xwoba_{sfx}"),
            pl.when(gate).then(pl.col(f"_ls_{sfx}")).otherwise(None).alias(f"launch_speed_{sfx}"),
            pl.when(gate).then(pl.col(f"_k_{sfx}") / pl.col(f"pa_{sfx}").clip(lower_bound=1))
              .otherwise(None).alias(f"k_rate_{sfx}"),
            pl.when(gate).then(pl.col(f"_bb_{sfx}") / pl.col(f"pa_{sfx}").clip(lower_bound=1))
              .otherwise(None).alias(f"bb_rate_{sfx}"),
            pl.when(gate).then(pl.col(f"_hr_{sfx}") / pl.col(f"pa_{sfx}").clip(lower_bound=1))
              .otherwise(None).alias(f"hr_rate_{sfx}"),
            pl.when(gate).then(pl.col(f"_hh_{sfx}") / pl.col(f"_bip_{sfx}").clip(lower_bound=1))
              .otherwise(None).alias(f"hard_hit_rate_{sfx}"),
        ]).drop([f"_k_{sfx}", f"_bb_{sfx}", f"_hr_{sfx}", f"_hh_{sfx}",
                 f"_bip_{sfx}", f"_xw_{sfx}", f"_ls_{sfx}"])

    # EWMA xwOBA (lenta α=0.2, rápida α=0.5) sobre la serie diaria shifteada.
    # ignore_nulls=True explícito: el default de Polars cambiará y alteraría
    # silenciosamente la semántica de la feature.
    daily = daily.with_columns([
        pl.col("xwoba_mean").shift(1)
          .ewm_mean(alpha=0.2, min_periods=1, adjust=True, ignore_nulls=True)
          .over("batter_id").alias("xwoba_ewma_alpha02"),
        pl.col("xwoba_mean").shift(1)
          .ewm_mean(alpha=0.5, min_periods=1, adjust=True, ignore_nulls=True)
          .over("batter_id").alias("xwoba_ewma_alpha05"),
    ])
    return daily.drop([f"_{c}_s" for c in cols_to_shift])


# ---------------------------------------------------------------------------
# Stats season-to-date estabilizadas (James-Stein / Marcel)
# ---------------------------------------------------------------------------

def _stabilize_exprs(raw_col: str, pa_col: str, T: float, prior: float, out: str) -> list[pl.Expr]:
    """θ_stab = μ + (1-B)·(θ_raw - μ),  B = T/(T + PA)."""
    b = pl.lit(T) / (pl.lit(T) + pl.col(pa_col).clip(lower_bound=0))
    return [
        b.alias(f"{out}_shrinkage_b"),
        (pl.lit(prior) + (1 - b) * (pl.col(raw_col) - pl.lit(prior))).alias(f"{out}_stabilized"),
    ]


def add_season_stabilized(daily: pl.DataFrame) -> pl.DataFrame:
    """Añade stats season-to-date estabilizadas, anti-leakage shift(1)."""
    daily = daily.sort(["batter_id", "season", "game_date"])
    grp = ["batter_id", "season"]
    daily = daily.with_columns([
        pl.col("pa").shift(1, fill_value=0).cum_sum().over(grp).alias("pa_std"),
        pl.col("k").shift(1, fill_value=0).cum_sum().over(grp).alias("k_std"),
        pl.col("bb").shift(1, fill_value=0).cum_sum().over(grp).alias("bb_std"),
        pl.col("hr").shift(1, fill_value=0).cum_sum().over(grp).alias("hr_std"),
        pl.col("hits").shift(1, fill_value=0).cum_sum().over(grp).alias("hits_std"),
        pl.col("bip").shift(1, fill_value=0).cum_sum().over(grp).alias("bip_std"),
        pl.col("tb").shift(1, fill_value=0).cum_sum().over(grp).alias("tb_std"),
        pl.col("xwoba_mean").shift(1, fill_value=None)
          .rolling_mean(window_size=200, min_periods=1)
          .over(grp).alias("woba_raw"),
    ])
    daily = daily.with_columns([
        (pl.col("k_std") / pl.col("pa_std").clip(lower_bound=1)).alias("k_rate_raw"),
        (pl.col("bb_std") / pl.col("pa_std").clip(lower_bound=1)).alias("bb_rate_raw"),
        ((pl.col("hits_std") - pl.col("hr_std"))
         / pl.col("bip_std").clip(lower_bound=1)).alias("babip_raw"),
        # ISO = (TB - H) / PA — aproximación consistente train/serve
        ((pl.col("tb_std") - pl.col("hits_std"))
         / pl.col("pa_std").clip(lower_bound=1)).alias("iso_raw"),
    ])
    exprs: list[pl.Expr] = []
    exprs += _stabilize_exprs("woba_raw",    "pa_std", STAB_T["woba"],    LEAGUE_AVG["woba"],    "woba")
    exprs += _stabilize_exprs("k_rate_raw",  "pa_std", STAB_T["k_rate"],  LEAGUE_AVG["k_rate"],  "k_rate")
    exprs += _stabilize_exprs("bb_rate_raw", "pa_std", STAB_T["bb_rate"], LEAGUE_AVG["bb_rate"], "bb_rate")
    exprs += _stabilize_exprs("babip_raw",   "pa_std", STAB_T["babip"],   LEAGUE_AVG["babip"],   "babip")
    exprs += _stabilize_exprs("iso_raw",     "pa_std", STAB_T["iso"],     LEAGUE_AVG["iso"],     "iso")
    return daily.with_columns(exprs)


# ---------------------------------------------------------------------------
# Platoon career-to-date vs mano del lanzador (grano diario, sin explosión m:n)
# ---------------------------------------------------------------------------

def _platoon_daily_grain(flagged: pl.DataFrame) -> pl.DataFrame:
    """Agrega flags PA-level al grano (batter_id, pitcher_throws, game_date)."""
    return (
        flagged.group_by(["batter_id", "pitcher_throws", "game_date"])
        .agg([
            pl.col("pa_flag").sum().alias("pa"),
            pl.col("k_flag").sum().alias("k"),
            pl.col("bb_flag").sum().alias("bb"),
            pl.col("hr_flag").sum().alias("hr"),
            pl.col("hit_flag").sum().alias("hits"),
            pl.col("bip_flag").sum().alias("bip"),
            pl.col("tb").sum().alias("tb"),
            pl.col("xwoba").drop_nulls().sum().alias("xwoba_sum"),
            pl.col("xwoba").is_not_null().sum().alias("xwoba_n"),
        ])
    )


def platoon_career_table(flagged: pl.DataFrame) -> pl.DataFrame:
    """Career-to-date vs mano del pitcher, James-Stein, una fila por
    (batter_id, pitcher_throws, game_date) — clave única, join seguro.

    El grano diario + shift(1) excluye TODOS los PAs del propio día
    (consistente con las rolling features).
    """
    return _platoon_from_daily(_platoon_daily_grain(flagged))


# ---------------------------------------------------------------------------
# Pitcher rolling FIP (30 juegos, shift(1))
# ---------------------------------------------------------------------------

def pitcher_rolling_table(silver: pl.DataFrame, n_games: int = 30) -> pl.DataFrame:
    """FIP y rates rolling por (pitcher_id, game_date), anti-leakage shift(1).

    Flags desde pa_result/pa_outcome_int (disponibles tanto en Silver como en
    el build de Gold) para que training y serving usen la misma definición.
    """
    df = silver.with_columns([
        pl.col("pa_result").is_in(_K_EVENTS).cast(pl.Int32).alias("p_k"),
        pl.col("pa_result").is_in(_BB_EVENTS + ["hit_by_pitch"]).cast(pl.Int32).alias("p_bb"),
        (pl.col("pa_result") == "home_run").cast(pl.Int32).alias("p_hr"),
        pl.col("pa_result").is_in(list(GIDP_EVENTS)).cast(pl.Int32).alias("p_dp"),
        ((pl.col("pa_outcome_int") == 0)
         & ~pl.col("pa_result").is_in(list(GIDP_EVENTS))).cast(pl.Int32).alias("p_out"),
    ])
    daily = (
        df.group_by(["pitcher_id", "game_date"])
        .agg([
            pl.col("p_k").sum(), pl.col("p_bb").sum(), pl.col("p_hr").sum(),
            pl.col("p_out").sum(), pl.col("p_dp").sum(),
            pl.len().alias("p_pa"),
        ])
        .sort(["pitcher_id", "game_date"])
    )
    stat_cols = ["p_k", "p_bb", "p_hr", "p_out", "p_dp", "p_pa"]
    daily = daily.with_columns([
        pl.col(c).shift(1, fill_value=0).over("pitcher_id").alias(f"s_{c}")
        for c in stat_cols
    ]).with_columns([
        pl.col(f"s_{c}").rolling_sum(n_games, min_periods=1).over("pitcher_id").alias(f"r_{c}")
        for c in stat_cols
    ])
    daily = daily.with_columns(
        ((pl.col("r_p_out") + pl.col("r_p_k") + 2 * pl.col("r_p_dp")) / 3.0).alias("r_ip")
    )

    # Shrinkage gradual hacia la media de liga: w = min(PA/FIP_MIN_PA, 1)
    w = (pl.col("r_p_pa") / FIP_MIN_PA).clip(upper_bound=1.0)
    fip_real = (
        (13 * pl.col("r_p_hr") + 3 * pl.col("r_p_bb") - 2 * pl.col("r_p_k"))
        / pl.col("r_ip").clip(lower_bound=0.33) + FIP_CONSTANT
    ).clip(lower_bound=FIP_FLOOR, upper_bound=FIP_CEIL)

    daily = daily.with_columns([
        (w * fip_real + (1 - w) * FIP_NEUTRAL).cast(pl.Float32).alias("pitcher_fip"),
        (w * (pl.col("r_p_k") / pl.col("r_p_pa").clip(lower_bound=1))
         + (1 - w) * LEAGUE_AVG["k_rate"]).cast(pl.Float32).alias("pitcher_k_rate"),
        (w * (pl.col("r_p_bb") / pl.col("r_p_pa").clip(lower_bound=1))
         + (1 - w) * LEAGUE_AVG["bb_rate"]).cast(pl.Float32).alias("pitcher_bb_rate"),
        (w * (pl.col("r_p_hr") / pl.col("r_p_pa").clip(lower_bound=1))
         + (1 - w) * 0.033).cast(pl.Float32).alias("pitcher_hr_rate"),
    ])
    return daily.select(["pitcher_id", "game_date"] + PITCHER_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Snapshots de serving (fila sintética + funciones batch)
# ---------------------------------------------------------------------------

def _synthetic_daily_row(batter_id: int, game_date: str, season: int) -> pl.DataFrame:
    """Fila diaria vacía para 'esta noche': counts=0, métricas null."""
    return pl.DataFrame({
        "batter_id": [batter_id], "game_date": [game_date], "season": [season],
        "pa": [0], "k": [0], "bb": [0], "hr": [0], "hits": [0],
        "hh": [0], "bip": [0], "tb": [0],
        "xwoba_mean": [None], "ls_mean": [None],
    }).with_columns([
        pl.col("pa").cast(pl.UInt32), pl.col("k").cast(pl.UInt32),
        pl.col("bb").cast(pl.UInt32), pl.col("hr").cast(pl.UInt32),
        pl.col("hits").cast(pl.UInt32), pl.col("hh").cast(pl.UInt32),
        pl.col("bip").cast(pl.UInt32), pl.col("tb").cast(pl.Int64),
        pl.col("xwoba_mean").cast(pl.Float64), pl.col("ls_mean").cast(pl.Float64),
        pl.col("season").cast(pl.Int32),
    ])


def batter_snapshot(
    rows: pl.DataFrame,
    batter_id: int,
    game_date: str,
    season: int,
    pitcher_throws: str,
) -> dict[str, float | None]:
    """Features de bateador "as of" game_date, vía las funciones batch.

    Args:
        rows: PAs del bateador en Silver (todas las temporadas). Puede estar vacío.
        batter_id: identificador MLB.
        game_date: fecha del partido (YYYY-MM-DD); debe ser > que todo el historial.
        season: temporada del partido.
        pitcher_throws: "R"/"L" del abridor rival (para las features platoon).

    Returns:
        Dict feature_name → valor (None cuando el batch produce null).
    """
    rows = rows.filter(pl.col("game_date") < game_date)
    flagged = add_event_flags(rows) if not rows.is_empty() else None

    # Grano diario + fila sintética de esta noche
    if flagged is not None:
        daily = daily_grain(flagged)
        daily = pl.concat(
            [daily, _synthetic_daily_row(batter_id, game_date, season)],
            how="vertical_relaxed",
        ).sort("game_date")
    else:
        daily = _synthetic_daily_row(batter_id, game_date, season)

    daily = add_shift_rolling(daily)
    daily = add_season_stabilized(daily)
    last = daily.tail(1).to_dicts()[0]

    out: dict[str, float | None] = {
        c: last.get(c) for c in ROLLING_FEATURE_COLS + STAB_FEATURE_COLS
    }

    # Platoon vs la mano de esta noche (misma fila sintética, en grano platoon)
    if flagged is not None:
        synth_p = pl.DataFrame({
            "batter_id": [batter_id], "pitcher_throws": [pitcher_throws],
            "game_date": [game_date],
            "pa": [0], "k": [0], "bb": [0], "hr": [0], "hits": [0],
            "bip": [0], "tb": [0], "xwoba_sum": [0.0], "xwoba_n": [0],
        })
        daily_p = pl.concat(
            [_platoon_daily_grain(flagged), synth_p], how="vertical_relaxed"
        )
        platoon = _platoon_from_daily(daily_p)
        prow = (
            platoon
            .filter((pl.col("pitcher_throws") == pitcher_throws)
                    & (pl.col("game_date") == game_date))
            .tail(1)
        )
        if not prow.is_empty():
            pdict = prow.to_dicts()[0]
            out.update({c: pdict.get(c) for c in PLATOON_FEATURE_COLS})
        else:
            out.update(_neutral_platoon())
    else:
        out.update(_neutral_platoon())

    return out


def _platoon_from_daily(daily_p: pl.DataFrame) -> pl.DataFrame:
    """Tramo cum_sum + James-Stein de platoon_career_table, sobre grano diario."""
    grp = ["batter_id", "pitcher_throws"]
    daily_p = daily_p.sort(["batter_id", "pitcher_throws", "game_date"])
    daily_p = daily_p.with_columns([
        pl.col(c).shift(1, fill_value=0).cum_sum().over(grp).alias(f"c_{c}")
        for c in ("pa", "k", "bb", "hr", "hits", "bip", "tb", "xwoba_sum", "xwoba_n")
    ])
    daily_p = daily_p.with_columns([
        pl.when(pl.col("c_xwoba_n") > 0)
          .then(pl.col("c_xwoba_sum") / pl.col("c_xwoba_n"))
          .otherwise(None).alias("woba_vh_raw"),
        (pl.col("c_k") / pl.col("c_pa").clip(lower_bound=1)).alias("k_vh_raw"),
        (pl.col("c_bb") / pl.col("c_pa").clip(lower_bound=1)).alias("bb_vh_raw"),
        ((pl.col("c_hits") - pl.col("c_hr"))
         / pl.col("c_bip").clip(lower_bound=1)).alias("babip_vh_raw"),
        ((pl.col("c_tb") - pl.col("c_hits"))
         / pl.col("c_pa").clip(lower_bound=1)).alias("iso_vh_raw"),
    ])
    exprs: list[pl.Expr] = []
    for raw, T, prior, out in [
        ("woba_vh_raw",  STAB_T["woba"],    LEAGUE_AVG["woba"],    "woba_vs_hand"),
        ("k_vh_raw",     STAB_T["k_rate"],  LEAGUE_AVG["k_rate"],  "k_rate_vs_hand"),
        ("bb_vh_raw",    STAB_T["bb_rate"], LEAGUE_AVG["bb_rate"], "bb_rate_vs_hand"),
        ("babip_vh_raw", STAB_T["babip"],   LEAGUE_AVG["babip"],   "babip_vs_hand"),
        ("iso_vh_raw",   STAB_T["iso"],     LEAGUE_AVG["iso"],     "iso_vs_hand"),
    ]:
        b = pl.lit(T) / (pl.lit(T) + pl.col("c_pa").clip(lower_bound=0))
        exprs.append(
            (pl.lit(prior) + (1 - b) * (pl.col(raw) - pl.lit(prior)))
            .alias(f"{out}_stabilized")
        )
    daily_p = daily_p.with_columns(exprs).with_columns(
        pl.col("c_pa").cast(pl.Float32).alias("pa_vs_hand")
    )
    return daily_p.select(
        ["batter_id", "pitcher_throws", "game_date"] + PLATOON_FEATURE_COLS
    )


def _neutral_platoon() -> dict[str, float]:
    """Platoon de jugador sin historial: prior de liga, 0 PAs."""
    return {
        "woba_vs_hand_stabilized":    LEAGUE_AVG["woba"],
        "k_rate_vs_hand_stabilized":  LEAGUE_AVG["k_rate"],
        "bb_rate_vs_hand_stabilized": LEAGUE_AVG["bb_rate"],
        "babip_vs_hand_stabilized":   LEAGUE_AVG["babip"],
        "iso_vs_hand_stabilized":     LEAGUE_AVG["iso"],
        "pa_vs_hand":                 0.0,
    }


def pitcher_snapshot(
    silver: pl.DataFrame,
    pitcher_id: int | None,
    game_date: str,
) -> dict[str, float]:
    """Features de pitcher "as of" game_date, vía pitcher_rolling_table.

    Devuelve valores neutrales de liga cuando el pitcher no tiene historial.
    """
    neutral = {
        "pitcher_fip":     FIP_NEUTRAL,
        "pitcher_k_rate":  LEAGUE_AVG["k_rate"],
        "pitcher_bb_rate": LEAGUE_AVG["bb_rate"],
        "pitcher_hr_rate": 0.033,
    }
    if not pitcher_id:
        return neutral
    rows = silver.filter(
        (pl.col("pitcher_id") == pitcher_id) & (pl.col("game_date") < game_date)
    )
    if rows.is_empty():
        return neutral
    # Fila sintética: un PA dummy en game_date que el shift(1) interno excluye
    synth = rows.tail(1).with_columns([
        pl.lit(game_date).alias("game_date"),
        pl.lit("field_out").alias("pa_result"),
        pl.lit(0).cast(pl.Int32).alias("pa_outcome_int"),
    ])
    table = pitcher_rolling_table(pl.concat([rows, synth], how="vertical_relaxed"))
    last = table.filter(pl.col("game_date") == game_date).tail(1)
    if last.is_empty():
        return neutral
    d = last.to_dicts()[0]
    return {c: d[c] for c in PITCHER_FEATURE_COLS}
