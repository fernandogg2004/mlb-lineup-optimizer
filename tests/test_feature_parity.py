"""
test_feature_parity.py — Guardián anti-skew training/serving.

Verifica que el snapshot de serving (batter_snapshot / pitcher_snapshot)
produce EXACTAMENTE los mismos valores que las funciones batch que generan el
Gold de entrenamiento, sobre un Silver sintético controlado.

Si este test falla, training y serving han divergido — el bug más caro del
sistema (no aparece en ninguna métrica offline).
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from src.features import shared_features as sf

BATTER = 1001
PITCHER = 2001


@pytest.fixture(scope="module")
def synthetic_silver() -> pl.DataFrame:
    """Silver sintético: 1 bateador, 40 juegos, 4 PAs/juego, 2 temporadas."""
    rows = []
    outcomes = [
        ("field_out", None, 0), ("strikeout", None, 1),
        ("single", "single", 3), ("walk", None, 2),
        ("home_run", "home_run", 6), ("double", "double", 4),
    ]
    i = 0
    for season, month_start in [(2023, 4), (2024, 4)]:
        for g in range(20):
            day = g + 1
            gd = f"{season}-{month_start:02d}-{day:02d}"
            for pa in range(4):
                pa_result, hit_type, oc = outcomes[i % len(outcomes)]
                rows.append({
                    "batter_id": BATTER,
                    "pitcher_id": PITCHER,
                    "game_date": gd,
                    "season": season,
                    "batter_stand": "R",
                    "pitcher_throws": "R" if (g + pa) % 3 else "L",
                    "pa_result": pa_result,
                    "hit_type": hit_type,
                    "xwoba": 0.250 + (i % 10) * 0.03,
                    "launch_speed": 85.0 + (i % 15),
                    "launch_angle": 12.0,
                    "pa_outcome_int": oc,
                })
                i += 1
    return pl.DataFrame(rows).with_columns([
        pl.col("xwoba").cast(pl.Float32),
        pl.col("launch_speed").cast(pl.Float32),
        pl.col("season").cast(pl.Int32),
        pl.col("pa_outcome_int").cast(pl.Int32),
    ])


def _close(a, b, tol=1e-5) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) and isinstance(b, float) and (math.isnan(a) or math.isnan(b)):
        return math.isnan(a) and math.isnan(b)
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))


def test_rolling_and_stab_parity(synthetic_silver):
    """Snapshot as-of una fecha real == fila batch de esa fecha."""
    flagged = sf.add_event_flags(synthetic_silver)
    daily = sf.add_shift_rolling(sf.daily_grain(flagged))
    stab = sf.add_season_stabilized(sf.daily_grain(flagged))

    test_date = "2024-04-15"
    hand = "R"
    rows = synthetic_silver.filter(pl.col("batter_id") == BATTER)
    snap = sf.batter_snapshot(rows, BATTER, test_date, 2024, hand)

    b = daily.filter(
        (pl.col("batter_id") == BATTER) & (pl.col("game_date") == test_date)
    ).to_dicts()[0]
    s = stab.filter(
        (pl.col("batter_id") == BATTER) & (pl.col("game_date") == test_date)
    ).to_dicts()[0]

    for c in sf.ROLLING_FEATURE_COLS:
        assert _close(b.get(c), snap.get(c)), f"rolling skew en {c}: {b.get(c)} vs {snap.get(c)}"
    for c in sf.STAB_FEATURE_COLS:
        assert _close(s.get(c), snap.get(c)), f"stab skew en {c}: {s.get(c)} vs {snap.get(c)}"


def test_platoon_parity(synthetic_silver):
    flagged = sf.add_event_flags(synthetic_silver)
    plat = sf.platoon_career_table(flagged)

    test_date = "2024-04-15"
    hand = "R"
    rows = synthetic_silver.filter(pl.col("batter_id") == BATTER)
    snap = sf.batter_snapshot(rows, BATTER, test_date, 2024, hand)

    p = plat.filter(
        (pl.col("batter_id") == BATTER)
        & (pl.col("pitcher_throws") == hand)
        & (pl.col("game_date") == test_date)
    ).to_dicts()[0]

    for c in sf.PLATOON_FEATURE_COLS:
        assert _close(p.get(c), snap.get(c)), f"platoon skew en {c}: {p.get(c)} vs {snap.get(c)}"


def test_pitcher_parity(synthetic_silver):
    pit = sf.pitcher_rolling_table(synthetic_silver)
    test_date = "2024-04-15"
    snap = sf.pitcher_snapshot(synthetic_silver, PITCHER, test_date)

    p = pit.filter(
        (pl.col("pitcher_id") == PITCHER) & (pl.col("game_date") == test_date)
    ).to_dicts()[0]

    for c in sf.PITCHER_FEATURE_COLS:
        assert _close(p.get(c), snap.get(c)), f"pitcher skew en {c}: {p.get(c)} vs {snap.get(c)}"


def test_rookie_sin_historial_produce_nan_no_ceros(synthetic_silver):
    """Jugador sin historia: rolling = None/NaN (rama default LGBM), NUNCA 0.0."""
    empty = synthetic_silver.filter(pl.col("batter_id") == -1)
    snap = sf.batter_snapshot(empty, 999_999, "2024-06-01", 2024, "R")

    for c in ("k_rate_30d", "xwoba_7d", "bb_rate_15d"):
        v = snap.get(c)
        assert v is None or (isinstance(v, float) and math.isnan(v)), (
            f"{c}={v}: imputar 0.0 a un rookie lo convierte en élite que nunca se poncha"
        )
    # Stabilized k/bb/babip/iso → prior de liga (B=1), igual que en training
    from src.constants import LEAGUE_AVG
    assert _close(snap["k_rate_stabilized"], LEAGUE_AVG["k_rate"])
    assert _close(snap["bb_rate_stabilized"], LEAGUE_AVG["bb_rate"])


def test_platoon_key_unica_sin_explosion(synthetic_silver):
    """La tabla platoon debe tener clave única (el join PA-level no multiplica filas)."""
    flagged = sf.add_event_flags(synthetic_silver)
    plat = sf.platoon_career_table(flagged)
    n_keys = plat.select(["batter_id", "pitcher_throws", "game_date"]).unique().height
    assert n_keys == plat.height


def test_anti_leakage_shift(synthetic_silver):
    """La primera fila de la carrera no puede tener rolling features (shift(1))."""
    flagged = sf.add_event_flags(synthetic_silver)
    daily = sf.add_shift_rolling(sf.daily_grain(flagged))
    first = daily.sort("game_date").head(1).to_dicts()[0]
    assert first["k_rate_7d"] is None
    assert first["xwoba_ewma_alpha02"] is None
