"""
predict_tonight.py
==================
Descarga los lineups de los partidos de esta noche via MLB Stats API,
calcula prob_vectors para cada bateador usando el AtBatPredictor entrenado,
y sugiere el orden de bateo optimo con el SabermetricSeeder.

Uso:
    python predict_tonight.py                     # lista partidos de hoy, elige uno
    python predict_tonight.py --team NYY          # busca partido de ese equipo
    python predict_tonight.py --team LAD --side home
    python predict_tonight.py --date 2026-05-20   # otra fecha

Nota: Los features rolling se calculan desde los Silver Parquets (datos 2015-2024).
Para jugadores sin historial (rookies post-2024) se usan valores cero.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import requests

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Fix pickle __main__ issue: the model was saved when run as __main__, so
# AtBatModelConfig is stored as __main__.AtBatModelConfig in the pickle.
# ---------------------------------------------------------------------------
import src.models.model_at_bat as _mat_module  # noqa: E402
for _n in dir(_mat_module):
    if not _n.startswith("__"):
        setattr(sys.modules["__main__"], _n, getattr(_mat_module, _n))

from src.optimizer.lineup_optimizer import PlayerStats, SabermetricSeeder  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MLB_API = "https://statsapi.mlb.com/api/v1"

# 8-class outcome names and run values (must match model_at_bat.PAOutcome)
#   0=OUT_IN_PLAY  1=STRIKEOUT  2=WALK_HBP  3=SINGLE  4=DOUBLE
#   5=TRIPLE       6=HOME_RUN   7=DOUBLE_PLAY (GIDP)
OUTCOME_NAMES = ["OUT", "K", "BB/HBP", "1B", "2B", "3B", "HR", "DP"]
from src.constants import RUN_VALUES  # noqa: E402  — single source of truth

# Post-hoc calibration scale for Monte Carlo E[R/game] output.
# RETIRADO (= 1.0) desde el retrain 2026-06-12: el modelo v5 (sin class
# weights, sin leakage intra-PA, con park factors como features) pasó el gate
# de prior drift en train_v3.py — sesgo E[R/PA] pred-obs = -0.0001.
# El valor 1.0 es además el sentinel que desactiva la auto-recalibración
# semanal en morning.py. Si un retrain futuro falla el gate, train_v3.py no
# copia el modelo a producción — NO volver a parchear con este escalar.
_MC_RUNS_SCALE: float = 1.0

MODEL_PATH = ROOT / "models" / "at_bat_predictor.pkl"
SILVER_DIR = ROOT / "data" / "silver" / "plate_appearances"

# James-Stein stabilization thresholds and league averages — from central constants.
from src.constants import LEAGUE_AVG as _LEAGUE_AVG  # noqa: E402

# League-average PA outcome distribution — from central constants.
from src.constants import LEAGUE_AVG_PA_PROBS as _LEAGUE_AVG_PA_PROBS, LEAGUE_AVG_LINEUP as _LEAGUE_AVG_LINEUP  # noqa: E402

# FEATURE_COLS is fetched from the model at runtime (see _load_model).
# This empty sentinel is replaced after the model loads.
FEATURE_COLS: list[str] = []

# Minimum gap (days) before issuing a data-staleness warning.
_STALENESS_WARN_DAYS = 90


# ---------------------------------------------------------------------------
# MLB Stats API helpers
# ---------------------------------------------------------------------------

def _api_get(path: str, params: dict | None = None) -> dict:
    resp = requests.get(f"{MLB_API}{path}", params=params or {}, timeout=12)
    resp.raise_for_status()
    return resp.json()


def fetch_games(game_date: str) -> list[dict]:
    data = _api_get("/schedule", {
        "sportId": 1, "date": game_date,
        "hydrate": "lineups,team,probablePitcher",
    })
    return [g for d in data.get("dates", []) for g in d.get("games", [])]


def _parse_lineup(game: dict, side: str) -> list[dict]:
    players = game.get("lineups", {}).get(f"{side}Players", [])
    return [{"id": p.get("id"), "fullName": p.get("fullName", f"Player {p.get('id')}")}
            for p in players]


def fetch_roster(team_id: int) -> list[dict]:
    try:
        data = _api_get(f"/teams/{team_id}/roster", {"rosterType": "active"})
        return [
            {"id": p["person"]["id"], "fullName": p["person"]["fullName"]}
            for p in data.get("roster", [])
            if p.get("position", {}).get("abbreviation", "") != "P"
        ][:9]
    except Exception:
        return []


def _fetch_pitcher_hand(pitcher_id: int | None) -> str:
    """Returns the throwing arm of a pitcher ('R' or 'L') from the MLB Stats API.

    Falls back to 'R' on any error so the pipeline never blocks on an API failure.
    """
    if not pitcher_id:
        return "R"
    try:
        data = _api_get(f"/people/{pitcher_id}")
        code = (
            data.get("people", [{}])[0]
                .get("pitchHand", {})
                .get("code", "R")
        )
        return code if code in ("R", "L") else "R"
    except Exception:
        return "R"


# ---------------------------------------------------------------------------
# Data-staleness check
# ---------------------------------------------------------------------------

def _check_silver_staleness(silver: pl.DataFrame, game_date: str) -> None:
    """Warns when Silver data lags more than _STALENESS_WARN_DAYS behind game_date.

    A large gap means player hot/cold streaks and roster changes from the
    missing period are invisible to the model — predictions default to
    career averages for the affected players.
    """
    if "game_date" not in silver.columns:
        return
    last_raw = silver["game_date"].max()
    if last_raw is None:
        return
    try:
        pred_dt = date.fromisoformat(game_date)
        last_dt = last_raw if isinstance(last_raw, date) else last_raw.date()
        gap_days = (pred_dt - last_dt).days
    except Exception:
        return

    if gap_days > _STALENESS_WARN_DAYS:
        pct_no_recent = (
            silver.filter(
                pl.col("game_date") >= pl.lit(str(pred_dt - timedelta(days=90)))
            ).height == 0
        )
        print(
            f"\n  AVISO: Datos Silver con {gap_days} dias de retraso "
            f"(ultimo: {last_dt}, prediccion: {game_date})."
        )
        print(
            "    Los jugadores sin datos de 2025+ recibiran valores de "
            "liga promedio — precision reducida."
        )
        print("    Ejecuta: python build_silver.py --years <año> --force para actualizar.\n")


# ---------------------------------------------------------------------------
# Park factors
# ---------------------------------------------------------------------------

def _get_park_factors(game: dict) -> tuple[float, float]:
    """Returns (park_factor_hr, park_factor_xb) for the game venue.

    Looks up the venue ID in src.constants.PARK_FACTORS. Falls back to
    (1.0, 1.0) for any unlisted venue (neutral assumption).
    """
    from src.constants import PARK_FACTORS
    venue_id = str(game.get("venue", {}).get("id", ""))
    pf = PARK_FACTORS.get(venue_id, {})
    return pf.get("hr", 1.0), pf.get("xb", 1.0)


# ---------------------------------------------------------------------------
# Feature computation (replicates training pipeline from features_rolling.py)
# ---------------------------------------------------------------------------

def _load_silver() -> pl.DataFrame:
    parts = [
        pl.read_parquet(pq, hive_partitioning=False)
        for sd in sorted(SILVER_DIR.glob("season=*"))
        if (pq := sd / "data.parquet").exists()
    ]
    if not parts:
        raise FileNotFoundError(f"No Silver Parquets in {SILVER_DIR}")
    return pl.concat(parts, how="diagonal_relaxed").sort(["batter_id", "game_date"])


def compute_features(
    batter_id: int,
    pitcher_throws: str,
    silver: pl.DataFrame,
    feature_names: list[str] | None = None,
    pitcher_id: int | None = None,
    game_date: str | None = None,
    park_factor_hr: float = 1.0,
    park_factor_xb: float = 1.0,
    is_home: float = 0.5,
) -> np.ndarray:
    """
    Construye el vector de features para un bateador vs un lanzador.

    Anti-skew: NO reimplementa el pipeline — llama a las MISMAS funciones de
    src/features/shared_features.py que usa build_gold_v3.py, evaluadas sobre
    el historial del jugador más una fila sintética con la fecha del partido.
    Los nulos sobreviven como NaN (rama default de LightGBM); PROHIBIDO
    imputar 0.0 en rolling stats.

    Args:
        batter_id:      MLB batter identifier.
        pitcher_throws: "R" or "L" for the opposing pitcher's handedness.
        silver:         Silver plate_appearances DataFrame (all seasons).
        feature_names:  Ordered list from the loaded model (_feature_names).
                        When None, falls back to the module-level FEATURE_COLS.
        pitcher_id:     MLB pitcher identifier. When provided, real FIP stats are
                        computed from Silver; otherwise league-average values are used.
        game_date:      Fecha del partido YYYY-MM-DD (default: hoy). Las features
                        usan solo datos estrictamente anteriores a esta fecha.
        park_factor_hr: Factor HR del estadio (feature del modelo, no post-hoc).
        park_factor_xb: Factor extra-bases del estadio.
        is_home:        1.0 si el bateador es local, 0.0 visitante, 0.5 desconocido.
    """
    from src.features.shared_features import batter_snapshot, pitcher_snapshot

    fcols = feature_names if feature_names is not None else FEATURE_COLS
    if game_date is None:
        game_date = str(date.today())
    season = int(game_date[:4])

    rows = silver.filter(pl.col("batter_id") == batter_id)

    # ── batter_stand (switch hitters batean del lado opuesto al pitcher) ─────
    if not rows.is_empty() and "batter_stand" in rows.columns:
        stands = set(rows["batter_stand"].drop_nulls().unique().to_list())
        if len(stands) >= 2:
            batter_stand_enc = 0 if pitcher_throws == "R" else 1
        else:
            top = rows["batter_stand"].drop_nulls().mode()[0]
            batter_stand_enc = 1 if top == "R" else 0
    else:
        batter_stand_enc = 1  # default right-handed

    # ── Snapshots vía shared_features (mismas funciones que el training) ─────
    feats: dict[str, float | None] = {}
    feats.update(batter_snapshot(rows, batter_id, game_date, season, pitcher_throws))
    feats.update(pitcher_snapshot(silver, pitcher_id, game_date))

    # ── Contexto de juego ─────────────────────────────────────────────────────
    feats.update({
        "batter_stand":             float(batter_stand_enc),
        "pitcher_throws":           1.0 if pitcher_throws == "R" else 0.0,
        "era_shift_ban":            1.0 if season >= 2023 else 0.0,
        "era_universal_dh":         1.0 if season >= 2020 else 0.0,
        "era_first_year_shift_ban": 1.0 if season == 2023 else 0.0,
        "park_factor_hr":           float(park_factor_hr),
        "park_factor_xb":           float(park_factor_xb),
        "is_home":                  float(is_home),
    })

    # Compat con modelos legacy (pre-retrain) que aún esperan las features
    # intra-PA eliminadas por leakage — imputes idénticos al serving anterior.
    feats.setdefault("pitch_count_in_pa", 4.0)
    feats.setdefault("last_pitch_type", 0.0)

    # Ensamblar en el orden del modelo. None → NaN (rama default de LightGBM).
    return np.array(
        [np.nan if feats.get(c) is None else float(feats.get(c, np.nan))
         for c in fcols],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_batter_table(results: list[dict]) -> None:
    cols = "  ".join(f"{n:>8}" for n in OUTCOME_NAMES)
    header = f"{'#':>2}  {'Batter':<22} {'E[R/PA]':>7}  {cols}"
    sep = "=" * len(header)
    print("\n" + sep)
    print(header)
    print("-" * len(header))
    for r in results:
        pv = r["prob_vector"]
        ev = float(pv @ RUN_VALUES)
        probs = "  ".join(f"{p*100:>7.1f}%" for p in pv)
        print(f"{r['slot']:>2}  {r['name']:<22} {ev:>7.4f}  {probs}")
    print(sep)


def _print_optimal_order(order: list[dict]) -> None:
    print("\n-- Optimal Batting Order (SabermetricSeeder) -----------------")
    for slot, b in enumerate(order, 1):
        ev = float(b["prob_vector"] @ RUN_VALUES)
        print(f"  {slot}. {b['name']:<22}  E[R/PA]={ev:.4f}  "
              f"wOBA~{b['woba']:.3f}  OBP~{b['obp']:.3f}")
    print()


# ---------------------------------------------------------------------------
# Monte Carlo simulation helper
# ---------------------------------------------------------------------------

def _run_game_simulation(
    my_lineup_probs: np.ndarray,
    opp_lineup_probs: np.ndarray,
    park_factor_hr: float = 1.0,
    park_factor_xb: float = 1.0,
    n_sims: int = 10_000,
) -> dict:
    """Runs a Monte Carlo simulation for one team vs. opponent.

    Uses the Numba-parallel engine (no Ray required). With 10k simulations
    the standard error on E[R] is < 0.03 runs — sufficient for game-day
    decisions. Set n_sims=50_000 for tighter percentile estimates.

    Args:
        my_lineup_probs:  Shape (9, 8) float32 — my batting order.
        opp_lineup_probs: Shape (9, 8) float32 — opponent batting order
                          (use _LEAGUE_AVG_LINEUP when opponent is unknown).
        park_factor_hr:   HR park factor (1.0 = neutral).
        park_factor_xb:   Extra-base hit park factor (1.0 = neutral).
        n_sims:           Number of games to simulate.

    Returns:
        Dict with simulation statistics ready to merge into the result JSON.
    """
    from src.simulation.simulation_engine import MonteCarloConfig, MonteCarloEngine

    cfg = MonteCarloConfig(
        n_simulations=n_sims,
        use_ray=False,
        park_factor_hr=park_factor_hr,
        park_factor_xb=park_factor_xb,
        use_extra_innings=False,
    )
    engine = MonteCarloEngine(cfg)
    sim = engine.run(my_lineup_probs.astype(np.float32),
                     opp_lineup_probs.astype(np.float32))

    pct = sim.runs_scored_percentiles
    s   = _MC_RUNS_SCALE   # post-hoc bias correction (see constant definition)
    return {
        "expected_runs_per_game": round(sim.expected_runs_scored * s, 2),
        "runs_p05":               round(pct[5]  * s, 1),
        "runs_p25":               round(pct[25] * s, 1),
        "runs_p50":               round(pct[50] * s, 1),
        "runs_p75":               round(pct[75] * s, 1),
        "runs_p95":               round(pct[95] * s, 1),
        "win_probability":        round(sim.win_probability, 3),      # no escalar
        "win_prob_ci_low":        round(sim.win_prob_ci_low, 3),      # no escalar
        "win_prob_ci_high":       round(sim.win_prob_ci_high, 3),     # no escalar
        "std_dev_runs":           round(sim.std_dev_runs_scored * s, 2),
        "uncertainty":            sim.uncertainty_level,
        "n_simulations":          sim.n_simulations,
    }


# ---------------------------------------------------------------------------
# Helpers reutilizables
# ---------------------------------------------------------------------------

def _load_model() -> tuple:
    """Load AtBatPredictor and populate the global FEATURE_COLS from model metadata.

    Returns:
        (predictor, n_features, feature_names)
    """
    global FEATURE_COLS

    predictor = _mat_module.AtBatPredictor.load(str(MODEL_PATH))

    # Sync the global FEATURE_COLS with what this model was actually trained on
    feature_names = predictor._feature_names or []
    FEATURE_COLS = feature_names

    n_classes = len(predictor._calibrated_model.classes_)
    n_feat    = predictor._n_features

    # Sanity checks
    expected_n_classes = _mat_module.N_CLASSES  # 8
    if n_classes != expected_n_classes:
        raise RuntimeError(
            f"Modelo tiene {n_classes} clases, se esperaban {expected_n_classes}. "
            "Usa el modelo retrenado con 8 clases (models/at_bat_predictor.pkl)."
        )

    return predictor, n_feat, feature_names


def _predict_one_side(
    game: dict,
    side: str,
    silver: pl.DataFrame,
    predictor,
    game_date: str,
    verbose: bool = True,
    feature_names: list[str] | None = None,
) -> dict | None:
    """Calcula el orden optimo para un equipo en un partido.

    Returns a JSON-serializable dict plus an internal '_lineup_probs_matrix'
    key (numpy array, shape 9x8) used by the caller to run Monte Carlo
    simulation. The caller must pop this key before saving to disk.

    Returns None if the roster has fewer than 9 eligible batters.
    """
    opp_side  = "home" if side == "away" else "away"
    team_info = game["teams"][side]["team"]
    away_name = game["teams"]["away"]["team"]["name"]
    home_name = game["teams"]["home"]["team"]["name"]
    pk        = game["gamePk"]

    opp_pitcher_data   = game["teams"][opp_side].get("probablePitcher", {})
    opp_pitcher        = opp_pitcher_data.get("fullName", "Unknown")
    opp_pitcher_id     = opp_pitcher_data.get("id")
    opp_pitcher_throws = _fetch_pitcher_hand(opp_pitcher_id)

    # Compute pitcher FIP from Silver data (context only — not a model input yet)
    from src.features.pitcher_fip import compute_pitcher_fip_for_inference
    pitcher_fip_ctx = compute_pitcher_fip_for_inference(opp_pitcher_id, silver)

    if verbose and opp_pitcher_id:
        fip_tag = " [est.]" if pitcher_fip_ctx.get("fip_is_estimated") else ""
        print(f"  Pitcher rival: {opp_pitcher} ({opp_pitcher_throws}HP)  "
              f"FIP={pitcher_fip_ctx['fip']:.2f}{fip_tag}  "
              f"K/9={pitcher_fip_ctx['k9']:.1f}  BB/9={pitcher_fip_ctx['bb9']:.1f}")

    lineup = _parse_lineup(game, side)
    if not lineup:
        lineup = fetch_roster(team_info["id"])
    if len(lineup) < 9:
        return None

    # Park factors + home/away entran como FEATURES del modelo (no post-hoc)
    pf_hr, pf_xb = _get_park_factors(game)
    is_home_val  = 1.0 if side == "home" else 0.0

    results = []
    low_data_batters: list[str] = []   # acumula nombres con datos insuficientes
    for slot, player in enumerate(lineup, 1):
        pid  = player["id"]
        name = player["fullName"]
        fcols = feature_names if feature_names is not None else FEATURE_COLS
        X    = compute_features(pid, opp_pitcher_throws, silver, fcols,
                                pitcher_id=opp_pitcher_id, game_date=game_date,
                                park_factor_hr=pf_hr, park_factor_xb=pf_xb,
                                is_home=is_home_val)
        pv   = predictor.predict_proba(X.reshape(1, -1))[0]
        ev   = float(pv @ RUN_VALUES)

        woba_s = float(X[fcols.index("woba_stabilized")]) if "woba_stabilized" in fcols else _LEAGUE_AVG["woba"]
        iso_s  = float(X[fcols.index("iso_stabilized")])  if "iso_stabilized"  in fcols else _LEAGUE_AVG["iso"]
        if not np.isfinite(woba_s):
            woba_s = _LEAGUE_AVG["woba"]
        if not np.isfinite(iso_s):
            iso_s = _LEAGUE_AVG["iso"]
        obp_e  = min(woba_s / 0.87 + 0.04, 0.450) if woba_s > 0.01 else 0.330
        in_hist = silver.filter(pl.col("batter_id") == pid).height > 0

        # Detectar bateadores con datos insuficientes (rookies o muy poca historia)
        pa_30d = float(X[fcols.index("pa_30d")]) if "pa_30d" in fcols else 0.0
        if not in_hist or not np.isfinite(pa_30d) or pa_30d < 30:
            low_data_batters.append(name)

        if verbose:
            tag = "" if in_hist else " [nuevo]"
            print(f"  {slot:>2}. {name:<22}  E[R/PA]={ev:.4f}  "
                  f"K={pv[1]*100:.1f}%  BB={pv[2]*100:.1f}%  HR={pv[6]*100:.1f}%{tag}")

        results.append({
            "slot": slot, "id": pid, "name": name,
            "prob_vector": pv, "ev": ev,
            "woba": woba_s if woba_s > 0.01 else _LEAGUE_AVG["woba"],
            "obp":  obp_e,
            "iso":  iso_s  if iso_s  > 0.001 else _LEAGUE_AVG["iso"],
            "in_hist": in_hist,
        })

    if verbose:
        _print_batter_table(results)

    players9  = [
        PlayerStats(
            player_id=r["id"], player_name=r["name"],
            obp=r["obp"], woba=r["woba"], iso=r["iso"],
            batter_stand="R", prob_vector=r["prob_vector"],
        )
        for r in results[:9]
    ]
    order_idx = SabermetricSeeder(players9).canonical_seed(pitcher_hand=opp_pitcher_throws)
    ordered   = [{"name": players9[i].player_name, "prob_vector": players9[i].prob_vector,
                  "woba": players9[i].woba, "obp": players9[i].obp}
                 for i in order_idx]

    if verbose:
        _print_optimal_order(ordered)

    # Build the lineup probability matrix in batting-order sequence (9 x 8)
    lineup_probs_matrix = np.array(
        [players9[i].prob_vector for i in order_idx], dtype=np.float32
    )

    abbr = team_info.get("abbreviation", team_info["name"].replace(" ", "")[:3]).upper()

    return {
        "game_date":   game_date,
        "game_pk":     pk,
        "matchup":     f"{away_name} @ {home_name}",
        "team":        team_info["name"],
        "team_abbr":   abbr,
        "side":        side,
        "opp_pitcher": opp_pitcher,
        "opp_pitcher_throws": opp_pitcher_throws,
        "roster_used": [r["name"] for r in results],
        "batting_order": [
            {
                "slot":      s + 1,
                "id":        players9[i].player_id,
                "name":      players9[i].player_name,
                "ev_per_pa": round(float(players9[i].prob_vector @ RUN_VALUES), 4),
                "prob_out":  round(float(players9[i].prob_vector[0]), 4),
                "prob_k":    round(float(players9[i].prob_vector[1]), 4),
                "prob_bb":   round(float(players9[i].prob_vector[2]), 4),
                "prob_1b":   round(float(players9[i].prob_vector[3]), 4),
                "prob_2b":   round(float(players9[i].prob_vector[4]), 4),
                "prob_3b":   round(float(players9[i].prob_vector[5]), 4),
                "prob_hr":   round(float(players9[i].prob_vector[6]), 4),
                "prob_dp":   round(float(players9[i].prob_vector[7])
                                   if len(players9[i].prob_vector) > 7 else 0.0, 4),
                "woba_stab": round(players9[i].woba, 4),
                "obp_est":   round(players9[i].obp, 4),
            }
            for s, i in enumerate(order_idx)
        ],
        # Provisional linear estimate (replaced by MC simulation when both lineups available)
        "expected_runs_per_game": round(
            sum(float(players9[i].prob_vector @ RUN_VALUES) for i in order_idx)
            / 9 * 27 * _MC_RUNS_SCALE, 2
        ),
        # Pitcher context (FIP and secondary rates — informational, not fed to model)
        "opp_pitcher_stats": pitcher_fip_ctx,
        # Data quality flags — indicate prediction confidence to the dashboard
        "data_quality": {
            "low_data_batters":    low_data_batters,
            "pitcher_fip_estimated": pitcher_fip_ctx.get("fip_is_estimated", True),
            "pitcher_fip_weight":    pitcher_fip_ctx.get("fip_data_weight", 0.0),
            "confidence": (
                "high"   if not low_data_batters and not pitcher_fip_ctx.get("fip_is_estimated") else
                "medium" if len(low_data_batters) <= 2 else
                "low"
            ),
        },
        # Internal field for Monte Carlo — caller must pop before saving JSON
        "_lineup_probs_matrix": lineup_probs_matrix,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predice probabilidades de PA para los partidos de esta noche"
    )
    parser.add_argument("--team",    default=None,
                        help="Abreviatura o nombre del equipo (ej. NYY, LAD, Yankees)")
    parser.add_argument("--date",    default=str(date.today()),
                        help="Fecha YYYY-MM-DD (default: hoy)")
    parser.add_argument("--side",    default="away", choices=["home", "away"],
                        help="Equipo a analizar: home o away (default: away)")
    parser.add_argument("--game-pk", type=int, default=None, dest="game_pk",
                        help="gamePk especifico (omite seleccion interactiva)")
    parser.add_argument("--output",  default=None, metavar="FILE",
                        help="Guardar resultado en JSON (ej. results/2026-05-19/NYY.json)")
    parser.add_argument("--all",     action="store_true", dest="all_games",
                        help="Procesar todos los partidos de la fecha (ambos equipos por juego)")
    parser.add_argument("--output-dir", default="results", dest="output_dir",
                        help="Carpeta de salida con --all (default: results/)")
    parser.add_argument("--no-sim",  action="store_true", dest="no_sim",
                        help="Omitir simulacion Monte Carlo (mas rapido, E[R] menos preciso)")
    parser.add_argument("--n-sims",  type=int, default=10_000, dest="n_sims",
                        help="Simulaciones Monte Carlo por partido (default: 10000)")
    args = parser.parse_args()

    game_date = args.date
    print(f"\nCargando partidos del {game_date}...")
    games = fetch_games(game_date)
    if not games:
        print(f"  No se encontraron partidos para {game_date}.")
        return
    print(f"  {len(games)} partidos encontrados")

    print("\nCargando historico Silver (2015-2024)...")
    silver = _load_silver()
    print(f"  {len(silver):,} PAs  |  {silver['batter_id'].n_unique():,} bateadores unicos")

    # Warn if Silver data is significantly behind the prediction date
    _check_silver_staleness(silver, game_date)

    print("\nCargando modelo AtBatPredictor...")
    predictor, n_feat, feature_names = _load_model()
    print(f"  Cargado | {n_feat} features | {len(predictor._calibrated_model.classes_)} clases\n")

    # Si el modelo ya aprende park factors como features, el motor MC debe
    # recibir factores NEUTRALES (aplicarlos dos veces descalibraría el output).
    # Con modelos legacy (sin park features) se mantiene el ajuste post-hoc.
    model_has_park = "park_factor_hr" in (feature_names or [])
    if model_has_park:
        print("  Park factors integrados en el modelo — motor MC en modo neutral.\n")

    # ------------------------------------------------------------------ --all
    if args.all_games:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved, skipped = [], []

        for game in games:
            away_name = game["teams"]["away"]["team"]["name"]
            home_name = game["teams"]["home"]["team"]["name"]
            pk        = game["gamePk"]
            t         = game.get("gameDate", "")[:16].replace("T", " ")
            print(f"{'='*66}")
            print(f"  {away_name} @ {home_name}  ({t} UTC)")

            # Predict both sides
            side_results: dict[str, dict] = {}
            for side in ("away", "home"):
                team = game["teams"][side]["team"]
                abbr = team.get("abbreviation", team["name"].replace(" ", "")[:3]).upper()
                print(f"\n  [{side.upper()}] {team['name']}")
                result = _predict_one_side(
                    game, side, silver, predictor, game_date,
                    verbose=True, feature_names=feature_names,
                )
                if result is None:
                    print("  Roster insuficiente — omitido.")
                    skipped.append(f"{abbr} ({side})")
                else:
                    side_results[side] = result

            # Run Monte Carlo simulation when both lineups are available.
            # Neutral cuando el modelo ya incluye park factors como features.
            pf_hr, pf_xb = (1.0, 1.0) if model_has_park else _get_park_factors(game)
            if not args.no_sim and len(side_results) == 2:
                print(f"\n  Simulando partido ({args.n_sims:,} juegos Monte Carlo)...")
                away_probs = side_results["away"].pop("_lineup_probs_matrix")
                home_probs = side_results["home"].pop("_lineup_probs_matrix")

                sim_away = _run_game_simulation(away_probs, home_probs,
                                                park_factor_hr=pf_hr,
                                                park_factor_xb=pf_xb,
                                                n_sims=args.n_sims)
                sim_home = _run_game_simulation(home_probs, away_probs,
                                                park_factor_hr=pf_hr,
                                                park_factor_xb=pf_xb,
                                                n_sims=args.n_sims)
                side_results["away"].update(sim_away)
                side_results["home"].update(sim_home)

                print(f"  {away_name}: E[R]={sim_away['expected_runs_per_game']}  "
                      f"P(W)={sim_away['win_probability']:.1%}  "
                      f"IC90=[{sim_away['runs_p05']}-{sim_away['runs_p95']}]")
                print(f"  {home_name}: E[R]={sim_home['expected_runs_per_game']}  "
                      f"P(W)={sim_home['win_probability']:.1%}  "
                      f"IC90=[{sim_home['runs_p05']}-{sim_home['runs_p95']}]")

            elif not args.no_sim and len(side_results) == 1:
                # Only one lineup available — simulate vs league-average opponent
                side = next(iter(side_results))
                my_probs = side_results[side].pop("_lineup_probs_matrix")
                sim = _run_game_simulation(my_probs, _LEAGUE_AVG_LINEUP,
                                           park_factor_hr=pf_hr,
                                           park_factor_xb=pf_xb,
                                           n_sims=args.n_sims)
                side_results[side].update(sim)
            else:
                # --no-sim or no results: remove internal field if present
                for r in side_results.values():
                    r.pop("_lineup_probs_matrix", None)

            # Save results to disk
            for side, result in side_results.items():
                team = game["teams"][side]["team"]
                abbr = team.get("abbreviation", team["name"].replace(" ", "")[:3]).upper()
                date_dir = out_dir / game_date
                date_dir.mkdir(parents=True, exist_ok=True)
                fname = date_dir / f"{abbr}.json"
                fname.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
                saved.append(str(fname))
                print(f"  Guardado: {fname}")

        print(f"\n{'='*66}")
        print(f"  Archivos guardados : {len(saved)}")
        if skipped:
            print(f"  Omitidos           : {', '.join(skipped)}")
        print(f"  Carpeta            : {(Path(args.output_dir) / game_date).resolve()}\n")
        return

    # -------------------------------------------------------------- un partido
    if args.game_pk:
        match = next((g for g in games if g["gamePk"] == args.game_pk), None)
        if not match:
            print(f"  gamePk {args.game_pk} no encontrado.")
            return
        game = match
    elif args.team:
        tu = args.team.upper()
        matches = [
            g for g in games
            if tu in g["teams"]["away"]["team"].get("abbreviation", "").upper()
            or tu in g["teams"]["home"]["team"].get("abbreviation", "").upper()
            or tu in g["teams"]["away"]["team"].get("name", "").upper()
            or tu in g["teams"]["home"]["team"].get("name", "").upper()
        ]
        if not matches:
            print(f"  Equipo '{args.team}' no encontrado. Partidos disponibles:")
            for i, g in enumerate(games):
                print(f"    [{i}] {g['teams']['away']['team']['name']} @ "
                      f"{g['teams']['home']['team']['name']}")
            return
        game = matches[0]
    else:
        print("  Partidos disponibles:")
        for i, g in enumerate(games):
            a = g["teams"]["away"]["team"]["name"]
            h = g["teams"]["home"]["team"]["name"]
            t = g.get("gameDate", "")[:16].replace("T", " ")
            s = g.get("status", {}).get("detailedState", "Sched.")
            print(f"    [{i:>2}] {a:26s} @ {h:26s}  {t} UTC  [{s}]")
        idx = input("\n  Elige el numero del partido: ").strip()
        try:
            game = games[int(idx)]
        except (ValueError, IndexError):
            print("  Seleccion invalida.")
            return

    away_name = game["teams"]["away"]["team"]["name"]
    home_name = game["teams"]["home"]["team"]["name"]
    pk        = game["gamePk"]
    print(f"  Partido: {away_name} @ {home_name}  (pk={pk})")

    result = _predict_one_side(game, args.side, silver, predictor, game_date,
                               verbose=True, feature_names=feature_names)
    if result is None:
        print("  No se pudo procesar: roster insuficiente.")
        return

    # Run Monte Carlo simulation
    if not args.no_sim:
        my_probs = result.pop("_lineup_probs_matrix")
        # Try to get the opponent's lineup for a real matchup simulation
        opp_side = "home" if args.side == "away" else "away"
        print(f"\n  Simulando ({args.n_sims:,} juegos Monte Carlo)...")
        opp_result = _predict_one_side(
            game, opp_side, silver, predictor, game_date,
            verbose=False, feature_names=feature_names,
        )
        if opp_result is not None:
            opp_probs = opp_result.pop("_lineup_probs_matrix")
        else:
            opp_probs = _LEAGUE_AVG_LINEUP
        pf_hr, pf_xb = (1.0, 1.0) if model_has_park else _get_park_factors(game)
        sim = _run_game_simulation(my_probs, opp_probs,
                                   park_factor_hr=pf_hr, park_factor_xb=pf_xb,
                                   n_sims=args.n_sims)
        result.update(sim)
        print(f"  E[R]={sim['expected_runs_per_game']}  "
              f"P(W)={sim['win_probability']:.1%}  "
              f"IC90=[{sim['runs_p05']}-{sim['runs_p95']}]  "
              f"sigma={sim['std_dev_runs']}")
    else:
        result.pop("_lineup_probs_matrix", None)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"  Resultados guardados en: {out_path}\n")


if __name__ == "__main__":
    main()
