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
from datetime import date
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

OUTCOME_NAMES = ["OUT", "K", "BB/HBP", "1B", "2B", "3B", "HR"]
RUN_VALUES    = np.array([0.0, 0.0, 0.33, 0.47, 0.75, 1.03, 1.40], dtype=np.float32)

MODEL_PATH = ROOT / "models" / "at_bat_predictor.pkl"
SILVER_DIR = ROOT / "data" / "silver" / "plate_appearances"

# James-Stein stabilization thresholds (Lichtman 2010 / "The Book")
_STAB_T = {"woba": 200, "k_rate": 60, "bb_rate": 120, "babip": 500, "iso": 160}
_LEAGUE_AVG = {
    "woba":    0.318,
    "k_rate":  0.224,
    "bb_rate": 0.083,
    "babip":   0.298,
    "iso":     0.147,
}

# 33 features in Gold column order (must match model training order exactly)
FEATURE_COLS = [
    "pa_7d", "k_rate_7d", "bb_rate_7d", "hr_rate_7d", "hard_hit_rate_7d",
    "xwoba_7d", "launch_speed_7d",
    "pa_15d", "k_rate_15d", "bb_rate_15d", "hr_rate_15d", "hard_hit_rate_15d",
    "xwoba_15d", "launch_speed_15d",
    "pa_30d", "k_rate_30d", "bb_rate_30d", "hr_rate_30d", "hard_hit_rate_30d",
    "xwoba_30d", "launch_speed_30d",
    "xwoba_ewma_alpha02", "xwoba_ewma_alpha05",
    "woba_stabilized", "woba_shrinkage_b",
    "k_rate_stabilized", "k_rate_shrinkage_b",
    "bb_rate_stabilized", "bb_rate_shrinkage_b",
    "babip_stabilized", "babip_shrinkage_b",
    "iso_stabilized", "iso_shrinkage_b",
]


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


def _stabilize(raw: float, n_pa: int, stat: str) -> tuple[float, float]:
    t = _STAB_T[stat]
    b = t / (t + max(n_pa, 0))
    return float(_LEAGUE_AVG[stat] + (1 - b) * (raw - _LEAGUE_AVG[stat])), float(b)


def compute_features(batter_id: int, pitcher_throws: str, silver: pl.DataFrame) -> np.ndarray:
    """
    Construye el vector de 33 features para un bateador vs un tipo de lanzador.

    Replica exactamente el pipeline de entrenamiento:
      1. _add_pa_event_flags (de features_rolling.py)
      2. _aggregate_to_daily  -> 1 fila por (batter_id, game_date)
      3. Rolling sobre ultimos N JUEGOS (no PAs) sin shift -> stats "as of last known game"
      4. EWMA sobre xwoba_mean diario
      5. Platoon stats (vs pitcher_throws) con shrinkage James-Stein
    """
    from src.features.features_rolling import _add_pa_event_flags, _aggregate_to_daily

    rows = silver.filter(pl.col("batter_id") == batter_id)
    if rows.is_empty():
        # No history: return league-average features so predictions are neutral
        league_defaults = {
            "babip_shrinkage_b": 1.0,   "babip_stabilized": _LEAGUE_AVG["babip"],
            "bb_rate_15d": _LEAGUE_AVG["bb_rate"],
            "bb_rate_30d": _LEAGUE_AVG["bb_rate"],
            "bb_rate_7d":  _LEAGUE_AVG["bb_rate"],
            "bb_rate_shrinkage_b": 1.0, "bb_rate_stabilized": _LEAGUE_AVG["bb_rate"],
            "hard_hit_rate_15d": 0.35,  "hard_hit_rate_30d": 0.35, "hard_hit_rate_7d": 0.35,
            "hr_rate_15d": 0.033,       "hr_rate_30d": 0.033,      "hr_rate_7d": 0.033,
            "iso_shrinkage_b": 1.0,     "iso_stabilized": _LEAGUE_AVG["iso"],
            "k_rate_15d": _LEAGUE_AVG["k_rate"],
            "k_rate_30d": _LEAGUE_AVG["k_rate"],
            "k_rate_7d":  _LEAGUE_AVG["k_rate"],
            "k_rate_shrinkage_b": 1.0,  "k_rate_stabilized": _LEAGUE_AVG["k_rate"],
            "launch_speed_15d": 88.0,   "launch_speed_30d": 88.0,  "launch_speed_7d": 88.0,
            "pa_15d": 60.0,             "pa_30d": 120.0,            "pa_7d": 28.0,
            "woba_shrinkage_b": 1.0,    "woba_stabilized": _LEAGUE_AVG["woba"],
            "xwoba_15d": 0.35,          "xwoba_30d": 0.35,          "xwoba_7d": 0.35,
            "xwoba_ewma_alpha02": 0.35, "xwoba_ewma_alpha05": 0.35,
        }
        return np.array([league_defaults[c] for c in FEATURE_COLS], dtype=np.float32)

    # ---- Game-level daily aggregation ----
    daily = _aggregate_to_daily(_add_pa_event_flags(rows.lazy())).collect()
    daily = daily.sort("game_date")

    def _pa_sum(n: int) -> float:
        return float(daily.tail(n)["pa"].sum())

    def _rate(n: int, k_col: str, denom_col: str = "pa") -> float:
        tail = daily.tail(n)
        k = float(tail[k_col].sum())
        d = float(tail[denom_col].sum())
        return k / d if d >= 1 else 0.0

    def _mean(n: int, col: str) -> float:
        vals = daily.tail(n)[col].drop_nulls()
        return float(vals.mean()) if len(vals) > 0 else 0.0

    def _gated(val: float, n: int, min_pa: int) -> float:
        return val if _pa_sum(n) >= min_pa else 0.0

    # ---- Rolling features (last N games, no anti-leakage shift for inference) ----
    k7  = _gated(_rate(7,  "k"),             7,  3)
    k15 = _gated(_rate(15, "k"),             15, 3)
    k30 = _gated(_rate(30, "k"),             30, 10)

    bb7  = _gated(_rate(7,  "bb"),            7,  3)
    bb15 = _gated(_rate(15, "bb"),            15, 3)
    bb30 = _gated(_rate(30, "bb"),            30, 10)

    hr7  = _gated(_rate(7,  "hr"),            7,  3)
    hr15 = _gated(_rate(15, "hr"),            15, 3)
    hr30 = _gated(_rate(30, "hr"),            30, 10)

    hh7  = _gated(_rate(7,  "hard_hits", "bip"), 7,  3)
    hh15 = _gated(_rate(15, "hard_hits", "bip"), 15, 3)
    hh30 = _gated(_rate(30, "hard_hits", "bip"), 30, 10)

    xw7  = _gated(_mean(7,  "xwoba_mean"),    7,  3)
    xw15 = _gated(_mean(15, "xwoba_mean"),    15, 3)
    xw30 = _gated(_mean(30, "xwoba_mean"),    30, 10)

    ls7  = _gated(_mean(7,  "launch_speed_mean"), 7,  3)
    ls15 = _gated(_mean(15, "launch_speed_mean"), 15, 3)
    ls30 = _gated(_mean(30, "launch_speed_mean"), 30, 10)

    pa7, pa15, pa30 = _pa_sum(7), _pa_sum(15), _pa_sum(30)

    # ---- EWMA over full career (no shift for inference) ----
    xw_series = daily["xwoba_mean"].drop_nulls().to_numpy().astype(np.float64)
    ewma02 = ewma05 = float(xw_series[0]) if len(xw_series) > 0 else _LEAGUE_AVG["woba"]
    for v in xw_series[1:]:
        ewma02 = 0.2 * v + 0.8 * ewma02
        ewma05 = 0.5 * v + 0.5 * ewma05

    # ---- Platoon features (vs tonight's pitcher hand) ----
    platoon_rows = rows.filter(pl.col("pitcher_throws") == pitcher_throws)
    if not platoon_rows.is_empty():
        daily_p = _aggregate_to_daily(
            _add_pa_event_flags(platoon_rows.lazy())
        ).collect()
        total_pa_p = float(daily_p["pa"].sum())
        k_raw      = float(daily_p["k"].sum())         / max(total_pa_p, 1)
        bb_raw     = float(daily_p["bb"].sum())        / max(total_pa_p, 1)
        hr_raw     = float(daily_p["hr"].sum())        / max(total_pa_p, 1)
        hits_p     = float(daily_p["hits"].sum())
        hr_p       = float(daily_p["hr"].sum())
        bip_p      = float(daily_p["bip"].sum())
        babip_raw  = (hits_p - hr_p) / max(bip_p, 1.0)
        xwoba_vals = platoon_rows["xwoba"].drop_nulls()
        woba_raw   = float(xwoba_vals.mean()) if len(xwoba_vals) > 0 else _LEAGUE_AVG["woba"]
        # ISO ≈ extra-base power (rough approximation)
        iso_raw    = max(woba_raw - (hits_p / max(total_pa_p, 1)) * 0.89, 0.0)
    else:
        total_pa_p = 0
        k_raw = bb_raw = iso_raw = 0.0
        babip_raw = _LEAGUE_AVG["babip"]
        woba_raw  = _LEAGUE_AVG["woba"]

    woba_stab,    woba_b    = _stabilize(woba_raw,    int(total_pa_p), "woba")
    k_rate_stab,  k_b       = _stabilize(k_raw,       int(total_pa_p), "k_rate")
    bb_rate_stab, bb_b      = _stabilize(bb_raw,      int(total_pa_p), "bb_rate")
    babip_stab,   babip_b   = _stabilize(babip_raw,   int(total_pa_p), "babip")
    iso_stab,     iso_b     = _stabilize(iso_raw,     int(total_pa_p), "iso")

    feat = {
        "babip_shrinkage_b":   babip_b,
        "babip_stabilized":    babip_stab,
        "bb_rate_15d":         bb15,
        "bb_rate_30d":         bb30,
        "bb_rate_7d":          bb7,
        "bb_rate_shrinkage_b": bb_b,
        "bb_rate_stabilized":  bb_rate_stab,
        "hard_hit_rate_15d":   hh15,
        "hard_hit_rate_30d":   hh30,
        "hard_hit_rate_7d":    hh7,
        "hr_rate_15d":         hr15,
        "hr_rate_30d":         hr30,
        "hr_rate_7d":          hr7,
        "iso_shrinkage_b":     iso_b,
        "iso_stabilized":      iso_stab,
        "k_rate_15d":          k15,
        "k_rate_30d":          k30,
        "k_rate_7d":           k7,
        "k_rate_shrinkage_b":  k_b,
        "k_rate_stabilized":   k_rate_stab,
        "launch_speed_15d":    ls15,
        "launch_speed_30d":    ls30,
        "launch_speed_7d":     ls7,
        "pa_15d":              pa15,
        "pa_30d":              pa30,
        "pa_7d":               pa7,
        "woba_shrinkage_b":    woba_b,
        "woba_stabilized":     woba_stab,
        "xwoba_15d":           xw15,
        "xwoba_30d":           xw30,
        "xwoba_7d":            xw7,
        "xwoba_ewma_alpha02":  ewma02,
        "xwoba_ewma_alpha05":  ewma05,
    }
    return np.array([feat[c] for c in FEATURE_COLS], dtype=np.float32)


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
# Helpers reutilizables
# ---------------------------------------------------------------------------

def _load_model() -> tuple:
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    predictor = _mat_module.AtBatPredictor(config=payload["config"])
    predictor._calibrated_model = payload["calibrated_model"]
    predictor._base_model       = payload["base_model"]
    predictor._feature_names    = payload["feature_names"]
    predictor._is_fitted        = True
    return predictor, len(payload["feature_names"])


def _predict_one_side(
    game: dict,
    side: str,
    silver: pl.DataFrame,
    predictor,
    game_date: str,
    verbose: bool = True,
) -> dict | None:
    """Calcula el orden optimo para un equipo en un partido.

    Devuelve un dict JSON-serializable, o None si no hay roster suficiente.
    """
    opp_side  = "home" if side == "away" else "away"
    team_info = game["teams"][side]["team"]
    away_name = game["teams"]["away"]["team"]["name"]
    home_name = game["teams"]["home"]["team"]["name"]
    pk        = game["gamePk"]

    opp_pitcher = (game["teams"][opp_side]
                   .get("probablePitcher", {})
                   .get("fullName", "Unknown"))
    opp_pitcher_throws = "R"

    lineup = _parse_lineup(game, side)
    if not lineup:
        lineup = fetch_roster(team_info["id"])
    if len(lineup) < 9:
        return None

    results = []
    for slot, player in enumerate(lineup, 1):
        pid  = player["id"]
        name = player["fullName"]
        X    = compute_features(pid, opp_pitcher_throws, silver)
        pv   = predictor.predict_proba(X.reshape(1, -1))[0]
        ev   = float(pv @ RUN_VALUES)

        woba_s = float(X[FEATURE_COLS.index("woba_stabilized")])
        iso_s  = float(X[FEATURE_COLS.index("iso_stabilized")])
        obp_e  = min(woba_s / 0.87 + 0.04, 0.450) if woba_s > 0.01 else 0.330
        in_hist = silver.filter(pl.col("batter_id") == pid).height > 0

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

    total_ev = sum(float(r["prob_vector"] @ RUN_VALUES) for r in ordered)
    avg_ev   = total_ev / 9

    if verbose:
        print(f"  E[R] suma 9 bateadores:        {total_ev:.4f}")
        print(f"  E[R/PA] promedio:              {avg_ev:.4f}")
        print(f"  E[R/partido] estimado (27 PA): {avg_ev * 27:.2f} carreras\n")

    abbr = team_info.get("abbreviation", team_info["name"].replace(" ", "")[:3]).upper()

    return {
        "game_date":   game_date,
        "game_pk":     pk,
        "matchup":     f"{away_name} @ {home_name}",
        "team":        team_info["name"],
        "team_abbr":   abbr,
        "side":        side,
        "opp_pitcher": opp_pitcher,
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
                "woba_stab": round(players9[i].woba, 4),
                "obp_est":   round(players9[i].obp, 4),
            }
            for s, i in enumerate(order_idx)
        ],
        "expected_runs_per_game": round(avg_ev * 27, 2),
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
                        help="Guardar resultado en JSON (ej. results/NYY_2026-05-19.json)")
    parser.add_argument("--all",     action="store_true", dest="all_games",
                        help="Procesar todos los partidos de la fecha (ambos equipos por juego)")
    parser.add_argument("--output-dir", default="results", dest="output_dir",
                        help="Carpeta de salida con --all (default: results/)")
    args = parser.parse_args()

    game_date = args.date
    print(f"\nCargando partidos del {game_date}...")
    games = fetch_games(game_date)
    if not games:
        print(f"  No se encontraron partidos para {game_date}.")
        return
    print(f"  {len(games)} partidos encontrados")

    print("\nCargando historico Silver (2021-2024)...")
    silver = _load_silver()
    print(f"  {len(silver):,} PAs  |  {silver['batter_id'].n_unique():,} bateadores unicos")

    print("\nCargando modelo AtBatPredictor...")
    predictor, n_feat = _load_model()
    print(f"  Cargado | {n_feat} features\n")

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

            for side in ("away", "home"):
                team = game["teams"][side]["team"]
                abbr = team.get("abbreviation", team["name"].replace(" ", "")[:3]).upper()
                print(f"\n  [{side.upper()}] {team['name']}")
                result = _predict_one_side(game, side, silver, predictor, game_date, verbose=True)
                if result is None:
                    print(f"  Roster insuficiente — omitido.")
                    skipped.append(f"{abbr} ({side})")
                    continue
                fname = out_dir / f"{abbr}_{game_date}.json"
                fname.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                saved.append(str(fname))
                print(f"  Guardado: {fname}")

        print(f"\n{'='*66}")
        print(f"  Archivos guardados : {len(saved)}")
        if skipped:
            print(f"  Omitidos           : {', '.join(skipped)}")
        print(f"  Carpeta            : {Path(args.output_dir).resolve()}\n")
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

    result = _predict_one_side(game, args.side, silver, predictor, game_date, verbose=True)
    if result is None:
        print("  No se pudo procesar: roster insuficiente.")
        return

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Resultados guardados en: {out_path}\n")


if __name__ == "__main__":
    main()
