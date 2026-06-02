"""
morning.py — Rutina diaria MLB Optimizer
=========================================
Ejecutar CADA MAÑANA antes de los partidos:

    python morning.py

Flujo automático:
  1. Reporte post-partido de AYER  (predicciones vs. resultado real)
  2. Lista de partidos de HOY con pitchers probables
  3. Predicciones para TODOS los partidos de hoy

Opciones:
    python morning.py --no-predict          # Solo reporte + schedule
    python morning.py --date 2026-05-20     # Tratar esa fecha como "hoy"

Para prediccion de un equipo especifico (cuando ya hay lineup oficial):
    python predict_tonight.py --team NYY
    python predict_tonight.py --team LAD --side home
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT         = Path(__file__).parent
RESULTS_DIR  = ROOT / "results"
REPORTS_DIR  = ROOT / "reports" / "comparison"
MLB_API      = "https://statsapi.mlb.com/api/v1"

SEP  = "=" * 62
SEP2 = "-" * 62


# ---------------------------------------------------------------------------
# File-path helpers (new date-subdir structure, backward-compatible)
# ---------------------------------------------------------------------------

def _pred_files(results_dir: Path, game_date: str) -> list[Path]:
    """Returns prediction JSONs for a date. Checks new date-subdir structure first,
    falls back to old flat structure for backward compatibility."""
    date_dir = results_dir / game_date
    if date_dir.is_dir():
        return sorted(date_dir.glob("*.json"))
    return sorted(results_dir.glob(f"*_{game_date}.json"))


def _comparison_path(reports_dir: Path, game_date: str) -> Path:
    """Returns the path where the comparison report for a date should be saved
    (new date-subdir structure)."""
    date_dir = reports_dir / game_date
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir / "comparison.json"


# ---------------------------------------------------------------------------
# MLB API helper
# ---------------------------------------------------------------------------

def _api_get(path: str, params: dict | None = None) -> dict:
    resp = requests.get(f"{MLB_API}{path}", params=params or {}, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# 1. Reporte post-partido
# ---------------------------------------------------------------------------

def post_game_report(game_date: str) -> None:
    print(f"\n{SEP}")
    print(f"  REPORTE POST-PARTIDO - {game_date}")
    print(SEP)

    try:
        data = _api_get("/schedule", {
            "sportId": 1,
            "date": game_date,
            "hydrate": "linescore,team",
        })
    except requests.RequestException as exc:
        print(f"  [ERROR] No se pudo conectar con la MLB API: {exc}")
        return

    games = [g for d in data.get("dates", []) for g in d.get("games", [])]
    if not games:
        print(f"  No hay partidos registrados para {game_date}.")
        return

    # Pre-load all prediction JSONs for this date
    pred_by_pk: dict[int, list[dict]] = {}
    for pf in _pred_files(RESULTS_DIR, game_date):
        try:
            p = json.loads(pf.read_text(encoding="utf-8"))
            pk = p.get("game_pk")
            if pk:
                pred_by_pk.setdefault(pk, []).append(p)
        except Exception:
            pass

    report_games: list[dict] = []

    for game in games:
        pk        = game["gamePk"]
        away_name = game["teams"]["away"]["team"]["name"]
        home_name = game["teams"]["home"]["team"]["name"]
        status    = game.get("status", {}).get("detailedState", "Unknown")

        ls        = game.get("linescore", {})
        ls_teams  = ls.get("teams", {})
        away_runs = ls_teams.get("away", {}).get("runs")
        home_runs = ls_teams.get("home", {}).get("runs")

        print(f"\n  {away_name} @ {home_name}  (pk={pk})")
        print(f"  Estado : {status}")

        if away_runs is not None or home_runs is not None:
            print(f"  Resultado real :  {away_name} {away_runs} - {home_name} {home_runs}")
        else:
            print(f"  Resultado real :  No disponible aún")

        preds = pred_by_pk.get(pk, [])
        pred_runs: dict[str, float] = {}
        pred_details: list[dict] = []

        if preds:
            for p in preds:
                side   = p.get("side", "?")
                team   = p.get("team", "?")
                exp_r  = p.get("expected_runs_per_game", 0)
                roster = len(p.get("batting_order", []))
                fip_ctx = p.get("opp_pitcher_stats", {})
                fip_str = ""
                if fip_ctx and not fip_ctx.get("fip_is_estimated"):
                    fip_str = f"  FIP={fip_ctx['fip']:.2f} K/9={fip_ctx['k9']:.1f}"
                print(f"  Prediccion ({side}) {team}: E[R/partido] = {exp_r}  ({roster} bateadores){fip_str}")
                pred_runs[team] = exp_r
                pred_details.append({"team": team, "side": side, "expected_runs_per_game": exp_r})
        else:
            print(f"  Prediccion : Sin archivo en results/ para este partido")

        # Winner comparison — works with 1 or 2 predictions
        comparison: dict = {}
        if away_runs is not None and home_runs is not None and pred_runs:
            try:
                real_winner = away_name if int(away_runs) > int(home_runs) else home_name
                actual_runs = {away_name: int(away_runs), home_name: int(home_runs)}
                pred_winner = max(pred_runs, key=pred_runs.__getitem__)
                if pred_winner in (away_name, home_name):
                    correct = pred_winner == real_winner
                    comparison = {
                        "predicted_winner": pred_winner,
                        "actual_winner":    real_winner,
                        "correct":          correct,
                        "actual_runs":      actual_runs,
                    }
                    mark = "[OK]" if correct else "[X]"
                    print(f"  Ganador predicho: {pred_winner} -> Real: {real_winner} {mark}")
            except TypeError:
                pass

        report_games.append({
            "game_pk":    pk,
            "away_team":  away_name,
            "home_team":  home_name,
            "status":     status,
            "actual": {
                "away_runs": away_runs,
                "home_runs": home_runs,
            },
            "predictions": pred_details,
            "comparison":  comparison,
        })

    print()

    # Summary stats
    with_preds   = [g for g in report_games if g["predictions"]]
    with_cmp     = [g for g in report_games if g["comparison"]]
    correct      = [g for g in with_cmp if g["comparison"].get("correct")]
    accuracy     = round(len(correct) / len(with_cmp), 3) if with_cmp else None

    print(f"  Resumen: {len(games)} partidos | {len(with_preds)} con prediccion | "
          f"{len(correct)}/{len(with_cmp)} ganadores correctos"
          + (f" ({accuracy:.1%})" if accuracy is not None else ""))
    print()

    report = {
        "report_date": str(date.today()),
        "game_date":   game_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "games": report_games,
        "summary": {
            "total_games":            len(games),
            "games_with_predictions": len(with_preds),
            "games_with_comparison":  len(with_cmp),
            "correct_winner":         len(correct),
            "incorrect_winner":       len(with_cmp) - len(correct),
            "accuracy":               accuracy,
        },
    }

    out_path = _comparison_path(REPORTS_DIR, game_date)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Informe guardado en: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# 2. Schedule de hoy
# ---------------------------------------------------------------------------

def show_schedule(game_date: str) -> list[dict]:
    print(f"{SEP}")
    print(f"  PARTIDOS HOY — {game_date}")
    print(SEP)

    try:
        data = _api_get("/schedule", {
            "sportId": 1,
            "date": game_date,
            "hydrate": "team,probablePitcher",
        })
    except requests.RequestException as exc:
        print(f"  [ERROR] No se pudo conectar con la MLB API: {exc}")
        return []

    games = [g for d in data.get("dates", []) for g in d.get("games", [])]
    if not games:
        print(f"  No hay partidos programados para {game_date}.\n")
        return []

    for i, game in enumerate(games):
        away = game["teams"]["away"]["team"]["name"]
        home = game["teams"]["home"]["team"]["name"]
        t    = game.get("gameDate", "")[:16].replace("T", " ")
        ap   = game["teams"]["away"].get("probablePitcher", {}).get("fullName", "TBD")
        hp   = game["teams"]["home"].get("probablePitcher", {}).get("fullName", "TBD")
        print(f"  [{i:>2}] {away:<28} @  {home}")
        print(f"       {t} UTC  |  P: {ap} vs {hp}")
    print()
    return games


# ---------------------------------------------------------------------------
# 2b. Feature drift check (semanal)
# ---------------------------------------------------------------------------

def _run_weekly_drift_check() -> None:
    """Runs PSI drift monitor on Mondays or if no report exists from this week."""
    today_dt   = date.today()
    drift_dir  = ROOT / "reports" / "drift"
    drift_dir.mkdir(parents=True, exist_ok=True)

    # Check if a drift report already exists this week (Mon–Sun)
    week_start = today_dt - timedelta(days=today_dt.weekday())
    existing   = list(drift_dir.glob(f"drift_{week_start}*.json"))
    if existing and today_dt.weekday() != 0:
        return  # Not Monday and a report already exists this week

    print(f"{SEP}")
    print("  CHEQUEO SEMANAL DE FEATURE DRIFT")
    print(SEP)
    try:
        from src.mlops.feature_drift_monitor import DriftMonitor
        monitor = DriftMonitor(silver_dir=str(ROOT / "data" / "silver" / "plate_appearances"))
        report  = monitor.run(recent_days=30)  # reference_seasons dinamico: [year-2, year-1]
        monitor.print_summary(report)
        out = drift_dir / f"drift_{today_dt}.json"
        out.write_text(
            __import__("json").dumps(monitor.to_json(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Reporte drift guardado: {out.resolve()}\n")
    except Exception as exc:
        print(f"  [AVISO] Drift monitor no ejecutado: {exc}\n")


# ---------------------------------------------------------------------------
# 2c. Auto-calibración de _MC_RUNS_SCALE (semanal, los lunes)
# ---------------------------------------------------------------------------

def _recalibrate_mc_scale() -> None:
    """Recalcula _MC_RUNS_SCALE leyendo todos los comparison JSONs acumulados.

    Solo actúa los lunes y solo si hay >= 30 observaciones (equipo-juego).
    Actualiza la constante en predict_tonight.py si el nuevo scale difiere
    más de 0.02 del actual. Protección: el scale nunca sale de [0.60, 0.95].
    """
    today_dt = date.today()
    if today_dt.weekday() != 0:   # solo lunes
        return

    print(f"{SEP}")
    print("  AUTO-CALIBRACION _MC_RUNS_SCALE")
    print(SEP)

    comp_dir = ROOT / "reports" / "comparison"
    all_comp  = list(comp_dir.glob("*/comparison.json")) + list(comp_dir.glob("comparison_*.json"))

    predicted, actual = [], []
    for cp in all_comp:
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            for g in data.get("games", []):
                act = g.get("actual", {})
                ar  = act.get("away_runs")
                hr  = act.get("home_runs")
                if ar is None or hr is None:
                    continue
                for pred in g.get("predictions", []):
                    er = pred.get("expected_runs_per_game")
                    if er is None:
                        continue
                    side  = pred.get("side", "")
                    real  = ar if side == "away" else hr
                    predicted.append(float(er))
                    actual.append(float(real))
        except Exception:
            pass

    n = len(predicted)
    if n < 30:
        print(f"  Insuficientes observaciones: {n} (necesarias >= 30). Saltando.\n")
        return

    import numpy as np

    # Leer scale actual de predict_tonight.py
    pt_path = ROOT / "predict_tonight.py"
    src     = pt_path.read_text(encoding="utf-8")
    import re
    m = re.search(r"_MC_RUNS_SCALE:\s*float\s*=\s*([\d.]+)", src)
    current_scale = float(m.group(1)) if m else 0.768

    # raw MC = predicho / scale_actual
    raw_mc_mean  = float(np.mean(predicted)) / current_scale
    actual_mean  = float(np.mean(actual))
    new_scale    = actual_mean / raw_mc_mean
    new_scale    = float(max(0.60, min(0.95, new_scale)))   # cap de seguridad

    print(f"  N observaciones  : {n}")
    print(f"  E[R] predicho    : {np.mean(predicted):.3f}  (raw MC: {raw_mc_mean:.3f})")
    print(f"  E[R] real        : {actual_mean:.3f}")
    print(f"  Scale actual     : {current_scale:.4f}")
    print(f"  Nuevo scale calc : {new_scale:.4f}")

    if abs(new_scale - current_scale) < 0.02:
        print(f"  Diferencia < 0.02 — sin cambios.\n")
        return

    # Actualizar la línea en predict_tonight.py
    new_src = re.sub(
        r"(_MC_RUNS_SCALE:\s*float\s*=\s*)[\d.]+",
        lambda _m: f"{_m.group(1)}{new_scale:.4f}",
        src,
    )
    pt_path.write_text(new_src, encoding="utf-8")
    print(f"  _MC_RUNS_SCALE actualizado: {current_scale:.4f} -> {new_scale:.4f} "
          f"(guardado en predict_tonight.py)\n")


# ---------------------------------------------------------------------------
# 3. Predicciones (delega a predict_tonight.py)
# ---------------------------------------------------------------------------

def run_predictions(game_date: str) -> None:
    print(f"{SEP}")
    print(f"  PREDICCIONES — {game_date}")
    print(SEP)
    print(f"  Ejecutando predict_tonight.py --all --date {game_date} ...\n")

    script = ROOT / "predict_tonight.py"
    result = subprocess.run(
        [sys.executable, str(script), "--all", "--date", game_date,
         "--output-dir", str(RESULTS_DIR)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n  [AVISO] predict_tonight.py terminó con código {result.returncode}.")
        print( "  Revisa el error arriba o ejecuta manualmente.")
    else:
        print(f"\n  Predicciones guardadas en: {RESULTS_DIR.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rutina diaria MLB Optimizer")
    parser.add_argument("--date",       default=str(date.today()),
                        help="Fecha de HOY en formato YYYY-MM-DD (default: hoy)")
    parser.add_argument("--no-predict", action="store_true",
                        help="Saltar la generacion de predicciones")
    args = parser.parse_args()

    today     = args.date
    yesterday = str(date.fromisoformat(today) - timedelta(days=1))

    print(f"\n{'MLB OPTIMIZER -- RUTINA DIARIA':^62}")
    print(f"{'Hoy: ' + today + ' | Ayer: ' + yesterday:^62}\n")

    yesterday_preds = _pred_files(RESULTS_DIR, yesterday)
    if yesterday_preds:
        post_game_report(yesterday)
    else:
        print(f"{SEP}")
        print(f"  REPORTE POST-PARTIDO - {yesterday}")
        print(SEP)
        print(f"  Sin predicciones guardadas para {yesterday}. Saltando reporte.\n")

    show_schedule(today)

    # Weekly feature drift check (every Monday or when no check ran this week)
    _run_weekly_drift_check()

    # Weekly MC scale recalibration (every Monday, requires >= 30 observations)
    _recalibrate_mc_scale()

    if not args.no_predict:
        run_predictions(today)
    else:
        print("  (Predicciones omitidas con --no-predict)")
        print(f"  Para predecir manualmente: python predict_tonight.py --all --date {today}\n")


if __name__ == "__main__":
    main()
