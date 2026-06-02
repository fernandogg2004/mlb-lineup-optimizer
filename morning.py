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
    for pf in RESULTS_DIR.glob(f"*_{game_date}.json"):
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
                print(f"  Prediccion ({side}) {team}: E[R/partido] = {exp_r}  ({roster} bateadores)")
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

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"comparison_{game_date}.json"
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
        report  = monitor.run(reference_seasons=[2022, 2023], recent_days=30)
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

    yesterday_preds = list(RESULTS_DIR.glob(f"*_{yesterday}.json"))
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

    if not args.no_predict:
        run_predictions(today)
    else:
        print("  (Predicciones omitidas con --no-predict)")
        print(f"  Para predecir manualmente: python predict_tonight.py --all --date {today}\n")


if __name__ == "__main__":
    main()
