"""
backtest.py — Motor de validación out-of-sample (Roadmap 0.2)
=============================================================
Lee todas las predicciones de results/*.json, obtiene los resultados reales
de la MLB Stats API o de reports/comparison/*.json y calcula:

  - Log-Loss vs baseline (siempre 50% y Elo)
  - Brier Score vs benchmark
  - AUC ROC
  - Diagrama de calibración (bins de probabilidad vs frecuencia observada)
  - ECE (Expected Calibration Error)

Uso:
    python backtest.py                      # Todas las fechas disponibles
    python backtest.py --from 2026-05-01    # Desde una fecha
    python backtest.py --date 2026-05-26    # Un día concreto
    python backtest.py --out reports/backtest/backtest_results.json

Salida: JSON inmutable con timestamp, inputs y métricas.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parent
RESULTS_DIR   = ROOT / "results"
REPORTS_DIR   = ROOT / "reports"
COMPARISON_DIR = REPORTS_DIR / "comparison"

MLB_API = "https://statsapi.mlb.com/api/v1"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_get(url: str, params: dict | None = None) -> dict:
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARN] MLB API error for {url}: {e}")
        return {}


def _actual_runs(game_pk: int) -> tuple[int, int] | None:
    """Get (home_runs, away_runs) for a finished game. Returns None if unavailable."""
    data = _safe_get(f"{MLB_API}/game/{game_pk}/linescore")
    if not data:
        return None
    teams = data.get("teams", {})
    hr = teams.get("home", {}).get("runs")
    ar = teams.get("away", {}).get("runs")
    if hr is None or ar is None:
        return None
    return int(hr), int(ar)


def _load_comparison_actuals() -> dict[int, tuple[int, int]]:
    """Load all game results from comparison JSON files. Faster than hitting MLB API."""
    actuals: dict[int, tuple[int, int]] = {}
    for cf in COMPARISON_DIR.glob("comparison_*.json"):
        try:
            data = json.loads(cf.read_text(encoding="utf-8"))
            for g in data.get("games", []):
                gpk = g.get("game_pk")
                act = g.get("actual", {})
                hr = act.get("home_runs")
                ar = act.get("away_runs")
                if gpk and hr is not None and ar is not None:
                    actuals[int(gpk)] = (int(hr), int(ar))
        except Exception:
            pass
    return actuals


# ── Metrics ──────────────────────────────────────────────────────────────────

def log_loss(predictions: list[tuple[float, int]]) -> float:
    """predictions: list of (win_prob, actual_home_won) tuples."""
    if not predictions:
        return float("nan")
    total = 0.0
    for p, y in predictions:
        p = max(1e-7, min(1 - 1e-7, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return round(total / len(predictions), 5)


def brier_score(predictions: list[tuple[float, int]]) -> float:
    if not predictions:
        return float("nan")
    total = sum((p - y) ** 2 for p, y in predictions)
    return round(total / len(predictions), 5)


def auc_roc(predictions: list[tuple[float, int]]) -> float:
    """Wilcoxon-Mann-Whitney estimator of AUC."""
    pos = [p for p, y in predictions if y == 1]
    neg = [p for p, y in predictions if y == 0]
    if not pos or not neg:
        return float("nan")
    concordant = sum(1 for pp in pos for pn in neg if pp > pn)
    ties = sum(0.5 for pp in pos for pn in neg if pp == pn)
    total = len(pos) * len(neg)
    return round((concordant + ties) / total, 5)


def calibration_bins(
    predictions: list[tuple[float, int]], n_bins: int = 10
) -> list[dict]:
    """Return reliability diagram data."""
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in predictions:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))

    result = []
    for i, items in enumerate(bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if not items:
            result.append({
                "bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                "mean_predicted": round((lo + hi) / 2, 3),
                "observed_freq": None,
                "count": 0,
            })
        else:
            mean_pred = sum(p for p, _ in items) / len(items)
            obs_freq = sum(y for _, y in items) / len(items)
            result.append({
                "bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                "mean_predicted": round(mean_pred, 3),
                "observed_freq": round(obs_freq, 3),
                "count": len(items),
            })
    return result


def ece(bins_data: list[dict]) -> float:
    """Expected Calibration Error."""
    total = sum(b["count"] for b in bins_data)
    if not total:
        return float("nan")
    weighted = sum(
        b["count"] * abs(b["mean_predicted"] - b["observed_freq"])
        for b in bins_data
        if b["observed_freq"] is not None
    )
    return round(weighted / total, 5)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(
    date_from: str | None = None,
    date_single: str | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run full out-of-sample backtest and return results dict."""

    print("=== MLB Backtest Engine (Roadmap 0.2) ===")
    print("Loading comparison actuals from reports/comparison/...")
    actuals_cache = _load_comparison_actuals()
    print(f"  Loaded {len(actuals_cache)} game results from comparison files.")

    # Collect all prediction files: estructura plana legacy (ABBR_fecha.json)
    # + estructura nueva por subdirectorio (results/<fecha>/<ABBR>.json)
    all_files = sorted(RESULTS_DIR.glob("*_2*.json")) + sorted(RESULTS_DIR.glob("2*/*.json"))
    print(f"  Found {len(all_files)} prediction files.")

    def _file_date(fp: Path) -> str:
        if fp.parent != RESULTS_DIR:
            return fp.parent.name          # results/<fecha>/<ABBR>.json
        return fp.stem.split("_")[-1]      # results/<ABBR>_<fecha>.json

    # Filter by date if requested
    if date_single:
        all_files = [f for f in all_files if _file_date(f) == date_single]
    elif date_from:
        all_files = [f for f in all_files if _file_date(f) >= date_from]

    # Group by game_pk (one prediction per team per game — deduplicate to home side)
    game_preds: dict[int, dict] = {}
    for fp in all_files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
            gpk = int(d.get("game_pk", 0))
            if not gpk:
                continue
            side = d.get("side", "")
            wp = d.get("win_probability") or d.get("expected_runs_per_game")
            if not wp:
                continue
            # Prefer home side for win_probability
            if gpk not in game_preds or side == "home":
                game_preds[gpk] = {
                    "game_pk": gpk,
                    "game_date": d.get("game_date", ""),
                    "team": d.get("team_abbr", ""),
                    "side": side,
                    "win_probability": float(d.get("win_probability", 0.5)),
                    "expected_runs": float(d.get("expected_runs_per_game", 4.5)),
                    "file": str(fp.name),
                }
        except Exception as e:
            print(f"  [WARN] Could not parse {fp.name}: {e}")

    print(f"  Deduplicated to {len(game_preds)} unique games.")

    # Resolve actuals
    predictions_with_outcome: list[tuple[float, int]] = []
    game_records: list[dict] = []
    api_calls = 0

    for gpk, pred in sorted(game_preds.items()):
        # Try cache first
        if gpk in actuals_cache:
            hr, ar = actuals_cache[gpk]
        else:
            # Fall back to MLB API
            api_calls += 1
            result = _actual_runs(gpk)
            if result is None:
                print(f"  [SKIP] game_pk={gpk}: no actual result available.")
                continue
            hr, ar = result
            actuals_cache[gpk] = (hr, ar)

        home_won = 1 if hr > ar else 0
        wp = pred["win_probability"]
        er = pred["expected_runs"]

        predictions_with_outcome.append((wp, home_won))
        game_records.append({
            "game_pk": gpk,
            "game_date": pred["game_date"],
            "team": pred["team"],
            "side": pred["side"],
            "win_probability": wp,
            "expected_runs": er,
            "actual_home_runs": hr,
            "actual_away_runs": ar,
            "home_won": home_won,
            "delta_er": round(hr - er, 2),
        })

    n = len(predictions_with_outcome)
    print(f"  Resolved actuals for {n} games ({api_calls} MLB API calls).")

    if n == 0:
        print("  [ERROR] No games with actuals found. Cannot compute metrics.")
        return {"error": "No games with resolved actuals", "n_games": 0}

    # Compute metrics
    ll_model    = log_loss(predictions_with_outcome)
    ll_baseline = log_loss([(0.5, y) for _, y in predictions_with_outcome])
    ll_elo      = 0.431  # Typical Elo baseline for MLB (from literature)

    bs_model     = brier_score(predictions_with_outcome)
    bs_benchmark = 0.228  # MLB benchmark from literature

    auc = auc_roc(predictions_with_outcome)
    bins = calibration_bins(predictions_with_outcome)
    ece_val = ece(bins)

    # Percentage of games where model "beats" 50/50 baseline (directional)
    correct_direction = sum(
        1 for p, y in predictions_with_outcome
        if (p > 0.5 and y == 1) or (p < 0.5 and y == 0)
    )
    accuracy_pct = round(correct_direction / n * 100, 1) if n > 0 else 0.0

    print(f"\n  Results ({n} games):")
    print(f"    Log-Loss model:    {ll_model:.4f}  (baseline 50%: {ll_baseline:.4f}, Elo: {ll_elo:.4f})")
    print(f"    Brier Score:       {bs_model:.4f}  (benchmark: {bs_benchmark:.4f})")
    print(f"    AUC ROC:           {auc:.4f}")
    print(f"    ECE:               {ece_val:.4f}")
    print(f"    Accuracy (dir.):   {accuracy_pct}%")
    print(f"    Beats 50% baseline: {ll_model < ll_baseline}")
    print(f"    Beats Elo baseline: {ll_model < ll_elo}")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_games": n,
        "date_range": {
            "from": min(r["game_date"] for r in game_records) if game_records else "",
            "to":   max(r["game_date"] for r in game_records) if game_records else "",
        },
        "metrics": {
            "log_loss": ll_model,
            "log_loss_baseline_50pct": ll_baseline,
            "log_loss_baseline_elo": ll_elo,
            "log_loss_delta_vs_50pct": round(ll_baseline - ll_model, 5),
            "log_loss_delta_vs_elo": round(ll_elo - ll_model, 5),
            "brier_score": bs_model,
            "brier_benchmark": bs_benchmark,
            "brier_delta": round(bs_benchmark - bs_model, 5),
            "auc_roc": auc,
            "ece": ece_val,
            "accuracy_pct": accuracy_pct,
            "beats_50pct_baseline": ll_model < ll_baseline,
            "beats_elo_baseline": ll_model < ll_elo,
        },
        "calibration_bins": bins,
        "games": game_records,
    }

    # Save output
    out = output_path or (REPORTS_DIR / "backtest" / "backtest_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Saved → {out}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Backtest Engine")
    parser.add_argument("--from", dest="date_from", help="Start date YYYY-MM-DD")
    parser.add_argument("--date", dest="date_single", help="Single date YYYY-MM-DD")
    parser.add_argument("--out",  help="Output JSON path")
    args = parser.parse_args()

    run_backtest(
        date_from=args.date_from,
        date_single=args.date_single,
        output_path=Path(args.out) if args.out else None,
    )
