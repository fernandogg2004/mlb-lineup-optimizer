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
    # Los ficheros reales viven en reports/comparison/<fecha>/comparison.json
    # (estructura por subdirectorio de fecha), no en comparison_*.json plano.
    # El patrón anterior cargaba 0 ficheros y forzaba a resolver TODOS los
    # actuals contra la MLB API (lento y con carga de red innecesaria).
    for cf in COMPARISON_DIR.glob("*/comparison.json"):
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


def _bootstrap_ci(
    predictions: list[tuple[float, int]],
    metric_fn,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.10,
) -> dict:
    """Intervalo de confianza por bootstrap (resampling con reemplazo) de una métrica.

    Con ~200 juegos los deltas de Log-Loss entre el modelo y un baseline 50/50
    son minúsculos; sin IC no se puede afirmar si el modelo es realmente mejor,
    peor o estadísticamente indistinguible. Esto último es el desenlace honesto
    más probable, y el IC lo demuestra.

    Args:
        predictions: Lista de (prob, outcome).
        metric_fn: Función métrica que toma la lista y devuelve un float.
        n_boot: Número de remuestras bootstrap.
        seed: Semilla para reproducibilidad.
        alpha: Nivel de significancia (0.10 -> IC del 90%).

    Returns:
        Dict con ``point``, ``ci_low``, ``ci_high`` (percentiles del bootstrap).
    """
    import random as _random

    n = len(predictions)
    point = metric_fn(predictions)
    if n == 0:
        return {"point": point, "ci_low": None, "ci_high": None}
    rng = _random.Random(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        resampled = [predictions[rng.randrange(n)] for _ in range(n)]
        val = metric_fn(resampled)
        if not (isinstance(val, float) and math.isnan(val)):
            samples.append(val)
    samples.sort()
    lo_idx = int((alpha / 2) * len(samples))
    hi_idx = min(len(samples) - 1, int((1 - alpha / 2) * len(samples)))
    return {
        "point": point,
        "ci_low": round(samples[lo_idx], 5),
        "ci_high": round(samples[hi_idx], 5),
    }


def _paired_bootstrap_vs_baseline(
    predictions: list[tuple[float, int]],
    baseline_probs: list[float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap pareado del delta de Log-Loss modelo vs baseline (mismos juegos).

    El pareado es más potente que comparar dos IC marginales: remuestrea los
    índices de juego una vez y evalúa ambas log-losses sobre la MISMA remuestra.

    Args:
        predictions: Lista de (prob_modelo, outcome).
        baseline_probs: Probabilidades del baseline alineadas por índice.
        n_boot: Número de remuestras.
        seed: Semilla.

    Returns:
        Dict con ``delta_logloss`` (baseline - modelo; >0 = modelo mejor),
        IC del 90% del delta y ``prob_model_better`` (fracción de remuestras
        donde el modelo gana).
    """
    import random as _random

    n = len(predictions)
    if n == 0:
        return {"delta_logloss": None, "ci_low": None, "ci_high": None, "prob_model_better": None}
    base = [(baseline_probs[i], predictions[i][1]) for i in range(n)]
    point = log_loss(base) - log_loss(predictions)
    rng = _random.Random(seed)
    deltas: list[float] = []
    wins = 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        m = [predictions[i] for i in idx]
        b = [base[i] for i in idx]
        d = log_loss(b) - log_loss(m)
        deltas.append(d)
        if d > 0:
            wins += 1
    deltas.sort()
    return {
        "delta_logloss": round(point, 5),
        "ci_low": round(deltas[int(0.05 * len(deltas))], 5),
        "ci_high": round(deltas[min(len(deltas) - 1, int(0.95 * len(deltas)))], 5),
        "prob_model_better": round(wins / n_boot, 3),
    }


def _sim_vs_reality(game_records: list[dict]) -> dict:
    """Compara E[R] simulado del local vs carreras reales del local (sesgo del sim).

    El PA model puede estar perfectamente calibrado en E[R/PA] y aun así el
    simulador inflar/deflactar las carreras al encadenar base-outs. Esta sección
    aísla ese sesgo, que es independiente del PA model.

    Args:
        game_records: Lista de registros con ``expected_runs`` y
            ``actual_home_runs``.

    Returns:
        Dict con sesgo medio, RMSE y correlación de Pearson E[R]-real.
    """
    pred = [r["expected_runs"] for r in game_records if r.get("expected_runs") is not None]
    actual = [r["actual_home_runs"] for r in game_records if r.get("expected_runs") is not None]
    n = len(pred)
    if n == 0:
        return {"n": 0}
    mp = sum(pred) / n
    ma = sum(actual) / n
    bias = mp - ma
    rmse = (sum((p - a) ** 2 for p, a in zip(pred, actual)) / n) ** 0.5
    # Pearson r
    cov = sum((p - mp) * (a - ma) for p, a in zip(pred, actual)) / n
    sp = (sum((p - mp) ** 2 for p in pred) / n) ** 0.5
    sa = (sum((a - ma) ** 2 for a in actual) / n) ** 0.5
    r = cov / (sp * sa) if sp > 0 and sa > 0 else float("nan")
    return {
        "n": n,
        "mean_expected_runs": round(mp, 3),
        "mean_actual_runs": round(ma, 3),
        "bias_runs": round(bias, 3),
        "rmse_runs": round(rmse, 3),
        "pearson_r": round(r, 4),
    }


def _winprob_shape(predictions: list[tuple[float, int]]) -> dict:
    """Forma de la distribución de win-prob: centrado y dispersión.

    Detecta el problema de centrado observado en Fase 0: media ~0.50 pero sólo
    una minoría de predicciones por encima de 0.5 (distribución sesgada), lo que
    arruina cualquier decisión con umbral 0.5.

    Args:
        predictions: Lista de (prob, outcome).

    Returns:
        Dict con media, std, mediana y fracción por encima de 0.5.
    """
    n = len(predictions)
    if n == 0:
        return {"n": 0}
    ps = sorted(p for p, _ in predictions)
    mean = sum(ps) / n
    std = (sum((p - mean) ** 2 for p in ps) / n) ** 0.5
    median = ps[n // 2]
    frac_above = sum(1 for p in ps if p > 0.5) / n
    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(median, 4),
        "frac_above_0.5": round(frac_above, 4),
        "min": round(ps[0], 4),
        "max": round(ps[-1], 4),
    }


def elo_baseline_predictions(
    elo_games: list[tuple[str, str, str, int]],
    *,
    k: float = 20.0,
    home_adv: float = 24.0,
    base_rating: float = 1500.0,
) -> list[tuple[float, int]]:
    """Elo REAL calculado sobre los MISMOS juegos (audit F07).

    Antes el backtest comparaba contra una constante 0.431 'de literatura', lo
    que hacía el claim 'beats Elo' no riguroso. Aquí se entrena un Elo estándar
    en orden cronológico: todos los equipos empiezan en ``base_rating``, con
    ventaja de local de ``home_adv`` puntos Elo y factor de actualización ``k``.
    Se devuelve la predicción P(gana local) ANTES de cada juego (out-of-sample
    secuencial, sin leakage) emparejada con el resultado real.

    Args:
        elo_games: lista de (game_date, home_name, away_name, home_won) ya
            resoluble; se ordena por fecha internamente.
        k: factor de actualización Elo.
        home_adv: ventaja del equipo local en puntos Elo.
        base_rating: rating inicial de cada equipo.

    Returns:
        Lista de (p_home_win, home_won) en el orden cronológico de los juegos.
    """
    ratings: dict[str, float] = {}
    preds: list[tuple[float, int]] = []
    for _date, home, away, home_won in sorted(elo_games, key=lambda g: g[0]):
        rh = ratings.get(home, base_rating)
        ra = ratings.get(away, base_rating)
        # Predicción ANTES del juego (no usa el resultado)
        exp_home = 1.0 / (1.0 + 10 ** (-((rh + home_adv) - ra) / 400.0))
        preds.append((exp_home, home_won))
        # Actualización tras observar el resultado
        ratings[home] = rh + k * (home_won - exp_home)
        ratings[away] = ra + k * ((1 - home_won) - (1 - exp_home))
    return preds


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

    # Agrupar por game_pk recogiendo AMBOS lados. Hallazgo Fase 0: ~52% de los
    # juegos tenían el archivo de un lado sin simular (win_probability ausente),
    # y el dedup anterior lo rellenaba con 0.5 por defecto. Esas predicciones
    # degeneradas contaminaban TODAS las métricas (accuracy direccional 24.9%,
    # win-prob aparentemente descentrado). Aquí se exige una predicción VÁLIDA:
    # ambos lados presentes con win_probability real y consistente (sum≈1).
    sides_by_game: dict[int, dict] = {}
    for fp in all_files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [WARN] Could not parse {fp.name}: {e}")
            continue
        gpk = int(d.get("game_pk", 0))
        side = d.get("side", "")
        if not gpk or side not in ("home", "away"):
            continue
        entry = sides_by_game.setdefault(gpk, {})
        entry[side] = {
            "win_probability": d.get("win_probability"),   # None si no se simuló
            "expected_runs": d.get("expected_runs_per_game"),
            "game_date": d.get("game_date", ""),
            "team": d.get("team_abbr", ""),
            "matchup": d.get("matchup", ""),
            "file": str(fp.name),
        }

    game_preds: dict[int, dict] = {}
    n_incomplete = 0
    for gpk, sides in sides_by_game.items():
        home = sides.get("home")
        away = sides.get("away")
        hwp = home.get("win_probability") if home else None
        awp = away.get("win_probability") if away else None
        # Predicción válida: ambos lados simulados y P(home)+P(away)≈1.
        complete = (
            hwp is not None and awp is not None
            and abs(float(hwp) + float(awp) - 1.0) <= 0.05
        )
        if not complete:
            n_incomplete += 1
            continue
        game_preds[gpk] = {
            "game_pk": gpk,
            "game_date": home["game_date"],
            "team": home["team"],
            "side": "home",
            "matchup": home["matchup"],          # "Away @ Home" — para Elo real (F07)
            "win_probability": float(hwp),        # P(home gana), real (no 0.5 por defecto)
            "expected_runs": float(home.get("expected_runs") or 4.5),
            "file": home["file"],
        }

    print(f"  {len(game_preds)} juegos con predicción VÁLIDA de dos lados "
          f"({n_incomplete} excluidos por predicción incompleta/degenerada).")

    # Resolve actuals
    predictions_with_outcome: list[tuple[float, int]] = []
    elo_games: list[tuple[str, str, str, int]] = []   # (date, home, away, home_won) — F07
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

        # Recolectar para el Elo real (F07): parsear "Away @ Home" del matchup.
        matchup = pred.get("matchup", "")
        if " @ " in matchup:
            away_name, home_name = (s.strip() for s in matchup.split(" @ ", 1))
            elo_games.append((pred.get("game_date", ""), home_name, away_name, home_won))
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

    # Baseline de LOCAL empírico: predecir siempre la tasa de victoria local
    # observada en ESTOS juegos (real, no constante de literatura — F07).
    home_rate = sum(y for _, y in predictions_with_outcome) / n
    ll_home   = log_loss([(home_rate, y) for _, y in predictions_with_outcome])

    # Elo REAL entrenado sobre los mismos juegos (F07). Si no hay matchups
    # parseables (predicciones antiguas sin el campo), cae a NaN y se marca.
    elo_preds = elo_baseline_predictions(elo_games) if elo_games else []
    ll_elo    = log_loss(elo_preds) if elo_preds else float("nan")
    elo_is_real = bool(elo_preds)

    bs_model     = brier_score(predictions_with_outcome)
    bs_home      = brier_score([(home_rate, y) for _, y in predictions_with_outcome])
    bs_benchmark = bs_home   # benchmark calculado sobre estos juegos, no de literatura

    auc = auc_roc(predictions_with_outcome)
    bins = calibration_bins(predictions_with_outcome)
    ece_val = ece(bins)

    # Percentage of games where model "beats" 50/50 baseline (directional)
    correct_direction = sum(
        1 for p, y in predictions_with_outcome
        if (p > 0.5 and y == 1) or (p < 0.5 and y == 0)
    )
    accuracy_pct = round(correct_direction / n * 100, 1) if n > 0 else 0.0

    # ── Intervalos de confianza (bootstrap) ─────────────────────────────────
    # Con ~200 juegos hay que saber si los deltas vs baseline son señal o ruido.
    ll_ci   = _bootstrap_ci(predictions_with_outcome, log_loss)
    auc_ci  = _bootstrap_ci(predictions_with_outcome, auc_roc)
    vs_50   = _paired_bootstrap_vs_baseline(predictions_with_outcome, [0.5] * n)
    vs_elo  = (
        _paired_bootstrap_vs_baseline(
            predictions_with_outcome,
            [p for p, _ in elo_preds],
        )
        if elo_is_real and len(elo_preds) == n
        else None
    )
    sim_real = _sim_vs_reality(game_records)
    wp_shape = _winprob_shape(predictions_with_outcome)

    beats_elo = (ll_model < ll_elo) if elo_is_real else None
    elo_str = f"{ll_elo:.4f}" if elo_is_real else "n/d (sin matchups)"
    print(f"\n  Results ({n} games):")
    print(f"    Log-Loss model:    {ll_model:.4f}  (50%: {ll_baseline:.4f}, local emp.: {ll_home:.4f}, Elo real: {elo_str})")
    print(f"    Brier Score:       {bs_model:.4f}  (benchmark local emp.: {bs_benchmark:.4f})")
    print(f"    AUC ROC:           {auc:.4f}")
    print(f"    ECE:               {ece_val:.4f}")
    print(f"    Accuracy (dir.):   {accuracy_pct}%")
    print(f"    Beats 50% baseline: {ll_model < ll_baseline}")
    print(f"    Beats local emp.:   {ll_model < ll_home}")
    print(f"    Beats Elo real:     {beats_elo if beats_elo is not None else 'n/d'}  "
          f"(n_elo={len(elo_preds)})")
    print(f"    Log-Loss IC90:     [{ll_ci['ci_low']}, {ll_ci['ci_high']}]   "
          f"AUC IC90: [{auc_ci['ci_low']}, {auc_ci['ci_high']}]")
    print(f"    Delta vs 50/50:    {vs_50['delta_logloss']:+}  "
          f"IC90 [{vs_50['ci_low']}, {vs_50['ci_high']}]  "
          f"P(modelo mejor)={vs_50['prob_model_better']}")
    print(f"    Sim vs realidad:   E[R]={sim_real.get('mean_expected_runs')} vs "
          f"real={sim_real.get('mean_actual_runs')}  sesgo={sim_real.get('bias_runs'):+} "
          f"r={sim_real.get('pearson_r')}")
    print(f"    Win-prob centrado: media={wp_shape.get('mean')} mediana={wp_shape.get('median')} "
          f"frac>0.5={wp_shape.get('frac_above_0.5')}")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_games": n,
        "n_excluded_incomplete": n_incomplete,
        "date_range": {
            "from": min(r["game_date"] for r in game_records) if game_records else "",
            "to":   max(r["game_date"] for r in game_records) if game_records else "",
        },
        "metrics": {
            "log_loss": ll_model,
            "log_loss_baseline_50pct": ll_baseline,
            "log_loss_baseline_home_empirical": round(ll_home, 5),
            "log_loss_baseline_elo": ll_elo if elo_is_real else None,
            "elo_baseline_is_real": elo_is_real,   # True = calculado sobre estos juegos
            "elo_n_games": len(elo_preds),
            "home_win_rate": round(home_rate, 5),
            "log_loss_delta_vs_50pct": round(ll_baseline - ll_model, 5),
            "log_loss_delta_vs_home": round(ll_home - ll_model, 5),
            "log_loss_delta_vs_elo": round(ll_elo - ll_model, 5) if elo_is_real else None,
            "brier_score": bs_model,
            "brier_benchmark_home_empirical": round(bs_benchmark, 5),
            "brier_delta": round(bs_benchmark - bs_model, 5),
            "auc_roc": auc,
            "ece": ece_val,
            "accuracy_pct": accuracy_pct,
            "beats_50pct_baseline": ll_model < ll_baseline,
            "beats_home_empirical": ll_model < ll_home,
            "beats_elo_baseline": (ll_model < ll_elo) if elo_is_real else None,
        },
        "confidence_intervals": {
            "log_loss_ci90": [ll_ci["ci_low"], ll_ci["ci_high"]],
            "auc_roc_ci90": [auc_ci["ci_low"], auc_ci["ci_high"]],
            "delta_vs_50pct": vs_50,
            "delta_vs_elo": vs_elo,
        },
        "sim_vs_reality": sim_real,
        "winprob_shape": wp_shape,
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
