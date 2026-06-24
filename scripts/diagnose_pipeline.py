"""diagnose_pipeline.py — Descomposición del error del pipeline (Fase 0).

Responde la pregunta central del roadmap de mejora de eficacia: cuando el
backtest a nivel JUEGO falla (Log-Loss > 50/50, AUC ~0.5, ECE alto, pierde vs
Elo), ¿el error vive en el MODELO PA, en el SIMULADOR o en el MODELADO DEL
RIVAL? Sin esta descomposición no se puede priorizar Fase 1 (PA) vs Fase 2
(simulador/opponent).

El script mide tres cosas, de la más limpia a la más agregada:

  A. PA-level OOS (``data/gold/features_train_v3.parquet`` temporada 2026):
     el modelo desplegado se entrenó con 2015-2024 y validó en 2025, así que
     2026 es un holdout COMPLETAMENTE VIRGEN. Se evalúa calibración (ECE,
     log-loss, sesgo E[R/PA]) y, sobre todo, DISCRIMINACIÓN one-vs-rest (AUC
     por clase). Si el PA model discrimina bien OOS, el fallo a nivel juego no
     es suyo.

  B. Dispersión del win-prob a nivel juego (``results/<fecha>/*.json``): si las
     win-probabilities están apiñadas en [0.45, 0.55] el sistema no distingue
     equipos fuertes de débiles aunque el PA model sea perfecto — síntoma de
     que el simulador/opponent es el cuello de botella.

Uso:
    python scripts/diagnose_pipeline.py
    python scripts/diagnose_pipeline.py --season 2026 --out reports/diagnostics/pipeline_diagnosis.json

Salida: JSON inmutable con las tres secciones + un veredicto heurístico.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.models.model_at_bat import AtBatPredictor, OUTCOME_NAMES  # noqa: E402
from src.constants import RUN_VALUES  # noqa: E402

GOLD_PATH = ROOT / "data" / "gold" / "features_train_v3.parquet"
MODEL_PATH = ROOT / "models" / "at_bat_predictor.pkl"
RESULTS_DIR = ROOT / "results"
DEFAULT_OUT = ROOT / "reports" / "diagnostics" / "pipeline_diagnosis.json"

# Temporada OOS por defecto: el modelo en producción entrena <=2024 y valida en
# 2025; 2026 es el primer año que el modelo nunca ha visto.
DEFAULT_OOS_SEASON = 2026


# ── A. PA-level OOS ────────────────────────────────────────────────────────────

def _load_oos_matrix(
    gold_path: Path, feature_names: list[str], season: int
) -> tuple[np.ndarray, np.ndarray]:
    """Carga la matriz de features OOS de una temporada en el orden del modelo.

    Args:
        gold_path: Ruta al Gold v3 (``features_train_v3.parquet``).
        feature_names: Columnas EXACTAS y en el ORDEN con que se entrenó el
            modelo (``AtBatPredictor._feature_names``). Garantiza paridad.
        season: Temporada a aislar como holdout OOS.

    Returns:
        Tupla ``(X, y)`` con ``X`` float32 shape (N, D) e ``y`` int32 shape (N,).

    Raises:
        ValueError: Si la temporada no tiene filas en el Gold.
    """
    df = pl.read_parquet(gold_path, columns=[*feature_names, "pa_outcome_idx", "season"])
    df = df.filter(pl.col("season") == season)
    if df.is_empty():
        raise ValueError(f"No hay filas para season={season} en {gold_path}")
    X = df.select(feature_names).to_numpy().astype(np.float32)
    y = df["pa_outcome_idx"].to_numpy().astype(np.int32)
    return X, y


def diagnose_pa_level(predictor: AtBatPredictor, X: np.ndarray, y: np.ndarray) -> dict:
    """Calibración + discriminación del PA model en un holdout OOS.

    Reutiliza ``predictor.evaluate_calibration`` (ECE, log-loss, distribuciones,
    E[R/PA]) y añade AUC one-vs-rest por clase, que mide la capacidad de
    DISCRIMINAR — el eje en el que el sistema falla a nivel juego (AUC ~0.5).

    Args:
        predictor: Modelo desplegado ya cargado.
        X: Matriz de features OOS, shape (N, D).
        y: Etiquetas reales, shape (N,).

    Returns:
        Dict con métricas de calibración y AUC por clase + el AUC medio
        ponderado por soporte de clase.
    """
    cal = predictor.evaluate_calibration(X, y)
    probs = predictor.predict_proba(X)

    per_class_auc: dict[str, float | None] = {}
    supports: dict[str, int] = {}
    for c, name in enumerate(OUTCOME_NAMES):
        y_bin = (y == c).astype(int)
        support = int(y_bin.sum())
        supports[name] = support
        if 0 < support < len(y):
            per_class_auc[name] = round(float(roc_auc_score(y_bin, probs[:, c])), 4)
        else:
            per_class_auc[name] = None

    # AUC medio ponderado por soporte (resumen de discriminación global)
    weighted = [
        (per_class_auc[n] * supports[n])
        for n in OUTCOME_NAMES
        if per_class_auc[n] is not None
    ]
    total_support = sum(supports[n] for n in OUTCOME_NAMES if per_class_auc[n] is not None)
    macro_auc = round(sum(weighted) / total_support, 4) if total_support else None

    ev_pred = float(np.mean(probs @ RUN_VALUES))
    ev_obs = float(np.asarray(RUN_VALUES)[y].mean())

    return {
        "n_pa": int(len(y)),
        "overall_logloss": cal["overall_logloss"],
        "overall_ece": cal["overall_ece"],
        "ece_target_met": cal["ece_target_met"],
        "expected_run_value_pred": round(ev_pred, 5),
        "expected_run_value_obs": round(ev_obs, 5),
        "expected_run_value_bias": round(ev_pred - ev_obs, 5),
        "per_class_logloss": cal["per_class_logloss"],
        "per_class_ece": cal["per_class_ece"],
        "per_class_auc": per_class_auc,
        "per_class_support": supports,
        "weighted_auc": macro_auc,
        "predicted_outcome_distribution": cal["predicted_outcome_distribution"],
        "observed_outcome_distribution": cal["observed_outcome_distribution"],
    }


# ── B. Dispersión del win-prob a nivel juego ───────────────────────────────────

def diagnose_winprob_dispersion(results_dir: Path) -> dict:
    """Mide la forma de la distribución de win-prob generada por el simulador.

    Un sistema que discrimina produce win-probs repartidas por [0, 1]; uno que
    no, las apiña en torno a 0.5. Offline: sólo lee los JSON de ``results/``.

    Args:
        results_dir: Directorio ``results/`` con subdirectorios por fecha.

    Returns:
        Dict con std, percentiles y fracción de predicciones en bandas
        estrechas alrededor de 0.5.
    """
    probs: list[float] = []
    files = sorted(results_dir.glob("2*/*.json"))
    for fp in files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        wp = d.get("win_probability")
        if wp is not None:
            probs.append(float(wp))

    if not probs:
        return {"n_predictions": 0, "note": "sin archivos de predicción en results/"}

    arr = np.asarray(probs, dtype=np.float64)
    return {
        "n_predictions": int(arr.size),
        "n_files_scanned": len(files),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "p05": round(float(np.percentile(arr, 5)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(arr.max()), 4),
        "frac_in_0.45_0.55": round(float(((arr >= 0.45) & (arr <= 0.55)).mean()), 4),
        "frac_in_0.40_0.60": round(float(((arr >= 0.40) & (arr <= 0.60)).mean()), 4),
    }


# ── Veredicto ──────────────────────────────────────────────────────────────────

def build_verdict(pa: dict, winprob: dict) -> dict:
    """Heurística que localiza dónde domina el error a partir de A y B.

    Args:
        pa: Salida de :func:`diagnose_pa_level`.
        winprob: Salida de :func:`diagnose_winprob_dispersion`.

    Returns:
        Dict con flags booleanas y un mensaje legible.
    """
    pa_calibrated = bool(pa["ece_target_met"]) and abs(pa["expected_run_value_bias"]) <= 0.005
    # HR y K son las clases de alto valor donde la discriminación importa.
    hr_auc = pa["per_class_auc"].get("HOME_RUN")
    k_auc = pa["per_class_auc"].get("STRIKEOUT")
    pa_discriminates = (
        pa["weighted_auc"] is not None and pa["weighted_auc"] >= 0.60
        and (hr_auc is None or hr_auc >= 0.70)
    )
    winprob_collapsed = (
        winprob.get("n_predictions", 0) > 0
        and winprob.get("frac_in_0.40_0.60", 1.0) >= 0.80
    )

    if pa_calibrated and pa_discriminates and winprob_collapsed:
        locus = "simulador/opponent"
        msg = (
            "El PA model está calibrado y discrimina OOS, pero el win-prob a "
            "nivel juego está colapsado en torno a 0.5. El error vive aguas "
            "abajo: simulador (sin bullpen) y/o modelado del rival. Prioriza Fase 2."
        )
    elif not pa_discriminates:
        locus = "PA model"
        msg = (
            "El PA model discrimina poco OOS (AUC bajo). El cuello de botella "
            "está en el modelo de plate appearance. Prioriza Fase 1."
        )
    elif not pa_calibrated:
        locus = "PA model (calibración)"
        msg = (
            "El PA model perdió calibración fuera de muestra (ECE/sesgo E[R]). "
            "Revisa drift y recalibración antes de tocar el simulador."
        )
    else:
        locus = "indeterminado"
        msg = (
            "PA model sano y win-prob no colapsado: el error a nivel juego puede "
            "ser muestra pequeña o un sesgo más sutil. Amplía el backtest (IC)."
        )

    return {
        "pa_calibrated_oos": pa_calibrated,
        "pa_discriminates_oos": pa_discriminates,
        "winprob_collapsed": winprob_collapsed,
        "error_locus": locus,
        "message": msg,
    }


def run_diagnosis(season: int, out_path: Path) -> dict:
    """Orquesta las tres secciones y persiste el reporte.

    Args:
        season: Temporada OOS a usar para el diagnóstico PA-level.
        out_path: Ruta de salida del JSON de diagnóstico.

    Returns:
        El dict de diagnóstico completo.
    """
    print(f"Cargando modelo desplegado: {MODEL_PATH}", flush=True)
    predictor = AtBatPredictor.load(str(MODEL_PATH))
    feature_names = predictor._feature_names
    if not feature_names:
        raise RuntimeError("El modelo no tiene feature_names; no se puede garantizar paridad.")

    print(f"PA-level OOS (season={season}) — cargando Gold...", flush=True)
    X, y = _load_oos_matrix(GOLD_PATH, feature_names, season)
    print(f"  {len(y):,} PA OOS, {X.shape[1]} features.", flush=True)
    pa = diagnose_pa_level(predictor, X, y)

    print(f"  overall ECE={pa['overall_ece']}  logloss={pa['overall_logloss']}  "
          f"E[R/PA] sesgo={pa['expected_run_value_bias']:+}", flush=True)
    print(f"  weighted AUC={pa['weighted_auc']}  AUC por clase:", flush=True)
    for name in OUTCOME_NAMES:
        print(f"    {name:<13} AUC={pa['per_class_auc'][name]}  "
              f"(n={pa['per_class_support'][name]})", flush=True)

    print("\nDispersión win-prob a nivel juego (results/)...", flush=True)
    winprob = diagnose_winprob_dispersion(RESULTS_DIR)
    if winprob.get("n_predictions"):
        print(f"  n={winprob['n_predictions']}  std={winprob['std']}  "
              f"p05={winprob['p05']} p50={winprob['p50']} p95={winprob['p95']}  "
              f"frac[0.40,0.60]={winprob['frac_in_0.40_0.60']}", flush=True)

    verdict = build_verdict(pa, winprob)
    print(f"\nVEREDICTO: error_locus = {verdict['error_locus']}", flush=True)
    print(f"  {verdict['message']}", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "oos_season": season,
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "pa_level_oos": pa,
        "winprob_dispersion": winprob,
        "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado -> {out_path}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descomposición del error del pipeline (Fase 0)")
    parser.add_argument("--season", type=int, default=DEFAULT_OOS_SEASON,
                        help="Temporada OOS para el diagnóstico PA-level")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                        help="Ruta del JSON de salida")
    args = parser.parse_args()
    run_diagnosis(season=args.season, out_path=Path(args.out))
