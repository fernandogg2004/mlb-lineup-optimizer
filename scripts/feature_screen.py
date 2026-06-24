"""feature_screen.py — Screening de features NUEVAS por lift de AUC OOS (Fase 1).

La prueba de techo (`ceiling_test.py`) mostró que (a) la identidad de jugador no
ayuda y (b) el techo está en la calidad de contacto. Para subir discriminación
hay que añadir INFORMACIÓN NUEVA que no esté ya en las 51 features (las
interacciones de features existentes no cuentan: el GBM ya las captura por
árboles). Este harness mide, sobre el MISMO split OOS que la prueba de techo
(train ≤2024, early-stop 2025, eval 2026), cuánto sube el AUC cada candidata
ANTES de pagar el coste de integrarla en el contrato anti-skew.

Candidatas (derivables de Gold, todas las temporadas, servibles en producción):
  - pitcher_xwoba_allowed_30d : calidad de contacto PERMITIDA por el pitcher
    (las features de pitcher son FIP/K/BB/HR, outcome-based; esto es nuevo).
  - pitcher_hard_hit_allowed_30d : % batazos duros permitidos (rolling).
  - batter_days_rest / pitcher_days_rest : fatiga/descanso (señal nueva).

Todas con shift(1) anti-leakage (sólo ven juegos completados), igual que las
features de producción.

Uso:
    python scripts/feature_screen.py
    python scripts/feature_screen.py --train-sample 600000 --n-estimators 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import log_loss as sk_log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.models.model_at_bat import AtBatPredictor, OUTCOME_NAMES  # noqa: E402

GOLD_PATH = ROOT / "data" / "gold" / "features_train_v3.parquet"
MODEL_PATH = ROOT / "models" / "at_bat_predictor.pkl"
DEFAULT_OUT = ROOT / "reports" / "diagnostics" / "feature_screen.json"

TRAIN_MAX_SEASON = 2024
EARLYSTOP_SEASON = 2025
EVAL_SEASON = 2026
N_CLASSES = len(OUTCOME_NAMES)
HARD_HIT_MPH = 95.0

CANDIDATE_COLS = [
    "pitcher_xwoba_allowed_30d",
    "pitcher_hard_hit_allowed_30d",
    "batter_days_rest",
    "pitcher_days_rest",
]


def _add_candidates(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula las features candidatas con shift(1) anti-leakage y las une a ``df``.

    Args:
        df: Gold v3 con al menos batter_id, pitcher_id, game_date, season,
            xwoba, launch_speed.

    Returns:
        ``df`` con las columnas de :data:`CANDIDATE_COLS` añadidas.
    """
    df = df.with_columns(pl.col("game_date").str.to_date(strict=False).alias("_gd"))

    # ── Pitcher: calidad de contacto permitida (grano diario → rolling 30g) ──
    pdaily = (
        df.with_columns(
            (pl.col("launch_speed") >= HARD_HIT_MPH).cast(pl.Float64).alias("_hh")
        )
        .group_by(["pitcher_id", "_gd"])
        .agg([
            pl.col("xwoba").drop_nulls().mean().alias("xw_allowed"),
            pl.col("_hh").drop_nulls().mean().alias("hh_allowed"),
        ])
        .sort(["pitcher_id", "_gd"])
    )
    pdaily = pdaily.with_columns([
        pl.col("xw_allowed").shift(1).rolling_mean(30, min_periods=3)
          .over("pitcher_id").alias("pitcher_xwoba_allowed_30d"),
        pl.col("hh_allowed").shift(1).rolling_mean(30, min_periods=3)
          .over("pitcher_id").alias("pitcher_hard_hit_allowed_30d"),
        (pl.col("_gd") - pl.col("_gd").shift(1)).dt.total_days()
          .over("pitcher_id").alias("pitcher_days_rest"),
    ]).select(["pitcher_id", "_gd", "pitcher_xwoba_allowed_30d",
               "pitcher_hard_hit_allowed_30d", "pitcher_days_rest"])

    # ── Batter: días de descanso ────────────────────────────────────────────
    bdaily = (
        df.select(["batter_id", "_gd"]).unique()
        .sort(["batter_id", "_gd"])
        .with_columns(
            (pl.col("_gd") - pl.col("_gd").shift(1)).dt.total_days()
            .over("batter_id").alias("batter_days_rest")
        )
    )

    df = df.join(pdaily, on=["pitcher_id", "_gd"], how="left")
    df = df.join(bdaily, on=["batter_id", "_gd"], how="left")
    # Cap de descanso (offseason / IL): valores enormes → tope razonable.
    return df.with_columns([
        pl.col("batter_days_rest").clip(upper_bound=30),
        pl.col("pitcher_days_rest").clip(upper_bound=30),
    ]).drop("_gd")


def _evaluate(model: LGBMClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    """AUC one-vs-rest por clase + AUC ponderado + log-loss OOS."""
    probs = model.predict_proba(X)
    per_class, wnum, wden = {}, 0.0, 0
    for c, name in enumerate(OUTCOME_NAMES):
        yb = (y == c).astype(int); sup = int(yb.sum())
        if 0 < sup < len(y):
            a = float(roc_auc_score(yb, probs[:, c])); per_class[name] = round(a, 4)
            wnum += a * sup; wden += sup
        else:
            per_class[name] = None
    return {
        "per_class_auc": per_class,
        "weighted_auc": round(wnum / wden, 4) if wden else None,
        "logloss": round(float(sk_log_loss(y, probs, labels=list(range(N_CLASSES)))), 5),
    }


def _fit_eval(cols, df, masks, y_all, n_estimators, cat=None) -> dict:
    """Entrena LGBM sobre ``cols`` y evalúa en el holdout OOS."""
    tr, es, ev = masks
    X = df.select(cols).to_numpy().astype(np.float32)
    model = LGBMClassifier(
        objective="multiclass", num_class=N_CLASSES, n_estimators=n_estimators,
        learning_rate=0.05, num_leaves=63, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.5,
        n_jobs=-1, verbosity=-1,
    )
    t0 = time.time()
    model.fit(
        X[tr], y_all[tr], eval_set=[(X[es], y_all[es])], eval_metric="multi_logloss",
        callbacks=[early_stopping(40, verbose=False), log_evaluation(0)],
    )
    out = _evaluate(model, X[ev], y_all[ev])
    out["best_iteration"] = int(model.best_iteration_ or n_estimators)
    out["secs"] = round(time.time() - t0, 1)
    return out


def run_screen(train_sample: int, n_estimators: int, out_path: Path) -> dict:
    """Compara V0 (51 features) vs V0+cada candidata y vs V0+todas (AUC OOS)."""
    base = AtBatPredictor.load(str(MODEL_PATH))._feature_names
    print(f"Base: {len(base)} features. Cargando Gold...", flush=True)
    need = list(dict.fromkeys([*base, "pa_outcome_idx", "season", "game_date",
                               "batter_id", "pitcher_id", "xwoba", "launch_speed"]))
    df = pl.read_parquet(GOLD_PATH, columns=need)
    print(f"  {len(df):,} filas. Calculando candidatas...", flush=True)
    df = _add_candidates(df)
    for c in CANDIDATE_COLS:
        cov = df[c].is_not_null().mean()
        print(f"  {c}: cobertura {cov:.1%}", flush=True)

    season = df["season"].to_numpy()
    tr = season <= TRAIN_MAX_SEASON
    es = season == EARLYSTOP_SEASON
    ev = season == EVAL_SEASON
    # Subsample de train
    rng = np.random.default_rng(42)
    tr_idx = np.where(tr)[0]
    if len(tr_idx) > train_sample:
        keep = np.zeros(len(season), bool)
        keep[np.sort(rng.choice(tr_idx, train_sample, replace=False))] = True
        tr = keep
    y_all = df["pa_outcome_idx"].to_numpy().astype(np.int32)
    masks = (tr, es, ev)
    print(f"  train={tr.sum():,} es={es.sum():,} eval={ev.sum():,}", flush=True)

    results = {}
    print("\n=== V0 (control) ===", flush=True)
    results["V0_base"] = _fit_eval(list(base), df, masks, y_all, n_estimators)
    v0 = results["V0_base"]["weighted_auc"]
    print(f"  wAUC={v0}  HR={results['V0_base']['per_class_auc']['HOME_RUN']}", flush=True)

    for c in CANDIDATE_COLS:
        print(f"\n=== V0 + {c} ===", flush=True)
        r = _fit_eval(list(base) + [c], df, masks, y_all, n_estimators)
        r["delta_wauc_vs_v0"] = round((r["weighted_auc"] or 0) - (v0 or 0), 4)
        results[f"V0+{c}"] = r
        print(f"  wAUC={r['weighted_auc']}  delta={r['delta_wauc_vs_v0']:+.4f}  "
              f"HR={r['per_class_auc']['HOME_RUN']} 1B={r['per_class_auc']['SINGLE']}", flush=True)

    print("\n=== V0 + TODAS las candidatas ===", flush=True)
    rall = _fit_eval(list(base) + CANDIDATE_COLS, df, masks, y_all, n_estimators)
    rall["delta_wauc_vs_v0"] = round((rall["weighted_auc"] or 0) - (v0 or 0), 4)
    results["V0+all"] = rall
    print(f"  wAUC={rall['weighted_auc']}  delta={rall['delta_wauc_vs_v0']:+.4f}", flush=True)

    winners = [c for c in CANDIDATE_COLS
               if results[f"V0+{c}"]["delta_wauc_vs_v0"] >= 0.003]
    verdict = (
        f"INTEGRAR: {winners} (lift wAUC ≥ 0.003)." if winners
        else "SIN GANADORAS: ninguna candidata sube el AUC ≥0.003 → la señal "
             "disponible ya está capturada; se requiere NUEVA fuente de datos "
             "(arsenal velo/spin, weather) para más discriminación."
    )
    print(f"\nVEREDICTO: {verdict}", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": {"train_max": TRAIN_MAX_SEASON, "eval": EVAL_SEASON, "train_rows": int(tr.sum())},
        "results": results, "winners": winners, "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado -> {out_path}", flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Screening de features nuevas (Fase 1)")
    p.add_argument("--train-sample", type=int, default=600000)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    a = p.parse_args()
    run_screen(a.train_sample, a.n_estimators, Path(a.out))
