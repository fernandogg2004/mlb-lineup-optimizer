"""
train_v3.py — Train AtBatPredictor v4 on full Statcast era (2015-2025 + 2026 partial).

Changes vs v3:
  - VAL_SEASON = 2025 (modelo ve datos 2025 por primera vez)
  - LOG_SCALED_CLASS_WEIGHTS para mejorar recall en eventos raros (3B, DP, HR)
  - Calibration split correcto: val_df partido en dos mitades cronológicas
    (primera mitad → calibración isotónica, segunda mitad → early stopping + ECE)
  - Memory-optimized: num_leaves=63, max_bin=127, n_estimators=1200
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.models.model_at_bat import (  # noqa: E402
    AtBatModelConfig,
    AtBatPredictor,
    LOG_SCALED_CLASS_WEIGHTS,
)

GOLD_PATH  = ROOT / "data" / "gold" / "features_train_v3.parquet"
OUT_PATH   = ROOT / "models" / "pa_predictor_v1.pkl"
DEST_PATH  = ROOT / "models" / "at_bat_predictor.pkl"

# 2025 = primera temporada post-entrenamiento anterior → es el holdout real
VAL_SEASON  = 2025
# 2026 parcial: incluir en entrenamiento para que el modelo vea datos actuales
# (pocas filas, no usarlo como val porque la temporada no ha terminado)
TRAIN_CAP_SEASON = 2026

_LEAKING    = {"xwoba", "launch_speed", "launch_angle"}
SAMPLE_FRAC = 0.70   # aumentado a 0.70 porque ahora hay ~12 temporadas


def main() -> None:
    print(f"Cargando Gold: {GOLD_PATH}", flush=True)
    t0 = time.time()
    df = pl.read_parquet(GOLD_PATH)
    seasons_available = sorted(df["season"].unique().to_list())
    print(f"  {len(df):,} filas, {len(df.columns)} cols  "
          f"temporadas={seasons_available}  ({time.time()-t0:.1f}s)", flush=True)

    label_col    = "pa_outcome_idx"
    # Exclude: label, identifiers (high-cardinality IDs), temporal index, leaking Statcast raw
    _NON_FEATURES = {label_col, "season", "game_date", "batter_id", "pitcher_id"}
    feature_cols = [
        c for c in df.columns
        if c not in _NON_FEATURES | _LEAKING
        and df[c].dtype not in (pl.Utf8, pl.String, pl.Categorical)
    ]
    print(f"  Features ({len(feature_cols)}): {feature_cols}", flush=True)

    # Training: todas las temporadas excepto VAL_SEASON
    # (incluye 2026 parcial si está disponible)
    train_df = df.filter(pl.col("season") != VAL_SEASON)
    val_df   = df.filter(pl.col("season") == VAL_SEASON).sort("game_date")

    if val_df.is_empty():
        print(f"ERROR: No hay datos para VAL_SEASON={VAL_SEASON}. "
              "¿Se descargó Silver 2025?", flush=True)
        sys.exit(1)

    # Calibration split: primera mitad cronológica → calibración
    #                    segunda mitad → early stopping + holdout ECE
    mid = len(val_df) // 2
    cal_df  = val_df[:mid]
    eval_df = val_df[mid:]

    # Subsample de training para caber en RAM
    rng = np.random.default_rng(42)
    n_train = len(train_df)
    idx = rng.choice(n_train, size=int(n_train * SAMPLE_FRAC), replace=False)
    idx.sort()

    print(f"\nConvirtiendo a numpy "
          f"(train subsample {SAMPLE_FRAC:.0%} -> {len(idx):,} filas)...", flush=True)
    t1 = time.time()
    X_train_full = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train_full = train_df[label_col].to_numpy().astype(np.int32)
    X_train = X_train_full[idx]
    y_train = y_train_full[idx]
    del X_train_full, y_train_full

    X_cal  = cal_df.select(feature_cols).to_numpy().astype(np.float32)
    y_cal  = cal_df[label_col].to_numpy().astype(np.int32)
    X_val  = eval_df.select(feature_cols).to_numpy().astype(np.float32)
    y_val  = eval_df[label_col].to_numpy().astype(np.int32)

    print(f"  X_train={X_train.shape}  X_cal={X_cal.shape}  X_val={X_val.shape}"
          f"  ({time.time()-t1:.1f}s)", flush=True)
    del df, train_df, val_df, cal_df, eval_df

    config = AtBatModelConfig(
        mlflow_experiment="pa-model-v4-2015-2025",
        num_leaves=63,
        n_estimators=1500,
        max_bin=127,
        min_child_samples=80,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=0.10,
        reg_lambda=1.50,
        class_weight=LOG_SCALED_CLASS_WEIGHTS,
        early_stopping_rounds=75,
        n_jobs=-1,
    )
    print(
        f"\nConfig: num_leaves={config.num_leaves}, n_estimators={config.n_estimators}, "
        f"max_bin={config.max_bin}, class_weight=log_scaled, n_jobs={config.n_jobs}",
        flush=True,
    )

    predictor = AtBatPredictor(config)

    print("\nEntrenando modelo...", flush=True)
    t2 = time.time()
    predictor.fit(
        X_train, y_train,
        X_val,   y_val,
        X_cal,   y_cal,
        feature_names=feature_cols,
    )
    elapsed = time.time() - t2
    print(f"Entrenamiento completado en {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

    # Evaluación final en holdout
    print("\nEvaluando en holdout (X_val)...", flush=True)
    metrics = predictor.evaluate_calibration(X_val, y_val)
    print(f"  ECE    : {metrics['overall_ece']:.5f}  (target ≤ 0.035)")
    print(f"  LogLoss: {metrics['overall_logloss']:.5f}")
    print(f"  E[R/PA]: {metrics['avg_expected_run_value']:.5f}  (sanity: ~0.105)")
    print(f"  ECE por clase:")
    for cls, ece in metrics["per_class_ece"].items():
        flag = " !" if ece > 0.05 else ""
        print(f"    {cls:<15} {ece:.5f}{flag}")

    # Guardar
    predictor.save(str(OUT_PATH))
    print(f"\nModelo guardado : {OUT_PATH}", flush=True)
    shutil.copy2(OUT_PATH, DEST_PATH)
    print(f"Copiado a       : {DEST_PATH}", flush=True)
    print("\nRetrain completado.", flush=True)


if __name__ == "__main__":
    main()
