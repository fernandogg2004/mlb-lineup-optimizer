"""
train_v3.py — Train AtBatPredictor v3 on full Statcast era (2015-2024).
Memory-optimized: num_leaves=63, max_bin=127, n_estimators=1200.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from models.model_at_bat import AtBatModelConfig, AtBatPredictor  # noqa: E402

GOLD_PATH  = ROOT / "data" / "gold" / "features_train_v2.parquet"
OUT_PATH   = ROOT / "models" / "pa_predictor_v1.pkl"
VAL_SEASON = 2024

_LEAKING   = {"xwoba", "launch_speed", "launch_angle"}

def main() -> None:
    print(f"Cargando Gold: {GOLD_PATH}", flush=True)
    t0 = time.time()
    df = pl.read_parquet(GOLD_PATH)
    print(f"  {len(df):,} filas, {len(df.columns)} cols  ({time.time()-t0:.1f}s)", flush=True)

    label_col    = "pa_outcome_idx"
    feature_cols = [
        c for c in df.columns
        if c not in {label_col, "season", "game_date"} | _LEAKING
    ]
    print(f"  Features ({len(feature_cols)}): {feature_cols}", flush=True)

    train_df = df.filter(pl.col("season") != VAL_SEASON)
    val_df   = df.filter(pl.col("season") == VAL_SEASON)

    # Subsample training data to 60% to fit in RAM — 5.5M rows is plenty for convergence
    SAMPLE_FRAC = 0.60
    rng = np.random.default_rng(42)
    n_train = len(train_df)
    idx = rng.choice(n_train, size=int(n_train * SAMPLE_FRAC), replace=False)
    idx.sort()

    print(f"Convirtiendo a numpy (subsample {SAMPLE_FRAC:.0%} → {len(idx):,} filas)...", flush=True)
    t1 = time.time()
    X_train_full = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train_full = train_df[label_col].to_numpy().astype(np.int32)
    X_train = X_train_full[idx]
    y_train = y_train_full[idx]
    del X_train_full, y_train_full
    X_val   = val_df.select(feature_cols).to_numpy().astype(np.float32)
    y_val   = val_df[label_col].to_numpy().astype(np.int32)
    print(f"  X_train={X_train.shape}, X_val={X_val.shape}  ({time.time()-t1:.1f}s)", flush=True)

    del df, train_df, val_df

    config = AtBatModelConfig(
        mlflow_experiment="pa-model-v3-statcast-era",
        num_leaves=31,
        n_estimators=1200,
        max_bin=63,
        min_child_samples=100,
        subsample=0.80,
        colsample_bytree=0.75,
        n_jobs=4,
    )
    print(f"Config: num_leaves={config.num_leaves}, n_estimators={config.n_estimators}, max_bin={config.max_bin}, n_jobs={config.n_jobs}", flush=True)

    predictor = AtBatPredictor(config)

    print("Entrenando...", flush=True)
    t2 = time.time()
    predictor.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    print(f"Entrenamiento completado en {time.time()-t2:.0f}s", flush=True)

    predictor.save(str(OUT_PATH))
    print(f"Modelo guardado: {OUT_PATH}", flush=True)

    import shutil
    dst = ROOT / "models" / "at_bat_predictor.pkl"
    shutil.copy2(OUT_PATH, dst)
    print(f"Copiado a:       {dst}", flush=True)


if __name__ == "__main__":
    main()
