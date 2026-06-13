"""
train_v3.py — Train AtBatPredictor v5 on full Statcast era (2015-present).

Cambios vs v4 (auditoría 2026-06):
  - Sin leakage intra-PA: pitch_count_in_pa y last_pitch_type excluidos
  - class_weight=None: las probabilidades alimentan un simulador; las
    frecuencias naturales son la señal (los class weights sesgaban E[R] +30%)
  - Cadena temporal estricta: train < cal < eval (sin temporadas futuras
    en train; el holdout H2 es virgen — solo se lee para la métrica final)
  - Gate de prior drift: |pred - obs| <= 0.5pp por clase y E[R/PA] ~ 0.105;
    si falla, el modelo NO se copia a producción
  - Features nuevas: platoon vs mano (James-Stein), park factors, is_home
"""
from __future__ import annotations

import shutil
import sys
import time
from datetime import date as _date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.models.model_at_bat import (  # noqa: E402
    OUTCOME_NAMES,
    AtBatModelConfig,
    AtBatPredictor,
)
from src.constants import RUN_VALUES  # noqa: E402

GOLD_PATH  = ROOT / "data" / "gold" / "features_train_v3.parquet"
OUT_PATH   = ROOT / "models" / "pa_predictor_v1.pkl"
DEST_PATH  = ROOT / "models" / "at_bat_predictor.pkl"

# VAL_SEASON dinámico: siempre la temporada anterior al año actual.
# Esto garantiza que el retrain anual siempre valida en el año más reciente completo
# sin necesidad de actualización manual.
VAL_SEASON       = _date.today().year - 1
TRAIN_CAP_SEASON = _date.today().year   # temporada actual (parcial) incluida en train

# Features que solo se conocen DESPUÉS de que el PA termina — usarlas en
# training es target leakage (en serving se imputarían constantes):
#   xwoba/launch_speed/launch_angle : métricas del batazo del propio PA
#   pitch_count_in_pa               : un walk exige ≥4 pitches, un K ≥3
#   last_pitch_type                 : es el pitch que TERMINÓ el PA
_LEAKING    = {"xwoba", "launch_speed", "launch_angle",
               "pitch_count_in_pa", "last_pitch_type"}
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

    # Cadena temporal ESTRICTA: train < cal < eval. Nunca datos posteriores a
    # VAL_SEASON en training (entrenar con 2026 y validar en 2025 deja que el
    # run environment del futuro se filtre hacia atrás e infla la métrica).
    val_season = VAL_SEASON
    if df.filter(pl.col("season") == val_season).is_empty():
        val_season = max(s for s in seasons_available if s < VAL_SEASON)
        print(f"AVISO: VAL_SEASON={VAL_SEASON} sin datos. Usando {val_season} como fallback.", flush=True)

    train_df = df.filter(pl.col("season") < val_season)
    val_df   = df.filter(pl.col("season") == val_season).sort("game_date")
    n_future = len(df.filter(pl.col("season") > val_season))
    if n_future:
        print(f"  Excluidas {n_future:,} filas de temporadas > {val_season} "
              f"(post-validación; entrarán en el retrain de producción).", flush=True)

    # Split de la temporada de validación en dos mitades cronológicas:
    #   H1 (cal_df)  → early stopping + calibración isotónica (ambos son "tuning")
    #   H2 (eval_df) → holdout VIRGEN: una sola lectura, la métrica que se publica
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
        # class_weight=None: el modelo alimenta un simulador Monte Carlo que
        # consume las probabilidades como pesos estocásticos — las frecuencias
        # naturales SON la señal. Upweighting de clases raras (TRIPLE×12 etc.)
        # sesga E[R] al alza (+30% runs) y obliga a parches post-hoc.
        class_weight=None,
        early_stopping_rounds=75,
        n_jobs=-1,
    )
    print(
        f"\nConfig: num_leaves={config.num_leaves}, n_estimators={config.n_estimators}, "
        f"max_bin={config.max_bin}, class_weight=None, n_jobs={config.n_jobs}",
        flush=True,
    )

    predictor = AtBatPredictor(config)

    print("\nEntrenando modelo...", flush=True)
    t2 = time.time()
    # X_cal hace de set de early stopping Y de calibración (ambos son tuning
    # sobre H1 de la temporada de validación). X_val (H2) queda virgen y solo
    # se toca en evaluate_calibration() más abajo.
    predictor.fit(
        X_train, y_train,
        X_cal,   y_cal,
        X_cal,   y_cal,
        feature_names=feature_cols,
    )
    elapsed = time.time() - t2
    print(f"Entrenamiento completado en {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

    # Evaluación final en holdout VIRGEN (H2 de la temporada de validación)
    print("\nEvaluando en holdout virgen (X_val)...", flush=True)
    metrics = predictor.evaluate_calibration(X_val, y_val)
    print(f"  ECE    : {metrics['overall_ece']:.5f}  (target ≤ 0.035)")
    print(f"  LogLoss: {metrics['overall_logloss']:.5f}")
    print(f"  E[R/PA]: {metrics['avg_expected_run_value']:.5f}  "
          f"(linear weights sobre baseline-out; comparar vs observado en el gate)")
    print(f"  ECE por clase:")
    for cls, ece in metrics["per_class_ece"].items():
        flag = " !" if ece > 0.05 else ""
        print(f"    {cls:<15} {ece:.5f}{flag}")

    # ── Gate de despliegue: sesgo de E[R] + prior drift por clase ────────────
    # La distribución media predicha debe igualar la observada. Si esto pasa,
    # el simulador no necesita escalares post-hoc tipo _MC_RUNS_SCALE.
    #
    # Umbrales por impacto en runs:
    #   - Clases con run value != 0 (BB,1B,2B,3B,HR,DP): drift <= 0.6pp —
    #     sesgan E[R] del simulador directamente.
    #   - OUT_IN_PLAY y STRIKEOUT (run value 0): drift <= 1.2pp — confundirlas
    #     entre sí no afecta E[R] a primer orden (solo via DP/avance de corredores).
    #   - Check principal: |E[R/PA] pred - obs| <= 0.005 runs (el sesgo que el
    #     viejo modelo tenía en +30% y parcheaba con _MC_RUNS_SCALE=0.768).
    probs_val = predictor.predict_proba(X_val)
    pred_dist = probs_val.mean(axis=0)
    obs_dist  = np.bincount(y_val, minlength=len(OUTCOME_NAMES)) / len(y_val)
    drift     = np.abs(pred_dist - obs_dist)
    ev_pred   = float(np.mean(probs_val @ RUN_VALUES))
    ev_obs    = float(np.asarray(RUN_VALUES)[y_val].mean())

    thresholds = np.where(np.asarray(RUN_VALUES) != 0.0, 0.006, 0.012)

    print("\nGate de prior drift (|pred - obs| por clase):")
    for name, p, o, d, t in zip(OUTCOME_NAMES, pred_dist, obs_dist, drift, thresholds):
        flag = "  <-- FAIL" if d > t else ""
        print(f"  {name:<13} pred={p:.4f}  obs={o:.4f}  drift={d:.4f}  (umbral {t:.3f}){flag}")

    ev_bias = ev_pred - ev_obs
    gate_ok = bool((drift <= thresholds).all()) and abs(ev_bias) <= 0.005
    print(f"  E[R/PA] pred={ev_pred:.4f}  obs={ev_obs:.4f}  sesgo={ev_bias:+.4f} (umbral ±0.005)")
    print(f"  GATE: {'PASS' if gate_ok else 'FAIL'}", flush=True)

    # Guardar
    predictor.save(str(OUT_PATH))
    print(f"\nModelo guardado : {OUT_PATH}", flush=True)
    if gate_ok:
        shutil.copy2(OUT_PATH, DEST_PATH)
        print(f"Copiado a       : {DEST_PATH}", flush=True)
        print("\nRetrain completado.", flush=True)
    else:
        print(f"NO copiado a {DEST_PATH}: el gate de prior drift falló.", flush=True)
        print("El modelo candidato queda en pa_predictor_v1.pkl para diagnóstico.", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
