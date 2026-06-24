"""ceiling_test.py — ¿La baja discriminación del PA model es arreglable o intrínseca?

Fase 0bis (prueba de techo). El diagnóstico mostró que el PA model está
perfectamente calibrado pero discrimina poco (AUC ponderado ~0.57 OOS). Antes de
invertir en la Fase 1 hay que saber si ese techo es del MÉTODO (features
demasiado encogidas, capacidad insuficiente) o del PROBLEMA (el outcome de un PA
es intrínsecamente casi aleatorio dado lo conocible antes del swing).

Estrategia: entrenar varias variantes diagnósticas sobre el MISMO split temporal
(train ≤2024, early-stop 2025, eval 2026 OOS) y medir SÓLO discriminación
(AUC one-vs-rest por clase, log-loss). No se calibra: la isotónica es monótona y
no cambia el ranking, así que el AUC crudo es la métrica de discriminación.

Variantes:
  V0_actual    : las 51 features de producción. Control — debe reproducir ~0.57.
  V1_identidad : V0 + batter_id + pitcher_id (categóricas). ¿Los promedios
                 encogidos están tirando señal de skill que la identidad recupera?
  V2_contacto  : V0 + xwoba/launch_speed/launch_angle del PROPIO PA (LEAKING).
                 Cota superior BLANDA: cuánto explicaría el outcome si pudiéramos
                 predecir el contacto a la perfección.
  V3_oraculo   : V0 + identidad + contacto. Techo combinado.

Lectura:
  - Si V2/V3 >> V0  -> hay margen: mejores features pre-PA (predecir contacto,
    matchup más nítido) pueden subir la discriminación. Fase 1 tiene recorrido.
  - Si V2/V3 ≈ V0   -> el outcome es casi aleatorio dado lo conocible; el techo
    es intrínseco y conviene apoyarse en Fase 2 (simulador) y en blends.

Uso:
    python scripts/ceiling_test.py
    python scripts/ceiling_test.py --train-sample 500000 --n-estimators 400
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
DEFAULT_OUT = ROOT / "reports" / "diagnostics" / "ceiling_test.json"

TRAIN_MAX_SEASON = 2024   # train: season <= esto
EARLYSTOP_SEASON = 2025   # validación de early stopping
EVAL_SEASON = 2026        # holdout OOS (nunca visto por el modelo de producción)

IDENTITY_COLS = ["batter_id", "pitcher_id"]
CONTACT_COLS = ["xwoba", "launch_speed", "launch_angle"]  # leaking (sólo para el techo)
N_CLASSES = len(OUTCOME_NAMES)


def _factorize(col: np.ndarray) -> np.ndarray:
    """Mapea ids arbitrarios a códigos enteros contiguos (para categóricas LGBM).

    Args:
        col: Columna de ids (enteros arbitrarios), shape (N,).

    Returns:
        Códigos contiguos int32 en [0, n_unique), shape (N,).
    """
    _, codes = np.unique(col, return_inverse=True)
    return codes.astype(np.int32)


def _evaluate(model: LGBMClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    """Discriminación OOS: AUC one-vs-rest por clase + log-loss multiclase.

    Args:
        model: Clasificador LightGBM ya entrenado.
        X: Features OOS, shape (N, D).
        y: Etiquetas, shape (N,).

    Returns:
        Dict con AUC por clase, AUC ponderado por soporte y log-loss.
    """
    probs = model.predict_proba(X)
    per_class: dict[str, float | None] = {}
    weighted_num = 0.0
    weighted_den = 0
    for c, name in enumerate(OUTCOME_NAMES):
        y_bin = (y == c).astype(int)
        sup = int(y_bin.sum())
        if 0 < sup < len(y):
            auc = float(roc_auc_score(y_bin, probs[:, c]))
            per_class[name] = round(auc, 4)
            weighted_num += auc * sup
            weighted_den += sup
        else:
            per_class[name] = None
    return {
        "per_class_auc": per_class,
        "weighted_auc": round(weighted_num / weighted_den, 4) if weighted_den else None,
        "logloss": round(float(sk_log_loss(y, probs, labels=list(range(N_CLASSES)))), 5),
    }


def run_ceiling_test(train_sample: int, n_estimators: int, out_path: Path) -> dict:
    """Entrena las variantes diagnósticas y compara su discriminación OOS.

    Args:
        train_sample: Nº de filas de train a muestrear (velocidad).
        n_estimators: Árboles máximos por variante (con early stopping).
        out_path: Ruta de salida del JSON.

    Returns:
        Dict con los resultados por variante y los metadatos del experimento.
    """
    print(f"Cargando feature_names de producción: {MODEL_PATH}", flush=True)
    base_features = AtBatPredictor.load(str(MODEL_PATH))._feature_names
    print(f"  {len(base_features)} features base.", flush=True)

    needed = list(dict.fromkeys(
        [*base_features, *IDENTITY_COLS, *CONTACT_COLS, "pa_outcome_idx", "season"]
    ))
    print("Cargando Gold...", flush=True)
    t0 = time.time()
    df = pl.read_parquet(GOLD_PATH, columns=needed)
    print(f"  {len(df):,} filas ({time.time()-t0:.1f}s)", flush=True)

    train_df = df.filter(pl.col("season") <= TRAIN_MAX_SEASON)
    es_df = df.filter(pl.col("season") == EARLYSTOP_SEASON)
    eval_df = df.filter(pl.col("season") == EVAL_SEASON)
    print(f"  train={len(train_df):,}  early_stop({EARLYSTOP_SEASON})={len(es_df):,}  "
          f"eval({EVAL_SEASON})={len(eval_df):,}", flush=True)

    # Subsample de train para velocidad (la comparación es relativa entre variantes)
    rng = np.random.default_rng(42)
    if len(train_df) > train_sample:
        idx = np.sort(rng.choice(len(train_df), size=train_sample, replace=False))
        train_df = train_df[idx]
    print(f"  train subsample -> {len(train_df):,}", flush=True)

    # Pre-factorizar identidad sobre TODO el df para códigos consistentes train/eval
    id_codes = {col: _factorize(df[col].to_numpy()) for col in IDENTITY_COLS}
    season_arr = df["season"].to_numpy()
    train_mask = season_arr <= TRAIN_MAX_SEASON
    es_mask = season_arr == EARLYSTOP_SEASON
    eval_mask = season_arr == EVAL_SEASON
    if len(train_df) < int(train_mask.sum()):
        # train fue submuestreado: re-derivar la máscara de filas elegidas
        train_pos = np.where(train_mask)[0][idx]
    else:
        train_pos = np.where(train_mask)[0]
    es_pos = np.where(es_mask)[0]
    eval_pos = np.where(eval_mask)[0]

    def build(cols: list[str], use_identity: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """Construye matrices (train, es, eval) y devuelve índices categóricos."""
        mats = df.select(cols).to_numpy().astype(np.float32)
        blocks = [mats]
        cat_idx: list[int] = []
        if use_identity:
            id_block = np.column_stack([id_codes[c] for c in IDENTITY_COLS]).astype(np.float32)
            cat_idx = [mats.shape[1] + i for i in range(len(IDENTITY_COLS))]
            blocks.append(id_block)
        full = np.hstack(blocks) if len(blocks) > 1 else mats
        return full[train_pos], full[es_pos], full[eval_pos], cat_idx

    y_train = df["pa_outcome_idx"].to_numpy().astype(np.int32)[train_pos]
    y_es = df["pa_outcome_idx"].to_numpy().astype(np.int32)[es_pos]
    y_eval = df["pa_outcome_idx"].to_numpy().astype(np.int32)[eval_pos]

    variants = {
        "V0_actual":    (list(base_features), False),
        "V1_identidad": (list(base_features), True),
        "V2_contacto":  (list(base_features) + CONTACT_COLS, False),
        "V3_oraculo":   (list(base_features) + CONTACT_COLS, True),
    }

    results: dict[str, dict] = {}
    for name, (cols, use_id) in variants.items():
        print(f"\n=== {name} ({len(cols)} cols{' +id' if use_id else ''}) ===", flush=True)
        Xtr, Xes, Xev, cat_idx = build(cols, use_id)
        t1 = time.time()
        model = LGBMClassifier(
            objective="multiclass",
            num_class=N_CLASSES,
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=80,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.5,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            Xtr, y_train,
            eval_set=[(Xes, y_es)],
            eval_metric="multi_logloss",
            categorical_feature=cat_idx if cat_idx else "auto",
            callbacks=[early_stopping(40, verbose=False), log_evaluation(0)],
        )
        ev = _evaluate(model, Xev, y_eval)
        ev["best_iteration"] = int(model.best_iteration_ or n_estimators)
        ev["train_seconds"] = round(time.time() - t1, 1)
        results[name] = ev
        print(f"  weighted AUC={ev['weighted_auc']}  logloss={ev['logloss']}  "
              f"(best_iter={ev['best_iteration']}, {ev['train_seconds']}s)", flush=True)
        print(f"  HR={ev['per_class_auc']['HOME_RUN']}  K={ev['per_class_auc']['STRIKEOUT']}  "
              f"1B={ev['per_class_auc']['SINGLE']}  BB={ev['per_class_auc']['WALK_HBP']}", flush=True)

    # Veredicto: ¿el techo (V3) supera al control (V0) de forma relevante?
    v0 = results["V0_actual"]["weighted_auc"]
    v3 = results["V3_oraculo"]["weighted_auc"]
    v1 = results["V1_identidad"]["weighted_auc"]
    headroom = round(v3 - v0, 4) if (v0 and v3) else None
    identity_gain = round(v1 - v0, 4) if (v0 and v1) else None
    verdict = (
        "MARGEN: el techo supera claramente al control -> Fase 1 (mejores features pre-PA) "
        "tiene recorrido."
        if headroom is not None and headroom >= 0.03
        else "TECHO INTRÍNSECO: el oráculo apenas supera al control -> el outcome es casi "
             "aleatorio dado lo conocible; apóyate en Fase 2 (simulador) y blends."
    )
    print(f"\nHeadroom V3-V0 = {headroom}  |  Ganancia identidad V1-V0 = {identity_gain}", flush=True)
    print(f"VEREDICTO: {verdict}", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": {"train_max_season": TRAIN_MAX_SEASON, "earlystop_season": EARLYSTOP_SEASON,
                  "eval_season": EVAL_SEASON, "train_rows": int(len(train_pos))},
        "variants": results,
        "headroom_v3_minus_v0": headroom,
        "identity_gain_v1_minus_v0": identity_gain,
        "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado -> {out_path}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de techo de discriminación (Fase 0bis)")
    parser.add_argument("--train-sample", type=int, default=600000)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()
    run_ceiling_test(args.train_sample, args.n_estimators, Path(args.out))
