"""
model_at_bat.py
===============
Multiclass classifier that predicts the probability distribution over 7
mutually-exclusive Plate Appearance outcomes for a given batter × pitcher
× context matchup.

Outcome classes (PA_OUTCOME enum):
    0 — OUT_IN_PLAY   (ground out, fly out, line out)
    1 — STRIKEOUT     (K)
    2 — WALK_HBP      (BB or Hit-By-Pitch; same base advancement)
    3 — SINGLE        (1B)
    4 — DOUBLE        (2B)
    5 — TRIPLE        (3B)
    6 — HOME_RUN      (HR)

Calibration requirement
-----------------------
Raw GBDT predicted probabilities are known to be poorly calibrated
(over-confident on the majority class). Because the simulation engine uses
these probabilities as *stochastic weights*, miscalibration propagates
directly into E[R] bias.

Target: Expected Calibration Error (ECE) ≤ 0.035 on the holdout set.

Implementation strategy:
    1. Train a LightGBM base classifier (+ optional CatBoost).
    2. Post-hoc calibrate with sklearn ``CalibratedClassifierCV`` using
       ``method='isotonic'`` (non-parametric; beats Platt for multiclass).
    3. Evaluate calibration quality with classwise ECE on the validation set.
    4. If ECE > 0.050 after isotonic, fall back to Platt Scaling (sigmoid).

Feature matrix
--------------
The full feature vector (~340 dims) is assembled upstream by the feature
pipeline and passed as a numpy float32 array. This module is feature-
agnostic: it receives X, y and trains/infers.

Usage:
    # Training
    python -m src.models.model_at_bat train \\
        --features-path gold/features_train.parquet \\
        --output-dir models/ --experiment pa-model-v1

    # Inference
    from src.models.model_at_bat import AtBatPredictor
    predictor = AtBatPredictor.load("models/pa_predictor_v1.pkl")
    probs = predictor.predict_proba(X)   # shape (N, 7)
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import polars as pl
import structlog
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ---------------------------------------------------------------------------
# Outcome registry
# ---------------------------------------------------------------------------

class PAOutcome(IntEnum):
    """Integer encoding for the 8 PA outcome classes.

    Class 7 (DOUBLE_PLAY) was added in Mejora 6 to explicitly model GIDP and
    strikeout-double-play events, which add 2 outs AND eliminate the runner on
    first base — a qualitatively different state transition from a regular out.
    The coordinated simulator change must be deployed simultaneously with the
    model retrain that produces 8-class probability vectors.
    """

    OUT_IN_PLAY   = 0
    STRIKEOUT     = 1
    WALK_HBP      = 2
    SINGLE        = 3
    DOUBLE        = 4
    TRIPLE        = 5
    HOME_RUN      = 6
    DOUBLE_PLAY   = 7   # GIDP or strikeout_double_play (Mejora 6)


OUTCOME_NAMES: list[str] = [o.name for o in PAOutcome]
N_CLASSES: int = len(PAOutcome)   # 8

# Linear run values — importados de src.constants (fuente única de verdad).
# Sources: FanGraphs 2024 linear weights (indices 0-6) +
#          Retrosheet 2015-2024 for DOUBLE_PLAY (index 7).
from src.constants import RUN_VALUES  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ⚠️ NO USAR CON LA SIMULACIÓN (audit F17) ⚠️
# Estos pesos de clase sesgan las probabilidades calibradas que el motor Monte
# Carlo consume como pesos estocásticos: cualquier upweighting de clases raras
# (3B, HR, DP) infla E[R] y rompe el gate de prior-drift de train_v3.py. El
# training de producción FUERZA class_weight=None a propósito (ver main() y
# train_v3.py). Esta constante se conserva solo como referencia histórica de por
# qué NO se reescala; no la conectes a AtBatModelConfig.class_weight.
#
# Rationale histórico: "balanced" produce un ratio 128:1 que desestabiliza
# LightGBM (best_iteration colapsa a 1); estos pesos log-escalados eran un
# intento intermedio, también descartado.
LOG_SCALED_CLASS_WEIGHTS: dict[int, float] = {
    0: 1.0,   # OUT_IN_PLAY  ~45%
    1: 1.2,   # STRIKEOUT    ~22%
    2: 2.5,   # WALK_HBP     ~9%
    3: 2.0,   # SINGLE       ~15%
    4: 3.5,   # DOUBLE       ~5%
    5: 12.0,  # TRIPLE       ~0.8%
    6: 4.0,   # HOME_RUN     ~3.2%
    7: 5.0,   # DOUBLE_PLAY  ~1.5%
}


@dataclass
class AtBatModelConfig:
    """Hyperparameters and training settings for the PA-level model.

    Attributes:
        n_estimators: Number of boosting rounds for LightGBM.
        learning_rate: Step size shrinkage.
        num_leaves: Maximum tree leaves (controls model complexity).
        max_depth: Maximum tree depth (-1 = no limit).
        min_child_samples: Minimum data in a leaf (regularization).
        subsample: Row sub-sampling ratio per tree.
        colsample_bytree: Feature sub-sampling ratio per tree.
        reg_alpha: L1 regularization term.
        reg_lambda: L2 regularization term.
        class_weight: Per-class weight dict, ``"balanced"``, or ``None``.
            Defaults to ``LOG_SCALED_CLASS_WEIGHTS``, which applies moderate
            upweighting to rare events (3B, DP, HR) without the 128:1 ratio
            instability caused by ``"balanced"``.  Pass ``None`` to disable.
        calibration_method: ``"isotonic"`` or ``"sigmoid"`` (Platt).
        ece_n_bins: Number of equal-width confidence bins for ECE.
        ece_target: Acceptable ECE threshold; warning if exceeded.
        cv_n_splits: Number of temporal cross-validation folds.
        early_stopping_rounds: Rounds without val-loss improvement.
        seed: Random seed for reproducibility.
        mlflow_experiment: MLflow experiment name.
        n_jobs: Parallel jobs for LightGBM tree building.
    """

    n_estimators: int = 1500
    learning_rate: float = 0.03
    num_leaves: int = 127
    max_depth: int = -1
    min_child_samples: int = 50
    subsample: float = 0.80
    colsample_bytree: float = 0.75
    reg_alpha: float = 0.10
    reg_lambda: float = 1.50
    max_bin: int = 255
    class_weight: Optional[dict | str] = None  # set to LOG_SCALED_CLASS_WEIGHTS at retrain
    calibration_method: str = "isotonic"
    ece_n_bins: int = 10
    ece_target: float = 0.035
    cv_n_splits: int = 5
    early_stopping_rounds: int = 75
    seed: int = 42
    mlflow_experiment: str = "pa-model-at-bat"
    n_jobs: int = -1


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

class ECEComputer:
    """Computes the Expected Calibration Error for a multiclass classifier.

    Uses the **classwise ECE** formulation: for each class c treated as
    one-vs-rest binary, compute the standard binned ECE and average across
    all C classes.

    ECE_c = Σ_b  (|B_b| / N) × |accuracy(B_b) - confidence(B_b)|
    ECE   = (1/C) × Σ_c ECE_c

    This formulation detects per-class over/under-confidence independently,
    which is critical for rare events (HR, 3B) that might be miscalibrated
    even when the aggregate ECE is acceptable.

    Args:
        n_bins: Number of equal-width confidence bins in [0, 1].
    """

    def __init__(self, n_bins: int = 10) -> None:
        """Initializes the ECE computer.

        Args:
            n_bins: Number of equal-width calibration bins.
        """
        self.n_bins = n_bins
        self.bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    def _binary_ece(self, y_true_binary: np.ndarray, y_prob: np.ndarray) -> float:
        """Computes ECE for a single one-vs-rest class.

        Args:
            y_true_binary: Boolean array shape (N,); 1 = this class, 0 = other.
            y_prob: Predicted probability for this class, shape (N,).

        Returns:
            Scalar ECE value in [0, 1].
        """
        n_samples = len(y_true_binary)
        ece = 0.0

        for i in range(self.n_bins):
            lo = self.bin_edges[i]
            hi = self.bin_edges[i + 1]
            # Include right edge in last bin
            mask = (y_prob >= lo) & (y_prob < hi if i < self.n_bins - 1 else y_prob <= hi)
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            conf = y_prob[mask].mean()
            acc  = y_true_binary[mask].mean()
            ece += (n_bin / n_samples) * abs(acc - conf)

        return float(ece)

    def compute(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> dict[str, float]:
        """Computes classwise ECE and overall mean ECE.

        Args:
            y_true: Integer class labels, shape (N,). Values in [0, C-1].
            y_prob: Predicted probabilities, shape (N, C).

        Returns:
            Dict with keys ``"overall_ece"``, ``"per_class_ece"``, and
            ``"per_class_names"``.

        Raises:
            ValueError: If ``y_true`` and ``y_prob`` have incompatible shapes.
        """
        if y_true.ndim != 1 or y_prob.ndim != 2:
            raise ValueError("y_true must be 1D, y_prob must be 2D (N, C).")
        n_classes = y_prob.shape[1]
        per_class: list[float] = []

        for c in range(n_classes):
            y_bin = (y_true == c).astype(np.float64)
            ece_c = self._binary_ece(y_bin, y_prob[:, c])
            per_class.append(ece_c)

        overall = float(np.mean(per_class))
        return {
            "overall_ece": overall,
            "per_class_ece": per_class,
            "per_class_names": [PAOutcome(c).name for c in range(n_classes)],
        }

    def reliability_data(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (fraction_positive, mean_predicted_prob, bin_counts) for plotting.

        Convenience wrapper around ``sklearn.calibration.calibration_curve`` for
        generating data to render reliability diagrams.

        Args:
            y_true: Integer class labels, shape (N,).
            y_prob: Predicted probabilities, shape (N, C).
            class_idx: Index of the class to inspect.

        Returns:
            Three arrays: (fraction_pos, mean_pred_prob, bin_counts).
        """
        y_bin = (y_true == class_idx).astype(int)
        frac_pos, mean_pred = calibration_curve(
            y_bin, y_prob[:, class_idx], n_bins=self.n_bins, strategy="uniform"
        )
        bin_counts = np.histogram(y_prob[:, class_idx], bins=self.n_bins)[0]
        return frac_pos, mean_pred, bin_counts


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class AtBatPredictor:
    """Trains, calibrates, and serves the PA-outcome multiclass classifier.

    Architecture:
        Base model  : LightGBM multiclass (GBDT with dart or gbdt booster)
        Calibration : ``CalibratedClassifierCV(cv='prefit', method='isotonic')``
        Validation  : Walk-forward temporal cross-validation (TimeSeriesSplit)

    The ``predict_proba`` method always returns calibrated probabilities from
    the post-hoc calibrated model. The raw (uncalibrated) probabilities are
    accessible via ``predict_proba_raw`` for diagnostic purposes.

    Persistence:
        The fitted calibrated model is serialized with ``pickle`` for local
        storage and logged as an MLflow artifact for experiment tracking.

    Args:
        config: Training and hyperparameter configuration.
    """

    def __init__(self, config: Optional[AtBatModelConfig] = None) -> None:
        """Initializes the predictor.

        Args:
            config: Optional configuration; uses defaults if not provided.
        """
        self.config = config or AtBatModelConfig()
        self._base_model: Optional[lgb.LGBMClassifier] = None
        self._calibrated_model = None
        self._ece_computer = ECEComputer(n_bins=self.config.ece_n_bins)
        self._feature_names: Optional[list[str]] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build_lgbm(self) -> lgb.LGBMClassifier:
        """Instantiates the LightGBM base classifier.

        Returns:
            Configured ``LGBMClassifier`` instance.
        """
        cfg = self.config
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=N_CLASSES,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            max_depth=cfg.max_depth,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            max_bin=cfg.max_bin,
            class_weight=cfg.class_weight,
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            verbosity=-1,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
    ) -> "AtBatPredictor":
        """Trains the base LightGBM model and applies post-hoc calibration.

        Training sequence:
        1. Train LightGBM with early stopping on ``(X_val, y_val)``.
        2. Calibrate the base model on ``(X_cal, y_cal)`` — a *separate* held-out
           set that was NOT used for early stopping.  If ``X_cal`` is ``None``,
           falls back to ``(X_val, y_val)`` with a ``calibration_LEAK_WARNING``
           log entry (backwards-compatible but biased ECE).
        3. Compute ECE on both ``X_cal`` (in-domain) and ``X_val`` (holdout) to
           expose the calibration-set vs. evaluation-set gap.
        4. If calibrated ECE > ``config.ece_target``, emit a warning but do
           NOT fall back silently — the caller must decide what to do.
        5. Log all metrics and the calibrated model artifact to MLflow.

        Args:
            X_train: Training feature matrix, shape (N_train, D).
            y_train: Training labels, shape (N_train,). Integer in [0, N_CLASSES-1].
            X_val: Validation feature matrix used for LightGBM early stopping
                and for the *holdout* ECE metric.  Shape (N_val, D).
            y_val: Validation labels for early stopping / holdout ECE, shape (N_val,).
            X_cal: Calibration feature matrix, shape (N_cal, D).  Should be a
                disjoint temporal split *earlier* than ``X_val`` so that the
                isotonic/sigmoid calibrator never sees the evaluation set.
                If ``None``, ``X_val``/``y_val`` are reused with a warning.
            y_cal: Calibration labels, shape (N_cal,).  Required when ``X_cal``
                is not ``None``.
            feature_names: Optional list of D feature names for SHAP/logging.

        Returns:
            ``self`` (fluent interface).

        Raises:
            ValueError: If labels contain values outside [0, N_CLASSES-1].
            ValueError: If ``X_cal`` is provided but ``y_cal`` is ``None``.
        """
        # --- Guard: X_cal and y_cal must be provided together ---
        if (X_cal is None) != (y_cal is None):
            raise ValueError("X_cal and y_cal must both be provided or both be None.")

        # --- Backwards-compatible leak fallback ---
        if X_cal is None:
            log.warning(
                "calibration_LEAK_WARNING",
                msg=(
                    "X_cal is None — using X_val for both early-stopping AND calibration. "
                    "ECE will be optimistically biased by ~0.005–0.012. "
                    "Pass separate X_cal / y_cal to fix this."
                ),
            )
            X_cal_eff, y_cal_eff = X_val, y_val
            _separate_cal = False
        else:
            X_cal_eff, y_cal_eff = X_cal, y_cal
            _separate_cal = True

        # Validate labels
        unique_labels = np.unique(np.concatenate([y_train, y_val, y_cal_eff]))
        invalid = set(unique_labels.tolist()) - set(range(N_CLASSES))
        if invalid:
            raise ValueError(
                f"Labels contain invalid values: {invalid}. Must be in [0, {N_CLASSES - 1}]."
            )

        self._feature_names = feature_names
        self._n_features = X_train.shape[1]
        cfg = self.config

        mlflow.set_experiment(cfg.mlflow_experiment)
        with mlflow.start_run(run_name="lgbm-calibrated-v1") as run:
            self._log_mlflow_params(run.info.run_id)
            mlflow.log_param("separate_calibration_set", _separate_cal)

            # ----------------------------------------------------------
            # Step 1: Train base LightGBM with early stopping on X_val
            # ----------------------------------------------------------
            self._base_model = self._build_lgbm()
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=cfg.early_stopping_rounds, verbose=False
                ),
                lgb.log_evaluation(period=-1),
            ]
            self._base_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=callbacks,
                feature_name=feature_names or "auto",
            )
            best_iter = self._base_model.best_iteration_
            log.info("lgbm_trained", best_iteration=best_iter)

            # Raw probabilities and metrics on the holdout (X_val)
            raw_probs_val = self._base_model.predict_proba(X_val)
            raw_logloss = log_loss(y_val, raw_probs_val)
            raw_ece = self._ece_computer.compute(y_val, raw_probs_val)["overall_ece"]
            log.info("raw_metrics", logloss=round(raw_logloss, 5), ece=round(raw_ece, 5))
            mlflow.log_metrics({"raw_logloss": raw_logloss, "raw_ece": raw_ece})

            # ----------------------------------------------------------
            # Step 2: Post-hoc calibration on the *calibration* set (X_cal)
            # ----------------------------------------------------------
            self._calibrated_model = CalibratedClassifierCV(
                estimator=self._base_model,
                cv="prefit",           # base model already fitted
                method=cfg.calibration_method,
                n_jobs=cfg.n_jobs,
            )
            self._calibrated_model.fit(X_cal_eff, y_cal_eff)   # ← fixed: uses X_cal, not X_val
            log.info(
                "calibration_applied",
                method=cfg.calibration_method,
                cal_samples=len(X_cal_eff),
                separate_set=_separate_cal,
            )

            # ----------------------------------------------------------
            # Step 3: Evaluate calibration quality — TWO metrics
            #   cal_ece_in_domain : ECE on X_cal (the set the calibrator was fit on)
            #   cal_ece_holdout   : ECE on X_val (true unseen evaluation set)
            # When a separate cal set is used, holdout ≥ in_domain should hold.
            # When they are the same set (leak fallback), both are equal and biased.
            # ----------------------------------------------------------
            cal_probs_cal = self._calibrated_model.predict_proba(X_cal_eff)
            ece_in_domain = self._ece_computer.compute(y_cal_eff, cal_probs_cal)["overall_ece"]

            cal_probs_val = self._calibrated_model.predict_proba(X_val)
            cal_logloss   = log_loss(y_val, cal_probs_val)
            ece_result    = self._ece_computer.compute(y_val, cal_probs_val)
            cal_ece       = ece_result["overall_ece"]   # holdout ECE (the real metric)

            log.info(
                "calibrated_metrics",
                cal_ece_in_domain=round(ece_in_domain, 5),
                cal_ece_holdout=round(cal_ece, 5),
                logloss=round(cal_logloss, 5),
                ece_target=cfg.ece_target,
                target_met=cal_ece <= cfg.ece_target,
            )
            mlflow.log_metrics({
                "cal_logloss": cal_logloss,
                "cal_ece": cal_ece,                      # alias kept for dashboard backwards-compat
                "cal_ece_in_domain": ece_in_domain,
                "cal_ece_holdout": cal_ece,
                "ece_reduction_pct": round((raw_ece - cal_ece) / max(raw_ece, 1e-9) * 100, 1),
            })

            # Per-class ECE logging (on holdout)
            for name, ece_c in zip(ece_result["per_class_names"], ece_result["per_class_ece"]):
                mlflow.log_metric(f"ece_{name.lower()}", round(ece_c, 5))

            if cal_ece > cfg.ece_target:
                log.warning(
                    "ece_target_exceeded",
                    achieved=round(cal_ece, 5),
                    target=cfg.ece_target,
                    recommendation=(
                        "Consider: (1) more calibration data, "
                        "(2) switch calibration_method to 'sigmoid', "
                        "(3) tune LightGBM leaf depth for smoother probabilities."
                    ),
                )

            # Log expected run value per PA as a business-level sanity check
            avg_ev = float(np.mean(cal_probs_val @ RUN_VALUES))
            mlflow.log_metric("avg_expected_run_value_per_pa", round(avg_ev, 4))

            # Persist model artifact
            mlflow.sklearn.log_model(self._calibrated_model, artifact_path="calibrated_model")

        self._is_fitted = True
        return self

    def _log_mlflow_params(self, run_id: str) -> None:
        """Logs all model hyperparameters to MLflow.

        Args:
            run_id: Active MLflow run identifier (for logging only).
        """
        cfg = self.config
        mlflow.log_params({
            "n_estimators": cfg.n_estimators,
            "learning_rate": cfg.learning_rate,
            "num_leaves": cfg.num_leaves,
            "min_child_samples": cfg.min_child_samples,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_alpha": cfg.reg_alpha,
            "reg_lambda": cfg.reg_lambda,
            "calibration_method": cfg.calibration_method,
            "class_weight": cfg.class_weight or "none",
            "n_classes": N_CLASSES,
        })

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated probability distribution over PA outcomes.

        This is the primary inference entry point for the simulation engine.
        The returned array is guaranteed to sum to 1.0 per row (up to float32
        precision) and to contain only non-negative values.

        Args:
            X: Feature matrix, shape (N, D). D must match training dimensionality.

        Returns:
            Calibrated probability array, shape (N, 7). dtype = float32.

        Raises:
            RuntimeError: If the predictor has not been fitted yet.
        """
        self._assert_fitted()
        probs = self._calibrated_model.predict_proba(X).astype(np.float32)
        # Renormalize rows to exactly 1.0 (guard against floating-point drift)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs /= np.maximum(row_sums, 1e-9)
        return probs

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Returns raw (uncalibrated) probabilities from the base LightGBM model.

        Use only for diagnostic comparison with calibrated outputs.

        Args:
            X: Feature matrix, shape (N, D).

        Returns:
            Raw probability array, shape (N, 7). dtype = float32.
        """
        self._assert_fitted()
        return self._base_model.predict_proba(X).astype(np.float32)

    def predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Calibrated probabilities for a single PA feature vector.

        Convenience wrapper for the simulation engine's inner loop that
        passes one PA at a time.

        Args:
            x: Single PA feature vector, shape (D,).

        Returns:
            Probability array of shape (7,). dtype = float32.
        """
        return self.predict_proba(x.reshape(1, -1))[0]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_calibration(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """Comprehensive calibration evaluation on a holdout set.

        Computes:
            - Log-loss (overall and per-class)
            - Classwise ECE with per-class breakdown
            - Expected run value sanity check
            - Outcome distribution comparison (predicted vs. actual)

        Args:
            X_val: Feature matrix for evaluation, shape (N, D).
            y_val: True labels, shape (N,).

        Returns:
            Dict with calibration metrics suitable for logging or display.
        """
        self._assert_fitted()
        probs = self.predict_proba(X_val)
        ece_result = self._ece_computer.compute(y_val, probs)

        # Per-class log-loss
        per_class_logloss = {}
        for c in range(N_CLASSES):
            mask = y_val == c
            if mask.sum() == 0:
                continue
            y_bin = (y_val == c).astype(int)
            ll = log_loss(y_bin, probs[:, c], labels=[0, 1])
            per_class_logloss[PAOutcome(c).name] = round(ll, 5)

        # Predicted vs. observed distribution
        pred_dist = probs.mean(axis=0)
        obs_dist = np.bincount(y_val.astype(int), minlength=N_CLASSES) / len(y_val)

        return {
            "overall_logloss": round(log_loss(y_val, probs), 5),
            "overall_ece": round(ece_result["overall_ece"], 5),
            "ece_target_met": ece_result["overall_ece"] <= self.config.ece_target,
            "per_class_logloss": per_class_logloss,
            "per_class_ece": {
                name: round(v, 5)
                for name, v in zip(ece_result["per_class_names"], ece_result["per_class_ece"])
            },
            "predicted_outcome_distribution": {
                PAOutcome(c).name: round(float(pred_dist[c]), 4)
                for c in range(N_CLASSES)
            },
            "observed_outcome_distribution": {
                PAOutcome(c).name: round(float(obs_dist[c]), 4)
                for c in range(N_CLASSES)
            },
            "avg_expected_run_value": round(float(np.mean(probs @ RUN_VALUES)), 5),
        }

    def walk_forward_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time_index: np.ndarray,
    ) -> list[dict]:
        """Temporal walk-forward cross-validation across MLB seasons.

        Respects causal ordering: the model is NEVER trained on data from
        a season that is being evaluated. This is the only correct validation
        strategy for sports time series.

        Args:
            X: Full feature matrix, shape (N, D). Must be sorted by time_index.
            y: Labels, shape (N,).
            time_index: Integer season/date index for each sample, shape (N,).
                Folds are built over the UNIQUE sorted values of this index so
                that a temporal period (e.g. a season) is never split across
                train and validation, regardless of row counts per period.

        Returns:
            List of per-fold metric dicts.

        Raises:
            ValueError: If X is not sorted by time_index (temporal CV would
                silently mix past and future otherwise).
        """
        time_index = np.asarray(time_index)
        if np.any(np.diff(time_index) < 0):
            raise ValueError("X/y must be sorted by time_index for temporal CV.")

        # Split over unique time periods, then map back to row indices.
        periods = np.unique(time_index)
        tscv = TimeSeriesSplit(
            n_splits=self.config.cv_n_splits,
            gap=0,
        )
        fold_results = []

        for fold, (train_p_idx, val_p_idx) in enumerate(tscv.split(periods)):
            train_periods = periods[train_p_idx]
            val_periods   = periods[val_p_idx]
            train_idx = np.flatnonzero(np.isin(time_index, train_periods))
            val_idx   = np.flatnonzero(np.isin(time_index, val_periods))
            log.info("cv_fold_start", fold=fold, train_size=len(train_idx), val_size=len(val_idx))
            fold_predictor = AtBatPredictor(self.config)
            # Split val into two disjoint temporal halves:
            #   cal_idx  → tuning set: early stopping + isotonic calibration
            #   eval_idx → VIRGIN holdout: only read once, in evaluate_calibration
            mid = len(val_idx) // 2
            cal_idx, eval_idx = val_idx[:mid], val_idx[mid:]

            fold_predictor.fit(
                X[train_idx], y[train_idx],     # training data
                X[cal_idx],   y[cal_idx],        # X_val — early stopping (tuning half)
                X[cal_idx],   y[cal_idx],        # X_cal — isotonic calibration (same half)
            )
            metrics = fold_predictor.evaluate_calibration(X[eval_idx], y[eval_idx])
            metrics["fold"] = fold
            fold_results.append(metrics)
            log.info("cv_fold_complete", fold=fold,
                     ece=metrics["overall_ece"], logloss=metrics["overall_logloss"])

        return fold_results

    # ------------------------------------------------------------------
    # SHAP explainability
    # ------------------------------------------------------------------

    def explain(
        self,
        X: np.ndarray,
        max_display: int = 20,
    ) -> np.ndarray:
        """Computes SHAP values for the base LightGBM model.

        Returns the mean absolute SHAP values per feature (global importance)
        across all classes. For per-class SHAP, access ``shap_values[class_idx]``.

        Args:
            X: Feature matrix to explain, shape (N, D). Keep N ≤ 5000 for speed.
            max_display: Number of top features to log.

        Returns:
            Mean absolute SHAP values array, shape (D,).

        Raises:
            RuntimeError: If model is not fitted.
            ImportError: If ``shap`` is not installed.
        """
        self._assert_fitted()
        try:
            import shap  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install 'shap' to use explain().") from exc

        explainer = shap.TreeExplainer(self._base_model)
        shap_values = explainer.shap_values(X)  # list of (N, D) arrays, one per class

        # Mean absolute SHAP across all classes
        mean_abs_shap = np.abs(np.stack(shap_values)).mean(axis=(0, 1))

        if self._feature_names:
            top_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
            log.info(
                "shap_top_features",
                top_features=[
                    {"name": self._feature_names[i], "importance": round(float(mean_abs_shap[i]), 5)}
                    for i in top_idx
                ],
            )
        return mean_abs_shap

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serializes the fitted predictor to disk via pickle.

        Saves both the calibrated model and the full config so the predictor
        can be fully reconstructed with ``AtBatPredictor.load()``.

        Args:
            path: Destination file path (e.g. ``models/pa_predictor_v1.pkl``).

        Raises:
            RuntimeError: If the predictor is not fitted.
        """
        self._assert_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        payload = {
            "calibrated_model": self._calibrated_model,
            "base_model": self._base_model,
            "config": self.config,
            "feature_names": self._feature_names,
            "n_features": self._n_features,
            # Fecha de despliegue: fuente de verdad para los gates de regresión,
            # que solo deben calificar predicciones generadas con ESTE modelo.
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("predictor_saved", path=path, n_features=self._n_features)

    @classmethod
    def load(cls, path: str) -> "AtBatPredictor":
        """Loads a previously saved predictor from disk.

        Args:
            path: Path to the pickle file created by ``save()``.

        Returns:
            Fitted ``AtBatPredictor`` ready for inference.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Custom unpickler: remaps __main__.ClassName → this module so pickles
        # created from a training script (where classes lived in __main__) can
        # be loaded safely inside gunicorn/uvicorn workers.
        #
        # Endurecimiento de seguridad (audit F12): pickle puede ejecutar código
        # arbitrario al deserializar (reduce/__reduce__). Restringimos find_class
        # a una allowlist de módulos de confianza (el propio modelo + el stack ML)
        # y bloqueamos cualquier otro global. Mitiga la carga de un .pkl manipulado
        # sin cambiar el formato del artefacto existente.
        import sys
        _this_module = sys.modules[__name__]

        _ALLOWED_PREFIXES = (
            "sklearn.", "lightgbm", "numpy", "scipy.",
            "collections", "copyreg", "functools",
            "src.models.model_at_bat", __name__,
        )
        _ALLOWED_BUILTINS = {
            "builtins": {"list", "dict", "set", "tuple", "frozenset",
                         "int", "float", "bool", "str", "bytes", "complex",
                         "object", "type", "slice", "range"},
        }

        class _SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "__main__" and hasattr(_this_module, name):
                    return getattr(_this_module, name)
                if module in _ALLOWED_BUILTINS:
                    if name in _ALLOWED_BUILTINS[module]:
                        return super().find_class(module, name)
                    raise pickle.UnpicklingError(
                        f"Builtin no permitido al cargar el modelo: {module}.{name}"
                    )
                if module == __name__ or module.startswith(_ALLOWED_PREFIXES):
                    return super().find_class(module, name)
                raise pickle.UnpicklingError(
                    f"Módulo no permitido al cargar el modelo: {module}.{name}. "
                    "Si es legítimo, añádelo a _ALLOWED_PREFIXES (audit F12)."
                )

        with open(path, "rb") as f:
            payload = _SafeUnpickler(f).load()
        predictor = cls(config=payload["config"])
        predictor._calibrated_model = payload["calibrated_model"]
        predictor._base_model       = payload["base_model"]
        predictor._feature_names    = payload["feature_names"]
        predictor._n_features       = payload.get("n_features")   # None for legacy pickles
        predictor._is_fitted        = True

        # Guard against silent dimensionality mismatches at serving time
        n_classes = getattr(predictor._calibrated_model, "classes_", None)
        if n_classes is not None and len(n_classes) != N_CLASSES:
            raise ValueError(
                f"Loaded model has {len(n_classes)} classes, expected {N_CLASSES}. "
                "The model was trained with a different N_CLASSES — retrain required."
            )

        log.info(
            "predictor_loaded",
            path=path,
            n_features=predictor._n_features,
            n_classes=N_CLASSES,
        )
        return predictor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        """Raises RuntimeError if the predictor has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "AtBatPredictor is not fitted. Call fit() before predict_proba()."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"AtBatPredictor({status}, "
            f"calibration={self.config.calibration_method}, "
            f"n_classes={N_CLASSES})"
        )


# ---------------------------------------------------------------------------
# Feature matrix builder (upstream pipeline contract)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Feature-specific imputation values (Mejora 2)
# ---------------------------------------------------------------------------
# When a context column is present but has nulls, these medians replace NaN
# instead of the generic 0.0 fallback, preserving distributional meaning.
# All values are Statcast 2015-2024 MLB medians unless noted.
_CONTEXT_MEDIAN_IMPUTES: dict[str, float] = {
    "pitch_count_in_pa":      4.0,   # median PA length (pitches)
    # Era indicator columns default to 0 (pre-era); no custom impute needed.
}


def build_feature_matrix(
    rolling_df: pl.DataFrame,
    platoon_df: pl.DataFrame,
    embeddings_batter: np.ndarray,
    embeddings_pitcher: np.ndarray,
    context_df: pl.DataFrame,
    pitcher_rolling_df: Optional[pl.DataFrame] = None,
) -> tuple[np.ndarray, list[str]]:
    """Assembles the full ~340-dimensional feature vector for each PA.

    This function is the sole bridge between the feature pipeline outputs
    (Phases 2.1, 2.2, 2.3) and the model inputs, ensuring a single
    definition of the feature space that is identical in training and serving
    (the Training-Serving Skew prevention contract).

    Feature groups and approximate sizes:
        - Rolling batter stats (7d/15d/30d xwOBA, EV, K%, BB%, HR%): ~20 dims
        - Platoon splits stabilized (by hand, by pitch type): ~28 dims
        - Batter embedding (16-dim latent): 16 dims
        - Pitcher embedding (16-dim latent): 16 dims
        - Pitcher arsenal (per pitch type: velocity, spin, break, usage): ~63 dims
        - Weather vector (temp, humidity, wind_u, wind_v, pressure): 5 dims
        - Park factors (dynamic HR, XB, 1B factors): 3 dims
        - Contextual (inning, outs, bases_state, score_diff,
                      pitch_count_in_pa [Mejora 2],
                      era_shift_ban, era_universal_dh,
                      era_first_year_shift_ban [Mejora 4]): ~9 dims
        - Season/era encoding (cyclic sin/cos): 4 dims
        - Pitcher rolling features (velo, spin, fatigue [Mejora 3]): ~18 dims (optional)
        Total: ~169 dims base + pitch arsenal ≈ ~224–242 dims

    Args:
        rolling_df: Batter rolling features from Phase 2.2 (one row per batter).
        platoon_df: Platoon split stabilized metrics from Phase 2.1.
        embeddings_batter: Batter latent vectors, shape (N_batters, 16).
        embeddings_pitcher: Pitcher latent vectors, shape (N_pitchers, 16).
        context_df: Game context features (venue, weather, inning, etc.).
            Must contain ``pitch_count_in_pa`` (Mejora 2) and era columns
            (Mejora 4) if available.  Missing columns are silently omitted.
        pitcher_rolling_df: Optional pitcher rolling features from
            ``features_pitcher_rolling.py`` (Mejora 3).  When ``None`` the
            pitcher rolling block is omitted — backwards compatible.

    Returns:
        Tuple of (feature_matrix, feature_names) where feature_matrix is
        shape (N, D) float32 and feature_names is a list of D column names.

    Note:
        In production, all DataFrames are pre-joined by
        (batter_id, pitcher_id, game_date) before passing here.

    Serving note:
        ``pitch_count_in_pa`` has two serving modes:
          - Pre-game lineup prediction: impute with 4.0 (historical median).
          - Live in-PA tracking: use the real-time pitch count from the
            Statcast polling feed (``statcast_ingestion.py``).
        Both modes call this function identically; the difference is upstream
        (the context_df row passed in).
    """
    # Collect feature blocks
    rolling_cols = [c for c in rolling_df.columns if c not in {"batter_id", "feature_date"}]
    platoon_cols = [c for c in platoon_df.columns if c.endswith("_stabilized")]
    context_cols = [c for c in context_df.columns if c not in {"batter_id", "pitcher_id", "game_date"}]

    rolling_arr = rolling_df.select(rolling_cols).to_numpy(allow_copy=True).astype(np.float32)
    platoon_arr = platoon_df.select(platoon_cols).to_numpy(allow_copy=True).astype(np.float32)
    context_arr = context_df.select(context_cols).to_numpy(allow_copy=True).astype(np.float32)

    # --- Feature-specific null imputation (Mejora 2) ---
    # Apply domain-aware imputes BEFORE the generic nan_to_num below.
    for col_name, fill_value in _CONTEXT_MEDIAN_IMPUTES.items():
        if col_name in context_cols:
            col_idx = context_cols.index(col_name)
            null_mask = np.isnan(context_arr[:, col_idx])
            if null_mask.any():
                context_arr[null_mask, col_idx] = fill_value

    # Assemble blocks
    blocks: list[np.ndarray] = [
        rolling_arr,
        platoon_arr,
        embeddings_batter.astype(np.float32),
        embeddings_pitcher.astype(np.float32),
        context_arr,
    ]
    names: list[str] = (
        rolling_cols
        + platoon_cols
        + [f"batter_z_{i}" for i in range(embeddings_batter.shape[1])]
        + [f"pitcher_z_{i}" for i in range(embeddings_pitcher.shape[1])]
        + context_cols
    )

    # --- Pitcher rolling features (Mejora 3, optional) ---
    if pitcher_rolling_df is not None:
        p_exclude = {"pitcher_id", "feature_date", "season"}
        pitcher_roll_cols = [c for c in pitcher_rolling_df.columns if c not in p_exclude]
        pitcher_roll_arr  = (
            pitcher_rolling_df.select(pitcher_roll_cols)
            .to_numpy(allow_copy=True)
            .astype(np.float32)
        )
        blocks.append(pitcher_roll_arr)
        names.extend(pitcher_roll_cols)

    feature_matrix = np.hstack(blocks)

    # Replace remaining NaN (missing rolling windows for new players) with 0.0
    np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    return feature_matrix, names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PA-level outcome predictor — train & evaluate")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--features-path", required=True, dest="features_path")
    train_p.add_argument("--output-dir", default="models/", dest="output_dir")
    train_p.add_argument("--val-seasons", default="2024", dest="val_seasons",
                         help="Comma-separated season years to use as validation. "
                              "Default changed to 2024 so that the era_shift_ban (2023+) "
                              "feature is included in training rather than held out.")
    train_p.add_argument("--experiment", default="pa-model-at-bat", dest="experiment")

    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--model-path", required=True, dest="model_path")
    eval_p.add_argument("--features-path", required=True, dest="features_path")
    eval_p.add_argument("--eval-season", type=int, default=2024, dest="eval_season")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for model training and evaluation.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]`` when ``None``.
    """
    args = _parse_args(argv)

    if args.command == "train":
        df = pl.read_parquet(args.features_path)
        val_seasons = [int(s) for s in args.val_seasons.split(",")]

        train_mask = ~df["season"].is_in(val_seasons)
        val_mask   = df["season"].is_in(val_seasons)

        label_col = "pa_outcome_idx"
        # Exclude per-PA Statcast outcome metrics: these are derived FROM the PA being
        # predicted and introduce target leakage, making the model unusable for inference.
        # pitch_count_in_pa and last_pitch_type are intra-PA information (only known
        # once the PA ends) — same leakage class as xwoba.
        # Also exclude identifier columns (batter_id, pitcher_id) that are not predictive
        # features (they are high-cardinality IDs, not numeric features).
        _LEAKING = {"xwoba", "launch_speed", "launch_angle",
                    "pitch_count_in_pa", "last_pitch_type"}
        _NON_FEATURES = {"batter_id", "pitcher_id", label_col, "season", "game_date"}
        feature_cols = [
            c for c in df.columns
            if c not in _NON_FEATURES | _LEAKING
            and df[c].dtype not in (pl.Utf8, pl.String, pl.Categorical)
        ]

        X_train = df.filter(train_mask).select(feature_cols).to_numpy().astype(np.float32)
        y_train = df.filter(train_mask)[label_col].to_numpy().astype(np.int32)

        # Split val into two chronological halves:
        #   First half  → calibration (X_cal): the isotonic/sigmoid calibrator is fit here.
        #   Second half → evaluation  (X_val): early stopping signal AND holdout ECE.
        # Sorting by game_date ensures temporal ordering before the split.
        val_df_sorted = df.filter(val_mask).sort("game_date")
        mid = len(val_df_sorted) // 2
        X_cal = val_df_sorted[:mid].select(feature_cols).to_numpy().astype(np.float32)
        y_cal = val_df_sorted[:mid][label_col].to_numpy().astype(np.int32)
        X_val = val_df_sorted[mid:].select(feature_cols).to_numpy().astype(np.float32)
        y_val = val_df_sorted[mid:][label_col].to_numpy().astype(np.int32)

        # class_weight=None: el modelo alimenta un simulador que consume las
        # probabilidades como pesos estocásticos — upweighting de clases raras
        # sesga E[R]. Ver train_v3.py (gate de prior drift).
        config = AtBatModelConfig(
            mlflow_experiment=args.experiment,
            class_weight=None,
        )
        predictor = AtBatPredictor(config)
        predictor.fit(
            X_train, y_train,
            X_val, y_val,
            X_cal, y_cal,
            feature_names=feature_cols,
        )
        predictor.save(f"{args.output_dir}/pa_predictor_v1.pkl")

    elif args.command == "evaluate":
        predictor = AtBatPredictor.load(args.model_path)
        df = pl.read_parquet(args.features_path).filter(pl.col("season") == args.eval_season)
        feature_cols = [c for c in df.columns if c not in {"pa_outcome_idx", "season", "game_date"}]
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y = df["pa_outcome_idx"].to_numpy().astype(np.int32)
        metrics = predictor.evaluate_calibration(X, y)
        import json  # noqa: PLC0415
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
