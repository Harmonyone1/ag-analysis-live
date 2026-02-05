"""Training pipeline for the AI gate XGBoost model.

Handles:
- Walk-forward cross-validation
- XGBoost training with early stopping
- Platt scaling (logistic regression on logit scores) for calibration
- Calibration quality metrics (Brier score)
- Model persistence with versioned metadata
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the gate model training pipeline."""

    # XGBoost params
    max_depth: int = 6
    n_estimators: int = 500
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 20
    early_stopping_rounds: int = 30
    eval_metric: str = "logloss"

    # Calibration
    calibration_method: str = "platt"  # "platt" or "isotonic"
    min_validation_samples_for_platt: int = 100

    # Walk-forward
    n_folds: int = 5
    validation_fraction: float = 0.2

    # Output
    feature_version: str = "v3.0.0"
    model_dir: str = "models"


@dataclass
class TrainingResult:
    """Result of a full training run."""

    model_path: str
    n_samples: int
    n_features: int
    feature_version: str
    calibration_method: str
    train_auc: float
    val_auc: float
    raw_brier: float
    calibrated_brier: float
    walk_forward_metrics: List[Dict[str, float]]
    metadata: Dict[str, Any]


class TrainingPipeline:
    """End-to-end training pipeline for the AI gate model."""

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or TrainingConfig()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        symbols: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> TrainingResult:
        """Train the gate model with walk-forward validation and calibration.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape ``(n_samples, n_features)``.
        y : np.ndarray
            Binary labels (1=WIN, 0=LOSS/TIMEOUT).
        symbols : optional
            Per-sample symbol array for stratified splitting.
        feature_names : optional
            Human-readable feature names.
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score, brier_score_loss
        from xgboost import XGBClassifier

        t0 = time.time()
        n_samples, n_features = X.shape
        logger.info(
            "Training pipeline start: %d samples, %d features, version=%s",
            n_samples,
            n_features,
            self.config.feature_version,
        )

        # ------------------------------------------------------------------
        # Walk-forward cross-validation
        # ------------------------------------------------------------------
        tscv = TimeSeriesSplit(n_splits=self.config.n_folds)
        fold_metrics: List[Dict[str, float]] = []
        all_val_probs: List[np.ndarray] = []
        all_val_labels: List[np.ndarray] = []

        best_model = None
        best_val_auc = -1.0

        for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = XGBClassifier(
                max_depth=self.config.max_depth,
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                min_child_weight=self.config.min_child_weight,
                eval_metric=self.config.eval_metric,
                use_label_encoder=False,
                random_state=42 + fold_i,
                early_stopping_rounds=self.config.early_stopping_rounds,
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            p_val = model.predict_proba(X_val)[:, 1]
            p_train = model.predict_proba(X_train)[:, 1]

            try:
                auc_val = roc_auc_score(y_val, p_val)
            except ValueError:
                auc_val = 0.5
            try:
                auc_train = roc_auc_score(y_train, p_train)
            except ValueError:
                auc_train = 0.5

            brier_val = brier_score_loss(y_val, p_val)

            fold_metrics.append({
                "fold": fold_i,
                "train_auc": auc_train,
                "val_auc": auc_val,
                "val_brier": brier_val,
                "train_size": len(y_train),
                "val_size": len(y_val),
                "win_rate_train": float(y_train.mean()),
                "win_rate_val": float(y_val.mean()),
                "overfit_gap": auc_train - auc_val,
            })

            if auc_train - auc_val > 0.15:
                logger.warning(
                    "Fold %d: overfit detected — train AUC %.3f vs val AUC %.3f (gap %.3f)",
                    fold_i,
                    auc_train,
                    auc_val,
                    auc_train - auc_val,
                )

            logger.info(
                "Fold %d: train_AUC=%.3f val_AUC=%.3f val_Brier=%.4f "
                "train=%d val=%d",
                fold_i,
                auc_train,
                auc_val,
                brier_val,
                len(y_train),
                len(y_val),
            )

            all_val_probs.append(p_val)
            all_val_labels.append(y_val)

            if auc_val > best_val_auc:
                best_val_auc = auc_val
                best_model = model

        # ------------------------------------------------------------------
        # Aggregate validation predictions for calibration
        # ------------------------------------------------------------------
        val_probs = np.concatenate(all_val_probs)
        val_labels = np.concatenate(all_val_labels)

        raw_brier = brier_score_loss(val_labels, val_probs)
        try:
            overall_val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            overall_val_auc = 0.5

        logger.info(
            "Overall walk-forward: val_AUC=%.3f raw_Brier=%.4f (%d samples)",
            overall_val_auc,
            raw_brier,
            len(val_labels),
        )

        # ------------------------------------------------------------------
        # Train final model on all data
        # ------------------------------------------------------------------
        final_model = XGBClassifier(
            max_depth=self.config.max_depth,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            eval_metric=self.config.eval_metric,
            use_label_encoder=False,
            random_state=42,
        )
        # Use last 20% as eval set for early stopping on final model
        split_idx = int(len(X) * 0.8)
        final_model.fit(
            X[:split_idx],
            y[:split_idx],
            eval_set=[(X[split_idx:], y[split_idx:])],
            verbose=False,
        )

        p_all = final_model.predict_proba(X)[:, 1]
        try:
            train_auc = roc_auc_score(y, p_all)
        except ValueError:
            train_auc = 0.5

        # ------------------------------------------------------------------
        # Calibration
        # ------------------------------------------------------------------
        calibrator, cal_method = self._fit_calibrator(val_probs, val_labels)

        # Compute calibrated Brier score
        cal_probs = self._apply_calibrator(calibrator, cal_method, val_probs)
        calibrated_brier = brier_score_loss(val_labels, cal_probs)

        logger.info(
            "Calibration (%s): raw_Brier=%.4f → calibrated_Brier=%.4f",
            cal_method,
            raw_brier,
            calibrated_brier,
        )

        # Unique calibrated values
        n_unique_raw = len(set(cal_probs.tolist()))
        n_unique_rounded = len(set(round(p, 3) for p in cal_probs.tolist()))
        logger.info(
            "Calibrated P(win) uniqueness: %d raw unique values, "
            "%d after rounding to 0.001",
            n_unique_raw,
            n_unique_rounded,
        )

        # P(win) range
        p_min, p_max = float(cal_probs.min()), float(cal_probs.max())
        logger.info("Calibrated P(win) range: [%.3f, %.3f]", p_min, p_max)

        # ------------------------------------------------------------------
        # Save model
        # ------------------------------------------------------------------
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_name = f"gate_model_{self.config.feature_version}-{timestamp}.pkl"
        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / model_name

        metadata = {
            "feature_version": self.config.feature_version,
            "calibration_method": cal_method,
            "n_features": n_features,
            "n_samples": n_samples,
            "train_auc": train_auc,
            "val_auc": overall_val_auc,
            "raw_brier": raw_brier,
            "calibrated_brier": calibrated_brier,
            "n_unique_calibrated": n_unique_raw,
            "p_win_range": [p_min, p_max],
            "trained_at": timestamp,
            "feature_names": feature_names,
            "config": {
                "max_depth": self.config.max_depth,
                "n_estimators": self.config.n_estimators,
                "learning_rate": self.config.learning_rate,
                "n_folds": self.config.n_folds,
            },
        }

        payload = {
            "model": final_model,
            "calibrator": calibrator,
            "metadata": metadata,
        }

        with open(model_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        elapsed = time.time() - t0
        logger.info(
            "Model saved to %s (%.1fs elapsed, %d samples, version=%s)",
            model_path,
            elapsed,
            n_samples,
            self.config.feature_version,
        )

        # ------------------------------------------------------------------
        # Go / no-go evaluation
        # ------------------------------------------------------------------
        self._evaluate_go_no_go(
            n_samples=n_samples,
            val_auc=overall_val_auc,
            calibrated_brier=calibrated_brier,
            cal_probs=cal_probs,
            val_labels=val_labels,
            n_unique=n_unique_raw,
            p_min=p_min,
            p_max=p_max,
        )

        return TrainingResult(
            model_path=str(model_path),
            n_samples=n_samples,
            n_features=n_features,
            feature_version=self.config.feature_version,
            calibration_method=cal_method,
            train_auc=train_auc,
            val_auc=overall_val_auc,
            raw_brier=raw_brier,
            calibrated_brier=calibrated_brier,
            walk_forward_metrics=fold_metrics,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _fit_calibrator(
        self,
        p_win: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Any, str]:
        """Fit a probability calibrator.

        Platt scaling operates on logit(p_win) — the standard approach
        that avoids range compression from working on raw probabilities.

        Falls back to isotonic regression if validation set is too small.
        """
        n_val = len(y)
        method = self.config.calibration_method

        if method == "platt" and n_val < self.config.min_validation_samples_for_platt:
            logger.warning(
                "Only %d validation samples (need %d for Platt). "
                "Falling back to isotonic regression.",
                n_val,
                self.config.min_validation_samples_for_platt,
            )
            method = "isotonic"

        if method == "platt":
            from sklearn.linear_model import LogisticRegression

            p_clipped = np.clip(p_win, 1e-6, 1 - 1e-6)
            logit_scores = np.log(p_clipped / (1.0 - p_clipped))
            calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
            calibrator.fit(logit_scores.reshape(-1, 1), y)
            logger.info(
                "Platt calibrator fitted: coef=%.4f intercept=%.4f",
                float(calibrator.coef_[0][0]),
                float(calibrator.intercept_[0]),
            )
            return calibrator, "platt"

        else:
            from sklearn.isotonic import IsotonicRegression

            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(p_win, y)
            logger.info("Isotonic calibrator fitted on %d samples", n_val)
            return calibrator, "isotonic"

    @staticmethod
    def _apply_calibrator(
        calibrator: Any,
        method: str,
        p_win: np.ndarray,
    ) -> np.ndarray:
        """Apply the fitted calibrator to raw P(win) values."""
        if method == "platt":
            p_clipped = np.clip(p_win, 1e-6, 1 - 1e-6)
            logit_scores = np.log(p_clipped / (1.0 - p_clipped))
            return calibrator.predict_proba(logit_scores.reshape(-1, 1))[:, 1]
        else:
            return np.array(calibrator.transform(p_win))

    # ------------------------------------------------------------------
    # Go / no-go checks
    # ------------------------------------------------------------------
    def _evaluate_go_no_go(
        self,
        *,
        n_samples: int,
        val_auc: float,
        calibrated_brier: float,
        cal_probs: np.ndarray,
        val_labels: np.ndarray,
        n_unique: int,
        p_min: float,
        p_max: float,
    ) -> None:
        """Log go/no-go evaluation against deployment criteria."""
        logger.info("=" * 60)
        logger.info("GO / NO-GO EVALUATION")
        logger.info("=" * 60)

        checks: List[Tuple[str, bool, str]] = []

        # Total samples
        checks.append((
            "Total samples",
            n_samples >= 1000,
            f"{n_samples} (min=1000, target=2000+)",
        ))

        # Walk-forward AUC
        checks.append((
            "Walk-forward AUC(win)",
            val_auc >= 0.55,
            f"{val_auc:.3f} (min=0.55, target=0.65+)",
        ))

        # Calibrated Brier
        checks.append((
            "Calibrated Brier score",
            calibrated_brier < 0.25,
            f"{calibrated_brier:.4f} (max=0.25, target=<0.20)",
        ))

        # Win rate at P(win) >= 0.55
        mask_55 = cal_probs >= 0.55
        if mask_55.sum() >= 50:
            wr_55 = float(val_labels[mask_55].mean())
            checks.append((
                "Win rate at P(win)>=0.55",
                wr_55 > 0.50,
                f"{wr_55:.3f} (min=0.50, N={mask_55.sum()})",
            ))
        else:
            checks.append((
                "Win rate at P(win)>=0.55",
                False,
                f"UNRELIABLE — only {mask_55.sum()} samples (need 50+)",
            ))

        # Win rate at P(win) >= 0.65
        mask_65 = cal_probs >= 0.65
        if mask_65.sum() >= 50:
            wr_65 = float(val_labels[mask_65].mean())
            checks.append((
                "Win rate at P(win)>=0.65",
                wr_65 > 0.55,
                f"{wr_65:.3f} (min=0.55, N={mask_65.sum()})",
            ))
        else:
            checks.append((
                "Win rate at P(win)>=0.65",
                False,
                f"UNRELIABLE — only {mask_65.sum()} samples (need 50+)",
            ))

        # Unique calibrated values
        checks.append((
            "Unique calibrated P(win) values",
            n_unique > 50,
            f"{n_unique} (min=50, target=100+)",
        ))

        # P(win) range
        range_ok = p_min <= 0.10 and p_max >= 0.85
        checks.append((
            "P(win) range",
            range_ok,
            f"[{p_min:.3f}, {p_max:.3f}] (min=[0.10, 0.85])",
        ))

        # Approval rate
        threshold = 0.55
        n_approved = int((cal_probs >= threshold).sum())
        approval_rate = n_approved / len(cal_probs) * 100 if len(cal_probs) > 0 else 0
        approval_ok = 1 <= approval_rate <= 15
        checks.append((
            "Approval rate",
            approval_ok,
            f"{approval_rate:.1f}% (target=1-15%, N_approved={n_approved})",
        ))

        # EV on approved trades
        if n_approved > 0:
            approved_labels = val_labels[cal_probs >= threshold]
            ev = float(approved_labels.mean()) * 2 - 1  # simplified EV
            checks.append((
                "EV on approved trades",
                ev > 0,
                f"{ev:.3f} (min=0, target=0.3+)",
            ))

        # Report
        all_pass = True
        for name, passed, detail in checks:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            logger.info("  [%s] %s: %s", status, name, detail)

        logger.info("=" * 60)
        if all_pass:
            logger.info("RESULT: ALL CHECKS PASSED — model is deployable")
        else:
            logger.warning("RESULT: SOME CHECKS FAILED — do NOT deploy without investigation")
        logger.info("=" * 60)
