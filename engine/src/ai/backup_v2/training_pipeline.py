"""Training Pipeline - Complete ML training system for AI gate.

Implements:
- Time-based splits (NO random splitting)
- Walk-forward validation
- 3-class classifier (WIN/LOSS/TIMEOUT)
- R-multiple regressor
- Platt scaling calibration
- Feature versioning
"""

import json
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model settings
    model_type: str = "xgboost"  # xgboost, lightgbm
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1

    # Split ratios (time-based, NOT random)
    train_ratio: float = 0.65
    val_ratio: float = 0.20
    test_ratio: float = 0.15

    # Walk-forward settings
    walk_forward_windows: int = 5
    min_train_samples: int = 200

    # Calibration
    calibrate: bool = True
    calibration_method: str = "isotonic"  # "platt" or "isotonic"

    # Feature versioning
    feature_version: str = "v1.0.0"

    # Random seed for reproducibility
    random_state: int = 42


@dataclass
class DataSplit:
    """Time-based data split."""
    X_train: np.ndarray
    y_train_class: np.ndarray  # 3-class labels
    y_train_r: np.ndarray  # R-multiple targets
    times_train: List[datetime]

    X_val: np.ndarray
    y_val_class: np.ndarray
    y_val_r: np.ndarray
    times_val: List[datetime]

    X_test: np.ndarray
    y_test_class: np.ndarray
    y_test_r: np.ndarray
    times_test: List[datetime]


@dataclass
class TrainedModel:
    """Container for trained model artifacts."""
    classifier: Any
    regressor: Any
    calibrator: Optional[Any]  # Platt/isotonic calibrator
    feature_names: List[str]
    feature_version: str
    config: TrainingConfig
    metrics: Dict[str, float]
    training_samples: int
    trained_at: datetime
    version: str


class TrainingPipeline:
    """Complete training pipeline for AI gate models.

    Example:
        pipeline = TrainingPipeline(config)
        pipeline.load_data(features, labels_class, labels_r, timestamps)
        pipeline.train()
        pipeline.evaluate()
        pipeline.save("models/")
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize training pipeline.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()

        self._features: Optional[np.ndarray] = None
        self._labels_class: Optional[np.ndarray] = None
        self._labels_r: Optional[np.ndarray] = None
        self._timestamps: Optional[List[datetime]] = None
        self._feature_names: List[str] = []

        self._split: Optional[DataSplit] = None
        self._classifier = None
        self._regressor = None
        self._calibrator = None
        self._metrics: Dict[str, float] = {}

    def load_data(
        self,
        features: np.ndarray,
        labels_class: np.ndarray,  # 0=LOSS, 1=WIN, 2=TIMEOUT
        labels_r: np.ndarray,  # R-multiples
        timestamps: List[datetime],
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Load training data.

        IMPORTANT: Data must be sorted by timestamp (oldest first).

        Args:
            features: Feature matrix (n_samples, n_features)
            labels_class: Classification labels (0, 1, 2)
            labels_r: R-multiple regression targets
            timestamps: Timestamp for each sample
            feature_names: Feature names for interpretability
        """
        # Verify data is time-sorted
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i-1]:
                raise ValueError("Data must be sorted by timestamp (oldest first)")

        self._features = features
        self._labels_class = labels_class
        self._labels_r = labels_r
        self._timestamps = timestamps
        self._feature_names = feature_names or [f"f_{i}" for i in range(features.shape[1])]

        logger.info(
            "Data loaded",
            samples=len(labels_class),
            features=features.shape[1],
            win_rate=np.mean(labels_class == 1),
            timeout_rate=np.mean(labels_class == 2),
            time_range=f"{timestamps[0]} to {timestamps[-1]}",
        )

    def create_time_splits(self) -> DataSplit:
        """Create time-based train/val/test splits.

        NO RANDOM SPLITTING - strictly chronological.

        Returns:
            DataSplit with train/val/test arrays
        """
        if self._features is None:
            raise ValueError("No data loaded")

        n = len(self._features)

        # Calculate split indices
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        logger.info(
            "Creating time-based splits",
            train=f"0:{train_end}",
            val=f"{train_end}:{val_end}",
            test=f"{val_end}:{n}",
        )

        self._split = DataSplit(
            X_train=self._features[:train_end],
            y_train_class=self._labels_class[:train_end],
            y_train_r=self._labels_r[:train_end],
            times_train=self._timestamps[:train_end],

            X_val=self._features[train_end:val_end],
            y_val_class=self._labels_class[train_end:val_end],
            y_val_r=self._labels_r[train_end:val_end],
            times_val=self._timestamps[train_end:val_end],

            X_test=self._features[val_end:],
            y_test_class=self._labels_class[val_end:],
            y_test_r=self._labels_r[val_end:],
            times_test=self._timestamps[val_end:],
        )

        return self._split

    def train(self) -> Dict[str, float]:
        """Train both classifier and regressor.

        Returns:
            Dict with training metrics
        """
        if self._split is None:
            self.create_time_splits()

        logger.info("Training classifier...")
        self._classifier = self._train_classifier()

        logger.info("Training regressor...")
        self._regressor = self._train_regressor()

        if self.config.calibrate:
            logger.info("Calibrating classifier...")
            self._calibrator = self._calibrate_classifier()

        # Evaluate on validation set
        self._metrics = self._evaluate()

        return self._metrics

    def _train_classifier(self):
        """Train 3-class classifier (WIN/LOSS/TIMEOUT)."""
        if self.config.model_type == "xgboost":
            return self._build_xgb_classifier()
        elif self.config.model_type == "lightgbm":
            return self._build_lgb_classifier()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _build_xgb_classifier(self):
        """Build and train XGBoost classifier."""
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                objective="multi:softprob",
                num_class=3,
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric="mlogloss",
            )

            model.fit(
                self._split.X_train,
                self._split.y_train_class,
                eval_set=[(self._split.X_val, self._split.y_val_class)],
                verbose=False,
            )

            return model

        except ImportError:
            logger.warning("XGBoost not available, using sklearn")
            return self._build_sklearn_classifier()

    def _build_lgb_classifier(self):
        """Build and train LightGBM classifier."""
        try:
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                objective="multiclass",
                num_class=3,
                random_state=self.config.random_state,
                verbose=-1,
            )

            model.fit(
                self._split.X_train,
                self._split.y_train_class,
                eval_set=[(self._split.X_val, self._split.y_val_class)],
            )

            return model

        except ImportError:
            logger.warning("LightGBM not available, using sklearn")
            return self._build_sklearn_classifier()

    def _build_sklearn_classifier(self):
        """Fallback sklearn classifier."""
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
        )

        model.fit(self._split.X_train, self._split.y_train_class)
        return model

    def _train_regressor(self):
        """Train R-multiple regressor."""
        if self.config.model_type == "xgboost":
            return self._build_xgb_regressor()
        elif self.config.model_type == "lightgbm":
            return self._build_lgb_regressor()
        else:
            return self._build_sklearn_regressor()

    def _build_xgb_regressor(self):
        """Build and train XGBoost regressor."""
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
            )

            model.fit(
                self._split.X_train,
                self._split.y_train_r,
                eval_set=[(self._split.X_val, self._split.y_val_r)],
                verbose=False,
            )

            return model

        except ImportError:
            return self._build_sklearn_regressor()

    def _build_lgb_regressor(self):
        """Build and train LightGBM regressor."""
        try:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                verbose=-1,
            )

            model.fit(
                self._split.X_train,
                self._split.y_train_r,
                eval_set=[(self._split.X_val, self._split.y_val_r)],
            )

            return model

        except ImportError:
            return self._build_sklearn_regressor()

    def _build_sklearn_regressor(self):
        """Fallback sklearn regressor."""
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
        )

        model.fit(self._split.X_train, self._split.y_train_r)
        return model

    def _calibrate_classifier(self):
        """Calibrate classifier probabilities.

        Uses validation set for calibration.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import clone

        if self.config.calibration_method == "platt":
            method = "sigmoid"
        else:
            method = "isotonic"

        # Get raw probabilities on validation set
        val_proba = self._classifier.predict_proba(self._split.X_val)

        # Create calibrator trained on validation predictions
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        # Calibrate P(WIN) specifically
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
        else:
            calibrator = LogisticRegression()

        # Fit calibrator on P(WIN) vs actual WIN outcomes
        p_win = val_proba[:, 1]  # P(WIN)
        y_win = (self._split.y_val_class == 1).astype(int)

        calibrator.fit(p_win.reshape(-1, 1), y_win)

        return calibrator

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate models on validation set.

        Returns:
            Dict with comprehensive metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, mean_squared_error, mean_absolute_error,
            brier_score_loss, log_loss,
        )

        # Get predictions
        y_pred_class = self._classifier.predict(self._split.X_val)
        y_pred_proba = self._classifier.predict_proba(self._split.X_val)
        y_pred_r = self._regressor.predict(self._split.X_val)

        # Classification metrics
        metrics = {
            "accuracy": accuracy_score(self._split.y_val_class, y_pred_class),
            "precision_win": precision_score(
                self._split.y_val_class, y_pred_class, labels=[1], average="macro", zero_division=0
            ),
            "recall_win": recall_score(
                self._split.y_val_class, y_pred_class, labels=[1], average="macro", zero_division=0
            ),
            "f1_win": f1_score(
                self._split.y_val_class, y_pred_class, labels=[1], average="macro", zero_division=0
            ),
        }

        # Binary AUC for WIN vs not-WIN
        y_binary = (self._split.y_val_class == 1).astype(int)
        p_win = y_pred_proba[:, 1]
        metrics["auc_win"] = roc_auc_score(y_binary, p_win)

        # Calibration metrics
        metrics["brier_score"] = brier_score_loss(y_binary, p_win)

        # Regression metrics
        metrics["rmse_r"] = np.sqrt(mean_squared_error(self._split.y_val_r, y_pred_r))
        metrics["mae_r"] = mean_absolute_error(self._split.y_val_r, y_pred_r)

        # Trading-specific metrics
        metrics.update(self._calculate_trading_metrics(y_pred_proba, y_pred_r))

        logger.info("Evaluation complete", **{k: f"{v:.4f}" for k, v in metrics.items()})

        return metrics

    def _calculate_trading_metrics(
        self,
        y_pred_proba: np.ndarray,
        y_pred_r: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate trading-specific evaluation metrics.

        This is what actually matters for trading performance.
        """
        p_win = y_pred_proba[:, 1]
        actual_class = self._split.y_val_class
        actual_r = self._split.y_val_r

        metrics = {}

        # Simulate different P(win) thresholds
        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
            mask = p_win >= threshold

            if mask.sum() == 0:
                continue

            # Win rate in bucket
            wins_in_bucket = (actual_class[mask] == 1).sum()
            total_in_bucket = mask.sum()
            win_rate = wins_in_bucket / total_in_bucket

            # Average R in bucket
            avg_r = actual_r[mask].mean()

            # Expected value
            ev = avg_r

            metrics[f"win_rate_p{int(threshold*100)}"] = win_rate
            metrics[f"avg_r_p{int(threshold*100)}"] = avg_r
            metrics[f"ev_p{int(threshold*100)}"] = ev
            metrics[f"n_trades_p{int(threshold*100)}"] = int(total_in_bucket)

        # Overall expected value if we took all trades
        metrics["ev_all"] = actual_r.mean()
        metrics["win_rate_all"] = (actual_class == 1).mean()

        # Profit factor (if we have win/loss trades)
        wins = actual_r[actual_class == 1]
        losses = actual_r[actual_class == 0]

        if len(wins) > 0 and len(losses) > 0:
            gross_profit = wins.sum()
            gross_loss = abs(losses.sum())
            if gross_loss > 0:
                metrics["profit_factor"] = gross_profit / gross_loss
            else:
                metrics["profit_factor"] = float('inf')

        return metrics

    def walk_forward_validate(self) -> List[Dict[str, float]]:
        """Perform walk-forward validation.

        This is the gold standard for trading model validation.

        Returns:
            List of metrics for each walk-forward window
        """
        if self._features is None:
            raise ValueError("No data loaded")

        n = len(self._features)
        window_size = n // self.config.walk_forward_windows
        results = []

        logger.info(
            "Starting walk-forward validation",
            windows=self.config.walk_forward_windows,
            window_size=window_size,
        )

        for i in range(self.config.walk_forward_windows - 1):
            # Train on windows 0..i, validate on window i+1
            train_end = (i + 1) * window_size
            val_start = train_end
            val_end = min(val_start + window_size, n)

            if train_end < self.config.min_train_samples:
                continue

            logger.info(f"Walk-forward window {i+1}: train 0:{train_end}, val {val_start}:{val_end}")

            # Create temporary split
            temp_split = DataSplit(
                X_train=self._features[:train_end],
                y_train_class=self._labels_class[:train_end],
                y_train_r=self._labels_r[:train_end],
                times_train=self._timestamps[:train_end],
                X_val=self._features[val_start:val_end],
                y_val_class=self._labels_class[val_start:val_end],
                y_val_r=self._labels_r[val_start:val_end],
                times_val=self._timestamps[val_start:val_end],
                X_test=np.array([]),
                y_test_class=np.array([]),
                y_test_r=np.array([]),
                times_test=[],
            )

            # Train models
            old_split = self._split
            self._split = temp_split

            self._classifier = self._train_classifier()
            self._regressor = self._train_regressor()

            # Evaluate
            metrics = self._evaluate()
            metrics["window"] = i + 1
            metrics["train_size"] = train_end
            metrics["val_size"] = val_end - val_start
            results.append(metrics)

            self._split = old_split

        # Log summary
        if results:
            avg_metrics = {
                k: np.mean([r[k] for r in results if k in r])
                for k in results[0].keys()
                if isinstance(results[0].get(k), (int, float))
            }
            logger.info("Walk-forward summary", **{k: f"{v:.4f}" for k, v in avg_metrics.items()})

        return results

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from classifier.

        Returns:
            Dict mapping feature name to importance
        """
        if self._classifier is None:
            raise ValueError("No model trained")

        if hasattr(self._classifier, "feature_importances_"):
            importances = self._classifier.feature_importances_
        else:
            return {}

        # Normalize
        total = importances.sum()
        if total > 0:
            importances = importances / total

        return {
            name: float(imp)
            for name, imp in zip(self._feature_names, importances)
        }

    def save(self, output_dir: str, version: str = None) -> Dict[str, str]:
        """Save trained models and metadata.

        Args:
            output_dir: Output directory
            version: Model version string

        Returns:
            Dict with saved file paths
        """
        if self._classifier is None or self._regressor is None:
            raise ValueError("No models trained")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        version = version or f"{self.config.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Save classifier
        classifier_path = output_path / f"classifier_{version}.pkl"
        with open(classifier_path, "wb") as f:
            pickle.dump(self._classifier, f)

        # Save regressor
        regressor_path = output_path / f"regressor_{version}.pkl"
        with open(regressor_path, "wb") as f:
            pickle.dump(self._regressor, f)

        # Save calibrator if exists
        calibrator_path = None
        if self._calibrator is not None:
            calibrator_path = output_path / f"calibrator_{version}.pkl"
            with open(calibrator_path, "wb") as f:
                pickle.dump(self._calibrator, f)

        # Save metadata
        metadata = {
            "version": version,
            "feature_version": self.config.feature_version,
            "feature_names": self._feature_names,
            "feature_count": len(self._feature_names),
            "config": {
                "model_type": self.config.model_type,
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "learning_rate": self.config.learning_rate,
                "calibrate": self.config.calibrate,
                "calibration_method": self.config.calibration_method,
            },
            "metrics": self._metrics,
            "feature_importance": self.get_feature_importance(),
            "training_samples": len(self._split.y_train_class) if self._split else 0,
            "trained_at": datetime.now().isoformat(),
        }

        metadata_path = output_path / f"metadata_{version}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save combined model for production
        combined = {
            "classifier": self._classifier,
            "regressor": self._regressor,
            "calibrator": self._calibrator,
            "feature_names": self._feature_names,
            "version": version,
            "feature_version": self.config.feature_version,
            "metrics": self._metrics,
        }

        combined_path = output_path / f"gate_model_{version}.pkl"
        with open(combined_path, "wb") as f:
            pickle.dump(combined, f)

        logger.info(
            "Models saved",
            output_dir=str(output_path),
            version=version,
            files=[str(classifier_path), str(regressor_path), str(combined_path)],
        )

        return {
            "classifier": str(classifier_path),
            "regressor": str(regressor_path),
            "calibrator": str(calibrator_path) if calibrator_path else None,
            "metadata": str(metadata_path),
            "combined": str(combined_path),
        }

    @staticmethod
    def load(model_path: str) -> "TrainedModel":
        """Load a trained model from file.

        Args:
            model_path: Path to combined model file

        Returns:
            TrainedModel object
        """
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        return TrainedModel(
            classifier=data["classifier"],
            regressor=data["regressor"],
            calibrator=data.get("calibrator"),
            feature_names=data["feature_names"],
            feature_version=data.get("feature_version", "unknown"),
            config=TrainingConfig(),  # Default config
            metrics=data.get("metrics", {}),
            training_samples=0,
            trained_at=datetime.now(),
            version=data.get("version", "unknown"),
        )
