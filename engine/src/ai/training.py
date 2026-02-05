"""Model training utilities for AI decision gate.

Provides tools for:
- Training data preparation
- Model training (XGBoost/LightGBM)
- Cross-validation
- Model export
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import pickle
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: str = "xgboost"  # xgboost, lightgbm, or logistic
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_samples_split: int = 10
    validation_split: float = 0.2
    random_state: int = 42
    class_weight: str = "balanced"


@dataclass
class TrainingResult:
    """Result of model training."""
    model: Any
    metrics: Dict[str, float]
    feature_importances: Dict[str, float]
    training_samples: int
    validation_samples: int
    version: str
    timestamp: datetime


class ModelTrainer:
    """Trains classification models for the AI gate.

    Supports XGBoost, LightGBM, and logistic regression.

    Example:
        trainer = ModelTrainer(config)
        trainer.load_training_data(trades_df)
        result = trainer.train()
        trainer.save_model("models/gate_v1.pkl")
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._feature_names: List[str] = []
        self._model = None
        self._training_result: Optional[TrainingResult] = None

    def load_training_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Load training data.

        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Binary labels (1=profitable, 0=loss)
            feature_names: Names of features
        """
        from sklearn.model_selection import train_test_split

        self._feature_names = feature_names or [
            f"feature_{i}" for i in range(features.shape[1])
        ]

        # Split into train/validation
        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(
            features,
            labels,
            test_size=self.config.validation_split,
            random_state=self.config.random_state,
            stratify=labels,
        )

        logger.info(
            "Training data loaded",
            train_samples=len(self._y_train),
            val_samples=len(self._y_val),
            positive_rate=float(labels.mean()),
        )

    def load_from_trades(
        self,
        trades: List[Dict],
        feature_extractor: Any = None,
    ) -> None:
        """Load training data from historical trades.

        Args:
            trades: List of trade records with features and outcome
            feature_extractor: FeatureExtractor instance
        """
        if not trades:
            raise ValueError("No trades provided")

        features_list = []
        labels = []

        for trade in trades:
            # Extract features if extractor provided
            if feature_extractor and "setup" in trade:
                feat_vec = feature_extractor.extract(
                    trade["setup"],
                    trade.get("market_view"),
                )
                features_list.append(feat_vec.feature_array)
                self._feature_names = feat_vec.feature_names
            elif "features" in trade:
                features_list.append(np.array(trade["features"]))

            # Determine label (1 = profitable, 0 = loss)
            pnl = trade.get("pnl", 0)
            labels.append(1 if pnl > 0 else 0)

        features = np.vstack(features_list)
        labels = np.array(labels)

        self.load_training_data(features, labels, self._feature_names)

    def train(self) -> TrainingResult:
        """Train the model.

        Returns:
            TrainingResult with model and metrics
        """
        if self._X_train is None:
            raise ValueError("No training data loaded")

        logger.info("Starting model training", model_type=self.config.model_type)

        # Build model based on type
        if self.config.model_type == "xgboost":
            self._model = self._build_xgboost()
        elif self.config.model_type == "lightgbm":
            self._model = self._build_lightgbm()
        else:
            self._model = self._build_logistic()

        # Train
        self._model.fit(self._X_train, self._y_train)

        # Evaluate
        metrics = self._evaluate()

        # Get feature importances
        importances = self._get_feature_importances()

        # Create version string
        version = f"{self.config.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        self._training_result = TrainingResult(
            model=self._model,
            metrics=metrics,
            feature_importances=importances,
            training_samples=len(self._y_train),
            validation_samples=len(self._y_val),
            version=version,
            timestamp=datetime.now(),
        )

        logger.info(
            "Model training complete",
            version=version,
            accuracy=metrics.get("accuracy"),
            auc=metrics.get("auc"),
        )

        return self._training_result

    def _build_xgboost(self):
        """Build XGBoost classifier."""
        try:
            from xgboost import XGBClassifier

            # Calculate scale_pos_weight for imbalanced data
            pos_count = self._y_train.sum()
            neg_count = len(self._y_train) - pos_count
            scale_pos = neg_count / pos_count if pos_count > 0 else 1.0

            return XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                scale_pos_weight=scale_pos,
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        except ImportError:
            logger.warning("XGBoost not installed, falling back to logistic regression")
            return self._build_logistic()

    def _build_lightgbm(self):
        """Build LightGBM classifier."""
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                verbose=-1,
            )
        except ImportError:
            logger.warning("LightGBM not installed, falling back to logistic regression")
            return self._build_logistic()

    def _build_logistic(self):
        """Build logistic regression classifier."""
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            max_iter=1000,
        )

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        y_pred = self._model.predict(self._X_val)
        y_proba = self._model.predict_proba(self._X_val)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(self._y_val, y_pred)),
            "precision": float(precision_score(self._y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(self._y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(self._y_val, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(self._y_val, y_proba)),
        }

        # Calculate profit factor (if we have returns data)
        # This would need actual P&L data

        return metrics

    def _get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances."""
        importances = {}

        if hasattr(self._model, "feature_importances_"):
            raw_imp = self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            raw_imp = np.abs(self._model.coef_[0])
        else:
            return importances

        for i, name in enumerate(self._feature_names):
            if i < len(raw_imp):
                importances[name] = float(raw_imp[i])

        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    def cross_validate(self, n_folds: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation.

        Args:
            n_folds: Number of CV folds

        Returns:
            Dict with lists of metrics per fold
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, roc_auc_score

        if self._X_train is None:
            raise ValueError("No training data loaded")

        X_full = np.vstack([self._X_train, self._X_val])
        y_full = np.concatenate([self._y_train, self._y_val])

        kfold = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        cv_results = {"accuracy": [], "auc": []}

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_full, y_full)):
            # Build fresh model
            if self.config.model_type == "xgboost":
                model = self._build_xgboost()
            elif self.config.model_type == "lightgbm":
                model = self._build_lightgbm()
            else:
                model = self._build_logistic()

            # Train
            model.fit(X_full[train_idx], y_full[train_idx])

            # Evaluate
            y_pred = model.predict(X_full[val_idx])
            y_proba = model.predict_proba(X_full[val_idx])[:, 1]

            cv_results["accuracy"].append(
                float(accuracy_score(y_full[val_idx], y_pred))
            )
            cv_results["auc"].append(
                float(roc_auc_score(y_full[val_idx], y_proba))
            )

            logger.info(
                f"CV fold {fold + 1}/{n_folds}",
                accuracy=cv_results["accuracy"][-1],
                auc=cv_results["auc"][-1],
            )

        # Log summary
        logger.info(
            "Cross-validation complete",
            mean_accuracy=np.mean(cv_results["accuracy"]),
            std_accuracy=np.std(cv_results["accuracy"]),
            mean_auc=np.mean(cv_results["auc"]),
            std_auc=np.std(cv_results["auc"]),
        )

        return cv_results

    def save_model(self, path: str) -> None:
        """Save trained model to file.

        Args:
            path: Output path (.pkl)
        """
        if self._model is None:
            raise ValueError("No model trained")

        model_data = {
            "model": self._model,
            "version": self._training_result.version if self._training_result else "unknown",
            "feature_names": self._feature_names,
            "config": {
                "model_type": self.config.model_type,
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
            },
            "metrics": self._training_result.metrics if self._training_result else {},
            "trained_at": datetime.now().isoformat(),
        }

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info("Model saved", path=path)

    def export_for_production(self, output_dir: str) -> Dict[str, str]:
        """Export model and metadata for production use.

        Args:
            output_dir: Output directory

        Returns:
            Dict with paths to exported files
        """
        if self._model is None:
            raise ValueError("No model trained")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        version = self._training_result.version if self._training_result else "v1"

        # Save model
        model_path = output_path / f"model_{version}.pkl"
        self.save_model(str(model_path))

        # Save metadata
        metadata = {
            "version": version,
            "model_type": self.config.model_type,
            "feature_names": self._feature_names,
            "feature_count": len(self._feature_names),
            "metrics": self._training_result.metrics if self._training_result else {},
            "training_samples": self._training_result.training_samples if self._training_result else 0,
            "exported_at": datetime.now().isoformat(),
        }

        metadata_path = output_path / f"metadata_{version}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model exported for production", output_dir=output_dir)

        return {
            "model": str(model_path),
            "metadata": str(metadata_path),
        }


def prepare_training_data_from_db(
    db_session,
    min_trades: int = 100,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare training data from database trade records.

    Args:
        db_session: Database session
        min_trades: Minimum trades required

    Returns:
        Tuple of (features, labels, feature_names)
    """
    from database.models import TradeLog
    from .features import FeatureExtractor

    # Query completed trades
    trades = db_session.query(TradeLog).filter(
        TradeLog.status == "CLOSED"
    ).all()

    if len(trades) < min_trades:
        raise ValueError(
            f"Insufficient trades: {len(trades)} < {min_trades}"
        )

    extractor = FeatureExtractor()
    features_list = []
    labels = []

    for trade in trades:
        # Extract features from stored data
        if trade.features:
            features_list.append(np.array(trade.features))
        else:
            # Would need to reconstruct from trade data
            continue

        # Label based on P&L
        labels.append(1 if trade.pnl and trade.pnl > 0 else 0)

    if not features_list:
        raise ValueError("No valid feature data found in trades")

    return (
        np.vstack(features_list),
        np.array(labels),
        extractor.feature_names,
    )
