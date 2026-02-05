"""AI Decision Gate for trade candidate approval.

Uses machine learning to make the final go/no-go decision
on trade candidates that pass rule-based filters.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import pickle
import numpy as np
import structlog

from .features import FeatureExtractor, FeatureVector
from src.scoring.confluence import TradeSetup
from src.analysis.analyzer import MarketView

logger = structlog.get_logger(__name__)


class GateDecisionType(Enum):
    """AI gate decision types."""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass
class GateConfig:
    """Configuration for AI gate.

    Two-layer decision logic:
    1. Deterministic checks (confluence, risk filters)
    2. AI gate: approve if E[R] > min_ev AND P(win) > min_prob
    """
    # AI thresholds (TWO conditions must be met)
    min_prob_win: float = 0.55  # Minimum P(win) from classifier
    min_expected_r: float = 0.15  # Minimum E[R] from regressor
    max_prob_timeout: float = 0.35  # Maximum P(timeout) acceptable

    # Absolute minimum thresholds (below these = always REJECTED)
    absolute_min_prob_win: float = 0.20  # Floor for NEEDS_REVIEW consideration
    absolute_min_expected_r: float = -0.5  # Floor for E[R] consideration

    # Legacy thresholds (for rule-based fallback)
    approval_threshold: float = 0.65
    review_threshold: float = 0.45

    # Deterministic layer
    min_confluence_score: int = 60  # Minimum confluence before AI check
    max_candidates_per_session: int = 10

    # Model settings
    model_path: Optional[str] = None
    fallback_to_rules: bool = True  # Use rules if model unavailable


@dataclass
class GateDecision:
    """Result of AI gate decision.

    Contains both P(win) from classifier and E[R] from regressor.
    """
    decision: GateDecisionType
    probability: float  # P(win) from classifier
    expected_r: float  # E[R] from regressor
    prob_timeout: float  # P(timeout) from classifier
    confidence: float
    reasons: List[str]
    feature_importances: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    model_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision.value,
            "probability": self.probability,
            "expected_r": self.expected_r,
            "prob_timeout": self.prob_timeout,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "feature_importances": self.feature_importances,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
        }


class AIGate:
    """AI-powered decision gate for trade approval.

    Uses trained classifier and regressor models to make go/no-go decisions
    on trade candidates. Falls back to rule-based logic if models
    are unavailable.

    Two-layer decision logic:
    1. Deterministic checks (confluence score, session limits)
    2. AI gate: approve if E[R] > min_ev AND P(win) > min_prob

    Example:
        gate = AIGate(config)
        decision = gate.evaluate(setup, market_view)
        if decision.decision == GateDecisionType.APPROVED:
            execute_trade(setup)
    """

    def __init__(self, config: Optional[GateConfig] = None):
        """Initialize AI gate.

        Args:
            config: Gate configuration
        """
        self.config = config or GateConfig()
        self.feature_extractor = FeatureExtractor()

        # Two models: classifier for P(win/loss/timeout), regressor for E[R]
        self._classifier = None
        self._regressor = None
        self._calibrator = None  # IsotonicRegression to calibrate P(win)
        self._model_version: Optional[str] = None
        self._model_feature_names: Optional[List[str]] = None
        self._decision_history: List[GateDecision] = []
        self._session_approved_count: int = 0
        self._last_session_reset: datetime = datetime.now()

        # Load model if path specified
        if self.config.model_path:
            self.load_model(self.config.model_path)

    def load_model(self, model_path: str) -> bool:
        """Load trained models (classifier + regressor) from directory or file.

        Args:
            model_path: Path to model directory or single model file

        Returns:
            True if at least one model loaded successfully
        """
        try:
            path = Path(model_path)

            # If directory, look for model files
            if path.is_dir():
                # First, try to find gate_model_*.pkl (combined model)
                gate_models = list(path.glob("gate_model_*.pkl"))
                if gate_models:
                    # Use the most recent one
                    gate_model_path = sorted(gate_models)[-1]
                    with open(gate_model_path, "rb") as f:
                        model_data = pickle.load(f)
                        self._classifier = model_data.get("classifier")
                        self._regressor = model_data.get("regressor")
                        self._calibrator = model_data.get("calibrator")
                        self._model_version = model_data.get("version", "unknown")
                        self._model_feature_names = model_data.get("feature_names")
                        logger.info("Combined model loaded",
                                   path=str(gate_model_path),
                                   has_calibrator=self._calibrator is not None)
                else:
                    # Fall back to individual files
                    classifier_path = path / "classifier.pkl"
                    regressor_path = path / "regressor.pkl"

                    if classifier_path.exists():
                        with open(classifier_path, "rb") as f:
                            clf_data = pickle.load(f)
                            # Handle both dict format and raw model
                            if isinstance(clf_data, dict):
                                self._classifier = clf_data.get("model") or clf_data.get("classifier")
                                self._model_version = clf_data.get("version", "unknown")
                                self._model_feature_names = clf_data.get("feature_names")
                            else:
                                self._classifier = clf_data
                            logger.info("Classifier loaded")

                    if regressor_path.exists():
                        with open(regressor_path, "rb") as f:
                            reg_data = pickle.load(f)
                            if isinstance(reg_data, dict):
                                self._regressor = reg_data.get("model") or reg_data.get("regressor")
                            else:
                                self._regressor = reg_data
                            logger.info("Regressor loaded")

            elif path.exists() and path.suffix == ".pkl":
                # Single file
                with open(path, "rb") as f:
                    model_data = pickle.load(f)
                    # Check if it contains both models (combined format)
                    if isinstance(model_data, dict) and "classifier" in model_data:
                        self._classifier = model_data.get("classifier")
                        self._regressor = model_data.get("regressor")
                        self._model_version = model_data.get("version", "unknown")
                        self._model_feature_names = model_data.get("feature_names")
                    elif isinstance(model_data, dict):
                        # Legacy single model dict
                        self._classifier = model_data.get("model")
                        self._model_version = model_data.get("version", "unknown")
                        self._model_feature_names = model_data.get("feature_names")
                    else:
                        # Raw model object
                        self._classifier = model_data
            else:
                logger.warning("Model path not found", path=model_path)
                return False

            logger.info(
                "AI models loaded",
                version=self._model_version,
                classifier=self._classifier is not None,
                regressor=self._regressor is not None,
            )
            return self._classifier is not None or self._regressor is not None

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            return False

    def evaluate(
        self,
        setup: TradeSetup,
        market_view: Optional[MarketView] = None,
        trade_history: Optional[List[Dict]] = None,
    ) -> GateDecision:
        """Evaluate a trade setup through the AI gate.

        Args:
            setup: Trade setup to evaluate
            market_view: Full market analysis
            trade_history: Recent trade history

        Returns:
            GateDecision with approval status
        """
        self._check_session_reset()

        # Pre-check: minimum confluence score
        if setup.confluence_score < self.config.min_confluence_score:
            return GateDecision(
                decision=GateDecisionType.REJECTED,
                probability=0.0,
                expected_r=0.0,
                prob_timeout=0.0,
                confidence=1.0,
                reasons=[
                    f"Confluence score {setup.confluence_score} below minimum {self.config.min_confluence_score}"
                ],
                feature_importances={},
            )

        # Check session limit
        if self._session_approved_count >= self.config.max_candidates_per_session:
            return GateDecision(
                decision=GateDecisionType.REJECTED,
                probability=0.0,
                expected_r=0.0,
                prob_timeout=0.0,
                confidence=1.0,
                reasons=["Session trade limit reached"],
                feature_importances={},
            )

        # Extract features
        features = self.feature_extractor.extract(
            setup, market_view, trade_history
        )

        # Evaluate with model or fallback
        if self._classifier is not None:
            decision = self._evaluate_with_model(setup, features)
        elif self.config.fallback_to_rules:
            decision = self._evaluate_with_rules(setup, features)
        else:
            decision = GateDecision(
                decision=GateDecisionType.NEEDS_REVIEW,
                probability=0.5,
                expected_r=0.0,
                prob_timeout=0.0,
                confidence=0.0,
                reasons=["No model loaded and fallback disabled"],
                feature_importances={},
            )

        # Track approved candidates
        if decision.decision == GateDecisionType.APPROVED:
            self._session_approved_count += 1

        # Record decision
        self._decision_history.append(decision)
        if len(self._decision_history) > 100:
            self._decision_history = self._decision_history[-100:]

        logger.info(
            "AI gate decision",
            symbol=setup.symbol,
            direction=setup.direction,
            decision=decision.decision.value,
            probability=decision.probability,
        )

        return decision

    def _evaluate_with_model(
        self,
        setup: TradeSetup,
        features: FeatureVector,
    ) -> GateDecision:
        """Evaluate using trained classifier and regressor.

        Two-layer decision logic:
        - Approve ONLY if E[R] > min_expected_r AND P(win) > min_prob_win
        - Also checks P(timeout) < max_prob_timeout
        """
        try:
            feature_input = features.feature_array.reshape(1, -1)
            reasons = []

            # --- Get P(win), P(loss), P(timeout) from classifier ---
            prob_win = 0.0
            prob_loss = 0.0
            prob_timeout = 0.0

            if hasattr(self._classifier, "predict_proba"):
                proba = self._classifier.predict_proba(feature_input)[0]
                # 3-class: [P(LOSS), P(WIN), P(TIMEOUT)] - classes 0, 1, 2
                if len(proba) >= 3:
                    prob_loss = float(proba[0])
                    prob_win = float(proba[1])
                    prob_timeout = float(proba[2])
                elif len(proba) == 2:
                    # Binary: [P(LOSS), P(WIN)]
                    prob_loss = float(proba[0])
                    prob_win = float(proba[1])

                # Apply calibrator to get realistic P(win) from raw XGBoost proba
                if self._calibrator is not None:
                    raw_prob_win = prob_win
                    try:
                        prob_win = float(self._calibrator.transform([raw_prob_win])[0])
                    except Exception:
                        pass  # Fall back to uncalibrated
            else:
                # No probability - use prediction
                pred = self._classifier.predict(feature_input)[0]
                prob_win = 1.0 if pred == 1 else 0.0

            # --- Get E[R] from regressor ---
            expected_r = 0.0
            if self._regressor is not None:
                expected_r = float(self._regressor.predict(feature_input)[0])
            else:
                # Estimate E[R] from probabilities if no regressor
                # E[R] ≈ P(win) * avg_win_r - P(loss) * 1.0
                avg_win_r = 2.0  # Assume 2R wins
                expected_r = prob_win * avg_win_r - prob_loss * 1.0

            # --- Two-layer decision logic ---
            passes_prob_win = prob_win >= self.config.min_prob_win
            passes_expected_r = expected_r >= self.config.min_expected_r
            passes_timeout = prob_timeout <= self.config.max_prob_timeout

            # Build reasons for decision
            if passes_prob_win:
                reasons.append(f"P(win) {prob_win:.1%} >= {self.config.min_prob_win:.1%}")
            else:
                reasons.append(f"P(win) {prob_win:.1%} < {self.config.min_prob_win:.1%}")

            if passes_expected_r:
                reasons.append(f"E[R] {expected_r:.2f} >= {self.config.min_expected_r:.2f}")
            else:
                reasons.append(f"E[R] {expected_r:.2f} < {self.config.min_expected_r:.2f}")

            if not passes_timeout:
                reasons.append(f"P(timeout) {prob_timeout:.1%} > {self.config.max_prob_timeout:.1%}")

            # Check absolute minimum thresholds (below these = always REJECTED)
            above_absolute_min_prob = prob_win >= self.config.absolute_min_prob_win
            above_absolute_min_r = expected_r >= self.config.absolute_min_expected_r

            # Determine decision: BOTH conditions must be met for APPROVAL
            if passes_prob_win and passes_expected_r and passes_timeout:
                decision_type = GateDecisionType.APPROVED
                reasons.insert(0, "AI gate APPROVED: all conditions met")
            elif not above_absolute_min_prob:
                # P(win) below absolute floor - always reject
                decision_type = GateDecisionType.REJECTED
                reasons.insert(0, f"AI gate REJECTED: P(win) {prob_win:.1%} below floor {self.config.absolute_min_prob_win:.0%}")
            elif not above_absolute_min_r:
                # E[R] below absolute floor - always reject
                decision_type = GateDecisionType.REJECTED
                reasons.insert(0, f"AI gate REJECTED: E[R] {expected_r:.2f} below floor {self.config.absolute_min_expected_r:.2f}")
            elif passes_prob_win or passes_expected_r:
                # Above absolute floors but not meeting full criteria
                decision_type = GateDecisionType.NEEDS_REVIEW
                reasons.insert(0, "AI gate REVIEW: partial conditions met")
            else:
                decision_type = GateDecisionType.REJECTED
                reasons.insert(0, "AI gate REJECTED: conditions not met")

            # Get feature importances
            importances = self._get_feature_importances(features)

            # Add top contributing features
            top_features = sorted(
                importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            for name, imp in top_features:
                direction = "+" if imp > 0 else ""
                reasons.append(f"Feature: {name} ({direction}{imp:.3f})")

            # Calculate confidence based on margin above thresholds
            prob_margin = (prob_win - self.config.min_prob_win) / (1.0 - self.config.min_prob_win) if prob_win > self.config.min_prob_win else 0
            r_margin = (expected_r - self.config.min_expected_r) / (1.0 - self.config.min_expected_r) if expected_r > self.config.min_expected_r else 0
            confidence = min(prob_margin, r_margin) if decision_type == GateDecisionType.APPROVED else max(prob_margin, r_margin)

            return GateDecision(
                decision=decision_type,
                probability=prob_win,
                expected_r=expected_r,
                prob_timeout=prob_timeout,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                reasons=reasons,
                feature_importances=importances,
                model_version=self._model_version,
            )

        except Exception as e:
            logger.error("Model evaluation failed", error=str(e))
            if self.config.fallback_to_rules:
                return self._evaluate_with_rules(setup, features)
            return GateDecision(
                decision=GateDecisionType.NEEDS_REVIEW,
                probability=0.5,
                expected_r=0.0,
                prob_timeout=0.0,
                confidence=0.0,
                reasons=[f"Model error: {str(e)}"],
                feature_importances={},
            )

    def _evaluate_with_rules(
        self,
        setup: TradeSetup,
        features: FeatureVector,
    ) -> GateDecision:
        """Fallback rule-based evaluation when no model is available."""
        score = 0.0
        max_score = 100.0
        reasons = []
        importances = {}

        # Confluence score contribution (30%)
        conf_contrib = (setup.confluence_score / 100.0) * 30
        score += conf_contrib
        importances["confluence_score"] = conf_contrib / max_score
        if setup.confluence_score >= 75:
            reasons.append("Strong confluence score")

        # Risk/reward contribution (25%)
        rr = setup.risk_reward or 0
        if rr >= 2.0:
            rr_contrib = 25
            reasons.append(f"Good R:R ratio ({rr:.1f})")
        elif rr >= 1.5:
            rr_contrib = 15
        else:
            rr_contrib = 5
        score += rr_contrib
        importances["risk_reward_ratio"] = rr_contrib / max_score

        # Momentum alignment (15%)
        mom_aligned = features.features.get("momentum_aligned", 0.5)
        mom_contrib = mom_aligned * 15
        score += mom_contrib
        importances["momentum_aligned"] = mom_contrib / max_score
        if mom_aligned > 0.5:
            reasons.append("Momentum confirms direction")

        # Strength alignment (15%)
        str_aligned = features.features.get("strength_trend_aligned", 0.5)
        str_contrib = str_aligned * 15
        score += str_contrib
        importances["strength_aligned"] = str_contrib / max_score
        if str_aligned > 0.5:
            reasons.append("Currency strength aligned")

        # Session timing (10%)
        session_overlap = features.features.get("session_overlap", 0)
        session_contrib = 10 if session_overlap > 0.5 else 5
        score += session_contrib
        importances["session_timing"] = session_contrib / max_score
        if session_overlap > 0.5:
            reasons.append("Active session overlap")

        # Zone quality (5%)
        zone_qual = features.features.get("zone_quality", 0.5)
        zone_contrib = zone_qual * 5
        score += zone_contrib
        importances["zone_quality"] = zone_contrib / max_score

        # Normalize to probability (treat as P(win))
        probability = score / max_score

        # Estimate E[R] based on rule score and R:R
        expected_r = (probability - 0.5) * rr if rr > 0 else (probability - 0.5) * 2.0

        # Determine decision using two-layer logic
        passes_prob = probability >= self.config.min_prob_win
        passes_er = expected_r >= self.config.min_expected_r

        if passes_prob and passes_er:
            decision_type = GateDecisionType.APPROVED
            if not reasons:
                reasons.append("Rule-based checks passed")
        elif passes_prob or passes_er:
            decision_type = GateDecisionType.NEEDS_REVIEW
            reasons.append("Borderline - manual review suggested")
        else:
            decision_type = GateDecisionType.REJECTED
            reasons.append("Insufficient rule-based score")

        return GateDecision(
            decision=decision_type,
            probability=probability,
            expected_r=expected_r,
            prob_timeout=0.1,  # Default estimate for rules
            confidence=abs(probability - 0.5) * 2,
            reasons=reasons,
            feature_importances=importances,
            model_version="rules-v1",
        )

    def _get_feature_importances(
        self,
        features: FeatureVector,
    ) -> Dict[str, float]:
        """Get feature importances from classifier model."""
        importances = {}

        if self._classifier is None:
            return importances

        try:
            # Try to get feature importances from classifier
            if hasattr(self._classifier, "feature_importances_"):
                raw_imp = self._classifier.feature_importances_
                for i, name in enumerate(features.feature_names):
                    if i < len(raw_imp):
                        importances[name] = float(raw_imp[i])
            elif hasattr(self._classifier, "coef_"):
                coef = self._classifier.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]
                for i, name in enumerate(features.feature_names):
                    if i < len(coef):
                        importances[name] = float(coef[i])
        except Exception:
            pass

        return importances

    def _check_session_reset(self) -> None:
        """Reset session counters if new session."""
        now = datetime.now()
        # Reset at start of each London session (7:00 UTC)
        if now.date() != self._last_session_reset.date():
            if now.hour >= 7:
                self._session_approved_count = 0
                self._last_session_reset = now
                logger.info("AI gate session reset")

    def batch_evaluate(
        self,
        setups: List[TradeSetup],
        market_views: Optional[Dict[str, MarketView]] = None,
    ) -> List[Tuple[TradeSetup, GateDecision]]:
        """Evaluate multiple setups and return sorted by probability.

        Args:
            setups: List of trade setups
            market_views: Dict of market views keyed by symbol

        Returns:
            List of (setup, decision) tuples sorted by approval probability
        """
        results = []
        for setup in setups:
            view = market_views.get(setup.symbol) if market_views else None
            decision = self.evaluate(setup, view)
            results.append((setup, decision))

        # Sort by probability (highest first)
        results.sort(key=lambda x: x[1].probability, reverse=True)
        return results

    def get_approved_setups(
        self,
        setups: List[TradeSetup],
        market_views: Optional[Dict[str, MarketView]] = None,
        max_setups: int = 3,
    ) -> List[TradeSetup]:
        """Get approved setups up to a maximum count.

        Args:
            setups: List of trade setups to evaluate
            market_views: Dict of market views
            max_setups: Maximum setups to return

        Returns:
            List of approved setups
        """
        evaluated = self.batch_evaluate(setups, market_views)
        approved = [
            setup for setup, decision in evaluated
            if decision.decision == GateDecisionType.APPROVED
        ]
        return approved[:max_setups]

    @property
    def model_loaded(self) -> bool:
        """Check if classifier model is loaded."""
        return self._classifier is not None

    @property
    def decision_history(self) -> List[GateDecision]:
        """Get recent decision history."""
        return self._decision_history[-20:]

    @property
    def session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        return {
            "approved_count": self._session_approved_count,
            "max_per_session": self.config.max_candidates_per_session,
            "remaining": max(
                0,
                self.config.max_candidates_per_session - self._session_approved_count
            ),
            "session_start": self._last_session_reset.isoformat(),
        }
