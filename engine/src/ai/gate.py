"""AI Gate -- probability-calibrated trade filter.

Bridges the legacy interface (``GateConfig``, ``GateDecisionType``,
``evaluate(setup, market_view, ...)``) with the v3 model format produced
by ``training_pipeline.py`` (XGBoost classifier + Platt calibrator +
versioned metadata).

Model format detection
~~~~~~~~~~~~~~~~~~~~~~
* **v3 format** -- top-level keys ``model``, ``calibrator``, ``metadata``
  with ``metadata["feature_version"]`` starting with ``"v3"``.
* **Legacy format** -- top-level keys ``classifier`` (required) and
  optional ``regressor``.

When a v3 model is loaded *and* the caller supplies raw candle arrays
(``m15_candles``, ``h1_candles``, ``h4_candles``), the gate runs the
v3 :class:`FeatureExtractor` internally.  Otherwise it falls back to a
rule-based heuristic.
"""
from __future__ import annotations

import enum
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:                      # graceful fallback
    import logging
    logger = logging.getLogger(__name__)

from .features import (
    FeatureExtractor,
    MarketView,
    N_TOTAL_FEATURES,
    FEATURE_VERSION,
)


# =====================================================================
# Public data structures
# =====================================================================

class GateDecisionType(str, enum.Enum):
    """Possible outcomes of a gate evaluation."""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    NO_MODEL = "NO_MODEL"


@dataclass
class GateConfig:
    """Configuration consumed by :class:`AIGate`.

    Parameters match the legacy call-site in ``main.py``::

        gate_config = GateConfig(
            model_path="./models",
            min_prob_win=0.55,
            ...
        )
    """
    model_path: str = "./models"
    min_prob_win: float = 0.55
    min_expected_r: float = 0.02
    min_confluence_score: float = 60.0
    max_candidates_per_session: int = 3
    fallback_to_rules: bool = True


@dataclass
class GateDecision:
    """Result of a single gate evaluation.

    Attributes expected by the legacy caller (``decision.decision``,
    ``decision.probability``, ``decision.expected_r``, etc.) and
    extended fields for the v3 pipeline.
    """
    decision: GateDecisionType
    probability: float = 0.0        # calibrated P(win)
    raw_probability: float = 0.0    # uncalibrated XGBoost P(win)
    expected_r: float = 0.0         # estimated E[R]
    prob_timeout: float = 0.0       # estimated P(timeout) -- heuristic
    confluence_score: float = 0.0
    reason: str = ""
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0
    features: Optional[np.ndarray] = field(default=None, repr=False)
    feature_importances: Dict[str, float] = field(default_factory=dict)
    model_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience aliases so both old and new callers work.
    # ------------------------------------------------------------------
    @property
    def p_win(self) -> float:
        """Alias used by the v3 ``LiveEngine``."""
        return self.probability

    @property
    def raw_p_win(self) -> float:
        return self.raw_probability

    @property
    def approved(self) -> bool:
        return self.decision == GateDecisionType.APPROVED

    @property
    def threshold(self) -> float:
        """Return the P(win) threshold stored in metadata (if any)."""
        return float(self.metadata.get("threshold", 0.0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "probability": self.probability,
            "expected_r": self.expected_r,
            "prob_timeout": self.prob_timeout,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "model_version": self.model_version,
        }


# =====================================================================
# AIGate
# =====================================================================

class AIGate:
    """Loads a trained gate model and evaluates trade setups.

    Supports both the legacy call convention and the v3 pipeline.
    """

    def __init__(self, config: GateConfig) -> None:
        self._config = config

        # Model artefacts
        self._classifier: Any = None
        self._regressor: Any = None      # legacy only
        self._calibrator: Any = None      # v3 Platt / isotonic
        self._metadata: Dict[str, Any] = {}

        # v3 feature extractor (lazy -- created when a v3 model is loaded)
        self._feature_extractor: Optional[FeatureExtractor] = None

        self._is_v3: bool = False
        self._model_loaded: bool = False

        self._load_model(config.model_path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def is_v3(self) -> bool:
        return self._is_v3

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    @property
    def feature_version(self) -> str:
        return self._metadata.get("feature_version", "unknown")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self, model_path: str) -> None:
        """Load a pickled gate model.

        *model_path* may be either a directory (search for the newest
        ``.pkl`` file) or a direct path to a pickle file.
        """
        path = Path(model_path)

        # If path is a directory, find the newest .pkl file inside it.
        if path.is_dir():
            pkl_files = sorted(path.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
            if not pkl_files:
                logger.warning("gate.load_model.no_pkl_files", directory=str(path))
                return
            path = pkl_files[-1]
            logger.info("gate.load_model.auto_selected", file=path.name)

        if not path.exists():
            logger.warning("gate.load_model.file_not_found", path=str(path))
            return

        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:
            logger.error("gate.load_model.pickle_error", path=str(path), error=str(exc))
            return

        # ----- Detect format -----
        if self._try_load_v3(payload, path):
            return
        if self._try_load_legacy(payload, path):
            return

        logger.error(
            "gate.load_model.unknown_format",
            keys=list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
        )

    # ---- v3 loader ----
    def _try_load_v3(self, payload: Any, path: Path) -> bool:
        """Return ``True`` if *payload* matches the v3 format and was loaded."""
        if not isinstance(payload, dict):
            return False
        meta = payload.get("metadata")
        if not isinstance(meta, dict):
            return False
        fv = meta.get("feature_version", "")
        if not isinstance(fv, str) or not fv.startswith("v3"):
            return False

        model = payload.get("model")
        if model is None:
            logger.error("gate.load_model.v3_missing_model", path=str(path))
            return False

        self._classifier = model
        self._calibrator = payload.get("calibrator")
        self._metadata = meta
        self._is_v3 = True
        self._model_loaded = True
        self._feature_extractor = FeatureExtractor()

        # Validate feature count
        n_model = meta.get("n_features")
        n_extractor = self._feature_extractor.n_features
        if isinstance(n_model, int) and n_model != n_extractor:
            logger.error(
                "gate.load_model.feature_count_mismatch",
                model_expects=n_model,
                extractor_produces=n_extractor,
                model_version=fv,
                extractor_version=FEATURE_VERSION,
            )
            self._model_loaded = False
            return True  # format was v3, but unusable

        logger.info(
            "gate.load_model.v3_loaded",
            path=path.name,
            feature_version=fv,
            calibration=meta.get("calibration_method", "none"),
            n_features=n_model,
            has_calibrator=self._calibrator is not None,
            val_auc=meta.get("val_auc"),
        )
        return True

    # ---- legacy loader ----
    def _try_load_legacy(self, payload: Any, path: Path) -> bool:
        """Return ``True`` if *payload* matches the legacy format."""
        if not isinstance(payload, dict):
            return False
        classifier = payload.get("classifier")
        if classifier is None:
            return False

        self._classifier = classifier
        self._regressor = payload.get("regressor")
        self._calibrator = payload.get("calibrator")
        self._metadata = payload.get("metadata", {})
        self._is_v3 = False
        self._model_loaded = True

        logger.info(
            "gate.load_model.legacy_loaded",
            path=path.name,
            has_regressor=self._regressor is not None,
            has_calibrator=self._calibrator is not None,
        )
        return True

    # ==================================================================
    # Evaluation -- public entry-point
    # ==================================================================

    def evaluate(
        self,
        setup: Any,
        *,
        market_view: Any = None,
        trade_history: Any = None,
        # v3 raw-candle kwargs
        m15_candles: Optional[Dict[str, Any]] = None,
        h1_candles: Optional[Dict[str, Any]] = None,
        h4_candles: Optional[Dict[str, Any]] = None,
    ) -> GateDecision:
        """Evaluate a trade *setup* and return a gate decision.

        Parameters
        ----------
        setup
            A setup object expected by the legacy engine.  Must expose
            at least ``.direction`` and should expose ``.symbol``.  Any
            additional attributes are accessed defensively.
        market_view
            Legacy market-view object (varies by caller).
        trade_history
            Optional recent trade history (used by rule heuristics).
        m15_candles, h1_candles, h4_candles
            Optional dicts with keys ``opens``, ``highs``, ``lows``,
            ``closes``, ``volumes`` (np.ndarrays), plus optional
            ``rsi_14``, ``atr_14``, ``ema_50``, ``trend_state``,
            ``timestamp_utc``.  When provided together with a v3 model,
            the v3 :class:`FeatureExtractor` is used directly.
        """
        if not self._model_loaded:
            if self._config.fallback_to_rules:
                return self._evaluate_with_rules(setup, market_view, trade_history)
            return self._no_model_decision("model not loaded")

        # ---- v3 path ----
        if self._is_v3:
            if m15_candles is not None:
                return self._evaluate_v3(
                    setup, m15_candles, h1_candles, h4_candles,
                )
            # v3 model present but no candle data -- fall back to rules
            if self._config.fallback_to_rules:
                logger.debug(
                    "gate.evaluate.v3_no_candles_fallback",
                    symbol=getattr(setup, "symbol", "?"),
                )
                return self._evaluate_with_rules(setup, market_view, trade_history)
            return self._no_model_decision("v3 model loaded but no candle data supplied")

        # ---- legacy path ----
        return self._evaluate_with_model(setup, market_view, trade_history)

    # ==================================================================
    # v3 evaluation
    # ==================================================================

    def _evaluate_v3(
        self,
        setup: Any,
        m15_candles: Dict[str, Any],
        h1_candles: Optional[Dict[str, Any]],
        h4_candles: Optional[Dict[str, Any]],
    ) -> GateDecision:
        """Run the full v3 pipeline: candles -> features -> model -> calibrate."""
        assert self._feature_extractor is not None

        try:
            m15_view = self._candles_to_market_view(m15_candles)
            h1_view = self._candles_to_market_view(h1_candles) if h1_candles else None
            h4_view = self._candles_to_market_view(h4_candles) if h4_candles else None

            features = self._feature_extractor.extract(
                m15_view, h1_view=h1_view, h4_view=h4_view,
            )

            # Raw P(win) from XGBoost
            raw_p_win = float(
                self._classifier.predict_proba(features.reshape(1, -1))[:, 1][0]
            )

            # Calibrated P(win)
            p_win = self._calibrate(raw_p_win)

            # Estimate E[R] from P(win) since v3 has no regressor.
            # Simple edge formula: E[R] = P(win) * RR - (1 - P(win))
            # Assume a 2:1 reward-risk ratio as the default.
            rr = 2.0
            expected_r = p_win * rr - (1.0 - p_win)

            # Heuristic P(timeout) -- driven by distance from 0.5
            prob_timeout = max(0.0, 1.0 - abs(p_win - 0.5) * 4.0)
            prob_timeout = round(min(prob_timeout, 0.40), 4)

            approved = (
                p_win >= self._config.min_prob_win
                and expected_r >= self._config.min_expected_r
            )

            decision_type = GateDecisionType.APPROVED if approved else GateDecisionType.REJECTED
            reason = (
                f"v3 model: P(win)={p_win:.3f} E[R]={expected_r:.3f}"
                if approved
                else self._rejection_reason_v3(p_win, expected_r)
            )

            logger.info(
                "gate.evaluate.v3",
                symbol=getattr(setup, "symbol", "?"),
                decision=decision_type.value,
                p_win=round(p_win, 4),
                raw_p_win=round(raw_p_win, 4),
                expected_r=round(expected_r, 4),
            )

            reasons = [reason]
            if p_win >= self._config.min_prob_win:
                reasons.append(f"P(win) {p_win:.1%} >= {self._config.min_prob_win:.1%}")
            else:
                reasons.append(f"P(win) {p_win:.1%} < {self._config.min_prob_win:.1%}")
            if expected_r >= self._config.min_expected_r:
                reasons.append(f"E[R] {expected_r:.2f} >= {self._config.min_expected_r:.2f}")
            else:
                reasons.append(f"E[R] {expected_r:.2f} < {self._config.min_expected_r:.2f}")

            confidence = float(np.clip(
                (p_win - self._config.min_prob_win) / max(1.0 - self._config.min_prob_win, 0.01),
                0.0, 1.0,
            )) if approved else 0.0

            return GateDecision(
                decision=decision_type,
                probability=p_win,
                raw_probability=raw_p_win,
                expected_r=expected_r,
                prob_timeout=prob_timeout,
                reason=reason,
                reasons=reasons,
                confidence=confidence,
                features=features,
                model_version=self._metadata.get("feature_version", "?"),
                metadata={
                    "threshold": self._config.min_prob_win,
                    "model_version": self._metadata.get("feature_version", "?"),
                    "pipeline": "v3",
                },
            )

        except Exception as exc:
            logger.error("gate.evaluate.v3_error", error=str(exc), exc_info=True)
            if self._config.fallback_to_rules:
                return self._evaluate_with_rules(setup, None, None)
            return self._no_model_decision(f"v3 evaluation failed: {exc}")

    # ------------------------------------------------------------------
    # Candle dict -> MarketView conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _candles_to_market_view(candles: Dict[str, Any]) -> MarketView:
        """Build a :class:`MarketView` from a raw candle dict.

        Expected keys: ``closes``, ``highs``, ``lows``, ``opens``,
        ``volumes``.  Optional: ``rsi_14``, ``atr_14``, ``ema_50``,
        ``trend_state``, ``timestamp_utc``.
        """
        closes = np.asarray(candles["closes"], dtype=np.float64)
        highs = np.asarray(candles["highs"], dtype=np.float64)
        lows = np.asarray(candles["lows"], dtype=np.float64)
        opens = np.asarray(candles["opens"], dtype=np.float64)
        volumes = np.asarray(candles["volumes"], dtype=np.float64)

        # Timestamps: accept either int-ms array or fall back to zeros
        if "timestamp_utc" in candles and candles["timestamp_utc"] is not None:
            timestamps = np.asarray(candles["timestamp_utc"], dtype=np.int64)
        else:
            timestamps = np.zeros(len(closes), dtype=np.int64)

        # Pre-computed scalars (optional)
        rsi = float(candles.get("rsi_14", 50.0))
        atr = float(candles.get("atr_14", 0.0))
        ema_50 = float(candles.get("ema_50", 0.0))
        trend_state = candles.get("trend_state", "unclear")

        # Derive the remaining MarketView scalars from what we have.
        n = len(closes)

        # EMA-12 (fast): compute on the fly if not provided
        def _ema_val(data: np.ndarray, span: int) -> float:
            if len(data) < 2:
                return float(data[-1]) if len(data) > 0 else 0.0
            alpha = 2.0 / (span + 1.0)
            ema = float(data[0])
            for i in range(1, len(data)):
                ema = alpha * float(data[i]) + (1.0 - alpha) * ema
            return ema

        ema_fast = _ema_val(closes, 12)
        ema_slow = ema_50 if ema_50 != 0.0 else _ema_val(closes, 50)

        trend_bullish = (ema_fast > ema_slow and closes[-1] > ema_slow) if n > 0 else False
        trend_bearish = (ema_fast < ema_slow and closes[-1] < ema_slow) if n > 0 else False

        momentum_aligned = False
        if n > 1:
            momentum_aligned = (
                (trend_bullish and closes[-1] > closes[-2])
                or (trend_bearish and closes[-1] < closes[-2])
            )

        # Structure state
        if isinstance(trend_state, str) and trend_state in ("trending", "ranging"):
            structure_state = trend_state
        elif trend_bullish or trend_bearish:
            structure_state = "trending"
        else:
            structure_state = "unclear"

        # ATR percentile: rough estimate if not provided
        atr_percentile = 0.5
        if atr > 0 and n > 30:
            recent_ranges = highs[-30:] - lows[-30:]
            mean_range = float(np.mean(recent_ranges))
            if mean_range > 0:
                atr_percentile = float(np.clip(atr / (mean_range * 2.0), 0.0, 1.0))

        return MarketView(
            closes=closes,
            highs=highs,
            lows=lows,
            opens=opens,
            volumes=volumes,
            timestamps=timestamps,
            rsi=rsi,
            atr=atr,
            atr_percentile=atr_percentile,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            trend_bullish=trend_bullish,
            trend_bearish=trend_bearish,
            momentum_aligned=momentum_aligned,
            structure_state=structure_state,
        )

    # ==================================================================
    # Legacy evaluation (old model format)
    # ==================================================================

    def _evaluate_with_model(
        self,
        setup: Any,
        market_view: Any,
        trade_history: Any,
    ) -> GateDecision:
        """Evaluate using the legacy ``classifier`` / ``regressor`` model."""
        try:
            feature_vec = self._extract_legacy_features(setup, market_view)

            # P(win) from the classifier
            proba = self._classifier.predict_proba(feature_vec.reshape(1, -1))
            raw_p_win = float(proba[:, 1][0]) if proba.shape[1] > 1 else float(proba[0][0])
            p_win = self._calibrate(raw_p_win)

            # E[R] from the regressor (if available)
            if self._regressor is not None:
                expected_r = float(
                    self._regressor.predict(feature_vec.reshape(1, -1))[0]
                )
            else:
                rr = 2.0
                expected_r = p_win * rr - (1.0 - p_win)

            prob_timeout = max(0.0, 1.0 - abs(p_win - 0.5) * 4.0)
            prob_timeout = round(min(prob_timeout, 0.40), 4)

            # Confluence score from market_view (if available)
            confluence = self._compute_confluence(setup, market_view)

            approved = (
                p_win >= self._config.min_prob_win
                and expected_r >= self._config.min_expected_r
                and confluence >= self._config.min_confluence_score
            )

            decision_type = GateDecisionType.APPROVED if approved else GateDecisionType.REJECTED
            reason = (
                f"legacy model: P(win)={p_win:.3f} E[R]={expected_r:.3f} conf={confluence:.0f}"
                if approved
                else self._rejection_reason_legacy(p_win, expected_r, confluence)
            )

            logger.info(
                "gate.evaluate.legacy",
                symbol=getattr(setup, "symbol", "?"),
                decision=decision_type.value,
                p_win=round(p_win, 4),
                expected_r=round(expected_r, 4),
                confluence=round(confluence, 1),
            )

            return GateDecision(
                decision=decision_type,
                probability=p_win,
                raw_probability=raw_p_win,
                expected_r=expected_r,
                prob_timeout=prob_timeout,
                confluence_score=confluence,
                reason=reason,
                reasons=[reason],
                confidence=float(np.clip(p_win - self._config.min_prob_win, 0, 1)),
                features=feature_vec,
                model_version=self._metadata.get("version", "legacy"),
                metadata={
                    "threshold": self._config.min_prob_win,
                    "pipeline": "legacy",
                },
            )

        except Exception as exc:
            logger.error("gate.evaluate.legacy_error", error=str(exc), exc_info=True)
            if self._config.fallback_to_rules:
                return self._evaluate_with_rules(setup, market_view, trade_history)
            return self._no_model_decision(f"legacy evaluation failed: {exc}")

    # ==================================================================
    # Rules-based fallback
    # ==================================================================

    def _evaluate_with_rules(
        self,
        setup: Any,
        market_view: Any,
        trade_history: Any,
    ) -> GateDecision:
        """Simple rule-based filter when no usable model is available."""
        confluence = self._compute_confluence(setup, market_view)
        approved = confluence >= self._config.min_confluence_score

        decision_type = GateDecisionType.APPROVED if approved else GateDecisionType.REJECTED
        reason = (
            f"rules: confluence={confluence:.0f}"
            if approved
            else f"rules rejected: confluence {confluence:.0f} < {self._config.min_confluence_score:.0f}"
        )

        logger.info(
            "gate.evaluate.rules",
            symbol=getattr(setup, "symbol", "?"),
            decision=decision_type.value,
            confluence=round(confluence, 1),
        )

        return GateDecision(
            decision=decision_type,
            probability=0.0,
            raw_probability=0.0,
            expected_r=0.0,
            prob_timeout=0.0,
            confluence_score=confluence,
            reason=reason,
            reasons=[reason],
            model_version="rules-v1",
            metadata={"pipeline": "rules"},
        )

    # ==================================================================
    # Calibration
    # ==================================================================

    def _calibrate(self, raw_p_win: float) -> float:
        """Apply the fitted calibrator to a single raw P(win).

        Dispatch:
        1. metadata says ``"platt"`` -> logit -> LogisticRegression
        2. metadata says ``"isotonic"`` -> IsotonicRegression.transform
        3. No metadata -> duck-type detection
        4. No calibrator -> return raw
        """
        if self._calibrator is None:
            return raw_p_win

        cal_method = self._metadata.get("calibration_method")

        if cal_method == "platt":
            return self._apply_platt(raw_p_win)
        elif cal_method == "isotonic":
            return self._apply_isotonic(raw_p_win)
        else:
            if hasattr(self._calibrator, "predict_proba"):
                return self._apply_platt(raw_p_win)
            elif hasattr(self._calibrator, "transform"):
                return self._apply_isotonic(raw_p_win)
            else:
                logger.warning("gate.calibrate.unknown_type")
                return raw_p_win

    def _apply_platt(self, raw_p_win: float) -> float:
        """Platt scaling: clip -> logit -> LogisticRegression -> P(win)."""
        p_clipped = np.clip(raw_p_win, 1e-6, 1.0 - 1e-6)
        logit = float(np.log(p_clipped / (1.0 - p_clipped)))
        proba = self._calibrator.predict_proba(np.array([[logit]]))[:, 1]
        return float(proba[0])

    def _apply_isotonic(self, raw_p_win: float) -> float:
        """Isotonic regression: direct transform."""
        result = self._calibrator.transform([raw_p_win])
        return float(result[0])

    # ==================================================================
    # Feature extraction helpers
    # ==================================================================

    @staticmethod
    def _extract_legacy_features(setup: Any, market_view: Any) -> np.ndarray:
        """Build a feature vector from the legacy setup + market_view.

        This is intentionally defensive: it pulls whatever attributes
        exist and zero-fills the rest so the classifier always receives
        a consistently-shaped input.
        """
        feats: List[float] = []

        # -- setup-level features --
        for attr in (
            "score", "strength", "confidence", "rsi", "atr",
            "momentum", "volatility", "trend_score",
        ):
            feats.append(float(getattr(setup, attr, 0.0)))

        # -- market_view-level features --
        if market_view is not None:
            for attr in (
                "trend_score", "volatility", "momentum",
                "volume_ratio", "spread", "rsi", "atr",
                "bb_width", "macd_hist",
            ):
                feats.append(float(getattr(market_view, attr, 0.0)))
        else:
            feats.extend([0.0] * 9)

        return np.array(feats, dtype=np.float32)

    @staticmethod
    def _compute_confluence(setup: Any, market_view: Any) -> float:
        """Compute a rule-based confluence score (0-100).

        Tries to import the project's ``ConfluenceScorer`` first; falls
        back to a simple attribute-based heuristic.
        """
        try:
            from src.scoring.confluence import ConfluenceScorer  # type: ignore[import-untyped]
            scorer = ConfluenceScorer()
            return float(scorer.score(setup, market_view))
        except Exception:
            pass

        # Fallback heuristic
        score = 50.0

        for attr, weight in (
            ("score", 1.0),
            ("strength", 0.8),
            ("confidence", 0.6),
        ):
            val = getattr(setup, attr, None)
            if val is not None:
                score += float(val) * weight

        if market_view is not None:
            trend = getattr(market_view, "trend_score", None)
            if trend is not None:
                score += float(trend) * 0.5

        return max(0.0, min(100.0, score))

    # ==================================================================
    # Rejection reason builders
    # ==================================================================

    def _rejection_reason_v3(self, p_win: float, expected_r: float) -> str:
        parts: List[str] = []
        if p_win < self._config.min_prob_win:
            parts.append(
                f"P(win) {p_win:.3f} < {self._config.min_prob_win:.2f}"
            )
        if expected_r < self._config.min_expected_r:
            parts.append(
                f"E[R] {expected_r:.3f} < {self._config.min_expected_r:.2f}"
            )
        return "v3 rejected: " + "; ".join(parts) if parts else "v3 rejected"

    def _rejection_reason_legacy(
        self, p_win: float, expected_r: float, confluence: float,
    ) -> str:
        parts: List[str] = []
        if p_win < self._config.min_prob_win:
            parts.append(
                f"P(win) {p_win:.3f} < {self._config.min_prob_win:.2f}"
            )
        if expected_r < self._config.min_expected_r:
            parts.append(
                f"E[R] {expected_r:.3f} < {self._config.min_expected_r:.2f}"
            )
        if confluence < self._config.min_confluence_score:
            parts.append(
                f"confluence {confluence:.0f} < {self._config.min_confluence_score:.0f}"
            )
        return "legacy rejected: " + "; ".join(parts) if parts else "legacy rejected"

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _no_model_decision(reason: str) -> GateDecision:
        return GateDecision(
            decision=GateDecisionType.NO_MODEL,
            reason=reason,
            metadata={"pipeline": "none"},
        )
