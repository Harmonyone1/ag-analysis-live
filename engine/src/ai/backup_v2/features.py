"""Feature extraction for AI decision gate.

Extracts normalized features from trade setups and market analysis
for input to the classification model.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import structlog

from src.scoring.confluence import TradeSetup
from src.analysis.analyzer import MarketView

logger = structlog.get_logger(__name__)


@dataclass
class FeatureVector:
    """Extracted feature vector for model input."""
    features: Dict[str, float]
    feature_array: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]


class FeatureExtractor:
    """Extracts and normalizes features for the AI gate model.

    Features are organized into categories:
    - Strength features (currency strength, divergence)
    - Structure features (trend, BOS, market phase)
    - Liquidity features (zone quality, sweep recency)
    - Momentum features (RSI, ATR ratio, impulse)
    - Context features (session, time, volatility regime)

    Example:
        extractor = FeatureExtractor()
        features = extractor.extract(setup, market_view)
        model_input = features.feature_array
    """

    # Feature normalization ranges
    FEATURE_RANGES = {
        "confluence_score": (0, 100),
        "strength_score": (-100, 100),
        "strength_divergence": (0, 100),
        "rsi": (0, 100),
        "atr_ratio": (0, 5),
        "impulse_ratio": (0, 3),
        "risk_reward": (0, 10),
        "stop_distance_atr": (0, 5),
        "zone_quality": (0, 100),
        "sweep_recency": (0, 100),
        "bos_count": (0, 10),
        "session_hour": (0, 24),
        "day_of_week": (0, 6),
    }

    def __init__(self):
        """Initialize feature extractor."""
        self._feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        """Build ordered list of feature names."""
        return [
            # Confluence
            "confluence_score",
            "reason_count",
            # Strength
            "base_strength",
            "quote_strength",
            "strength_divergence",
            "strength_trend_aligned",
            # Structure
            "structure_score",
            "trend_bullish",
            "trend_bearish",
            "bos_count_recent",
            "higher_tf_aligned",
            # Liquidity
            "zone_quality",
            "sweep_recency_minutes",
            "entry_zone_width_pips",
            "stop_behind_liquidity",
            # Momentum
            "rsi_value",
            "rsi_oversold",
            "rsi_overbought",
            "atr_ratio",
            "impulse_ratio",
            "momentum_aligned",
            # Trade parameters
            "risk_reward_ratio",
            "stop_distance_atr",
            "entry_type_limit",
            # Context
            "session_london",
            "session_ny",
            "session_overlap",
            "hour_of_day",
            "day_of_week",
            "news_within_2h",
            # Historical (requires trade history)
            "symbol_recent_win_rate",
            "setup_type_win_rate",
        ]

    def extract(
        self,
        setup: TradeSetup,
        market_view: Optional[MarketView] = None,
        trade_history: Optional[List[Dict]] = None,
    ) -> FeatureVector:
        """Extract features from a trade setup.

        Args:
            setup: Trade setup to extract features from
            market_view: Full market analysis view
            trade_history: Recent trade history for this symbol

        Returns:
            FeatureVector with normalized features
        """
        features: Dict[str, float] = {}

        # Confluence features
        features["confluence_score"] = self._normalize(
            setup.confluence_score, "confluence_score"
        )
        features["reason_count"] = min(len(setup.reasons) / 10.0, 1.0)

        # Extract strength features
        self._extract_strength_features(features, setup, market_view)

        # Extract structure features
        self._extract_structure_features(features, setup, market_view)

        # Extract liquidity features
        self._extract_liquidity_features(features, setup, market_view)

        # Extract momentum features
        self._extract_momentum_features(features, setup, market_view)

        # Extract trade parameter features
        self._extract_trade_features(features, setup)

        # Extract context features
        self._extract_context_features(features, setup)

        # Extract historical features
        self._extract_historical_features(features, setup, trade_history)

        # Build feature array in consistent order
        feature_array = np.array([
            features.get(name, 0.0) for name in self._feature_names
        ], dtype=np.float32)

        return FeatureVector(
            features=features,
            feature_array=feature_array,
            feature_names=self._feature_names.copy(),
            metadata={
                "symbol": setup.symbol,
                "direction": setup.direction,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def _extract_strength_features(
        self,
        features: Dict[str, float],
        setup: TradeSetup,
        market_view: Optional[MarketView],
    ) -> None:
        """Extract currency strength features."""
        if market_view and market_view.strength:
            strength = market_view.strength
            base_ccy = setup.symbol[:3].upper()
            quote_ccy = setup.symbol[3:6].upper()

            base_str = strength.currencies.get(base_ccy, 0)
            quote_str = strength.currencies.get(quote_ccy, 0)

            features["base_strength"] = self._normalize(base_str, "strength_score")
            features["quote_strength"] = self._normalize(quote_str, "strength_score")
            features["strength_divergence"] = self._normalize(
                abs(base_str - quote_str), "strength_divergence"
            )

            # Check if strength aligns with direction
            if setup.direction == "LONG":
                aligned = base_str > quote_str
            else:
                aligned = quote_str > base_str
            features["strength_trend_aligned"] = 1.0 if aligned else 0.0
        else:
            features["base_strength"] = 0.5
            features["quote_strength"] = 0.5
            features["strength_divergence"] = 0.0
            features["strength_trend_aligned"] = 0.5

    def _extract_structure_features(
        self,
        features: Dict[str, float],
        setup: TradeSetup,
        market_view: Optional[MarketView],
    ) -> None:
        """Extract market structure features."""
        if market_view and market_view.structure:
            struct = market_view.structure

            # Map structure state to a score
            state_scores = {"trending": 0.8, "ranging": 0.5, "breakout": 0.7, "reversal": 0.6}
            features["structure_score"] = state_scores.get(str(struct.state.value).lower(), 0.5)

            # Trend encoding using trend_direction
            trend = str(struct.trend_direction.value).upper() if struct.trend_direction else ""
            features["trend_bullish"] = 1.0 if trend == "BULLISH" else 0.0
            features["trend_bearish"] = 1.0 if trend == "BEARISH" else 0.0

            # BOS indicator (recent_bos is singular, not a list)
            features["bos_count_recent"] = 1.0 if struct.recent_bos else 0.0

            # Higher timeframe alignment - check if setup direction matches structure trend
            htf_aligned = self._check_higher_tf_alignment(setup, struct, market_view)
            features["higher_tf_aligned"] = htf_aligned
        else:
            features["structure_score"] = 0.5
            features["trend_bullish"] = 0.0
            features["trend_bearish"] = 0.0
            features["bos_count_recent"] = 0.0
            features["higher_tf_aligned"] = 0.5

    def _check_higher_tf_alignment(
        self,
        setup: TradeSetup,
        struct,
        market_view: Optional[MarketView],
    ) -> float:
        """Check if setup direction aligns with higher timeframe trend."""
        alignment_score = 0.5  # Neutral default

        # Check structure trend direction
        if struct and struct.trend_direction:
            trend = str(struct.trend_direction.value).upper()
            if setup.direction == "LONG" and trend == "BULLISH":
                alignment_score = 1.0
            elif setup.direction == "SHORT" and trend == "BEARISH":
                alignment_score = 1.0
            elif setup.direction == "LONG" and trend == "BEARISH":
                alignment_score = 0.0  # Counter-trend
            elif setup.direction == "SHORT" and trend == "BULLISH":
                alignment_score = 0.0  # Counter-trend

        # Additional check: Bias confidence from market view
        if market_view and market_view.bias_confidence:
            bias = market_view.directional_bias
            if setup.direction == bias:
                # Same direction as overall bias, boost alignment
                alignment_score = min(1.0, alignment_score + 0.2)
            elif bias == "NEUTRAL":
                # Neutral bias, keep current score
                pass
            else:
                # Against bias, reduce alignment
                alignment_score = max(0.0, alignment_score - 0.2)

        return alignment_score

    def _extract_liquidity_features(
        self,
        features: Dict[str, float],
        setup: TradeSetup,
        market_view: Optional[MarketView],
    ) -> None:
        """Extract liquidity zone features."""
        if market_view and market_view.liquidity:
            liq = market_view.liquidity

            # Zone quality from most recent sweep (use recent_sweeps, not sweeps)
            if liq.recent_sweeps:
                recent_sweep = liq.recent_sweeps[0]
                # quality_score is already 0-1
                features["zone_quality"] = min(recent_sweep.quality_score, 1.0)

                # Sweep recency in normalized form
                recency_minutes = (
                    datetime.now() - recent_sweep.timestamp
                ).total_seconds() / 60.0
                features["sweep_recency_minutes"] = max(0, 1.0 - recency_minutes / 60.0)
            else:
                features["zone_quality"] = 0.5
                features["sweep_recency_minutes"] = 0.0

            # Entry zone width
            if setup.entry_zone:
                zone_width = abs(setup.entry_zone[1] - setup.entry_zone[0])
                # Normalize assuming typical 10 pip zones
                features["entry_zone_width_pips"] = min(zone_width * 10000 / 20.0, 1.0)
            else:
                features["entry_zone_width_pips"] = 0.5

            # Stop behind liquidity - check if stop is protected by a liquidity zone
            features["stop_behind_liquidity"] = self._check_stop_behind_liquidity(
                setup, liq
            )
        else:
            features["zone_quality"] = 0.5
            features["sweep_recency_minutes"] = 0.0
            features["entry_zone_width_pips"] = 0.5
            features["stop_behind_liquidity"] = 0.5

    def _check_stop_behind_liquidity(
        self,
        setup: TradeSetup,
        liq,
    ) -> float:
        """Check if stop loss is placed behind a liquidity zone for protection."""
        if not setup.stop_price:
            return 0.5

        stop_price = float(setup.stop_price)
        protection_score = 0.0

        # For LONG trades, check if stop is below a support/liquidity zone
        if setup.direction == "LONG":
            # Check sell-side liquidity (lows) - stop should be behind these
            for zone in (liq.sell_side_liquidity or []):
                zone_price = float(zone.price)
                # Stop is behind liquidity if it's below the zone
                if stop_price < zone_price:
                    # The closer the stop is to the zone, the better protected
                    distance = zone_price - stop_price
                    # Normalize: within 20 pips is ideal
                    if distance < 0.0020:  # 20 pips for most pairs
                        protection_score = max(protection_score, 1.0 - (distance / 0.0020) * 0.5)
                    else:
                        protection_score = max(protection_score, 0.5)

            # Also check nearest support
            if liq.nearest_support:
                support_price = float(liq.nearest_support.price)
                if stop_price < support_price:
                    protection_score = max(protection_score, 0.8)

        # For SHORT trades, check if stop is above a resistance/liquidity zone
        else:
            # Check buy-side liquidity (highs) - stop should be behind these
            for zone in (liq.buy_side_liquidity or []):
                zone_price = float(zone.price)
                # Stop is behind liquidity if it's above the zone
                if stop_price > zone_price:
                    distance = stop_price - zone_price
                    if distance < 0.0020:
                        protection_score = max(protection_score, 1.0 - (distance / 0.0020) * 0.5)
                    else:
                        protection_score = max(protection_score, 0.5)

            # Also check nearest resistance
            if liq.nearest_resistance:
                resistance_price = float(liq.nearest_resistance.price)
                if stop_price > resistance_price:
                    protection_score = max(protection_score, 0.8)

        return protection_score

    def _extract_momentum_features(
        self,
        features: Dict[str, float],
        setup: TradeSetup,
        market_view: Optional[MarketView],
    ) -> None:
        """Extract momentum features."""
        if market_view and market_view.momentum:
            mom = market_view.momentum

            # RSI
            features["rsi_value"] = self._normalize(mom.rsi, "rsi")
            features["rsi_oversold"] = 1.0 if mom.rsi < 30 else 0.0
            features["rsi_overbought"] = 1.0 if mom.rsi > 70 else 0.0

            # ATR percentile (use atr_percentile instead of atr_ratio)
            features["atr_ratio"] = mom.atr_percentile if mom.atr_percentile else 0.5

            # Impulse ratio
            features["impulse_ratio"] = self._normalize(
                mom.impulse_ratio if mom.impulse_ratio else 1.0, "impulse_ratio"
            )

            # Momentum alignment with direction (use rsi_regime instead of bias)
            if setup.direction == "LONG":
                regime = str(mom.rsi_regime.value).upper() if mom.rsi_regime else ""
                aligned = regime == "BULLISH"
            else:
                regime = str(mom.rsi_regime.value).upper() if mom.rsi_regime else ""
                aligned = regime == "BEARISH"
            features["momentum_aligned"] = 1.0 if aligned else 0.0
        else:
            features["rsi_value"] = 0.5
            features["rsi_oversold"] = 0.0
            features["rsi_overbought"] = 0.0
            features["atr_ratio"] = 0.5
            features["impulse_ratio"] = 0.5
            features["momentum_aligned"] = 0.5

    def _extract_trade_features(
        self,
        features: Dict[str, float],
        setup: TradeSetup,
    ) -> None:
        """Extract trade parameter features."""
        # Risk/reward ratio
        rr = setup.risk_reward or 0.0
        features["risk_reward_ratio"] = self._normalize(rr, "risk_reward")

        # Stop distance in ATR terms (approximate)
        if setup.entry_zone and setup.stop_price:
            entry_mid = (setup.entry_zone[0] + setup.entry_zone[1]) / 2
            stop_dist = abs(entry_mid - setup.stop_price)
            # Normalize assuming 1 ATR = 50 pips roughly
            # Normalize: convert to pips, then scale by ~50 pip ATR
            pip_divisor = 0.01 if "JPY" in setup.symbol else 0.0001
            stop_pips = stop_dist / pip_divisor
            features["stop_distance_atr"] = min(stop_pips / 50.0, 1.0)
        else:
            features["stop_distance_atr"] = 0.5

        # Entry type encoding
        features["entry_type_limit"] = 1.0 if setup.entry_type == "LIMIT" else 0.0

    def _extract_context_features(
        self,
        features: Dict[str, float],
        setup: TradeSetup,
    ) -> None:
        """Extract contextual features."""
        now = datetime.now()

        # Session encoding (UTC-based)
        hour = now.hour
        features["session_london"] = 1.0 if 7 <= hour < 16 else 0.0
        features["session_ny"] = 1.0 if 12 <= hour < 21 else 0.0
        features["session_overlap"] = 1.0 if 12 <= hour < 16 else 0.0

        # Time features
        features["hour_of_day"] = hour / 24.0
        features["day_of_week"] = now.weekday() / 6.0

        # News/high volatility risk detection
        features["news_within_2h"] = self._detect_high_volatility_window(now, setup.symbol)

    def _detect_high_volatility_window(self, now: datetime, symbol: str) -> float:
        """Detect if we're in a high volatility window (session opens, common news times).

        Returns:
            0.0 = No expected volatility
            0.5 = Moderate volatility expected
            1.0 = High volatility expected (avoid trading)
        """
        hour = now.hour
        minute = now.minute
        day = now.weekday()  # 0=Monday, 4=Friday

        # High volatility windows (UTC times)
        high_vol_windows = [
            # London open (7:00-8:30 UTC)
            (7, 0, 8, 30),
            # US open / US data releases (12:30-14:30 UTC)
            (12, 30, 14, 30),
            # FOMC / Major US news (typically 18:00-19:00 UTC on Wednesdays)
            (18, 0, 19, 0) if day == 2 else None,
        ]

        # Check major currency-specific windows
        base_ccy = symbol[:3].upper()
        quote_ccy = symbol[3:6].upper()

        # EUR pairs: ECB news typically Thursday 12:45 UTC
        if "EUR" in [base_ccy, quote_ccy] and day == 3:
            if 12 <= hour < 14:
                return 1.0

        # GBP pairs: BOE news typically Thursday
        if "GBP" in [base_ccy, quote_ccy] and day == 3:
            if 11 <= hour < 13:
                return 1.0

        # JPY pairs: BOJ news (varies)
        # USD pairs: NFP first Friday (usually 12:30 UTC)
        if "USD" in [base_ccy, quote_ccy] and day == 4:
            # First Friday - likely NFP week
            if now.day <= 7 and 12 <= hour < 14:
                return 1.0

        # Check general high volatility windows
        current_minutes = hour * 60 + minute
        for window in high_vol_windows:
            if window is None:
                continue
            start_h, start_m, end_h, end_m = window
            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m

            if start_minutes <= current_minutes <= end_minutes:
                return 0.8  # High but not maximum (allows trading with caution)

        # Sunday evening / Monday early - low liquidity
        if day == 6 or (day == 0 and hour < 7):
            return 0.6

        # Friday late - weekend risk
        if day == 4 and hour >= 20:
            return 0.5

        return 0.0  # Normal conditions

    def _extract_historical_features(
        self,
        features: Dict[str, float],
        setup: TradeSetup,
        trade_history: Optional[List[Dict]],
    ) -> None:
        """Extract historical performance features."""
        if trade_history:
            # Filter recent trades for this symbol
            symbol_trades = [
                t for t in trade_history
                if t.get("symbol") == setup.symbol
            ][-20:]

            if symbol_trades:
                wins = sum(1 for t in symbol_trades if t.get("pnl", 0) > 0)
                features["symbol_recent_win_rate"] = wins / len(symbol_trades)
            else:
                features["symbol_recent_win_rate"] = 0.5

            # Setup type win rate - based on similar setups
            features["setup_type_win_rate"] = self._calculate_setup_type_win_rate(
                setup, trade_history
            )
        else:
            features["symbol_recent_win_rate"] = 0.5
            features["setup_type_win_rate"] = 0.5

    def _calculate_setup_type_win_rate(
        self,
        setup: TradeSetup,
        trade_history: List[Dict],
    ) -> float:
        """Calculate win rate for similar setup types."""
        if not trade_history:
            return 0.5

        # Filter by direction
        direction_trades = [
            t for t in trade_history
            if t.get("direction") == setup.direction
        ][-30:]

        if not direction_trades:
            return 0.5

        # Further filter by similar confluence score range (+/- 10)
        similar_trades = [
            t for t in direction_trades
            if abs(t.get("confluence_score", 0) - setup.confluence_score) <= 10
        ]

        if len(similar_trades) >= 5:
            wins = sum(1 for t in similar_trades if t.get("pnl", 0) > 0)
            return wins / len(similar_trades)

        # Fall back to direction-only win rate
        wins = sum(1 for t in direction_trades if t.get("pnl", 0) > 0)
        return wins / len(direction_trades)

    def _normalize(self, value: float, feature_name: str) -> float:
        """Normalize a value to [0, 1] range."""
        if feature_name not in self.FEATURE_RANGES:
            return float(value)

        min_val, max_val = self.FEATURE_RANGES[feature_name]
        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)
        return float(np.clip(normalized, 0.0, 1.0))

    @property
    def feature_count(self) -> int:
        """Get number of features."""
        return len(self._feature_names)

    @property
    def feature_names(self) -> List[str]:
        """Get ordered feature names."""
        return self._feature_names.copy()
