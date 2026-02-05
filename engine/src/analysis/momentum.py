"""Momentum Engine for confirming move quality.

Analyzes:
- RSI regime and divergence
- ATR expansion/contraction
- Candle efficiency (impulse vs correction)
- Momentum confirmation signals
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class MomentumBias(Enum):
    """Momentum directional bias."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class DivergenceType(Enum):
    """Types of divergence."""
    BULLISH_REGULAR = "BULLISH_REGULAR"  # Lower price low, higher RSI low
    BEARISH_REGULAR = "BEARISH_REGULAR"  # Higher price high, lower RSI high
    BULLISH_HIDDEN = "BULLISH_HIDDEN"    # Higher price low, lower RSI low
    BEARISH_HIDDEN = "BEARISH_HIDDEN"    # Lower price high, higher RSI high
    NONE = "NONE"


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXPANDING = "EXPANDING"
    CONTRACTING = "CONTRACTING"


@dataclass
class MomentumState:
    """Current momentum state."""
    timestamp: datetime
    symbol: str
    timeframe: str

    # RSI
    rsi: float
    rsi_regime: MomentumBias  # Above/below 50

    # Divergence
    divergence: DivergenceType
    divergence_strength: float  # 0-1

    # ATR/Volatility
    atr: float
    atr_percentile: float  # Current ATR vs historical
    volatility_regime: VolatilityRegime

    # Efficiency
    impulse_ratio: float  # Ratio of impulse to correction moves
    candle_efficiency: float  # Body/range ratio

    # Overall
    bias: MomentumBias
    confidence: float  # 0-1
    warnings: List[str]


class MomentumEngine:
    """Analyzes momentum quality for trade confirmation.

    Uses RSI, ATR, and candle analysis to confirm or warn
    against potential trade setups.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        atr_period: int = 14,
        lookback_swings: int = 5,
    ):
        """Initialize momentum engine.

        Args:
            rsi_period: RSI calculation period
            atr_period: ATR calculation period
            lookback_swings: Swings to check for divergence
        """
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.lookback_swings = lookback_swings

    def analyze(
        self,
        symbol: str,
        timeframe: str,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: List[datetime],
        swing_highs: Optional[List[Tuple[int, float]]] = None,
        swing_lows: Optional[List[Tuple[int, float]]] = None,
    ) -> MomentumState:
        """Analyze momentum state.

        Args:
            symbol: Instrument symbol
            timeframe: Chart timeframe
            highs, lows, closes: Price arrays
            timestamps: Bar timestamps
            swing_highs: Optional list of (index, price) swing highs
            swing_lows: Optional list of (index, price) swing lows

        Returns:
            MomentumState with analysis results
        """
        analysis_time = datetime.now()

        if len(closes) < max(self.rsi_period, self.atr_period) + 10:
            return self._empty_state(analysis_time, symbol, timeframe)

        # Calculate RSI
        rsi = self._calculate_rsi(closes)
        rsi_current = rsi[-1] if len(rsi) > 0 else 50

        # RSI regime
        if rsi_current > 60:
            rsi_regime = MomentumBias.BULLISH
        elif rsi_current < 40:
            rsi_regime = MomentumBias.BEARISH
        else:
            rsi_regime = MomentumBias.NEUTRAL

        # Calculate ATR
        atr = self._calculate_atr(highs, lows, closes)
        atr_current = atr[-1] if len(atr) > 0 else 0
        atr_percentile = self._calculate_percentile(atr, atr_current)

        # Volatility regime
        vol_regime = self._classify_volatility(atr)

        # Check for divergence
        divergence, div_strength = self._detect_divergence(
            closes, rsi, swing_highs, swing_lows
        )

        # Calculate efficiency metrics
        impulse_ratio = self._calculate_impulse_ratio(highs, lows, closes)
        candle_efficiency = self._calculate_candle_efficiency(
            highs[-20:], lows[-20:], closes[-20:]
        )

        # Determine overall bias and warnings
        bias, confidence, warnings = self._determine_bias(
            rsi_current, rsi_regime, divergence, vol_regime,
            impulse_ratio, candle_efficiency
        )

        return MomentumState(
            timestamp=analysis_time,
            symbol=symbol,
            timeframe=timeframe,
            rsi=rsi_current,
            rsi_regime=rsi_regime,
            divergence=divergence,
            divergence_strength=div_strength,
            atr=atr_current,
            atr_percentile=atr_percentile,
            volatility_regime=vol_regime,
            impulse_ratio=impulse_ratio,
            candle_efficiency=candle_efficiency,
            bias=bias,
            confidence=confidence,
            warnings=warnings,
        )

    def _calculate_rsi(self, closes: np.ndarray) -> np.ndarray:
        """Calculate RSI indicator."""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Initial average
        avg_gain = np.mean(gains[:self.rsi_period])
        avg_loss = np.mean(losses[:self.rsi_period])

        rsi = np.zeros(len(closes))
        rsi[:self.rsi_period] = 50  # Default for warmup period

        for i in range(self.rsi_period, len(closes)):
            avg_gain = (avg_gain * (self.rsi_period - 1) + gains[i - 1]) / self.rsi_period
            avg_loss = (avg_loss * (self.rsi_period - 1) + losses[i - 1]) / self.rsi_period

            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> np.ndarray:
        """Calculate Average True Range."""
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]

        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        # Smoothed ATR
        atr = np.zeros(len(closes))
        atr[:self.atr_period] = np.mean(tr[:self.atr_period])

        for i in range(self.atr_period, len(closes)):
            atr[i] = (atr[i - 1] * (self.atr_period - 1) + tr[i]) / self.atr_period

        return atr

    def _calculate_percentile(self, values: np.ndarray, current: float) -> float:
        """Calculate percentile of current value in historical distribution."""
        if len(values) < 10:
            return 50.0
        return float(np.sum(values <= current) / len(values) * 100)

    def _classify_volatility(self, atr: np.ndarray) -> VolatilityRegime:
        """Classify current volatility regime."""
        if len(atr) < 20:
            return VolatilityRegime.MEDIUM

        current = atr[-1]
        recent_avg = np.mean(atr[-20:])
        older_avg = np.mean(atr[-50:-20]) if len(atr) >= 50 else recent_avg

        # Check for expansion/contraction
        if current > recent_avg * 1.2:
            return VolatilityRegime.EXPANDING
        elif current < recent_avg * 0.8:
            return VolatilityRegime.CONTRACTING

        # Classify level
        percentile = self._calculate_percentile(atr, current)
        if percentile > 70:
            return VolatilityRegime.HIGH
        elif percentile < 30:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.MEDIUM

    def _detect_divergence(
        self,
        closes: np.ndarray,
        rsi: np.ndarray,
        swing_highs: Optional[List[Tuple[int, float]]],
        swing_lows: Optional[List[Tuple[int, float]]]
    ) -> Tuple[DivergenceType, float]:
        """Detect RSI divergence from price."""
        if swing_highs is None or swing_lows is None:
            return DivergenceType.NONE, 0.0

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return DivergenceType.NONE, 0.0

        # Check for bearish divergence (higher price high, lower RSI high)
        if len(swing_highs) >= 2:
            h1_idx, h1_price = swing_highs[-2]
            h2_idx, h2_price = swing_highs[-1]

            if h2_idx < len(rsi) and h1_idx < len(rsi):
                h1_rsi = rsi[h1_idx]
                h2_rsi = rsi[h2_idx]

                if h2_price > h1_price and h2_rsi < h1_rsi:
                    strength = abs(h1_rsi - h2_rsi) / 20  # Normalize
                    return DivergenceType.BEARISH_REGULAR, min(strength, 1.0)

                if h2_price < h1_price and h2_rsi > h1_rsi:
                    strength = abs(h1_rsi - h2_rsi) / 20
                    return DivergenceType.BEARISH_HIDDEN, min(strength, 1.0)

        # Check for bullish divergence (lower price low, higher RSI low)
        if len(swing_lows) >= 2:
            l1_idx, l1_price = swing_lows[-2]
            l2_idx, l2_price = swing_lows[-1]

            if l2_idx < len(rsi) and l1_idx < len(rsi):
                l1_rsi = rsi[l1_idx]
                l2_rsi = rsi[l2_idx]

                if l2_price < l1_price and l2_rsi > l1_rsi:
                    strength = abs(l2_rsi - l1_rsi) / 20
                    return DivergenceType.BULLISH_REGULAR, min(strength, 1.0)

                if l2_price > l1_price and l2_rsi < l1_rsi:
                    strength = abs(l2_rsi - l1_rsi) / 20
                    return DivergenceType.BULLISH_HIDDEN, min(strength, 1.0)

        return DivergenceType.NONE, 0.0

    def _calculate_impulse_ratio(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> float:
        """Calculate ratio of impulse to correction moves."""
        if len(closes) < 20:
            return 0.5

        # Calculate up and down moves
        changes = np.diff(closes)
        up_moves = np.sum(np.abs(changes[changes > 0]))
        down_moves = np.sum(np.abs(changes[changes < 0]))

        total = up_moves + down_moves
        if total == 0:
            return 0.5

        # Return ratio favoring recent direction
        recent_change = closes[-1] - closes[-20]
        if recent_change > 0:
            return up_moves / total
        else:
            return down_moves / total

    def _calculate_candle_efficiency(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> float:
        """Calculate average candle efficiency (body/range ratio)."""
        if len(closes) < 2:
            return 0.5

        opens = np.roll(closes, 1)
        opens[0] = closes[0]

        bodies = np.abs(closes - opens)
        ranges = highs - lows

        # Avoid division by zero
        ranges = np.where(ranges == 0, 1, ranges)
        efficiencies = bodies / ranges

        return float(np.mean(efficiencies))

    def _determine_bias(
        self,
        rsi: float,
        rsi_regime: MomentumBias,
        divergence: DivergenceType,
        vol_regime: VolatilityRegime,
        impulse_ratio: float,
        candle_efficiency: float
    ) -> Tuple[MomentumBias, float, List[str]]:
        """Determine overall momentum bias and warnings."""
        warnings = []
        confidence = 0.5

        # Start with RSI regime
        bias = rsi_regime

        # Adjust for divergence
        if divergence == DivergenceType.BEARISH_REGULAR:
            if bias == MomentumBias.BULLISH:
                warnings.append("Bearish RSI divergence against bullish momentum")
                confidence -= 0.2
            else:
                bias = MomentumBias.BEARISH
                confidence += 0.1

        elif divergence == DivergenceType.BULLISH_REGULAR:
            if bias == MomentumBias.BEARISH:
                warnings.append("Bullish RSI divergence against bearish momentum")
                confidence -= 0.2
            else:
                bias = MomentumBias.BULLISH
                confidence += 0.1

        # Volatility warnings
        if vol_regime == VolatilityRegime.LOW:
            warnings.append("Low volatility - potential breakout pending")
        elif vol_regime == VolatilityRegime.HIGH:
            warnings.append("High volatility - increased risk")

        # Efficiency considerations
        if candle_efficiency < 0.4:
            warnings.append("Low candle efficiency - choppy price action")
            confidence -= 0.1

        # RSI extremes
        if rsi > 80:
            warnings.append("RSI overbought (>80)")
            if bias == MomentumBias.BULLISH:
                confidence -= 0.15
        elif rsi < 20:
            warnings.append("RSI oversold (<20)")
            if bias == MomentumBias.BEARISH:
                confidence -= 0.15

        # Normalize confidence
        confidence = max(0.1, min(0.95, confidence))

        return bias, confidence, warnings

    def _empty_state(
        self,
        timestamp: datetime,
        symbol: str,
        timeframe: str
    ) -> MomentumState:
        """Return empty state when data is insufficient."""
        return MomentumState(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            rsi=50,
            rsi_regime=MomentumBias.NEUTRAL,
            divergence=DivergenceType.NONE,
            divergence_strength=0,
            atr=0,
            atr_percentile=50,
            volatility_regime=VolatilityRegime.MEDIUM,
            impulse_ratio=0.5,
            candle_efficiency=0.5,
            bias=MomentumBias.NEUTRAL,
            confidence=0,
            warnings=["Insufficient data for momentum analysis"],
        )

    def to_dict(self, state: MomentumState) -> Dict:
        """Convert state to dictionary for storage."""
        return {
            "timestamp": state.timestamp.isoformat(),
            "symbol": state.symbol,
            "timeframe": state.timeframe,
            "rsi": state.rsi,
            "rsi_regime": state.rsi_regime.value,
            "divergence": state.divergence.value,
            "divergence_strength": state.divergence_strength,
            "atr": state.atr,
            "atr_percentile": state.atr_percentile,
            "volatility_regime": state.volatility_regime.value,
            "impulse_ratio": state.impulse_ratio,
            "candle_efficiency": state.candle_efficiency,
            "bias": state.bias.value,
            "confidence": state.confidence,
            "warnings": state.warnings,
        }
