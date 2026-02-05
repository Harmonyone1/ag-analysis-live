"""Market Structure Engine for trend and range detection.

Detects swing highs/lows, higher-high/lower-low sequences,
breaks of structure, and range/distribution conditions.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class StructureState(Enum):
    """Market structure states."""
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    DISTRIBUTION = "DISTRIBUTION"
    ACCUMULATION = "ACCUMULATION"
    UNKNOWN = "UNKNOWN"


class TrendDirection(Enum):
    """Trend direction."""
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class SwingType(Enum):
    """Swing point types."""
    HIGH = "HIGH"
    LOW = "LOW"
    HIGHER_HIGH = "HIGHER_HIGH"
    LOWER_HIGH = "LOWER_HIGH"
    HIGHER_LOW = "HIGHER_LOW"
    LOWER_LOW = "LOWER_LOW"


@dataclass
class SwingPoint:
    """A swing high or low point."""
    timestamp: datetime
    price: float
    swing_type: SwingType
    index: int
    confirmed: bool = True


@dataclass
class BreakOfStructure:
    """Break of structure event."""
    timestamp: datetime
    price: float
    direction: TrendDirection  # Direction of the break
    broken_level: float
    swing_point: SwingPoint


@dataclass
class StructureAnalysis:
    """Complete structure analysis output."""
    timestamp: datetime
    symbol: str
    timeframe: str

    # Current state
    state: StructureState
    trend_direction: TrendDirection

    # Swing points
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    last_swing_high: Optional[SwingPoint]
    last_swing_low: Optional[SwingPoint]

    # Key levels
    range_high: Optional[float]
    range_low: Optional[float]

    # Invalidation
    bullish_invalidation: Optional[float]  # Below this = bearish
    bearish_invalidation: Optional[float]  # Above this = bullish

    # Recent events
    recent_bos: Optional[BreakOfStructure]


class StructureEngine:
    """Analyzes market structure using price action.

    Detects:
    - Swing highs and lows using fractal method
    - Higher-high/lower-low sequences for trend
    - Range conditions when no clear trend
    - Breaks of structure (BOS) events
    """

    def __init__(self, swing_lookback: int = 5):
        """Initialize structure engine.

        Args:
            swing_lookback: Bars on each side to confirm swing
        """
        self.swing_lookback = swing_lookback

    def analyze(
        self,
        symbol: str,
        timeframe: str,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: List[datetime],
    ) -> StructureAnalysis:
        """Analyze market structure from OHLC data.

        Args:
            symbol: Instrument symbol
            timeframe: Chart timeframe
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices
            timestamps: List of bar timestamps

        Returns:
            StructureAnalysis with detected structure
        """
        analysis_time = datetime.now()

        if len(highs) < self.swing_lookback * 2 + 1:
            return self._empty_analysis(analysis_time, symbol, timeframe)

        # Detect swing points
        swing_highs = self._detect_swing_highs(highs, timestamps)
        swing_lows = self._detect_swing_lows(lows, timestamps)

        # Classify swing points as HH/LH/HL/LL
        swing_highs = self._classify_swings(swing_highs, is_high=True)
        swing_lows = self._classify_swings(swing_lows, is_high=False)

        # Determine structure state and trend
        state, trend = self._determine_structure(swing_highs, swing_lows, closes)

        # Find range boundaries
        range_high, range_low = self._find_range(swing_highs, swing_lows)

        # Calculate invalidation levels
        bullish_inv, bearish_inv = self._calculate_invalidations(
            swing_highs, swing_lows, trend
        )

        # Detect recent break of structure
        recent_bos = self._detect_bos(swing_highs, swing_lows, closes, timestamps)

        return StructureAnalysis(
            timestamp=analysis_time,
            symbol=symbol,
            timeframe=timeframe,
            state=state,
            trend_direction=trend,
            swing_highs=swing_highs[-10:],  # Last 10
            swing_lows=swing_lows[-10:],
            last_swing_high=swing_highs[-1] if swing_highs else None,
            last_swing_low=swing_lows[-1] if swing_lows else None,
            range_high=range_high,
            range_low=range_low,
            bullish_invalidation=bullish_inv,
            bearish_invalidation=bearish_inv,
            recent_bos=recent_bos,
        )

    def _detect_swing_highs(
        self,
        highs: np.ndarray,
        timestamps: List[datetime]
    ) -> List[SwingPoint]:
        """Detect swing highs using fractal method."""
        swing_points = []
        n = len(highs)
        lb = self.swing_lookback

        for i in range(lb, n - lb):
            # Check if this is a local maximum
            is_swing = True
            for j in range(1, lb + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing = False
                    break

            if is_swing:
                swing_points.append(SwingPoint(
                    timestamp=timestamps[i],
                    price=float(highs[i]),
                    swing_type=SwingType.HIGH,
                    index=i,
                ))

        return swing_points

    def _detect_swing_lows(
        self,
        lows: np.ndarray,
        timestamps: List[datetime]
    ) -> List[SwingPoint]:
        """Detect swing lows using fractal method."""
        swing_points = []
        n = len(lows)
        lb = self.swing_lookback

        for i in range(lb, n - lb):
            # Check if this is a local minimum
            is_swing = True
            for j in range(1, lb + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing = False
                    break

            if is_swing:
                swing_points.append(SwingPoint(
                    timestamp=timestamps[i],
                    price=float(lows[i]),
                    swing_type=SwingType.LOW,
                    index=i,
                ))

        return swing_points

    def _classify_swings(
        self,
        swings: List[SwingPoint],
        is_high: bool
    ) -> List[SwingPoint]:
        """Classify swings as HH/LH or HL/LL based on sequence."""
        if len(swings) < 2:
            return swings

        for i in range(1, len(swings)):
            prev = swings[i - 1]
            curr = swings[i]

            if is_high:
                if curr.price > prev.price:
                    curr.swing_type = SwingType.HIGHER_HIGH
                else:
                    curr.swing_type = SwingType.LOWER_HIGH
            else:
                if curr.price > prev.price:
                    curr.swing_type = SwingType.HIGHER_LOW
                else:
                    curr.swing_type = SwingType.LOWER_LOW

        return swings

    def _determine_structure(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        closes: np.ndarray
    ) -> Tuple[StructureState, TrendDirection]:
        """Determine market structure state and trend direction."""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return StructureState.UNKNOWN, TrendDirection.NEUTRAL

        # Count HH/LH and HL/LL in recent swings
        recent_highs = swing_highs[-5:]
        recent_lows = swing_lows[-5:]

        hh_count = sum(1 for s in recent_highs if s.swing_type == SwingType.HIGHER_HIGH)
        lh_count = sum(1 for s in recent_highs if s.swing_type == SwingType.LOWER_HIGH)
        hl_count = sum(1 for s in recent_lows if s.swing_type == SwingType.HIGHER_LOW)
        ll_count = sum(1 for s in recent_lows if s.swing_type == SwingType.LOWER_LOW)

        # Determine trend
        if hh_count >= 2 and hl_count >= 2:
            return StructureState.TREND_UP, TrendDirection.UP
        elif lh_count >= 2 and ll_count >= 2:
            return StructureState.TREND_DOWN, TrendDirection.DOWN

        # Check for range/distribution
        if swing_highs and swing_lows:
            high_range = max(s.price for s in recent_highs) - min(s.price for s in recent_highs)
            low_range = max(s.price for s in recent_lows) - min(s.price for s in recent_lows)
            price_range = max(s.price for s in recent_highs) - min(s.price for s in recent_lows)

            # If highs and lows are relatively flat, it's a range
            if price_range > 0 and (high_range / price_range < 0.3 and low_range / price_range < 0.3):
                # Check if near highs (distribution) or lows (accumulation)
                current_price = closes[-1]
                mid_range = (max(s.price for s in recent_highs) + min(s.price for s in recent_lows)) / 2

                if current_price > mid_range:
                    return StructureState.DISTRIBUTION, TrendDirection.NEUTRAL
                else:
                    return StructureState.ACCUMULATION, TrendDirection.NEUTRAL

                return StructureState.RANGE, TrendDirection.NEUTRAL

        return StructureState.UNKNOWN, TrendDirection.NEUTRAL

    def _find_range(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find current range boundaries."""
        if not swing_highs or not swing_lows:
            return None, None

        recent_highs = swing_highs[-5:]
        recent_lows = swing_lows[-5:]

        range_high = max(s.price for s in recent_highs)
        range_low = min(s.price for s in recent_lows)

        return range_high, range_low

    def _calculate_invalidations(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        trend: TrendDirection
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate invalidation levels for directional bias."""
        bullish_inv = None
        bearish_inv = None

        if swing_lows:
            # Bullish invalidation = last significant swing low
            bullish_inv = swing_lows[-1].price

        if swing_highs:
            # Bearish invalidation = last significant swing high
            bearish_inv = swing_highs[-1].price

        return bullish_inv, bearish_inv

    def _detect_bos(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        closes: np.ndarray,
        timestamps: List[datetime]
    ) -> Optional[BreakOfStructure]:
        """Detect most recent break of structure."""
        if len(closes) < 2:
            return None

        current_close = closes[-1]
        prev_close = closes[-2]

        # Check for bullish BOS (break above recent swing high)
        if swing_highs:
            last_high = swing_highs[-1]
            if prev_close <= last_high.price < current_close:
                return BreakOfStructure(
                    timestamp=timestamps[-1],
                    price=float(current_close),
                    direction=TrendDirection.UP,
                    broken_level=last_high.price,
                    swing_point=last_high,
                )

        # Check for bearish BOS (break below recent swing low)
        if swing_lows:
            last_low = swing_lows[-1]
            if prev_close >= last_low.price > current_close:
                return BreakOfStructure(
                    timestamp=timestamps[-1],
                    price=float(current_close),
                    direction=TrendDirection.DOWN,
                    broken_level=last_low.price,
                    swing_point=last_low,
                )

        return None

    def _empty_analysis(
        self,
        timestamp: datetime,
        symbol: str,
        timeframe: str
    ) -> StructureAnalysis:
        """Return empty analysis when data is insufficient."""
        return StructureAnalysis(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            state=StructureState.UNKNOWN,
            trend_direction=TrendDirection.NEUTRAL,
            swing_highs=[],
            swing_lows=[],
            last_swing_high=None,
            last_swing_low=None,
            range_high=None,
            range_low=None,
            bullish_invalidation=None,
            bearish_invalidation=None,
            recent_bos=None,
        )

    def to_dict(self, analysis: StructureAnalysis) -> Dict:
        """Convert analysis to dictionary for storage."""
        return {
            "timestamp": analysis.timestamp.isoformat(),
            "symbol": analysis.symbol,
            "timeframe": analysis.timeframe,
            "state": analysis.state.value,
            "trend_direction": analysis.trend_direction.value,
            "swing_highs": [
                {"time": s.timestamp.isoformat(), "price": s.price, "type": s.swing_type.value}
                for s in analysis.swing_highs
            ],
            "swing_lows": [
                {"time": s.timestamp.isoformat(), "price": s.price, "type": s.swing_type.value}
                for s in analysis.swing_lows
            ],
            "range_high": analysis.range_high,
            "range_low": analysis.range_low,
            "bullish_invalidation": analysis.bullish_invalidation,
            "bearish_invalidation": analysis.bearish_invalidation,
            "recent_bos": {
                "time": analysis.recent_bos.timestamp.isoformat(),
                "direction": analysis.recent_bos.direction.value,
                "broken_level": analysis.recent_bos.broken_level,
            } if analysis.recent_bos else None,
        }
