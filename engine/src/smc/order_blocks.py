"""
Order Block Detection - Smart Money Concepts

Order blocks are areas where institutional orders were placed.
- Bullish OB: Last bearish candle before a bullish impulse move
- Bearish OB: Last bullish candle before a bearish impulse move
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class OrderBlock:
    """Represents an Order Block zone."""
    ob_type: str  # "bullish" or "bearish"
    high: float
    low: float
    open_price: float
    close_price: float
    index: int
    timestamp: pd.Timestamp
    impulse_strength: float  # Strength of the impulse move that followed
    is_mitigated: bool = False  # Has price returned and touched the OB?
    mitigation_index: Optional[int] = None

    @property
    def midpoint(self) -> float:
        """Get the midpoint of the order block."""
        return (self.high + self.low) / 2

    @property
    def zone(self) -> Tuple[float, float]:
        """Get the OB zone as (lower, upper)."""
        return (self.low, self.high)


class OrderBlockDetector:
    """
    Detects Order Blocks using Smart Money Concepts methodology.

    Order blocks are identified as the last opposing candle before
    a significant impulse move.
    """

    def __init__(
        self,
        impulse_atr_multiplier: float = 1.5,
        min_impulse_candles: int = 2,
        max_ob_age: int = 100
    ):
        """
        Initialize Order Block detector.

        Args:
            impulse_atr_multiplier: Minimum impulse size as multiple of ATR
            min_impulse_candles: Minimum candles in impulse move
            max_ob_age: Maximum age (in candles) for valid OB
        """
        self.impulse_atr_multiplier = impulse_atr_multiplier
        self.min_impulse_candles = min_impulse_candles
        self.max_ob_age = max_ob_age
        self.order_blocks: List[OrderBlock] = []

    def detect(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detect order blocks in price data.

        Args:
            df: DataFrame with OHLC data

        Returns:
            List of detected OrderBlock objects
        """
        self.order_blocks = []

        if len(df) < 20:
            return self.order_blocks

        # Calculate ATR for impulse threshold
        atr = self._calculate_atr(df, period=14)

        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        timestamps = df['timestamp'] if 'timestamp' in df.columns else df.index

        n = len(df)

        for i in range(2, n - self.min_impulse_candles):
            current_atr = atr[i] if i < len(atr) else atr[-1]
            impulse_threshold = current_atr * self.impulse_atr_multiplier

            # Check for bullish impulse (look for bearish OB before it)
            # A bullish impulse is a strong move up
            impulse_up = self._detect_impulse_up(
                highs, lows, closes, i,
                impulse_threshold, self.min_impulse_candles
            )

            if impulse_up:
                # Look for the last bearish candle before the impulse
                ob_candle = self._find_bearish_candle_before(opens, closes, i)
                if ob_candle is not None:
                    self.order_blocks.append(OrderBlock(
                        ob_type="bullish",
                        high=highs[ob_candle],
                        low=lows[ob_candle],
                        open_price=opens[ob_candle],
                        close_price=closes[ob_candle],
                        index=ob_candle,
                        timestamp=timestamps.iloc[ob_candle] if hasattr(timestamps, 'iloc') else timestamps[ob_candle],
                        impulse_strength=impulse_up
                    ))

            # Check for bearish impulse (look for bullish OB before it)
            impulse_down = self._detect_impulse_down(
                highs, lows, closes, i,
                impulse_threshold, self.min_impulse_candles
            )

            if impulse_down:
                # Look for the last bullish candle before the impulse
                ob_candle = self._find_bullish_candle_before(opens, closes, i)
                if ob_candle is not None:
                    self.order_blocks.append(OrderBlock(
                        ob_type="bearish",
                        high=highs[ob_candle],
                        low=lows[ob_candle],
                        open_price=opens[ob_candle],
                        close_price=closes[ob_candle],
                        index=ob_candle,
                        timestamp=timestamps.iloc[ob_candle] if hasattr(timestamps, 'iloc') else timestamps[ob_candle],
                        impulse_strength=impulse_down
                    ))

        # Check for mitigation (price returned to OB)
        self._check_mitigation(df)

        # Filter out old OBs
        current_index = len(df) - 1
        self.order_blocks = [
            ob for ob in self.order_blocks
            if (current_index - ob.index) <= self.max_ob_age
        ]

        # Remove duplicates (keep strongest)
        self.order_blocks = self._remove_duplicates()

        return self.order_blocks

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.zeros(len(df))
        tr[0] = high[0] - low[0]

        for i in range(1, len(df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )

        atr = np.zeros(len(df))
        atr[:period] = np.mean(tr[:period])

        for i in range(period, len(df)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return atr

    def _detect_impulse_up(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        start_idx: int,
        threshold: float,
        min_candles: int
    ) -> Optional[float]:
        """Detect bullish impulse move."""
        if start_idx + min_candles >= len(closes):
            return None

        start_price = lows[start_idx]
        max_high = start_price

        bullish_candles = 0
        for j in range(start_idx, min(start_idx + 5, len(closes))):
            if closes[j] > closes[j-1] if j > 0 else True:
                bullish_candles += 1
            max_high = max(max_high, highs[j])

        move_size = max_high - start_price

        if move_size >= threshold and bullish_candles >= min_candles:
            return move_size

        return None

    def _detect_impulse_down(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        start_idx: int,
        threshold: float,
        min_candles: int
    ) -> Optional[float]:
        """Detect bearish impulse move."""
        if start_idx + min_candles >= len(closes):
            return None

        start_price = highs[start_idx]
        min_low = start_price

        bearish_candles = 0
        for j in range(start_idx, min(start_idx + 5, len(closes))):
            if closes[j] < closes[j-1] if j > 0 else True:
                bearish_candles += 1
            min_low = min(min_low, lows[j])

        move_size = start_price - min_low

        if move_size >= threshold and bearish_candles >= min_candles:
            return move_size

        return None

    def _find_bearish_candle_before(
        self,
        opens: np.ndarray,
        closes: np.ndarray,
        impulse_start: int,
        lookback: int = 5
    ) -> Optional[int]:
        """Find the last bearish candle before an impulse."""
        for i in range(impulse_start - 1, max(0, impulse_start - lookback - 1), -1):
            if closes[i] < opens[i]:  # Bearish candle
                return i
        return None

    def _find_bullish_candle_before(
        self,
        opens: np.ndarray,
        closes: np.ndarray,
        impulse_start: int,
        lookback: int = 5
    ) -> Optional[int]:
        """Find the last bullish candle before an impulse."""
        for i in range(impulse_start - 1, max(0, impulse_start - lookback - 1), -1):
            if closes[i] > opens[i]:  # Bullish candle
                return i
        return None

    def _check_mitigation(self, df: pd.DataFrame):
        """Check if order blocks have been mitigated (price returned to zone)."""
        lows = df['low'].values
        highs = df['high'].values

        for ob in self.order_blocks:
            if ob.is_mitigated:
                continue

            # Check candles after the OB
            for i in range(ob.index + 1, len(df)):
                if ob.ob_type == "bullish":
                    # Bullish OB is mitigated when price returns to the zone
                    if lows[i] <= ob.high:  # Price touched the OB
                        ob.is_mitigated = True
                        ob.mitigation_index = i
                        break
                else:  # bearish
                    # Bearish OB is mitigated when price returns to the zone
                    if highs[i] >= ob.low:  # Price touched the OB
                        ob.is_mitigated = True
                        ob.mitigation_index = i
                        break

    def _remove_duplicates(self) -> List[OrderBlock]:
        """Remove duplicate OBs, keeping the strongest."""
        if not self.order_blocks:
            return []

        # Sort by index
        sorted_obs = sorted(self.order_blocks, key=lambda x: x.index)

        unique_obs = []
        for ob in sorted_obs:
            # Check if there's already a similar OB
            is_duplicate = False
            for existing in unique_obs:
                if existing.ob_type == ob.ob_type:
                    # Check if zones overlap significantly
                    overlap = min(existing.high, ob.high) - max(existing.low, ob.low)
                    min_size = min(existing.high - existing.low, ob.high - ob.low)
                    if overlap > min_size * 0.5:
                        # Keep the one with stronger impulse
                        if ob.impulse_strength > existing.impulse_strength:
                            unique_obs.remove(existing)
                            unique_obs.append(ob)
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_obs.append(ob)

        return unique_obs

    def get_active_obs(self, current_price: float) -> dict:
        """
        Get active (unmitigated) order blocks near current price.

        Returns:
            Dictionary with nearest bullish and bearish OBs
        """
        active_bullish = [
            ob for ob in self.order_blocks
            if ob.ob_type == "bullish" and not ob.is_mitigated
        ]
        active_bearish = [
            ob for ob in self.order_blocks
            if ob.ob_type == "bearish" and not ob.is_mitigated
        ]

        # Find nearest OBs
        nearest_bullish = None
        nearest_bearish = None

        # Nearest bullish OB below price (potential buy zone)
        below_price = [ob for ob in active_bullish if ob.high < current_price]
        if below_price:
            nearest_bullish = max(below_price, key=lambda x: x.high)

        # Nearest bearish OB above price (potential sell zone)
        above_price = [ob for ob in active_bearish if ob.low > current_price]
        if above_price:
            nearest_bearish = min(above_price, key=lambda x: x.low)

        return {
            "nearest_bullish_ob": nearest_bullish,
            "nearest_bearish_ob": nearest_bearish,
            "all_bullish_obs": active_bullish,
            "all_bearish_obs": active_bearish,
        }
