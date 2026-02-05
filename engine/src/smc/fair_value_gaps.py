"""
Fair Value Gap (FVG) Detection - Smart Money Concepts

FVGs are imbalances in price where the market moved so quickly
that a gap was left between candle wicks.

- Bullish FVG: Gap between candle 1's high and candle 3's low (in upward move)
- Bearish FVG: Gap between candle 1's low and candle 3's high (in downward move)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap (imbalance)."""
    fvg_type: str  # "bullish" or "bearish"
    high: float  # Upper boundary of the gap
    low: float   # Lower boundary of the gap
    index: int   # Index of the middle candle
    timestamp: pd.Timestamp
    size: float  # Size of the gap
    is_filled: bool = False  # Has price filled the gap?
    fill_percentage: float = 0.0  # How much of the gap is filled

    @property
    def midpoint(self) -> float:
        """Get the midpoint of the FVG (optimal trade entry)."""
        return (self.high + self.low) / 2

    @property
    def zone(self) -> Tuple[float, float]:
        """Get the FVG zone as (lower, upper)."""
        return (self.low, self.high)


class FVGDetector:
    """
    Detects Fair Value Gaps (imbalances) in price data.

    FVGs represent areas where price moved inefficiently and
    may return to "fill" the gap.
    """

    def __init__(
        self,
        min_gap_atr_ratio: float = 0.2,
        max_fvg_age: int = 50
    ):
        """
        Initialize FVG detector.

        Args:
            min_gap_atr_ratio: Minimum gap size as ratio of ATR
            max_fvg_age: Maximum age (in candles) for valid FVG
        """
        self.min_gap_atr_ratio = min_gap_atr_ratio
        self.max_fvg_age = max_fvg_age
        self.fvgs: List[FairValueGap] = []

    def detect(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps in price data.

        Args:
            df: DataFrame with OHLC data

        Returns:
            List of detected FairValueGap objects
        """
        self.fvgs = []

        if len(df) < 5:
            return self.fvgs

        # Calculate ATR for minimum gap size
        atr = self._calculate_atr(df, period=14)

        highs = df['high'].values
        lows = df['low'].values
        timestamps = df['timestamp'] if 'timestamp' in df.columns else df.index

        n = len(df)

        # Need at least 3 candles to form an FVG
        for i in range(1, n - 1):
            current_atr = atr[i] if i < len(atr) else atr[-1]
            min_gap_size = current_atr * self.min_gap_atr_ratio

            # Check for Bullish FVG
            # Candle 1's high < Candle 3's low (gap up)
            gap_low = highs[i - 1]  # Top of candle 1
            gap_high = lows[i + 1]  # Bottom of candle 3

            if gap_high > gap_low:  # There is a gap
                gap_size = gap_high - gap_low
                if gap_size >= min_gap_size:
                    self.fvgs.append(FairValueGap(
                        fvg_type="bullish",
                        high=gap_high,
                        low=gap_low,
                        index=i,
                        timestamp=timestamps.iloc[i] if hasattr(timestamps, 'iloc') else timestamps[i],
                        size=gap_size
                    ))

            # Check for Bearish FVG
            # Candle 1's low > Candle 3's high (gap down)
            gap_high = lows[i - 1]   # Bottom of candle 1
            gap_low = highs[i + 1]   # Top of candle 3

            if gap_high > gap_low:  # There is a gap
                gap_size = gap_high - gap_low
                if gap_size >= min_gap_size:
                    self.fvgs.append(FairValueGap(
                        fvg_type="bearish",
                        high=gap_high,
                        low=gap_low,
                        index=i,
                        timestamp=timestamps.iloc[i] if hasattr(timestamps, 'iloc') else timestamps[i],
                        size=gap_size
                    ))

        # Check which FVGs have been filled
        self._check_fills(df)

        # Filter out old FVGs
        current_index = len(df) - 1
        self.fvgs = [
            fvg for fvg in self.fvgs
            if (current_index - fvg.index) <= self.max_fvg_age
        ]

        return self.fvgs

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
        if len(tr) >= period:
            atr[:period] = np.mean(tr[:period])
            for i in range(period, len(df)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        else:
            atr[:] = np.mean(tr)

        return atr

    def _check_fills(self, df: pd.DataFrame):
        """Check which FVGs have been filled by subsequent price action."""
        highs = df['high'].values
        lows = df['low'].values

        for fvg in self.fvgs:
            if fvg.is_filled:
                continue

            # Check candles after the FVG
            for i in range(fvg.index + 2, len(df)):  # +2 because FVG forms on candle 2 of 3
                if fvg.fvg_type == "bullish":
                    # Bullish FVG is filled when price drops into it
                    if lows[i] <= fvg.high:
                        # Calculate fill percentage
                        penetration = fvg.high - lows[i]
                        fvg.fill_percentage = min(penetration / fvg.size, 1.0) * 100
                        if lows[i] <= fvg.low:
                            fvg.is_filled = True
                        break
                else:  # bearish
                    # Bearish FVG is filled when price rises into it
                    if highs[i] >= fvg.low:
                        penetration = highs[i] - fvg.low
                        fvg.fill_percentage = min(penetration / fvg.size, 1.0) * 100
                        if highs[i] >= fvg.high:
                            fvg.is_filled = True
                        break

    def get_unfilled_fvgs(self, current_price: float) -> dict:
        """
        Get unfilled FVGs relative to current price.

        Returns:
            Dictionary with FVGs above and below current price
        """
        unfilled = [fvg for fvg in self.fvgs if not fvg.is_filled]

        # FVGs below price (potential support for longs)
        bullish_fvgs = [
            fvg for fvg in unfilled
            if fvg.fvg_type == "bullish" and fvg.high < current_price
        ]

        # FVGs above price (potential resistance for shorts)
        bearish_fvgs = [
            fvg for fvg in unfilled
            if fvg.fvg_type == "bearish" and fvg.low > current_price
        ]

        # Nearest FVGs
        nearest_bullish = max(bullish_fvgs, key=lambda x: x.high) if bullish_fvgs else None
        nearest_bearish = min(bearish_fvgs, key=lambda x: x.low) if bearish_fvgs else None

        return {
            "nearest_bullish_fvg": nearest_bullish,
            "nearest_bearish_fvg": nearest_bearish,
            "all_bullish_fvgs": sorted(bullish_fvgs, key=lambda x: x.high, reverse=True),
            "all_bearish_fvgs": sorted(bearish_fvgs, key=lambda x: x.low),
        }

    def get_fvg_at_price(self, price: float, tolerance: float = 0.0001) -> Optional[FairValueGap]:
        """
        Check if price is currently within any FVG.

        Args:
            price: Current price to check
            tolerance: Price tolerance for matching

        Returns:
            FVG if price is within one, None otherwise
        """
        for fvg in self.fvgs:
            if not fvg.is_filled:
                if fvg.low - tolerance <= price <= fvg.high + tolerance:
                    return fvg
        return None
