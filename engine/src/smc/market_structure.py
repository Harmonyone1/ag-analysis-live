"""
Market Structure Analysis - Smart Money Concepts

Detects:
- Swing Highs and Swing Lows
- Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), Lower Lows (LL)
- Break of Structure (BOS) - continuation
- Change of Character (ChoCH) - reversal
- Premium and Discount zones
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


class StructureType(Enum):
    """Type of structure point."""
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL"
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"


class TrendBias(Enum):
    """Market trend bias."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class StructurePoint:
    """Represents a swing high or low in market structure."""
    index: int
    price: float
    timestamp: pd.Timestamp
    structure_type: StructureType
    is_broken: bool = False
    broken_at_index: Optional[int] = None


@dataclass
class StructureShift:
    """Represents a Break of Structure (BOS) or Change of Character (ChoCH)."""
    shift_type: str  # "BOS" or "ChoCH"
    direction: str  # "bullish" or "bearish"
    broken_point: StructurePoint
    break_index: int
    break_price: float
    timestamp: pd.Timestamp


class MarketStructure:
    """
    Analyzes market structure using Smart Money Concepts.

    Identifies swing points, trend structure, and structure shifts.
    """

    def __init__(self, swing_lookback: int = 5):
        """
        Initialize Market Structure analyzer.

        Args:
            swing_lookback: Number of candles on each side to confirm swing point
        """
        self.swing_lookback = swing_lookback
        self.swing_highs: List[StructurePoint] = []
        self.swing_lows: List[StructurePoint] = []
        self.structure_shifts: List[StructureShift] = []
        self.current_bias: TrendBias = TrendBias.NEUTRAL

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Perform full market structure analysis.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'timestamp']

        Returns:
            Dictionary with structure analysis results
        """
        # Reset state
        self.swing_highs = []
        self.swing_lows = []
        self.structure_shifts = []

        # Find swing points
        self._find_swing_points(df)

        # Classify structure (HH, HL, LH, LL)
        self._classify_structure()

        # Detect structure shifts (BOS, ChoCH)
        self._detect_structure_shifts(df)

        # Determine current bias
        self._determine_bias()

        # Calculate premium/discount zones
        premium_discount = self._calculate_premium_discount(df)

        return {
            "bias": self.current_bias.value,
            "swing_highs": self.swing_highs,
            "swing_lows": self.swing_lows,
            "structure_shifts": self.structure_shifts,
            "last_hh": self._get_last_of_type(StructureType.HIGHER_HIGH),
            "last_hl": self._get_last_of_type(StructureType.HIGHER_LOW),
            "last_lh": self._get_last_of_type(StructureType.LOWER_HIGH),
            "last_ll": self._get_last_of_type(StructureType.LOWER_LOW),
            "premium_zone": premium_discount["premium"],
            "discount_zone": premium_discount["discount"],
            "equilibrium": premium_discount["equilibrium"],
            "current_price_zone": premium_discount["current_zone"],
        }

    def _find_swing_points(self, df: pd.DataFrame):
        """Find swing highs and lows using lookback method."""
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df['timestamp'] if 'timestamp' in df.columns else df.index

        n = len(df)
        lookback = self.swing_lookback

        for i in range(lookback, n - lookback):
            # Check for swing high
            is_swing_high = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break

            if is_swing_high:
                self.swing_highs.append(StructurePoint(
                    index=i,
                    price=highs[i],
                    timestamp=timestamps.iloc[i] if hasattr(timestamps, 'iloc') else timestamps[i],
                    structure_type=StructureType.SWING_HIGH
                ))

            # Check for swing low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                self.swing_lows.append(StructurePoint(
                    index=i,
                    price=lows[i],
                    timestamp=timestamps.iloc[i] if hasattr(timestamps, 'iloc') else timestamps[i],
                    structure_type=StructureType.SWING_LOW
                ))

    def _classify_structure(self):
        """Classify swing points as HH, HL, LH, LL."""
        # Classify swing highs
        for i, sh in enumerate(self.swing_highs):
            if i > 0:
                prev_sh = self.swing_highs[i - 1]
                if sh.price > prev_sh.price:
                    sh.structure_type = StructureType.HIGHER_HIGH
                else:
                    sh.structure_type = StructureType.LOWER_HIGH

        # Classify swing lows
        for i, sl in enumerate(self.swing_lows):
            if i > 0:
                prev_sl = self.swing_lows[i - 1]
                if sl.price > prev_sl.price:
                    sl.structure_type = StructureType.HIGHER_LOW
                else:
                    sl.structure_type = StructureType.LOWER_LOW

    def _detect_structure_shifts(self, df: pd.DataFrame):
        """
        Detect Break of Structure (BOS) and Change of Character (ChoCH).

        BOS: Break in the direction of trend (continuation)
        ChoCH: Break against the trend (reversal signal)
        """
        closes = df['close'].values
        timestamps = df['timestamp'] if 'timestamp' in df.columns else df.index

        # Check for breaks of swing highs
        for sh in self.swing_highs:
            if sh.is_broken:
                continue

            # Look for close above this swing high
            for i in range(sh.index + 1, len(df)):
                if closes[i] > sh.price:
                    sh.is_broken = True
                    sh.broken_at_index = i

                    # Determine if BOS or ChoCH
                    # If we were bearish (making LH) and break a high = ChoCH bullish
                    # If we were bullish (making HH) and break a high = BOS bullish
                    if sh.structure_type == StructureType.LOWER_HIGH:
                        shift_type = "ChoCH"
                        direction = "bullish"
                    else:
                        shift_type = "BOS"
                        direction = "bullish"

                    self.structure_shifts.append(StructureShift(
                        shift_type=shift_type,
                        direction=direction,
                        broken_point=sh,
                        break_index=i,
                        break_price=closes[i],
                        timestamp=timestamps.iloc[i] if hasattr(timestamps, 'iloc') else timestamps[i]
                    ))
                    break

        # Check for breaks of swing lows
        for sl in self.swing_lows:
            if sl.is_broken:
                continue

            # Look for close below this swing low
            for i in range(sl.index + 1, len(df)):
                if closes[i] < sl.price:
                    sl.is_broken = True
                    sl.broken_at_index = i

                    # If we were bullish (making HL) and break a low = ChoCH bearish
                    # If we were bearish (making LL) and break a low = BOS bearish
                    if sl.structure_type == StructureType.HIGHER_LOW:
                        shift_type = "ChoCH"
                        direction = "bearish"
                    else:
                        shift_type = "BOS"
                        direction = "bearish"

                    self.structure_shifts.append(StructureShift(
                        shift_type=shift_type,
                        direction=direction,
                        broken_point=sl,
                        break_index=i,
                        break_price=closes[i],
                        timestamp=timestamps.iloc[i] if hasattr(timestamps, 'iloc') else timestamps[i]
                    ))
                    break

    def _determine_bias(self):
        """Determine overall market bias from structure."""
        if not self.swing_highs or not self.swing_lows:
            self.current_bias = TrendBias.NEUTRAL
            return

        # Get last 3 swing highs and lows
        recent_highs = self.swing_highs[-3:] if len(self.swing_highs) >= 3 else self.swing_highs
        recent_lows = self.swing_lows[-3:] if len(self.swing_lows) >= 3 else self.swing_lows

        # Count HH/HL vs LH/LL
        hh_count = sum(1 for sh in recent_highs if sh.structure_type == StructureType.HIGHER_HIGH)
        hl_count = sum(1 for sl in recent_lows if sl.structure_type == StructureType.HIGHER_LOW)
        lh_count = sum(1 for sh in recent_highs if sh.structure_type == StructureType.LOWER_HIGH)
        ll_count = sum(1 for sl in recent_lows if sl.structure_type == StructureType.LOWER_LOW)

        bullish_score = hh_count + hl_count
        bearish_score = lh_count + ll_count

        # Also check most recent structure shift
        if self.structure_shifts:
            last_shift = self.structure_shifts[-1]
            if last_shift.shift_type == "ChoCH":
                # ChoCH is a strong signal
                if last_shift.direction == "bullish":
                    bullish_score += 2
                else:
                    bearish_score += 2

        if bullish_score > bearish_score:
            self.current_bias = TrendBias.BULLISH
        elif bearish_score > bullish_score:
            self.current_bias = TrendBias.BEARISH
        else:
            self.current_bias = TrendBias.NEUTRAL

    def _calculate_premium_discount(self, df: pd.DataFrame) -> dict:
        """
        Calculate premium and discount zones.

        Premium: Above 50% of range (sell zone)
        Discount: Below 50% of range (buy zone)
        """
        if not self.swing_highs or not self.swing_lows:
            return {
                "premium": None,
                "discount": None,
                "equilibrium": None,
                "current_zone": "neutral"
            }

        # Use most recent significant swing high and low
        recent_high = max(self.swing_highs[-3:], key=lambda x: x.price) if self.swing_highs else None
        recent_low = min(self.swing_lows[-3:], key=lambda x: x.price) if self.swing_lows else None

        if not recent_high or not recent_low:
            return {
                "premium": None,
                "discount": None,
                "equilibrium": None,
                "current_zone": "neutral"
            }

        range_high = recent_high.price
        range_low = recent_low.price
        range_size = range_high - range_low

        equilibrium = range_low + (range_size * 0.5)
        premium_start = range_low + (range_size * 0.5)  # 50%+
        discount_end = range_low + (range_size * 0.5)   # 50%-

        current_price = df['close'].iloc[-1]

        if current_price > equilibrium:
            current_zone = "premium"
        elif current_price < equilibrium:
            current_zone = "discount"
        else:
            current_zone = "equilibrium"

        return {
            "premium": (premium_start, range_high),
            "discount": (range_low, discount_end),
            "equilibrium": equilibrium,
            "current_zone": current_zone,
            "range_high": range_high,
            "range_low": range_low,
        }

    def _get_last_of_type(self, structure_type: StructureType) -> Optional[StructurePoint]:
        """Get the most recent structure point of a given type."""
        all_points = self.swing_highs + self.swing_lows
        matching = [p for p in all_points if p.structure_type == structure_type]
        return matching[-1] if matching else None

    def get_key_levels(self) -> List[Tuple[float, str]]:
        """Get key price levels from structure."""
        levels = []

        # Add unbroken swing highs as resistance
        for sh in self.swing_highs:
            if not sh.is_broken:
                levels.append((sh.price, f"resistance_{sh.structure_type.value}"))

        # Add unbroken swing lows as support
        for sl in self.swing_lows:
            if not sl.is_broken:
                levels.append((sl.price, f"support_{sl.structure_type.value}"))

        return sorted(levels, key=lambda x: x[0], reverse=True)
