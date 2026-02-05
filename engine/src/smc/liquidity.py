"""
Liquidity Zone Detection - Smart Money Concepts

Liquidity zones are areas where stop losses cluster:
- Equal Highs (EQH): Retail stops above (sell-side liquidity)
- Equal Lows (EQL): Retail stops below (buy-side liquidity)
- Previous highs/lows: Obvious stop placement areas
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class LiquidityZone:
    """Represents a liquidity zone (stop hunt target)."""
    zone_type: str  # "buy_side" (above price) or "sell_side" (below price)
    price: float
    strength: int  # Number of touches/equal levels
    indices: List[int]  # Candle indices that formed this level
    is_swept: bool = False  # Has liquidity been taken?
    sweep_index: Optional[int] = None

    @property
    def description(self) -> str:
        """Human-readable description of the zone."""
        if self.zone_type == "buy_side":
            return f"Buy-side liquidity at {self.price} (EQH x{self.strength})"
        else:
            return f"Sell-side liquidity at {self.price} (EQL x{self.strength})"


class LiquidityZoneDetector:
    """
    Detects liquidity zones where stop orders likely cluster.

    Institutional traders often hunt these zones before making
    their real move in the opposite direction.
    """

    def __init__(
        self,
        price_tolerance: float = 0.0002,  # 2 pips for forex
        min_touches: int = 2,
        lookback: int = 50
    ):
        """
        Initialize Liquidity Zone detector.

        Args:
            price_tolerance: Tolerance for considering levels "equal"
            min_touches: Minimum touches to form a valid zone
            lookback: Number of candles to look back
        """
        self.price_tolerance = price_tolerance
        self.min_touches = min_touches
        self.lookback = lookback
        self.liquidity_zones: List[LiquidityZone] = []

    def detect(self, df: pd.DataFrame) -> List[LiquidityZone]:
        """
        Detect liquidity zones in price data.

        Args:
            df: DataFrame with OHLC data

        Returns:
            List of detected LiquidityZone objects
        """
        self.liquidity_zones = []

        if len(df) < 10:
            return self.liquidity_zones

        highs = df['high'].values
        lows = df['low'].values

        # Use recent data based on lookback
        start_idx = max(0, len(df) - self.lookback)

        # Detect Equal Highs (buy-side liquidity)
        eq_highs = self._find_equal_levels(highs[start_idx:], is_high=True)
        for price, indices in eq_highs.items():
            if len(indices) >= self.min_touches:
                self.liquidity_zones.append(LiquidityZone(
                    zone_type="buy_side",
                    price=price,
                    strength=len(indices),
                    indices=[i + start_idx for i in indices]
                ))

        # Detect Equal Lows (sell-side liquidity)
        eq_lows = self._find_equal_levels(lows[start_idx:], is_high=False)
        for price, indices in eq_lows.items():
            if len(indices) >= self.min_touches:
                self.liquidity_zones.append(LiquidityZone(
                    zone_type="sell_side",
                    price=price,
                    strength=len(indices),
                    indices=[i + start_idx for i in indices]
                ))

        # Check for sweeps
        self._check_sweeps(df)

        return self.liquidity_zones

    def _find_equal_levels(
        self,
        prices: np.ndarray,
        is_high: bool
    ) -> dict:
        """
        Find clusters of equal price levels.

        Args:
            prices: Array of highs or lows
            is_high: True if finding equal highs, False for equal lows

        Returns:
            Dictionary mapping price level to list of indices
        """
        levels = {}

        for i, price in enumerate(prices):
            # Find if this price matches any existing level
            matched = False
            for level_price in list(levels.keys()):
                if abs(price - level_price) <= self.price_tolerance * price:
                    # Add to existing level (use average)
                    levels[level_price].append(i)
                    matched = True
                    break

            if not matched:
                levels[price] = [i]

        return levels

    def _check_sweeps(self, df: pd.DataFrame):
        """Check if liquidity zones have been swept."""
        highs = df['high'].values
        lows = df['low'].values

        for zone in self.liquidity_zones:
            if zone.is_swept:
                continue

            last_touch_idx = max(zone.indices)

            # Check candles after the zone formed
            for i in range(last_touch_idx + 1, len(df)):
                if zone.zone_type == "buy_side":
                    # Buy-side liquidity is swept when price goes above
                    if highs[i] > zone.price * (1 + self.price_tolerance):
                        zone.is_swept = True
                        zone.sweep_index = i
                        break
                else:
                    # Sell-side liquidity is swept when price goes below
                    if lows[i] < zone.price * (1 - self.price_tolerance):
                        zone.is_swept = True
                        zone.sweep_index = i
                        break

    def get_active_zones(self, current_price: float) -> dict:
        """
        Get active (unswept) liquidity zones.

        Args:
            current_price: Current market price

        Returns:
            Dictionary with zones above and below current price
        """
        active = [z for z in self.liquidity_zones if not z.is_swept]

        # Buy-side liquidity (above price - targets for shorts to take)
        buy_side = [z for z in active if z.zone_type == "buy_side" and z.price > current_price]

        # Sell-side liquidity (below price - targets for longs to take)
        sell_side = [z for z in active if z.zone_type == "sell_side" and z.price < current_price]

        # Sort by distance from current price
        buy_side.sort(key=lambda z: z.price)
        sell_side.sort(key=lambda z: z.price, reverse=True)

        return {
            "buy_side_liquidity": buy_side,
            "sell_side_liquidity": sell_side,
            "nearest_above": buy_side[0] if buy_side else None,
            "nearest_below": sell_side[0] if sell_side else None,
        }

    def get_sweep_targets(self, current_price: float, direction: str) -> List[LiquidityZone]:
        """
        Get potential sweep targets for a trade direction.

        Args:
            current_price: Current market price
            direction: "long" or "short"

        Returns:
            List of liquidity zones that could be swept
        """
        active_zones = self.get_active_zones(current_price)

        if direction == "long":
            # Longs would target buy-side liquidity (equal highs above)
            return active_zones["buy_side_liquidity"][:3]  # Top 3
        else:
            # Shorts would target sell-side liquidity (equal lows below)
            return active_zones["sell_side_liquidity"][:3]  # Top 3
