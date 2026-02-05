"""Liquidity Engine for detecting liquidity zones and sweep events.

Detects:
- Previous day/session highs and lows (PDH/PDL)
- Equal highs/lows clusters
- Sweep and rejection events
- Entry zone recommendations
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class ZoneType(Enum):
    """Types of liquidity zones."""
    PDH = "PDH"  # Previous Day High
    PDL = "PDL"  # Previous Day Low
    PWH = "PWH"  # Previous Week High
    PWL = "PWL"  # Previous Week Low
    ASIA_HIGH = "ASIA_HIGH"
    ASIA_LOW = "ASIA_LOW"
    LONDON_HIGH = "LONDON_HIGH"
    LONDON_LOW = "LONDON_LOW"
    NY_HIGH = "NY_HIGH"
    NY_LOW = "NY_LOW"
    EQUAL_HIGHS = "EQUAL_HIGHS"
    EQUAL_LOWS = "EQUAL_LOWS"
    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"


class SweepType(Enum):
    """Types of liquidity sweeps."""
    BUY_SIDE = "BUY_SIDE"  # Swept highs then reversed down
    SELL_SIDE = "SELL_SIDE"  # Swept lows then reversed up


@dataclass
class LiquidityZone:
    """A liquidity zone with price level and metadata."""
    zone_type: ZoneType
    price: float
    timestamp: datetime
    strength: float  # 0-1, based on touches and recency
    touch_count: int = 0
    swept: bool = False
    sweep_time: Optional[datetime] = None


@dataclass
class SweepEvent:
    """A liquidity sweep event."""
    sweep_type: SweepType
    timestamp: datetime
    zone: LiquidityZone
    sweep_price: float  # Price that went beyond the zone
    close_price: float  # Where price closed after sweep
    wick_size: float  # Size of the rejection wick
    quality_score: float  # 0-1, sweep quality


@dataclass
class LiquidityAnalysis:
    """Complete liquidity analysis output."""
    timestamp: datetime
    symbol: str
    timeframe: str

    # All detected zones
    zones: List[LiquidityZone]

    # Key levels
    buy_side_liquidity: List[LiquidityZone]  # Highs to target
    sell_side_liquidity: List[LiquidityZone]  # Lows to target

    # Recent sweeps
    recent_sweeps: List[SweepEvent]

    # Entry recommendations
    long_entry_zone: Optional[Tuple[float, float]]  # (min, max) price
    short_entry_zone: Optional[Tuple[float, float]]

    # Nearest zones
    nearest_resistance: Optional[LiquidityZone]
    nearest_support: Optional[LiquidityZone]


class LiquidityEngine:
    """Analyzes liquidity zones and sweep events.

    Detects key liquidity levels where stop losses are likely
    clustered and identifies sweep events that indicate
    smart money activity.
    """

    def __init__(
        self,
        equal_threshold_pct: float = 0.001,  # 0.1% tolerance for equal levels
        sweep_rejection_threshold: float = 0.5,  # 50% wick required
    ):
        """Initialize liquidity engine.

        Args:
            equal_threshold_pct: Percentage threshold for equal levels
            sweep_rejection_threshold: Min wick ratio for valid sweep
        """
        self.equal_threshold = equal_threshold_pct
        self.rejection_threshold = sweep_rejection_threshold

        # Session times (UTC)
        self.sessions = {
            "asia": (time(0, 0), time(8, 0)),
            "london": (time(7, 0), time(16, 0)),
            "ny": (time(12, 0), time(21, 0)),
        }

    def analyze(
        self,
        symbol: str,
        timeframe: str,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: List[datetime],
        daily_data: Optional[Dict] = None,  # For PDH/PDL
        weekly_data: Optional[Dict] = None,  # For PWH/PWL
    ) -> LiquidityAnalysis:
        """Analyze liquidity zones and sweeps.

        Args:
            symbol: Instrument symbol
            timeframe: Chart timeframe
            opens, highs, lows, closes: OHLC arrays
            timestamps: Bar timestamps
            daily_data: Optional daily OHLC for PDH/PDL
            weekly_data: Optional weekly OHLC for PWH/PWL

        Returns:
            LiquidityAnalysis with detected zones and sweeps
        """
        analysis_time = datetime.now()

        if len(highs) < 10:
            return self._empty_analysis(analysis_time, symbol, timeframe)

        zones = []

        # Detect time-based liquidity zones
        zones.extend(self._detect_session_levels(highs, lows, timestamps))

        if daily_data:
            zones.extend(self._detect_daily_levels(daily_data))

        if weekly_data:
            zones.extend(self._detect_weekly_levels(weekly_data))

        # Detect equal highs/lows clusters
        zones.extend(self._detect_equal_levels(highs, lows, timestamps))

        # Detect sweeps
        sweeps = self._detect_sweeps(
            zones, opens, highs, lows, closes, timestamps
        )

        # Mark swept zones
        for sweep in sweeps:
            sweep.zone.swept = True
            sweep.zone.sweep_time = sweep.timestamp

        # Categorize zones
        current_price = closes[-1]
        buy_side = [z for z in zones if z.price > current_price and not z.swept]
        sell_side = [z for z in zones if z.price < current_price and not z.swept]

        # Sort by distance
        buy_side.sort(key=lambda z: z.price)
        sell_side.sort(key=lambda z: z.price, reverse=True)

        # Calculate entry zones based on recent sweeps
        long_entry, short_entry = self._calculate_entry_zones(
            sweeps, current_price, zones
        )

        return LiquidityAnalysis(
            timestamp=analysis_time,
            symbol=symbol,
            timeframe=timeframe,
            zones=zones,
            buy_side_liquidity=buy_side[:5],
            sell_side_liquidity=sell_side[:5],
            recent_sweeps=sweeps[-5:],
            long_entry_zone=long_entry,
            short_entry_zone=short_entry,
            nearest_resistance=buy_side[0] if buy_side else None,
            nearest_support=sell_side[0] if sell_side else None,
        )

    def _detect_session_levels(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        timestamps: List[datetime]
    ) -> List[LiquidityZone]:
        """Detect session high/low levels."""
        zones = []

        # Group bars by session
        session_bars = {"asia": [], "london": [], "ny": []}

        for i, ts in enumerate(timestamps):
            bar_time = ts.time()
            for session, (start, end) in self.sessions.items():
                if start <= bar_time <= end:
                    session_bars[session].append(i)

        # Extract session highs/lows from previous sessions
        for session, indices in session_bars.items():
            if len(indices) < 2:
                continue

            # Get bars from previous complete session
            session_highs = highs[indices]
            session_lows = lows[indices]

            if len(session_highs) > 0:
                session_high = float(np.max(session_highs))
                session_low = float(np.min(session_lows))
                session_time = timestamps[indices[-1]]

                zone_type_high = {
                    "asia": ZoneType.ASIA_HIGH,
                    "london": ZoneType.LONDON_HIGH,
                    "ny": ZoneType.NY_HIGH,
                }[session]

                zone_type_low = {
                    "asia": ZoneType.ASIA_LOW,
                    "london": ZoneType.LONDON_LOW,
                    "ny": ZoneType.NY_LOW,
                }[session]

                zones.append(LiquidityZone(
                    zone_type=zone_type_high,
                    price=session_high,
                    timestamp=session_time,
                    strength=0.7,
                ))
                zones.append(LiquidityZone(
                    zone_type=zone_type_low,
                    price=session_low,
                    timestamp=session_time,
                    strength=0.7,
                ))

        return zones

    def _detect_daily_levels(self, daily_data: Dict) -> List[LiquidityZone]:
        """Detect previous day high/low."""
        zones = []

        if "prev_high" in daily_data:
            zones.append(LiquidityZone(
                zone_type=ZoneType.PDH,
                price=daily_data["prev_high"],
                timestamp=daily_data.get("timestamp", datetime.now()),
                strength=0.9,
            ))

        if "prev_low" in daily_data:
            zones.append(LiquidityZone(
                zone_type=ZoneType.PDL,
                price=daily_data["prev_low"],
                timestamp=daily_data.get("timestamp", datetime.now()),
                strength=0.9,
            ))

        return zones

    def _detect_weekly_levels(self, weekly_data: Dict) -> List[LiquidityZone]:
        """Detect previous week high/low."""
        zones = []

        if "prev_high" in weekly_data:
            zones.append(LiquidityZone(
                zone_type=ZoneType.PWH,
                price=weekly_data["prev_high"],
                timestamp=weekly_data.get("timestamp", datetime.now()),
                strength=0.95,
            ))

        if "prev_low" in weekly_data:
            zones.append(LiquidityZone(
                zone_type=ZoneType.PWL,
                price=weekly_data["prev_low"],
                timestamp=weekly_data.get("timestamp", datetime.now()),
                strength=0.95,
            ))

        return zones

    def _detect_equal_levels(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        timestamps: List[datetime]
    ) -> List[LiquidityZone]:
        """Detect equal highs/lows clusters."""
        zones = []

        # Find equal highs
        equal_highs = self._find_equal_levels(highs, timestamps, is_high=True)
        for price, ts, strength in equal_highs:
            zones.append(LiquidityZone(
                zone_type=ZoneType.EQUAL_HIGHS,
                price=price,
                timestamp=ts,
                strength=strength,
            ))

        # Find equal lows
        equal_lows = self._find_equal_levels(lows, timestamps, is_high=False)
        for price, ts, strength in equal_lows:
            zones.append(LiquidityZone(
                zone_type=ZoneType.EQUAL_LOWS,
                price=price,
                timestamp=ts,
                strength=strength,
            ))

        return zones

    def _find_equal_levels(
        self,
        prices: np.ndarray,
        timestamps: List[datetime],
        is_high: bool
    ) -> List[Tuple[float, datetime, float]]:
        """Find clusters of equal price levels."""
        if len(prices) < 3:
            return []

        # Find local extremes
        extremes = []
        for i in range(1, len(prices) - 1):
            if is_high:
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    extremes.append((i, float(prices[i])))
            else:
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    extremes.append((i, float(prices[i])))

        if len(extremes) < 2:
            return []

        # Find clusters within threshold
        clusters = []
        used = set()

        for i, (idx1, price1) in enumerate(extremes):
            if i in used:
                continue

            cluster = [(idx1, price1)]
            for j, (idx2, price2) in enumerate(extremes[i+1:], i+1):
                if j in used:
                    continue
                if abs(price2 - price1) / price1 < self.equal_threshold:
                    cluster.append((idx2, price2))
                    used.add(j)

            if len(cluster) >= 2:
                avg_price = sum(p for _, p in cluster) / len(cluster)
                last_idx = max(idx for idx, _ in cluster)
                # Strength based on number of touches
                strength = min(0.5 + len(cluster) * 0.15, 1.0)
                clusters.append((avg_price, timestamps[last_idx], strength))
                used.add(i)

        return clusters

    def _detect_sweeps(
        self,
        zones: List[LiquidityZone],
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: List[datetime]
    ) -> List[SweepEvent]:
        """Detect liquidity sweep events."""
        sweeps = []

        for i in range(1, len(closes)):
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            bar_open = opens[i]
            bar_range = bar_high - bar_low

            if bar_range == 0:
                continue

            # Check for buy-side sweeps (wick above zone, close back below)
            for zone in zones:
                if zone.zone_type in [ZoneType.EQUAL_HIGHS, ZoneType.PDH, ZoneType.PWH,
                                       ZoneType.ASIA_HIGH, ZoneType.LONDON_HIGH, ZoneType.NY_HIGH]:
                    # Check if we swept the level
                    if bar_high > zone.price and bar_close < zone.price:
                        wick_above = bar_high - max(bar_open, bar_close)
                        wick_ratio = wick_above / bar_range

                        if wick_ratio >= self.rejection_threshold:
                            quality = min(wick_ratio, 1.0) * zone.strength
                            sweeps.append(SweepEvent(
                                sweep_type=SweepType.BUY_SIDE,
                                timestamp=timestamps[i],
                                zone=zone,
                                sweep_price=float(bar_high),
                                close_price=float(bar_close),
                                wick_size=float(wick_above),
                                quality_score=quality,
                            ))

            # Check for sell-side sweeps (wick below zone, close back above)
            for zone in zones:
                if zone.zone_type in [ZoneType.EQUAL_LOWS, ZoneType.PDL, ZoneType.PWL,
                                       ZoneType.ASIA_LOW, ZoneType.LONDON_LOW, ZoneType.NY_LOW]:
                    if bar_low < zone.price and bar_close > zone.price:
                        wick_below = min(bar_open, bar_close) - bar_low
                        wick_ratio = wick_below / bar_range

                        if wick_ratio >= self.rejection_threshold:
                            quality = min(wick_ratio, 1.0) * zone.strength
                            sweeps.append(SweepEvent(
                                sweep_type=SweepType.SELL_SIDE,
                                timestamp=timestamps[i],
                                zone=zone,
                                sweep_price=float(bar_low),
                                close_price=float(bar_close),
                                wick_size=float(wick_below),
                                quality_score=quality,
                            ))

        return sweeps

    def _calculate_entry_zones(
        self,
        sweeps: List[SweepEvent],
        current_price: float,
        zones: List[LiquidityZone]
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Calculate optimal entry zones based on sweeps."""
        long_entry = None
        short_entry = None

        # Look for recent quality sweeps
        recent_sweeps = sweeps[-3:] if sweeps else []

        for sweep in recent_sweeps:
            if sweep.quality_score >= 0.6:
                if sweep.sweep_type == SweepType.SELL_SIDE:
                    # Sell-side sweep suggests long entry
                    entry_low = sweep.zone.price
                    entry_high = sweep.close_price
                    if entry_low < current_price:
                        long_entry = (entry_low, entry_high)

                elif sweep.sweep_type == SweepType.BUY_SIDE:
                    # Buy-side sweep suggests short entry
                    entry_high = sweep.zone.price
                    entry_low = sweep.close_price
                    if entry_high > current_price:
                        short_entry = (entry_low, entry_high)

        return long_entry, short_entry

    def _empty_analysis(
        self,
        timestamp: datetime,
        symbol: str,
        timeframe: str
    ) -> LiquidityAnalysis:
        """Return empty analysis when data is insufficient."""
        return LiquidityAnalysis(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            zones=[],
            buy_side_liquidity=[],
            sell_side_liquidity=[],
            recent_sweeps=[],
            long_entry_zone=None,
            short_entry_zone=None,
            nearest_resistance=None,
            nearest_support=None,
        )

    def to_dict(self, analysis: LiquidityAnalysis) -> Dict:
        """Convert analysis to dictionary for storage."""
        return {
            "timestamp": analysis.timestamp.isoformat(),
            "symbol": analysis.symbol,
            "timeframe": analysis.timeframe,
            "zones": [
                {
                    "type": z.zone_type.value,
                    "price": z.price,
                    "strength": z.strength,
                    "swept": z.swept,
                }
                for z in analysis.zones
            ],
            "buy_side": [z.price for z in analysis.buy_side_liquidity],
            "sell_side": [z.price for z in analysis.sell_side_liquidity],
            "recent_sweeps": [
                {
                    "type": s.sweep_type.value,
                    "time": s.timestamp.isoformat(),
                    "zone_price": s.zone.price,
                    "quality": s.quality_score,
                }
                for s in analysis.recent_sweeps
            ],
            "long_entry_zone": analysis.long_entry_zone,
            "short_entry_zone": analysis.short_entry_zone,
        }
