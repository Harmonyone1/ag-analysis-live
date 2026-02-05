"""
Multi-Timeframe Analysis - Smart Money Concepts

Combines analysis from multiple timeframes:
- HTF (Higher Timeframe): H4/Daily - Determines bias and key levels
- MTF (Medium Timeframe): H1 - Structure and order blocks
- LTF (Lower Timeframe): M15/M5 - Entry timing
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .market_structure import MarketStructure, TrendBias
from .order_blocks import OrderBlockDetector, OrderBlock
from .fair_value_gaps import FVGDetector, FairValueGap
from .liquidity import LiquidityZoneDetector, LiquidityZone


@dataclass
class TimeframeAnalysis:
    """Analysis results for a single timeframe."""
    timeframe: str
    bias: str
    structure: dict
    order_blocks: List[OrderBlock]
    fvgs: List[FairValueGap]
    liquidity_zones: List[LiquidityZone]
    key_levels: List[Tuple[float, str]]


@dataclass
class TradeSetup:
    """A potential trade setup identified by MTF analysis."""
    direction: str  # "long" or "short"
    entry_zone: Tuple[float, float]  # (low, high)
    stop_loss: float
    take_profit: float
    risk_reward: float
    confidence: float  # 0-100
    htf_bias: str
    entry_reason: str
    confluences: List[str]


class MultiTimeframeAnalyzer:
    """
    Analyzes multiple timeframes to find high-probability trade setups.

    Uses top-down analysis:
    1. HTF determines overall bias
    2. MTF identifies structure and POIs
    3. LTF times the entry
    """

    def __init__(self):
        """Initialize the multi-timeframe analyzer."""
        self.structure_analyzer = MarketStructure(swing_lookback=5)
        self.ob_detector = OrderBlockDetector()
        self.fvg_detector = FVGDetector()
        self.liquidity_detector = LiquidityZoneDetector()

        self.htf_analysis: Optional[TimeframeAnalysis] = None
        self.mtf_analysis: Optional[TimeframeAnalysis] = None
        self.ltf_analysis: Optional[TimeframeAnalysis] = None

    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> TimeframeAnalysis:
        """
        Analyze a single timeframe.

        Args:
            df: OHLC DataFrame for the timeframe
            timeframe: Timeframe string (e.g., "H4", "H1", "M15")

        Returns:
            TimeframeAnalysis object
        """
        # Market structure
        structure = self.structure_analyzer.analyze(df)

        # Order blocks
        order_blocks = self.ob_detector.detect(df)

        # Fair value gaps
        fvgs = self.fvg_detector.detect(df)

        # Liquidity zones
        liquidity_zones = self.liquidity_detector.detect(df)

        # Key levels
        key_levels = self.structure_analyzer.get_key_levels()

        return TimeframeAnalysis(
            timeframe=timeframe,
            bias=structure["bias"],
            structure=structure,
            order_blocks=order_blocks,
            fvgs=fvgs,
            liquidity_zones=liquidity_zones,
            key_levels=key_levels
        )

    def full_analysis(
        self,
        htf_df: pd.DataFrame,
        mtf_df: pd.DataFrame,
        ltf_df: pd.DataFrame,
        htf_name: str = "H4",
        mtf_name: str = "H1",
        ltf_name: str = "M15"
    ) -> dict:
        """
        Perform full multi-timeframe analysis.

        Args:
            htf_df: Higher timeframe data (H4/Daily)
            mtf_df: Medium timeframe data (H1)
            ltf_df: Lower timeframe data (M15/M5)
            htf_name: Name of higher timeframe
            mtf_name: Name of medium timeframe
            ltf_name: Name of lower timeframe

        Returns:
            Dictionary with full analysis and trade setups
        """
        # Analyze each timeframe
        self.htf_analysis = self.analyze_timeframe(htf_df, htf_name)
        self.mtf_analysis = self.analyze_timeframe(mtf_df, mtf_name)
        self.ltf_analysis = self.analyze_timeframe(ltf_df, ltf_name)

        current_price = ltf_df['close'].iloc[-1]

        # Find trade setups
        setups = self._find_trade_setups(current_price)

        return {
            "htf": self.htf_analysis,
            "mtf": self.mtf_analysis,
            "ltf": self.ltf_analysis,
            "current_price": current_price,
            "htf_bias": self.htf_analysis.bias,
            "trade_setups": setups,
            "summary": self._generate_summary(current_price)
        }

    def _find_trade_setups(self, current_price: float) -> List[TradeSetup]:
        """
        Find potential trade setups based on MTF analysis.

        Looks for confluence between:
        - HTF bias
        - MTF order blocks/FVGs
        - LTF entry triggers
        """
        setups = []

        if not all([self.htf_analysis, self.mtf_analysis, self.ltf_analysis]):
            return setups

        htf_bias = self.htf_analysis.bias

        # Only look for setups in the direction of HTF bias
        if htf_bias == "bullish":
            setups.extend(self._find_long_setups(current_price))
        elif htf_bias == "bearish":
            setups.extend(self._find_short_setups(current_price))

        return setups

    def _find_long_setups(self, current_price: float) -> List[TradeSetup]:
        """Find potential long setups."""
        setups = []

        # Get MTF bullish order blocks below price
        mtf_obs = self.ob_detector.get_active_obs(current_price)
        bullish_ob = mtf_obs.get("nearest_bullish_ob")

        # Get MTF bullish FVGs below price
        mtf_fvgs = self.fvg_detector.get_unfilled_fvgs(current_price)
        bullish_fvg = mtf_fvgs.get("nearest_bullish_fvg")

        # Get LTF structure
        ltf_structure = self.ltf_analysis.structure

        # Setup 1: Price at bullish OB with LTF bullish structure
        if bullish_ob and not bullish_ob.is_mitigated:
            distance_to_ob = current_price - bullish_ob.high
            # If price is near the OB (within 2x the OB size)
            ob_size = bullish_ob.high - bullish_ob.low
            if distance_to_ob < ob_size * 2 and distance_to_ob > 0:
                confluences = ["HTF bullish bias", f"MTF bullish OB at {bullish_ob.low:.5f}"]

                # Check for FVG confluence
                if bullish_fvg and abs(bullish_fvg.midpoint - bullish_ob.midpoint) < ob_size:
                    confluences.append(f"FVG confluence at {bullish_fvg.midpoint:.5f}")

                # Check LTF structure
                if ltf_structure.get("bias") == "bullish":
                    confluences.append("LTF bullish structure")

                entry_low = bullish_ob.low
                entry_high = bullish_ob.high
                stop_loss = bullish_ob.low - ob_size * 0.5

                # Target: Next liquidity or structure
                take_profit = current_price + (current_price - stop_loss) * 2

                risk = current_price - stop_loss
                reward = take_profit - current_price
                rr = reward / risk if risk > 0 else 0

                confidence = min(len(confluences) * 25, 100)

                setups.append(TradeSetup(
                    direction="long",
                    entry_zone=(entry_low, entry_high),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=rr,
                    confidence=confidence,
                    htf_bias="bullish",
                    entry_reason="Bullish OB in discount",
                    confluences=confluences
                ))

        return setups

    def _find_short_setups(self, current_price: float) -> List[TradeSetup]:
        """Find potential short setups."""
        setups = []

        # Get MTF bearish order blocks above price
        mtf_obs = self.ob_detector.get_active_obs(current_price)
        bearish_ob = mtf_obs.get("nearest_bearish_ob")

        # Get MTF bearish FVGs above price
        mtf_fvgs = self.fvg_detector.get_unfilled_fvgs(current_price)
        bearish_fvg = mtf_fvgs.get("nearest_bearish_fvg")

        # Get LTF structure
        ltf_structure = self.ltf_analysis.structure

        # Setup 1: Price at bearish OB with LTF bearish structure
        if bearish_ob and not bearish_ob.is_mitigated:
            distance_to_ob = bearish_ob.low - current_price
            ob_size = bearish_ob.high - bearish_ob.low
            if distance_to_ob < ob_size * 2 and distance_to_ob > 0:
                confluences = ["HTF bearish bias", f"MTF bearish OB at {bearish_ob.high:.5f}"]

                if bearish_fvg and abs(bearish_fvg.midpoint - bearish_ob.midpoint) < ob_size:
                    confluences.append(f"FVG confluence at {bearish_fvg.midpoint:.5f}")

                if ltf_structure.get("bias") == "bearish":
                    confluences.append("LTF bearish structure")

                entry_low = bearish_ob.low
                entry_high = bearish_ob.high
                stop_loss = bearish_ob.high + ob_size * 0.5

                take_profit = current_price - (stop_loss - current_price) * 2

                risk = stop_loss - current_price
                reward = current_price - take_profit
                rr = reward / risk if risk > 0 else 0

                confidence = min(len(confluences) * 25, 100)

                setups.append(TradeSetup(
                    direction="short",
                    entry_zone=(entry_low, entry_high),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=rr,
                    confidence=confidence,
                    htf_bias="bearish",
                    entry_reason="Bearish OB in premium",
                    confluences=confluences
                ))

        return setups

    def _generate_summary(self, current_price: float) -> dict:
        """Generate a summary of the MTF analysis."""
        htf = self.htf_analysis
        mtf = self.mtf_analysis
        ltf = self.ltf_analysis

        # Determine alignment
        biases = [htf.bias, mtf.bias, ltf.bias]
        bullish_count = biases.count("bullish")
        bearish_count = biases.count("bearish")

        if bullish_count >= 2:
            alignment = "bullish"
            alignment_strength = bullish_count / 3
        elif bearish_count >= 2:
            alignment = "bearish"
            alignment_strength = bearish_count / 3
        else:
            alignment = "neutral"
            alignment_strength = 0

        # Get premium/discount zone
        htf_pd = htf.structure.get("current_price_zone", "neutral")

        return {
            "htf_bias": htf.bias,
            "mtf_bias": mtf.bias,
            "ltf_bias": ltf.bias,
            "alignment": alignment,
            "alignment_strength": alignment_strength,
            "price_zone": htf_pd,
            "recommendation": self._get_recommendation(alignment, htf_pd, current_price)
        }

    def _get_recommendation(
        self,
        alignment: str,
        price_zone: str,
        current_price: float
    ) -> str:
        """Generate trade recommendation based on analysis."""
        if alignment == "neutral":
            return "NO TRADE - Conflicting timeframe biases"

        if alignment == "bullish":
            if price_zone == "discount":
                return "LOOK FOR LONGS - Bullish alignment in discount zone"
            elif price_zone == "premium":
                return "WAIT - Bullish but price in premium (wait for pullback)"
            else:
                return "CAUTIOUS LONGS - Bullish alignment at equilibrium"

        if alignment == "bearish":
            if price_zone == "premium":
                return "LOOK FOR SHORTS - Bearish alignment in premium zone"
            elif price_zone == "discount":
                return "WAIT - Bearish but price in discount (wait for rally)"
            else:
                return "CAUTIOUS SHORTS - Bearish alignment at equilibrium"

        return "NO TRADE - Insufficient data"
