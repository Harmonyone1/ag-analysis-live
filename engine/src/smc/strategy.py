"""
Enhanced SMC Trading Strategy

Combines all Smart Money Concepts modules with external data
for high-probability trade setups:

1. Multi-Timeframe Analysis (HTF bias -> LTF entry)
2. Order Block / FVG confluence zones
3. Liquidity sweep detection
4. COT institutional positioning
5. Retail sentiment (contrarian)
6. Economic calendar filtering

Entry Criteria:
- HTF bias established (structure + COT alignment)
- MTF POI identified (OB/FVG in premium/discount)
- LTF entry trigger (BOS/ChoCH in direction of bias)
- No high-impact news within buffer
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import pandas as pd

from .market_structure import MarketStructure, TrendBias
from .order_blocks import OrderBlockDetector, OrderBlock
from .fair_value_gaps import FVGDetector, FairValueGap
from .liquidity import LiquidityZoneDetector, LiquidityZone
from .multi_timeframe import MultiTimeframeAnalyzer, TradeSetup, TimeframeAnalysis


class SetupQuality(Enum):
    """Quality rating for trade setups."""
    A_PLUS = "A+"  # All confluences aligned
    A = "A"        # Strong setup, minor missing confluence
    B = "B"        # Decent setup, some risk
    C = "C"        # Marginal, only trade with tight stop


@dataclass
class EnhancedSetup:
    """
    Enhanced trade setup with full SMC analysis.

    This is the main output of the strategy - a complete
    trade recommendation with all supporting analysis.
    """
    # Basic setup info
    symbol: str
    direction: str  # "long" or "short"
    quality: SetupQuality

    # Entry details
    entry_price: float
    entry_zone: Tuple[float, float]  # (low, high)
    stop_loss: float
    take_profit_1: float  # Conservative target
    take_profit_2: float  # Extended target
    take_profit_3: float  # Full target (liquidity sweep)

    # Risk management
    risk_pips: float
    reward_pips: float
    risk_reward: float
    position_size_factor: float  # 0.25 to 1.0 based on quality

    # Analysis components
    htf_bias: str
    mtf_structure: str
    ltf_trigger: str
    price_zone: str  # "premium", "discount", "equilibrium"

    # Confluence factors
    confluences: List[str] = field(default_factory=list)
    confluence_score: int = 0  # 0-100

    # External data alignment
    cot_bias: Optional[str] = None
    retail_sentiment_bias: Optional[str] = None
    news_safe: bool = True
    news_warning: Optional[str] = None

    # Entry trigger
    entry_reason: str = ""
    invalidation_level: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        """Check if setup is still valid."""
        return self.quality in [SetupQuality.A_PLUS, SetupQuality.A, SetupQuality.B]

    @property
    def recommended_lots(self) -> float:
        """Get recommended lot size based on quality."""
        base_lots = 0.1
        return base_lots * self.position_size_factor

    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "quality": self.quality.value,
            "entry_zone": f"{self.entry_zone[0]:.5f} - {self.entry_zone[1]:.5f}",
            "stop_loss": f"{self.stop_loss:.5f}",
            "tp1": f"{self.take_profit_1:.5f}",
            "tp2": f"{self.take_profit_2:.5f}",
            "tp3": f"{self.take_profit_3:.5f}",
            "risk_reward": f"{self.risk_reward:.2f}",
            "confluence_score": self.confluence_score,
            "confluences": self.confluences,
            "htf_bias": self.htf_bias,
            "cot_alignment": self.cot_bias == self.htf_bias if self.cot_bias else "N/A",
            "sentiment_alignment": self.retail_sentiment_bias == self.htf_bias if self.retail_sentiment_bias else "N/A",
            "news_safe": self.news_safe,
            "entry_reason": self.entry_reason,
        }


class SMCStrategy:
    """
    Professional SMC-based trading strategy.

    Uses institutional trading concepts:
    1. Top-down analysis (HTF -> LTF)
    2. Order flow concepts (OB, FVG, liquidity)
    3. External data validation (COT, sentiment)
    4. News-aware execution
    """

    def __init__(
        self,
        min_rr: float = 2.0,
        min_confluence: int = 3,
        news_buffer_minutes: int = 30
    ):
        """
        Initialize SMC Strategy.

        Args:
            min_rr: Minimum risk:reward ratio
            min_confluence: Minimum confluence factors
            news_buffer_minutes: Minutes to avoid before/after news
        """
        self.min_rr = min_rr
        self.min_confluence = min_confluence
        self.news_buffer_minutes = news_buffer_minutes

        # Initialize analyzers
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.structure = MarketStructure(swing_lookback=5)
        self.ob_detector = OrderBlockDetector()
        self.fvg_detector = FVGDetector()
        self.liquidity_detector = LiquidityZoneDetector()

    def analyze(
        self,
        symbol: str,
        htf_data: pd.DataFrame,
        mtf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        cot_bias: Optional[str] = None,
        sentiment_bias: Optional[str] = None,
        news_safe: bool = True,
        news_warning: Optional[str] = None
    ) -> Optional[EnhancedSetup]:
        """
        Perform full SMC analysis and generate trade setup.

        Args:
            symbol: Trading symbol
            htf_data: H4/Daily OHLC data
            mtf_data: H1 OHLC data
            ltf_data: M15/M5 OHLC data
            cot_bias: COT data bias ("bullish", "bearish", "neutral")
            sentiment_bias: Retail sentiment contrarian bias
            news_safe: Whether it's safe from news
            news_warning: Any news-related warning

        Returns:
            EnhancedSetup if valid setup found, None otherwise
        """
        # 1. Multi-timeframe analysis
        mtf_result = self.mtf_analyzer.full_analysis(
            htf_data, mtf_data, ltf_data,
            htf_name="H4", mtf_name="H1", ltf_name="M15"
        )

        current_price = mtf_result["current_price"]
        htf_bias = mtf_result["htf_bias"]
        summary = mtf_result["summary"]

        # 2. Check for alignment
        if summary["alignment"] == "neutral":
            return None  # No clear direction

        # 3. Find the best POI (Point of Interest)
        poi = self._find_best_poi(
            mtf_result,
            current_price,
            summary["alignment"]
        )

        if not poi:
            return None  # No valid POI

        # 4. Calculate confluence score
        confluences, score = self._calculate_confluence(
            mtf_result,
            poi,
            cot_bias,
            sentiment_bias,
            news_safe
        )

        if score < self.min_confluence * 20:  # Each confluence worth ~20 points
            return None

        # 5. Calculate entry, SL, TP
        entry_zone, sl, tp1, tp2, tp3 = self._calculate_levels(
            poi,
            current_price,
            mtf_result,
            summary["alignment"]
        )

        # 6. Calculate risk/reward
        entry_mid = (entry_zone[0] + entry_zone[1]) / 2

        if summary["alignment"] == "bullish":
            risk_pips = (entry_mid - sl) * 10000  # For forex pairs
            reward_pips = (tp2 - entry_mid) * 10000
        else:
            risk_pips = (sl - entry_mid) * 10000
            reward_pips = (entry_mid - tp2) * 10000

        rr = reward_pips / risk_pips if risk_pips > 0 else 0

        if rr < self.min_rr:
            return None  # RR too low

        # 7. Determine quality
        quality = self._determine_quality(score, rr, news_safe, cot_bias, htf_bias)

        # 8. Calculate position size factor
        size_factor = self._calculate_size_factor(quality, news_safe, score)

        # 9. Build the setup
        setup = EnhancedSetup(
            symbol=symbol,
            direction="long" if summary["alignment"] == "bullish" else "short",
            quality=quality,
            entry_price=entry_mid,
            entry_zone=entry_zone,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            risk_pips=risk_pips,
            reward_pips=reward_pips,
            risk_reward=rr,
            position_size_factor=size_factor,
            htf_bias=htf_bias,
            mtf_structure=summary["mtf_bias"],
            ltf_trigger=summary["ltf_bias"],
            price_zone=summary["price_zone"],
            confluences=confluences,
            confluence_score=score,
            cot_bias=cot_bias,
            retail_sentiment_bias=sentiment_bias,
            news_safe=news_safe,
            news_warning=news_warning,
            entry_reason=poi.get("reason", "POI confluence"),
            invalidation_level=sl,
        )

        return setup

    def _find_best_poi(
        self,
        mtf_result: dict,
        current_price: float,
        direction: str
    ) -> Optional[dict]:
        """
        Find the best Point of Interest for entry.

        Prioritizes:
        1. OB + FVG overlap (strongest)
        2. OB at key level
        3. FVG in premium/discount
        """
        mtf_analysis = mtf_result["mtf"]

        if direction == "bullish":
            # Look for bullish POI below price (discount zone)
            obs = [ob for ob in mtf_analysis.order_blocks
                   if ob.ob_type == "bullish" and not ob.is_mitigated
                   and ob.high < current_price]

            fvgs = [fvg for fvg in mtf_analysis.fvgs
                    if fvg.fvg_type == "bullish" and not fvg.is_filled
                    and fvg.high < current_price]

        else:
            # Look for bearish POI above price (premium zone)
            obs = [ob for ob in mtf_analysis.order_blocks
                   if ob.ob_type == "bearish" and not ob.is_mitigated
                   and ob.low > current_price]

            fvgs = [fvg for fvg in mtf_analysis.fvgs
                    if fvg.fvg_type == "bearish" and not fvg.is_filled
                    and fvg.low > current_price]

        if not obs and not fvgs:
            return None

        # Check for OB + FVG overlap (best setup)
        for ob in obs:
            for fvg in fvgs:
                # Check if they overlap
                overlap_low = max(ob.low, fvg.low)
                overlap_high = min(ob.high, fvg.high)
                if overlap_low < overlap_high:
                    return {
                        "type": "ob_fvg_overlap",
                        "low": overlap_low,
                        "high": overlap_high,
                        "ob": ob,
                        "fvg": fvg,
                        "reason": "OB + FVG overlap",
                        "strength": 5
                    }

        # Use best OB
        if obs:
            best_ob = max(obs, key=lambda x: x.strength)
            return {
                "type": "order_block",
                "low": best_ob.low,
                "high": best_ob.high,
                "ob": best_ob,
                "reason": f"{best_ob.ob_type.capitalize()} OB",
                "strength": 3
            }

        # Use best FVG
        if fvgs:
            best_fvg = min(fvgs, key=lambda x: abs(x.midpoint - current_price))
            return {
                "type": "fvg",
                "low": best_fvg.low,
                "high": best_fvg.high,
                "fvg": best_fvg,
                "reason": f"{best_fvg.fvg_type.capitalize()} FVG",
                "strength": 2
            }

        return None

    def _calculate_confluence(
        self,
        mtf_result: dict,
        poi: dict,
        cot_bias: Optional[str],
        sentiment_bias: Optional[str],
        news_safe: bool
    ) -> Tuple[List[str], int]:
        """Calculate confluence factors and score."""
        confluences = []
        score = 0

        summary = mtf_result["summary"]
        direction = summary["alignment"]

        # 1. HTF bias alignment (25 points)
        if mtf_result["htf_bias"] == direction:
            confluences.append(f"HTF {direction} bias")
            score += 25

        # 2. MTF structure alignment (20 points)
        if summary["mtf_bias"] == direction:
            confluences.append(f"MTF {direction} structure")
            score += 20

        # 3. LTF confirmation (15 points)
        if summary["ltf_bias"] == direction:
            confluences.append(f"LTF {direction} confirmation")
            score += 15

        # 4. POI quality (10-20 points)
        if poi["type"] == "ob_fvg_overlap":
            confluences.append("OB + FVG overlap")
            score += 20
        elif poi["type"] == "order_block":
            confluences.append(f"Order Block")
            score += 15
        elif poi["type"] == "fvg":
            confluences.append("Fair Value Gap")
            score += 10

        # 5. Price zone (10 points)
        if direction == "bullish" and summary["price_zone"] == "discount":
            confluences.append("Price in discount zone")
            score += 10
        elif direction == "bearish" and summary["price_zone"] == "premium":
            confluences.append("Price in premium zone")
            score += 10

        # 6. COT alignment (10 points)
        if cot_bias and cot_bias == direction:
            confluences.append(f"COT {direction} positioning")
            score += 10

        # 7. Sentiment alignment (5 points) - contrarian
        if sentiment_bias and sentiment_bias == direction:
            confluences.append(f"Retail fading ({direction})")
            score += 5

        # 8. News clear (5 points)
        if news_safe:
            confluences.append("No imminent news")
            score += 5

        # 9. Liquidity target available
        htf_analysis = mtf_result["htf"]
        liq_zones = [z for z in htf_analysis.liquidity_zones if not z.is_swept]

        if direction == "bullish":
            targets = [z for z in liq_zones if z.zone_type == "buy_side"]
        else:
            targets = [z for z in liq_zones if z.zone_type == "sell_side"]

        if targets:
            confluences.append(f"Liquidity target available")
            score += 5

        return confluences, min(score, 100)

    def _calculate_levels(
        self,
        poi: dict,
        current_price: float,
        mtf_result: dict,
        direction: str
    ) -> Tuple[Tuple[float, float], float, float, float, float]:
        """
        Calculate entry zone, stop loss, and take profits.

        Returns:
            (entry_zone, stop_loss, tp1, tp2, tp3)
        """
        poi_low = poi["low"]
        poi_high = poi["high"]
        poi_size = poi_high - poi_low

        # Get HTF levels for targets
        htf_analysis = mtf_result["htf"]
        liquidity_zones = [z for z in htf_analysis.liquidity_zones if not z.is_swept]

        if direction == "bullish":
            # Entry at POI
            entry_zone = (poi_low, poi_high)

            # Stop loss below POI
            stop_loss = poi_low - poi_size * 0.5

            # Take profits
            risk = poi_high - stop_loss

            # TP1: 1:1 RR (conservative)
            tp1 = poi_high + risk

            # TP2: 1:2 RR
            tp2 = poi_high + risk * 2

            # TP3: Liquidity sweep or 1:3 RR
            buy_side_liq = [z for z in liquidity_zones
                           if z.zone_type == "buy_side" and z.price > current_price]
            if buy_side_liq:
                tp3 = max(buy_side_liq, key=lambda z: z.price).price
            else:
                tp3 = poi_high + risk * 3

        else:  # bearish
            entry_zone = (poi_low, poi_high)

            # Stop loss above POI
            stop_loss = poi_high + poi_size * 0.5

            risk = stop_loss - poi_low

            # TP1: 1:1 RR
            tp1 = poi_low - risk

            # TP2: 1:2 RR
            tp2 = poi_low - risk * 2

            # TP3: Liquidity sweep or 1:3 RR
            sell_side_liq = [z for z in liquidity_zones
                            if z.zone_type == "sell_side" and z.price < current_price]
            if sell_side_liq:
                tp3 = min(sell_side_liq, key=lambda z: z.price).price
            else:
                tp3 = poi_low - risk * 3

        return entry_zone, stop_loss, tp1, tp2, tp3

    def _determine_quality(
        self,
        score: int,
        rr: float,
        news_safe: bool,
        cot_bias: Optional[str],
        htf_bias: str
    ) -> SetupQuality:
        """Determine setup quality grade."""
        if score >= 85 and rr >= 3.0 and news_safe:
            if cot_bias == htf_bias:
                return SetupQuality.A_PLUS
            return SetupQuality.A

        if score >= 70 and rr >= 2.5:
            return SetupQuality.A

        if score >= 55 and rr >= 2.0:
            return SetupQuality.B

        return SetupQuality.C

    def _calculate_size_factor(
        self,
        quality: SetupQuality,
        news_safe: bool,
        score: int
    ) -> float:
        """Calculate position size factor based on quality."""
        base_factors = {
            SetupQuality.A_PLUS: 1.0,
            SetupQuality.A: 0.75,
            SetupQuality.B: 0.5,
            SetupQuality.C: 0.25,
        }

        factor = base_factors.get(quality, 0.25)

        # Reduce if news risk
        if not news_safe:
            factor *= 0.5

        return factor

    def scan_symbols(
        self,
        symbol_data: Dict[str, Dict[str, pd.DataFrame]],
        cot_data: Optional[Dict[str, str]] = None,
        sentiment_data: Optional[Dict[str, str]] = None,
        calendar_safe: Optional[Dict[str, bool]] = None
    ) -> List[EnhancedSetup]:
        """
        Scan multiple symbols for valid setups.

        Args:
            symbol_data: Dict mapping symbol to {htf, mtf, ltf} DataFrames
            cot_data: Dict mapping symbol to COT bias
            sentiment_data: Dict mapping symbol to sentiment bias
            calendar_safe: Dict mapping symbol to news safety

        Returns:
            List of valid EnhancedSetup objects, sorted by quality
        """
        setups = []

        for symbol, timeframes in symbol_data.items():
            try:
                cot_bias = cot_data.get(symbol) if cot_data else None
                sent_bias = sentiment_data.get(symbol) if sentiment_data else None
                news_safe = calendar_safe.get(symbol, True) if calendar_safe else True

                setup = self.analyze(
                    symbol=symbol,
                    htf_data=timeframes["htf"],
                    mtf_data=timeframes["mtf"],
                    ltf_data=timeframes["ltf"],
                    cot_bias=cot_bias,
                    sentiment_bias=sent_bias,
                    news_safe=news_safe
                )

                if setup and setup.is_valid:
                    setups.append(setup)

            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue

        # Sort by quality then confluence score
        quality_order = [SetupQuality.A_PLUS, SetupQuality.A, SetupQuality.B, SetupQuality.C]
        setups.sort(key=lambda s: (quality_order.index(s.quality), -s.confluence_score))

        return setups


def format_setup_report(setup: EnhancedSetup) -> str:
    """Format a setup for display."""
    report = f"""
{'='*60}
SMC TRADE SETUP - {setup.symbol} {setup.direction.upper()}
{'='*60}

QUALITY: {setup.quality.value} | CONFLUENCE: {setup.confluence_score}/100

DIRECTION: {setup.direction.upper()}
HTF BIAS: {setup.htf_bias}
PRICE ZONE: {setup.price_zone}

ENTRY ZONE: {setup.entry_zone[0]:.5f} - {setup.entry_zone[1]:.5f}
STOP LOSS: {setup.stop_loss:.5f}

TARGETS:
  TP1 (1:1): {setup.take_profit_1:.5f}
  TP2 (2:1): {setup.take_profit_2:.5f}
  TP3 (LIQ): {setup.take_profit_3:.5f}

RISK/REWARD: {setup.risk_reward:.2f}
RISK: {setup.risk_pips:.1f} pips
REWARD: {setup.reward_pips:.1f} pips

POSITION SIZE: {setup.position_size_factor:.0%} of normal

CONFLUENCE FACTORS:
{chr(10).join(f'  + {c}' for c in setup.confluences)}

EXTERNAL DATA:
  COT: {setup.cot_bias or 'N/A'}
  Sentiment: {setup.retail_sentiment_bias or 'N/A'}
  News Safe: {'Yes' if setup.news_safe else 'No - ' + (setup.news_warning or 'Risk')}

ENTRY REASON: {setup.entry_reason}
INVALIDATION: {setup.invalidation_level:.5f}

{'='*60}
"""
    return report
