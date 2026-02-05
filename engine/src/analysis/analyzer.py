"""Market Analyzer - Orchestrates all analysis engines.

Combines strength, structure, liquidity, momentum, and event risk
analysis into a unified market view for trade candidate generation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import structlog

from .strength import StrengthEngine, StrengthAnalysis
from .structure import StructureEngine, StructureAnalysis, StructureState, TrendDirection
from .liquidity import LiquidityEngine, LiquidityAnalysis
from .momentum import MomentumEngine, MomentumState, MomentumBias
from .events import EventRiskEngine, EventRisk

logger = structlog.get_logger(__name__)


@dataclass
class MarketView:
    """Complete market analysis for a symbol."""
    timestamp: datetime
    symbol: str
    timeframe: str

    # Individual analyses
    strength: Optional[StrengthAnalysis]
    structure: Optional[StructureAnalysis]
    liquidity: Optional[LiquidityAnalysis]
    momentum: Optional[MomentumState]
    event_risk: Optional[EventRisk]

    # Synthesized view
    directional_bias: str  # LONG, SHORT, NEUTRAL
    bias_confidence: float  # 0-1
    trade_allowed: bool
    key_levels: Dict[str, float]  # invalidation, entry_zone, targets
    warnings: List[str]


class MarketAnalyzer:
    """Orchestrates all analysis engines for comprehensive market view.

    Usage:
        analyzer = MarketAnalyzer()
        view = analyzer.analyze(symbol, timeframe, ohlc_data)
    """

    def __init__(
        self,
        swing_lookback: int = 5,
        rsi_period: int = 14,
        atr_period: int = 14,
    ):
        """Initialize market analyzer with all engines.

        Args:
            swing_lookback: Bars for swing detection
            rsi_period: RSI calculation period
            atr_period: ATR calculation period
        """
        self.strength_engine = StrengthEngine()
        self.structure_engine = StructureEngine(swing_lookback=swing_lookback)
        self.liquidity_engine = LiquidityEngine()
        self.momentum_engine = MomentumEngine(
            rsi_period=rsi_period,
            atr_period=atr_period,
        )
        self.event_engine = EventRiskEngine()

    def analyze(
        self,
        symbol: str,
        timeframe: str,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: List[datetime],
        pair_returns: Optional[Dict[str, float]] = None,
        daily_data: Optional[Dict] = None,
        weekly_data: Optional[Dict] = None,
    ) -> MarketView:
        """Perform complete market analysis.

        Args:
            symbol: Instrument symbol
            timeframe: Chart timeframe
            opens, highs, lows, closes: OHLC arrays
            timestamps: Bar timestamps
            pair_returns: FX pair returns for strength calculation
            daily_data: Daily OHLC for PDH/PDL
            weekly_data: Weekly OHLC for PWH/PWL

        Returns:
            MarketView with complete analysis
        """
        analysis_time = datetime.now()
        warnings = []

        # 1. Strength Analysis
        strength = None
        if pair_returns:
            try:
                strength = self.strength_engine.calculate_fx_strength(
                    pair_returns, timeframe
                )
            except Exception as e:
                logger.error("Strength analysis failed", error=str(e))
                warnings.append("Strength analysis unavailable")

        # 2. Structure Analysis
        structure = None
        try:
            structure = self.structure_engine.analyze(
                symbol, timeframe, highs, lows, closes, timestamps
            )
        except Exception as e:
            logger.error("Structure analysis failed", error=str(e))
            warnings.append("Structure analysis unavailable")

        # 3. Liquidity Analysis
        liquidity = None
        try:
            liquidity = self.liquidity_engine.analyze(
                symbol, timeframe, opens, highs, lows, closes, timestamps,
                daily_data=daily_data,
                weekly_data=weekly_data,
            )
        except Exception as e:
            logger.error("Liquidity analysis failed", error=str(e))
            warnings.append("Liquidity analysis unavailable")

        # 4. Momentum Analysis
        momentum = None
        swing_highs = None
        swing_lows = None

        if structure:
            swing_highs = [(s.index, s.price) for s in structure.swing_highs]
            swing_lows = [(s.index, s.price) for s in structure.swing_lows]

        try:
            momentum = self.momentum_engine.analyze(
                symbol, timeframe, highs, lows, closes, timestamps,
                swing_highs=swing_highs,
                swing_lows=swing_lows,
            )
            warnings.extend(momentum.warnings)
        except Exception as e:
            logger.error("Momentum analysis failed", error=str(e))
            warnings.append("Momentum analysis unavailable")

        # 5. Event Risk
        event_risk = None
        try:
            event_risk = self.event_engine.assess_risk(symbol)
            warnings.extend(event_risk.warnings)
        except Exception as e:
            logger.error("Event risk analysis failed", error=str(e))

        # 6. Synthesize directional bias
        bias, confidence = self._synthesize_bias(
            symbol, strength, structure, momentum
        )

        # 7. Determine if trading is allowed
        trade_allowed = True
        if event_risk and not event_risk.trade_allowed:
            trade_allowed = False

        # 8. Extract key levels
        key_levels = self._extract_key_levels(
            structure, liquidity, closes[-1] if len(closes) > 0 else 0
        )

        return MarketView(
            timestamp=analysis_time,
            symbol=symbol,
            timeframe=timeframe,
            strength=strength,
            structure=structure,
            liquidity=liquidity,
            momentum=momentum,
            event_risk=event_risk,
            directional_bias=bias,
            bias_confidence=confidence,
            trade_allowed=trade_allowed,
            key_levels=key_levels,
            warnings=warnings,
        )

    def _synthesize_bias(
        self,
        symbol: str,
        strength: Optional[StrengthAnalysis],
        structure: Optional[StructureAnalysis],
        momentum: Optional[MomentumState],
    ) -> tuple[str, float]:
        """Synthesize overall directional bias from all analyses."""
        votes = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
        total_weight = 0

        # Structure vote (weight: 30%)
        if structure:
            weight = 0.30
            total_weight += weight
            if structure.trend_direction == TrendDirection.UP:
                votes["LONG"] += weight
            elif structure.trend_direction == TrendDirection.DOWN:
                votes["SHORT"] += weight
            else:
                votes["NEUTRAL"] += weight

        # Strength vote (weight: 25%)
        if strength:
            weight = 0.25
            total_weight += weight
            pair_bias, _ = self.strength_engine.get_pair_bias(
                symbol, strength.strengths
            )
            if pair_bias == "LONG":
                votes["LONG"] += weight
            elif pair_bias == "SHORT":
                votes["SHORT"] += weight
            else:
                votes["NEUTRAL"] += weight

        # Momentum vote (weight: 25%)
        if momentum:
            weight = 0.25
            total_weight += weight
            if momentum.bias == MomentumBias.BULLISH:
                votes["LONG"] += weight
            elif momentum.bias == MomentumBias.BEARISH:
                votes["SHORT"] += weight
            else:
                votes["NEUTRAL"] += weight

        # Default weight for missing analyses
        if total_weight == 0:
            return "NEUTRAL", 0.0

        # Normalize votes
        for direction in votes:
            votes[direction] /= total_weight

        # Determine winner
        max_vote = max(votes.values())
        if max_vote < 0.4:  # No clear winner
            return "NEUTRAL", max_vote

        for direction, vote in votes.items():
            if vote == max_vote:
                return direction, vote

        return "NEUTRAL", 0.0

    def _extract_key_levels(
        self,
        structure: Optional[StructureAnalysis],
        liquidity: Optional[LiquidityAnalysis],
        current_price: float,
    ) -> Dict[str, float]:
        """Extract key trading levels from analyses."""
        levels = {}

        if structure:
            if structure.bullish_invalidation:
                levels["bullish_invalidation"] = structure.bullish_invalidation
            if structure.bearish_invalidation:
                levels["bearish_invalidation"] = structure.bearish_invalidation
            if structure.range_high:
                levels["range_high"] = structure.range_high
            if structure.range_low:
                levels["range_low"] = structure.range_low

        if liquidity:
            if liquidity.nearest_resistance:
                levels["nearest_resistance"] = liquidity.nearest_resistance.price
            if liquidity.nearest_support:
                levels["nearest_support"] = liquidity.nearest_support.price
            if liquidity.long_entry_zone:
                levels["long_entry_low"] = liquidity.long_entry_zone[0]
                levels["long_entry_high"] = liquidity.long_entry_zone[1]
            if liquidity.short_entry_zone:
                levels["short_entry_low"] = liquidity.short_entry_zone[0]
                levels["short_entry_high"] = liquidity.short_entry_zone[1]

        levels["current_price"] = current_price

        return levels

    def to_snapshot_dict(self, view: MarketView) -> Dict[str, Any]:
        """Convert MarketView to dictionary for database storage."""
        return {
            "symbol": view.symbol,
            "timeframe": view.timeframe,
            "snapshot_time": view.timestamp,
            "structure_state": view.structure.state.value if view.structure else None,
            "trend_direction": view.structure.trend_direction.value if view.structure else None,
            "strength_scores": self.strength_engine.to_dict(view.strength) if view.strength else None,
            "liquidity_zones": self.liquidity_engine.to_dict(view.liquidity) if view.liquidity else None,
            "momentum_state": self.momentum_engine.to_dict(view.momentum) if view.momentum else None,
            "event_risk": self.event_engine.to_dict(view.event_risk) if view.event_risk else None,
            "computed_levels": view.key_levels,
        }

    def update_economic_calendar(self, events: List) -> None:
        """Update the economic calendar."""
        self.event_engine.update_calendar(events)
