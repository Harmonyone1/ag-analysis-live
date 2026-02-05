"""Confluence Scoring System for trade candidate ranking.

Combines multiple analysis factors into a single confluence score
with explainable reason codes for each contributing factor.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import structlog

from src.analysis.analyzer import MarketView
from src.analysis.structure import StructureState, TrendDirection
from src.analysis.liquidity import SweepType, ZoneType
from src.analysis.momentum import MomentumBias, DivergenceType
from .reasons import ReasonCode, get_reason_description

logger = structlog.get_logger(__name__)


@dataclass
class ScoringWeights:
    """Configurable weights for confluence scoring.

    Optimized weights based on ICT/SMC methodology:
    - Structure: Most important - trading with trend/structure
    - Liquidity: Critical - sweeps and zones define entries
    - Momentum: Confirmation factor
    - Strength: Context factor (less reliable intraday)
    - Regime: Market condition awareness
    - Sentiment: Minor factor
    """
    # Optimized weights (research-backed)
    strength: float = 15.0   # Reduced - less reliable on short TF
    structure: float = 30.0  # Increased - trend alignment critical
    liquidity: float = 28.0  # Increased - SMC core concept
    momentum: float = 15.0   # Confirmation
    regime: float = 8.0      # Context
    sentiment: float = 4.0   # Minor

    def normalize(self) -> 'ScoringWeights':
        """Normalize weights to sum to 100."""
        total = (self.strength + self.structure + self.liquidity +
                 self.momentum + self.regime + self.sentiment)
        if total == 0:
            return self
        factor = 100.0 / total
        return ScoringWeights(
            strength=self.strength * factor,
            structure=self.structure * factor,
            liquidity=self.liquidity * factor,
            momentum=self.momentum * factor,
            regime=self.regime * factor,
            sentiment=self.sentiment * factor,
        )


@dataclass
class ConfluenceBonus:
    """Bonus scores for specific confluence patterns."""
    # High-probability pattern bonuses
    sweep_plus_structure_bonus: float = 8.0  # Sweep + trend alignment
    divergence_plus_zone_bonus: float = 6.0  # Divergence at liquidity zone
    multi_tf_alignment_bonus: float = 10.0   # Multiple timeframe agreement
    session_optimal_bonus: float = 5.0       # Trading optimal session for pair

    # Risk penalty factors
    counter_trend_penalty: float = -15.0     # Trading against structure
    news_proximity_penalty: float = -10.0    # Near high-impact news
    low_liquidity_penalty: float = -8.0      # Off-hours trading
    overextended_penalty: float = -12.0      # RSI extreme + far from mean


@dataclass
class AdaptiveWeightConfig:
    """Configuration for adaptive weight adjustment."""
    # Volatility-based adjustments
    high_vol_structure_boost: float = 1.2    # Boost structure weight in high vol
    high_vol_momentum_reduce: float = 0.8    # Reduce momentum weight in high vol
    low_vol_liquidity_boost: float = 1.15    # Boost liquidity weight in low vol

    # Session-based adjustments
    london_structure_weight: float = 1.1     # London favors structure plays
    asian_liquidity_weight: float = 1.2      # Asian session is range-bound
    ny_momentum_weight: float = 1.15         # NY has momentum moves

    # Trend strength adjustments
    strong_trend_momentum_boost: float = 1.2  # Momentum matters more in trends
    ranging_liquidity_boost: float = 1.25     # Liquidity matters in ranges


@dataclass
class TradeSetup:
    """A scored trade setup candidate."""
    symbol: str
    direction: str  # LONG or SHORT
    entry_type: str  # MARKET or LIMIT

    # Entry details
    entry_zone: Tuple[float, float]  # (min, max) price
    stop_price: float
    invalidation_price: float
    tp_targets: List[Dict]  # [{"r": 1.5, "price": 1.09}]

    # Scoring
    confluence_score: int  # 0-100
    sub_scores: Dict[str, float]  # Individual component scores
    reasons: List[str]  # Human-readable reasons
    reason_codes: List[str]  # Machine-readable codes

    # Risk metrics
    risk_reward: float
    atr_distance: float  # Stop distance in ATR units

    # Metadata
    timestamp: datetime
    timeframe: str
    snapshot_id: Optional[str] = None


class ConfluenceScorer:
    """Scores trade setups based on confluence of analysis factors.

    Uses adaptive weights that adjust based on market conditions,
    confluence bonuses for high-probability patterns, and session-aware
    scoring adjustments.

    Example:
        scorer = ConfluenceScorer()
        setups = scorer.generate_setups(market_view)
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        bonus_config: Optional[ConfluenceBonus] = None,
        adaptive_config: Optional[AdaptiveWeightConfig] = None,
    ):
        """Initialize confluence scorer.

        Args:
            weights: Base scoring weights (defaults to optimized)
            bonus_config: Confluence bonus configuration
            adaptive_config: Adaptive weight adjustment config
        """
        self.base_weights = (weights or ScoringWeights()).normalize()
        self.weights = self.base_weights  # Current active weights
        self.bonus_config = bonus_config or ConfluenceBonus()
        self.adaptive_config = adaptive_config or AdaptiveWeightConfig()
        self.min_score_threshold = 20  # Minimum score to generate candidate

    def _get_adaptive_weights(self, view: MarketView) -> ScoringWeights:
        """Calculate adaptive weights based on market conditions.

        Args:
            view: Current market analysis

        Returns:
            Adjusted ScoringWeights for current conditions
        """
        from datetime import datetime

        # Start with base weights
        strength = self.base_weights.strength
        structure = self.base_weights.structure
        liquidity = self.base_weights.liquidity
        momentum = self.base_weights.momentum
        regime = self.base_weights.regime
        sentiment = self.base_weights.sentiment

        ac = self.adaptive_config

        # Volatility adjustments
        if view.momentum and view.momentum.atr_percentile:
            atr_pct = view.momentum.atr_percentile
            if atr_pct > 0.7:  # High volatility
                structure *= ac.high_vol_structure_boost
                momentum *= ac.high_vol_momentum_reduce
            elif atr_pct < 0.3:  # Low volatility
                liquidity *= ac.low_vol_liquidity_boost

        # Session-based adjustments
        hour = datetime.now().hour
        if 7 <= hour < 16:  # London session
            structure *= ac.london_structure_weight
        elif 0 <= hour < 9:  # Asian session
            liquidity *= ac.asian_liquidity_weight
        elif 12 <= hour < 21:  # NY session
            momentum *= ac.ny_momentum_weight

        # Trend strength adjustments
        if view.structure:
            state = str(view.structure.state.value).lower() if view.structure.state else ""
            if "trend" in state:
                momentum *= ac.strong_trend_momentum_boost
            elif "rang" in state or "accumul" in state or "distribut" in state:
                liquidity *= ac.ranging_liquidity_boost

        # Return normalized weights
        return ScoringWeights(
            strength=strength,
            structure=structure,
            liquidity=liquidity,
            momentum=momentum,
            regime=regime,
            sentiment=sentiment,
        ).normalize()

    def _calculate_confluence_bonus(
        self,
        view: MarketView,
        direction: str,
        sub_scores: Dict[str, float]
    ) -> Tuple[float, List[str]]:
        """Calculate bonus/penalty for confluence patterns.

        Args:
            view: Market analysis
            direction: Trade direction
            sub_scores: Individual component scores

        Returns:
            Tuple of (bonus_amount, reason_list)
        """
        from datetime import datetime

        bonus = 0.0
        reasons = []
        bc = self.bonus_config

        # Check for sweep + structure alignment (high probability)
        has_sweep = sub_scores.get("liquidity", 50) > 65
        has_structure = sub_scores.get("structure", 50) > 65

        if has_sweep and has_structure:
            bonus += bc.sweep_plus_structure_bonus
            reasons.append("Liquidity sweep with structure alignment")

        # Check for divergence at liquidity zone
        has_divergence = False
        if view.momentum:
            if direction == "LONG" and view.momentum.divergence and "BULLISH" in str(view.momentum.divergence.value):
                has_divergence = True
            elif direction == "SHORT" and view.momentum.divergence and "BEARISH" in str(view.momentum.divergence.value):
                has_divergence = True

        at_liquidity_zone = view.liquidity and (
            (direction == "LONG" and view.liquidity.long_entry_zone) or
            (direction == "SHORT" and view.liquidity.short_entry_zone)
        )

        if has_divergence and at_liquidity_zone:
            bonus += bc.divergence_plus_zone_bonus
            reasons.append("Divergence at liquidity zone")

        # Session optimal bonus
        hour = datetime.now().hour
        symbol = view.symbol
        is_jpy_pair = "JPY" in symbol
        is_eur_gbp = "EUR" in symbol or "GBP" in symbol
        is_aud_nzd = "AUD" in symbol or "NZD" in symbol

        if 0 <= hour < 9 and is_jpy_pair:  # Tokyo session for JPY
            bonus += bc.session_optimal_bonus
            reasons.append("Optimal session for JPY pair")
        elif 7 <= hour < 16 and is_eur_gbp:  # London for EUR/GBP
            bonus += bc.session_optimal_bonus
            reasons.append("Optimal session for EUR/GBP pair")
        elif hour >= 21 or hour < 6:  # Sydney for AUD/NZD
            if is_aud_nzd:
                bonus += bc.session_optimal_bonus
                reasons.append("Optimal session for AUD/NZD pair")

        # Counter-trend penalty
        if view.structure and view.structure.trend_direction:
            trend = str(view.structure.trend_direction.value).upper()
            if (direction == "LONG" and trend == "DOWN") or (direction == "SHORT" and trend == "UP"):
                bonus += bc.counter_trend_penalty
                reasons.append("Counter-trend trade (penalty)")

        # Overextended penalty (RSI extreme without mean reversion signal)
        if view.momentum and view.momentum.rsi:
            rsi = view.momentum.rsi
            if direction == "LONG" and rsi > 75:
                bonus += bc.overextended_penalty
                reasons.append("Overextended momentum (penalty)")
            elif direction == "SHORT" and rsi < 25:
                bonus += bc.overextended_penalty
                reasons.append("Overextended momentum (penalty)")

        return bonus, reasons

    def generate_setups(
        self,
        view: MarketView,
        atr: float = 0.0,
    ) -> List[TradeSetup]:
        """Generate scored trade setups from market view.

        Args:
            view: Complete market analysis
            atr: Current ATR for position sizing

        Returns:
            List of TradeSetup candidates (may be empty)
        """
        setups = []

        # Only generate if trading is allowed
        if not view.trade_allowed:
            logger.debug("Trading not allowed - no setups generated",
                        symbol=view.symbol, warnings=view.warnings)
            return setups

        # Try to generate long setup
        long_setup = self._evaluate_direction(view, "LONG", atr)
        if long_setup and long_setup.confluence_score >= self.min_score_threshold:
            setups.append(long_setup)

        # Try to generate short setup
        short_setup = self._evaluate_direction(view, "SHORT", atr)
        if short_setup and short_setup.confluence_score >= self.min_score_threshold:
            setups.append(short_setup)

        return setups

    def _evaluate_direction(
        self,
        view: MarketView,
        direction: str,
        atr: float,
    ) -> Optional[TradeSetup]:
        """Evaluate a specific direction for trade setup.

        Args:
            view: Market analysis
            direction: LONG or SHORT
            atr: Current ATR

        Returns:
            TradeSetup if valid, None otherwise
        """
        reasons = []
        reason_codes = []
        sub_scores = {}

        # Score each component
        strength_score, str_reasons = self._score_strength(view, direction)
        sub_scores["strength"] = strength_score
        reasons.extend(str_reasons)
        reason_codes.extend([r.value for r in self._get_strength_codes(view, direction)])

        structure_score, struct_reasons = self._score_structure(view, direction)
        sub_scores["structure"] = structure_score
        reasons.extend(struct_reasons)
        reason_codes.extend([r.value for r in self._get_structure_codes(view, direction)])

        liquidity_score, liq_reasons = self._score_liquidity(view, direction)
        sub_scores["liquidity"] = liquidity_score
        reasons.extend(liq_reasons)
        reason_codes.extend([r.value for r in self._get_liquidity_codes(view, direction)])

        momentum_score, mom_reasons = self._score_momentum(view, direction)
        sub_scores["momentum"] = momentum_score
        reasons.extend(mom_reasons)
        reason_codes.extend([r.value for r in self._get_momentum_codes(view, direction)])

        regime_score = self._score_regime(view, direction)
        sub_scores["regime"] = regime_score

        # Get adaptive weights for current market conditions
        adaptive_weights = self._get_adaptive_weights(view)

        # Calculate weighted total using adaptive weights
        total_score = (
            strength_score * adaptive_weights.strength / 100 +
            structure_score * adaptive_weights.structure / 100 +
            liquidity_score * adaptive_weights.liquidity / 100 +
            momentum_score * adaptive_weights.momentum / 100 +
            regime_score * adaptive_weights.regime / 100
        )

        # Apply confluence bonuses/penalties
        bonus_score, bonus_reasons = self._calculate_confluence_bonus(
            view, direction, sub_scores
        )
        total_score += bonus_score
        reasons.extend(bonus_reasons)

        confluence_score = int(min(100, max(0, total_score)))

        # Store adaptive weights in sub_scores for debugging
        sub_scores["_adaptive_weights"] = {
            "strength": adaptive_weights.strength,
            "structure": adaptive_weights.structure,
            "liquidity": adaptive_weights.liquidity,
            "momentum": adaptive_weights.momentum,
            "regime": adaptive_weights.regime,
        }
        sub_scores["_bonus"] = bonus_score

        # Skip if score too low
        if confluence_score < self.min_score_threshold:
            return None

        # Calculate entry zone and stops
        entry_zone, stop_price, invalidation = self._calculate_levels(
            view, direction, atr
        )

        if entry_zone is None:
            return None

        # Validate stop is on correct side of entry
        entry_mid = (entry_zone[0] + entry_zone[1]) / 2
        if direction == "LONG" and stop_price >= entry_mid:
            # Invalid: stop above entry for LONG
            logger.debug(f"Skipping LONG setup: stop {stop_price} >= entry {entry_mid}")
            return None
        elif direction == "SHORT" and stop_price <= entry_mid:
            # Invalid: stop below entry for SHORT
            logger.debug(f"Skipping SHORT setup: stop {stop_price} <= entry {entry_mid}")
            return None

        # Calculate targets
        tp_targets = self._calculate_targets(entry_zone, stop_price, direction)

        # Calculate risk/reward
        entry_mid = (entry_zone[0] + entry_zone[1]) / 2
        if direction == "LONG":
            risk = entry_mid - stop_price
            reward = tp_targets[0]["price"] - entry_mid if tp_targets else 0
        else:
            risk = stop_price - entry_mid
            reward = entry_mid - tp_targets[0]["price"] if tp_targets else 0

        risk_reward = reward / risk if risk > 0 else 0

        # Determine entry type
        current_price = view.key_levels.get("current_price", entry_mid)
        if direction == "LONG":
            entry_type = "LIMIT" if current_price > entry_zone[0] else "MARKET"
        else:
            entry_type = "LIMIT" if current_price < entry_zone[1] else "MARKET"

        return TradeSetup(
            symbol=view.symbol,
            direction=direction,
            entry_type=entry_type,
            entry_zone=entry_zone,
            stop_price=stop_price,
            invalidation_price=invalidation,
            tp_targets=tp_targets,
            confluence_score=confluence_score,
            sub_scores=sub_scores,
            reasons=reasons[:5],  # Top 5 reasons
            reason_codes=reason_codes[:10],  # Top 10 codes
            risk_reward=risk_reward,
            atr_distance=abs(entry_mid - stop_price) / atr if atr > 0 else 0,
            timestamp=view.timestamp,
            timeframe=view.timeframe,
        )

    def _score_strength(
        self,
        view: MarketView,
        direction: str
    ) -> Tuple[float, List[str]]:
        """Score based on currency strength."""
        if not view.strength:
            return 50.0, []

        reasons = []
        score = 50.0

        # Get pair bias from strength
        from analysis.strength import StrengthEngine
        engine = StrengthEngine()
        bias, confidence = engine.get_pair_bias(view.symbol, view.strength.strengths)

        if bias == direction:
            score = 70 + confidence * 30
            if direction == "LONG":
                reasons.append("Base currency stronger than quote")
            else:
                reasons.append("Quote currency stronger than base")
        elif bias == "NEUTRAL":
            score = 50.0
        else:
            score = 30 - confidence * 20
            reasons.append("Strength conflicts with direction")

        return score, reasons

    def _score_structure(
        self,
        view: MarketView,
        direction: str
    ) -> Tuple[float, List[str]]:
        """Score based on market structure."""
        if not view.structure:
            return 50.0, []

        reasons = []
        score = 50.0

        struct = view.structure

        # Trend alignment
        if direction == "LONG":
            if struct.trend_direction == TrendDirection.UP:
                score += 30
                reasons.append("Structure in uptrend")
            elif struct.state in [StructureState.ACCUMULATION, StructureState.RANGE]:
                score += 15
                reasons.append("Potential accumulation zone")
            elif struct.trend_direction == TrendDirection.DOWN:
                score -= 20
                reasons.append("Counter-trend setup (risky)")

            # Break of structure
            if struct.recent_bos and struct.recent_bos.direction == TrendDirection.UP:
                score += 20
                reasons.append("Recent bullish BOS")

        else:  # SHORT
            if struct.trend_direction == TrendDirection.DOWN:
                score += 30
                reasons.append("Structure in downtrend")
            elif struct.state in [StructureState.DISTRIBUTION, StructureState.RANGE]:
                score += 15
                reasons.append("Potential distribution zone")
            elif struct.trend_direction == TrendDirection.UP:
                score -= 20
                reasons.append("Counter-trend setup (risky)")

            if struct.recent_bos and struct.recent_bos.direction == TrendDirection.DOWN:
                score += 20
                reasons.append("Recent bearish BOS")

        return min(100, max(0, score)), reasons

    def _score_liquidity(
        self,
        view: MarketView,
        direction: str
    ) -> Tuple[float, List[str]]:
        """Score based on liquidity analysis."""
        if not view.liquidity:
            return 50.0, []

        reasons = []
        score = 50.0

        liq = view.liquidity

        # Check for recent sweeps in the right direction
        for sweep in liq.recent_sweeps:
            if direction == "LONG" and sweep.sweep_type == SweepType.SELL_SIDE:
                score += 25 * sweep.quality_score
                reasons.append(f"Sell-side sweep at {sweep.zone.price:.5f}")
            elif direction == "SHORT" and sweep.sweep_type == SweepType.BUY_SIDE:
                score += 25 * sweep.quality_score
                reasons.append(f"Buy-side sweep at {sweep.zone.price:.5f}")

        # Check for entry zone availability
        if direction == "LONG" and liq.long_entry_zone:
            score += 15
            reasons.append("Optimal long entry zone identified")
        elif direction == "SHORT" and liq.short_entry_zone:
            score += 15
            reasons.append("Optimal short entry zone identified")

        return min(100, max(0, score)), reasons

    def _score_momentum(
        self,
        view: MarketView,
        direction: str
    ) -> Tuple[float, List[str]]:
        """Score based on momentum analysis."""
        if not view.momentum:
            return 50.0, []

        reasons = []
        score = 50.0
        mom = view.momentum

        # Momentum bias alignment
        if direction == "LONG":
            if mom.bias == MomentumBias.BULLISH:
                score += 20
                reasons.append("Bullish momentum")
            elif mom.bias == MomentumBias.BEARISH:
                score -= 15
                reasons.append("Momentum conflicts")

            # Divergence
            if mom.divergence == DivergenceType.BULLISH_REGULAR:
                score += 20
                reasons.append("Bullish RSI divergence")

            # RSI
            if mom.rsi < 30:
                score += 15
                reasons.append("RSI oversold")
            elif mom.rsi > 70:
                score -= 10

        else:  # SHORT
            if mom.bias == MomentumBias.BEARISH:
                score += 20
                reasons.append("Bearish momentum")
            elif mom.bias == MomentumBias.BULLISH:
                score -= 15
                reasons.append("Momentum conflicts")

            if mom.divergence == DivergenceType.BEARISH_REGULAR:
                score += 20
                reasons.append("Bearish RSI divergence")

            if mom.rsi > 70:
                score += 15
                reasons.append("RSI overbought")
            elif mom.rsi < 30:
                score -= 10

        # Apply momentum confidence
        score = 50 + (score - 50) * mom.confidence

        return min(100, max(0, score)), reasons

    def _score_regime(self, view: MarketView, direction: str) -> float:
        """Score based on market regime.

        Evaluates:
        - Volatility environment (trending vs ranging)
        - Session timing
        - Overall market conditions
        """
        from datetime import datetime

        score = 50.0

        # Evaluate volatility regime
        if view.momentum and view.momentum.atr_percentile:
            atr_pct = view.momentum.atr_percentile

            # For trending plays, higher volatility is better
            if view.structure and view.structure.trend_direction:
                trend = str(view.structure.trend_direction.value).upper()
                is_with_trend = (
                    (direction == "LONG" and trend == "UP") or
                    (direction == "SHORT" and trend == "DOWN")
                )
                if is_with_trend:
                    # Higher vol better for trend trades
                    if atr_pct > 0.5:
                        score += 15
                    elif atr_pct < 0.3:
                        score -= 10  # Low vol not ideal for trend trades
                else:
                    # Lower vol better for counter-trend/reversal
                    if atr_pct < 0.3:
                        score += 10
                    elif atr_pct > 0.7:
                        score -= 15  # High vol risky for reversals

        # Session timing score
        hour = datetime.now().hour
        symbol = view.symbol

        # Determine if current session is favorable
        is_major_session = 7 <= hour < 21  # London + NY
        is_overlap = 12 <= hour < 16  # London-NY overlap

        if is_overlap:
            score += 10  # Best liquidity
        elif is_major_session:
            score += 5
        else:
            score -= 5  # Off-hours

        # Pair-session alignment
        if "JPY" in symbol and 0 <= hour < 9:
            score += 5  # JPY in Asian session
        elif ("EUR" in symbol or "GBP" in symbol) and 7 <= hour < 16:
            score += 5  # EUR/GBP in London
        elif "USD" in symbol and 12 <= hour < 21:
            score += 5  # USD in NY

        return min(100, max(0, score))

    def _calculate_dynamic_stop_multiplier(self, view: MarketView) -> float:
        """Calculate dynamic stop loss multiplier based on market conditions.

        Returns:
            Multiplier for ATR-based stop distance (typically 1.0 - 2.5)
        """
        from datetime import datetime

        base_multiplier = 1.5  # Base stop at 1.5 ATR

        # Adjust for volatility regime
        if view.momentum and view.momentum.atr_percentile:
            atr_pct = view.momentum.atr_percentile
            if atr_pct > 0.75:
                # High volatility - wider stops
                base_multiplier *= 1.3
            elif atr_pct < 0.25:
                # Low volatility - tighter stops
                base_multiplier *= 0.8

        # Adjust for market structure
        if view.structure:
            state = str(view.structure.state.value).lower() if view.structure.state else ""
            if state == "trending":
                # Trending markets - can use tighter stops in direction
                base_multiplier *= 0.9
            elif state == "ranging":
                # Ranging markets - wider stops to avoid whipsaw
                base_multiplier *= 1.2
            elif state == "breakout":
                # Breakouts - standard
                pass

        # Adjust for time of day (session volatility)
        hour = datetime.now().hour
        # London/NY overlap (12-16 UTC) - highest volatility
        if 12 <= hour < 16:
            base_multiplier *= 1.1
        # Asian session (0-7 UTC) - lower volatility
        elif 0 <= hour < 7:
            base_multiplier *= 0.9
        # Session opens (7-8 UTC London, 12-13 UTC NY)
        elif hour in [7, 8, 12, 13]:
            base_multiplier *= 1.15

        # Clamp to reasonable range
        return max(1.0, min(2.5, base_multiplier))

    def _calculate_levels(
        self,
        view: MarketView,
        direction: str,
        atr: float
    ) -> Tuple[Optional[Tuple[float, float]], float, float]:
        """Calculate entry zone, stop, and invalidation levels."""
        levels = view.key_levels
        current = levels.get("current_price", 0)

        if current == 0:
            return None, 0, 0

        # Get dynamic stop multiplier
        stop_multiplier = self._calculate_dynamic_stop_multiplier(view)

        if direction == "LONG":
            # Entry zone
            if "long_entry_low" in levels and "long_entry_high" in levels:
                entry_zone = (levels["long_entry_low"], levels["long_entry_high"])
            else:
                # Default: current price to 0.5 ATR below
                atr_buffer = atr * 0.5 if atr > 0 else current * 0.002
                entry_zone = (current - atr_buffer, current)

            # Stop and invalidation - ensure invalidation is BELOW entry for LONG
            has_valid_invalidation = (
                "bullish_invalidation" in levels and
                levels["bullish_invalidation"] < entry_zone[0]
            )

            if has_valid_invalidation:
                invalidation = levels["bullish_invalidation"]
                stop_buffer = atr * 0.2 if atr > 0 else abs(entry_zone[0] - invalidation) * 0.1
                stop_price = invalidation - abs(stop_buffer)
            else:
                # Dynamic stop buffer based on market conditions
                stop_buffer = atr * stop_multiplier if atr > 0 else current * 0.01
                stop_price = entry_zone[0] - abs(stop_buffer)
                invalidation = stop_price

            # Final validation: stop MUST be below entry
            if stop_price >= entry_zone[0]:
                stop_price = entry_zone[0] - (current * 0.005)  # Fallback: 0.5% below
                invalidation = stop_price

        else:  # SHORT
            if "short_entry_low" in levels and "short_entry_high" in levels:
                entry_zone = (levels["short_entry_low"], levels["short_entry_high"])
            else:
                atr_buffer = atr * 0.5 if atr > 0 else current * 0.002
                entry_zone = (current, current + atr_buffer)

            # Stop and invalidation - ensure invalidation is ABOVE entry for SHORT
            has_valid_invalidation = (
                "bearish_invalidation" in levels and
                levels["bearish_invalidation"] > entry_zone[1]
            )

            if has_valid_invalidation:
                invalidation = levels["bearish_invalidation"]
                stop_buffer = atr * 0.2 if atr > 0 else abs(invalidation - entry_zone[1]) * 0.1
                stop_price = invalidation + abs(stop_buffer)
            else:
                # Dynamic stop buffer based on market conditions
                stop_buffer = atr * stop_multiplier if atr > 0 else current * 0.01
                stop_price = entry_zone[1] + abs(stop_buffer)
                invalidation = stop_price

            # Final validation: stop MUST be above entry
            if stop_price <= entry_zone[1]:
                stop_price = entry_zone[1] + (current * 0.005)  # Fallback: 0.5% above
                invalidation = stop_price

        return entry_zone, stop_price, invalidation

    def _calculate_targets(
        self,
        entry_zone: Tuple[float, float],
        stop_price: float,
        direction: str
    ) -> List[Dict]:
        """Calculate take profit targets based on R-multiples."""
        entry_mid = (entry_zone[0] + entry_zone[1]) / 2
        risk = abs(entry_mid - stop_price)

        targets = []
        r_multiples = [1.5, 2.5, 4.0]

        for r in r_multiples:
            if direction == "LONG":
                tp_price = entry_mid + risk * r
            else:
                tp_price = entry_mid - risk * r

            targets.append({
                "r": r,
                "price": round(tp_price, 5),
            })

        return targets

    def _get_strength_codes(self, view: MarketView, direction: str) -> List[ReasonCode]:
        """Get reason codes for strength analysis."""
        codes = []
        if view.directional_bias == direction:
            if direction == "LONG":
                codes.append(ReasonCode.STR_STRONG_BASE)
            else:
                codes.append(ReasonCode.STR_WEAK_QUOTE)
        return codes

    def _get_structure_codes(self, view: MarketView, direction: str) -> List[ReasonCode]:
        """Get reason codes for structure analysis."""
        codes = []
        if not view.structure:
            return codes

        if direction == "LONG":
            if view.structure.trend_direction == TrendDirection.UP:
                codes.append(ReasonCode.STRUCT_TREND_UP)
                codes.append(ReasonCode.STRUCT_HH_HL)
            if view.structure.recent_bos and view.structure.recent_bos.direction == TrendDirection.UP:
                codes.append(ReasonCode.STRUCT_BOS_BULL)
        else:
            if view.structure.trend_direction == TrendDirection.DOWN:
                codes.append(ReasonCode.STRUCT_TREND_DOWN)
                codes.append(ReasonCode.STRUCT_LH_LL)
            if view.structure.recent_bos and view.structure.recent_bos.direction == TrendDirection.DOWN:
                codes.append(ReasonCode.STRUCT_BOS_BEAR)

        return codes

    def _get_liquidity_codes(self, view: MarketView, direction: str) -> List[ReasonCode]:
        """Get reason codes for liquidity analysis."""
        codes = []
        if not view.liquidity:
            return codes

        for sweep in view.liquidity.recent_sweeps:
            if direction == "LONG" and sweep.sweep_type == SweepType.SELL_SIDE:
                codes.append(ReasonCode.LIQ_SWEEP_SELL)
                if sweep.zone.zone_type == ZoneType.PDL:
                    codes.append(ReasonCode.LIQ_PDL_REJECT)
                elif sweep.zone.zone_type == ZoneType.EQUAL_LOWS:
                    codes.append(ReasonCode.LIQ_EQUAL_LOWS_SWEPT)
            elif direction == "SHORT" and sweep.sweep_type == SweepType.BUY_SIDE:
                codes.append(ReasonCode.LIQ_SWEEP_BUY)
                if sweep.zone.zone_type == ZoneType.PDH:
                    codes.append(ReasonCode.LIQ_PDH_REJECT)
                elif sweep.zone.zone_type == ZoneType.EQUAL_HIGHS:
                    codes.append(ReasonCode.LIQ_EQUAL_HIGHS_SWEPT)

        return codes

    def _get_momentum_codes(self, view: MarketView, direction: str) -> List[ReasonCode]:
        """Get reason codes for momentum analysis."""
        codes = []
        if not view.momentum:
            return codes

        mom = view.momentum
        if direction == "LONG":
            if mom.rsi < 30:
                codes.append(ReasonCode.MOM_RSI_OVERSOLD)
            if mom.divergence == DivergenceType.BULLISH_REGULAR:
                codes.append(ReasonCode.MOM_BULL_DIV)
        else:
            if mom.rsi > 70:
                codes.append(ReasonCode.MOM_RSI_OVERBOUGHT)
            if mom.divergence == DivergenceType.BEARISH_REGULAR:
                codes.append(ReasonCode.MOM_BEAR_DIV)

        return codes
