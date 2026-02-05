"""Advanced Trade Management - Real-time position monitoring and scaling.

Features:
- Trade health monitoring (momentum, structure, divergence analysis)
- Position scaling (pyramid into strong trends)
- Dynamic take profit adjustment
- Early exit on reversal signals
- Aggressive position sizing options
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class TradeHealth(Enum):
    """Trade health status."""
    STRONG = "strong"           # Momentum aligned, structure intact
    HEALTHY = "healthy"         # Normal conditions
    WEAKENING = "weakening"     # Momentum fading, minor concerns
    CRITICAL = "critical"       # Reversal signals, exit recommended
    UNKNOWN = "unknown"


class ScaleAction(Enum):
    """Scaling action recommendation."""
    ADD = "add"                 # Add to position
    HOLD = "hold"               # Maintain current position
    REDUCE = "reduce"           # Partial close recommended
    EXIT = "exit"               # Full exit recommended


@dataclass
class TradeHealthReport:
    """Comprehensive trade health analysis."""
    health: TradeHealth
    score: float  # 0-100, higher = healthier
    momentum_score: float
    structure_score: float
    divergence_detected: bool
    reversal_signals: List[str]
    continuation_signals: List[str]
    recommended_action: ScaleAction
    tp_adjustment: Optional[float]  # Suggested TP change (+ = extend, - = tighten)
    reasons: List[str]


@dataclass
class ScalingConfig:
    """Configuration for position scaling."""
    # Scaling thresholds
    min_r_to_scale: float = 1.5          # Minimum R profit before scaling
    max_scale_count: int = 2              # Maximum times to add to position
    scale_size_percent: float = 0.5       # Size of add relative to original

    # Health requirements for scaling
    min_health_score: float = 70.0        # Minimum health score to scale
    require_momentum_alignment: bool = True

    # TP adjustment
    enable_dynamic_tp: bool = True
    tp_extension_per_scale: float = 0.5   # Extend TP by 0.5R per scale
    max_tp_extension: float = 2.0         # Maximum additional R for TP

    # Exit triggers
    exit_on_divergence: bool = True
    exit_health_threshold: float = 30.0   # Exit if health drops below
    partial_exit_health: float = 50.0     # Partial exit threshold


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""
    # Risk per trade
    base_risk_percent: float = 0.01       # 1% base risk
    aggressive_risk_percent: float = 0.02  # 2% aggressive risk
    max_risk_percent: float = 0.03        # 3% maximum

    # Lot size limits
    min_lots: float = 0.01
    max_lots: float = 0.50                # Higher max for aggressive trading

    # Confidence-based sizing
    enable_confidence_scaling: bool = True
    high_confidence_multiplier: float = 1.5  # 1.5x for high confidence trades
    confidence_threshold: float = 0.80       # P(win) threshold for boost

    # Account-based limits
    max_account_risk: float = 0.10        # Max 10% total account risk


class TradeHealthMonitor:
    """Monitors trade health using real-time market analysis."""

    def __init__(self, analyzer, scorer):
        """Initialize health monitor.

        Args:
            analyzer: MarketAnalyzer instance
            scorer: ConfluenceScorer instance
        """
        self.analyzer = analyzer
        self.scorer = scorer
        self._health_cache: Dict[str, Tuple[TradeHealthReport, datetime]] = {}
        self._cache_duration = timedelta(seconds=30)

    def analyze_trade_health(
        self,
        symbol: str,
        direction: str,  # "buy" or "sell"
        entry_price: float,
        current_price: float,
        candles: List,  # Recent candles
    ) -> TradeHealthReport:
        """Analyze the health of an open trade.

        Args:
            symbol: Trading symbol
            direction: Trade direction
            entry_price: Original entry price
            current_price: Current market price
            candles: Recent price candles

        Returns:
            TradeHealthReport with analysis
        """
        # Check cache
        cache_key = f"{symbol}_{direction}"
        if cache_key in self._health_cache:
            cached, timestamp = self._health_cache[cache_key]
            if datetime.now() - timestamp < self._cache_duration:
                return cached

        reversal_signals = []
        continuation_signals = []

        # Extract OHLC
        opens = np.array([float(c.open) for c in candles])
        highs = np.array([float(c.high) for c in candles])
        lows = np.array([float(c.low) for c in candles])
        closes = np.array([float(c.close) for c in candles])
        timestamps = [c.timestamp for c in candles]

        # Get market analysis
        market_view = self.analyzer.analyze(
            symbol, "M15", opens, highs, lows, closes, timestamps
        )

        # Normalize direction
        trade_direction = "LONG" if direction.lower() == "buy" else "SHORT"

        # === Momentum Analysis ===
        momentum_score = self._analyze_momentum(market_view, trade_direction)

        # === Structure Analysis ===
        structure_score = self._analyze_structure(market_view, trade_direction)

        # === Divergence Detection ===
        divergence_detected = self._detect_divergence(market_view, trade_direction)
        if divergence_detected:
            reversal_signals.append("Momentum divergence detected")

        # === Price Action Signals ===
        pa_signals = self._analyze_price_action(
            highs, lows, closes, trade_direction, entry_price, current_price
        )
        reversal_signals.extend(pa_signals["reversal"])
        continuation_signals.extend(pa_signals["continuation"])

        # === Calculate Overall Health ===
        health_score = (momentum_score * 0.4 + structure_score * 0.4 +
                       (0 if divergence_detected else 20))

        # Adjust for reversal signals
        health_score -= len(reversal_signals) * 10
        health_score += len(continuation_signals) * 5
        health_score = max(0, min(100, health_score))

        # Determine health status
        if health_score >= 75:
            health = TradeHealth.STRONG
        elif health_score >= 55:
            health = TradeHealth.HEALTHY
        elif health_score >= 35:
            health = TradeHealth.WEAKENING
        else:
            health = TradeHealth.CRITICAL

        # Determine recommended action
        action, tp_adjustment = self._determine_action(
            health_score, momentum_score, divergence_detected,
            len(reversal_signals), len(continuation_signals)
        )

        # Build reasons
        reasons = []
        if momentum_score >= 70:
            reasons.append(f"Strong momentum alignment ({momentum_score:.0f})")
        elif momentum_score < 40:
            reasons.append(f"Weak momentum ({momentum_score:.0f})")

        if structure_score >= 70:
            reasons.append(f"Structure supports trade ({structure_score:.0f})")
        elif structure_score < 40:
            reasons.append(f"Structure weakening ({structure_score:.0f})")

        if divergence_detected:
            reasons.append("⚠️ Divergence detected - potential reversal")

        report = TradeHealthReport(
            health=health,
            score=health_score,
            momentum_score=momentum_score,
            structure_score=structure_score,
            divergence_detected=divergence_detected,
            reversal_signals=reversal_signals,
            continuation_signals=continuation_signals,
            recommended_action=action,
            tp_adjustment=tp_adjustment,
            reasons=reasons,
        )

        # Cache result
        self._health_cache[cache_key] = (report, datetime.now())

        return report

    def _analyze_momentum(self, view, direction: str) -> float:
        """Analyze momentum alignment with trade direction."""
        score = 50.0  # Neutral baseline

        if not view.momentum:
            return score

        mom = view.momentum

        # RSI analysis
        if mom.rsi:
            if direction == "LONG":
                if 40 <= mom.rsi <= 70:
                    score += 15  # Healthy bullish RSI
                elif mom.rsi > 70:
                    score -= 10  # Overbought risk
                elif mom.rsi < 40:
                    score -= 15  # Momentum against trade
            else:  # SHORT
                if 30 <= mom.rsi <= 60:
                    score += 15  # Healthy bearish RSI
                elif mom.rsi < 30:
                    score -= 10  # Oversold risk
                elif mom.rsi > 60:
                    score -= 15  # Momentum against trade

        # MACD analysis
        if mom.macd_histogram is not None:
            if direction == "LONG" and mom.macd_histogram > 0:
                score += 10
            elif direction == "SHORT" and mom.macd_histogram < 0:
                score += 10
            else:
                score -= 10

        # Trend strength (ADX if available)
        if hasattr(mom, 'adx') and mom.adx:
            if mom.adx > 25:
                score += 10  # Strong trend
            elif mom.adx < 20:
                score -= 5   # Weak trend

        return max(0, min(100, score))

    def _analyze_structure(self, view, direction: str) -> float:
        """Analyze market structure alignment."""
        score = 50.0

        if not view.structure:
            return score

        struct = view.structure

        # Trend direction
        if struct.trend_direction:
            trend = str(struct.trend_direction.value).upper()
            if direction == "LONG" and trend in ("UP", "BULLISH"):
                score += 25
            elif direction == "SHORT" and trend in ("DOWN", "BEARISH"):
                score += 25
            elif direction == "LONG" and trend in ("DOWN", "BEARISH"):
                score -= 25
            elif direction == "SHORT" and trend in ("UP", "BULLISH"):
                score -= 25

        # Market state
        if struct.state:
            state = str(struct.state.value).lower()
            if "trend" in state:
                score += 10
            elif "rang" in state:
                score -= 5  # Ranging is harder

        # Bias alignment
        if hasattr(struct, 'bias') and struct.bias:
            bias = str(struct.bias).upper()
            if direction == "LONG" and bias == "BULLISH":
                score += 10
            elif direction == "SHORT" and bias == "BEARISH":
                score += 10

        return max(0, min(100, score))

    def _detect_divergence(self, view, direction: str) -> bool:
        """Detect momentum divergence against trade direction."""
        if not view.momentum or not view.momentum.divergence:
            return False

        div = str(view.momentum.divergence.value).upper()

        # Bearish divergence is bad for longs
        if direction == "LONG" and "BEARISH" in div:
            return True
        # Bullish divergence is bad for shorts
        if direction == "SHORT" and "BULLISH" in div:
            return True

        return False

    def _analyze_price_action(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        direction: str,
        entry: float,
        current: float,
    ) -> Dict[str, List[str]]:
        """Analyze recent price action for signals."""
        signals = {"reversal": [], "continuation": []}

        if len(closes) < 10:
            return signals

        recent_closes = closes[-10:]
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]

        # Check for lower highs / higher lows
        if direction == "LONG":
            # Bad: Making lower highs
            if recent_highs[-1] < recent_highs[-3] < recent_highs[-5]:
                signals["reversal"].append("Lower highs forming")
            # Good: Making higher lows
            if recent_lows[-1] > recent_lows[-3] > recent_lows[-5]:
                signals["continuation"].append("Higher lows intact")
            # Bad: Broke below recent swing low
            swing_low = min(recent_lows[-5:])
            if current < swing_low:
                signals["reversal"].append("Broke below swing low")
        else:  # SHORT
            # Bad: Making higher lows
            if recent_lows[-1] > recent_lows[-3] > recent_lows[-5]:
                signals["reversal"].append("Higher lows forming")
            # Good: Making lower highs
            if recent_highs[-1] < recent_highs[-3] < recent_highs[-5]:
                signals["continuation"].append("Lower highs intact")
            # Bad: Broke above recent swing high
            swing_high = max(recent_highs[-5:])
            if current > swing_high:
                signals["reversal"].append("Broke above swing high")

        # Check momentum of recent candles
        recent_momentum = closes[-1] - closes[-5]
        if direction == "LONG" and recent_momentum > 0:
            signals["continuation"].append("Positive recent momentum")
        elif direction == "SHORT" and recent_momentum < 0:
            signals["continuation"].append("Negative recent momentum")
        elif direction == "LONG" and recent_momentum < 0:
            signals["reversal"].append("Momentum turning negative")
        elif direction == "SHORT" and recent_momentum > 0:
            signals["reversal"].append("Momentum turning positive")

        return signals

    def _determine_action(
        self,
        health_score: float,
        momentum_score: float,
        divergence: bool,
        reversal_count: int,
        continuation_count: int,
    ) -> Tuple[ScaleAction, Optional[float]]:
        """Determine recommended action based on analysis."""
        tp_adjustment = None

        # Critical health - exit
        if health_score < 30 or (divergence and reversal_count >= 2):
            return ScaleAction.EXIT, -0.5  # Tighten TP

        # Weakening - reduce or hold
        if health_score < 50 or divergence:
            return ScaleAction.REDUCE, -0.25

        # Healthy - hold
        if health_score < 70:
            return ScaleAction.HOLD, None

        # Strong - consider adding
        if health_score >= 75 and momentum_score >= 70 and continuation_count >= 2:
            return ScaleAction.ADD, 0.5  # Extend TP

        return ScaleAction.HOLD, 0.25  # Slight TP extension


class PositionScaler:
    """Manages position scaling (pyramiding) for winning trades."""

    def __init__(
        self,
        broker,
        health_monitor: TradeHealthMonitor,
        config: Optional[ScalingConfig] = None,
    ):
        """Initialize position scaler.

        Args:
            broker: Broker adapter for order execution
            health_monitor: Trade health monitor
            config: Scaling configuration
        """
        self.broker = broker
        self.health_monitor = health_monitor
        self.config = config or ScalingConfig()

        # Track scaling state per position
        self._scale_state: Dict[int, Dict] = {}  # position_id -> state

    def get_scale_state(self, position_id: int) -> Dict:
        """Get or create scale state for position."""
        if position_id not in self._scale_state:
            self._scale_state[position_id] = {
                "original_size": None,
                "scale_count": 0,
                "total_size": None,
                "scale_prices": [],
                "original_tp": None,
                "current_tp": None,
            }
        return self._scale_state[position_id]

    def evaluate_scaling(
        self,
        position,
        candles: List,
        current_r: float,
    ) -> Tuple[bool, Optional[float], str]:
        """Evaluate if position should be scaled.

        Args:
            position: Current position
            candles: Recent candles for analysis
            current_r: Current R-multiple profit

        Returns:
            Tuple of (should_scale, scale_size, reason)
        """
        state = self.get_scale_state(position.position_id)

        # Initialize state if needed
        if state["original_size"] is None:
            state["original_size"] = float(position.quantity)
            state["total_size"] = float(position.quantity)
            state["original_tp"] = float(position.take_profit) if position.take_profit else None
            state["current_tp"] = state["original_tp"]

        # Check if max scales reached
        if state["scale_count"] >= self.config.max_scale_count:
            return False, None, "Max scale count reached"

        # Check minimum R requirement
        if current_r < self.config.min_r_to_scale:
            return False, None, f"Need {self.config.min_r_to_scale}R to scale (current: {current_r:.2f}R)"

        # Check trade health
        health_report = self.health_monitor.analyze_trade_health(
            symbol=position.symbol,
            direction=position.side,
            entry_price=float(position.avg_price),
            current_price=float(position.current_price),
            candles=candles,
        )

        if health_report.score < self.config.min_health_score:
            return False, None, f"Health score too low ({health_report.score:.0f})"

        if health_report.recommended_action != ScaleAction.ADD:
            return False, None, f"Health recommends {health_report.recommended_action.value}"

        # Calculate scale size
        scale_size = state["original_size"] * self.config.scale_size_percent
        scale_size = round(scale_size, 2)

        return True, scale_size, f"Strong trend, health={health_report.score:.0f}"

    def execute_scale(
        self,
        position,
        scale_size: float,
        current_price: float,
    ) -> bool:
        """Execute a scale-in order.

        Args:
            position: Position to scale into
            scale_size: Size to add
            current_price: Current market price

        Returns:
            True if scale order placed successfully
        """
        from src.adapters.broker import OrderRequest

        state = self.get_scale_state(position.position_id)

        try:
            # Place market order in same direction
            order_request = OrderRequest(
                symbol=position.symbol,
                side=position.side,
                quantity=Decimal(str(scale_size)),
                order_type="market",
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
            )

            result = self.broker.place_order(order_request)

            if result and result.order_id:
                # Update state
                state["scale_count"] += 1
                state["total_size"] += scale_size
                state["scale_prices"].append(current_price)

                logger.info(
                    "Position scaled",
                    symbol=position.symbol,
                    scale_count=state["scale_count"],
                    added_size=scale_size,
                    total_size=state["total_size"],
                )
                return True

        except Exception as e:
            logger.error(f"Scale order failed: {e}")

        return False

    def adjust_take_profit(
        self,
        position,
        health_report: TradeHealthReport,
    ) -> Optional[float]:
        """Calculate adjusted take profit based on health.

        Args:
            position: Current position
            health_report: Health analysis

        Returns:
            New TP price if adjustment recommended, None otherwise
        """
        if not self.config.enable_dynamic_tp:
            return None

        if health_report.tp_adjustment is None:
            return None

        state = self.get_scale_state(position.position_id)

        if state["original_tp"] is None:
            return None

        # Calculate new TP
        entry = float(position.avg_price)
        original_tp = state["original_tp"]
        pip_size = 0.01 if "JPY" in position.symbol else 0.0001

        # Calculate original TP distance
        if position.side == "buy":
            original_distance = original_tp - entry
        else:
            original_distance = entry - original_tp

        # Calculate adjustment
        adjustment = original_distance * health_report.tp_adjustment

        # Cap adjustment
        max_adjustment = original_distance * self.config.max_tp_extension
        adjustment = max(-original_distance * 0.5, min(max_adjustment, adjustment))

        # Calculate new TP
        if position.side == "buy":
            new_tp = original_tp + adjustment
        else:
            new_tp = original_tp - adjustment

        # Only return if meaningfully different
        if abs(new_tp - state["current_tp"]) > pip_size * 5:
            state["current_tp"] = new_tp
            return new_tp

        return None


class AggressivePositionSizer:
    """Calculates position sizes with aggressive growth options."""

    def __init__(self, config: Optional[PositionSizingConfig] = None):
        """Initialize position sizer.

        Args:
            config: Position sizing configuration
        """
        self.config = config or PositionSizingConfig()

    def calculate_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_price: float,
        symbol: str,
        probability: float = 0.5,
        aggressive_mode: bool = False,
        existing_risk: float = 0.0,
    ) -> Tuple[float, Dict]:
        """Calculate position size with optional aggressive scaling.

        Args:
            account_balance: Current account balance
            entry_price: Planned entry price
            stop_price: Stop loss price
            symbol: Trading symbol
            probability: AI probability (for confidence scaling)
            aggressive_mode: Use aggressive risk settings
            existing_risk: Already committed risk (for account limit)

        Returns:
            Tuple of (lot_size, details_dict)
        """
        details = {
            "base_risk": 0,
            "confidence_multiplier": 1.0,
            "final_risk_percent": 0,
            "stop_pips": 0,
            "raw_lots": 0,
            "capped_lots": 0,
            "reason": "",
        }

        # Determine base risk
        if aggressive_mode:
            base_risk = self.config.aggressive_risk_percent
            details["reason"] = "Aggressive mode"
        else:
            base_risk = self.config.base_risk_percent
            details["reason"] = "Standard mode"

        details["base_risk"] = base_risk

        # Apply confidence scaling
        if self.config.enable_confidence_scaling:
            if probability >= self.config.confidence_threshold:
                multiplier = self.config.high_confidence_multiplier
                details["confidence_multiplier"] = multiplier
                details["reason"] += f" + high confidence ({probability:.1%})"
            else:
                multiplier = 1.0
        else:
            multiplier = 1.0

        final_risk = base_risk * multiplier

        # Cap at max risk
        final_risk = min(final_risk, self.config.max_risk_percent)

        # Check account-level risk limit
        available_risk = self.config.max_account_risk - existing_risk
        if final_risk > available_risk:
            final_risk = max(0, available_risk)
            details["reason"] += " (account risk limited)"

        details["final_risk_percent"] = final_risk

        if final_risk <= 0:
            return 0.0, details

        # Calculate stop distance
        pip_size = 0.01 if "JPY" in symbol else 0.0001
        pip_value = 10.0  # USD per pip per standard lot

        stop_pips = abs(entry_price - stop_price) / pip_size
        details["stop_pips"] = stop_pips

        # Minimum stop distance
        if stop_pips < 5:
            stop_pips = 5
            details["reason"] += " (min stop enforced)"

        # Calculate risk amount
        risk_amount = account_balance * final_risk

        # Calculate lots
        lots = risk_amount / (stop_pips * pip_value)
        details["raw_lots"] = lots

        # Apply limits
        lots = max(self.config.min_lots, min(self.config.max_lots, lots))
        lots = round(lots, 2)
        details["capped_lots"] = lots

        return lots, details


# Convenience function for creating the full advanced management suite
def create_advanced_manager(broker, analyzer, scorer, aggressive: bool = False):
    """Create a complete advanced trade management suite.

    Args:
        broker: Broker adapter
        analyzer: MarketAnalyzer
        scorer: ConfluenceScorer
        aggressive: Use aggressive settings

    Returns:
        Dict with health_monitor, scaler, and sizer
    """
    health_monitor = TradeHealthMonitor(analyzer, scorer)

    scaling_config = ScalingConfig(
        min_r_to_scale=1.0 if aggressive else 1.5,
        max_scale_count=3 if aggressive else 2,
        scale_size_percent=0.75 if aggressive else 0.5,
    )

    sizing_config = PositionSizingConfig(
        base_risk_percent=0.02 if aggressive else 0.01,
        aggressive_risk_percent=0.03 if aggressive else 0.02,
        max_risk_percent=0.05 if aggressive else 0.03,
        max_lots=1.0 if aggressive else 0.50,
    )

    scaler = PositionScaler(broker, health_monitor, scaling_config)
    sizer = AggressivePositionSizer(sizing_config)

    return {
        "health_monitor": health_monitor,
        "scaler": scaler,
        "sizer": sizer,
        "scaling_config": scaling_config,
        "sizing_config": sizing_config,
    }
