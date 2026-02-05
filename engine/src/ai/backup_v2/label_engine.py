"""Label Engine - Generates training labels from historical candidates.

CRITICAL: This engine MUST match live execution logic exactly.
Any deviation will cause train/live mismatch and model failure.

Labels generated:
- Classification: WIN (1), LOSS (0), TIMEOUT (2)
- Regression: R-multiple (realized PnL / risk)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class TradeOutcome(Enum):
    """Trade outcome classification."""
    WIN = 1      # TP hit before SL within horizon
    LOSS = 0     # SL hit before TP
    TIMEOUT = 2  # Neither hit within horizon


@dataclass
class ExecutionConfig:
    """Execution parameters - MUST match live trading config.

    These values are critical for label accuracy.
    """
    # Entry mechanics
    entry_type: str = "market"  # "market" or "limit"
    entry_slippage_pips: float = 0.5  # Expected slippage on market orders

    # Costs (must match your broker)
    spread_pips: float = 1.2  # Average spread for majors
    commission_per_lot: float = 0.0  # If applicable

    # Trade management
    default_tp_r: float = 2.0  # Default TP in R-multiples
    timeout_bars: int = 96  # Max bars before timeout (96 = 24h on M15)

    # Position sizing (for cost calculation)
    pip_value_per_lot: float = 10.0  # USD per pip per lot


@dataclass
class CandidateLabel:
    """Complete label for a trade candidate."""
    # Identification
    candidate_id: str
    symbol: str
    direction: str
    signal_time: datetime

    # Entry/Exit details
    entry_price: float
    stop_price: float
    tp_price: float
    exit_price: float
    exit_time: datetime

    # Classification label
    outcome: TradeOutcome
    outcome_class: int  # 0=LOSS, 1=WIN, 2=TIMEOUT

    # Regression label
    r_multiple: float  # Realized R (can be negative)
    pnl_pips: float

    # Execution details
    bars_to_exit: int
    total_cost_pips: float

    # Quality metrics (for analysis, not training)
    max_adverse_excursion: float  # MAE in pips
    max_favorable_excursion: float  # MFE in pips

    # Versioning
    label_engine_version: str = "1.0.0"


class LabelEngine:
    """Generates labels by simulating trade execution on historical data.

    IMPORTANT: This class defines the "ground truth" for training.
    Every parameter here affects what the model learns.

    Example:
        engine = LabelEngine(config)
        label = engine.generate_label(candidate, future_candles)
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize label engine.

        Args:
            config: Execution configuration matching live trading
        """
        self.config = config or ExecutionConfig()

    def generate_label(
        self,
        candidate_id: str,
        symbol: str,
        direction: str,  # "LONG" or "SHORT"
        signal_time: datetime,
        entry_price: float,
        stop_price: float,
        tp_targets: List[Dict],  # [{"r": 1.5, "price": 1.0950}, ...]
        future_candles: List[Dict],  # [{"open": x, "high": x, "low": x, "close": x, "time": dt}, ...]
    ) -> CandidateLabel:
        """Generate label for a single candidate.

        Args:
            candidate_id: Unique candidate identifier
            symbol: Trading symbol
            direction: Trade direction
            signal_time: Time when signal was generated
            entry_price: Planned entry price
            stop_price: Stop loss price
            tp_targets: Take profit targets
            future_candles: Candles AFTER signal time for simulation

        Returns:
            CandidateLabel with outcome and R-multiple
        """
        # Calculate actual entry with costs
        actual_entry = self._calculate_entry_price(entry_price, direction)

        # Get TP price (use first target or calculate from R)
        tp_price = self._get_tp_price(tp_targets, actual_entry, stop_price, direction)

        # Calculate risk (distance to stop)
        if direction == "LONG":
            risk_pips = (actual_entry - stop_price) / 0.0001
        else:
            risk_pips = (stop_price - actual_entry) / 0.0001

        if risk_pips <= 0:
            logger.warning(f"Invalid risk: {risk_pips} pips for {symbol}")
            risk_pips = 1.0  # Prevent division by zero

        # Simulate trade through future candles
        outcome, exit_price, exit_bar, mae, mfe = self._simulate_trade(
            direction=direction,
            entry_price=actual_entry,
            stop_price=stop_price,
            tp_price=tp_price,
            future_candles=future_candles,
        )

        # Calculate exit time
        if exit_bar < len(future_candles):
            exit_time = future_candles[exit_bar].get("time", signal_time + timedelta(minutes=15 * exit_bar))
        else:
            exit_time = signal_time + timedelta(minutes=15 * len(future_candles))

        # Calculate realized P&L
        if direction == "LONG":
            pnl_pips = (exit_price - actual_entry) / 0.0001
        else:
            pnl_pips = (actual_entry - exit_price) / 0.0001

        # Subtract costs
        total_cost = self.config.spread_pips + self.config.entry_slippage_pips
        pnl_pips -= total_cost

        # Calculate R-multiple
        r_multiple = pnl_pips / risk_pips

        return CandidateLabel(
            candidate_id=candidate_id,
            symbol=symbol,
            direction=direction,
            signal_time=signal_time,
            entry_price=actual_entry,
            stop_price=stop_price,
            tp_price=tp_price,
            exit_price=exit_price,
            exit_time=exit_time,
            outcome=outcome,
            outcome_class=outcome.value,
            r_multiple=r_multiple,
            pnl_pips=pnl_pips,
            bars_to_exit=exit_bar + 1,
            total_cost_pips=total_cost,
            max_adverse_excursion=mae,
            max_favorable_excursion=mfe,
            label_engine_version=self.VERSION,
        )

    def _calculate_entry_price(self, price: float, direction: str) -> float:
        """Calculate actual entry price including spread and slippage.

        For LONG: we buy at ask (price + half spread)
        For SHORT: we sell at bid (price - half spread)
        Plus slippage on market orders.
        """
        half_spread = (self.config.spread_pips / 2) * 0.0001
        slippage = self.config.entry_slippage_pips * 0.0001

        if self.config.entry_type == "market":
            if direction == "LONG":
                return price + half_spread + slippage
            else:
                return price - half_spread - slippage
        else:
            # Limit orders - assume fill at limit price
            return price

    def _get_tp_price(
        self,
        tp_targets: List[Dict],
        entry: float,
        stop: float,
        direction: str,
    ) -> float:
        """Get take profit price from targets or calculate from R."""
        if tp_targets and len(tp_targets) > 0:
            return tp_targets[0].get("price", 0)

        # Calculate from default R
        risk = abs(entry - stop)
        reward = risk * self.config.default_tp_r

        if direction == "LONG":
            return entry + reward
        else:
            return entry - reward

    def _simulate_trade(
        self,
        direction: str,
        entry_price: float,
        stop_price: float,
        tp_price: float,
        future_candles: List[Dict],
    ) -> Tuple[TradeOutcome, float, int, float, float]:
        """Simulate trade through future candles.

        Returns:
            Tuple of (outcome, exit_price, exit_bar_index, mae, mfe)
        """
        mae = 0.0  # Max adverse excursion (pips)
        mfe = 0.0  # Max favorable excursion (pips)

        for bar_idx, candle in enumerate(future_candles):
            if bar_idx >= self.config.timeout_bars:
                break

            high = candle.get("high", candle.get("h", 0))
            low = candle.get("low", candle.get("l", 0))
            close = candle.get("close", candle.get("c", 0))

            if direction == "LONG":
                # Check stop first (worst-case assumption)
                if low <= stop_price:
                    return TradeOutcome.LOSS, stop_price, bar_idx, mae, mfe

                # Check TP
                if high >= tp_price:
                    return TradeOutcome.WIN, tp_price, bar_idx, mae, mfe

                # Track excursions
                adverse = (entry_price - low) / 0.0001
                favorable = (high - entry_price) / 0.0001
                mae = max(mae, adverse)
                mfe = max(mfe, favorable)

            else:  # SHORT
                # Check stop first
                if high >= stop_price:
                    return TradeOutcome.LOSS, stop_price, bar_idx, mae, mfe

                # Check TP
                if low <= tp_price:
                    return TradeOutcome.WIN, tp_price, bar_idx, mae, mfe

                # Track excursions
                adverse = (high - entry_price) / 0.0001
                favorable = (entry_price - low) / 0.0001
                mae = max(mae, adverse)
                mfe = max(mfe, favorable)

        # Timeout - exit at last available close
        if future_candles:
            last_close = future_candles[-1].get("close", future_candles[-1].get("c", entry_price))
            return TradeOutcome.TIMEOUT, last_close, len(future_candles) - 1, mae, mfe

        return TradeOutcome.TIMEOUT, entry_price, 0, mae, mfe

    def generate_labels_batch(
        self,
        candidates: List[Dict],
        price_data: Dict[str, List[Dict]],  # symbol -> candles
    ) -> List[CandidateLabel]:
        """Generate labels for a batch of candidates.

        Args:
            candidates: List of candidate dicts
            price_data: Price data keyed by symbol

        Returns:
            List of CandidateLabel objects
        """
        labels = []

        for candidate in candidates:
            symbol = candidate["symbol"]
            signal_time = candidate["signal_time"]

            # Get future candles after signal time
            all_candles = price_data.get(symbol, [])
            future_candles = [
                c for c in all_candles
                if c.get("time", c.get("t")) > signal_time
            ][:self.config.timeout_bars]

            if len(future_candles) < 10:  # Need some future data
                continue

            label = self.generate_label(
                candidate_id=candidate["id"],
                symbol=symbol,
                direction=candidate["direction"],
                signal_time=signal_time,
                entry_price=candidate["entry_price"],
                stop_price=candidate["stop_price"],
                tp_targets=candidate.get("tp_targets", []),
                future_candles=future_candles,
            )

            labels.append(label)

        return labels
