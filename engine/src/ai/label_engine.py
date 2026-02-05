"""Label engine for generating trade outcome labels.

Given a candle bar and execution config, simulates a trade entry with
per-symbol spread + slippage, then checks whether SL or TP was hit first
in subsequent bars to produce WIN / LOSS / TIMEOUT labels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

from .utils import get_spread_pips, pip_size, spread_in_price, SPREAD_TABLE

logger = logging.getLogger(__name__)


class TradeOutcome(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    TIMEOUT = "TIMEOUT"


@dataclass
class ExecutionConfig:
    """Execution-cost parameters for label generation.

    Per-symbol spreads are looked up from :data:`SPREAD_TABLE` via
    :meth:`get_spread`.  A flat ``slippage_pips`` is applied on top.
    """

    slippage_pips: float = 0.3
    tp_pips: float = 30.0
    sl_pips: float = 15.0
    max_hold_bars: int = 96  # 96 × 15 min = 24 h
    candle_close_is_bid: bool = True  # TradeLocker candles close on bid

    # If set, overrides the SPREAD_TABLE lookup for ALL symbols.
    # Leave as None to use per-symbol spreads (recommended).
    spread_pips_override: Optional[float] = None

    def get_spread(self, symbol: str) -> float:
        """Return spread in pips for *symbol*."""
        if self.spread_pips_override is not None:
            return self.spread_pips_override
        return get_spread_pips(symbol)


@dataclass
class LabelRecord:
    """One labelled trade."""

    symbol: str
    bar_index: int
    direction: str  # "LONG" or "SHORT"
    candle_close: float
    entry_price: float
    sl_price: float
    tp_price: float
    exit_price: float
    exit_bar: int
    outcome: TradeOutcome
    spread_applied_price: float
    pip_size_used: float


class LabelEngine:
    """Generates trade-outcome labels from OHLC arrays.

    The engine walks forward through the provided close/high/low arrays,
    simulates entry with realistic costs, and records whether each trade
    would have hit TP, SL, or timed out.
    """

    def __init__(self, config: Optional[ExecutionConfig] = None) -> None:
        self.config = config or ExecutionConfig()
        self._verification_count = 0

    # ------------------------------------------------------------------
    # Entry price calculation — bid/ask consistency
    # ------------------------------------------------------------------
    def _calculate_entry_price(
        self,
        candle_close: float,
        direction: str,
        symbol: str,
    ) -> float:
        """Compute realistic entry price from candle close.

        If ``candle_close_is_bid`` (TradeLocker default):
          * LONG entry  = close + full_spread + slippage
          * SHORT entry = close − slippage

        If candle close is mid-price:
          * LONG entry  = close + half_spread + slippage
          * SHORT entry = close − half_spread − slippage
        """
        ps = pip_size(symbol)
        spread_price = self.config.get_spread(symbol) * ps
        slippage_price = self.config.slippage_pips * ps

        if self.config.candle_close_is_bid:
            if direction == "LONG":
                entry = candle_close + spread_price + slippage_price
            else:
                entry = candle_close - slippage_price
        else:
            # Mid-price assumption
            half_spread = spread_price / 2.0
            if direction == "LONG":
                entry = candle_close + half_spread + slippage_price
            else:
                entry = candle_close - half_spread - slippage_price

        return entry

    # ------------------------------------------------------------------
    # Label generation for a single trade setup
    # ------------------------------------------------------------------
    def generate_label(
        self,
        *,
        symbol: str,
        bar_index: int,
        direction: str,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> Optional[LabelRecord]:
        """Simulate one trade entry at *bar_index* and label the outcome.

        Parameters
        ----------
        symbol : str
            Instrument name (for spread/pip lookup).
        bar_index : int
            Index of the entry candle in the arrays.
        direction : str
            ``"LONG"`` or ``"SHORT"``.
        closes, highs, lows : np.ndarray
            Price arrays (must all have the same length).

        Returns ``None`` if there are not enough forward bars.
        """
        n = len(closes)
        if bar_index >= n - 1:
            return None

        candle_close = float(closes[bar_index])
        entry = self._calculate_entry_price(candle_close, direction, symbol)

        ps = pip_size(symbol)
        tp_dist = self.config.tp_pips * ps
        sl_dist = self.config.sl_pips * ps

        if direction == "LONG":
            tp_price = entry + tp_dist
            sl_price = entry - sl_dist
        else:
            tp_price = entry - tp_dist
            sl_price = entry + sl_dist

        spread_applied = self.config.get_spread(symbol) * ps

        # Verification logging for first 10 labels
        if self._verification_count < 10:
            logger.info(
                "LABEL VERIFY [%d] symbol=%s dir=%s candle_close=%.5f "
                "entry=%.5f spread_price=%.5f pip_size=%.5f tp=%.5f sl=%.5f",
                self._verification_count,
                symbol,
                direction,
                candle_close,
                entry,
                spread_applied,
                ps,
                tp_price,
                sl_price,
            )
            self._verification_count += 1

        # Walk forward to find outcome
        end_index = min(bar_index + 1 + self.config.max_hold_bars, n)
        exit_price = candle_close
        exit_bar = bar_index
        outcome = TradeOutcome.TIMEOUT

        for i in range(bar_index + 1, end_index):
            bar_high = float(highs[i])
            bar_low = float(lows[i])

            if direction == "LONG":
                # Check SL first (conservative — assume worst case intra-bar)
                if bar_low <= sl_price:
                    outcome = TradeOutcome.LOSS
                    exit_price = sl_price
                    exit_bar = i
                    break
                if bar_high >= tp_price:
                    outcome = TradeOutcome.WIN
                    exit_price = tp_price
                    exit_bar = i
                    break
            else:  # SHORT
                if bar_high >= sl_price:
                    outcome = TradeOutcome.LOSS
                    exit_price = sl_price
                    exit_bar = i
                    break
                if bar_low <= tp_price:
                    outcome = TradeOutcome.WIN
                    exit_price = tp_price
                    exit_bar = i
                    break

        # TIMEOUT → use last bar's close
        if outcome == TradeOutcome.TIMEOUT and end_index > bar_index + 1:
            exit_price = float(closes[end_index - 1])
            exit_bar = end_index - 1

        return LabelRecord(
            symbol=symbol,
            bar_index=bar_index,
            direction=direction,
            candle_close=candle_close,
            entry_price=entry,
            sl_price=sl_price,
            tp_price=tp_price,
            exit_price=exit_price,
            exit_bar=exit_bar,
            outcome=outcome,
            spread_applied_price=spread_applied,
            pip_size_used=ps,
        )

    # ------------------------------------------------------------------
    # Batch label generation
    # ------------------------------------------------------------------
    def generate_labels_batch(
        self,
        *,
        symbol: str,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        directions: Sequence[str],
        bar_indices: Sequence[int],
    ) -> List[LabelRecord]:
        """Generate labels for multiple trade setups."""
        records: List[LabelRecord] = []
        for idx, direction in zip(bar_indices, directions):
            rec = self.generate_label(
                symbol=symbol,
                bar_index=idx,
                direction=direction,
                closes=closes,
                highs=highs,
                lows=lows,
            )
            if rec is not None:
                records.append(rec)
        return records

    def log_class_balance(self, labels: List[LabelRecord]) -> Dict[str, int]:
        """Log and return outcome distribution."""
        counts: Dict[str, int] = {"WIN": 0, "LOSS": 0, "TIMEOUT": 0}
        for rec in labels:
            counts[rec.outcome.value] += 1
        total = len(labels)
        if total > 0:
            logger.info(
                "Class balance: WIN=%d (%.1f%%) LOSS=%d (%.1f%%) TIMEOUT=%d (%.1f%%) total=%d",
                counts["WIN"],
                100 * counts["WIN"] / total,
                counts["LOSS"],
                100 * counts["LOSS"] / total,
                counts["TIMEOUT"],
                100 * counts["TIMEOUT"] / total,
                total,
            )
            for key, count in counts.items():
                if count < 0.10 * total:
                    logger.warning(
                        "Class %s has < 10%% of total (%d / %d). "
                        "Consider rebalancing via scale_pos_weight.",
                        key,
                        count,
                        total,
                    )
        return counts
