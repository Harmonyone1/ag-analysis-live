"""Backtest engine for the AI gate.

Runs a walk-forward simulation over historical M15 data with optional
H1 / H4 multi-timeframe alignment.  Produces labelled training samples
with feature vectors for the XGBoost gate model.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import FeatureExtractor, MarketView, N_TOTAL_FEATURES
from .label_engine import ExecutionConfig, LabelEngine, LabelRecord, TradeOutcome
from .utils import pip_size, get_spread_pips

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """Parameters for a walk-forward backtest run."""

    symbols: List[str] = field(default_factory=lambda: ["EURUSD"])
    lookback_bars: int = 20000  # M15 bars to fetch
    fetch_h1: bool = True
    fetch_h4: bool = True
    tp_pips: float = 30.0
    sl_pips: float = 15.0
    slippage_pips: float = 0.3
    max_hold_bars: int = 96
    candle_close_is_bid: bool = True
    signal_interval: int = 4  # generate a signal every N bars
    min_bars_warmup: int = 100  # skip first N bars for indicator warm-up
    walk_forward_windows: int = 5  # number of train/val folds


# ------------------------------------------------------------------
# Backtest result
# ------------------------------------------------------------------
@dataclass
class BacktestSample:
    """One feature-vector + label pair ready for training."""

    symbol: str
    bar_index: int
    features: np.ndarray  # shape (42,)
    label: int  # 1 = WIN, 0 = LOSS/TIMEOUT
    outcome: str
    direction: str
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float


@dataclass
class BacktestResult:
    samples: List[BacktestSample]
    n_symbols: int
    n_bars_total: int
    duration_seconds: float


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def candles_to_arrays(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a candle DataFrame to aligned numpy arrays.

    Returns (closes, highs, lows, opens, volumes, timestamps_ms).
    """
    closes = df["close"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    opens = df["open"].to_numpy(dtype=np.float64)
    volumes = df["volume"].to_numpy(dtype=np.float64)
    timestamps = df["timestamp"].astype(np.int64).to_numpy()
    # Normalise to ms if they look like seconds
    if len(timestamps) > 0 and timestamps[0] < 1e12:
        timestamps = timestamps * 1000
    return closes, highs, lows, opens, volumes, timestamps


def _build_market_view(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
    *,
    ema_fast_span: int = 12,
    ema_slow_span: int = 50,
) -> MarketView:
    """Construct a ``MarketView`` from raw arrays with indicator computation."""
    n = len(closes)
    c = closes.astype(np.float64)

    # EMA computation
    def _ema_val(data, span):
        if len(data) < 2:
            return float(data[-1]) if len(data) > 0 else 0.0
        alpha = 2.0 / (span + 1.0)
        ema = float(data[0])
        for i in range(1, len(data)):
            ema = alpha * float(data[i]) + (1.0 - alpha) * ema
        return ema

    ema_f = _ema_val(c, ema_fast_span)
    ema_s = _ema_val(c, ema_slow_span)

    # RSI
    rsi = 50.0
    period = 14
    if n > period:
        deltas = np.diff(c[-(period + 1):])
        gains = np.clip(deltas, 0.0, None)
        losses = np.clip(-deltas, 0.0, None)
        avg_gain = float(gains.mean())
        avg_loss = float(losses.mean())
        if avg_loss > 1e-12:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        elif avg_gain > 0:
            rsi = 100.0

    # ATR + percentile
    atr_period = 14
    if n > atr_period + 1:
        tr = np.maximum(
            highs[-atr_period:] - lows[-atr_period:],
            np.maximum(
                np.abs(highs[-atr_period:] - closes[-atr_period - 1:-1]),
                np.abs(lows[-atr_period:] - closes[-atr_period - 1:-1]),
            ),
        )
        atr_val = float(np.mean(tr))
        # Percentile of current ATR within all rolling ATRs
        if n > atr_period * 3:
            rolling_atrs = []
            for j in range(atr_period + 1, n):
                tr_j = max(
                    highs[j] - lows[j],
                    abs(highs[j] - closes[j - 1]),
                    abs(lows[j] - closes[j - 1]),
                )
                rolling_atrs.append(tr_j)
            atr_pct = float(np.mean(np.array(rolling_atrs) <= atr_val))
        else:
            atr_pct = 0.5
    else:
        atr_val = float(np.mean(highs - lows)) if n > 0 else 0.0
        atr_pct = 0.5

    trend_bullish = ema_f > ema_s and c[-1] > ema_s if n > 0 else False
    trend_bearish = ema_f < ema_s and c[-1] < ema_s if n > 0 else False
    momentum_aligned = (
        (trend_bullish and c[-1] > c[-2] if n > 1 else False)
        or (trend_bearish and c[-1] < c[-2] if n > 1 else False)
    )

    # Structure state
    if trend_bullish or trend_bearish:
        structure = "trending"
    elif atr_pct < 0.35:
        structure = "ranging"
    else:
        structure = "unclear"

    return MarketView(
        closes=closes,
        highs=highs,
        lows=lows,
        opens=opens,
        volumes=volumes,
        timestamps=timestamps,
        rsi=rsi,
        atr=atr_val,
        atr_percentile=atr_pct,
        ema_fast=ema_f,
        ema_slow=ema_s,
        trend_bullish=trend_bullish,
        trend_bearish=trend_bearish,
        momentum_aligned=momentum_aligned,
        structure_state=structure,
    )


# ------------------------------------------------------------------
# Main backtest runner
# ------------------------------------------------------------------
class BacktestEngine:
    """Walk-forward backtest producing labelled feature vectors."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()
        self._feature_extractor = FeatureExtractor()
        self._label_engine = LabelEngine(
            ExecutionConfig(
                slippage_pips=self.config.slippage_pips,
                tp_pips=self.config.tp_pips,
                sl_pips=self.config.sl_pips,
                max_hold_bars=self.config.max_hold_bars,
                candle_close_is_bid=self.config.candle_close_is_bid,
            )
        )

    def run(
        self,
        candle_data: Dict[str, Dict[str, pd.DataFrame]],
    ) -> BacktestResult:
        """Execute the backtest across all symbols.

        Parameters
        ----------
        candle_data:
            ``{symbol: {"m15": df, "h1": df, "h4": df}}``.
            DataFrames must have columns:
            ``timestamp, open, high, low, close, volume``.
        """
        t0 = time.time()
        all_samples: List[BacktestSample] = []
        total_bars = 0

        for symbol, tf_data in candle_data.items():
            m15_df = tf_data.get("m15")
            if m15_df is None or m15_df.empty:
                logger.warning("No M15 data for %s — skipping", symbol)
                continue

            h1_df = tf_data.get("h1", pd.DataFrame())
            h4_df = tf_data.get("h4", pd.DataFrame())

            samples = self._process_symbol(symbol, m15_df, h1_df, h4_df)
            all_samples.extend(samples)
            total_bars += len(m15_df)

        elapsed = time.time() - t0
        logger.info(
            "Backtest complete: %d samples from %d symbols (%d total bars) in %.1fs",
            len(all_samples),
            len(candle_data),
            total_bars,
            elapsed,
        )
        return BacktestResult(
            samples=all_samples,
            n_symbols=len(candle_data),
            n_bars_total=total_bars,
            duration_seconds=elapsed,
        )

    def _process_symbol(
        self,
        symbol: str,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        h4_df: pd.DataFrame,
    ) -> List[BacktestSample]:
        """Generate labelled samples for one symbol."""
        m15_c, m15_h, m15_l, m15_o, m15_v, m15_ts = candles_to_arrays(m15_df)
        n_bars = len(m15_c)
        if n_bars < self.config.min_bars_warmup + self.config.max_hold_bars:
            logger.warning(
                "%s: only %d bars — need at least %d. Skipping.",
                symbol,
                n_bars,
                self.config.min_bars_warmup + self.config.max_hold_bars,
            )
            return []

        # Prepare H1/H4 aligned index arrays
        h1_ts = h1_c = h1_h = h1_l = h1_o = h1_v = None
        h4_ts = h4_c = h4_h = h4_l = h4_o = h4_v = None
        h1_sorted_idx: Optional[np.ndarray] = None
        h4_sorted_idx: Optional[np.ndarray] = None

        if not h1_df.empty:
            h1_c, h1_h, h1_l, h1_o, h1_v, h1_ts = candles_to_arrays(h1_df)
            h1_sorted_idx = np.searchsorted(h1_ts, m15_ts, side="right") - 1
            self._sanity_check_alignment(symbol, "H1", m15_ts, h1_ts, h1_sorted_idx)

        if not h4_df.empty:
            h4_c, h4_h, h4_l, h4_o, h4_v, h4_ts = candles_to_arrays(h4_df)
            h4_sorted_idx = np.searchsorted(h4_ts, m15_ts, side="right") - 1
            self._sanity_check_alignment(symbol, "H4", m15_ts, h4_ts, h4_sorted_idx)

        # Walk-forward label + feature generation
        samples: List[BacktestSample] = []
        prev_h1_idx = -1
        prev_h4_idx = -1
        cached_h1_view: Optional[MarketView] = None
        cached_h4_view: Optional[MarketView] = None

        for bar_i in range(self.config.min_bars_warmup, n_bars - self.config.max_hold_bars):
            if (bar_i - self.config.min_bars_warmup) % self.config.signal_interval != 0:
                continue

            # Determine direction heuristic: alternate LONG/SHORT
            direction = "LONG" if bar_i % 2 == 0 else "SHORT"

            # Label
            label_rec = self._label_engine.generate_label(
                symbol=symbol,
                bar_index=bar_i,
                direction=direction,
                closes=m15_c,
                highs=m15_h,
                lows=m15_l,
            )
            if label_rec is None:
                continue

            # M15 view (use bars up to bar_i — no look-ahead)
            lookback = min(bar_i + 1, 200)
            m15_view = _build_market_view(
                m15_c[bar_i + 1 - lookback: bar_i + 1],
                m15_h[bar_i + 1 - lookback: bar_i + 1],
                m15_l[bar_i + 1 - lookback: bar_i + 1],
                m15_o[bar_i + 1 - lookback: bar_i + 1],
                m15_v[bar_i + 1 - lookback: bar_i + 1],
                m15_ts[bar_i + 1 - lookback: bar_i + 1],
            )

            # H1 view (cached — only recompute on index change)
            h1_view: Optional[MarketView] = None
            if h1_sorted_idx is not None and h1_c is not None:
                hi = int(h1_sorted_idx[bar_i])
                if hi >= 0:
                    if hi != prev_h1_idx:
                        lb = min(hi + 1, 100)
                        cached_h1_view = _build_market_view(
                            h1_c[hi + 1 - lb: hi + 1],
                            h1_h[hi + 1 - lb: hi + 1],
                            h1_l[hi + 1 - lb: hi + 1],
                            h1_o[hi + 1 - lb: hi + 1],
                            h1_v[hi + 1 - lb: hi + 1],
                            h1_ts[hi + 1 - lb: hi + 1],
                        )
                        prev_h1_idx = hi
                    h1_view = cached_h1_view

            # H4 view (cached)
            h4_view: Optional[MarketView] = None
            if h4_sorted_idx is not None and h4_c is not None:
                hi4 = int(h4_sorted_idx[bar_i])
                if hi4 >= 0:
                    if hi4 != prev_h4_idx:
                        lb4 = min(hi4 + 1, 50)
                        cached_h4_view = _build_market_view(
                            h4_c[hi4 + 1 - lb4: hi4 + 1],
                            h4_h[hi4 + 1 - lb4: hi4 + 1],
                            h4_l[hi4 + 1 - lb4: hi4 + 1],
                            h4_o[hi4 + 1 - lb4: hi4 + 1],
                            h4_v[hi4 + 1 - lb4: hi4 + 1],
                            h4_ts[hi4 + 1 - lb4: hi4 + 1],
                        )
                        prev_h4_idx = hi4
                    h4_view = cached_h4_view

            # Feature extraction
            features = self._feature_extractor.extract(m15_view, h1_view=h1_view, h4_view=h4_view)

            # Convert outcome to binary label
            binary_label = 1 if label_rec.outcome == TradeOutcome.WIN else 0

            samples.append(
                BacktestSample(
                    symbol=symbol,
                    bar_index=bar_i,
                    features=features,
                    label=binary_label,
                    outcome=label_rec.outcome.value,
                    direction=direction,
                    entry_price=label_rec.entry_price,
                    exit_price=label_rec.exit_price,
                    tp_price=label_rec.tp_price,
                    sl_price=label_rec.sl_price,
                )
            )

        logger.info(
            "%s: %d samples (WIN=%d LOSS/TIMEOUT=%d)",
            symbol,
            len(samples),
            sum(1 for s in samples if s.label == 1),
            sum(1 for s in samples if s.label == 0),
        )
        return samples

    @staticmethod
    def _sanity_check_alignment(
        symbol: str,
        tf_name: str,
        m15_ts: np.ndarray,
        htf_ts: np.ndarray,
        idx_arr: np.ndarray,
    ) -> None:
        """Log one sample to verify that searchsorted alignment has no look-ahead."""
        # Pick the middle M15 bar
        mid = len(m15_ts) // 2
        htf_i = int(idx_arr[mid])
        if htf_i < 0 or htf_i >= len(htf_ts):
            logger.warning(
                "%s %s alignment: index %d out of range at M15 bar %d",
                symbol,
                tf_name,
                htf_i,
                mid,
            )
            return

        m15_time = m15_ts[mid]
        htf_time = htf_ts[htf_i]
        if htf_time >= m15_time:
            logger.error(
                "LOOK-AHEAD DETECTED! %s %s: htf_ts=%d >= m15_ts=%d at bar %d. "
                "The %s candle is in the future relative to the M15 decision point.",
                symbol,
                tf_name,
                htf_time,
                m15_time,
                mid,
                tf_name,
            )
        else:
            logger.info(
                "%s %s alignment OK: M15 bar %d ts=%d → %s bar %d ts=%d (delta=%dms)",
                symbol,
                tf_name,
                mid,
                m15_time,
                tf_name,
                htf_i,
                htf_time,
                m15_time - htf_time,
            )
