"""Feature extractor for the AI gate model (v3.0.0).

Extracts 42 features from M15 market data plus optional H1 and H4 views:
  - Features  0-31: Base M15 features (backward-compatible with v2.0)
  - Features 32-41: MTF + regime features (new in v3.0)
"""
from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from math import tau
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_VERSION = "v3.0.0"
N_BASE_FEATURES = 32
N_MTF_FEATURES = 10
N_TOTAL_FEATURES = N_BASE_FEATURES + N_MTF_FEATURES  # 42


@dataclass
class MarketView:
    """Lightweight snapshot of analysed price data for one timeframe.

    All arrays should have the same length (the lookback window).
    Scalars represent the *latest* state.
    """

    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    opens: np.ndarray
    volumes: np.ndarray
    timestamps: np.ndarray  # int64 ms UTC

    # Pre-computed indicators (scalars — latest bar)
    rsi: float = 50.0
    atr: float = 0.0
    atr_percentile: float = 0.5
    ema_fast: float = 0.0  # e.g. EMA-12
    ema_slow: float = 0.0  # e.g. EMA-50
    trend_bullish: bool = False
    trend_bearish: bool = False
    momentum_aligned: bool = False
    structure_state: str = "unclear"  # "trending" | "ranging" | "unclear"


class FeatureExtractor:
    """Produces a fixed-length feature vector for the AI gate model."""

    def __init__(self) -> None:
        self._feature_names: List[str] = self._build_feature_names()
        assert len(self._feature_names) == N_TOTAL_FEATURES, (
            f"Expected {N_TOTAL_FEATURES} features, got {len(self._feature_names)}"
        )

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def n_features(self) -> int:
        return N_TOTAL_FEATURES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(
        self,
        m15_view: MarketView,
        *,
        h1_view: Optional[MarketView] = None,
        h4_view: Optional[MarketView] = None,
    ) -> np.ndarray:
        """Return a ``(42,)`` float32 feature vector.

        When *h1_view* or *h4_view* are ``None`` the corresponding MTF
        features default to 0.5 (neutral).
        """
        base = self._extract_base_features(m15_view)
        mtf = self._extract_mtf_features(h1_view, h4_view, m15_view)
        vec = np.concatenate([base, mtf]).astype(np.float32)
        assert vec.shape == (N_TOTAL_FEATURES,), f"Feature vector shape {vec.shape}"
        return vec

    # ------------------------------------------------------------------
    # Feature name registry
    # ------------------------------------------------------------------
    @staticmethod
    def _build_feature_names() -> List[str]:
        names: List[str] = []
        # --- Base M15 features (0-31) ---
        for h in (1, 3, 12, 60):
            names.append(f"logret_{h}")
        names.extend([
            "range_pct", "body_pct", "upper_wick", "lower_wick",
            "gap_pct", "dow", "hour_sin", "hour_cos",
        ])
        for w in (10, 30, 60):
            names.append(f"realized_vol_{w}")
            names.append(f"parkinson_vol_{w}")
        # RSI, MACD, BB, momentum, ATR-derived, additional
        names.extend([
            "rsi_14", "rsi_7", "macd_hist", "bb_percent", "bb_width",
            "momentum_5", "momentum_12", "atr_norm", "trend_strength",
            "candle_score", "volume_ratio", "price_vs_ema50",
            "close_vs_high", "close_vs_low",
        ])
        assert len(names) == N_BASE_FEATURES, f"Base features: {len(names)}"
        # --- MTF + Regime features (32-41) ---
        names.extend([
            "h1_trend_bullish",
            "h1_trend_bearish",
            "h1_rsi_value",
            "h1_momentum_aligned",
            "h4_trend_bullish",
            "h4_trend_bearish",
            "h4_structure_aligned",
            "m15_aligns_h4",
            "regime_trending",
            "regime_ranging",
        ])
        return names

    # ------------------------------------------------------------------
    # Base M15 feature extraction
    # ------------------------------------------------------------------
    def _extract_base_features(self, v: MarketView) -> np.ndarray:
        """Extract 32 base features from the M15 view."""
        c = v.closes.astype(np.float64)
        h = v.highs.astype(np.float64)
        lo = v.lows.astype(np.float64)
        o = v.opens.astype(np.float64)
        vol = v.volumes.astype(np.float64)
        n = len(c)

        feats = np.zeros(N_BASE_FEATURES, dtype=np.float64)

        # Log returns at different horizons
        idx = 0
        for horizon in (1, 3, 12, 60):
            if n > horizon and c[-1] > 0 and c[-1 - horizon] > 0:
                feats[idx] = np.log(c[-1] / c[-1 - horizon])
            idx += 1

        # Candle anatomy
        if c[-1] > 0:
            feats[idx] = (h[-1] - lo[-1]) / c[-1]  # range_pct
        idx += 1
        if c[-1] > 0:
            feats[idx] = (c[-1] - o[-1]) / c[-1]  # body_pct
        idx += 1
        top = max(o[-1], c[-1])
        bot = min(o[-1], c[-1])
        if c[-1] > 0:
            feats[idx] = (h[-1] - top) / c[-1]  # upper_wick
        idx += 1
        if c[-1] > 0:
            feats[idx] = (bot - lo[-1]) / c[-1]  # lower_wick
        idx += 1

        # Gap
        if n > 1 and c[-2] > 0:
            feats[idx] = (o[-1] - c[-2]) / c[-2]  # gap_pct
        idx += 1

        # Time features (use last timestamp)
        if len(v.timestamps) > 0:
            ts_ms = int(v.timestamps[-1])
            dt = _dt.datetime.fromtimestamp(ts_ms / 1000, tz=_dt.timezone.utc)
            feats[idx] = dt.weekday()  # dow
            idx += 1
            hour_frac = dt.hour + dt.minute / 60.0
            feats[idx] = np.sin(hour_frac * tau / 24)  # hour_sin
            idx += 1
            feats[idx] = np.cos(hour_frac * tau / 24)  # hour_cos
            idx += 1
        else:
            idx += 3

        # Volatility windows
        for w in (10, 30, 60):
            feats[idx] = self._realized_vol(c, w)
            idx += 1
            feats[idx] = self._parkinson_vol(h, lo, w)
            idx += 1

        # RSI-14
        feats[idx] = self._rsi(c, 14) / 100.0  # normalise to [0, 1]
        idx += 1

        # RSI-7
        feats[idx] = self._rsi(c, 7) / 100.0
        idx += 1

        # MACD histogram (normalised)
        macd_hist = self._macd_histogram(c)
        if c[-1] > 0:
            feats[idx] = macd_hist / c[-1]
        idx += 1

        # Bollinger %B and width
        bb_pct, bb_w = self._bollinger(c, 20, 2.0)
        feats[idx] = bb_pct
        idx += 1
        feats[idx] = bb_w
        idx += 1

        # Momentum (5 bars)
        if n > 5 and c[-6] > 0:
            feats[idx] = (c[-1] - c[-6]) / c[-6]
        idx += 1

        # Momentum (12 bars)
        if n > 12 and c[-13] > 0:
            feats[idx] = (c[-1] - c[-13]) / c[-13]
        idx += 1

        # ATR normalised
        atr = self._atr(h, lo, c, 14)
        if c[-1] > 0:
            feats[idx] = atr / c[-1]
        idx += 1

        # Trend strength: (EMA12 - EMA50) / ATR
        ema12 = self._ema(c, 12)
        ema50 = self._ema(c, 50)
        if atr > 0:
            feats[idx] = (ema12 - ema50) / atr
        idx += 1

        # Candle score: body / range
        bar_range = h[-1] - lo[-1]
        if bar_range > 0:
            feats[idx] = abs(c[-1] - o[-1]) / bar_range
        idx += 1

        # Volume ratio: last bar vs 20-bar mean
        if n >= 20:
            mean_vol = np.mean(vol[-20:])
            if mean_vol > 0:
                feats[idx] = vol[-1] / mean_vol
        idx += 1

        # Price vs EMA-50
        if ema50 > 0:
            feats[idx] = (c[-1] - ema50) / ema50
        idx += 1

        # Close vs High (how close to the high of the bar)
        if bar_range > 0:
            feats[idx] = (c[-1] - lo[-1]) / bar_range
        idx += 1

        # Close vs Low (distance from low as fraction of range)
        if bar_range > 0:
            feats[idx] = (h[-1] - c[-1]) / bar_range
        idx += 1

        assert idx == N_BASE_FEATURES
        return feats

    # ------------------------------------------------------------------
    # MTF + regime features
    # ------------------------------------------------------------------
    def _extract_mtf_features(
        self,
        h1: Optional[MarketView],
        h4: Optional[MarketView],
        m15: MarketView,
    ) -> np.ndarray:
        """Extract 10 multi-timeframe and regime features."""
        feats = np.full(N_MTF_FEATURES, 0.5, dtype=np.float64)

        # H1 features (indices 0-3)
        if h1 is not None and len(h1.closes) > 0:
            feats[0] = 1.0 if h1.trend_bullish else 0.0
            feats[1] = 1.0 if h1.trend_bearish else 0.0
            feats[2] = h1.rsi / 100.0 if h1.rsi is not None else 0.5
            feats[3] = 1.0 if h1.momentum_aligned else 0.0

        # H4 features (indices 4-7)
        if h4 is not None and len(h4.closes) > 0:
            feats[4] = 1.0 if h4.trend_bullish else 0.0
            feats[5] = 1.0 if h4.trend_bearish else 0.0
            # H4 structure aligned
            feats[6] = 1.0 if h4.structure_state == "trending" else 0.0
            # M15 aligns with H4 trend
            m15_bull = m15.trend_bullish
            m15_bear = m15.trend_bearish
            feats[7] = 1.0 if (
                (h4.trend_bullish and m15_bull) or (h4.trend_bearish and m15_bear)
            ) else 0.0

        # Regime features (indices 8-9)
        feats[8], feats[9] = self._extract_regime_features(m15, h4)

        return feats

    def _extract_regime_features(
        self,
        m15: MarketView,
        h4: Optional[MarketView],
    ) -> tuple[float, float]:
        """Return (regime_trending, regime_ranging).

        These are NOT mutually exclusive: both 0 → unclear regime.
        Uses ATR percentile and structure state to determine regime.
        """
        # Start with ATR percentile from M15
        atr_pct = m15.atr_percentile if m15.atr_percentile is not None else 0.5

        # High ATR + directional structure → trending
        trending = 0.0
        ranging = 0.0

        if atr_pct > 0.65:
            if m15.structure_state == "trending":
                trending = min(1.0, atr_pct)
            elif m15.structure_state == "ranging":
                ranging = min(1.0, 1.0 - atr_pct + 0.3)
            else:
                trending = atr_pct * 0.5
        elif atr_pct < 0.35:
            ranging = min(1.0, 1.0 - atr_pct)
            if m15.structure_state == "trending":
                trending = 0.3
        else:
            # Mid ATR: lean on structure
            if m15.structure_state == "trending":
                trending = 0.6
            elif m15.structure_state == "ranging":
                ranging = 0.6

        # Refine with H4 if available
        if h4 is not None and len(h4.closes) > 0:
            if h4.structure_state == "trending":
                trending = min(1.0, trending + 0.2)
            elif h4.structure_state == "ranging":
                ranging = min(1.0, ranging + 0.2)

        return trending, ranging

    # ------------------------------------------------------------------
    # Technical indicator helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ema(data: np.ndarray, span: int) -> float:
        if len(data) < span:
            return float(data[-1]) if len(data) > 0 else 0.0
        alpha = 2.0 / (span + 1.0)
        ema = float(data[0])
        for i in range(1, len(data)):
            ema = alpha * float(data[i]) + (1.0 - alpha) * ema
        return ema

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        if len(closes) <= period:
            return 50.0
        window = closes[-(period + 1):]
        deltas = np.diff(window)
        gains = np.clip(deltas, 0.0, None)
        losses = np.clip(-deltas, 0.0, None)
        avg_gain = float(gains.mean())
        avg_loss = float(losses.mean())
        if avg_loss < 1e-12:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _macd_histogram(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        if len(closes) < slow + signal:
            return 0.0
        alpha_f = 2.0 / (fast + 1)
        alpha_s = 2.0 / (slow + 1)
        ema_f = float(closes[0])
        ema_s = float(closes[0])
        macd_line = np.zeros(len(closes))
        for i in range(1, len(closes)):
            ema_f = alpha_f * float(closes[i]) + (1 - alpha_f) * ema_f
            ema_s = alpha_s * float(closes[i]) + (1 - alpha_s) * ema_s
            macd_line[i] = ema_f - ema_s
        alpha_sig = 2.0 / (signal + 1)
        sig = macd_line[0]
        for i in range(1, len(macd_line)):
            sig = alpha_sig * macd_line[i] + (1 - alpha_sig) * sig
        return float(macd_line[-1] - sig)

    @staticmethod
    def _bollinger(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple[float, float]:
        if len(closes) < period:
            return 0.5, 0.0
        window = closes[-period:]
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=0))
        if std < 1e-12 or abs(mean) < 1e-12:
            return 0.5, 0.0
        upper = mean + num_std * std
        lower = mean - num_std * std
        pct_b = (float(window[-1]) - lower) / (upper - lower)
        width = (upper - lower) / mean
        return float(np.clip(pct_b, 0, 1)), float(width)

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return float(np.mean(highs[-period:] - lows[-period:])) if len(closes) > 0 else 0.0
        tr = np.maximum(
            highs[-period:] - lows[-period:],
            np.maximum(
                np.abs(highs[-period:] - closes[-period - 1:-1]),
                np.abs(lows[-period:] - closes[-period - 1:-1]),
            ),
        )
        return float(np.mean(tr))

    @staticmethod
    def _realized_vol(closes: np.ndarray, window: int) -> float:
        if len(closes) < window + 1:
            return 0.0
        log_returns = np.diff(np.log(np.clip(closes[-window - 1:], 1e-10, None)))
        return float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0

    @staticmethod
    def _parkinson_vol(highs: np.ndarray, lows: np.ndarray, window: int) -> float:
        if len(highs) < window:
            return 0.0
        h = highs[-window:]
        lo = lows[-window:]
        ratio = np.clip(h / np.clip(lo, 1e-10, None), 1e-10, None)
        log_hl = np.log(ratio)
        return float(np.sqrt(np.mean(log_hl ** 2) / (4 * np.log(2))))
