"""Relative Strength Engine for currency and index analysis.

Computes per-currency strength using basket returns and ranks
strongest-vs-weakest pairs across multiple time horizons.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import optimize
import structlog

logger = structlog.get_logger(__name__)

# Major currencies for FX analysis
FX_CURRENCIES = ["EUR", "USD", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]

# FX pairs for strength calculation
FX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
    "AUDJPY", "AUDCHF", "AUDNZD", "AUDCAD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
]

# Time horizons for analysis
HORIZONS = {
    "H1": 1,      # 1 hour
    "H4": 4,      # 4 hours
    "1D": 24,     # 1 day
    "5D": 120,    # 5 days
    "20D": 480,   # 20 days (trading days)
}


@dataclass
class CurrencyStrength:
    """Currency strength data for a single horizon."""
    currency: str
    strength: float
    rank: int
    change: float  # Change from previous period
    timestamp: datetime


@dataclass
class StrengthAnalysis:
    """Complete strength analysis output."""
    timestamp: datetime
    horizon: str
    strengths: Dict[str, CurrencyStrength]
    pair_rankings: List[Tuple[str, float]]  # Sorted strongest to weakest
    strongest_currency: str
    weakest_currency: str


class StrengthEngine:
    """Computes relative currency strength using basket decomposition.

    For FX: Uses constrained least squares to decompose pair returns
    into individual currency strengths.

    For indices: Uses volatility-normalized returns.
    """

    def __init__(self):
        """Initialize strength engine."""
        self._currency_map = self._build_currency_map()
        self._last_strengths: Dict[str, Dict[str, float]] = {}

    def _build_currency_map(self) -> Dict[str, Tuple[str, str]]:
        """Build mapping of pairs to base/quote currencies."""
        pair_map = {}
        for pair in FX_PAIRS:
            if len(pair) == 6:
                base = pair[:3]
                quote = pair[3:]
                pair_map[pair] = (base, quote)
        return pair_map

    def calculate_fx_strength(
        self,
        returns: Dict[str, float],
        horizon: str = "1D"
    ) -> StrengthAnalysis:
        """Calculate currency strength from pair returns.

        Uses constrained least squares: for pair A/B, return ≈ strength(A) - strength(B)
        Subject to: sum of all strengths = 0 (zero-sum constraint)

        Args:
            returns: Dictionary of {pair: return} (e.g., {"EURUSD": 0.005})
            horizon: Time horizon label

        Returns:
            StrengthAnalysis with computed strengths
        """
        timestamp = datetime.now()

        # Build matrices for least squares
        currencies = FX_CURRENCIES.copy()
        n_currencies = len(currencies)
        currency_idx = {c: i for i, c in enumerate(currencies)}

        # Count valid pairs
        valid_pairs = [(p, r) for p, r in returns.items() if p in self._currency_map]
        if len(valid_pairs) < n_currencies - 1:
            logger.warning("Insufficient pairs for strength calculation",
                         pairs=len(valid_pairs), required=n_currencies-1)
            return self._empty_analysis(timestamp, horizon)

        # Build design matrix A and observation vector b
        # Each row: A[i, base] = 1, A[i, quote] = -1, b[i] = return
        A = np.zeros((len(valid_pairs), n_currencies))
        b = np.zeros(len(valid_pairs))

        for i, (pair, ret) in enumerate(valid_pairs):
            base, quote = self._currency_map[pair]
            if base in currency_idx and quote in currency_idx:
                A[i, currency_idx[base]] = 1
                A[i, currency_idx[quote]] = -1
                b[i] = ret

        # Add zero-sum constraint: sum(strengths) = 0
        constraint_row = np.ones((1, n_currencies))
        A_constrained = np.vstack([A, constraint_row])
        b_constrained = np.append(b, 0)

        # Solve using least squares
        try:
            strengths_raw, _, _, _ = np.linalg.lstsq(A_constrained, b_constrained, rcond=None)
        except np.linalg.LinAlgError:
            logger.error("Failed to solve strength system")
            return self._empty_analysis(timestamp, horizon)

        # Build strength results
        strengths = {}
        for i, currency in enumerate(currencies):
            strength_val = float(strengths_raw[i])

            # Calculate change from last period
            change = 0.0
            if horizon in self._last_strengths and currency in self._last_strengths[horizon]:
                change = strength_val - self._last_strengths[horizon][currency]

            strengths[currency] = CurrencyStrength(
                currency=currency,
                strength=strength_val,
                rank=0,  # Will be set after sorting
                change=change,
                timestamp=timestamp,
            )

        # Rank currencies
        sorted_currencies = sorted(strengths.values(), key=lambda x: x.strength, reverse=True)
        for rank, cs in enumerate(sorted_currencies, 1):
            strengths[cs.currency].rank = rank

        # Cache for change calculation
        self._last_strengths[horizon] = {c: s.strength for c, s in strengths.items()}

        # Calculate pair rankings (strongest vs weakest)
        pair_rankings = self._calculate_pair_rankings(strengths)

        return StrengthAnalysis(
            timestamp=timestamp,
            horizon=horizon,
            strengths=strengths,
            pair_rankings=pair_rankings,
            strongest_currency=sorted_currencies[0].currency,
            weakest_currency=sorted_currencies[-1].currency,
        )

    def _calculate_pair_rankings(
        self,
        strengths: Dict[str, CurrencyStrength]
    ) -> List[Tuple[str, float]]:
        """Rank pairs by strength differential (strongest base vs weakest quote)."""
        pair_scores = []

        for pair, (base, quote) in self._currency_map.items():
            if base in strengths and quote in strengths:
                # Positive score = base stronger than quote (bullish for pair)
                score = strengths[base].strength - strengths[quote].strength
                pair_scores.append((pair, score))

        # Sort by absolute score (strongest setups first)
        return sorted(pair_scores, key=lambda x: abs(x[1]), reverse=True)

    def calculate_index_strength(
        self,
        returns: Dict[str, float],
        volatilities: Dict[str, float],
        horizon: str = "1D"
    ) -> Dict[str, CurrencyStrength]:
        """Calculate index strength using volatility-normalized returns.

        Args:
            returns: Dictionary of {index: return}
            volatilities: Dictionary of {index: volatility}
            horizon: Time horizon label

        Returns:
            Dictionary of index strengths
        """
        timestamp = datetime.now()
        strengths = {}

        for index, ret in returns.items():
            vol = volatilities.get(index, 0.01)  # Default 1% vol
            normalized_return = ret / vol if vol > 0 else 0

            strengths[index] = CurrencyStrength(
                currency=index,
                strength=normalized_return,
                rank=0,
                change=0,
                timestamp=timestamp,
            )

        # Rank indices
        sorted_indices = sorted(strengths.values(), key=lambda x: x.strength, reverse=True)
        for rank, cs in enumerate(sorted_indices, 1):
            strengths[cs.currency].rank = rank

        return strengths

    def get_pair_bias(
        self,
        pair: str,
        strengths: Dict[str, CurrencyStrength]
    ) -> Tuple[str, float]:
        """Get directional bias for a pair based on currency strengths.

        Args:
            pair: Currency pair (e.g., "EURUSD")
            strengths: Current strength analysis

        Returns:
            Tuple of (direction, confidence) where direction is "LONG" or "SHORT"
        """
        if pair not in self._currency_map:
            return ("NEUTRAL", 0.0)

        base, quote = self._currency_map[pair]
        if base not in strengths or quote not in strengths:
            return ("NEUTRAL", 0.0)

        differential = strengths[base].strength - strengths[quote].strength

        # Normalize to confidence score (0-1)
        # Typical differential range is -0.03 to +0.03
        confidence = min(abs(differential) / 0.02, 1.0)

        if differential > 0.005:
            return ("LONG", confidence)
        elif differential < -0.005:
            return ("SHORT", confidence)
        else:
            return ("NEUTRAL", confidence)

    def _empty_analysis(self, timestamp: datetime, horizon: str) -> StrengthAnalysis:
        """Return empty analysis when calculation fails."""
        return StrengthAnalysis(
            timestamp=timestamp,
            horizon=horizon,
            strengths={},
            pair_rankings=[],
            strongest_currency="",
            weakest_currency="",
        )

    def to_dict(self, analysis: StrengthAnalysis) -> Dict:
        """Convert analysis to dictionary for storage."""
        return {
            "timestamp": analysis.timestamp.isoformat(),
            "horizon": analysis.horizon,
            "strengths": {
                c: {
                    "strength": s.strength,
                    "rank": s.rank,
                    "change": s.change,
                }
                for c, s in analysis.strengths.items()
            },
            "pair_rankings": analysis.pair_rankings[:10],  # Top 10
            "strongest": analysis.strongest_currency,
            "weakest": analysis.weakest_currency,
        }
