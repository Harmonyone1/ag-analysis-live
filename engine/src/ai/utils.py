"""Shared helpers for the AI gate subsystem.

Centralises pip-size logic and per-symbol spread tables so that every
module uses exactly the same values.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pip sizing
# ------------------------------------------------------------------
_JPY_PAIRS = frozenset({
    "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY",
    "CHFJPY", "NZDJPY", "SGDJPY",
})


def pip_size(symbol: str) -> float:
    """Return the pip size for *symbol*.

    * JPY pairs → ``0.01``
    * Everything else → ``0.0001``

    Every pip-to-price conversion in the codebase **must** call this
    function.  No hardcoded ``0.0001``.
    """
    sym = symbol.upper().replace("/", "").replace("_", "")
    if sym in _JPY_PAIRS or sym.endswith("JPY"):
        return 0.01
    return 0.0001


# ------------------------------------------------------------------
# Per-symbol spread table (in pips)
# ------------------------------------------------------------------
SPREAD_TABLE: Dict[str, float] = {
    # Majors
    "EURUSD": 0.8,
    "GBPUSD": 1.0,
    "USDJPY": 1.0,
    "USDCHF": 1.0,
    "AUDUSD": 1.0,
    "USDCAD": 1.2,
    "NZDUSD": 1.2,
    # Minor crosses
    "EURGBP": 1.2,
    "EURJPY": 1.3,
    "GBPJPY": 1.8,
    "EURAUD": 1.5,
    "EURCHF": 1.5,
    "EURCAD": 1.8,
    "EURNZD": 2.0,
    "GBPAUD": 2.0,
    "GBPCAD": 2.0,
    "GBPCHF": 2.0,
    "GBPNZD": 2.5,
    "AUDCAD": 1.5,
    "AUDCHF": 1.5,
    "AUDNZD": 1.5,
    "AUDJPY": 1.5,
    "CADJPY": 1.5,
    "CHFJPY": 1.5,
    "CADCHF": 1.5,
    "NZDCAD": 1.8,
    "NZDJPY": 1.5,
    "NZDCHF": 2.0,
    # Exotic crosses
    "USDSGD": 2.5,
    "USDNOK": 3.0,
    "USDSEK": 3.0,
    "USDMXN": 3.0,
    "USDTRY": 3.0,
    "USDZAR": 3.0,
    "EURSEK": 3.0,
    "EURNOK": 3.0,
    "EURTRY": 3.0,
    "SGDJPY": 2.5,
    # Crypto-CFDs (if traded)
    "ETHUSD": 2.0,
    "BTCUSD": 2.0,
    "XAUUSD": 2.5,
    "XAGUSD": 2.5,
}

_DEFAULT_SPREAD_PIPS = 1.5


def get_spread_pips(symbol: str) -> float:
    """Return the typical spread in pips for *symbol*.

    Logs a warning the first time an unknown symbol is seen so that
    missing entries don't silently pass.
    """
    sym = symbol.upper().replace("/", "").replace("_", "")
    spread = SPREAD_TABLE.get(sym)
    if spread is not None:
        return spread

    logger.warning(
        "Symbol %r not in SPREAD_TABLE — defaulting to %.1f pips.  "
        "Add it to engine/src/ai/utils.py for accurate cost modelling.",
        sym,
        _DEFAULT_SPREAD_PIPS,
    )
    return _DEFAULT_SPREAD_PIPS


def spread_in_price(symbol: str) -> float:
    """Return the full spread in *price units* (pips × pip_size)."""
    return get_spread_pips(symbol) * pip_size(symbol)
