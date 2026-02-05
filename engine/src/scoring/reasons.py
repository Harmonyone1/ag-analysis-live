"""Reason codes for trade candidates.

Provides standardized, machine-readable reason codes with
human-readable descriptions for explainability.
"""

from enum import Enum
from typing import Dict


class ReasonCode(Enum):
    """Standardized reason codes for trade setups."""

    # Strength-based reasons
    STR_STRONG_BASE = "STR_STRONG_BASE"
    STR_WEAK_QUOTE = "STR_WEAK_QUOTE"
    STR_DIVERGENCE = "STR_DIVERGENCE"
    STR_MOMENTUM_ALIGN = "STR_MOMENTUM_ALIGN"

    # Structure-based reasons
    STRUCT_TREND_UP = "STRUCT_TREND_UP"
    STRUCT_TREND_DOWN = "STRUCT_TREND_DOWN"
    STRUCT_HH_HL = "STRUCT_HH_HL"
    STRUCT_LH_LL = "STRUCT_LH_LL"
    STRUCT_BOS_BULL = "STRUCT_BOS_BULL"
    STRUCT_BOS_BEAR = "STRUCT_BOS_BEAR"
    STRUCT_RANGE_SUPPORT = "STRUCT_RANGE_SUPPORT"
    STRUCT_RANGE_RESIST = "STRUCT_RANGE_RESIST"

    # Liquidity-based reasons
    LIQ_SWEEP_SELL = "LIQ_SWEEP_SELL"
    LIQ_SWEEP_BUY = "LIQ_SWEEP_BUY"
    LIQ_PDH_REJECT = "LIQ_PDH_REJECT"
    LIQ_PDL_REJECT = "LIQ_PDL_REJECT"
    LIQ_EQUAL_HIGHS_SWEPT = "LIQ_EQUAL_HIGHS_SWEPT"
    LIQ_EQUAL_LOWS_SWEPT = "LIQ_EQUAL_LOWS_SWEPT"
    LIQ_SESSION_HIGH_SWEPT = "LIQ_SESSION_HIGH_SWEPT"
    LIQ_SESSION_LOW_SWEPT = "LIQ_SESSION_LOW_SWEPT"

    # Momentum-based reasons
    MOM_RSI_OVERSOLD = "MOM_RSI_OVERSOLD"
    MOM_RSI_OVERBOUGHT = "MOM_RSI_OVERBOUGHT"
    MOM_BULL_DIV = "MOM_BULL_DIV"
    MOM_BEAR_DIV = "MOM_BEAR_DIV"
    MOM_IMPULSE_STRONG = "MOM_IMPULSE_STRONG"
    MOM_VOL_EXPAND = "MOM_VOL_EXPAND"

    # Regime-based reasons
    REG_LOW_VOL_BREAKOUT = "REG_LOW_VOL_BREAKOUT"
    REG_RISK_ON = "REG_RISK_ON"
    REG_RISK_OFF = "REG_RISK_OFF"

    # Entry quality
    ENTRY_OPTIMAL_ZONE = "ENTRY_OPTIMAL_ZONE"
    ENTRY_LIMIT_AVAILABLE = "ENTRY_LIMIT_AVAILABLE"
    ENTRY_CLEAR_INVALIDATION = "ENTRY_CLEAR_INVALIDATION"

    # Warning codes (negative)
    WARN_MOMENTUM_CONFLICT = "WARN_MOMENTUM_CONFLICT"
    WARN_STRUCTURE_UNCLEAR = "WARN_STRUCTURE_UNCLEAR"
    WARN_HIGH_SPREAD = "WARN_HIGH_SPREAD"
    WARN_LOW_VOLUME = "WARN_LOW_VOLUME"
    WARN_EVENT_NEAR = "WARN_EVENT_NEAR"


# Human-readable descriptions
REASON_DESCRIPTIONS: Dict[ReasonCode, str] = {
    # Strength
    ReasonCode.STR_STRONG_BASE: "Base currency showing relative strength",
    ReasonCode.STR_WEAK_QUOTE: "Quote currency showing relative weakness",
    ReasonCode.STR_DIVERGENCE: "Currency strength diverging from price",
    ReasonCode.STR_MOMENTUM_ALIGN: "Strength momentum aligned with direction",

    # Structure
    ReasonCode.STRUCT_TREND_UP: "Market structure in uptrend (HH/HL)",
    ReasonCode.STRUCT_TREND_DOWN: "Market structure in downtrend (LH/LL)",
    ReasonCode.STRUCT_HH_HL: "Higher high and higher low sequence",
    ReasonCode.STRUCT_LH_LL: "Lower high and lower low sequence",
    ReasonCode.STRUCT_BOS_BULL: "Bullish break of structure confirmed",
    ReasonCode.STRUCT_BOS_BEAR: "Bearish break of structure confirmed",
    ReasonCode.STRUCT_RANGE_SUPPORT: "Price at range support level",
    ReasonCode.STRUCT_RANGE_RESIST: "Price at range resistance level",

    # Liquidity
    ReasonCode.LIQ_SWEEP_SELL: "Sell-side liquidity swept with rejection",
    ReasonCode.LIQ_SWEEP_BUY: "Buy-side liquidity swept with rejection",
    ReasonCode.LIQ_PDH_REJECT: "Previous day high rejected",
    ReasonCode.LIQ_PDL_REJECT: "Previous day low rejected",
    ReasonCode.LIQ_EQUAL_HIGHS_SWEPT: "Equal highs liquidity swept",
    ReasonCode.LIQ_EQUAL_LOWS_SWEPT: "Equal lows liquidity swept",
    ReasonCode.LIQ_SESSION_HIGH_SWEPT: "Session high swept",
    ReasonCode.LIQ_SESSION_LOW_SWEPT: "Session low swept",

    # Momentum
    ReasonCode.MOM_RSI_OVERSOLD: "RSI in oversold territory",
    ReasonCode.MOM_RSI_OVERBOUGHT: "RSI in overbought territory",
    ReasonCode.MOM_BULL_DIV: "Bullish RSI divergence detected",
    ReasonCode.MOM_BEAR_DIV: "Bearish RSI divergence detected",
    ReasonCode.MOM_IMPULSE_STRONG: "Strong impulse momentum",
    ReasonCode.MOM_VOL_EXPAND: "Volatility expanding",

    # Regime
    ReasonCode.REG_LOW_VOL_BREAKOUT: "Breakout from low volatility regime",
    ReasonCode.REG_RISK_ON: "Risk-on market regime",
    ReasonCode.REG_RISK_OFF: "Risk-off market regime",

    # Entry
    ReasonCode.ENTRY_OPTIMAL_ZONE: "Price in optimal entry zone",
    ReasonCode.ENTRY_LIMIT_AVAILABLE: "Limit order entry available",
    ReasonCode.ENTRY_CLEAR_INVALIDATION: "Clear invalidation level defined",

    # Warnings
    ReasonCode.WARN_MOMENTUM_CONFLICT: "Momentum conflicts with direction",
    ReasonCode.WARN_STRUCTURE_UNCLEAR: "Market structure unclear",
    ReasonCode.WARN_HIGH_SPREAD: "Spread higher than normal",
    ReasonCode.WARN_LOW_VOLUME: "Low trading volume",
    ReasonCode.WARN_EVENT_NEAR: "High-impact event approaching",
}


def get_reason_description(code: ReasonCode) -> str:
    """Get human-readable description for a reason code."""
    return REASON_DESCRIPTIONS.get(code, f"Unknown reason: {code.value}")


def get_reason_codes_for_direction(direction: str) -> list[ReasonCode]:
    """Get positive reason codes for a trade direction."""
    if direction == "LONG":
        return [
            ReasonCode.STR_STRONG_BASE,
            ReasonCode.STR_WEAK_QUOTE,
            ReasonCode.STRUCT_TREND_UP,
            ReasonCode.STRUCT_HH_HL,
            ReasonCode.STRUCT_BOS_BULL,
            ReasonCode.STRUCT_RANGE_SUPPORT,
            ReasonCode.LIQ_SWEEP_SELL,
            ReasonCode.LIQ_PDL_REJECT,
            ReasonCode.LIQ_EQUAL_LOWS_SWEPT,
            ReasonCode.LIQ_SESSION_LOW_SWEPT,
            ReasonCode.MOM_RSI_OVERSOLD,
            ReasonCode.MOM_BULL_DIV,
        ]
    elif direction == "SHORT":
        return [
            ReasonCode.STR_WEAK_QUOTE,
            ReasonCode.STR_STRONG_BASE,  # In reverse pair
            ReasonCode.STRUCT_TREND_DOWN,
            ReasonCode.STRUCT_LH_LL,
            ReasonCode.STRUCT_BOS_BEAR,
            ReasonCode.STRUCT_RANGE_RESIST,
            ReasonCode.LIQ_SWEEP_BUY,
            ReasonCode.LIQ_PDH_REJECT,
            ReasonCode.LIQ_EQUAL_HIGHS_SWEPT,
            ReasonCode.LIQ_SESSION_HIGH_SWEPT,
            ReasonCode.MOM_RSI_OVERBOUGHT,
            ReasonCode.MOM_BEAR_DIV,
        ]
    return []
