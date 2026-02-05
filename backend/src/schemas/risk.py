"""Risk schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class RiskEventResponse(BaseModel):
    """Risk event response."""
    id: str
    event_type: str
    severity: str  # INFO, WARNING, CRITICAL
    message: str
    payload: Dict[str, Any]
    timestamp: datetime
    acknowledged: bool = False


class RiskEventsListResponse(BaseModel):
    """List of risk events response."""
    events: List[RiskEventResponse]
    total_count: int
    critical_count: int
    warning_count: int


class CurrencyExposure(BaseModel):
    """Currency exposure detail."""
    currency: str
    long_exposure: float
    short_exposure: float
    net_exposure: float
    exposure_pct: float  # As percentage of account


class ExposureResponse(BaseModel):
    """Exposure breakdown response."""
    timestamp: datetime
    total_exposure: float
    exposure_pct: float  # As percentage of account
    by_currency: List[CurrencyExposure]
    by_symbol: Dict[str, float]
    correlation_risk: float  # 0-100


class RiskLimitsResponse(BaseModel):
    """Risk limits configuration response."""
    max_risk_per_trade: float
    max_daily_loss: float
    max_open_positions: int
    max_correlated_exposure: float
    max_spread_multiplier: float
    slippage_halt_threshold: float


class RiskStateResponse(BaseModel):
    """Current risk state response."""
    trading_enabled: bool
    daily_pnl: float
    daily_pnl_pct: float
    daily_trades: int
    open_positions: int
    max_positions: int
    current_exposure_pct: float
    max_daily_loss_pct: float
    at_daily_limit: bool
    at_position_limit: bool
    limits: RiskLimitsResponse
    recent_events: List[RiskEventResponse]


class RiskCheckRequest(BaseModel):
    """Risk check request for new trade."""
    symbol: str
    side: str
    risk_amount: float
    spread: Optional[float] = None


class RiskCheckResponse(BaseModel):
    """Risk check response."""
    passed: bool
    reason: str
    suggested_size_multiplier: float = 1.0
    warnings: List[str] = []
