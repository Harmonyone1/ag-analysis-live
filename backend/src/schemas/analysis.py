"""Analysis schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CurrencyStrength(BaseModel):
    """Single currency strength."""
    currency: str
    strength: float
    rank: int
    change_1h: Optional[float] = None
    change_4h: Optional[float] = None


class CurrencyStrengthResponse(BaseModel):
    """Currency strength rankings response."""
    timestamp: datetime
    currencies: List[CurrencyStrength]
    strongest: str
    weakest: str


class StructureInfo(BaseModel):
    """Market structure information."""
    trend: str  # BULLISH, BEARISH, RANGING
    confidence: float
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None
    last_bos_type: Optional[str] = None
    last_bos_price: Optional[float] = None


class LiquidityInfo(BaseModel):
    """Liquidity analysis information."""
    zones: List[Dict[str, Any]]
    recent_sweeps: List[Dict[str, Any]]
    pdh: Optional[float] = None
    pdl: Optional[float] = None


class MomentumInfo(BaseModel):
    """Momentum analysis information."""
    rsi: float
    atr: float
    atr_ratio: float
    bias: str
    impulse_ratio: float
    warnings: List[str] = []


class AnalysisSnapshotResponse(BaseModel):
    """Full analysis snapshot for a symbol."""
    symbol: str
    timeframe: str
    timestamp: datetime
    current_price: float

    # Component analyses
    structure: Optional[StructureInfo] = None
    liquidity: Optional[LiquidityInfo] = None
    momentum: Optional[MomentumInfo] = None

    # Derived values
    directional_bias: str  # LONG, SHORT, NEUTRAL
    bias_strength: float
    key_levels: List[Dict[str, Any]] = []


class TradeCandidateResponse(BaseModel):
    """Trade candidate response."""
    id: str
    symbol: str
    direction: str
    confluence_score: int
    entry_zone: List[float]
    stop_price: float
    tp_targets: List[Dict[str, Any]]
    risk_reward_ratio: float
    reasons: List[str]
    status: str  # PENDING, APPROVED, REJECTED, EXPIRED
    created_at: datetime
    expires_at: Optional[datetime] = None
    ai_probability: Optional[float] = None
    ai_decision: Optional[str] = None


class ScannerResponse(BaseModel):
    """Scanner results response."""
    timestamp: datetime
    scan_duration_ms: int
    candidates: List[TradeCandidateResponse]
    symbols_scanned: int
    filters_applied: List[str] = []


class ChartDataPoint(BaseModel):
    """Single chart data point."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class ChartDataResponse(BaseModel):
    """Chart data response."""
    symbol: str
    timeframe: str
    candles: List[ChartDataPoint]
    annotations: List[Dict[str, Any]] = []
