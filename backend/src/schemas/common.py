"""Common schemas."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    components: Dict[str, str] = {}


class BotStateResponse(BaseModel):
    """Bot state response."""
    mode: str  # SCANNING, PAUSED, ERROR
    trading_enabled: bool
    last_scan: Optional[datetime]
    active_candidates: int
    open_positions: int
    daily_pnl: float
    error_message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class PaginatedResponse(BaseModel):
    """Base for paginated responses."""
    total: int
    page: int
    page_size: int
    has_more: bool


class MessageResponse(BaseModel):
    """Simple message response."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
