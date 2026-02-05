"""Trading schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class PositionResponse(BaseModel):
    """Position response."""
    id: str
    broker_position_id: Optional[int] = None
    symbol: str
    side: str  # buy/sell
    quantity: float
    entry_price: float
    current_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    opened_at: datetime
    candidate_id: Optional[str] = None


class PositionsListResponse(BaseModel):
    """List of positions response."""
    positions: List[PositionResponse]
    total_count: int
    total_unrealized_pnl: float


class OrderResponse(BaseModel):
    """Order response."""
    id: str
    broker_order_id: Optional[str] = None
    symbol: str
    side: str
    order_type: str  # market, limit, stop
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str  # pending, placed, filled, cancelled, rejected
    created_at: datetime
    filled_at: Optional[datetime] = None
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None


class OrdersListResponse(BaseModel):
    """List of orders response."""
    orders: List[OrderResponse]
    total_count: int


class ExecutionResponse(BaseModel):
    """Execution response."""
    id: str
    order_id: str
    candidate_id: Optional[str] = None
    symbol: str
    side: str
    quantity: float
    price: float
    slippage_pips: Optional[float] = None
    executed_at: datetime
    status: str


class ExecutionsListResponse(BaseModel):
    """List of executions response."""
    executions: List[ExecutionResponse]
    total_count: int


class ApprovalRequest(BaseModel):
    """Trade approval request."""
    position_size: Optional[float] = None  # Override calculated size
    notes: Optional[str] = None


class ApprovalResponse(BaseModel):
    """Trade approval response."""
    success: bool
    candidate_id: str
    order_id: Optional[str] = None
    message: str
    execution_result: Optional[Dict[str, Any]] = None


class RejectionResponse(BaseModel):
    """Trade rejection response."""
    success: bool
    candidate_id: str
    message: str


class ModifyPositionRequest(BaseModel):
    """Modify position request."""
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None


class ClosePositionRequest(BaseModel):
    """Close position request."""
    quantity: Optional[float] = None  # Partial close if specified
