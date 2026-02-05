"""Pydantic schemas for API requests and responses."""

from .analysis import (
    CurrencyStrengthResponse,
    AnalysisSnapshotResponse,
    TradeCandidateResponse,
    ScannerResponse,
)
from .trading import (
    PositionResponse,
    OrderResponse,
    ExecutionResponse,
    ApprovalRequest,
    ApprovalResponse,
)
from .risk import (
    RiskEventResponse,
    ExposureResponse,
    RiskStateResponse,
)
from .common import (
    HealthResponse,
    BotStateResponse,
    ErrorResponse,
)

__all__ = [
    "CurrencyStrengthResponse",
    "AnalysisSnapshotResponse",
    "TradeCandidateResponse",
    "ScannerResponse",
    "PositionResponse",
    "OrderResponse",
    "ExecutionResponse",
    "ApprovalRequest",
    "ApprovalResponse",
    "RiskEventResponse",
    "ExposureResponse",
    "RiskStateResponse",
    "HealthResponse",
    "BotStateResponse",
    "ErrorResponse",
]
