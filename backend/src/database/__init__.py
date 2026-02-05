"""Database layer for AG Analyzer Backend."""

from .connection import DatabaseManager, get_db, init_db
from .models import (
    Base,
    Instrument,
    PriceHistory,
    AnalysisSnapshot,
    TradeCandidate,
    Order,
    Execution,
    Position,
    RiskEvent,
    BotState,
    ModelRegistry,
    Setting,
    EconomicCalendar,
    TradeLog,
)

__all__ = [
    "DatabaseManager",
    "get_db",
    "init_db",
    "Base",
    "Instrument",
    "PriceHistory",
    "AnalysisSnapshot",
    "TradeCandidate",
    "Order",
    "Execution",
    "Position",
    "RiskEvent",
    "BotState",
    "ModelRegistry",
    "Setting",
    "EconomicCalendar",
    "TradeLog",
]
