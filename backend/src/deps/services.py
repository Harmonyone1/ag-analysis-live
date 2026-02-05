"""Service dependencies."""

from fastapi import Depends
from sqlalchemy.orm import Session

from .database import get_db
from services.analysis_service import AnalysisService
from services.trading_service import TradingService
from services.risk_service import RiskService


def get_analysis_service(db: Session = Depends(get_db)) -> AnalysisService:
    """Get analysis service instance.

    Args:
        db: Database session

    Returns:
        AnalysisService instance
    """
    return AnalysisService(db)


def get_trading_service(db: Session = Depends(get_db)) -> TradingService:
    """Get trading service instance.

    Args:
        db: Database session

    Returns:
        TradingService instance
    """
    return TradingService(db)


def get_risk_service(db: Session = Depends(get_db)) -> RiskService:
    """Get risk service instance.

    Args:
        db: Database session

    Returns:
        RiskService instance
    """
    return RiskService(db)
