"""Analysis API routes."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from deps import get_analysis_service
from services.analysis_service import AnalysisService

router = APIRouter()


@router.get("/strength")
async def get_strength(
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get currency strength rankings.

    Returns:
        Currency strength data with rankings
    """
    try:
        return await service.get_currency_strength()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scanner")
async def get_scanner(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get trade candidates from scanner.

    Args:
        status: Filter by status (PENDING, APPROVED, etc.)
        limit: Maximum candidates to return

    Returns:
        List of trade candidates
    """
    try:
        return await service.get_trade_candidates(status=status, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chart/{symbol}")
async def get_chart_data(
    symbol: str,
    timeframe: str = Query("M15", description="Chart timeframe"),
    limit: int = Query(200, ge=1, le=1000),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get chart data for a symbol.

    Args:
        symbol: Trading symbol
        timeframe: Chart timeframe (M15, H1, H4, D1)
        limit: Number of candles

    Returns:
        Chart data with candles
    """
    try:
        return await service.get_chart_data(symbol, timeframe, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}")
async def get_analysis(
    symbol: str,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get latest analysis snapshot for a symbol.

    Args:
        symbol: Trading symbol (e.g., EURUSD)

    Returns:
        Full analysis snapshot including structure, liquidity, momentum
    """
    try:
        return await service.get_analysis_snapshot(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
