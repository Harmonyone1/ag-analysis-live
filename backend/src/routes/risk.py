"""Risk API routes."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from deps import get_risk_service
from services.risk_service import RiskService

router = APIRouter()


class RiskCheckRequest(BaseModel):
    """Risk check request."""
    symbol: str
    side: str
    risk_amount: float
    spread: Optional[float] = None


class ToggleTradingRequest(BaseModel):
    """Toggle trading request."""
    enabled: bool


@router.get("/events")
async def get_risk_events(
    limit: int = Query(50, ge=1, le=200),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    service: RiskService = Depends(get_risk_service),
):
    """Get recent risk events.

    Args:
        limit: Maximum events to return
        severity: Filter by severity (INFO, WARNING, CRITICAL)

    Returns:
        List of risk events
    """
    try:
        return await service.get_risk_events(limit=limit, severity=severity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exposure")
async def get_exposure(
    service: RiskService = Depends(get_risk_service),
):
    """Get current exposure breakdown.

    Returns:
        Exposure by currency and symbol
    """
    try:
        return await service.get_exposure()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state")
async def get_risk_state(
    service: RiskService = Depends(get_risk_service),
):
    """Get current risk state.

    Returns:
        Complete risk state including limits, daily P&L, positions
    """
    try:
        return await service.get_risk_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check")
async def check_trade_risk(
    request: RiskCheckRequest,
    service: RiskService = Depends(get_risk_service),
):
    """Check if a trade passes risk filters.

    Args:
        request: Trade details to check

    Returns:
        Risk check result with pass/fail and any warnings
    """
    try:
        return await service.check_trade_risk(
            symbol=request.symbol,
            side=request.side,
            risk_amount=request.risk_amount,
            spread=request.spread,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trading/toggle")
async def toggle_trading(
    request: ToggleTradingRequest,
    service: RiskService = Depends(get_risk_service),
):
    """Enable or disable trading (kill switch).

    Args:
        request: Whether to enable trading

    Returns:
        Result of toggle operation
    """
    try:
        result = await service.toggle_trading(request.enabled)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/{event_id}/acknowledge")
async def acknowledge_event(
    event_id: str,
    service: RiskService = Depends(get_risk_service),
):
    """Acknowledge a risk event.

    Args:
        event_id: ID of the event to acknowledge

    Returns:
        Acknowledgment result
    """
    try:
        result = await service.acknowledge_event(event_id)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
