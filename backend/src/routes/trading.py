"""Trading API routes."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from deps import get_trading_service
from services.trading_service import TradingService

router = APIRouter()


class ApprovalRequest(BaseModel):
    """Trade approval request."""
    position_size: Optional[float] = None
    notes: Optional[str] = None


class RejectionRequest(BaseModel):
    """Trade rejection request."""
    reason: Optional[str] = None


class ModifyPositionRequest(BaseModel):
    """Modify position request."""
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None


class ClosePositionRequest(BaseModel):
    """Close position request."""
    quantity: Optional[float] = None


@router.get("/positions")
async def get_positions(
    service: TradingService = Depends(get_trading_service),
):
    """Get open positions.

    Returns:
        List of open positions with P&L
    """
    try:
        return await service.get_positions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders")
async def get_orders(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    service: TradingService = Depends(get_trading_service),
):
    """Get orders.

    Args:
        status: Filter by status
        limit: Maximum orders to return

    Returns:
        List of orders
    """
    try:
        return await service.get_orders(status=status, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions")
async def get_executions(
    limit: int = Query(50, ge=1, le=200),
    service: TradingService = Depends(get_trading_service),
):
    """Get recent executions.

    Args:
        limit: Maximum executions to return

    Returns:
        List of executions
    """
    try:
        return await service.get_executions(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/candidates/{candidate_id}/approve")
async def approve_candidate(
    candidate_id: str,
    request: ApprovalRequest = None,
    service: TradingService = Depends(get_trading_service),
):
    """Approve a trade candidate for execution.

    Args:
        candidate_id: ID of the candidate to approve
        request: Optional approval parameters

    Returns:
        Approval result with order details
    """
    try:
        position_size = request.position_size if request else None
        notes = request.notes if request else None
        result = await service.approve_candidate(
            candidate_id,
            position_size=position_size,
            notes=notes,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/candidates/{candidate_id}/reject")
async def reject_candidate(
    candidate_id: str,
    request: RejectionRequest = None,
    service: TradingService = Depends(get_trading_service),
):
    """Reject a trade candidate.

    Args:
        candidate_id: ID of the candidate to reject
        request: Optional rejection reason

    Returns:
        Rejection result
    """
    try:
        reason = request.reason if request else None
        result = await service.reject_candidate(candidate_id, reason=reason)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/positions/{position_id}")
async def modify_position(
    position_id: str,
    request: ModifyPositionRequest,
    service: TradingService = Depends(get_trading_service),
):
    """Modify position stop loss or take profit.

    Args:
        position_id: ID of the position to modify
        request: New stop loss and/or take profit values

    Returns:
        Modification result
    """
    try:
        result = await service.modify_position(
            position_id,
            new_stop_loss=request.new_stop_loss,
            new_take_profit=request.new_take_profit,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/positions/{position_id}/close")
async def close_position(
    position_id: str,
    request: ClosePositionRequest = None,
    service: TradingService = Depends(get_trading_service),
):
    """Close a position.

    Args:
        position_id: ID of the position to close
        request: Optional partial close quantity

    Returns:
        Close result
    """
    try:
        quantity = request.quantity if request else None
        result = await service.close_position(position_id, quantity=quantity)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
