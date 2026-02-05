"""Performance API routes."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/summary")
async def get_summary():
    """Get overall performance stats."""
    return {"win_rate": 0, "expectancy": 0, "profit_factor": 0}


@router.get("/by-setup")
async def get_by_setup():
    """Get stats by setup type."""
    return {"setups": []}


@router.get("/by-session")
async def get_by_session():
    """Get stats by session."""
    return {"sessions": []}


@router.get("/equity-curve")
async def get_equity_curve():
    """Get equity curve data."""
    return {"data": []}
