"""AG Analyzer Backend API - FastAPI Application."""

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/app/engine/src")

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import structlog

from routes import analysis, trading, risk, performance, models
from deps import get_db
from sqlalchemy.orm import Session

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting AG Analyzer Backend API")
    yield
    logger.info("Shutting down AG Analyzer Backend API")


app = FastAPI(
    title="AG Analyzer API",
    description="AI-driven trading analyzer and bot API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(trading.router, prefix="/api", tags=["Trading"])
app.include_router(risk.router, prefix="/api/risk", tags=["Risk"])
app.include_router(performance.router, prefix="/api/performance", tags=["Performance"])
app.include_router(models.router, prefix="/api/models", tags=["AI Models"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AG Analyzer API", "version": "0.1.0"}


@app.get("/health")
async def health(db: Session = Depends(get_db)):
    """Health check endpoint."""
    try:
        # Check database connection
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "components": {
            "database": db_status,
            "api": "healthy",
        }
    }


@app.get("/api/bot/state")
async def get_bot_state(db: Session = Depends(get_db)):
    """Get current bot state."""
    try:
        from database.models import BotState, Position

        bot_state = db.query(BotState).filter(BotState.id == 1).first()
        position_count = db.query(Position).filter(Position.close_time.is_(None)).count()

        if bot_state:
            return {
                "trading_enabled": bot_state.trading_enabled,
                "mode": bot_state.mode,
                "last_heartbeat": bot_state.last_heartbeat.isoformat() if bot_state.last_heartbeat else None,
                "current_model_version": bot_state.current_model_version,
                "daily_pnl": float(bot_state.daily_pnl) if bot_state.daily_pnl else 0,
                "daily_trades": bot_state.daily_trades,
                "open_positions": position_count,
                "open_risk_percent": float(bot_state.open_risk_percent) if bot_state.open_risk_percent else 0,
            }
        else:
            return {
                "trading_enabled": False,
                "mode": "paper",
                "last_heartbeat": None,
                "current_model_version": "v1.0.0",
                "daily_pnl": 0,
                "daily_trades": 0,
                "open_positions": 0,
                "open_risk_percent": 0,
            }
    except Exception as e:
        logger.error("Failed to get bot state", error=str(e))
        return {
            "trading_enabled": False,
            "mode": "paper",
            "last_heartbeat": None,
            "current_model_version": "v1.0.0",
            "daily_pnl": 0,
            "daily_trades": 0,
            "open_positions": 0,
            "open_risk_percent": 0,
            "error": str(e),
        }


@app.post("/api/bot/enable")
async def enable_trading(db: Session = Depends(get_db)):
    """Enable trading."""
    try:
        from database.models import BotState

        bot_state = db.query(BotState).filter(BotState.id == 1).first()
        if bot_state:
            bot_state.trading_enabled = True
            db.commit()

        return {"success": True, "trading_enabled": True}
    except Exception as e:
        logger.error("Failed to enable trading", error=str(e))
        db.rollback()
        return {"success": False, "error": str(e)}


@app.post("/api/bot/disable")
async def disable_trading(db: Session = Depends(get_db)):
    """Disable trading (kill switch)."""
    try:
        from database.models import BotState

        bot_state = db.query(BotState).filter(BotState.id == 1).first()
        if bot_state:
            bot_state.trading_enabled = False
            db.commit()

        return {"success": True, "trading_enabled": False}
    except Exception as e:
        logger.error("Failed to disable trading", error=str(e))
        db.rollback()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
