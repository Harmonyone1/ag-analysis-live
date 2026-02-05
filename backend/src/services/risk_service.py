"""Risk service for risk management operations."""

import sys
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from decimal import Decimal

sys.path.insert(0, "/app/engine/src")

from sqlalchemy.orm import Session
import structlog

logger = structlog.get_logger(__name__)


class RiskService:
    """Service for risk management operations."""

    def __init__(self, db: Session):
        """Initialize service.

        Args:
            db: Database session
        """
        self.db = db

    async def get_risk_events(
        self,
        limit: int = 50,
        severity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get recent risk events.

        Args:
            limit: Maximum events to return
            severity: Filter by severity

        Returns:
            Dict with events list
        """
        try:
            from database.models import RiskEvent

            query = self.db.query(RiskEvent)

            if severity:
                query = query.filter(RiskEvent.severity == severity)

            events = (
                query.order_by(RiskEvent.created_at.desc())
                .limit(limit)
                .all()
            )

            critical_count = sum(1 for e in events if e.severity == "CRITICAL")
            warning_count = sum(1 for e in events if e.severity == "WARNING")

            result = {
                "events": [
                    {
                        "id": str(e.id),
                        "event_type": e.event_type,
                        "severity": e.severity,
                        "message": e.event_type,  # Use event_type as message
                        "payload": e.payload or {},
                        "timestamp": e.created_at.isoformat(),
                        "acknowledged": e.resolved or False,
                    }
                    for e in events
                ],
                "total_count": len(events),
                "critical_count": critical_count,
                "warning_count": warning_count,
            }

            return result

        except Exception as e:
            logger.error("Failed to get risk events", error=str(e))
            raise

    async def get_exposure(self) -> Dict[str, Any]:
        """Get current exposure breakdown.

        Returns:
            Dict with exposure data
        """
        try:
            from database.models import Position, BotState, Setting

            # Get open positions
            positions = (
                self.db.query(Position)
                .filter(Position.close_time.is_(None))
                .all()
            )

            # Get account balance from bot state
            bot_state = (
                self.db.query(BotState)
                .filter(BotState.id == 1)
                .first()
            )
            # Get account balance from settings or use default
            settings = {s.key: s.value for s in self.db.query(Setting).all()}
            account_balance = float(settings.get("account_balance", {}).get("value", 10000.0)) if settings.get("account_balance") else 10000.0

            # Calculate exposure by currency
            currency_exposure: Dict[str, Dict[str, float]] = {}
            symbol_exposure: Dict[str, float] = {}

            for pos in positions:
                # Extract currencies from symbol
                symbol = pos.symbol.upper()
                if len(symbol) == 6:
                    base = symbol[:3]
                    quote = symbol[3:]
                else:
                    base = symbol
                    quote = "USD"

                # Calculate position value
                pos_value = float(pos.quantity) * float(pos.entry_price or 0)
                symbol_exposure[symbol] = symbol_exposure.get(symbol, 0) + pos_value

                # Track by currency
                if base not in currency_exposure:
                    currency_exposure[base] = {"long": 0, "short": 0}
                if quote not in currency_exposure:
                    currency_exposure[quote] = {"long": 0, "short": 0}

                if pos.side == "buy":
                    currency_exposure[base]["long"] += pos_value
                    currency_exposure[quote]["short"] += pos_value
                else:
                    currency_exposure[base]["short"] += pos_value
                    currency_exposure[quote]["long"] += pos_value

            # Build response
            by_currency = []
            total_exposure = 0

            for ccy, exp in currency_exposure.items():
                net = exp["long"] - exp["short"]
                total = exp["long"] + exp["short"]
                total_exposure += total / 2  # Avoid double counting

                by_currency.append({
                    "currency": ccy,
                    "long_exposure": exp["long"],
                    "short_exposure": exp["short"],
                    "net_exposure": net,
                    "exposure_pct": (total / account_balance) * 100 if account_balance else 0,
                })

            # Calculate correlation risk (simplified)
            corr_risk = min(100, len(positions) * 15)  # Rough estimate

            result = {
                "timestamp": datetime.now().isoformat(),
                "total_exposure": total_exposure,
                "exposure_pct": (total_exposure / account_balance) * 100 if account_balance else 0,
                "by_currency": by_currency,
                "by_symbol": symbol_exposure,
                "correlation_risk": corr_risk,
            }

            return result

        except Exception as e:
            logger.error("Failed to get exposure", error=str(e))
            raise

    async def get_risk_state(self) -> Dict[str, Any]:
        """Get current risk state.

        Returns:
            Dict with risk state
        """
        try:
            from database.models import Position, BotState, RiskEvent, Setting

            # Get bot state
            bot_state = (
                self.db.query(BotState)
                .filter(BotState.id == 1)
                .first()
            )

            # Get settings
            settings = {
                s.key: s.value
                for s in self.db.query(Setting).all()
            }

            # Get position count
            position_count = (
                self.db.query(Position)
                .filter(Position.close_time.is_(None))
                .count()
            )

            # Get recent events
            recent_events = (
                self.db.query(RiskEvent)
                .order_by(RiskEvent.created_at.desc())
                .limit(5)
                .all()
            )

            # Parse risk limits from settings
            max_risk_per_trade = float(settings.get("max_risk_per_trade", 0.01))
            max_daily_loss = float(settings.get("max_daily_loss", 0.03))
            max_positions = int(settings.get("max_open_positions", 5))

            # Get account balance from settings or use default
            account_balance = float(settings.get("account_balance", {}).get("value", 10000.0)) if settings.get("account_balance") else 10000.0
            daily_pnl = float(bot_state.daily_pnl) if bot_state else 0.0
            daily_pnl_pct = (daily_pnl / account_balance) * 100 if account_balance else 0

            result = {
                "trading_enabled": bot_state.trading_enabled if bot_state else True,
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": daily_pnl_pct,
                "daily_trades": bot_state.daily_trades if bot_state else 0,
                "open_positions": position_count,
                "max_positions": max_positions,
                "current_exposure_pct": 0,  # Would calculate from positions
                "max_daily_loss_pct": max_daily_loss * 100,
                "at_daily_limit": daily_pnl_pct <= -max_daily_loss * 100,
                "at_position_limit": position_count >= max_positions,
                "limits": {
                    "max_risk_per_trade": max_risk_per_trade,
                    "max_daily_loss": max_daily_loss,
                    "max_open_positions": max_positions,
                    "max_correlated_exposure": float(settings.get("max_correlated_exposure", 0.02)),
                    "max_spread_multiplier": float(settings.get("max_spread_multiplier", 3.0)),
                    "slippage_halt_threshold": float(settings.get("slippage_halt_threshold", 5.0)),
                },
                "recent_events": [
                    {
                        "id": str(e.id),
                        "event_type": e.event_type,
                        "severity": e.severity,
                        "message": e.event_type,  # Use event_type as message
                        "payload": e.payload or {},
                        "timestamp": e.created_at.isoformat(),
                        "acknowledged": e.resolved or False,
                    }
                    for e in recent_events
                ],
            }

            return result

        except Exception as e:
            logger.error("Failed to get risk state", error=str(e))
            raise

    async def check_trade_risk(
        self,
        symbol: str,
        side: str,
        risk_amount: float,
        spread: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check if a trade passes risk filters.

        Args:
            symbol: Trading symbol
            side: Trade side
            risk_amount: Risk in account currency
            spread: Current spread

        Returns:
            Dict with risk check result
        """
        try:
            from database.models import Position, BotState, Setting

            warnings = []

            # Get settings
            settings = {
                s.key: s.value
                for s in self.db.query(Setting).all()
            }

            # Get bot state
            bot_state = (
                self.db.query(BotState)
                .filter(BotState.id == 1)
                .first()
            )

            # Check trading enabled
            if bot_state and not bot_state.trading_enabled:
                return {
                    "passed": False,
                    "reason": "Trading is disabled",
                    "suggested_size_multiplier": 0,
                    "warnings": [],
                }

            # Get position count
            position_count = (
                self.db.query(Position)
                .filter(Position.close_time.is_(None))
                .count()
            )

            max_positions = int(settings.get("max_open_positions", 5))
            if position_count >= max_positions:
                return {
                    "passed": False,
                    "reason": f"Max positions ({max_positions}) reached",
                    "suggested_size_multiplier": 0,
                    "warnings": [],
                }

            # Check max risk per trade
            # Get account balance from settings or use default
            account_balance = float(settings.get("account_balance", {}).get("value", 10000.0)) if settings.get("account_balance") else 10000.0
            max_risk_pct = float(settings.get("max_risk_per_trade", 0.01))
            max_risk = account_balance * max_risk_pct

            if risk_amount > max_risk:
                size_mult = max_risk / risk_amount
                warnings.append(f"Risk reduced to {max_risk_pct*100}% of account")
                return {
                    "passed": True,
                    "reason": "Risk exceeds limit, size reduced",
                    "suggested_size_multiplier": size_mult,
                    "warnings": warnings,
                }

            # Check spread if provided
            if spread:
                max_spread_mult = float(settings.get("max_spread_multiplier", 3.0))
                # Would need typical spread data to compare
                if spread > 5:  # Rough check
                    warnings.append("Elevated spread detected")

            return {
                "passed": True,
                "reason": "All risk checks passed",
                "suggested_size_multiplier": 1.0,
                "warnings": warnings,
            }

        except Exception as e:
            logger.error("Failed to check trade risk", error=str(e))
            return {
                "passed": False,
                "reason": str(e),
                "suggested_size_multiplier": 0,
                "warnings": [],
            }

    async def toggle_trading(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable trading.

        Args:
            enabled: Whether to enable trading

        Returns:
            Dict with result
        """
        try:
            from database.models import BotState, RiskEvent
            import uuid

            bot_state = (
                self.db.query(BotState)
                .filter(BotState.id == 1)
                .first()
            )

            if bot_state:
                bot_state.trading_enabled = enabled

                # Log risk event
                event = RiskEvent(
                    id=uuid.uuid4(),
                    event_type="KILL_SWITCH" if not enabled else "TRADING_ENABLED",
                    severity="WARNING" if not enabled else "INFO",
                    payload={"action": "manual", "message": "Trading disabled via kill switch" if not enabled else "Trading enabled"},
                )
                self.db.add(event)
                self.db.commit()

            return {
                "success": True,
                "trading_enabled": enabled,
                "message": f"Trading {'enabled' if enabled else 'disabled'}",
            }

        except Exception as e:
            logger.error("Failed to toggle trading", error=str(e))
            self.db.rollback()
            return {
                "success": False,
                "message": str(e),
            }

    async def acknowledge_event(self, event_id: str) -> Dict[str, Any]:
        """Acknowledge a risk event.

        Args:
            event_id: Event ID

        Returns:
            Dict with result
        """
        try:
            from database.models import RiskEvent

            event = (
                self.db.query(RiskEvent)
                .filter(RiskEvent.id == event_id)
                .first()
            )

            if not event:
                return {
                    "success": False,
                    "message": "Event not found",
                }

            event.resolved = True
            event.resolved_at = datetime.now()
            self.db.commit()

            return {
                "success": True,
                "message": "Event acknowledged",
            }

        except Exception as e:
            logger.error("Failed to acknowledge event", error=str(e))
            self.db.rollback()
            return {
                "success": False,
                "message": str(e),
            }
