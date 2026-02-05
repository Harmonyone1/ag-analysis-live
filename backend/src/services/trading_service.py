"""Trading service for order and position management."""

import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
import uuid

sys.path.insert(0, "/app/engine/src")

from sqlalchemy.orm import Session
import structlog

logger = structlog.get_logger(__name__)


class TradingService:
    """Service for trading operations."""

    def __init__(self, db: Session):
        """Initialize service.

        Args:
            db: Database session
        """
        self.db = db

    async def get_positions(self) -> Dict[str, Any]:
        """Get open positions.

        Returns:
            Dict with positions list
        """
        try:
            from database.models import Position

            positions = (
                self.db.query(Position)
                .filter(Position.close_time.is_(None))
                .order_by(Position.open_time.desc())
                .all()
            )

            total_pnl = sum(
                float(p.unrealized_pnl or 0) for p in positions
            )

            result = {
                "positions": [
                    {
                        "id": str(p.id),
                        "broker_position_id": p.broker_position_id,
                        "symbol": p.symbol,
                        "side": p.side,
                        "quantity": float(p.quantity),
                        "entry_price": float(p.avg_entry_price),
                        "current_price": None,  # Would need to fetch from broker
                        "stop_loss": float(p.current_stop_loss) if p.current_stop_loss else None,
                        "take_profit": float(p.current_take_profit) if p.current_take_profit else None,
                        "unrealized_pnl": float(p.unrealized_pnl) if p.unrealized_pnl else 0,
                        "unrealized_pnl_pct": self._calc_pnl_pct(p),
                        "opened_at": p.open_time.isoformat(),
                        "candidate_id": str(p.candidate_id) if p.candidate_id else None,
                    }
                    for p in positions
                ],
                "total_count": len(positions),
                "total_unrealized_pnl": total_pnl,
            }

            return result

        except Exception as e:
            logger.error("Failed to get positions", error=str(e))
            raise

    async def get_orders(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get orders.

        Args:
            status: Filter by status
            limit: Maximum orders to return

        Returns:
            Dict with orders list
        """
        try:
            from database.models import Order

            query = self.db.query(Order)

            if status:
                query = query.filter(Order.status == status)

            orders = (
                query.order_by(Order.created_at.desc())
                .limit(limit)
                .all()
            )

            result = {
                "orders": [
                    {
                        "id": str(o.id),
                        "broker_order_id": o.broker_order_id,
                        "symbol": o.symbol,
                        "side": o.side,
                        "order_type": o.order_type,
                        "quantity": float(o.quantity),
                        "price": float(o.price) if o.price else None,
                        "stop_loss": float(o.stop_loss) if o.stop_loss else None,
                        "take_profit": float(o.take_profit) if o.take_profit else None,
                        "status": o.status,
                        "created_at": o.created_at.isoformat(),
                        "filled_at": None,  # Would come from executions
                        "fill_price": None,  # Would come from executions
                        "fill_quantity": float(o.filled_qty) if o.filled_qty else None,
                    }
                    for o in orders
                ],
                "total_count": len(orders),
            }

            return result

        except Exception as e:
            logger.error("Failed to get orders", error=str(e))
            raise

    async def get_executions(self, limit: int = 50) -> Dict[str, Any]:
        """Get recent executions.

        Args:
            limit: Maximum executions to return

        Returns:
            Dict with executions list
        """
        try:
            from database.models import Execution, Order

            executions = (
                self.db.query(Execution)
                .order_by(Execution.fill_time.desc())
                .limit(limit)
                .all()
            )

            result = {
                "executions": [
                    {
                        "id": str(e.id),
                        "order_id": str(e.order_id),
                        "broker_trade_id": e.broker_trade_id,
                        "quantity": float(e.fill_qty),
                        "price": float(e.fill_price),
                        "fees": float(e.fees) if e.fees else 0,
                        "slippage_pips": float(e.slippage_pips) if e.slippage_pips else None,
                        "executed_at": e.fill_time.isoformat(),
                    }
                    for e in executions
                ],
                "total_count": len(executions),
            }

            return result

        except Exception as e:
            logger.error("Failed to get executions", error=str(e))
            raise

    async def approve_candidate(
        self,
        candidate_id: str,
        position_size: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Approve a trade candidate for execution.

        Args:
            candidate_id: Candidate ID to approve
            position_size: Optional override for position size
            notes: Optional notes

        Returns:
            Dict with approval result
        """
        try:
            from database.models import TradeCandidate, Order

            # Get candidate
            candidate = (
                self.db.query(TradeCandidate)
                .filter(TradeCandidate.id == candidate_id)
                .first()
            )

            if not candidate:
                return {
                    "success": False,
                    "candidate_id": candidate_id,
                    "message": "Candidate not found",
                }

            if candidate.status not in ("new", "pending"):
                return {
                    "success": False,
                    "candidate_id": candidate_id,
                    "message": f"Candidate already {candidate.status}",
                }

            # Check if expired
            if candidate.expires_at and candidate.expires_at < datetime.now():
                candidate.status = "expired"
                self.db.commit()
                return {
                    "success": False,
                    "candidate_id": candidate_id,
                    "message": "Candidate has expired",
                }

            # Update candidate status
            candidate.status = "approved"

            # Create order
            order_id = uuid.uuid4()
            entry_price = None
            if candidate.entry_zone:
                # entry_zone is a dict with "min" and "max" keys
                min_price = candidate.entry_zone.get("min", 0)
                max_price = candidate.entry_zone.get("max", 0)
                entry_price = Decimal(str((min_price + max_price) / 2))

            order = Order(
                id=order_id,
                candidate_id=candidate.id,
                symbol=candidate.symbol,
                side="buy" if candidate.direction == "LONG" else "sell",
                order_type="limit" if entry_price else "market",
                quantity=Decimal(str(position_size or 0.1)),
                price=entry_price,
                stop_loss=candidate.stop_price,
                take_profit=Decimal(str(candidate.tp_targets[0]["price"])) if candidate.tp_targets else None,
                status="pending",
            )

            self.db.add(order)
            self.db.commit()

            logger.info(
                "Candidate approved",
                candidate_id=candidate_id,
                order_id=str(order_id),
                symbol=candidate.symbol,
            )

            return {
                "success": True,
                "candidate_id": candidate_id,
                "order_id": str(order_id),
                "message": "Trade approved and order created",
                "execution_result": {
                    "symbol": candidate.symbol,
                    "direction": candidate.direction,
                    "order_type": order.order_type,
                },
            }

        except Exception as e:
            logger.error("Failed to approve candidate", error=str(e))
            self.db.rollback()
            return {
                "success": False,
                "candidate_id": candidate_id,
                "message": str(e),
            }

    async def reject_candidate(
        self,
        candidate_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reject a trade candidate.

        Args:
            candidate_id: Candidate ID to reject
            reason: Optional rejection reason

        Returns:
            Dict with rejection result
        """
        try:
            from database.models import TradeCandidate

            candidate = (
                self.db.query(TradeCandidate)
                .filter(TradeCandidate.id == candidate_id)
                .first()
            )

            if not candidate:
                return {
                    "success": False,
                    "candidate_id": candidate_id,
                    "message": "Candidate not found",
                }

            if candidate.status not in ("new", "pending"):
                return {
                    "success": False,
                    "candidate_id": candidate_id,
                    "message": f"Candidate already {candidate.status}",
                }

            candidate.status = "rejected"
            self.db.commit()

            logger.info(
                "Candidate rejected",
                candidate_id=candidate_id,
                reason=reason,
            )

            return {
                "success": True,
                "candidate_id": candidate_id,
                "message": "Trade rejected",
            }

        except Exception as e:
            logger.error("Failed to reject candidate", error=str(e))
            self.db.rollback()
            return {
                "success": False,
                "candidate_id": candidate_id,
                "message": str(e),
            }

    async def modify_position(
        self,
        position_id: str,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Modify position stop loss or take profit.

        Args:
            position_id: Position ID
            new_stop_loss: New stop loss price
            new_take_profit: New take profit price

        Returns:
            Dict with result
        """
        try:
            from database.models import Position

            position = (
                self.db.query(Position)
                .filter(Position.id == position_id)
                .first()
            )

            if not position:
                return {
                    "success": False,
                    "message": "Position not found",
                }

            if new_stop_loss:
                position.current_stop_loss = Decimal(str(new_stop_loss))
            if new_take_profit:
                position.current_take_profit = Decimal(str(new_take_profit))

            self.db.commit()

            return {
                "success": True,
                "message": "Position modified",
                "position_id": position_id,
            }

        except Exception as e:
            logger.error("Failed to modify position", error=str(e))
            self.db.rollback()
            return {
                "success": False,
                "message": str(e),
            }

    async def close_position(
        self,
        position_id: str,
        quantity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Close a position.

        Args:
            position_id: Position ID
            quantity: Quantity to close (None = full close)

        Returns:
            Dict with result
        """
        try:
            from database.models import Position

            position = (
                self.db.query(Position)
                .filter(Position.id == position_id)
                .first()
            )

            if not position:
                return {
                    "success": False,
                    "message": "Position not found",
                }

            if quantity and Decimal(str(quantity)) < position.quantity:
                # Partial close
                position.quantity -= Decimal(str(quantity))
                message = f"Partial close of {quantity}"
            else:
                # Full close
                position.close_time = datetime.now()
                position.close_reason = "MANUAL"
                message = "Position closed"

            self.db.commit()

            return {
                "success": True,
                "message": message,
                "position_id": position_id,
            }

        except Exception as e:
            logger.error("Failed to close position", error=str(e))
            self.db.rollback()
            return {
                "success": False,
                "message": str(e),
            }

    def _calc_pnl_pct(self, position) -> Optional[float]:
        """Calculate P&L percentage."""
        if not position.avg_entry_price or not position.unrealized_pnl:
            return None

        entry = float(position.avg_entry_price)
        pnl = float(position.unrealized_pnl)
        pos_value = entry * float(position.quantity)

        if pos_value > 0:
            return (pnl / pos_value) * 100
        return None
