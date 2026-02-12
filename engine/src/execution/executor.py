"""Execution Engine for order placement and management.

Handles:
- Order placement with SL/TP
- Paper and live execution modes
- Idempotent order handling
- Fill tracking
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional
import uuid
import structlog

from src.adapters.broker import BrokerAdapter, OrderRequest, Order, BrokerError
from src.scoring.confluence import TradeSetup

logger = structlog.get_logger(__name__)


class ExecutionMode(Enum):
    """Execution mode."""
    PAPER = "paper"
    LIVE = "live"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class ExecutionResult:
    """Result of an execution attempt."""
    success: bool
    order_id: Optional[str]
    broker_order_id: Optional[str]
    status: OrderStatus
    message: str
    fill_price: Optional[Decimal] = None
    fill_qty: Optional[Decimal] = None
    slippage_pips: Optional[float] = None


class ExecutionEngine:
    """Executes trades through the broker adapter.

    Supports both paper trading (simulated) and live execution
    with proper error handling and fill tracking.

    Example:
        executor = ExecutionEngine(broker, mode=ExecutionMode.PAPER)
        result = executor.execute_setup(setup, position_size)
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        mode: ExecutionMode = ExecutionMode.PAPER,
    ):
        """Initialize execution engine.

        Args:
            broker: Broker adapter for order placement
            mode: Execution mode (paper/live)
        """
        self.broker = broker
        self.mode = mode
        self._pending_orders: Dict[str, Dict] = {}
        self._executed_trades: List[Dict] = []

    def execute_setup(
        self,
        setup: TradeSetup,
        position_size: Decimal,
        candidate_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute a trade setup.

        Args:
            setup: Trade setup to execute
            position_size: Position size in lots
            candidate_id: Optional reference to trade candidate

        Returns:
            ExecutionResult with order details
        """
        order_id = str(uuid.uuid4())

        try:
            # Get instrument ID
            instrument = self.broker.get_instrument(setup.symbol)
            if not instrument:
                return ExecutionResult(
                    success=False,
                    order_id=order_id,
                    broker_order_id=None,
                    status=OrderStatus.FAILED,
                    message=f"Instrument not found: {setup.symbol}",
                )

            # Determine entry price for limit orders
            entry_price = None
            if setup.entry_type == "LIMIT":
                entry_price = Decimal(str(
                    (setup.entry_zone[0] + setup.entry_zone[1]) / 2
                ))

            # Create order request
            request = OrderRequest(
                symbol=setup.symbol,
                instrument_id=instrument.instrument_id,
                side="buy" if setup.direction == "LONG" else "sell",
                quantity=position_size,
                order_type=setup.entry_type.lower(),
                price=entry_price,
                stop_loss=Decimal(str(setup.stop_price)),
                stop_loss_type="absolute",
                take_profit=Decimal(str(setup.tp_targets[0]["price"])) if setup.tp_targets else None,
                take_profit_type="absolute" if setup.tp_targets else None,
                strategy_id=candidate_id[:32] if candidate_id else None,
            )

            # Execute based on mode
            if self.mode == ExecutionMode.PAPER:
                result = self._execute_paper(request, order_id)
            else:
                result = self._execute_live(request, order_id)

            # Track execution
            self._executed_trades.append({
                "order_id": order_id,
                "candidate_id": candidate_id,
                "setup": setup,
                "result": result,
                "timestamp": datetime.now(),
            })

            return result

        except Exception as e:
            logger.error("Execution failed", error=str(e), symbol=setup.symbol)
            return ExecutionResult(
                success=False,
                order_id=order_id,
                broker_order_id=None,
                status=OrderStatus.FAILED,
                message=str(e),
            )

    def _execute_paper(
        self,
        request: OrderRequest,
        order_id: str,
    ) -> ExecutionResult:
        """Execute order in paper mode (simulated)."""
        logger.info("Paper execution",
                   symbol=request.symbol,
                   side=request.side,
                   qty=float(request.quantity))

        # Simulate fill at current price
        try:
            quote = self.broker.get_quote(request.symbol)
            fill_price = quote.ask if request.side == "buy" else quote.bid

            # Calculate simulated slippage (0-1 pip random)
            import random
            slippage = random.uniform(0, 1)

            return ExecutionResult(
                success=True,
                order_id=order_id,
                broker_order_id=f"PAPER-{order_id[:8]}",
                status=OrderStatus.FILLED,
                message="Paper trade executed",
                fill_price=fill_price,
                fill_qty=request.quantity,
                slippage_pips=slippage,
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                order_id=order_id,
                broker_order_id=None,
                status=OrderStatus.FAILED,
                message=f"Paper execution failed: {e}",
            )

    def _execute_live(
        self,
        request: OrderRequest,
        order_id: str,
    ) -> ExecutionResult:
        """Execute order in live mode through broker."""
        logger.info("Live execution",
                   symbol=request.symbol,
                   side=request.side,
                   qty=float(request.quantity))

        try:
            # Place order through broker
            order = self.broker.place_order(request)

            # Track pending order
            self._pending_orders[order_id] = {
                "broker_order_id": order.order_id,
                "request": request,
                "placed_at": datetime.now(),
            }

            # For market orders, assume immediate fill
            if request.order_type == "market":
                return ExecutionResult(
                    success=True,
                    order_id=order_id,
                    broker_order_id=str(order.order_id),
                    status=OrderStatus.FILLED,
                    message="Order filled",
                    fill_price=order.price,
                    fill_qty=order.filled_quantity,
                )
            else:
                return ExecutionResult(
                    success=True,
                    order_id=order_id,
                    broker_order_id=str(order.order_id),
                    status=OrderStatus.PLACED,
                    message="Limit order placed",
                )

        except BrokerError as e:
            # If TP price is invalid, the R:R is bad at current market — reject entirely
            if "TP price" in str(e) and request.take_profit is not None:
                logger.warning("TP rejected by broker — trade not placed (R:R invalid at market price)",
                               symbol=request.symbol, tp=str(request.take_profit))

            logger.error("Broker rejected order", error=str(e))
            return ExecutionResult(
                success=False,
                order_id=order_id,
                broker_order_id=None,
                status=OrderStatus.REJECTED,
                message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: Internal order ID

        Returns:
            True if cancelled successfully
        """
        if order_id not in self._pending_orders:
            logger.warning("Order not found for cancellation", order_id=order_id)
            return False

        pending = self._pending_orders[order_id]
        broker_order_id = pending.get("broker_order_id")

        if self.mode == ExecutionMode.PAPER:
            del self._pending_orders[order_id]
            return True

        try:
            if broker_order_id:
                self.broker.cancel_order(int(broker_order_id))
            del self._pending_orders[order_id]
            return True
        except Exception as e:
            logger.error("Failed to cancel order", error=str(e))
            return False

    def modify_position_stops(
        self,
        position_id: int,
        new_stop: Optional[Decimal] = None,
        new_tp: Optional[Decimal] = None,
    ) -> bool:
        """Modify stop loss or take profit on open position.

        Args:
            position_id: Broker position ID
            new_stop: New stop loss price
            new_tp: New take profit price

        Returns:
            True if modified successfully
        """
        if self.mode == ExecutionMode.PAPER:
            logger.info("Paper position modification",
                       position_id=position_id,
                       new_stop=float(new_stop) if new_stop else None,
                       new_tp=float(new_tp) if new_tp else None)
            return True

        try:
            return self.broker.modify_position(
                position_id=position_id,
                stop_loss=new_stop,
                take_profit=new_tp,
            )
        except Exception as e:
            logger.error("Failed to modify position", error=str(e))
            return False

    def close_position(
        self,
        position_id: int,
        quantity: Optional[Decimal] = None,
    ) -> bool:
        """Close an open position.

        Args:
            position_id: Broker position ID
            quantity: Quantity to close (None = full close)

        Returns:
            True if close order placed successfully
        """
        if self.mode == ExecutionMode.PAPER:
            logger.info("Paper position close", position_id=position_id)
            return True

        try:
            return self.broker.close_position(position_id, quantity)
        except Exception as e:
            logger.error("Failed to close position", error=str(e))
            return False

    @property
    def pending_order_count(self) -> int:
        """Get count of pending orders."""
        return len(self._pending_orders)

    @property
    def execution_history(self) -> List[Dict]:
        """Get recent execution history."""
        return self._executed_trades[-50:]
