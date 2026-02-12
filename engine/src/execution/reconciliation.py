"""Position Reconciliation Engine.

Handles:
- Syncing local state with broker positions
- Detecting orphaned positions
- Reconciling fills and partial fills
- Position state consistency
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set
import structlog

from src.adapters.broker import BrokerAdapter, Position as BrokerPosition

logger = structlog.get_logger(__name__)


class ReconciliationStatus(Enum):
    """Status of reconciliation."""
    SYNCED = "synced"
    MISMATCH = "mismatch"
    ORPHANED = "orphaned"
    MISSING = "missing"


@dataclass
class LocalPosition:
    """Local position record."""
    position_id: str
    broker_position_id: Optional[int]
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    opened_at: datetime
    candidate_id: Optional[str] = None


@dataclass
class ReconciliationResult:
    """Result of position reconciliation."""
    status: ReconciliationStatus
    synced_count: int
    mismatch_count: int
    orphaned_broker: List[BrokerPosition]
    missing_local: List[str]
    mismatches: List[Dict]
    message: str


class PositionReconciler:
    """Reconciles local position state with broker positions.

    Ensures consistency between the trading engine's view
    of open positions and the actual broker state.

    Example:
        reconciler = PositionReconciler(broker)
        result = reconciler.reconcile(local_positions)
        if result.orphaned_broker:
            handle_orphaned_positions(result.orphaned_broker)
    """

    def __init__(self, broker: BrokerAdapter):
        """Initialize reconciler.

        Args:
            broker: Broker adapter for position queries
        """
        self.broker = broker
        self._last_reconciliation: Optional[datetime] = None
        self._reconciliation_history: List[ReconciliationResult] = []

    def reconcile(
        self,
        local_positions: List[LocalPosition],
    ) -> ReconciliationResult:
        """Reconcile local positions with broker positions.

        Args:
            local_positions: List of locally tracked positions

        Returns:
            ReconciliationResult with sync status
        """
        try:
            # Get broker positions
            broker_positions = self.broker.get_positions()

            # Index by broker position ID
            broker_by_id: Dict[int, BrokerPosition] = {
                p.position_id: p for p in broker_positions
            }
            local_by_broker_id: Dict[int, LocalPosition] = {
                int(p.broker_position_id): p
                for p in local_positions
                if p.broker_position_id is not None
            }

            synced = 0
            mismatches = []
            orphaned_broker = []
            missing_local = []

            # Check each broker position
            broker_ids_checked: Set[int] = set()
            for bp in broker_positions:
                broker_ids_checked.add(bp.position_id)

                if bp.position_id in local_by_broker_id:
                    local = local_by_broker_id[bp.position_id]
                    mismatch = self._check_mismatch(local, bp)
                    if mismatch:
                        mismatches.append(mismatch)
                    else:
                        synced += 1
                else:
                    # Broker position not in local state
                    orphaned_broker.append(bp)
                    logger.warning(
                        "Orphaned broker position",
                        position_id=bp.position_id,
                        symbol=bp.symbol,
                    )

            # Check for local positions not on broker
            for local in local_positions:
                if local.broker_position_id:
                    if int(local.broker_position_id) not in broker_ids_checked:
                        missing_local.append(local.position_id)
                        logger.warning(
                            "Local position missing from broker",
                            position_id=local.position_id,
                            broker_id=local.broker_position_id,
                        )

            # Determine overall status
            if mismatches or orphaned_broker or missing_local:
                status = ReconciliationStatus.MISMATCH
                message = (
                    f"Reconciliation issues: {len(mismatches)} mismatches, "
                    f"{len(orphaned_broker)} orphaned, {len(missing_local)} missing"
                )
            else:
                status = ReconciliationStatus.SYNCED
                message = f"All {synced} positions synced"

            result = ReconciliationResult(
                status=status,
                synced_count=synced,
                mismatch_count=len(mismatches),
                orphaned_broker=orphaned_broker,
                missing_local=missing_local,
                mismatches=mismatches,
                message=message,
            )

            self._last_reconciliation = datetime.now()
            self._reconciliation_history.append(result)

            # Keep only last 50 reconciliations
            if len(self._reconciliation_history) > 50:
                self._reconciliation_history = self._reconciliation_history[-50:]

            logger.info(
                "Position reconciliation complete",
                status=status.value,
                synced=synced,
                mismatches=len(mismatches),
            )

            return result

        except Exception as e:
            logger.error("Reconciliation failed", error=str(e))
            return ReconciliationResult(
                status=ReconciliationStatus.MISMATCH,
                synced_count=0,
                mismatch_count=0,
                orphaned_broker=[],
                missing_local=[],
                mismatches=[],
                message=f"Reconciliation error: {e}",
            )

    def _check_mismatch(
        self,
        local: LocalPosition,
        broker: BrokerPosition,
    ) -> Optional[Dict]:
        """Check for mismatches between local and broker position."""
        issues = []

        # Check symbol
        if local.symbol.upper() != broker.symbol.upper():
            issues.append({
                "field": "symbol",
                "local": local.symbol,
                "broker": broker.symbol,
            })

        # Check side
        local_side = local.side.lower()
        broker_side = broker.side.lower()
        if local_side != broker_side:
            issues.append({
                "field": "side",
                "local": local_side,
                "broker": broker_side,
            })

        # Check quantity (allow small tolerance for partial fills)
        qty_diff = abs(local.quantity - broker.quantity)
        if qty_diff > Decimal("0.001"):
            issues.append({
                "field": "quantity",
                "local": float(local.quantity),
                "broker": float(broker.quantity),
            })

        # Check stop loss
        if local.stop_loss and broker.stop_loss:
            sl_diff = abs(local.stop_loss - broker.stop_loss)
            if sl_diff > Decimal("0.0001"):
                issues.append({
                    "field": "stop_loss",
                    "local": float(local.stop_loss),
                    "broker": float(broker.stop_loss),
                })

        if issues:
            return {
                "position_id": local.position_id,
                "broker_position_id": broker.position_id,
                "symbol": local.symbol,
                "issues": issues,
            }

        return None

    def sync_position_from_broker(
        self,
        broker_position: BrokerPosition,
    ) -> LocalPosition:
        """Create local position record from broker position.

        Args:
            broker_position: Broker position to sync

        Returns:
            New LocalPosition record
        """
        return LocalPosition(
            position_id=f"SYNC-{broker_position.position_id}",
            broker_position_id=broker_position.position_id,
            symbol=broker_position.symbol,
            side=broker_position.side,
            quantity=broker_position.quantity,
            entry_price=broker_position.avg_price,
            stop_loss=broker_position.stop_loss,
            take_profit=broker_position.take_profit,
            opened_at=datetime.now(),
        )

    def get_orphaned_positions(
        self,
        local_positions: List[LocalPosition],
    ) -> List[BrokerPosition]:
        """Get broker positions not tracked locally.

        Args:
            local_positions: Currently tracked positions

        Returns:
            List of untracked broker positions
        """
        result = self.reconcile(local_positions)
        return result.orphaned_broker

    @property
    def last_reconciliation_time(self) -> Optional[datetime]:
        """Get timestamp of last reconciliation."""
        return self._last_reconciliation

    @property
    def reconciliation_history(self) -> List[ReconciliationResult]:
        """Get recent reconciliation history."""
        return self._reconciliation_history[-10:]
