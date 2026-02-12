"""Live trade management: breakeven, trailing stop, and time stop.

Runs every cycle (~15s) to actively manage open positions on the broker.
Adapts the proven logic from trade_manager.py (paper) for live execution.
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any

import structlog

from src.adapters import BrokerAdapter
from src.database.models import Position

logger = structlog.get_logger(__name__)


# ── Session classification (same as trade_review.py) ──

def get_session(utc_hour: int) -> str:
    """Classify UTC hour into trading session."""
    if 12 <= utc_hour < 16:
        return "peak"
    elif 8 <= utc_hour < 12:
        return "london"
    elif 16 <= utc_hour < 20:
        return "ny"
    elif 20 <= utc_hour < 24:
        return "late_ny"
    else:
        return "asia"


# ── Configuration ──

@dataclass
class LiveTradeConfig:
    """Configuration for live trade management."""

    # Break-even: move SL to entry + buffer when trade reaches this R-multiple
    breakeven_trigger_r: float = 1.0
    breakeven_buffer_pips: float = 2.0

    # Trailing: start trailing SL when trade reaches this R-multiple
    trailing_trigger_r: float = 1.5
    trailing_distance_r: float = 0.5  # Trail 0.5R behind price

    # Time stop: close trade after this many hours
    max_duration_hours: int = 48

    # Session overrides (merged onto base config per position)
    session_overrides: Dict[str, Dict] = field(default_factory=lambda: {
        "peak": {
            "trailing_trigger_r": 1.2,     # Trail earlier in peak
            "trailing_distance_r": 0.4,    # Tighter trail to capture more
            "max_duration_hours": 72,      # Let peak winners ride longer
        },
        "asia": {
            "max_duration_hours": 24,      # Shorter timeout for low-vol session
        },
    })

    def for_session(self, session: str) -> "LiveTradeConfig":
        """Return a copy of this config with session overrides applied."""
        overrides = self.session_overrides.get(session, {})
        if not overrides:
            return self
        return LiveTradeConfig(
            breakeven_trigger_r=overrides.get("breakeven_trigger_r", self.breakeven_trigger_r),
            breakeven_buffer_pips=overrides.get("breakeven_buffer_pips", self.breakeven_buffer_pips),
            trailing_trigger_r=overrides.get("trailing_trigger_r", self.trailing_trigger_r),
            trailing_distance_r=overrides.get("trailing_distance_r", self.trailing_distance_r),
            max_duration_hours=overrides.get("max_duration_hours", self.max_duration_hours),
            session_overrides={},  # No recursion
        )


# ── Live Trade Manager ──

class LiveTradeManager:
    """Manages open positions: breakeven moves, trailing stops, time stops.

    Called every ~15s from _monitor_positions(). Works with real broker
    positions and modifies SL/TP via broker.modify_position().

    Safety invariant: stops are only ever tightened, never widened.
    """

    def __init__(
        self,
        config: Optional[LiveTradeConfig] = None,
        broker: Optional[BrokerAdapter] = None,
    ):
        self.config = config or LiveTradeConfig()
        self.broker = broker

    def manage_positions(
        self,
        broker_positions: List,
        db_session_factory: Callable,
    ) -> None:
        """Run trade management on all open positions.

        Args:
            broker_positions: List of broker Position objects from get_positions()
            db_session_factory: Context manager factory for DB sessions (db.session)
        """
        if not broker_positions or not self.broker:
            return

        utc_now = datetime.now(timezone.utc)
        session_name = get_session(utc_now.hour)
        cfg = self.config.for_session(session_name)

        with db_session_factory() as db_session:
            for bp in broker_positions:
                try:
                    self._manage_single_position(bp, cfg, utc_now, db_session)
                except Exception as e:
                    print(f"[TRADE_MGMT] ERROR managing {bp.symbol} "
                          f"pos={bp.position_id}: {e}", file=sys.stderr)

            db_session.commit()

    def _manage_single_position(
        self,
        bp: Any,
        cfg: LiveTradeConfig,
        utc_now: datetime,
        db_session: Any,
    ) -> None:
        """Apply trade management rules to a single position."""
        # Look up DB record for this broker position
        db_pos = db_session.query(Position).filter(
            Position.broker_position_id == str(bp.position_id)
        ).first()

        if not db_pos:
            return  # Not tracked in DB (e.g. manual trade)

        symbol = bp.symbol.upper()
        side = bp.side  # "buy" or "sell"
        is_long = side == "buy"

        # Resolve prices
        entry = float(db_pos.avg_entry_price)
        current_sl = float(bp.stop_loss) if bp.stop_loss else None

        # Get original SL (from DB field, fall back to current SL)
        original_sl = float(db_pos.original_stop_loss) if db_pos.original_stop_loss else current_sl
        if not original_sl or original_sl == 0:
            return  # Can't compute R without SL

        # Compute risk (entry-to-SL distance in price)
        if is_long:
            risk = entry - original_sl
        else:
            risk = original_sl - entry

        if risk <= 0:
            return  # Invalid SL placement (SL is on wrong side of entry)

        # Get current market price
        try:
            quote = self.broker.get_quote(symbol)
            # Use exit price: bid for longs (sell to close), ask for shorts (buy to close)
            current_price = float(quote.bid) if is_long else float(quote.ask)
        except Exception:
            return  # Can't get price, skip this cycle

        # Compute R-multiple
        if is_long:
            r_multiple = (current_price - entry) / risk
        else:
            r_multiple = (entry - current_price) / risk

        # Infer state for pre-migration positions (break_even_moved is NULL)
        be_moved = db_pos.break_even_moved if db_pos.break_even_moved is not None else False
        trail_active = db_pos.trailing_active if db_pos.trailing_active is not None else False

        # If state is unknown but SL is tighter than original, infer BE was done
        if not be_moved and current_sl and original_sl:
            if is_long and current_sl > original_sl:
                be_moved = True
                db_pos.break_even_moved = True
            elif not is_long and current_sl < original_sl:
                be_moved = True
                db_pos.break_even_moved = True

        is_jpy = "JPY" in symbol
        pip_size = Decimal("0.01") if is_jpy else Decimal("0.0001")

        # ── 1. Breakeven ──
        if not be_moved and r_multiple >= cfg.breakeven_trigger_r:
            buffer = float(cfg.breakeven_buffer_pips) * float(pip_size)
            if is_long:
                new_sl = entry + buffer
            else:
                new_sl = entry - buffer

            # Safety: only tighten (new SL must be more protective than current)
            if current_sl is None or (is_long and new_sl > current_sl) or (not is_long and new_sl < current_sl):
                new_sl_dec = Decimal(str(round(new_sl, 5 if not is_jpy else 3)))
                try:
                    self.broker.modify_position(
                        position_id=int(bp.position_id),
                        stop_loss=new_sl_dec,
                    )
                    db_pos.current_stop_loss = new_sl_dec
                    db_pos.break_even_moved = True
                    be_moved = True
                    print(f"[TRADE_MGMT] BE MOVED {symbol} {side} "
                          f"pos={bp.position_id} R={r_multiple:.2f} "
                          f"SL: {current_sl} -> {new_sl_dec}", file=sys.stderr)
                except Exception as e:
                    print(f"[TRADE_MGMT] BE FAILED {symbol}: {e}", file=sys.stderr)

        # ── 2. Trailing Stop ──
        if r_multiple >= cfg.trailing_trigger_r:
            trail_distance = risk * cfg.trailing_distance_r

            if is_long:
                new_sl = current_price - trail_distance
            else:
                new_sl = current_price + trail_distance

            # Safety: only tighten
            if current_sl is not None and (
                (is_long and new_sl > current_sl) or
                (not is_long and new_sl < current_sl)
            ):
                new_sl_dec = Decimal(str(round(new_sl, 5 if not is_jpy else 3)))
                try:
                    self.broker.modify_position(
                        position_id=int(bp.position_id),
                        stop_loss=new_sl_dec,
                    )
                    db_pos.current_stop_loss = new_sl_dec
                    if not trail_active:
                        db_pos.trailing_active = True
                        trail_active = True
                    print(f"[TRADE_MGMT] TRAIL {symbol} {side} "
                          f"pos={bp.position_id} R={r_multiple:.2f} "
                          f"SL: {current_sl} -> {new_sl_dec}", file=sys.stderr)
                except Exception as e:
                    print(f"[TRADE_MGMT] TRAIL FAILED {symbol}: {e}", file=sys.stderr)

        # ── 3. Time Stop ──
        open_time = db_pos.open_time
        if open_time:
            # Ensure open_time is timezone-aware for comparison
            if open_time.tzinfo is None:
                open_time = open_time.replace(tzinfo=timezone.utc)
            duration = utc_now - open_time
            max_duration = timedelta(hours=cfg.max_duration_hours)

            if duration > max_duration:
                try:
                    self.broker.close_position(
                        position_id=int(bp.position_id),
                    )
                    db_pos.close_time = datetime.now()
                    db_pos.close_reason = "TIMEOUT"
                    print(f"[TRADE_MGMT] TIMEOUT {symbol} {side} "
                          f"pos={bp.position_id} duration={duration.total_seconds()/3600:.1f}h "
                          f"limit={cfg.max_duration_hours}h", file=sys.stderr)
                except Exception as e:
                    print(f"[TRADE_MGMT] TIMEOUT CLOSE FAILED {symbol}: {e}",
                          file=sys.stderr)
