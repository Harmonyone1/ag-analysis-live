"""Trade Manager - Full lifecycle management for paper and live trades.

Handles:
- Trade entry and fill tracking
- Position monitoring with P&L updates
- Stop loss management (trailing, break-even)
- Take profit management (partial exits)
- Trade exit and performance recording
- Session-aware timeouts (weekend, news, market hours)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import uuid
import structlog

from src.adapters.broker import BrokerAdapter, Quote
from src.scoring.confluence import TradeSetup

logger = structlog.get_logger(__name__)


# Forex market sessions (UTC times)
class MarketSession(Enum):
    """Forex market sessions."""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    OFF_HOURS = "off_hours"


@dataclass
class SessionConfig:
    """Configuration for session-aware trading."""
    # Session hours (UTC)
    sydney_open: int = 21  # 21:00 UTC (Sunday)
    sydney_close: int = 6
    tokyo_open: int = 0
    tokyo_close: int = 9
    london_open: int = 7
    london_close: int = 16
    new_york_open: int = 12
    new_york_close: int = 21

    # Weekend handling
    friday_close_hour: int = 21  # 21:00 UTC Friday
    sunday_open_hour: int = 21  # 21:00 UTC Sunday
    close_before_weekend_hours: float = 2.0  # Close positions 2h before market close
    warn_before_weekend_hours: float = 4.0  # Warn 4h before close

    # High-impact news windows (hours before/after)
    news_blackout_before: float = 0.5  # 30 min before news
    news_blackout_after: float = 0.25  # 15 min after news

    # Session-specific timeouts (hours)
    max_duration_by_session: Dict[str, int] = field(default_factory=lambda: {
        "sydney": 8,
        "tokyo": 8,
        "london": 10,
        "new_york": 10,
        "overlap_london_ny": 6,  # Shorter during high volatility
        "off_hours": 4,
    })

    # Symbols best traded in specific sessions
    session_preferred_pairs: Dict[str, Set[str]] = field(default_factory=lambda: {
        "sydney": {"AUDUSD", "NZDUSD", "AUDJPY", "AUDNZD", "NZDJPY", "AUDCAD", "AUDCHF"},
        "tokyo": {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CHFJPY", "CADJPY"},
        "london": {"EURUSD", "GBPUSD", "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "GBPCHF"},
        "new_york": {"EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY"},
    })


class TradeStatus(Enum):
    """Trade status."""
    PENDING = "pending"  # Waiting for entry fill
    OPEN = "open"  # Position open
    PARTIAL_CLOSE = "partial_close"  # Some TPs hit
    CLOSED = "closed"  # Fully closed


class ExitReason(Enum):
    """Reason for trade exit."""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    BREAK_EVEN = "break_even"
    MANUAL = "manual"
    TIMEOUT = "timeout"
    INVALIDATION = "invalidation"
    WEEKEND_CLOSE = "weekend_close"  # Closed before weekend gap
    SESSION_TIMEOUT = "session_timeout"  # Session-specific timeout
    NEWS_EVENT = "news_event"  # Closed due to high-impact news


@dataclass
class PaperPosition:
    """A paper trading position."""
    id: str
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: Decimal
    current_price: Decimal
    quantity: Decimal
    stop_loss: Decimal
    take_profit: Optional[Decimal]
    tp_targets: List[Dict]  # [{r: 1.5, price: 1.09, hit: False}, ...]

    # Tracking
    opened_at: datetime
    candidate_id: Optional[str]
    ai_probability: float
    ai_expected_r: float
    confluence_score: int

    # P&L
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    # Excursions
    max_favorable_price: Optional[Decimal] = None
    max_adverse_price: Optional[Decimal] = None

    # Management
    original_stop: Optional[Decimal] = None
    break_even_moved: bool = False
    trailing_active: bool = False

    # Status
    status: TradeStatus = TradeStatus.OPEN
    closed_at: Optional[datetime] = None
    exit_reason: Optional[ExitReason] = None
    exit_price: Optional[Decimal] = None

    # Session tracking
    opened_session: Optional[str] = None  # Session when trade was opened
    weekend_warning_sent: bool = False  # Whether weekend warning was sent


@dataclass
class TradeResult:
    """Result of a completed trade."""
    trade_id: str
    symbol: str
    direction: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl_pips: float
    pnl_r: float  # R-multiple
    duration_minutes: int
    exit_reason: ExitReason
    ai_probability: float
    ai_expected_r: float
    confluence_score: int
    max_favorable_excursion: float  # In pips
    max_adverse_excursion: float  # In pips
    timestamp: datetime


@dataclass
class TradeManagerConfig:
    """Configuration for trade management."""
    # Break-even settings
    move_to_breakeven_r: float = 1.0  # Move SL to BE after 1R profit
    breakeven_buffer_pips: float = 2.0  # Add buffer above/below entry

    # Trailing stop settings
    trailing_start_r: float = 1.5  # Start trailing after 1.5R
    trailing_distance_r: float = 0.5  # Trail 0.5R behind price

    # Partial exit settings
    partial_exit_enabled: bool = True
    partial_exit_percent: float = 0.5  # Close 50% at first TP

    # Timeout settings
    max_trade_duration_hours: int = 48  # Max trade duration
    session_aware_timeouts: bool = True  # Enable session-aware timeouts

    # Risk settings
    max_open_trades: int = 5

    # Session configuration
    session_config: SessionConfig = field(default_factory=SessionConfig)

    # High-impact news events (UTC hours on weekdays)
    # Format: {weekday: [hour1, hour2, ...]} where Monday=0
    high_impact_news_schedule: Dict[int, List[int]] = field(default_factory=lambda: {
        # NFP first Friday of month at 13:30 UTC (approximated to 13:00)
        4: [13],  # Friday
        # FOMC typically Wednesday at 19:00 UTC
        2: [19],  # Wednesday
        # ECB typically Thursday at 12:45 UTC
        3: [12],  # Thursday
        # BOE typically Thursday at 12:00 UTC
        3: [12],
        # Common news times across all days
        0: [13, 15],  # Monday
        1: [13, 15],  # Tuesday
        2: [13, 15, 19],  # Wednesday
        3: [12, 13, 15],  # Thursday
        4: [13, 15],  # Friday
    })


class TradeManager:
    """Manages the full lifecycle of paper and live trades.

    Example:
        manager = TradeManager(config, broker=broker)

        # Open a trade
        position = manager.open_trade(setup, quantity, ai_prob, ai_er)

        # Update positions with current prices
        manager.update_positions()

        # Get performance stats
        stats = manager.get_performance_stats()
    """

    def __init__(
        self,
        config: Optional[TradeManagerConfig] = None,
        broker: Optional[BrokerAdapter] = None,
        paper_mode: bool = False,
    ):
        """Initialize trade manager.

        Args:
            config: Trade management configuration
            broker: Broker adapter for quotes (optional in paper mode)
            paper_mode: If True, use simulated prices
        """
        self.config = config or TradeManagerConfig()
        self.broker = broker
        self.paper_mode = paper_mode

        # Paper positions (keyed by position ID)
        self._positions: Dict[str, PaperPosition] = {}

        # Trade history
        self._trade_results: List[TradeResult] = []

        # Stats
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl_r = 0.0

        # Price cache for paper mode
        self._price_cache: Dict[str, Decimal] = {}

    def update_price(self, symbol: str, price: float) -> None:
        """Update cached price for a symbol (used in paper mode)."""
        self._price_cache[symbol] = Decimal(str(price))

    def _get_current_price(self, symbol: str, for_buy: bool = True) -> Optional[Decimal]:
        """Get current price from broker or cache."""
        if self.broker:
            try:
                quote = self.broker.get_quote(symbol)
                return quote.ask if for_buy else quote.bid
            except Exception:
                pass

        # Fall back to cache
        return self._price_cache.get(symbol)

    # =========================================================================
    # Session-Aware Methods
    # =========================================================================

    def get_current_session(self, utc_time: Optional[datetime] = None) -> MarketSession:
        """Determine the current forex market session.

        Args:
            utc_time: UTC time to check (defaults to now)

        Returns:
            Current MarketSession
        """
        if utc_time is None:
            utc_time = datetime.now(timezone.utc)

        hour = utc_time.hour
        sc = self.config.session_config

        # Check for London-NY overlap first (highest priority)
        if sc.new_york_open <= hour < sc.london_close:
            return MarketSession.OVERLAP_LONDON_NY

        # London session
        if sc.london_open <= hour < sc.london_close:
            return MarketSession.LONDON

        # New York session
        if sc.new_york_open <= hour < sc.new_york_close:
            return MarketSession.NEW_YORK

        # Tokyo session
        if sc.tokyo_open <= hour < sc.tokyo_close:
            return MarketSession.TOKYO

        # Sydney session (wraps around midnight)
        if hour >= sc.sydney_open or hour < sc.sydney_close:
            return MarketSession.SYDNEY

        return MarketSession.OFF_HOURS

    def is_near_weekend_close(self, utc_time: Optional[datetime] = None) -> Tuple[bool, float]:
        """Check if we're approaching weekend market close.

        Args:
            utc_time: UTC time to check (defaults to now)

        Returns:
            Tuple of (is_near_close, hours_until_close)
        """
        if utc_time is None:
            utc_time = datetime.now(timezone.utc)

        # Market closes Friday at ~21:00 UTC
        sc = self.config.session_config
        weekday = utc_time.weekday()
        hour = utc_time.hour

        # Only relevant on Friday
        if weekday != 4:  # Friday
            return False, float('inf')

        hours_until_close = sc.friday_close_hour - hour
        if hours_until_close < 0:
            hours_until_close = 0

        is_near = hours_until_close <= sc.warn_before_weekend_hours

        return is_near, hours_until_close

    def should_close_for_weekend(self, utc_time: Optional[datetime] = None) -> bool:
        """Determine if positions should be closed before weekend.

        Args:
            utc_time: UTC time to check (defaults to now)

        Returns:
            True if positions should be closed
        """
        if utc_time is None:
            utc_time = datetime.now(timezone.utc)

        is_near, hours_until = self.is_near_weekend_close(utc_time)
        return is_near and hours_until <= self.config.session_config.close_before_weekend_hours

    def is_market_open(self, utc_time: Optional[datetime] = None) -> bool:
        """Check if forex market is currently open.

        Args:
            utc_time: UTC time to check (defaults to now)

        Returns:
            True if market is open
        """
        if utc_time is None:
            utc_time = datetime.now(timezone.utc)

        weekday = utc_time.weekday()
        hour = utc_time.hour
        sc = self.config.session_config

        # Market closed Saturday
        if weekday == 5:
            return False

        # Market closed Sunday until ~21:00 UTC
        if weekday == 6 and hour < sc.sunday_open_hour:
            return False

        # Market closed Friday after ~21:00 UTC
        if weekday == 4 and hour >= sc.friday_close_hour:
            return False

        return True

    def is_near_high_impact_news(
        self,
        symbol: str,
        utc_time: Optional[datetime] = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if we're in a high-impact news window.

        Args:
            symbol: The trading symbol
            utc_time: UTC time to check (defaults to now)

        Returns:
            Tuple of (is_near_news, news_description)
        """
        if utc_time is None:
            utc_time = datetime.now(timezone.utc)

        weekday = utc_time.weekday()
        hour = utc_time.hour
        minute = utc_time.minute
        current_decimal_hour = hour + minute / 60.0

        sc = self.config.session_config
        news_hours = self.config.high_impact_news_schedule.get(weekday, [])

        for news_hour in news_hours:
            hours_until = news_hour - current_decimal_hour

            # Check if within blackout window
            if -sc.news_blackout_after <= hours_until <= sc.news_blackout_before:
                # Determine news type based on hour and symbol
                news_type = self._identify_news_event(weekday, news_hour, symbol)
                return True, news_type

        return False, None

    def _identify_news_event(self, weekday: int, hour: int, symbol: str) -> str:
        """Identify the type of news event.

        Args:
            weekday: Day of week (0=Monday)
            hour: UTC hour of news
            symbol: Trading symbol

        Returns:
            Description of the news event
        """
        # Common high-impact news events
        if weekday == 4 and hour == 13:
            if symbol in ("EURUSD", "GBPUSD", "USDCHF", "USDJPY", "USDCAD"):
                return "NFP (Non-Farm Payrolls)"
            return "US Employment Data"

        if weekday == 2 and hour == 19:
            if "USD" in symbol:
                return "FOMC Rate Decision"
            return "Fed Announcement"

        if weekday == 3 and hour == 12:
            if "EUR" in symbol:
                return "ECB Rate Decision"
            if "GBP" in symbol:
                return "BOE Rate Decision"
            return "Central Bank Announcement"

        if hour in (13, 15):
            return "Scheduled Economic Release"

        return "High-Impact News Window"

    def get_session_timeout(self, position: PaperPosition) -> timedelta:
        """Get the timeout for a position based on its session.

        Args:
            position: The position to check

        Returns:
            Maximum duration as timedelta
        """
        if not self.config.session_aware_timeouts:
            return timedelta(hours=self.config.max_trade_duration_hours)

        session = position.opened_session or "off_hours"
        session_hours = self.config.session_config.max_duration_by_session.get(
            session,
            self.config.max_trade_duration_hours
        )

        return timedelta(hours=session_hours)

    def is_optimal_session_for_pair(self, symbol: str) -> Tuple[bool, str]:
        """Check if current session is optimal for trading a pair.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (is_optimal, current_session_name)
        """
        current_session = self.get_current_session()
        session_name = current_session.value

        preferred = self.config.session_config.session_preferred_pairs.get(session_name, set())
        is_optimal = symbol in preferred or not preferred

        return is_optimal, session_name

    def _check_session_timeout(self, position: PaperPosition) -> Tuple[bool, Optional[ExitReason]]:
        """Check if trade has exceeded session-aware timeout.

        Args:
            position: Position to check

        Returns:
            Tuple of (should_close, exit_reason)
        """
        now = datetime.now(timezone.utc) if self.config.session_aware_timeouts else datetime.now()
        duration = now - position.opened_at.replace(tzinfo=timezone.utc if self.config.session_aware_timeouts else None)

        # Check standard timeout
        max_duration = self.get_session_timeout(position)
        if duration > max_duration:
            return True, ExitReason.SESSION_TIMEOUT

        # Check weekend close
        if self.should_close_for_weekend():
            return True, ExitReason.WEEKEND_CLOSE

        # Check high-impact news
        is_near_news, news_type = self.is_near_high_impact_news(position.symbol)
        if is_near_news:
            # Only close if position is in profit and news affects the pair
            pnl_r = self._get_pnl_in_r(position, position.current_price)
            if pnl_r > 0.5:  # Only close profitable trades before news
                logger.warning(
                    "Closing profitable trade before news",
                    id=position.id,
                    symbol=position.symbol,
                    news=news_type,
                    pnl_r=f"{pnl_r:.2f}R",
                )
                return True, ExitReason.NEWS_EVENT

        return False, None

    def check_weekend_warning(self, position: PaperPosition) -> Optional[str]:
        """Check if weekend warning should be sent for a position.

        Args:
            position: Position to check

        Returns:
            Warning message if applicable, None otherwise
        """
        if position.weekend_warning_sent:
            return None

        is_near, hours_until = self.is_near_weekend_close()
        if is_near and hours_until <= self.config.session_config.warn_before_weekend_hours:
            position.weekend_warning_sent = True
            return (
                f"Position {position.id} ({position.symbol}): "
                f"Weekend close in {hours_until:.1f}h. "
                f"Consider closing or moving SL to BE."
            )

        return None

    # =========================================================================
    # Trade Operations
    # =========================================================================

    def open_trade(
        self,
        setup: TradeSetup,
        quantity: Decimal,
        ai_probability: float,
        ai_expected_r: float,
        candidate_id: Optional[str] = None,
    ) -> Optional[PaperPosition]:
        """Open a new paper trade.

        Args:
            setup: Trade setup
            quantity: Position size in lots
            ai_probability: AI P(win)
            ai_expected_r: AI E[R]
            candidate_id: Reference to trade candidate

        Returns:
            PaperPosition if opened, None if rejected
        """
        # Check max trades
        open_count = len([p for p in self._positions.values() if p.status == TradeStatus.OPEN])
        if open_count >= self.config.max_open_trades:
            logger.warning("Max open trades reached", max=self.config.max_open_trades)
            return None

        # Get current price for entry
        entry_price = self._get_current_price(setup.symbol, for_buy=(setup.direction == "LONG"))
        if entry_price is None:
            logger.error("Failed to get quote for entry", symbol=setup.symbol)
            return None

        # Create position
        position_id = str(uuid.uuid4())[:8]

        # Prepare TP targets
        tp_targets = []
        for tp in (setup.tp_targets or []):
            tp_targets.append({
                "r": tp.get("r", 2.0),
                "price": Decimal(str(tp.get("price", 0))),
                "hit": False,
            })

        # Get current session for session-aware timeouts
        current_session = self.get_current_session()

        position = PaperPosition(
            id=position_id,
            symbol=setup.symbol,
            direction=setup.direction,
            entry_price=entry_price,
            current_price=entry_price,
            quantity=quantity,
            stop_loss=Decimal(str(setup.stop_price)),
            take_profit=Decimal(str(tp_targets[0]["price"])) if tp_targets else None,
            tp_targets=tp_targets,
            opened_at=datetime.now(),
            candidate_id=candidate_id,
            ai_probability=ai_probability,
            ai_expected_r=ai_expected_r,
            confluence_score=setup.confluence_score,
            original_stop=Decimal(str(setup.stop_price)),
            max_favorable_price=entry_price,
            max_adverse_price=entry_price,
            opened_session=current_session.value,
        )

        self._positions[position_id] = position
        self._total_trades += 1

        logger.info(
            "Paper trade opened",
            id=position_id,
            symbol=setup.symbol,
            direction=setup.direction,
            entry=float(entry_price),
            stop=float(position.stop_loss),
            tp=float(position.take_profit) if position.take_profit else None,
            qty=float(quantity),
            session=current_session.value,
        )

        return position

    def update_positions(self) -> List[PaperPosition]:
        """Update all open positions with current prices.

        Returns:
            List of positions that were closed
        """
        closed_positions = []

        for position_id, position in list(self._positions.items()):
            if position.status != TradeStatus.OPEN:
                continue

            try:
                # Get current quote
                current_price = self._get_current_price(position.symbol, for_buy=False) if position.direction == "LONG" else self._get_current_price(position.symbol, for_buy=True)
                if current_price is None:
                    logger.warning("No price for position update", symbol=position.symbol)
                    continue
                position.current_price = current_price

                # Update excursions
                if position.direction == "LONG":
                    if current_price > position.max_favorable_price:
                        position.max_favorable_price = current_price
                    if current_price < position.max_adverse_price:
                        position.max_adverse_price = current_price
                else:
                    if current_price < position.max_favorable_price:
                        position.max_favorable_price = current_price
                    if current_price > position.max_adverse_price:
                        position.max_adverse_price = current_price

                # Calculate unrealized P&L
                position.unrealized_pnl = self._calculate_pnl(position, current_price)

                # Check stop loss
                if self._check_stop_loss(position, current_price):
                    self._close_position(position, current_price, ExitReason.STOP_LOSS)
                    closed_positions.append(position)
                    continue

                # Check take profit
                if self._check_take_profit(position, current_price):
                    # May be partial close, check if fully closed
                    if position.status == TradeStatus.CLOSED:
                        closed_positions.append(position)
                        continue

                # Check timeout (session-aware)
                if self.config.session_aware_timeouts:
                    should_close, exit_reason = self._check_session_timeout(position)
                    if should_close and exit_reason:
                        self._close_position(position, current_price, exit_reason)
                        closed_positions.append(position)
                        continue
                elif self._check_timeout(position):
                    self._close_position(position, current_price, ExitReason.TIMEOUT)
                    closed_positions.append(position)
                    continue

                # Check weekend warnings
                if self.config.session_aware_timeouts:
                    warning = self.check_weekend_warning(position)
                    if warning:
                        logger.warning(warning)

                # Apply trade management (trailing, break-even)
                self._apply_trade_management(position, current_price)

            except Exception as e:
                logger.error("Failed to update position", id=position_id, error=str(e))

        return closed_positions

    def _calculate_pnl(self, position: PaperPosition, current_price: Decimal) -> Decimal:
        """Calculate P&L in account currency (assuming 10 USD per pip per lot)."""
        pip_size = Decimal("0.0001")
        if "JPY" in position.symbol:
            pip_size = Decimal("0.01")

        if position.direction == "LONG":
            pnl_pips = (current_price - position.entry_price) / pip_size
        else:
            pnl_pips = (position.entry_price - current_price) / pip_size

        # USD per pip per lot (approximate)
        pip_value = Decimal("10")
        return pnl_pips * pip_value * position.quantity

    def _get_pnl_in_r(self, position: PaperPosition, price: Decimal) -> float:
        """Calculate P&L in R-multiples."""
        pip_size = Decimal("0.0001")
        if "JPY" in position.symbol:
            pip_size = Decimal("0.01")

        # Risk in pips
        if position.direction == "LONG":
            risk_pips = float((position.entry_price - position.original_stop) / pip_size)
            current_pips = float((price - position.entry_price) / pip_size)
        else:
            risk_pips = float((position.original_stop - position.entry_price) / pip_size)
            current_pips = float((position.entry_price - price) / pip_size)

        if risk_pips <= 0:
            return 0.0

        return current_pips / risk_pips

    def _check_stop_loss(self, position: PaperPosition, current_price: Decimal) -> bool:
        """Check if stop loss was hit."""
        if position.direction == "LONG":
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss

    def _check_take_profit(self, position: PaperPosition, current_price: Decimal) -> bool:
        """Check if take profit was hit."""
        for tp in position.tp_targets:
            if tp["hit"]:
                continue

            tp_price = tp["price"]
            hit = False

            if position.direction == "LONG":
                hit = current_price >= tp_price
            else:
                hit = current_price <= tp_price

            if hit:
                tp["hit"] = True
                logger.info(
                    "Take profit hit",
                    id=position.id,
                    symbol=position.symbol,
                    tp_r=tp["r"],
                    price=float(current_price),
                )

                # Partial close if configured
                if self.config.partial_exit_enabled and not position.status == TradeStatus.PARTIAL_CLOSE:
                    # Close partial position
                    close_qty = position.quantity * Decimal(str(self.config.partial_exit_percent))
                    position.quantity -= close_qty
                    position.status = TradeStatus.PARTIAL_CLOSE

                    # Record partial P&L
                    pnl_r = self._get_pnl_in_r(position, current_price)
                    position.realized_pnl += self._calculate_pnl(position, current_price) * Decimal(str(self.config.partial_exit_percent))

                    logger.info(
                        "Partial close executed",
                        id=position.id,
                        symbol=position.symbol,
                        closed_qty=float(close_qty),
                        remaining_qty=float(position.quantity),
                        pnl_r=pnl_r,
                    )
                else:
                    # Full close on final TP
                    if all(t["hit"] for t in position.tp_targets) or tp == position.tp_targets[-1]:
                        self._close_position(position, current_price, ExitReason.TAKE_PROFIT)
                        return True

        return False

    def _check_timeout(self, position: PaperPosition) -> bool:
        """Check if trade has exceeded max duration."""
        duration = datetime.now() - position.opened_at
        max_duration = timedelta(hours=self.config.max_trade_duration_hours)
        return duration > max_duration

    def _apply_trade_management(self, position: PaperPosition, current_price: Decimal) -> None:
        """Apply trailing stop and break-even logic."""
        pnl_r = self._get_pnl_in_r(position, current_price)

        pip_size = Decimal("0.0001")
        if "JPY" in position.symbol:
            pip_size = Decimal("0.01")

        # Move to break-even
        if not position.break_even_moved and pnl_r >= self.config.move_to_breakeven_r:
            buffer = Decimal(str(self.config.breakeven_buffer_pips)) * pip_size

            if position.direction == "LONG":
                new_stop = position.entry_price + buffer
            else:
                new_stop = position.entry_price - buffer

            position.stop_loss = new_stop
            position.break_even_moved = True

            logger.info(
                "Moved to break-even",
                id=position.id,
                symbol=position.symbol,
                new_stop=float(new_stop),
                pnl_r=pnl_r,
            )

        # Trailing stop
        if pnl_r >= self.config.trailing_start_r:
            position.trailing_active = True

            # Calculate risk in price terms
            if position.direction == "LONG":
                risk = position.entry_price - position.original_stop
                trail_distance = risk * Decimal(str(self.config.trailing_distance_r))
                new_stop = current_price - trail_distance

                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    logger.debug(
                        "Trailing stop updated",
                        id=position.id,
                        new_stop=float(new_stop),
                    )
            else:
                risk = position.original_stop - position.entry_price
                trail_distance = risk * Decimal(str(self.config.trailing_distance_r))
                new_stop = current_price + trail_distance

                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop

    def _close_position(
        self,
        position: PaperPosition,
        exit_price: Decimal,
        exit_reason: ExitReason,
    ) -> None:
        """Close a position and record the result."""
        position.status = TradeStatus.CLOSED
        position.closed_at = datetime.now()
        position.exit_reason = exit_reason
        position.exit_price = exit_price

        # Calculate final P&L
        final_pnl = self._calculate_pnl(position, exit_price)
        position.realized_pnl += final_pnl

        # Calculate R-multiple
        pnl_r = self._get_pnl_in_r(position, exit_price)

        # Calculate excursions in pips
        pip_size = Decimal("0.0001")
        if "JPY" in position.symbol:
            pip_size = Decimal("0.01")

        if position.direction == "LONG":
            mfe_pips = float((position.max_favorable_price - position.entry_price) / pip_size)
            mae_pips = float((position.entry_price - position.max_adverse_price) / pip_size)
            pnl_pips = float((exit_price - position.entry_price) / pip_size)
        else:
            mfe_pips = float((position.entry_price - position.max_favorable_price) / pip_size)
            mae_pips = float((position.max_adverse_price - position.entry_price) / pip_size)
            pnl_pips = float((position.entry_price - exit_price) / pip_size)

        # Calculate duration
        duration_minutes = int((position.closed_at - position.opened_at).total_seconds() / 60)

        # Record result
        result = TradeResult(
            trade_id=position.id,
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl_pips=pnl_pips,
            pnl_r=pnl_r,
            duration_minutes=duration_minutes,
            exit_reason=exit_reason,
            ai_probability=position.ai_probability,
            ai_expected_r=position.ai_expected_r,
            confluence_score=position.confluence_score,
            max_favorable_excursion=mfe_pips,
            max_adverse_excursion=mae_pips,
            timestamp=datetime.now(),
        )

        self._trade_results.append(result)

        # Update stats
        if pnl_r > 0:
            self._winning_trades += 1
        self._total_pnl_r += pnl_r

        logger.info(
            "Trade closed",
            id=position.id,
            symbol=position.symbol,
            direction=position.direction,
            exit_reason=exit_reason.value,
            pnl_pips=f"{pnl_pips:.1f}",
            pnl_r=f"{pnl_r:.2f}R",
            duration=f"{duration_minutes}min",
            mfe=f"{mfe_pips:.1f}",
            mae=f"{mae_pips:.1f}",
        )

    def close_all_positions(self, reason: ExitReason = ExitReason.MANUAL) -> int:
        """Close all open positions.

        Returns:
            Number of positions closed
        """
        closed = 0
        for position in list(self._positions.values()):
            if position.status == TradeStatus.OPEN:
                try:
                    current_price = self._get_current_price(position.symbol, for_buy=False) if position.direction == "LONG" else self._get_current_price(position.symbol, for_buy=True)
                    if current_price is None:
                        # Use last known price
                        current_price = position.current_price
                    self._close_position(position, current_price, reason)
                    closed += 1
                except Exception as e:
                    logger.error("Failed to close position", id=position.id, error=str(e))
        return closed

    def get_open_positions(self) -> List[PaperPosition]:
        """Get all open positions."""
        return [p for p in self._positions.values() if p.status in (TradeStatus.OPEN, TradeStatus.PARTIAL_CLOSE)]

    def _calculate_current_r(self, position: PaperPosition, current_price: Decimal) -> float:
        """Calculate current R-multiple for a position."""
        return self._get_pnl_in_r(position, current_price)

    def get_position_status(self, position: PaperPosition) -> Dict:
        """Get detailed status for a position."""
        current_r = self._get_pnl_in_r(position, position.current_price)
        current_pnl = self._calculate_pnl(position, position.current_price)

        return {
            "id": position.id,
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_price": float(position.entry_price),
            "current_price": float(position.current_price),
            "stop_loss": float(position.stop_loss),
            "quantity": float(position.quantity),
            "current_r": current_r,
            "pnl_dollars": float(current_pnl),
            "break_even_moved": position.break_even_moved,
            "trailing_active": position.trailing_active,
            "duration_minutes": int((datetime.now() - position.opened_at).total_seconds() / 60),
        }

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self._trade_results:
            return {
                "total_trades": 0,
                "winners": 0,
                "losers": 0,
                "win_rate": None,
                "total_r": 0.0,
                "avg_r_per_trade": None,
                "profit_factor": None,
                "avg_winner_r": None,
                "avg_loser_r": None,
                "avg_duration_min": None,
                "ai_accuracy": None,
            }

        winners = [r for r in self._trade_results if r.pnl_r > 0]
        losers = [r for r in self._trade_results if r.pnl_r <= 0]

        gross_wins = sum(r.pnl_r for r in winners) if winners else 0
        gross_losses = abs(sum(r.pnl_r for r in losers)) if losers else 0

        # AI accuracy: did high probability trades win?
        high_prob_trades = [r for r in self._trade_results if r.ai_probability >= 0.55]
        high_prob_wins = [r for r in high_prob_trades if r.pnl_r > 0]

        return {
            "total_trades": len(self._trade_results),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(self._trade_results) if self._trade_results else 0,
            "total_r": sum(r.pnl_r for r in self._trade_results),
            "avg_r_per_trade": sum(r.pnl_r for r in self._trade_results) / len(self._trade_results),
            "profit_factor": gross_wins / gross_losses if gross_losses > 0 else float('inf'),
            "avg_winner_r": sum(r.pnl_r for r in winners) / len(winners) if winners else 0,
            "avg_loser_r": sum(r.pnl_r for r in losers) / len(losers) if losers else 0,
            "avg_duration_min": sum(r.duration_minutes for r in self._trade_results) / len(self._trade_results),
            "avg_mfe": sum(r.max_favorable_excursion for r in self._trade_results) / len(self._trade_results),
            "avg_mae": sum(r.max_adverse_excursion for r in self._trade_results) / len(self._trade_results),
            "ai_high_prob_trades": len(high_prob_trades),
            "ai_high_prob_wins": len(high_prob_wins),
            "ai_accuracy": len(high_prob_wins) / len(high_prob_trades) if high_prob_trades else 0,
            "exit_reasons": {
                reason.value: len([r for r in self._trade_results if r.exit_reason == reason])
                for reason in ExitReason
            },
        }

    def get_trade_history(self) -> List[TradeResult]:
        """Get trade history."""
        return self._trade_results.copy()
