"""Event Risk Engine for news and economic calendar filtering.

Blocks or reduces risk during high-impact news windows
to avoid volatility spikes around major announcements.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
import structlog

logger = structlog.get_logger(__name__)


class ImpactLevel(Enum):
    """News impact levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class EventAction(Enum):
    """Recommended action during event windows."""
    TRADE_NORMALLY = "TRADE_NORMALLY"
    REDUCE_RISK = "REDUCE_RISK"
    NO_NEW_TRADES = "NO_NEW_TRADES"
    CLOSE_POSITIONS = "CLOSE_POSITIONS"


@dataclass
class EconomicEvent:
    """An economic calendar event."""
    event_id: str
    name: str
    currency: str
    impact: ImpactLevel
    event_time: datetime
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


@dataclass
class EventRisk:
    """Event risk assessment for a symbol."""
    timestamp: datetime
    symbol: str
    currencies_affected: Set[str]

    # Current state
    trade_allowed: bool
    action: EventAction
    risk_multiplier: float  # 0.0 to 1.0

    # Upcoming events
    next_event: Optional[EconomicEvent]
    minutes_until_next: Optional[int]
    events_in_window: List[EconomicEvent]

    # Active windows
    in_high_impact_window: bool
    in_medium_impact_window: bool

    # Warnings
    warnings: List[str]


class EventRiskEngine:
    """Manages event-based trading restrictions.

    Monitors economic calendar and blocks/reduces trading
    around high-impact news events to avoid volatility spikes.
    """

    def __init__(
        self,
        high_impact_buffer_minutes: int = 30,
        medium_impact_buffer_minutes: int = 15,
        low_impact_buffer_minutes: int = 5,
        high_impact_risk_multiplier: float = 0.0,  # No trading
        medium_impact_risk_multiplier: float = 0.5,  # Half risk
    ):
        """Initialize event risk engine.

        Args:
            high_impact_buffer_minutes: Minutes before/after high impact events
            medium_impact_buffer_minutes: Minutes before/after medium impact
            low_impact_buffer_minutes: Minutes before/after low impact
            high_impact_risk_multiplier: Risk multiplier during high impact
            medium_impact_risk_multiplier: Risk multiplier during medium impact
        """
        self.high_buffer = timedelta(minutes=high_impact_buffer_minutes)
        self.medium_buffer = timedelta(minutes=medium_impact_buffer_minutes)
        self.low_buffer = timedelta(minutes=low_impact_buffer_minutes)
        self.high_multiplier = high_impact_risk_multiplier
        self.medium_multiplier = medium_impact_risk_multiplier

        # Currency to symbol mapping
        self.currency_symbols = self._build_currency_map()

        # Event cache
        self._events: List[EconomicEvent] = []

    def _build_currency_map(self) -> Dict[str, List[str]]:
        """Build mapping of currencies to affected symbols."""
        return {
            "USD": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
                   "US30", "US500", "US100"],
            "EUR": ["EURUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
                   "GER40"],
            "GBP": ["GBPUSD", "EURGBP", "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
                   "UK100"],
            "JPY": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
                   "JPN225"],
            "CHF": ["USDCHF", "EURCHF", "GBPCHF", "AUDCHF", "NZDCHF", "CADCHF", "CHFJPY"],
            "AUD": ["AUDUSD", "EURAUD", "GBPAUD", "AUDJPY", "AUDCHF", "AUDNZD", "AUDCAD"],
            "NZD": ["NZDUSD", "EURNZD", "GBPNZD", "NZDJPY", "NZDCHF", "AUDNZD", "NZDCAD"],
            "CAD": ["USDCAD", "EURCAD", "GBPCAD", "AUDCAD", "NZDCAD", "CADJPY", "CADCHF"],
        }

    def update_calendar(self, events: List[EconomicEvent]) -> None:
        """Update the economic calendar.

        Args:
            events: List of upcoming economic events
        """
        self._events = sorted(events, key=lambda e: e.event_time)
        logger.info(f"Updated economic calendar with {len(events)} events")

    def add_event(self, event: EconomicEvent) -> None:
        """Add a single event to the calendar."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.event_time)

    def assess_risk(
        self,
        symbol: str,
        current_time: Optional[datetime] = None
    ) -> EventRisk:
        """Assess event risk for a symbol at current time.

        Args:
            symbol: Trading symbol
            current_time: Time to check (defaults to now)

        Returns:
            EventRisk assessment
        """
        if current_time is None:
            current_time = datetime.now()

        # Get currencies affected by this symbol
        affected_currencies = self._get_affected_currencies(symbol)

        # Find relevant events
        relevant_events = self._get_relevant_events(
            affected_currencies, current_time
        )

        # Check if we're in any event windows
        in_high_window = False
        in_medium_window = False
        events_in_window = []
        warnings = []

        for event in relevant_events:
            time_to_event = event.event_time - current_time
            time_since_event = current_time - event.event_time

            # Check high impact window
            if event.impact == ImpactLevel.HIGH:
                if -self.high_buffer <= time_to_event <= self.high_buffer:
                    in_high_window = True
                    events_in_window.append(event)

            # Check medium impact window
            elif event.impact == ImpactLevel.MEDIUM:
                if -self.medium_buffer <= time_to_event <= self.medium_buffer:
                    in_medium_window = True
                    events_in_window.append(event)

        # Determine action and risk multiplier
        if in_high_window:
            action = EventAction.NO_NEW_TRADES
            risk_multiplier = self.high_multiplier
            trade_allowed = False
            warnings.append(f"In high-impact event window")
        elif in_medium_window:
            action = EventAction.REDUCE_RISK
            risk_multiplier = self.medium_multiplier
            trade_allowed = True
            warnings.append(f"In medium-impact event window - risk reduced")
        else:
            action = EventAction.TRADE_NORMALLY
            risk_multiplier = 1.0
            trade_allowed = True

        # Find next event
        next_event = None
        minutes_until = None
        for event in relevant_events:
            if event.event_time > current_time:
                next_event = event
                minutes_until = int((event.event_time - current_time).total_seconds() / 60)
                break

        # Add warning for upcoming events
        if next_event and minutes_until and minutes_until <= 60:
            warnings.append(
                f"Upcoming {next_event.impact.value} impact event: "
                f"{next_event.name} in {minutes_until} minutes"
            )

        return EventRisk(
            timestamp=current_time,
            symbol=symbol,
            currencies_affected=affected_currencies,
            trade_allowed=trade_allowed,
            action=action,
            risk_multiplier=risk_multiplier,
            next_event=next_event,
            minutes_until_next=minutes_until,
            events_in_window=events_in_window,
            in_high_impact_window=in_high_window,
            in_medium_impact_window=in_medium_window,
            warnings=warnings,
        )

    def _get_affected_currencies(self, symbol: str) -> Set[str]:
        """Get currencies that affect a symbol."""
        affected = set()
        symbol_upper = symbol.upper()

        # Check direct currency pairs
        if len(symbol_upper) == 6:
            affected.add(symbol_upper[:3])
            affected.add(symbol_upper[3:])

        # Check currency mappings
        for currency, symbols in self.currency_symbols.items():
            if symbol_upper in [s.upper() for s in symbols]:
                affected.add(currency)

        return affected

    def _get_relevant_events(
        self,
        currencies: Set[str],
        current_time: datetime
    ) -> List[EconomicEvent]:
        """Get events relevant to specified currencies."""
        # Look at events within the next 24 hours and past 1 hour
        window_start = current_time - timedelta(hours=1)
        window_end = current_time + timedelta(hours=24)

        relevant = []
        for event in self._events:
            if window_start <= event.event_time <= window_end:
                if event.currency in currencies:
                    relevant.append(event)

        return relevant

    def get_daily_schedule(
        self,
        symbol: str,
        date: Optional[datetime] = None
    ) -> List[Dict]:
        """Get schedule of events affecting a symbol for a day.

        Args:
            symbol: Trading symbol
            date: Date to check (defaults to today)

        Returns:
            List of event dictionaries
        """
        if date is None:
            date = datetime.now()

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        affected_currencies = self._get_affected_currencies(symbol)

        schedule = []
        for event in self._events:
            if start_of_day <= event.event_time < end_of_day:
                if event.currency in affected_currencies:
                    schedule.append({
                        "time": event.event_time.isoformat(),
                        "name": event.name,
                        "currency": event.currency,
                        "impact": event.impact.value,
                        "forecast": event.forecast,
                        "previous": event.previous,
                    })

        return schedule

    def to_dict(self, risk: EventRisk) -> Dict:
        """Convert risk assessment to dictionary."""
        return {
            "timestamp": risk.timestamp.isoformat(),
            "symbol": risk.symbol,
            "currencies_affected": list(risk.currencies_affected),
            "trade_allowed": risk.trade_allowed,
            "action": risk.action.value,
            "risk_multiplier": risk.risk_multiplier,
            "next_event": {
                "name": risk.next_event.name,
                "currency": risk.next_event.currency,
                "impact": risk.next_event.impact.value,
                "time": risk.next_event.event_time.isoformat(),
            } if risk.next_event else None,
            "minutes_until_next": risk.minutes_until_next,
            "in_high_impact_window": risk.in_high_impact_window,
            "in_medium_impact_window": risk.in_medium_impact_window,
            "warnings": risk.warnings,
        }


# Common high-impact events
HIGH_IMPACT_EVENTS = [
    "Non-Farm Payrolls",
    "NFP",
    "FOMC",
    "Fed Interest Rate Decision",
    "ECB Interest Rate Decision",
    "BOE Interest Rate Decision",
    "BOJ Interest Rate Decision",
    "CPI",
    "GDP",
    "Retail Sales",
    "PMI",
]

MEDIUM_IMPACT_EVENTS = [
    "Unemployment Rate",
    "Employment Change",
    "Trade Balance",
    "Industrial Production",
    "Consumer Confidence",
    "Housing Starts",
    "Durable Goods Orders",
]


def classify_event_impact(event_name: str) -> ImpactLevel:
    """Classify event impact level based on name."""
    name_upper = event_name.upper()

    for high in HIGH_IMPACT_EVENTS:
        if high.upper() in name_upper:
            return ImpactLevel.HIGH

    for medium in MEDIUM_IMPACT_EVENTS:
        if medium.upper() in name_upper:
            return ImpactLevel.MEDIUM

    return ImpactLevel.LOW
