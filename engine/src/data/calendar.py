"""
Economic Calendar - News Event Filter

Fetches economic calendar data to avoid trading during high-impact news.
Sources:
- Forex Factory (scraped)
- Investing.com Economic Calendar API

High-impact news can cause:
- Wide spreads
- Slippage
- Stop hunts
- Unpredictable price action
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import requests


class NewsImpact(Enum):
    """Impact level of economic news."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class NewsEvent:
    """Economic news event."""
    timestamp: datetime
    currency: str
    event_name: str
    impact: NewsImpact
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None

    @property
    def is_high_impact(self) -> bool:
        """Check if this is a high-impact event."""
        return self.impact == NewsImpact.HIGH

    @property
    def time_until(self) -> timedelta:
        """Time until this event occurs."""
        return self.timestamp - datetime.now()

    @property
    def minutes_until(self) -> float:
        """Minutes until this event."""
        return self.time_until.total_seconds() / 60

    def affects_pair(self, pair: str) -> bool:
        """Check if this event affects a currency pair."""
        pair_upper = pair.upper().replace("/", "")
        return self.currency.upper() in pair_upper


class EconomicCalendar:
    """
    Fetches and filters economic calendar events.

    Use this to:
    - Avoid trading during high-impact news
    - Adjust position sizing around news
    - Time entries after news volatility settles
    """

    # Known high-impact events to watch
    HIGH_IMPACT_EVENTS = [
        "Non-Farm Payrolls",
        "NFP",
        "Interest Rate Decision",
        "FOMC",
        "Fed Chair",
        "ECB President",
        "GDP",
        "CPI",
        "Inflation Rate",
        "Unemployment Rate",
        "Retail Sales",
        "PMI",
        "Trade Balance",
        "BOE",
        "BOJ",
        "RBA",
        "SNB",
    ]

    def __init__(self, buffer_minutes: int = 30):
        """
        Initialize economic calendar.

        Args:
            buffer_minutes: Minutes before/after news to avoid trading
        """
        self.buffer_minutes = buffer_minutes
        self.events: List[NewsEvent] = []
        self.last_fetch: Optional[datetime] = None
        self.cache_duration = timedelta(hours=1)

    def fetch_events(self, days_ahead: int = 7) -> List[NewsEvent]:
        """
        Fetch economic calendar events.

        Args:
            days_ahead: Number of days to fetch

        Returns:
            List of NewsEvent objects
        """
        # Try multiple sources
        events = self._fetch_from_forex_factory()

        if not events:
            events = self._fetch_from_investing()

        if events:
            self.events = events
            self.last_fetch = datetime.now()

        return self.events

    def _fetch_from_forex_factory(self) -> List[NewsEvent]:
        """Fetch events from Forex Factory."""
        events = []

        try:
            # Forex Factory calendar page
            url = "https://www.forexfactory.com/calendar"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code != 200:
                return events

            content = response.text

            # Parse the calendar HTML
            # This is simplified - actual parsing would need BeautifulSoup
            # Look for event rows with impact indicators

            # Pattern for high impact events (red folder icon)
            high_impact_pattern = r'class="calendar__cell.*?impact.*?high.*?currency">(.*?)</.*?event">(.*?)<'

            matches = re.findall(high_impact_pattern, content, re.DOTALL)

            for currency, event_name in matches:
                # Create event (simplified - actual implementation would parse times)
                events.append(NewsEvent(
                    timestamp=datetime.now() + timedelta(hours=1),  # Placeholder
                    currency=currency.strip()[:3],
                    event_name=event_name.strip(),
                    impact=NewsImpact.HIGH
                ))

        except Exception as e:
            print(f"Error fetching Forex Factory calendar: {e}")

        return events

    def _fetch_from_investing(self) -> List[NewsEvent]:
        """Fetch events from Investing.com."""
        events = []

        try:
            # Investing.com economic calendar API
            url = "https://www.investing.com/economic-calendar/"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'X-Requested-With': 'XMLHttpRequest'
            }

            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code != 200:
                return events

            # Parse response (simplified)
            content = response.text

            # Look for high impact events (3 bulls)
            # Actual implementation would parse the full calendar

        except Exception as e:
            print(f"Error fetching Investing.com calendar: {e}")

        return events

    def get_upcoming_events(
        self,
        currency: Optional[str] = None,
        hours_ahead: int = 24,
        impact_filter: Optional[NewsImpact] = None
    ) -> List[NewsEvent]:
        """
        Get upcoming events with optional filters.

        Args:
            currency: Filter by currency (e.g., "USD", "EUR")
            hours_ahead: Hours to look ahead
            impact_filter: Minimum impact level

        Returns:
            Filtered list of upcoming events
        """
        # Refresh cache if needed
        if (not self.last_fetch or
                datetime.now() - self.last_fetch > self.cache_duration):
            self.fetch_events()

        now = datetime.now()
        cutoff = now + timedelta(hours=hours_ahead)

        filtered = []
        for event in self.events:
            # Time filter
            if event.timestamp < now or event.timestamp > cutoff:
                continue

            # Currency filter
            if currency and event.currency.upper() != currency.upper():
                continue

            # Impact filter
            if impact_filter:
                impact_order = [NewsImpact.LOW, NewsImpact.MEDIUM, NewsImpact.HIGH]
                if impact_order.index(event.impact) < impact_order.index(impact_filter):
                    continue

            filtered.append(event)

        return sorted(filtered, key=lambda e: e.timestamp)

    def is_safe_to_trade(self, pair: str) -> dict:
        """
        Check if it's safe to trade a pair (no imminent news).

        Args:
            pair: Currency pair (e.g., "EURUSD")

        Returns:
            Dictionary with safety status and details
        """
        # Get base and quote currencies
        pair_clean = pair.upper().replace("/", "")
        base = pair_clean[:3]
        quote = pair_clean[3:6] if len(pair_clean) >= 6 else "USD"

        # Check for upcoming high-impact news
        upcoming = self.get_upcoming_events(
            hours_ahead=self.buffer_minutes / 60 * 2,  # Double buffer
            impact_filter=NewsImpact.HIGH
        )

        affecting_events = []
        for event in upcoming:
            if event.affects_pair(pair):
                affecting_events.append(event)

        if not affecting_events:
            return {
                "safe": True,
                "message": "No high-impact news imminent",
                "events": []
            }

        # Check timing
        nearest_event = min(affecting_events, key=lambda e: e.timestamp)
        minutes_until = nearest_event.minutes_until

        if minutes_until < self.buffer_minutes:
            return {
                "safe": False,
                "message": f"High-impact news in {minutes_until:.0f} minutes: {nearest_event.event_name}",
                "events": affecting_events,
                "wait_minutes": self.buffer_minutes - minutes_until
            }

        return {
            "safe": True,
            "message": f"News in {minutes_until:.0f} minutes - trade with caution",
            "events": affecting_events,
            "caution": True
        }

    def get_news_for_pair(self, pair: str, hours_ahead: int = 24) -> List[NewsEvent]:
        """
        Get all news events affecting a currency pair.

        Args:
            pair: Currency pair
            hours_ahead: Hours to look ahead

        Returns:
            List of events affecting this pair
        """
        all_events = self.get_upcoming_events(hours_ahead=hours_ahead)
        return [e for e in all_events if e.affects_pair(pair)]

    def should_reduce_position(self, pair: str) -> dict:
        """
        Check if position size should be reduced due to upcoming news.

        Args:
            pair: Currency pair

        Returns:
            Dictionary with recommendation
        """
        events = self.get_news_for_pair(pair, hours_ahead=4)

        high_impact = [e for e in events if e.is_high_impact]

        if not high_impact:
            return {
                "reduce": False,
                "factor": 1.0,
                "reason": "No high-impact news in next 4 hours"
            }

        nearest = min(high_impact, key=lambda e: e.timestamp)
        hours_until = nearest.minutes_until / 60

        if hours_until < 1:
            return {
                "reduce": True,
                "factor": 0.25,  # Quarter position
                "reason": f"{nearest.event_name} in < 1 hour"
            }
        elif hours_until < 2:
            return {
                "reduce": True,
                "factor": 0.5,  # Half position
                "reason": f"{nearest.event_name} in < 2 hours"
            }
        else:
            return {
                "reduce": True,
                "factor": 0.75,  # 75% position
                "reason": f"{nearest.event_name} in < 4 hours"
            }

    def get_trading_windows(self, pair: str, hours_ahead: int = 24) -> List[dict]:
        """
        Get safe trading windows avoiding news events.

        Args:
            pair: Currency pair
            hours_ahead: Hours to analyze

        Returns:
            List of safe trading windows
        """
        events = self.get_news_for_pair(pair, hours_ahead)
        high_impact = [e for e in events if e.is_high_impact]

        if not high_impact:
            return [{
                "start": datetime.now(),
                "end": datetime.now() + timedelta(hours=hours_ahead),
                "duration_hours": hours_ahead,
                "quality": "excellent"
            }]

        # Sort events by time
        high_impact.sort(key=lambda e: e.timestamp)

        windows = []
        current_time = datetime.now()

        for event in high_impact:
            event_start = event.timestamp - timedelta(minutes=self.buffer_minutes)
            event_end = event.timestamp + timedelta(minutes=self.buffer_minutes)

            if current_time < event_start:
                window_hours = (event_start - current_time).total_seconds() / 3600
                windows.append({
                    "start": current_time,
                    "end": event_start,
                    "duration_hours": window_hours,
                    "quality": "good" if window_hours > 2 else "limited"
                })

            current_time = event_end

        # Add final window
        end_time = datetime.now() + timedelta(hours=hours_ahead)
        if current_time < end_time:
            window_hours = (end_time - current_time).total_seconds() / 3600
            windows.append({
                "start": current_time,
                "end": end_time,
                "duration_hours": window_hours,
                "quality": "good" if window_hours > 2 else "limited"
            })

        return windows
