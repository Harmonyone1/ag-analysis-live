"""Session-aware trading policy layer for FX bot.

Adjusts approval thresholds and position sizing based on:
- Time-of-day (UTC session)
- Day-of-week
- Calibrated P(win)

This is NOT a model change -- it's a policy layer that sits on top of
the AI gate's calibrated probabilities, allocating capital to where the
existing edge is strongest.

FX Diagnostic evidence (fx_model_diagnostics.py on 2026-02 data):
    - Asia (00-08):   WR=59.3%, EV=+0.779  -- strongest session
    - Peak (12-16):   WR=58.9%, EV=+0.768  -- strong
    - London (08-12): WR=54.0%, EV=+0.621  -- moderate
    - Late NY (20-24):WR=52.2%, EV=+0.567  -- marginal
    - NY (16-20):     WR=47.1%, EV=+0.412  -- weakest session
    - Monday:         WR=49.2%             -- weakest day
    - Optimal global threshold: 0.60 (EV=+0.730)
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class SessionType(str, enum.Enum):
    """Trading session classification (UTC)."""
    PEAK = "peak"           # 12:00-16:00 UTC (London/NY overlap)
    LONDON = "london"       # 08:00-12:00 UTC
    NEW_YORK = "new_york"   # 16:00-20:00 UTC
    ASIA = "asia"           # 00:00-08:00 UTC
    LATE_NY = "late_ny"     # 20:00-24:00 UTC


@dataclass
class PolicyConfig:
    """Configuration for session-aware FX trading policy.

    Thresholds are the minimum calibrated P(win) required per session.
    Size multipliers scale position size relative to the base size.
    """
    # Session thresholds (min P(win) to approve)
    threshold_peak: float = 0.55       # Best overlap session
    threshold_london: float = 0.58     # Moderate
    threshold_new_york: float = 0.62   # Weakest — require more conviction
    threshold_asia: float = 0.55       # Strongest session
    threshold_late_ny: float = 0.60    # Marginal

    # Monday surcharge (weakest day, WR=49.2%)
    threshold_monday_add: float = 0.03

    # Size multipliers (relative to base position size)
    size_mult_peak: float = 1.0        # Full size during peak
    size_mult_london: float = 0.8      # 80% during London pre-overlap
    size_mult_new_york: float = 0.4    # 40% during weakest session
    size_mult_asia: float = 1.0        # Full size — strongest session
    size_mult_late_ny: float = 0.6     # 60% late NY
    size_mult_monday: float = 0.5      # 50% on Monday

    # Global enabled flag
    enabled: bool = True


class TradingPolicy:
    """Session-aware FX trading policy.

    Usage in the gate or main loop:
        policy = TradingPolicy()
        result = policy.evaluate(p_win=0.62, utc_now=datetime.now(timezone.utc))

        if result.approved:
            size = base_size * result.size_multiplier
    """

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = config or PolicyConfig()

    def get_session(self, utc_now: datetime) -> SessionType:
        """Classify the current UTC time into a trading session."""
        hour = utc_now.hour
        if 12 <= hour < 16:
            return SessionType.PEAK
        elif 8 <= hour < 12:
            return SessionType.LONDON
        elif 16 <= hour < 20:
            return SessionType.NEW_YORK
        elif 20 <= hour < 24:
            return SessionType.LATE_NY
        else:  # 0-8
            return SessionType.ASIA

    def get_threshold(self, utc_now: datetime) -> float:
        """Get the dynamic P(win) threshold for the current time."""
        if not self.config.enabled:
            return 0.55  # fallback to default

        session = self.get_session(utc_now)
        threshold_map = {
            SessionType.PEAK: self.config.threshold_peak,
            SessionType.LONDON: self.config.threshold_london,
            SessionType.NEW_YORK: self.config.threshold_new_york,
            SessionType.ASIA: self.config.threshold_asia,
            SessionType.LATE_NY: self.config.threshold_late_ny,
        }
        threshold = threshold_map[session]

        # Monday surcharge (weekday() == 0 is Monday)
        is_monday = utc_now.weekday() == 0
        if is_monday:
            threshold += self.config.threshold_monday_add

        return threshold

    def get_size_multiplier(self, p_win: float, utc_now: datetime) -> float:
        """Get position size multiplier based on session.

        Returns a multiplier in (0, 1] to apply to the base position size.
        """
        if not self.config.enabled:
            return 1.0

        session = self.get_session(utc_now)
        is_monday = utc_now.weekday() == 0

        # Session-based multiplier
        session_mult_map = {
            SessionType.PEAK: self.config.size_mult_peak,
            SessionType.LONDON: self.config.size_mult_london,
            SessionType.NEW_YORK: self.config.size_mult_new_york,
            SessionType.ASIA: self.config.size_mult_asia,
            SessionType.LATE_NY: self.config.size_mult_late_ny,
        }
        session_mult = session_mult_map[session]

        # Monday override (use the smaller of session mult and monday mult)
        if is_monday:
            session_mult = min(session_mult, self.config.size_mult_monday)

        return round(session_mult, 3)

    def evaluate(self, p_win: float, utc_now: Optional[datetime] = None) -> PolicyResult:
        """Full policy evaluation: threshold check + sizing.

        Parameters
        ----------
        p_win : float
            Calibrated probability of winning from the AI gate.
        utc_now : datetime, optional
            Current UTC time. Defaults to now.

        Returns
        -------
        PolicyResult
            Whether the trade is approved and at what size.
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        session = self.get_session(utc_now)
        threshold = self.get_threshold(utc_now)
        is_monday = utc_now.weekday() == 0

        approved = p_win >= threshold
        size_multiplier = self.get_size_multiplier(p_win, utc_now) if approved else 0.0

        reasons = []
        if not approved:
            reasons.append(
                f"P(win) {p_win:.3f} < session threshold {threshold:.3f} "
                f"({session.value}{' +mon' if is_monday else ''})"
            )

        result = PolicyResult(
            approved=approved,
            threshold=threshold,
            size_multiplier=size_multiplier,
            session=session,
            is_monday=is_monday,
            reasons=reasons,
        )

        logger.debug(
            "policy.evaluate",
            p_win=round(p_win, 4),
            session=session.value,
            threshold=round(threshold, 3),
            approved=approved,
            size_mult=round(size_multiplier, 3),
            is_monday=is_monday,
        )

        return result


@dataclass
class PolicyResult:
    """Result of a policy evaluation."""
    approved: bool
    threshold: float
    size_multiplier: float
    session: SessionType
    is_monday: bool
    reasons: list
