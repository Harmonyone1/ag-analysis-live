"""Risk Management Engine for trade safety and position limits.

Enforces:
- Max risk per trade
- Daily loss limits
- Position limits
- Correlation caps
- Spread/slippage filters
"""

from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set
import structlog

logger = structlog.get_logger(__name__)


class RiskEventType(Enum):
    """Types of risk events."""
    DAILY_LOSS_HIT = "DAILY_LOSS_HIT"
    MAX_POSITIONS_HIT = "MAX_POSITIONS_HIT"
    CORRELATION_LIMIT = "CORRELATION_LIMIT"
    SPREAD_FILTER = "SPREAD_FILTER"
    SLIPPAGE_HALT = "SLIPPAGE_HALT"
    DATA_STALE = "DATA_STALE"
    KILL_SWITCH = "KILL_SWITCH"


class Severity(Enum):
    """Risk event severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    max_risk_per_trade: float = 0.01  # 1% of account
    max_daily_loss: float = 0.03  # 3% of account
    max_open_positions: int = 5
    max_correlated_exposure: float = 0.02  # 2% per currency
    max_spread_multiplier: float = 3.0  # Max 3x normal spread
    slippage_halt_threshold: float = 5.0  # Pips
    slippage_lookback_trades: int = 5
    # Portfolio heat limits
    max_portfolio_heat: float = 0.05  # 5% total heat (realized + unrealized)
    heat_warning_threshold: float = 0.8  # Warn at 80% of heat limit
    max_drawdown: float = 0.10  # 10% max drawdown from peak


@dataclass
class RiskCheck:
    """Result of a risk check."""
    passed: bool
    reason: str
    event_type: Optional[RiskEventType] = None
    severity: Severity = Severity.INFO
    risk_multiplier: float = 1.0  # Suggested position size multiplier


@dataclass
class OpenPosition:
    """Simplified position for risk calculations."""
    symbol: str
    side: str
    size: Decimal
    risk_amount: Decimal  # Account currency risk
    currencies: Set[str]


class RiskManager:
    """Manages all risk controls for the trading system.

    Example:
        risk_mgr = RiskManager(limits)
        check = risk_mgr.check_new_trade(symbol, size, risk_amount)
        if not check.passed:
            logger.warning(f"Trade rejected: {check.reason}")
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        account_balance: Decimal = Decimal("10000"),
    ):
        """Initialize risk manager.

        Args:
            limits: Risk limit configuration
            account_balance: Current account balance
        """
        self.limits = limits or RiskLimits()
        self.account_balance = account_balance

        # State tracking
        self._daily_pnl: Decimal = Decimal("0")
        self._daily_trades: int = 0
        self._last_reset_date: date = date.today()
        self._open_positions: List[OpenPosition] = []
        self._recent_slippage: List[float] = []
        self._trading_enabled: bool = True
        self._risk_events: List[Dict] = []

        # Portfolio heat tracking
        self._unrealized_pnl: Decimal = Decimal("0")
        self._peak_balance: Decimal = account_balance
        self._max_drawdown_seen: float = 0.0

        # Currency correlation groups
        self._usd_pairs = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"}

    def update_account_balance(self, balance: Decimal) -> None:
        """Update account balance for risk calculations."""
        self.account_balance = balance
        # Update peak balance for drawdown calculation
        if balance > self._peak_balance:
            self._peak_balance = balance

    def update_unrealized_pnl(self, unrealized_pnl: Decimal) -> None:
        """Update unrealized P&L from open positions."""
        self._unrealized_pnl = unrealized_pnl
        self._check_portfolio_heat()
        self._check_drawdown()

    def _check_portfolio_heat(self) -> None:
        """Check total portfolio heat (realized + unrealized losses)."""
        if self.account_balance <= 0:
            return

        # Total heat = realized daily loss + unrealized loss
        realized_loss = min(Decimal("0"), self._daily_pnl)
        unrealized_loss = min(Decimal("0"), self._unrealized_pnl)
        total_heat = abs(realized_loss) + abs(unrealized_loss)

        heat_pct = float(total_heat) / float(self.account_balance)
        max_heat = self.limits.max_portfolio_heat

        # Warning at threshold
        if heat_pct >= max_heat * self.limits.heat_warning_threshold:
            if heat_pct < max_heat:
                self._log_risk_event(
                    RiskEventType.DAILY_LOSS_HIT,
                    Severity.WARNING,
                    {"heat_pct": heat_pct, "threshold": max_heat * self.limits.heat_warning_threshold,
                     "realized": float(realized_loss), "unrealized": float(unrealized_loss)}
                )
                logger.warning("Portfolio heat approaching limit",
                             heat_pct=f"{heat_pct:.1%}", limit=f"{max_heat:.1%}")

        # Critical at limit
        if heat_pct >= max_heat:
            self._trading_enabled = False
            self._log_risk_event(
                RiskEventType.DAILY_LOSS_HIT,
                Severity.CRITICAL,
                {"heat_pct": heat_pct, "limit": max_heat,
                 "realized": float(realized_loss), "unrealized": float(unrealized_loss)}
            )
            logger.critical("Portfolio heat limit hit - trading disabled",
                          heat_pct=f"{heat_pct:.1%}", limit=f"{max_heat:.1%}")

    def _check_drawdown(self) -> None:
        """Check drawdown from peak equity."""
        if self._peak_balance <= 0:
            return

        current_equity = self.account_balance + self._unrealized_pnl
        drawdown = float(self._peak_balance - current_equity) / float(self._peak_balance)

        # Track max drawdown seen
        if drawdown > self._max_drawdown_seen:
            self._max_drawdown_seen = drawdown

        # Check against limit
        if drawdown >= self.limits.max_drawdown:
            self._trading_enabled = False
            self._log_risk_event(
                RiskEventType.DAILY_LOSS_HIT,
                Severity.CRITICAL,
                {"drawdown": drawdown, "limit": self.limits.max_drawdown,
                 "peak": float(self._peak_balance), "current": float(current_equity)}
            )
            logger.critical("Max drawdown hit - trading disabled",
                          drawdown=f"{drawdown:.1%}", limit=f"{self.limits.max_drawdown:.1%}")

    def get_portfolio_heat(self) -> Dict:
        """Get current portfolio heat metrics."""
        realized_loss = abs(min(Decimal("0"), self._daily_pnl))
        unrealized_loss = abs(min(Decimal("0"), self._unrealized_pnl))
        total_heat = realized_loss + unrealized_loss

        heat_pct = float(total_heat) / float(self.account_balance) if self.account_balance > 0 else 0
        current_equity = self.account_balance + self._unrealized_pnl
        drawdown = float(self._peak_balance - current_equity) / float(self._peak_balance) if self._peak_balance > 0 else 0

        return {
            "realized_loss": float(realized_loss),
            "unrealized_loss": float(unrealized_loss),
            "total_heat": float(total_heat),
            "heat_pct": heat_pct,
            "heat_limit": self.limits.max_portfolio_heat,
            "heat_remaining": max(0, self.limits.max_portfolio_heat - heat_pct),
            "current_drawdown": drawdown,
            "max_drawdown_seen": self._max_drawdown_seen,
            "drawdown_limit": self.limits.max_drawdown,
            "peak_balance": float(self._peak_balance),
            "current_equity": float(current_equity),
        }

    def update_daily_pnl(self, pnl: Decimal) -> None:
        """Update daily P&L."""
        self._check_daily_reset()
        self._daily_pnl = pnl

        # Check daily loss limit
        if self.account_balance > 0:
            loss_pct = -float(pnl) / float(self.account_balance)
            if loss_pct >= self.limits.max_daily_loss:
                self._trading_enabled = False
                self._log_risk_event(
                    RiskEventType.DAILY_LOSS_HIT,
                    Severity.CRITICAL,
                    {"daily_loss_pct": loss_pct, "limit": self.limits.max_daily_loss}
                )
                logger.critical("Daily loss limit hit - trading disabled",
                              loss_pct=loss_pct, limit=self.limits.max_daily_loss)

    def update_positions(self, positions: List[OpenPosition]) -> None:
        """Update current open positions."""
        self._open_positions = positions

    def record_slippage(self, slippage_pips: float) -> None:
        """Record trade slippage for monitoring."""
        self._recent_slippage.append(slippage_pips)
        if len(self._recent_slippage) > self.limits.slippage_lookback_trades:
            self._recent_slippage.pop(0)

        # Check slippage halt
        if len(self._recent_slippage) >= self.limits.slippage_lookback_trades:
            avg_slippage = sum(self._recent_slippage) / len(self._recent_slippage)
            if avg_slippage > self.limits.slippage_halt_threshold:
                self._log_risk_event(
                    RiskEventType.SLIPPAGE_HALT,
                    Severity.WARNING,
                    {"avg_slippage": avg_slippage, "threshold": self.limits.slippage_halt_threshold}
                )

    def check_new_trade(
        self,
        symbol: str,
        side: str,
        risk_amount: Decimal,
        spread: Optional[float] = None,
        typical_spread: Optional[float] = None,
    ) -> RiskCheck:
        """Check if a new trade passes all risk filters.

        Args:
            symbol: Trading symbol
            side: Trade side (buy/sell)
            risk_amount: Risk in account currency
            spread: Current spread
            typical_spread: Normal spread for comparison

        Returns:
            RiskCheck result
        """
        self._check_daily_reset()

        # Check trading enabled
        if not self._trading_enabled:
            return RiskCheck(
                passed=False,
                reason="Trading disabled - daily loss limit hit or kill switch active",
                event_type=RiskEventType.KILL_SWITCH,
                severity=Severity.CRITICAL,
            )

        # Check portfolio heat before adding risk
        heat_info = self.get_portfolio_heat()
        if heat_info["heat_pct"] >= self.limits.max_portfolio_heat * 0.9:
            return RiskCheck(
                passed=False,
                reason=f"Portfolio heat {heat_info['heat_pct']:.1%} near limit {self.limits.max_portfolio_heat:.1%}",
                event_type=RiskEventType.DAILY_LOSS_HIT,
                severity=Severity.WARNING,
            )

        # Check max risk per trade
        max_risk = self.account_balance * Decimal(str(self.limits.max_risk_per_trade))
        if risk_amount > max_risk:
            return RiskCheck(
                passed=False,
                reason=f"Risk amount {risk_amount} exceeds max {max_risk}",
                severity=Severity.WARNING,
                risk_multiplier=float(max_risk / risk_amount),
            )

        # Check position count
        if len(self._open_positions) >= self.limits.max_open_positions:
            return RiskCheck(
                passed=False,
                reason=f"Max positions ({self.limits.max_open_positions}) reached",
                event_type=RiskEventType.MAX_POSITIONS_HIT,
                severity=Severity.WARNING,
            )

        # Check correlation exposure
        corr_check = self._check_correlation(symbol, side, risk_amount)
        if not corr_check.passed:
            return corr_check

        # Check spread filter
        if spread is not None and typical_spread is not None and typical_spread > 0:
            spread_multiple = spread / typical_spread
            if spread_multiple > self.limits.max_spread_multiplier:
                return RiskCheck(
                    passed=False,
                    reason=f"Spread {spread:.1f} is {spread_multiple:.1f}x normal",
                    event_type=RiskEventType.SPREAD_FILTER,
                    severity=Severity.INFO,
                )

        return RiskCheck(passed=True, reason="All risk checks passed")

    def _check_correlation(
        self,
        symbol: str,
        side: str,
        risk_amount: Decimal,
    ) -> RiskCheck:
        """Check correlation/currency exposure limits."""
        # Extract currencies from symbol
        currencies = self._get_currencies(symbol, side)

        # Calculate existing exposure per currency
        exposure: Dict[str, Decimal] = {}
        for pos in self._open_positions:
            for ccy in pos.currencies:
                exposure[ccy] = exposure.get(ccy, Decimal("0")) + pos.risk_amount

        # Check if new trade would exceed limits
        max_exposure = self.account_balance * Decimal(str(self.limits.max_correlated_exposure))

        for ccy in currencies:
            current = exposure.get(ccy, Decimal("0"))
            new_total = current + risk_amount
            if new_total > max_exposure:
                return RiskCheck(
                    passed=False,
                    reason=f"{ccy} exposure {new_total} exceeds limit {max_exposure}",
                    event_type=RiskEventType.CORRELATION_LIMIT,
                    severity=Severity.WARNING,
                )

        return RiskCheck(passed=True, reason="Correlation check passed")

    def _get_currencies(self, symbol: str, side: str) -> Set[str]:
        """Extract currency exposure from symbol and side."""
        currencies = set()
        symbol_upper = symbol.upper()

        if len(symbol_upper) == 6:
            base = symbol_upper[:3]
            quote = symbol_upper[3:]

            # Long = long base, short quote
            # Short = short base, long quote
            if side.lower() == "buy":
                currencies.add(f"{base}_LONG")
                currencies.add(f"{quote}_SHORT")
            else:
                currencies.add(f"{base}_SHORT")
                currencies.add(f"{quote}_LONG")

        return currencies

    def _check_daily_reset(self) -> None:
        """Reset daily counters if new day."""
        today = date.today()
        if today != self._last_reset_date:
            self._daily_pnl = Decimal("0")
            self._daily_trades = 0
            self._last_reset_date = today
            self._trading_enabled = True  # Re-enable trading
            logger.info("Daily risk counters reset")

    def _log_risk_event(
        self,
        event_type: RiskEventType,
        severity: Severity,
        payload: Dict,
    ) -> None:
        """Log a risk event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type.value,
            "severity": severity.value,
            "payload": payload,
        }
        self._risk_events.append(event)
        logger.warning("Risk event", **event)

    def enable_trading(self) -> None:
        """Enable trading (admin action)."""
        self._trading_enabled = True
        logger.info("Trading enabled")

    def disable_trading(self) -> None:
        """Disable trading (kill switch)."""
        self._trading_enabled = False
        self._log_risk_event(
            RiskEventType.KILL_SWITCH,
            Severity.CRITICAL,
            {"action": "manual_disable"}
        )
        logger.warning("Trading disabled via kill switch")

    @property
    def is_trading_enabled(self) -> bool:
        """Check if trading is enabled."""
        return self._trading_enabled

    @property
    def daily_pnl(self) -> Decimal:
        """Get current daily P&L."""
        return self._daily_pnl

    @property
    def open_position_count(self) -> int:
        """Get current open position count."""
        return len(self._open_positions)

    @property
    def recent_events(self) -> List[Dict]:
        """Get recent risk events."""
        return self._risk_events[-20:]

    def calculate_position_size(
        self,
        account_balance: Decimal,
        risk_percent: float,
        stop_distance: float,
        pip_value: float,
    ) -> Decimal:
        """Calculate position size based on risk parameters.

        Args:
            account_balance: Account balance
            risk_percent: Risk as decimal (0.01 = 1%)
            stop_distance: Stop distance in pips
            pip_value: Value per pip per lot

        Returns:
            Position size in lots
        """
        if stop_distance <= 0 or pip_value <= 0:
            return Decimal("0")

        risk_amount = account_balance * Decimal(str(risk_percent))
        position_size = risk_amount / (Decimal(str(stop_distance)) * Decimal(str(pip_value)))

        # Round to 2 decimal places (standard lot precision)
        return position_size.quantize(Decimal("0.01"))

    def get_state(self) -> Dict:
        """Get current risk manager state."""
        return {
            "trading_enabled": self._trading_enabled,
            "daily_pnl": float(self._daily_pnl),
            "daily_trades": self._daily_trades,
            "open_positions": self.open_position_count,
            "account_balance": float(self.account_balance),
            "max_positions": self.limits.max_open_positions,
            "max_daily_loss_pct": self.limits.max_daily_loss,
            "current_daily_loss_pct": float(-self._daily_pnl / self.account_balance) if self.account_balance else 0,
        }
