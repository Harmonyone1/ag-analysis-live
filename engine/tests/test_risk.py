"""Tests for risk management."""

import sys
import os
from decimal import Decimal

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk.manager import RiskManager, RiskLimits, RiskEventType


class TestRiskManager:
    """Tests for RiskManager class."""

    def test_init_default_limits(self):
        """Test initialization with default limits."""
        rm = RiskManager()
        assert rm.limits.max_risk_per_trade == 0.01
        assert rm.limits.max_daily_loss == 0.03
        assert rm.limits.max_open_positions == 5

    def test_init_custom_limits(self):
        """Test initialization with custom limits."""
        limits = RiskLimits(
            max_risk_per_trade=0.02,
            max_daily_loss=0.05,
            max_open_positions=3,
        )
        rm = RiskManager(limits=limits)
        assert rm.limits.max_risk_per_trade == 0.02
        assert rm.limits.max_daily_loss == 0.05
        assert rm.limits.max_open_positions == 3

    def test_check_new_trade_passes(self):
        """Test that valid trades pass risk checks."""
        rm = RiskManager(account_balance=Decimal("10000"))
        result = rm.check_new_trade(
            symbol="EURUSD",
            side="buy",
            risk_amount=Decimal("50"),  # 0.5% of account
        )
        assert result.passed is True
        assert result.reason == "All risk checks passed"

    def test_check_new_trade_exceeds_risk(self):
        """Test that trades exceeding risk limit fail."""
        rm = RiskManager(account_balance=Decimal("10000"))
        result = rm.check_new_trade(
            symbol="EURUSD",
            side="buy",
            risk_amount=Decimal("200"),  # 2% > 1% limit
        )
        assert result.passed is False
        assert "exceeds max" in result.reason

    def test_check_new_trade_trading_disabled(self):
        """Test that trades fail when trading is disabled."""
        rm = RiskManager()
        rm.disable_trading()
        result = rm.check_new_trade(
            symbol="EURUSD",
            side="buy",
            risk_amount=Decimal("50"),
        )
        assert result.passed is False
        assert result.event_type == RiskEventType.KILL_SWITCH

    def test_check_new_trade_max_positions(self):
        """Test that trades fail when max positions reached."""
        limits = RiskLimits(max_open_positions=1)
        rm = RiskManager(limits=limits, account_balance=Decimal("10000"))

        # Add a position
        from risk.manager import OpenPosition
        rm.update_positions([
            OpenPosition(
                symbol="EURUSD",
                side="buy",
                size=Decimal("0.1"),
                risk_amount=Decimal("50"),
                currencies={"EUR_LONG", "USD_SHORT"},
            )
        ])

        result = rm.check_new_trade(
            symbol="GBPUSD",
            side="buy",
            risk_amount=Decimal("50"),
        )
        assert result.passed is False
        assert result.event_type == RiskEventType.MAX_POSITIONS_HIT

    def test_daily_pnl_tracking(self):
        """Test daily P&L tracking."""
        rm = RiskManager(account_balance=Decimal("10000"))
        rm.update_daily_pnl(Decimal("-100"))
        assert rm.daily_pnl == Decimal("-100")

    def test_daily_loss_limit_hit(self):
        """Test that trading is disabled when daily loss limit hit."""
        rm = RiskManager(account_balance=Decimal("10000"))
        rm.update_daily_pnl(Decimal("-350"))  # 3.5% > 3% limit
        assert rm.is_trading_enabled is False

    def test_enable_disable_trading(self):
        """Test enable/disable trading."""
        rm = RiskManager()
        assert rm.is_trading_enabled is True

        rm.disable_trading()
        assert rm.is_trading_enabled is False

        rm.enable_trading()
        assert rm.is_trading_enabled is True

    def test_calculate_position_size(self):
        """Test position size calculation."""
        rm = RiskManager()
        size = rm.calculate_position_size(
            account_balance=Decimal("10000"),
            risk_percent=0.01,  # 1% = $100 risk
            stop_distance=20,  # 20 pips
            pip_value=10,  # $10 per pip per lot
        )
        # $100 / (20 pips * $10/pip) = 0.5 lots
        assert size == Decimal("0.50")

    def test_spread_filter(self):
        """Test spread filter."""
        limits = RiskLimits(max_spread_multiplier=2.0)
        rm = RiskManager(limits=limits, account_balance=Decimal("10000"))

        # Normal spread should pass
        result = rm.check_new_trade(
            symbol="EURUSD",
            side="buy",
            risk_amount=Decimal("50"),
            spread=1.5,
            typical_spread=1.0,
        )
        assert result.passed is True

        # Elevated spread should fail
        result = rm.check_new_trade(
            symbol="EURUSD",
            side="buy",
            risk_amount=Decimal("50"),
            spread=3.0,
            typical_spread=1.0,
        )
        assert result.passed is False
        assert result.event_type == RiskEventType.SPREAD_FILTER

    def test_get_state(self):
        """Test state retrieval."""
        rm = RiskManager(account_balance=Decimal("10000"))
        state = rm.get_state()
        assert "trading_enabled" in state
        assert "daily_pnl" in state
        assert "account_balance" in state
        assert state["account_balance"] == 10000.0
