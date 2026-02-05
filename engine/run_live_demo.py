#!/usr/bin/env python
"""Run AG Analyzer with REAL orders on TradeLocker Demo Account.

This script places actual trades on your TradeLocker demo account.
It's safe to test because it's a demo account with virtual money.

Features:
- Places real market orders with SL/TP
- Monitors and manages live positions
- Applies trailing stop management
- Tracks performance in real-time

Usage: python run_live_demo.py [--duration HOURS] [--lower-thresholds]
"""

import sys
import os
import asyncio
import logging
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from decimal import Decimal
from pathlib import Path

# Set up paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
import structlog
import numpy as np

# Load environment
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Symbols to scan
# All 28 major and cross forex pairs
SYMBOLS = [
    # Majors (7)
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    # EUR crosses (6)
    "EURGBP", "EURJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF",
    # GBP crosses (5)
    "GBPJPY", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
    # AUD/NZD crosses (4)
    "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF",
    # NZD crosses (2)
    "NZDCAD", "NZDCHF",
    # CAD/CHF/JPY crosses (4)
    "CADJPY", "CHFJPY", "CADCHF",
]

# Results file
RESULTS_FILE = Path(__file__).parent.parent / "live_demo_results.json"


def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_price: float,
    pip_value: float = 10.0,
    symbol: str = ""
) -> float:
    """Calculate position size based on risk.

    Args:
        account_balance: Account balance
        risk_percent: Risk per trade (e.g., 0.01 for 1%)
        entry_price: Entry price
        stop_price: Stop loss price
        pip_value: Value per pip per lot (default $10)
        symbol: Trading symbol (for JPY pair detection)

    Returns:
        Position size in lots
    """
    risk_amount = account_balance * risk_percent

    # Detect JPY pairs (2 decimal places instead of 4)
    if "JPY" in symbol:
        stop_distance_pips = abs(entry_price - stop_price) * 100  # JPY: 1 pip = 0.01
    else:
        stop_distance_pips = abs(entry_price - stop_price) * 10000  # Standard: 1 pip = 0.0001

    # Enforce minimum stop distance (5 pips) to prevent huge positions
    min_stop_pips = 5.0
    if stop_distance_pips < min_stop_pips:
        logger.warning(f"Stop distance {stop_distance_pips:.1f} pips < minimum {min_stop_pips} pips, using minimum")
        stop_distance_pips = min_stop_pips

    if stop_distance_pips <= 0:
        return 0.01

    lots = risk_amount / (stop_distance_pips * pip_value)

    # Cap position size for safety (0.01 min, 0.1 max for demo)
    lots = max(0.01, min(0.10, lots))
    return round(lots, 2)


class LiveDemoSession:
    """Manages live demo trading session with advanced trade management."""

    def __init__(
        self,
        broker,
        analyzer,
        scorer,
        ai_gate,
        config,
        lower_thresholds: bool = False,
        aggressive_mode: bool = False,
    ):
        self.broker = broker
        self.analyzer = analyzer
        self.scorer = scorer
        self.ai_gate = ai_gate
        self.config = config
        self.lower_thresholds = lower_thresholds
        self.aggressive_mode = aggressive_mode

        # Session stats
        self.cycle_count = 0
        self.setups_found = 0
        self.approved_count = 0
        self.rejected_count = 0
        self.orders_placed = 0
        self.scales_executed = 0
        self.early_exits = 0
        self.start_time = None
        self.starting_balance = None

        # Trade tracking
        self.trade_records: List[Dict] = []
        self.managed_positions: Dict[int, Dict] = {}  # position_id -> management info

        # Thresholds
        if lower_thresholds:
            self.min_probability = 0.40
            self.min_expected_r = 0.10
            self.min_confluence = 55
            logger.warning("Using LOWERED thresholds for testing!")
        else:
            self.min_probability = config.ai.min_ai_probability
            self.min_expected_r = config.ai.min_expected_value
            self.min_confluence = config.ai.min_confluence_score

        # Trade management settings
        self.move_to_breakeven_r = 1.0
        self.breakeven_buffer_pips = 2.0
        self.max_positions = 5 if aggressive_mode else 3

        # Initialize advanced trade management
        from src.execution.advanced_manager import (
            create_advanced_manager,
            TradeHealth,
            ScaleAction,
        )
        self.TradeHealth = TradeHealth
        self.ScaleAction = ScaleAction

        advanced = create_advanced_manager(
            broker, analyzer, scorer, aggressive=aggressive_mode
        )
        self.health_monitor = advanced["health_monitor"]
        self.scaler = advanced["scaler"]
        self.position_sizer = advanced["sizer"]
        self.scaling_config = advanced["scaling_config"]
        self.sizing_config = advanced["sizing_config"]

        if aggressive_mode:
            logger.warning("AGGRESSIVE MODE: Larger positions, more scaling")
            logger.info(f"  Max positions: {self.max_positions}")
            logger.info(f"  Base risk: {self.sizing_config.base_risk_percent:.1%}")
            logger.info(f"  Max risk: {self.sizing_config.max_risk_percent:.1%}")
            logger.info(f"  Max lots: {self.sizing_config.max_lots}")

    def should_approve_trade(self, decision) -> bool:
        """Check if trade meets approval criteria."""
        from src.ai import GateDecisionType

        if self.lower_thresholds:
            return (
                decision.probability >= self.min_probability and
                decision.expected_r >= self.min_expected_r
            )
        else:
            return decision.decision == GateDecisionType.APPROVED

    def get_pip_size(self, symbol: str) -> float:
        """Get pip size for symbol."""
        return 0.01 if "JPY" in symbol else 0.0001

    async def manage_positions(self):
        """Monitor and manage live positions with advanced features."""
        try:
            positions = self.broker.list_positions()

            for pos in positions:
                position_id = pos.position_id
                symbol = pos.symbol
                pip_size = self.get_pip_size(symbol)

                # Get or create management info
                if position_id not in self.managed_positions:
                    self.managed_positions[position_id] = {
                        "entry_price": float(pos.avg_price),
                        "original_stop": float(pos.stop_loss) if pos.stop_loss else None,
                        "break_even_moved": False,
                        "direction": pos.side,
                        "health_warnings": 0,
                    }

                mgmt = self.managed_positions[position_id]
                entry = mgmt["entry_price"]
                current = float(pos.current_price)

                # Calculate current R (handle missing stop)
                stop = mgmt["original_stop"]
                if stop:
                    if pos.side == "buy":
                        risk_pips = (entry - stop) / pip_size
                        current_pips = (current - entry) / pip_size
                    else:
                        risk_pips = (stop - entry) / pip_size
                        current_pips = (entry - current) / pip_size

                    current_r = current_pips / risk_pips if risk_pips > 0 else 0
                else:
                    current_r = 0

                # === ADVANCED: Trade Health Monitoring ===
                try:
                    candles = self.broker.get_candles(symbol, "M15", limit=50)
                    if candles and len(candles) >= 20:
                        health_report = self.health_monitor.analyze_trade_health(
                            symbol=symbol,
                            direction=pos.side,
                            entry_price=entry,
                            current_price=current,
                            candles=candles,
                        )

                        # Log health status
                        health_emoji = {
                            self.TradeHealth.STRONG: "💪",
                            self.TradeHealth.HEALTHY: "✅",
                            self.TradeHealth.WEAKENING: "⚠️",
                            self.TradeHealth.CRITICAL: "🚨",
                        }.get(health_report.health, "❓")

                        logger.info(
                            f"  {symbol} {pos.side.upper()}: {current_r:.2f}R | "
                            f"Health: {health_emoji} {health_report.health.value} ({health_report.score:.0f}) | "
                            f"P&L: ${float(pos.unrealized_pnl):.2f}"
                        )

                        # Log reasons if concerning
                        if health_report.health in (self.TradeHealth.WEAKENING, self.TradeHealth.CRITICAL):
                            for reason in health_report.reasons[:2]:
                                logger.warning(f"    → {reason}")
                            mgmt["health_warnings"] += 1

                        # === CRITICAL HEALTH: Consider early exit ===
                        if health_report.health == self.TradeHealth.CRITICAL:
                            if current_r > 0:  # Only exit if in profit
                                logger.warning(
                                    f"  🚨 CRITICAL: Closing {symbol} early (health={health_report.score:.0f})"
                                )
                                try:
                                    self.broker.close_position(position_id)
                                    self.early_exits += 1
                                    logger.info(f"  ✅ Early exit executed at {current_r:.2f}R")
                                    continue
                                except Exception as e:
                                    logger.error(f"  Failed to close: {e}")

                        # === STRONG HEALTH: Consider scaling ===
                        if (health_report.health == self.TradeHealth.STRONG and
                            health_report.recommended_action == self.ScaleAction.ADD):

                            should_scale, scale_size, reason = self.scaler.evaluate_scaling(
                                position=pos,
                                candles=candles,
                                current_r=current_r,
                            )

                            if should_scale and scale_size:
                                logger.info(
                                    f"  📈 SCALING: Adding {scale_size} lots to {symbol} ({reason})"
                                )
                                if self.scaler.execute_scale(pos, scale_size, current):
                                    self.scales_executed += 1

                        # === DYNAMIC TP ADJUSTMENT ===
                        new_tp = self.scaler.adjust_take_profit(pos, health_report)
                        if new_tp:
                            try:
                                self.broker.modify_position(
                                    position_id=position_id,
                                    take_profit=Decimal(str(new_tp))
                                )
                                logger.info(f"  📊 TP adjusted to {new_tp:.5f}")
                            except Exception as e:
                                logger.debug(f"  TP adjustment failed: {e}")

                except Exception as e:
                    logger.debug(f"Health check failed for {symbol}: {e}")
                    # Fallback to basic logging
                    logger.info(
                        f"  {symbol} {pos.side.upper()}: {current_r:.2f}R | "
                        f"Entry: {entry:.5f} | Current: {current:.5f} | "
                        f"P&L: ${float(pos.unrealized_pnl):.2f}"
                    )

                # === BASIC: Move to break-even at 1R ===
                if stop and not mgmt["break_even_moved"] and current_r >= self.move_to_breakeven_r:
                    buffer = self.breakeven_buffer_pips * pip_size

                    if pos.side == "buy":
                        new_stop = entry + buffer
                    else:
                        new_stop = entry - buffer

                    try:
                        success = self.broker.modify_position(
                            position_id=position_id,
                            stop_loss=Decimal(str(new_stop))
                        )
                        if success:
                            mgmt["break_even_moved"] = True
                            logger.info(
                                f"  🔒 Moved to break-even: SL={new_stop:.5f}"
                            )
                    except Exception as e:
                        logger.debug(f"Failed to move SL: {e}")

        except Exception as e:
            logger.error(f"Error managing positions: {e}")

    async def run_cycle(self) -> Dict:
        """Run a single trading cycle."""
        from src.ai import GateDecisionType
        from src.adapters.broker import OrderRequest

        cycle_start = datetime.now()
        self.cycle_count += 1
        cycle_stats = {
            "setups": 0,
            "approved": 0,
            "orders_placed": 0,
        }

        logger.info(f"--- Cycle {self.cycle_count} @ {cycle_start.strftime('%H:%M:%S')} ---")

        # First, manage existing positions
        positions = self.broker.list_positions()
        if positions:
            logger.info(f"Managing {len(positions)} open positions:")
            await self.manage_positions()

        # Check if we can open more trades
        current_position_count = len(positions)
        if current_position_count >= self.max_positions:
            logger.info(f"Max positions ({self.max_positions}) reached, skipping scan")
            return cycle_stats

        # Calculate how many new orders we can place this cycle
        orders_available = self.max_positions - current_position_count
        orders_placed_this_cycle = 0

        # Get symbols we already have positions in
        position_symbols = {p.symbol for p in positions}

        # Scan for new setups
        for symbol in SYMBOLS:
            if symbol in position_symbols:
                continue

            try:
                candles = self.broker.get_candles(symbol, "M15", limit=200)
                if not candles or len(candles) < 50:
                    continue

                opens = np.array([float(c.open) for c in candles])
                highs = np.array([float(c.high) for c in candles])
                lows = np.array([float(c.low) for c in candles])
                closes = np.array([float(c.close) for c in candles])
                timestamps = [c.timestamp for c in candles]
                current_price = closes[-1]

                market_view = self.analyzer.analyze(symbol, "M15", opens, highs, lows, closes, timestamps)
                setups = self.scorer.generate_setups(market_view)

                for setup in setups:
                    if setup.confluence_score >= self.min_confluence:
                        self.setups_found += 1
                        cycle_stats["setups"] += 1

                        decision = self.ai_gate.evaluate(setup, market_view)

                        if self.should_approve_trade(decision):
                            self.approved_count += 1
                            cycle_stats["approved"] += 1

                            logger.info("")
                            logger.info("=" * 50)
                            logger.info(f"TRADE SIGNAL: {symbol} {setup.direction}")
                            logger.info("=" * 50)
                            logger.info(f"  Confluence: {setup.confluence_score}")
                            logger.info(f"  P(win): {decision.probability:.1%}")
                            logger.info(f"  E[R]: {decision.expected_r:.2f}")

                            # Calculate position size using advanced sizer
                            account = self.broker.get_account()
                            stop_price = float(setup.stop_price)

                            # Validate stop is in correct direction
                            if setup.direction == "LONG" and stop_price >= current_price:
                                logger.warning(f"Invalid SL {stop_price} for LONG at {current_price}, skipping")
                                continue
                            elif setup.direction == "SHORT" and stop_price <= current_price:
                                logger.warning(f"Invalid SL {stop_price} for SHORT at {current_price}, skipping")
                                continue

                            # Calculate existing risk exposure
                            existing_risk = len(positions) * self.sizing_config.base_risk_percent

                            # Use advanced position sizer
                            quantity, size_details = self.position_sizer.calculate_size(
                                account_balance=float(account.balance),
                                entry_price=current_price,
                                stop_price=stop_price,
                                symbol=symbol,
                                probability=decision.probability,
                                aggressive_mode=self.aggressive_mode,
                                existing_risk=existing_risk,
                            )

                            if quantity <= 0:
                                logger.warning(f"Position size 0 for {symbol}, skipping")
                                continue

                            logger.info(f"  Size: {quantity} lots ({size_details['reason']})")

                            # Get first TP and validate it
                            tp_price = None
                            if setup.tp_targets:
                                candidate_tp = float(setup.tp_targets[0].get("price", 0))
                                # Validate TP is in correct direction
                                if setup.direction == "LONG" and candidate_tp > current_price:
                                    tp_price = candidate_tp
                                elif setup.direction == "SHORT" and candidate_tp < current_price:
                                    tp_price = candidate_tp
                                else:
                                    logger.warning(f"Invalid TP {candidate_tp} for {setup.direction} at {current_price}, skipping TP")

                            # Get instrument ID
                            instrument = self.broker.get_instrument(symbol)
                            if not instrument:
                                logger.error(f"Instrument not found: {symbol}")
                                continue

                            # Place order
                            side = "buy" if setup.direction == "LONG" else "sell"

                            try:
                                order_request = OrderRequest(
                                    instrument_id=instrument.instrument_id,
                                    symbol=symbol,
                                    side=side,
                                    quantity=Decimal(str(quantity)),
                                    order_type="market",
                                    stop_loss=Decimal(str(stop_price)),
                                    stop_loss_type="absolute",
                                    take_profit=Decimal(str(tp_price)) if tp_price else None,
                                    take_profit_type="absolute" if tp_price else None,
                                )

                                order = self.broker.place_order(order_request)
                                self.orders_placed += 1
                                cycle_stats["orders_placed"] += 1

                                logger.info(f"  ORDER PLACED: {order.order_id}")
                                logger.info(f"  Size: {quantity} lots")
                                logger.info(f"  Stop: {stop_price:.5f}")
                                logger.info(f"  TP: {tp_price:.5f}" if tp_price else "  TP: None")
                                logger.info("=" * 50)

                                # Record trade
                                self.trade_records.append({
                                    "order_id": order.order_id,
                                    "symbol": symbol,
                                    "direction": setup.direction,
                                    "quantity": quantity,
                                    "ai_probability": decision.probability,
                                    "ai_expected_r": decision.expected_r,
                                    "confluence_score": setup.confluence_score,
                                    "stop_price": stop_price,
                                    "tp_price": tp_price,
                                    "timestamp": datetime.now().isoformat(),
                                })

                                # Track orders placed and prevent duplicates
                                orders_placed_this_cycle += 1
                                position_symbols.add(symbol)

                                # Check if we've hit the limit
                                if orders_placed_this_cycle >= orders_available:
                                    logger.info(f"Max new orders ({orders_available}) placed this cycle")
                                    return cycle_stats

                            except Exception as e:
                                logger.error(f"Failed to place order: {e}")

                        elif decision.decision == GateDecisionType.NEEDS_REVIEW:
                            logger.info(f"{symbol} {setup.direction}: NEEDS_REVIEW - P(win)={decision.probability:.1%}")
                        else:
                            self.rejected_count += 1

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue

        cycle_time = (datetime.now() - cycle_start).total_seconds()
        logger.info(f"Cycle complete: {cycle_time:.1f}s | Setups: {cycle_stats['setups']} | Orders: {cycle_stats['orders_placed']}")

        return cycle_stats

    def get_session_summary(self) -> Dict:
        """Get session summary."""
        try:
            account = self.broker.get_account()
            current_balance = float(account.balance)
            pnl = current_balance - self.starting_balance if self.starting_balance else 0
        except:
            current_balance = 0
            pnl = 0

        return {
            "session": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": datetime.now().isoformat(),
                "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60 if self.start_time else 0,
                "cycles": self.cycle_count,
                "lower_thresholds": self.lower_thresholds,
            },
            "account": {
                "starting_balance": self.starting_balance,
                "ending_balance": current_balance,
                "pnl": pnl,
            },
            "activity": {
                "setups_found": self.setups_found,
                "approved": self.approved_count,
                "rejected": self.rejected_count,
                "orders_placed": self.orders_placed,
            },
            "trades": self.trade_records,
        }

    def save_results(self):
        """Save results to file."""
        summary = self.get_session_summary()

        if RESULTS_FILE.exists():
            with open(RESULTS_FILE, "r") as f:
                all_results = json.load(f)
        else:
            all_results = {"sessions": []}

        all_results["sessions"].append(summary)
        all_results["last_updated"] = datetime.now().isoformat()

        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info(f"Results saved to {RESULTS_FILE}")


async def run_live_demo(
    duration_hours: float = 1.0,
    lower_thresholds: bool = False,
    aggressive_mode: bool = False,
):
    """Run live demo trading session.

    Args:
        duration_hours: How long to run the session
        lower_thresholds: Use lower approval thresholds for testing
        aggressive_mode: Enable aggressive position sizing and scaling
    """
    from src.config import load_config
    from src.adapters import TradeLockerAdapter
    from src.analysis import MarketAnalyzer
    from src.scoring import ConfluenceScorer
    from src.ai import AIGate, GateConfig

    config = load_config(env_path)

    logger.info("=" * 60)
    logger.info("AG ANALYZER - LIVE DEMO TRADING")
    logger.info("=" * 60)
    if aggressive_mode:
        logger.warning("*** AGGRESSIVE MODE ENABLED ***")
    logger.info("*** PLACING REAL ORDERS ON DEMO ACCOUNT ***")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration_hours} hours")
    logger.info(f"Lower Thresholds: {lower_thresholds}")
    logger.info(f"Aggressive Mode: {aggressive_mode}")
    logger.info("=" * 60)

    # Connect to broker
    logger.info("Connecting to TradeLocker Demo...")
    broker = TradeLockerAdapter(
        environment=config.tradelocker.environment,
        email=config.tradelocker.email,
        password=config.tradelocker.password,
        server=config.tradelocker.server,
        acc_num=0,
        account_id=config.tradelocker.acc_num,
    )

    if not broker.connect():
        logger.error("Failed to connect to TradeLocker")
        return

    account = broker.get_account()
    logger.info(f"Connected: {account.account_name}")
    logger.info(f"Balance: ${float(account.balance):.2f}")
    logger.info(f"Equity: ${float(account.equity):.2f}")

    # Check existing positions
    positions = broker.list_positions()
    if positions:
        logger.info(f"Existing positions: {len(positions)}")
        for p in positions:
            logger.info(f"  {p.symbol} {p.side} {p.quantity} @ {p.avg_price}")

    # Initialize components
    analyzer = MarketAnalyzer()
    scorer = ConfluenceScorer()

    gate_config = GateConfig(
        model_path=config.ai.model_path,
        min_prob_win=config.ai.min_ai_probability,
        min_expected_r=config.ai.min_expected_value,
        min_confluence_score=config.ai.min_confluence_score,
        fallback_to_rules=True,
    )
    ai_gate = AIGate(gate_config)

    if ai_gate.model_loaded:
        logger.info(f"AI Gate loaded: {ai_gate._model_version}")
    else:
        logger.warning("AI Gate: Using rule-based fallback")

    # Create session with advanced trade management
    session = LiveDemoSession(
        broker=broker,
        analyzer=analyzer,
        scorer=scorer,
        ai_gate=ai_gate,
        config=config,
        lower_thresholds=lower_thresholds,
        aggressive_mode=aggressive_mode,
    )
    session.start_time = datetime.now()
    session.starting_balance = float(account.balance)

    end_time = datetime.now() + timedelta(hours=duration_hours)

    logger.info("")
    logger.info(f"Starting live demo trading... (Until {end_time.strftime('%H:%M:%S')})")
    logger.info("Press Ctrl+C to stop")
    logger.info("")

    try:
        while datetime.now() < end_time:
            await session.run_cycle()
            await asyncio.sleep(30)  # 30 second cycles

    except KeyboardInterrupt:
        logger.info("")
        logger.info("Stopping...")

    finally:
        # Final position check
        positions = broker.list_positions()
        if positions:
            logger.info(f"Open positions at end: {len(positions)}")
            for p in positions:
                logger.info(f"  {p.symbol} {p.side} P&L: ${float(p.unrealized_pnl):.2f}")

        broker.disconnect()

        # Summary
        summary = session.get_session_summary()

        logger.info("")
        logger.info("=" * 60)
        logger.info("LIVE DEMO SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration: {summary['session']['duration_minutes']:.1f} minutes")
        logger.info(f"Cycles: {summary['session']['cycles']}")
        logger.info("")
        logger.info("ACCOUNT:")
        logger.info(f"  Starting: ${summary['account']['starting_balance']:.2f}")
        logger.info(f"  Ending: ${summary['account']['ending_balance']:.2f}")
        logger.info(f"  P&L: ${summary['account']['pnl']:.2f}")
        logger.info("")
        logger.info("ACTIVITY:")
        logger.info(f"  Setups Found: {summary['activity']['setups_found']}")
        logger.info(f"  Approved: {summary['activity']['approved']}")
        logger.info(f"  Orders Placed: {summary['activity']['orders_placed']}")
        logger.info(f"  Scales Executed: {session.scales_executed}")
        logger.info(f"  Early Exits: {session.early_exits}")
        logger.info("=" * 60)

        session.save_results()


def main():
    parser = argparse.ArgumentParser(description="Run AG Analyzer on TradeLocker Demo")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration in hours")
    parser.add_argument("--lower-thresholds", action="store_true", help="Lower thresholds for more trades")
    parser.add_argument("--aggressive", action="store_true",
                       help="Aggressive mode: larger positions, more scaling, higher risk")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("WARNING: This will place REAL orders on your DEMO account!")
    if args.aggressive:
        print("** AGGRESSIVE MODE: Larger positions, more risk! **")
    print("=" * 60)
    response = input("Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        return

    asyncio.run(run_live_demo(
        duration_hours=args.duration,
        lower_thresholds=args.lower_thresholds,
        aggressive_mode=args.aggressive,
    ))


if __name__ == "__main__":
    main()
