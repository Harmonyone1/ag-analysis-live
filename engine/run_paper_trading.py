#!/usr/bin/env python
"""Run AG Analyzer in paper trading mode with full trade lifecycle management.

This script:
- Executes paper trades (simulated)
- Manages full trade lifecycle (entry, management, exit)
- Applies risk management (trailing stops, break-even)
- Tracks and reports performance metrics
- Evaluates AI prediction accuracy

Usage: python run_paper_trading.py [--duration HOURS] [--lower-thresholds]
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
RESULTS_FILE = Path(__file__).parent.parent / "paper_trading_results.json"


def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_price: float,
    pip_value: float = 10.0  # Standard lot pip value
) -> float:
    """Calculate position size based on risk."""
    risk_amount = account_balance * risk_percent
    stop_distance_pips = abs(entry_price - stop_price) * 10000  # For forex pairs

    if stop_distance_pips <= 0:
        return 0.01  # Minimum lot size

    # Risk amount / (stop distance * pip value) = lots
    lots = risk_amount / (stop_distance_pips * pip_value)

    # Clamp to reasonable range
    lots = max(0.01, min(1.0, lots))  # 0.01 to 1.0 lots
    return round(lots, 2)


class PaperTradingSession:
    """Manages a paper trading session with full lifecycle tracking."""

    def __init__(
        self,
        broker,
        analyzer,
        scorer,
        ai_gate,
        config,
        lower_thresholds: bool = False
    ):
        self.broker = broker
        self.analyzer = analyzer
        self.scorer = scorer
        self.ai_gate = ai_gate
        self.config = config
        self.lower_thresholds = lower_thresholds

        # Import trade manager
        from src.execution import TradeManager, TradeManagerConfig

        # Initialize trade manager with broker for price fetching
        trade_config = TradeManagerConfig(
            move_to_breakeven_r=1.0,
            trailing_start_r=1.5,
            trailing_distance_r=0.5,
            partial_exit_enabled=True,
            partial_exit_percent=0.5,
            max_trade_duration_hours=48,
            max_open_trades=5,
        )
        self.trade_manager = TradeManager(config=trade_config, broker=broker, paper_mode=True)

        # Session stats
        self.cycle_count = 0
        self.setups_found = 0
        self.approved_count = 0
        self.rejected_count = 0
        self.start_time = None

        # AI evaluation tracking
        self.ai_predictions: List[Dict] = []

        # Thresholds (possibly lowered for testing)
        if lower_thresholds:
            self.min_probability = 0.35  # Lower from 0.55 to get more trades
            self.min_expected_r = 0.10   # Lower from 0.15
            self.min_confluence = 50     # Lower from 60
            logger.warning("Using LOWERED thresholds for testing!")
            logger.warning(f"  min_probability: {self.min_probability}")
            logger.warning(f"  min_expected_r: {self.min_expected_r}")
            logger.warning(f"  min_confluence: {self.min_confluence}")
        else:
            self.min_probability = config.ai.min_ai_probability
            self.min_expected_r = config.ai.min_expected_value
            self.min_confluence = config.ai.min_confluence_score

    def should_approve_trade(self, decision) -> bool:
        """Check if trade meets approval criteria (with optional lower thresholds)."""
        from src.ai import GateDecisionType

        if self.lower_thresholds:
            # Use lowered thresholds for testing
            return (
                decision.probability >= self.min_probability and
                decision.expected_r >= self.min_expected_r
            )
        else:
            # Use standard approval
            return decision.decision == GateDecisionType.APPROVED

    async def run_cycle(self) -> Dict:
        """Run a single scanning/trading cycle."""
        from src.ai import GateDecisionType

        cycle_start = datetime.now()
        self.cycle_count += 1
        cycle_stats = {
            "setups": 0,
            "approved": 0,
            "rejected": 0,
            "trades_opened": 0,
            "trades_closed": 0,
        }

        logger.info(f"--- Cycle {self.cycle_count} @ {cycle_start.strftime('%H:%M:%S')} ---")

        # First, update existing positions
        closed_positions = self.trade_manager.update_positions()
        cycle_stats["trades_closed"] = len(closed_positions)

        for pos in closed_positions:
            # Get P&L info from the trade result in history
            history = self.trade_manager.get_trade_history()
            trade_result = next((r for r in history if r.trade_id == pos.id), None)
            pnl_r = trade_result.pnl_r if trade_result else 0.0
            pnl_dollars = float(pos.realized_pnl)

            logger.info(f"Position closed: {pos.symbol} {pos.direction}")
            logger.info(f"  Exit Reason: {pos.exit_reason}")
            logger.info(f"  P&L: {pnl_r:.2f}R (${pnl_dollars:.2f})")

            # Record for AI evaluation
            self.ai_predictions.append({
                "symbol": pos.symbol,
                "direction": pos.direction,
                "ai_probability": pos.ai_probability,
                "ai_expected_r": pos.ai_expected_r,
                "actual_r": pnl_r,
                "won": pnl_r > 0,
                "exit_reason": pos.exit_reason.value if pos.exit_reason else "unknown",
                "timestamp": datetime.now().isoformat(),
            })

        # Log open positions
        open_positions = self.trade_manager.get_open_positions()
        if open_positions:
            logger.info(f"Open positions: {len(open_positions)}")
            for pos in open_positions:
                current_r = self.trade_manager._calculate_current_r(pos, pos.current_price)
                logger.info(f"  {pos.symbol} {pos.direction}: {current_r:.2f}R (SL: {pos.stop_loss:.5f})")

        # Check if we can open more trades
        can_open_more = len(open_positions) < self.trade_manager.config.max_open_trades

        if not can_open_more:
            logger.info("Max positions reached, skipping scan")
        else:
            # Scan all symbols for new setups
            for symbol in SYMBOLS:
                # Skip if we already have a position in this symbol
                if any(p.symbol == symbol for p in open_positions):
                    continue

                try:
                    # Get candles
                    candles = self.broker.get_candles(symbol, "M15", limit=200)
                    if not candles or len(candles) < 50:
                        continue

                    # Extract OHLC
                    opens = np.array([float(c.open) for c in candles])
                    highs = np.array([float(c.high) for c in candles])
                    lows = np.array([float(c.low) for c in candles])
                    closes = np.array([float(c.close) for c in candles])
                    timestamps = [c.timestamp for c in candles]
                    current_price = closes[-1]

                    # Analyze
                    market_view = self.analyzer.analyze(symbol, "M15", opens, highs, lows, closes, timestamps)

                    # Generate setups
                    setups = self.scorer.generate_setups(market_view)

                    for setup in setups:
                        if setup.confluence_score >= self.min_confluence:
                            self.setups_found += 1
                            cycle_stats["setups"] += 1

                            # Apply AI gate
                            decision = self.ai_gate.evaluate(setup, market_view)

                            # Check approval (with optional lower thresholds)
                            if self.should_approve_trade(decision):
                                self.approved_count += 1
                                cycle_stats["approved"] += 1

                                logger.info("")
                                logger.info("=" * 50)
                                logger.info(f"TRADE APPROVED: {symbol} {setup.direction}")
                                logger.info("=" * 50)
                                logger.info(f"  Confluence: {setup.confluence_score}")
                                logger.info(f"  P(win): {decision.probability:.1%}")
                                logger.info(f"  E[R]: {decision.expected_r:.2f}")
                                logger.info(f"  Entry Zone: {setup.entry_zone}")
                                logger.info(f"  Stop: {setup.stop_price}")
                                logger.info(f"  TPs: {setup.tp_targets}")

                                # Calculate position size
                                account = self.broker.get_account()
                                entry_price = float(setup.entry_zone[0]) if setup.entry_zone else current_price
                                stop_price = float(setup.stop_price)

                                quantity = calculate_position_size(
                                    account_balance=float(account.balance),
                                    risk_percent=0.01,  # 1% risk
                                    entry_price=entry_price,
                                    stop_price=stop_price,
                                )

                                # Open paper trade (convert quantity to Decimal)
                                position = self.trade_manager.open_trade(
                                    setup=setup,
                                    quantity=Decimal(str(quantity)),
                                    ai_probability=decision.probability,
                                    ai_expected_r=decision.expected_r,
                                    candidate_id=None,
                                )

                                if position:
                                    cycle_stats["trades_opened"] += 1
                                    logger.info(f"  Position opened: {quantity} lots @ {position.entry_price:.5f}")
                                    logger.info("=" * 50)
                                else:
                                    logger.warning("  Failed to open position")

                            elif decision.decision == GateDecisionType.NEEDS_REVIEW:
                                logger.info(f"{symbol} {setup.direction}: NEEDS_REVIEW - P(win)={decision.probability:.1%}, E[R]={decision.expected_r:.2f}")
                            else:
                                self.rejected_count += 1
                                cycle_stats["rejected"] += 1
                                logger.debug(f"{symbol} {setup.direction}: REJECTED")

                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    continue

        # Cycle summary
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        logger.info(f"Cycle complete: {cycle_time:.1f}s | Setups: {cycle_stats['setups']} | Opened: {cycle_stats['trades_opened']} | Closed: {cycle_stats['trades_closed']}")

        return cycle_stats

    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary."""
        stats = self.trade_manager.get_performance_stats()

        # Calculate AI accuracy
        if self.ai_predictions:
            high_prob_trades = [p for p in self.ai_predictions if p["ai_probability"] >= 0.55]
            if high_prob_trades:
                high_prob_accuracy = sum(1 for p in high_prob_trades if p["won"]) / len(high_prob_trades)
            else:
                high_prob_accuracy = None

            all_trades_accuracy = sum(1 for p in self.ai_predictions if p["won"]) / len(self.ai_predictions)
        else:
            high_prob_accuracy = None
            all_trades_accuracy = None

        return {
            "session": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": datetime.now().isoformat(),
                "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60 if self.start_time else 0,
                "cycles": self.cycle_count,
                "lower_thresholds": self.lower_thresholds,
            },
            "setups": {
                "total_found": self.setups_found,
                "approved": self.approved_count,
                "rejected": self.rejected_count,
            },
            "trades": stats,
            "ai_evaluation": {
                "total_predictions": len(self.ai_predictions),
                "all_trades_accuracy": all_trades_accuracy,
                "high_probability_accuracy": high_prob_accuracy,
                "predictions": self.ai_predictions,
            },
        }

    def save_results(self):
        """Save session results to file."""
        summary = self.get_session_summary()

        # Load existing results or create new
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


async def run_paper_trading(duration_hours: float = 2.0, lower_thresholds: bool = False):
    """Run paper trading session."""
    from src.config import load_config
    from src.adapters import TradeLockerAdapter
    from src.analysis import MarketAnalyzer
    from src.scoring import ConfluenceScorer
    from src.ai import AIGate, GateConfig

    # Load config
    config = load_config(env_path)

    logger.info("=" * 60)
    logger.info("AG ANALYZER - PAPER TRADING SESSION")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration_hours} hours")
    logger.info(f"Lower Thresholds: {lower_thresholds}")
    logger.info(f"Model Path: {config.ai.model_path}")
    logger.info("=" * 60)

    # Connect to broker
    logger.info("Connecting to TradeLocker...")
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

    # Initialize components
    analyzer = MarketAnalyzer()
    scorer = ConfluenceScorer()

    # Load AI gate
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
        logger.warning("AI Gate: No model loaded, using rule-based fallback")

    # Create trading session
    session = PaperTradingSession(
        broker=broker,
        analyzer=analyzer,
        scorer=scorer,
        ai_gate=ai_gate,
        config=config,
        lower_thresholds=lower_thresholds,
    )
    session.start_time = datetime.now()

    end_time = datetime.now() + timedelta(hours=duration_hours)

    logger.info("")
    logger.info(f"Starting paper trading... (Will run until {end_time.strftime('%H:%M:%S')})")
    logger.info("Press Ctrl+C to stop early")
    logger.info("")

    try:
        while datetime.now() < end_time:
            await session.run_cycle()

            # Wait for next cycle (30 seconds for paper trading)
            await asyncio.sleep(30)

    except KeyboardInterrupt:
        logger.info("")
        logger.info("Stopping paper trading...")

    finally:
        # Close any remaining positions and record them
        open_positions = session.trade_manager.get_open_positions()
        closed = session.trade_manager.close_all_positions()
        if closed:
            logger.info(f"Closed {closed} remaining positions")

            # Record the closed positions for AI evaluation
            for pos in open_positions:
                history = session.trade_manager.get_trade_history()
                trade_result = next((r for r in history if r.trade_id == pos.id), None)
                if trade_result:
                    session.ai_predictions.append({
                        "symbol": pos.symbol,
                        "direction": pos.direction,
                        "ai_probability": pos.ai_probability,
                        "ai_expected_r": pos.ai_expected_r,
                        "actual_r": trade_result.pnl_r,
                        "won": trade_result.pnl_r > 0,
                        "exit_reason": "session_end",
                        "timestamp": datetime.now().isoformat(),
                    })

        broker.disconnect()

        # Print summary
        summary = session.get_session_summary()

        logger.info("")
        logger.info("=" * 60)
        logger.info("PAPER TRADING SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration: {summary['session']['duration_minutes']:.1f} minutes")
        logger.info(f"Cycles: {summary['session']['cycles']}")
        logger.info("")
        logger.info("SETUPS:")
        logger.info(f"  Total Found: {summary['setups']['total_found']}")
        logger.info(f"  Approved: {summary['setups']['approved']}")
        logger.info(f"  Rejected: {summary['setups']['rejected']}")
        logger.info("")
        logger.info("TRADES:")
        logger.info(f"  Total: {summary['trades']['total_trades']}")
        logger.info(f"  Winners: {summary['trades']['winners']}")
        logger.info(f"  Losers: {summary['trades']['losers']}")
        logger.info(f"  Win Rate: {summary['trades']['win_rate']:.1%}" if summary['trades']['win_rate'] else "  Win Rate: N/A")
        logger.info(f"  Total R: {summary['trades']['total_r']:.2f}")
        logger.info(f"  Avg R per Trade: {summary['trades']['avg_r_per_trade']:.2f}" if summary['trades']['avg_r_per_trade'] else "  Avg R per Trade: N/A")
        logger.info("")
        logger.info("AI EVALUATION:")
        logger.info(f"  Predictions Made: {summary['ai_evaluation']['total_predictions']}")
        if summary['ai_evaluation']['all_trades_accuracy'] is not None:
            logger.info(f"  Overall Accuracy: {summary['ai_evaluation']['all_trades_accuracy']:.1%}")
        if summary['ai_evaluation']['high_probability_accuracy'] is not None:
            logger.info(f"  High-Prob (>55%) Accuracy: {summary['ai_evaluation']['high_probability_accuracy']:.1%}")
        logger.info("=" * 60)

        # Save results
        session.save_results()


def main():
    parser = argparse.ArgumentParser(description="Run AG Analyzer paper trading")
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration in hours (default: 2.0)"
    )
    parser.add_argument(
        "--lower-thresholds",
        action="store_true",
        help="Lower AI thresholds to get more trades for testing"
    )
    args = parser.parse_args()

    asyncio.run(run_paper_trading(
        duration_hours=args.duration,
        lower_thresholds=args.lower_thresholds,
    ))


if __name__ == "__main__":
    main()
