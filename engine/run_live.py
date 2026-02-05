#!/usr/bin/env python
"""Run AG Analyzer in live/paper trading mode.

Usage: python run_live.py

This script runs the trading engine outside of Docker for easier testing.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

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


async def run_trading_loop():
    """Main trading loop."""
    from src.config import load_config
    from src.adapters import TradeLockerAdapter
    from src.analysis import MarketAnalyzer
    from src.scoring import ConfluenceScorer
    from src.ai import AIGate, GateConfig, GateDecisionType

    # Load config
    config = load_config(env_path)

    logger.info("=" * 60)
    logger.info("AG ANALYZER - LIVE TRADING MODE")
    logger.info("=" * 60)
    logger.info(f"Mode: {config.bot.mode}")
    logger.info(f"Trading Enabled: {config.bot.trading_enabled}")
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
    logger.info(f"Equity: ${float(account.equity):.2f}")

    # Initialize components
    analyzer = MarketAnalyzer()
    scorer = ConfluenceScorer()

    # Load AI gate with trained model
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

    # Stats tracking
    cycle_count = 0
    setups_found = 0
    approved_count = 0
    rejected_count = 0

    logger.info("")
    logger.info("Starting trading loop... (Press Ctrl+C to stop)")
    logger.info("")

    try:
        while True:
            cycle_start = datetime.now()
            cycle_count += 1

            logger.info(f"--- Cycle {cycle_count} @ {cycle_start.strftime('%H:%M:%S')} ---")

            # Scan all symbols
            for symbol in SYMBOLS:
                try:
                    # Get candles
                    candles = broker.get_candles(symbol, "M15", limit=200)
                    if not candles or len(candles) < 50:
                        continue

                    # Extract OHLC
                    opens = np.array([float(c.open) for c in candles])
                    highs = np.array([float(c.high) for c in candles])
                    lows = np.array([float(c.low) for c in candles])
                    closes = np.array([float(c.close) for c in candles])
                    timestamps = [c.timestamp for c in candles]

                    # Analyze
                    market_view = analyzer.analyze(symbol, "M15", opens, highs, lows, closes, timestamps)

                    # Generate setups
                    setups = scorer.generate_setups(market_view)

                    for setup in setups:
                        if setup.confluence_score >= config.ai.min_confluence_score:
                            setups_found += 1

                            # Apply AI gate
                            decision = ai_gate.evaluate(setup, market_view)

                            if decision.decision == GateDecisionType.APPROVED:
                                approved_count += 1
                                logger.info("")
                                logger.info("=" * 50)
                                logger.info(f"TRADE SIGNAL APPROVED: {symbol} {setup.direction}")
                                logger.info("=" * 50)
                                logger.info(f"  Confluence Score: {setup.confluence_score}")
                                logger.info(f"  P(win): {decision.probability:.1%}")
                                logger.info(f"  E[R]: {decision.expected_r:.2f}")
                                logger.info(f"  Entry Zone: {setup.entry_zone}")
                                logger.info(f"  Stop: {setup.stop_price}")
                                logger.info(f"  TPs: {setup.tp_targets}")
                                logger.info(f"  Reasons: {decision.reasons[:3]}")
                                logger.info("=" * 50)

                                # In paper mode, just log - don't execute
                                if config.bot.mode == "paper":
                                    logger.info("(Paper mode - not executing)")
                                else:
                                    # Would execute trade here
                                    logger.info("(Live mode - would execute trade)")

                            elif decision.decision == GateDecisionType.NEEDS_REVIEW:
                                logger.info(f"{symbol} {setup.direction}: NEEDS_REVIEW - P(win)={decision.probability:.1%}, E[R]={decision.expected_r:.2f}")

                            else:
                                rejected_count += 1
                                logger.debug(f"{symbol} {setup.direction}: REJECTED - P(win)={decision.probability:.1%}")

                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    continue

            # Cycle summary
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"Cycle complete: {cycle_time:.1f}s | Setups: {setups_found} | Approved: {approved_count} | Rejected: {rejected_count}")

            # Wait for next cycle (15 seconds)
            await asyncio.sleep(15)

    except KeyboardInterrupt:
        logger.info("")
        logger.info("Stopping trading loop...")

    finally:
        broker.disconnect()
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRADING SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Cycles: {cycle_count}")
        logger.info(f"Setups Found: {setups_found}")
        logger.info(f"Approved: {approved_count}")
        logger.info(f"Rejected: {rejected_count}")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_trading_loop())
