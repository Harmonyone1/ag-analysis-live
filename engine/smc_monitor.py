#!/usr/bin/env python
"""
SMC Setup Monitor - Periodic checking for trade setups

Monitors watchlist symbols and alerts when:
1. Price reaches entry zones (OB/FVG)
2. Timeframe alignment improves
3. Structure shifts occur
"""

import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('../.env', override=True)

from src.config import load_config
from src.adapters import TradeLockerAdapter

from src.smc import (
    MarketStructure,
    OrderBlockDetector,
    FVGDetector,
    LiquidityZoneDetector,
    MultiTimeframeAnalyzer,
)


# Watchlist with entry conditions
WATCHLIST = {
    "ETHUSD": {
        "bias": "short",
        "entry_zone": (2815, 2908),  # FVG / equilibrium zone
        "stop_above": 2950,
        "target": 2750,
    },
    "USDCAD": {
        "bias": "long",
        "entry_zone": (1.3770, 1.3780),  # FVG zone
        "stop_below": 1.3760,
        "target": 1.3850,
    },
    "EURGBP": {
        "bias": "short",
        "entry_zone": (0.8785, 0.8810),  # Potential supply
        "stop_above": 0.8830,
        "target": 0.8740,
    },
    "GBPUSD": {
        "bias": "neutral",
        "watch_for": "alignment",
    },
    "EURUSD": {
        "bias": "neutral",
        "watch_for": "alignment",
    },
}


def connect():
    """Connect to broker."""
    config = load_config('../.env')
    broker = TradeLockerAdapter(
        environment=config.tradelocker.environment,
        email=config.tradelocker.email,
        password=config.tradelocker.password,
        server=config.tradelocker.server,
        acc_num=0,
        account_id=config.tradelocker.acc_num,
    )
    if broker.connect():
        return broker
    return None


def candles_to_df(candles) -> pd.DataFrame:
    """Convert candles to DataFrame."""
    return pd.DataFrame({
        'open': [float(c.open) for c in candles],
        'high': [float(c.high) for c in candles],
        'low': [float(c.low) for c in candles],
        'close': [float(c.close) for c in candles],
    })


def check_symbol(broker, symbol: str, config: dict) -> dict:
    """Check a single symbol for setup conditions."""
    result = {
        "symbol": symbol,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "price": None,
        "alert": False,
        "alert_type": None,
        "message": "",
    }

    try:
        # Get current price from 15m candles
        ltf = broker.get_candles(symbol, "15m", limit=50)
        mtf = broker.get_candles(symbol, "1H", limit=50)
        htf = broker.get_candles(symbol, "4H", limit=50)

        if not ltf:
            result["message"] = "No data"
            return result

        ltf_df = candles_to_df(ltf)
        current_price = ltf_df['close'].iloc[-1]
        result["price"] = current_price

        # Check entry zone
        if "entry_zone" in config:
            zone_low, zone_high = config["entry_zone"]

            if config["bias"] == "long" and zone_low <= current_price <= zone_high:
                result["alert"] = True
                result["alert_type"] = "ENTRY_ZONE"
                result["message"] = f"LONG ENTRY ZONE! Price {current_price:.5f} in zone {zone_low}-{zone_high}"

            elif config["bias"] == "short" and zone_low <= current_price <= zone_high:
                result["alert"] = True
                result["alert_type"] = "ENTRY_ZONE"
                result["message"] = f"SHORT ENTRY ZONE! Price {current_price:.5f} in zone {zone_low}-{zone_high}"

            else:
                # Calculate distance to zone
                if config["bias"] == "long":
                    dist = current_price - zone_high
                    result["message"] = f"Price {current_price:.5f} | {abs(dist):.5f} above entry zone"
                elif config["bias"] == "short":
                    dist = zone_low - current_price
                    result["message"] = f"Price {current_price:.5f} | {abs(dist):.5f} below entry zone"

        # Check for alignment changes on neutral symbols
        if config.get("watch_for") == "alignment" and mtf and htf:
            mtf_df = candles_to_df(mtf)
            htf_df = candles_to_df(htf)

            analyzer = MultiTimeframeAnalyzer()
            analysis = analyzer.full_analysis(htf_df, mtf_df, ltf_df, "4H", "1H", "15m")

            alignment = analysis["summary"]["alignment"]
            strength = analysis["summary"]["alignment_strength"]

            if alignment != "neutral" and strength >= 0.67:
                result["alert"] = True
                result["alert_type"] = "ALIGNMENT"
                result["message"] = f"ALIGNMENT! {alignment.upper()} ({strength:.0%}) - Check for setup"
            else:
                result["message"] = f"Price {current_price:.5f} | Alignment: {alignment} ({strength:.0%})"

        # Check for structure shift
        if mtf:
            mtf_df = candles_to_df(mtf)
            structure = MarketStructure(swing_lookback=5)
            struct_result = structure.analyze(mtf_df)

            # Check for BOS/ChoCH
            shifts = struct_result.get("shifts", [])
            if shifts:
                latest_shift = shifts[-1]
                # If shift happened in last 3 candles
                if latest_shift.get("index", 0) > len(mtf_df) - 4:
                    result["alert"] = True
                    result["alert_type"] = "STRUCTURE_SHIFT"
                    result["message"] = f"STRUCTURE SHIFT! {latest_shift.get('type', 'BOS')} detected"

    except Exception as e:
        result["message"] = f"Error: {str(e)[:50]}"

    return result


def run_monitor(interval_seconds: int = 300, max_iterations: int = None):
    """
    Run the monitor loop.

    Args:
        interval_seconds: Check interval (default 5 minutes)
        max_iterations: Max checks before stopping (None = infinite)
    """
    print("=" * 60, flush=True)
    print("SMC SETUP MONITOR STARTED", flush=True)
    print(f"Interval: {interval_seconds} seconds", flush=True)
    print(f"Watchlist: {list(WATCHLIST.keys())}", flush=True)
    print("=" * 60, flush=True)

    iteration = 0

    while True:
        iteration += 1

        if max_iterations and iteration > max_iterations:
            print("\nMax iterations reached. Stopping monitor.")
            break

        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Check #{iteration}")
        print("-" * 50)

        broker = connect()
        if not broker:
            print("ERROR: Failed to connect. Retrying in 60 seconds...")
            time.sleep(60)
            continue

        alerts = []

        try:
            for symbol, config in WATCHLIST.items():
                result = check_symbol(broker, symbol, config)

                # Format output
                if result["alert"]:
                    print(f"🚨 {result['symbol']}: {result['message']}")
                    alerts.append(result)
                else:
                    print(f"   {result['symbol']}: {result['message']}")

            # Summary
            if alerts:
                print("\n" + "=" * 50)
                print(f"⚠️  {len(alerts)} ALERT(S) - REVIEW REQUIRED")
                print("=" * 50)
                for alert in alerts:
                    print(f"  {alert['symbol']}: {alert['alert_type']}")
                    print(f"    {alert['message']}")

        finally:
            broker.disconnect()

        # Wait for next check
        if max_iterations is None or iteration < max_iterations:
            print(f"\nNext check in {interval_seconds} seconds...")
            time.sleep(interval_seconds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SMC Setup Monitor")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--iterations", type=int, default=None, help="Max iterations (default: infinite)")
    args = parser.parse_args()

    run_monitor(interval_seconds=args.interval, max_iterations=args.iterations)
