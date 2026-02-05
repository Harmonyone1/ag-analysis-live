#!/usr/bin/env python
"""
SMC-Based Market Scanner

Scans forex pairs using Smart Money Concepts methodology:
- Multi-timeframe structure analysis
- Order block detection
- Fair value gap identification
- Liquidity zone mapping
- External data validation (COT, sentiment)
- News calendar filtering
"""

import sys
import os
import json
from datetime import datetime
from decimal import Decimal
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('../.env', override=True)

from src.config import load_config
from src.adapters import TradeLockerAdapter
from src.adapters.broker import OrderRequest

# Import SMC modules
from src.smc import (
    MarketStructure,
    OrderBlockDetector,
    FVGDetector,
    LiquidityZoneDetector,
    MultiTimeframeAnalyzer,
    SMCStrategy,
    format_setup_report,
)

# Import data modules
from src.data import (
    COTFetcher,
    SentimentFetcher,
    EconomicCalendar,
    NewsImpact,
)


# Pairs to scan
MAJOR_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"]
CROSS_PAIRS = ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "GBPAUD"]


def connect():
    """Connect to TradeLocker."""
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


def candles_to_dataframe(candles) -> pd.DataFrame:
    """Convert broker candles to pandas DataFrame."""
    data = {
        'open': [float(c.open) for c in candles],
        'high': [float(c.high) for c in candles],
        'low': [float(c.low) for c in candles],
        'close': [float(c.close) for c in candles],
    }
    return pd.DataFrame(data)


def get_mtf_data(broker, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Get multi-timeframe data for a symbol.

    Returns HTF (H4), MTF (H1), and LTF (M15) data.
    """
    try:
        # Get candle data for each timeframe (TradeLocker format)
        htf_candles = broker.get_candles(symbol, "4H", limit=100)
        mtf_candles = broker.get_candles(symbol, "1H", limit=100)
        ltf_candles = broker.get_candles(symbol, "15m", limit=100)

        if not all([htf_candles, mtf_candles, ltf_candles]):
            return None

        return {
            "htf": candles_to_dataframe(htf_candles),
            "mtf": candles_to_dataframe(mtf_candles),
            "ltf": candles_to_dataframe(ltf_candles),
        }
    except Exception as e:
        print(f"Error getting MTF data for {symbol}: {e}")
        return None


def analyze_with_smc(symbol: str, data: Dict[str, pd.DataFrame]) -> dict:
    """
    Perform full SMC analysis on a symbol.

    Returns analysis results including structure, order blocks, FVGs, and liquidity.
    """
    mtf_analyzer = MultiTimeframeAnalyzer()

    # Full analysis
    result = mtf_analyzer.full_analysis(
        data["htf"], data["mtf"], data["ltf"],
        htf_name="H4", mtf_name="H1", ltf_name="M15"
    )

    htf = result["htf"]
    mtf = result["mtf"]
    ltf = result["ltf"]

    # Format for output
    return {
        "symbol": symbol,
        "current_price": result["current_price"],
        "htf_bias": htf.bias,
        "mtf_bias": mtf.bias,
        "ltf_bias": ltf.bias,
        "alignment": result["summary"]["alignment"],
        "alignment_strength": result["summary"]["alignment_strength"],
        "price_zone": result["summary"]["price_zone"],
        "recommendation": result["summary"]["recommendation"],
        "order_blocks": {
            "htf_bullish": len([ob for ob in htf.order_blocks if ob.ob_type == "bullish" and not ob.is_mitigated]),
            "htf_bearish": len([ob for ob in htf.order_blocks if ob.ob_type == "bearish" and not ob.is_mitigated]),
            "mtf_bullish": len([ob for ob in mtf.order_blocks if ob.ob_type == "bullish" and not ob.is_mitigated]),
            "mtf_bearish": len([ob for ob in mtf.order_blocks if ob.ob_type == "bearish" and not ob.is_mitigated]),
        },
        "fvgs": {
            "htf_bullish": len([f for f in htf.fvgs if f.fvg_type == "bullish" and not f.is_filled]),
            "htf_bearish": len([f for f in htf.fvgs if f.fvg_type == "bearish" and not f.is_filled]),
            "mtf_bullish": len([f for f in mtf.fvgs if f.fvg_type == "bullish" and not f.is_filled]),
            "mtf_bearish": len([f for f in mtf.fvgs if f.fvg_type == "bearish" and not f.is_filled]),
        },
        "liquidity": {
            "buy_side": len([z for z in htf.liquidity_zones if z.zone_type == "buy_side" and not z.is_swept]),
            "sell_side": len([z for z in htf.liquidity_zones if z.zone_type == "sell_side" and not z.is_swept]),
        },
        "trade_setups": len(result["trade_setups"]),
    }


def get_cot_bias(symbol: str) -> Optional[str]:
    """Get COT-based bias for a symbol."""
    try:
        cot = COTFetcher()
        data = cot.fetch_cot_data(symbol)
        if data:
            return data.bias
    except Exception as e:
        print(f"Error fetching COT for {symbol}: {e}")
    return None


def get_sentiment_bias(symbol: str) -> Optional[str]:
    """Get retail sentiment contrarian bias."""
    try:
        sentiment = SentimentFetcher()
        data = sentiment.get_sentiment(symbol)
        if data:
            return data.bias  # Already contrarian
    except Exception as e:
        print(f"Error fetching sentiment for {symbol}: {e}")
    return None


def check_news_safety(symbol: str) -> dict:
    """Check if pair is safe from imminent news."""
    try:
        calendar = EconomicCalendar(buffer_minutes=30)
        return calendar.is_safe_to_trade(symbol)
    except Exception as e:
        print(f"Error checking news for {symbol}: {e}")
        return {"safe": True, "message": "Unable to check news"}


def full_scan(broker, symbols: List[str]) -> List[dict]:
    """
    Perform full SMC scan on multiple symbols.

    Returns list of analysis results sorted by setup quality.
    """
    results = []
    strategy = SMCStrategy(min_rr=2.0, min_confluence=3)

    for symbol in symbols:
        try:
            print(f"Scanning {symbol}...", end=" ", flush=True)

            # Get MTF data
            data = get_mtf_data(broker, symbol)
            if not data:
                print("No data")
                continue

            # SMC analysis
            analysis = analyze_with_smc(symbol, data)

            # Get external data
            cot_bias = get_cot_bias(symbol)
            sentiment_bias = get_sentiment_bias(symbol)
            news_check = check_news_safety(symbol)

            analysis["cot_bias"] = cot_bias
            analysis["sentiment_bias"] = sentiment_bias
            analysis["news_safe"] = news_check["safe"]
            analysis["news_message"] = news_check.get("message", "")

            # Generate setup if conditions met
            setup = strategy.analyze(
                symbol=symbol,
                htf_data=data["htf"],
                mtf_data=data["mtf"],
                ltf_data=data["ltf"],
                cot_bias=cot_bias,
                sentiment_bias=sentiment_bias,
                news_safe=news_check["safe"],
                news_warning=news_check.get("message")
            )

            if setup:
                analysis["has_setup"] = True
                analysis["setup"] = setup.to_dict()
            else:
                analysis["has_setup"] = False
                analysis["setup"] = None

            results.append(analysis)
            print(f"{analysis['alignment'].upper()}" if analysis['alignment'] != 'neutral' else "NEUTRAL")

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Sort by setup availability, then alignment strength
    results.sort(key=lambda x: (not x.get("has_setup", False), -x.get("alignment_strength", 0)))

    return results


def print_scan_results(results: List[dict]):
    """Print scan results in a formatted way."""
    print("\n" + "=" * 70)
    print("SMC MARKET SCAN RESULTS")
    print("=" * 70)

    for r in results:
        has_setup = "**SETUP**" if r.get("has_setup") else ""
        news_status = "NEWS_RISK" if not r.get("news_safe", True) else ""

        print(f"\n{r['symbol']} | {r['alignment'].upper()} ({r['alignment_strength']:.0%}) {has_setup} {news_status}")
        print(f"  HTF: {r['htf_bias']} | MTF: {r['mtf_bias']} | LTF: {r['ltf_bias']}")
        print(f"  Zone: {r['price_zone']} | Price: {r['current_price']:.5f}")
        print(f"  OBs: B{r['order_blocks']['mtf_bullish']}/S{r['order_blocks']['mtf_bearish']} | FVGs: B{r['fvgs']['mtf_bullish']}/S{r['fvgs']['mtf_bearish']}")

        if r.get("cot_bias"):
            print(f"  COT: {r['cot_bias']}")
        if r.get("sentiment_bias"):
            print(f"  Sentiment: {r['sentiment_bias']}")

        print(f"  -> {r['recommendation']}")

        if r.get("has_setup") and r.get("setup"):
            s = r["setup"]
            print(f"\n  === TRADE SETUP ===")
            print(f"  Quality: {s['quality']} | Confluence: {s['confluence_score']}/100")
            print(f"  Direction: {s['direction'].upper()}")
            print(f"  Entry: {s['entry_zone']}")
            print(f"  SL: {s['stop_loss']} | TP1: {s['tp1']} | TP2: {s['tp2']}")
            print(f"  R:R = {s['risk_reward']}")
            print(f"  Reason: {s['entry_reason']}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SMC Market Scanner")
    parser.add_argument("--action", choices=["scan", "analyze", "setups"], required=True)
    parser.add_argument("--symbol", type=str, help="Symbol for single analysis")
    parser.add_argument("--pairs", choices=["major", "cross", "all"], default="major")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    broker = connect()
    if not broker:
        print("ERROR: Failed to connect to TradeLocker")
        return

    try:
        if args.action == "scan":
            # Select pairs to scan
            if args.pairs == "major":
                symbols = MAJOR_PAIRS
            elif args.pairs == "cross":
                symbols = CROSS_PAIRS
            else:
                symbols = MAJOR_PAIRS + CROSS_PAIRS

            results = full_scan(broker, symbols)

            if args.json:
                # Clean up for JSON serialization
                for r in results:
                    if r.get("setup"):
                        # Already dict format
                        pass
                print(json.dumps(results, indent=2, default=str))
            else:
                print_scan_results(results)

        elif args.action == "analyze":
            if not args.symbol:
                print("ERROR: --symbol required for analyze")
                return

            data = get_mtf_data(broker, args.symbol)
            if not data:
                print(f"ERROR: No data for {args.symbol}")
                return

            analysis = analyze_with_smc(args.symbol, data)
            analysis["cot_bias"] = get_cot_bias(args.symbol)
            analysis["sentiment_bias"] = get_sentiment_bias(args.symbol)
            analysis["news"] = check_news_safety(args.symbol)

            print(json.dumps(analysis, indent=2, default=str))

        elif args.action == "setups":
            # Only show pairs with valid setups
            if args.pairs == "major":
                symbols = MAJOR_PAIRS
            elif args.pairs == "cross":
                symbols = CROSS_PAIRS
            else:
                symbols = MAJOR_PAIRS + CROSS_PAIRS

            results = full_scan(broker, symbols)
            setups_only = [r for r in results if r.get("has_setup")]

            if not setups_only:
                print("No valid SMC setups found at this time.")
                return

            print(f"\nFound {len(setups_only)} valid SMC setup(s):\n")

            for r in setups_only:
                s = r["setup"]
                print("=" * 60)
                print(f"{r['symbol']} {s['direction'].upper()} - Quality: {s['quality']}")
                print("=" * 60)
                print(f"Confluence Score: {s['confluence_score']}/100")
                print(f"Entry Zone: {s['entry_zone']}")
                print(f"Stop Loss: {s['stop_loss']}")
                print(f"TP1: {s['tp1']} | TP2: {s['tp2']} | TP3: {s['tp3']}")
                print(f"Risk:Reward = {s['risk_reward']}")
                print(f"\nConfluences:")
                for c in s.get("confluences", []):
                    print(f"  + {c}")
                print(f"\nReason: {s['entry_reason']}")
                print(f"HTF Bias: {s['htf_bias']}")
                print(f"News Safe: {'Yes' if r.get('news_safe', True) else 'No'}")
                print()

    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
