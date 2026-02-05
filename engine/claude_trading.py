#!/usr/bin/env python
"""Claude's Direct Trading Session - I analyze and trade using my own judgment."""

import sys
import os
import json
from datetime import datetime
from decimal import Decimal
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('../.env', override=True)

from src.config import load_config
from src.adapters import TradeLockerAdapter
from src.adapters.broker import OrderRequest

# All 28 forex pairs
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF",
    "GBPJPY", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
    "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF",
    "NZDCAD", "NZDCHF",
    "CADJPY", "CHFJPY", "CADCHF",
]


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


def get_account_status(broker):
    """Get account and position status."""
    account = broker.get_account()
    positions = broker.list_positions()

    result = {
        "account": {
            "name": account.account_name,
            "balance": float(account.balance),
            "equity": float(account.equity),
        },
        "positions": []
    }

    for p in positions:
        result["positions"].append({
            "symbol": p.symbol,
            "side": p.side,
            "quantity": float(p.quantity),
            "entry": float(p.avg_price),
            "current": float(p.current_price),
            "pnl": float(p.unrealized_pnl),
            "stop_loss": float(p.stop_loss) if p.stop_loss else None,
            "take_profit": float(p.take_profit) if p.take_profit else None,
            "position_id": p.position_id,
        })

    return result


def get_market_data(broker, symbol, timeframe="M15", limit=100):
    """Get candle data for analysis."""
    candles = broker.get_candles(symbol, timeframe, limit=limit)
    if not candles:
        return None

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": len(candles),
        "opens": [float(c.open) for c in candles],
        "highs": [float(c.high) for c in candles],
        "lows": [float(c.low) for c in candles],
        "closes": [float(c.close) for c in candles],
        "current_price": float(candles[-1].close),
        "last_high": float(candles[-1].high),
        "last_low": float(candles[-1].low),
    }


def analyze_pair(data):
    """Analyze a single pair and return key metrics."""
    closes = np.array(data["closes"])
    highs = np.array(data["highs"])
    lows = np.array(data["lows"])

    # Calculate indicators
    # SMA 20 and 50
    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-50:])

    # RSI 14
    deltas = np.diff(closes[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # ATR 14
    tr_list = []
    for i in range(-14, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_list.append(tr)
    atr = np.mean(tr_list)

    # Price relative to recent range
    recent_high = max(highs[-20:])
    recent_low = min(lows[-20:])
    price_position = (closes[-1] - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5

    # Trend direction
    trend = "BULLISH" if sma20 > sma50 else "BEARISH"

    # Momentum (price vs SMA20)
    momentum = (closes[-1] - sma20) / sma20 * 100

    return {
        "symbol": data["symbol"],
        "current_price": data["current_price"],
        "sma20": round(sma20, 5),
        "sma50": round(sma50, 5),
        "rsi": round(rsi, 1),
        "atr": round(atr, 5),
        "trend": trend,
        "momentum": round(momentum, 2),
        "price_position": round(price_position, 2),
        "recent_high": round(recent_high, 5),
        "recent_low": round(recent_low, 5),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["status", "scan", "analyze", "trade", "close"], required=True)
    parser.add_argument("--symbol", type=str, help="Symbol for analyze/trade")
    parser.add_argument("--direction", choices=["LONG", "SHORT"], help="Trade direction")
    parser.add_argument("--lots", type=float, default=0.1, help="Position size")
    parser.add_argument("--stop-pips", type=float, default=20, help="Stop loss in pips")
    parser.add_argument("--tp-pips", type=float, default=30, help="Take profit in pips")
    parser.add_argument("--position-id", type=int, help="Position ID for close")
    args = parser.parse_args()

    broker = connect()
    if not broker:
        print("ERROR: Failed to connect to TradeLocker")
        return

    try:
        if args.action == "status":
            status = get_account_status(broker)
            print(json.dumps(status, indent=2))

        elif args.action == "scan":
            # Scan all pairs and return analysis
            results = []
            for symbol in SYMBOLS[:10]:  # First 10 for speed
                try:
                    data = get_market_data(broker, symbol)
                    if data:
                        analysis = analyze_pair(data)
                        results.append(analysis)
                except Exception as e:
                    print(f"Error scanning {symbol}: {e}")
            print(json.dumps(results, indent=2))

        elif args.action == "analyze":
            if not args.symbol:
                print("ERROR: --symbol required")
                return
            data = get_market_data(broker, args.symbol, limit=100)
            if data:
                analysis = analyze_pair(data)
                # Add raw price data for context
                analysis["last_5_closes"] = data["closes"][-5:]
                analysis["last_5_highs"] = data["highs"][-5:]
                analysis["last_5_lows"] = data["lows"][-5:]
                print(json.dumps(analysis, indent=2))
            else:
                print(f"ERROR: No data for {args.symbol}")

        elif args.action == "trade":
            if not args.symbol or not args.direction:
                print("ERROR: --symbol and --direction required")
                return

            # Get current price
            data = get_market_data(broker, args.symbol, limit=10)
            if not data:
                print(f"ERROR: No data for {args.symbol}")
                return

            current_price = data["current_price"]
            # Calculate pip size based on instrument type
            if args.symbol in ["ETHUSD", "BTCUSD", "XRPUSD", "XAUUSD", "US30", "NAS100", "SPX500"]:
                pip_size = 1.0  # Crypto: $1 per pip
            elif "JPY" in args.symbol:
                pip_size = 0.01  # JPY pairs
            else:
                pip_size = 0.0001  # Standard forex

            # Calculate SL and TP
            if args.direction == "LONG":
                stop_price = current_price - (args.stop_pips * pip_size)
                tp_price = current_price + (args.tp_pips * pip_size)
                side = "buy"
            else:
                stop_price = current_price + (args.stop_pips * pip_size)
                tp_price = current_price - (args.tp_pips * pip_size)
                side = "sell"

            # Get instrument
            instrument = broker.get_instrument(args.symbol)
            if not instrument:
                print(f"ERROR: Instrument not found: {args.symbol}")
                return

            # Place order
            order_request = OrderRequest(
                instrument_id=instrument.instrument_id,
                symbol=args.symbol,
                side=side,
                quantity=Decimal(str(args.lots)),
                order_type="market",
                stop_loss=Decimal(str(round(stop_price, 5))),
                stop_loss_type="absolute",
                take_profit=Decimal(str(round(tp_price, 5))),
                take_profit_type="absolute",
            )

            order = broker.place_order(order_request)
            print(json.dumps({
                "status": "ORDER_PLACED",
                "order_id": order.order_id,
                "symbol": args.symbol,
                "direction": args.direction,
                "lots": args.lots,
                "entry_price": current_price,
                "stop_loss": round(stop_price, 5),
                "take_profit": round(tp_price, 5),
            }, indent=2))

        elif args.action == "close":
            if not args.position_id:
                print("ERROR: --position-id required")
                return

            try:
                broker.close_position(args.position_id)
                print(json.dumps({"status": "POSITION_CLOSED", "position_id": args.position_id}))
            except Exception as e:
                print(f"ERROR: Failed to close position: {e}")

    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
