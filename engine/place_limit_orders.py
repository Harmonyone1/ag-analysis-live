#!/usr/bin/env python
"""
Place limit orders for SMC setups.
"""

import sys
import os
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('../.env', override=True)

from src.config import load_config
from src.adapters import TradeLockerAdapter
from src.adapters.broker import OrderRequest


def main():
    config = load_config('../.env')
    broker = TradeLockerAdapter(
        environment=config.tradelocker.environment,
        email=config.tradelocker.email,
        password=config.tradelocker.password,
        server=config.tradelocker.server,
        acc_num=0,
        account_id=config.tradelocker.acc_num,
    )

    if not broker.connect():
        print("ERROR: Failed to connect")
        return

    try:
        # Get account status
        account = broker.get_account()
        print(f"Account Balance: ${float(account.balance):.2f}")
        print(f"Account Equity: ${float(account.equity):.2f}")

        # Define our limit orders based on SMC analysis
        # Using conservative lot sizes (0.1) given account size

        orders_to_place = [
            {
                "symbol": "USDCAD",
                "direction": "LONG",
                "entry": 1.3775,  # Middle of FVG zone 1.3770-1.3780
                "stop_loss": 1.3755,  # 20 pips below entry
                "take_profit": 1.3835,  # 60 pips (3:1 RR)
                "lots": 0.1,
            },
            {
                "symbol": "EURGBP",
                "direction": "SHORT",
                "entry": 0.8795,  # Middle of supply zone
                "stop_loss": 0.8825,  # 30 pips above
                "take_profit": 0.8705,  # 90 pips (3:1 RR)
                "lots": 0.1,
            },
            {
                "symbol": "ETHUSD",
                "direction": "SHORT",
                "entry": 2860,  # In the FVG zone
                "stop_loss": 2920,  # Above FVG
                "take_profit": 2680,  # 3:1 RR
                "lots": 0.05,  # Smaller size for crypto
            },
        ]

        print("\n" + "="*60)
        print("PLACING LIMIT ORDERS")
        print("="*60)

        for order_config in orders_to_place:
            symbol = order_config["symbol"]

            # Get instrument
            instrument = broker.get_instrument(symbol)
            if not instrument:
                print(f"\n{symbol}: Instrument not found, skipping")
                continue

            # Get current price
            candles = broker.get_candles(symbol, "1m", limit=1)
            if not candles:
                print(f"\n{symbol}: No price data, skipping")
                continue

            current_price = float(candles[-1].close)
            entry_price = order_config["entry"]
            direction = order_config["direction"]

            print(f"\n{symbol} {direction}:")
            print(f"  Current: {current_price}")
            print(f"  Entry: {entry_price}")
            print(f"  Stop: {order_config['stop_loss']}")
            print(f"  TP: {order_config['take_profit']}")
            print(f"  Lots: {order_config['lots']}")

            # Check if price is already past entry
            if direction == "LONG" and current_price < entry_price:
                # Price below entry - limit buy makes sense
                order_type = "limit"
                side = "buy"
            elif direction == "SHORT" and current_price > entry_price:
                # Price above entry - limit sell makes sense
                order_type = "limit"
                side = "sell"
            elif direction == "LONG" and current_price <= entry_price * 1.001:
                # Price at or just above entry zone
                print(f"  -> Price AT entry zone, placing limit buy")
                order_type = "limit"
                side = "buy"
            elif direction == "SHORT" and current_price >= entry_price * 0.999:
                # Price at or just below entry zone
                print(f"  -> Price AT entry zone, placing limit sell")
                order_type = "limit"
                side = "sell"
            else:
                print(f"  -> Price past entry ({current_price} vs {entry_price}), skipping")
                continue

            # Place the order
            try:
                order_request = OrderRequest(
                    instrument_id=instrument.instrument_id,
                    symbol=symbol,
                    side=side,
                    quantity=Decimal(str(order_config["lots"])),
                    order_type=order_type,
                    price=Decimal(str(entry_price)),
                    stop_loss=Decimal(str(order_config["stop_loss"])),
                    stop_loss_type="absolute",
                    take_profit=Decimal(str(order_config["take_profit"])),
                    take_profit_type="absolute",
                )

                result = broker.place_order(order_request)
                print(f"  [OK] ORDER PLACED - ID: {result.order_id}")

            except Exception as e:
                print(f"  [FAIL] Error placing order: {e}")

        # Show pending orders
        print("\n" + "="*60)
        print("PENDING ORDERS:")
        print("="*60)

        orders = broker.list_orders()
        if orders:
            for o in orders:
                print(f"  {o.symbol} {o.side} {o.quantity} @ {o.price} (ID: {o.order_id})")
        else:
            print("  No pending orders")

    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
