#!/usr/bin/env python
"""Check current positions on TradeLocker demo account."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('../.env')

from src.config import load_config
from src.adapters import TradeLockerAdapter

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
    account = broker.get_account()
    print(f"Account: {account.account_name}")
    print(f"Balance: ${float(account.balance):.2f}")
    print(f"Equity: ${float(account.equity):.2f}")
    print()

    positions = broker.list_positions()
    if positions:
        print(f"Open Positions ({len(positions)}):")
        for p in positions:
            print(f"  {p.symbol} {p.side} {p.quantity} @ {p.avg_price}")
            print(f"    Current: {p.current_price} | P&L: ${float(p.unrealized_pnl):.2f}")
            print(f"    SL: {p.stop_loss} | TP: {p.take_profit}")
            print()
    else:
        print("No open positions")

    # Check pending orders (SL/TP might be separate orders)
    orders = broker.list_orders()
    if orders:
        print(f"Pending Orders ({len(orders)}):")
        for o in orders:
            print(f"  {o.symbol} {o.side} {o.quantity} {o.order_type}")
            print(f"    Price: {o.price} | Status: {o.status}")
    else:
        print("No pending orders")

    broker.disconnect()
else:
    print("Failed to connect")
