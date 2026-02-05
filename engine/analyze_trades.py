#!/usr/bin/env python
"""Analyze trading history from TradeLocker account."""

import sys
import os
import json
from datetime import datetime
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('../.env', override=True)

from src.config import load_config
from src.adapters import TradeLockerAdapter


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


def get_trade_history(broker):
    """Get all historical trades."""
    # Access the underlying API directly for more data
    api = broker._api

    # Get order history
    orders_df = api.get_all_orders(history=True)

    # Get position history if available
    try:
        # Try to get closed positions/trades
        positions_history = api.get_all_positions()
    except:
        positions_history = None

    return orders_df, positions_history


def main():
    broker = connect()
    if not broker:
        print("ERROR: Failed to connect to TradeLocker")
        return

    try:
        api = broker._api

        # Get account info
        account = broker.get_account()
        print(f"\n=== ACCOUNT INFO ===")
        print(f"Account: {account.account_name}")
        print(f"Balance: ${float(account.balance):.2f}")
        print(f"Equity: ${float(account.equity):.2f}")

        # Get order history
        print(f"\n=== ORDER HISTORY ===")
        orders_df = api.get_all_orders(history=True)
        print(f"Total orders in history: {len(orders_df)}")

        if len(orders_df) > 0:
            print("\nColumns available:", orders_df.columns.tolist())
            print("\nFull order history:")
            print(orders_df.to_string())

        # Try to get executions/fills
        print(f"\n=== EXECUTION REPORTS ===")
        try:
            executions = api.get_execution_reports()
            print(f"Total executions: {len(executions) if executions is not None else 0}")
            if executions is not None and len(executions) > 0:
                print(executions.to_string())
        except Exception as e:
            print(f"Could not get executions: {e}")

        # Try trade history endpoint
        print(f"\n=== TRADE HISTORY ===")
        try:
            # Some brokers have this endpoint
            trades = api.get_trade_history() if hasattr(api, 'get_trade_history') else None
            if trades is not None:
                print(trades.to_string())
        except Exception as e:
            print(f"Trade history not available: {e}")

    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
