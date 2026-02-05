#!/usr/bin/env python
"""Deep analysis of trading history to identify patterns and issues."""

import sys
import os
import json
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('../.env', override=True)

from src.config import load_config
from src.adapters import TradeLockerAdapter


# Instrument ID to Symbol mapping (based on observed data)
INSTRUMENT_MAP = {
    3457: "USDCAD",
    3492: "EURAUD",
    3500: "USDJPY",
    3505: "EURGBP",
    3451: "EURJPY",
    3389: "XAUUSD",  # Gold
    3512: "USDCHF",
    3888: "US100",   # Nasdaq
    3884: "SP500",   # S&P 500
    3470: "EURUSD",
    3466: "GBPUSD",
    3493: "AUDUSD",
    11337: "NQ100",  # Nasdaq futures
    3366: "BTCUSD",  # Bitcoin
}


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


def analyze_trades(orders_df):
    """Analyze trading patterns from order history."""

    # Filter to only filled orders (actual trades)
    filled = orders_df[orders_df['status'] == 'Filled'].copy()

    # Convert timestamps
    filled['timestamp'] = pd.to_datetime(filled['createdDate'], unit='ms')
    filled['date'] = filled['timestamp'].dt.date
    filled['hour'] = filled['timestamp'].dt.hour
    filled['day_of_week'] = filled['timestamp'].dt.dayofweek

    # Map instrument IDs to symbols
    filled['symbol'] = filled['tradableInstrumentId'].map(INSTRUMENT_MAP)
    filled['symbol'] = filled['symbol'].fillna(filled['tradableInstrumentId'].astype(str))

    # Calculate trade pairs (entry + exit)
    # Group by positionId to match entries with exits
    trades = []
    position_ids = filled['positionId'].unique()

    for pos_id in position_ids:
        pos_orders = filled[filled['positionId'] == pos_id].sort_values('timestamp')

        if len(pos_orders) < 2:
            continue

        # First order is entry, last order(s) are exit
        entry = pos_orders.iloc[0]
        exits = pos_orders.iloc[1:]

        # Calculate P&L
        entry_price = float(entry['avgPrice'])
        entry_qty = float(entry['qty'])
        entry_side = entry['side']
        symbol = entry['symbol']

        total_exit_value = 0
        total_exit_qty = 0

        for _, exit_order in exits.iterrows():
            exit_price = float(exit_order['avgPrice'])
            exit_qty = float(exit_order['filledQty'])
            total_exit_value += exit_price * exit_qty
            total_exit_qty += exit_qty

        if total_exit_qty == 0:
            continue

        avg_exit_price = total_exit_value / total_exit_qty

        # Calculate P&L based on direction
        if entry_side == 'buy':
            pnl_pips = avg_exit_price - entry_price
        else:
            pnl_pips = entry_price - avg_exit_price

        # Estimate dollar P&L (rough calculation)
        # For forex, 1 pip = ~$10 per lot for most pairs
        # For indices and gold, different calculations
        if 'USD' in str(symbol) and len(str(symbol)) == 6:
            pip_value = 10 * entry_qty  # Forex
            if 'JPY' in str(symbol):
                pnl_dollars = pnl_pips * 100 * entry_qty * 0.65  # JPY pairs
            else:
                pnl_dollars = pnl_pips * 10000 * entry_qty * 0.10  # Standard forex
        elif symbol in ['XAUUSD', 3389]:
            pnl_dollars = pnl_pips * entry_qty * 100  # Gold
        elif symbol in ['US100', 'NQ100', 3888, 11337]:
            pnl_dollars = pnl_pips * entry_qty * 20  # Nasdaq
        elif symbol in ['SP500', 3884]:
            pnl_dollars = pnl_pips * entry_qty * 50  # S&P
        elif symbol in ['BTCUSD', 3366]:
            pnl_dollars = pnl_pips * entry_qty  # Bitcoin
        else:
            pnl_dollars = pnl_pips * entry_qty * 10  # Default

        # Holding time
        entry_time = entry['timestamp']
        exit_time = exits.iloc[-1]['timestamp']
        holding_time = (exit_time - entry_time).total_seconds() / 60  # minutes

        trades.append({
            'position_id': pos_id,
            'symbol': symbol,
            'side': entry_side,
            'entry_price': entry_price,
            'exit_price': avg_exit_price,
            'quantity': entry_qty,
            'pnl_pips': pnl_pips,
            'pnl_dollars': pnl_dollars,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'holding_minutes': holding_time,
            'hour': entry['hour'],
            'day_of_week': entry['day_of_week'],
            'had_stop_loss': entry['stopLoss'] > 0 if pd.notna(entry['stopLoss']) else False,
            'had_take_profit': entry['takeProfit'] > 0 if pd.notna(entry['takeProfit']) else False,
        })

    return pd.DataFrame(trades)


def generate_report(trades_df, orders_df):
    """Generate comprehensive trading analysis report."""

    print("\n" + "="*70)
    print("           TRADING ANALYSIS REPORT")
    print("="*70)

    if len(trades_df) == 0:
        print("\nNo completed trades found to analyze.")
        return

    # Overall Statistics
    print("\n" + "-"*70)
    print("OVERALL STATISTICS")
    print("-"*70)

    total_trades = len(trades_df)
    winners = trades_df[trades_df['pnl_dollars'] > 0]
    losers = trades_df[trades_df['pnl_dollars'] < 0]
    breakeven = trades_df[trades_df['pnl_dollars'] == 0]

    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

    total_profit = winners['pnl_dollars'].sum() if len(winners) > 0 else 0
    total_loss = abs(losers['pnl_dollars'].sum()) if len(losers) > 0 else 0
    net_pnl = total_profit - total_loss

    avg_win = winners['pnl_dollars'].mean() if len(winners) > 0 else 0
    avg_loss = abs(losers['pnl_dollars'].mean()) if len(losers) > 0 else 0

    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    print(f"Total Trades: {total_trades}")
    print(f"Winners: {len(winners)} ({win_rate:.1f}%)")
    print(f"Losers: {len(losers)} ({100-win_rate:.1f}%)")
    print(f"Breakeven: {len(breakeven)}")
    print(f"\nTotal Profit: ${total_profit:,.2f}")
    print(f"Total Loss: ${total_loss:,.2f}")
    print(f"Net P&L: ${net_pnl:,.2f}")
    print(f"\nAverage Win: ${avg_win:,.2f}")
    print(f"Average Loss: ${avg_loss:,.2f}")
    print(f"Risk/Reward: {avg_win/avg_loss:.2f}:1" if avg_loss > 0 else "N/A")
    print(f"Profit Factor: {profit_factor:.2f}")

    # By Instrument
    print("\n" + "-"*70)
    print("PERFORMANCE BY INSTRUMENT")
    print("-"*70)

    by_symbol = trades_df.groupby('symbol').agg({
        'pnl_dollars': ['count', 'sum', 'mean'],
        'position_id': lambda x: (trades_df.loc[x.index, 'pnl_dollars'] > 0).sum()
    }).round(2)
    by_symbol.columns = ['Trades', 'Total P&L', 'Avg P&L', 'Wins']
    by_symbol['Win %'] = (by_symbol['Wins'] / by_symbol['Trades'] * 100).round(1)
    by_symbol = by_symbol.sort_values('Total P&L', ascending=False)

    print(by_symbol.to_string())

    # Worst Performing
    print("\n" + "-"*70)
    print("BIGGEST LOSERS (Individual Trades)")
    print("-"*70)

    worst_trades = trades_df.nsmallest(10, 'pnl_dollars')[['symbol', 'side', 'entry_price', 'exit_price', 'quantity', 'pnl_dollars', 'holding_minutes']]
    print(worst_trades.to_string())

    # By Direction
    print("\n" + "-"*70)
    print("PERFORMANCE BY DIRECTION")
    print("-"*70)

    by_side = trades_df.groupby('side').agg({
        'pnl_dollars': ['count', 'sum', 'mean'],
        'position_id': lambda x: (trades_df.loc[x.index, 'pnl_dollars'] > 0).sum()
    }).round(2)
    by_side.columns = ['Trades', 'Total P&L', 'Avg P&L', 'Wins']
    by_side['Win %'] = (by_side['Wins'] / by_side['Trades'] * 100).round(1)
    print(by_side.to_string())

    # Holding Time Analysis
    print("\n" + "-"*70)
    print("HOLDING TIME ANALYSIS")
    print("-"*70)

    avg_hold_winners = winners['holding_minutes'].mean() if len(winners) > 0 else 0
    avg_hold_losers = losers['holding_minutes'].mean() if len(losers) > 0 else 0

    print(f"Average holding time (winners): {avg_hold_winners:.1f} minutes")
    print(f"Average holding time (losers): {avg_hold_losers:.1f} minutes")

    # Quick trades (< 5 min)
    quick_trades = trades_df[trades_df['holding_minutes'] < 5]
    if len(quick_trades) > 0:
        quick_win_rate = (quick_trades['pnl_dollars'] > 0).sum() / len(quick_trades) * 100
        print(f"\nQuick trades (<5 min): {len(quick_trades)} trades, {quick_win_rate:.1f}% win rate")

    # Risk Management
    print("\n" + "-"*70)
    print("RISK MANAGEMENT ANALYSIS")
    print("-"*70)

    with_sl = trades_df[trades_df['had_stop_loss'] == True]
    without_sl = trades_df[trades_df['had_stop_loss'] == False]

    print(f"Trades WITH stop loss: {len(with_sl)} ({len(with_sl)/len(trades_df)*100:.1f}%)")
    print(f"Trades WITHOUT stop loss: {len(without_sl)} ({len(without_sl)/len(trades_df)*100:.1f}%)")

    if len(with_sl) > 0:
        sl_win_rate = (with_sl['pnl_dollars'] > 0).sum() / len(with_sl) * 100
        print(f"Win rate with SL: {sl_win_rate:.1f}%")

    if len(without_sl) > 0:
        no_sl_win_rate = (without_sl['pnl_dollars'] > 0).sum() / len(without_sl) * 100
        print(f"Win rate without SL: {no_sl_win_rate:.1f}%")

    # Time of Day
    print("\n" + "-"*70)
    print("PERFORMANCE BY HOUR (UTC)")
    print("-"*70)

    by_hour = trades_df.groupby('hour').agg({
        'pnl_dollars': ['count', 'sum']
    }).round(2)
    by_hour.columns = ['Trades', 'P&L']
    by_hour = by_hour[by_hour['Trades'] >= 5]  # Only show hours with 5+ trades
    by_hour = by_hour.sort_values('P&L', ascending=False)
    print(by_hour.head(10).to_string())

    # Position Sizing
    print("\n" + "-"*70)
    print("POSITION SIZING ANALYSIS")
    print("-"*70)

    print(f"Average position size: {trades_df['quantity'].mean():.2f} lots")
    print(f"Max position size: {trades_df['quantity'].max():.2f} lots")
    print(f"Min position size: {trades_df['quantity'].min():.2f} lots")

    # Large positions performance
    median_size = trades_df['quantity'].median()
    large = trades_df[trades_df['quantity'] > median_size]
    small = trades_df[trades_df['quantity'] <= median_size]

    if len(large) > 0:
        large_win_rate = (large['pnl_dollars'] > 0).sum() / len(large) * 100
        print(f"\nLarge positions (>{median_size:.2f} lots): {len(large)} trades, {large_win_rate:.1f}% win rate, ${large['pnl_dollars'].sum():,.2f} P&L")

    if len(small) > 0:
        small_win_rate = (small['pnl_dollars'] > 0).sum() / len(small) * 100
        print(f"Small positions (<={median_size:.2f} lots): {len(small)} trades, {small_win_rate:.1f}% win rate, ${small['pnl_dollars'].sum():,.2f} P&L")

    # Key Issues Identified
    print("\n" + "="*70)
    print("KEY ISSUES IDENTIFIED")
    print("="*70)

    issues = []

    if win_rate < 50:
        issues.append(f"1. LOW WIN RATE ({win_rate:.1f}%): More trades are losing than winning")

    if avg_loss > avg_win:
        issues.append(f"2. POOR RISK/REWARD: Average loss (${avg_loss:.2f}) > Average win (${avg_win:.2f})")

    if len(without_sl) > len(with_sl):
        issues.append(f"3. NO STOP LOSSES: {len(without_sl)/len(trades_df)*100:.0f}% of trades have no stop loss")

    if avg_hold_losers > avg_hold_winners * 2:
        issues.append(f"4. HOLDING LOSERS TOO LONG: Avg losing hold ({avg_hold_losers:.0f}m) >> Avg winning hold ({avg_hold_winners:.0f}m)")

    worst_symbol = by_symbol['Total P&L'].idxmin()
    worst_pnl = by_symbol.loc[worst_symbol, 'Total P&L']
    if worst_pnl < -100:
        issues.append(f"5. PROBLEM INSTRUMENT: {worst_symbol} has ${worst_pnl:,.2f} in losses")

    quick_trades = trades_df[trades_df['holding_minutes'] < 2]
    if len(quick_trades) > total_trades * 0.2:
        issues.append(f"6. OVERTRADING: {len(quick_trades)/total_trades*100:.0f}% of trades last < 2 minutes (scalping/impulsive)")

    for i, issue in enumerate(issues):
        print(f"\n{issue}")

    if not issues:
        print("\nNo major issues identified. Keep refining your strategy!")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    print("""
1. ALWAYS USE STOP LOSSES
   - Every trade should have a defined stop loss before entry
   - This prevents catastrophic losses

2. IMPROVE RISK/REWARD
   - Target at least 2:1 reward-to-risk ratio
   - Don't take trades where potential loss > potential gain

3. REDUCE OVERTRADING
   - Wait for quality setups instead of forcing trades
   - Quick scalps often lead to death by a thousand cuts

4. CUT LOSERS QUICKLY
   - If a trade isn't working, exit early
   - Don't let small losses become big losses

5. FOCUS ON BEST INSTRUMENTS
   - Trade the pairs where you're profitable
   - Avoid or reduce size on problem instruments
""")


def main():
    broker = connect()
    if not broker:
        print("ERROR: Failed to connect to TradeLocker")
        return

    try:
        api = broker._api

        # Get order history
        print("Fetching order history...")
        orders_df = api.get_all_orders(history=True)
        print(f"Found {len(orders_df)} orders")

        # Analyze trades
        print("Analyzing trades...")
        trades_df = analyze_trades(orders_df)
        print(f"Matched {len(trades_df)} complete trades")

        # Generate report
        generate_report(trades_df, orders_df)

        # Save to file
        trades_df.to_csv('trade_analysis.csv', index=False)
        print("\n\nDetailed trade data saved to: trade_analysis.csv")

    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
