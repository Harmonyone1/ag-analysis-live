#!/usr/bin/env python
"""Get trade statistics - win rate and R:R"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))
id_to_name = {v: k for k, v in name_to_id.items()}

print('=' * 70)
print('TRADING STATISTICS')
print('=' * 70)
print()

# Get current account state
state = api.get_account_state()
balance = state['balance']
equity = balance + state['openGrossPnL']
open_pnl = state['openGrossPnL']

print('ACCOUNT STATUS:')
print('  Balance: $%.2f' % balance)
print('  Equity: $%.2f' % equity)
print('  Open P/L: $%.2f' % open_pnl)
print()

# Get current open positions with details
print('CURRENT OPEN POSITIONS:')
positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    pos_data = []
    for _, pos in positions.iterrows():
        tid = str(pos.get('tradableInstrumentId', ''))
        symbol = id_to_name.get(int(tid), tid) if tid.isdigit() else tid
        qty = pos.get('qty', 0)
        entry = pos.get('avgPrice', 0)
        upnl = pos.get('unrealizedPnL', 0)
        sl = pos.get('stopLoss', 0)
        tp = pos.get('takeProfit', 0)
        side = pos.get('side', 'buy')

        pos_data.append({
            'symbol': symbol,
            'qty': qty,
            'entry': entry,
            'pnl': upnl,
            'sl': sl,
            'tp': tp,
            'side': side
        })

    # Group by symbol
    pos_summary = {}
    for p in pos_data:
        sym = p['symbol']
        if sym not in pos_summary:
            pos_summary[sym] = {'qty': 0, 'total_pnl': 0, 'entries': [], 'sl': 0, 'tp': 0}
        pos_summary[sym]['qty'] += p['qty']
        pos_summary[sym]['total_pnl'] += p['pnl']
        pos_summary[sym]['entries'].append(p['entry'])
        pos_summary[sym]['sl'] = p['sl']
        pos_summary[sym]['tp'] = p['tp']

    for sym, data in pos_summary.items():
        avg_entry = sum(data['entries']) / len(data['entries']) if data['entries'] else 0
        print('  %s LONG %.2f lots' % (sym, data['qty']))
        print('    Entry: %.5f | P/L: $%.2f' % (avg_entry, data['total_pnl']))
        if data['sl'] and data['tp']:
            print('    SL: %.5f | TP: %.5f' % (data['sl'], data['tp']))
        print()

print()

# Session summary
starting_balance = 336.50  # From session start
session_pnl = equity - starting_balance
session_pct = (session_pnl / starting_balance) * 100

print('=' * 70)
print('SESSION PERFORMANCE')
print('=' * 70)
print()
print('  Starting Balance: $%.2f' % starting_balance)
print('  Current Equity: $%.2f' % equity)
print('  Session P/L: $%.2f (%.1f%%)' % (session_pnl, session_pct))
print()

# Calculate trade stats for this session
# All trades entered this session:
# 1. NZDCAD LONG 0.20 lots - still open, profitable
# 2. NZDUSD LONG 0.60 lots - still open, profitable
# 3. JNJ LONG 0.02 lots - still open, small loss
# 4. CVX LONG 0.01 lots - still open, small loss

print('TRADES THIS SESSION:')
print('  Total Trades: 4 instruments (9 order entries)')
print('  All positions still OPEN')
print()

# Current status by trade
print('TRADE STATUS:')
trades_profit = 0
trades_loss = 0
for sym, data in pos_summary.items():
    if data['total_pnl'] > 0:
        trades_profit += 1
        status = 'WINNING'
    else:
        trades_loss += 1
        status = 'LOSING'
    print('  %s: %s ($%.2f)' % (sym, status, data['total_pnl']))

print()
total_trades = trades_profit + trades_loss
if total_trades > 0:
    win_rate = (trades_profit / total_trades) * 100
    print('CURRENT WIN RATE: %.0f%% (%d/%d positions profitable)' % (win_rate, trades_profit, total_trades))

print()

# Risk/Reward for forex positions
print('RISK/REWARD ANALYSIS:')
print()

# NZDCAD
if 'NZDCAD' in pos_summary:
    nzd_entry = sum(pos_summary['NZDCAD']['entries']) / len(pos_summary['NZDCAD']['entries'])
    nzd_sl = 0.7907  # From our entries
    nzd_tp = 0.7963
    risk_pips = (nzd_entry - nzd_sl) * 10000
    reward_pips = (nzd_tp - nzd_entry) * 10000
    rr = reward_pips / risk_pips if risk_pips > 0 else 0
    print('  NZDCAD:')
    print('    Risk: %.1f pips | Reward: %.1f pips' % (risk_pips, reward_pips))
    print('    R:R = 1:%.2f' % rr)
    print()

# NZDUSD
if 'NZDUSD' in pos_summary:
    nzu_entry = sum(pos_summary['NZDUSD']['entries']) / len(pos_summary['NZDUSD']['entries'])
    nzu_sl = 0.5730
    nzu_tp = 0.5805
    risk_pips = (nzu_entry - nzu_sl) * 10000
    reward_pips = (nzu_tp - nzu_entry) * 10000
    rr = reward_pips / risk_pips if risk_pips > 0 else 0
    print('  NZDUSD:')
    print('    Risk: %.1f pips | Reward: %.1f pips' % (risk_pips, reward_pips))
    print('    R:R = 1:%.2f' % rr)
    print()

# Stocks
print('  JNJ & CVX: 3%% SL / 6%% TP = 1:2.0 R:R')
print()

print('=' * 70)
print('SUMMARY: All 4 positions still open. Forex positions profitable,')
print('stocks slightly negative. Overall session +$%.2f (+%.1f%%)' % (session_pnl, session_pct))
print('=' * 70)
