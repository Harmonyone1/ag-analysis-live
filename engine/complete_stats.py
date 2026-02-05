#!/usr/bin/env python
"""Complete account statistics with win rate and R:R"""

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
print('COMPLETE ACCOUNT STATISTICS')
print('=' * 70)
print()

# Get executions
executions = api.get_all_executions()
state = api.get_account_state()

# Analyze closed trades
closed_trades = []
if executions is not None and len(executions) > 0:
    # Group by symbol
    trades_by_symbol = {}
    for _, exe in executions.iterrows():
        tid = exe.get('tradableInstrumentId', '')
        symbol = id_to_name.get(tid, str(tid))
        side = exe.get('side', '')
        qty = exe.get('qty', 0)
        price = exe.get('price', 0)

        if symbol not in trades_by_symbol:
            trades_by_symbol[symbol] = {'buys': [], 'sells': []}

        trades_by_symbol[symbol][side + 's'].append({'qty': qty, 'price': price})

    print('CLOSED TRADES:')
    print('-' * 70)

    wins = 0
    losses = 0
    total_profit = 0
    total_loss = 0

    for sym, trades in trades_by_symbol.items():
        buys = trades['buys']
        sells = trades['sells']

        if buys and sells:
            # This is a closed trade
            buy_qty = sum(t['qty'] for t in buys)
            sell_qty = sum(t['qty'] for t in sells)
            closed_qty = min(buy_qty, sell_qty)

            avg_buy = sum(t['qty'] * t['price'] for t in buys) / buy_qty
            avg_sell = sum(t['qty'] * t['price'] for t in sells) / sell_qty

            # Calculate P/L based on pair type
            if 'JPY' in sym:
                # JPY pairs: 1 pip = 0.01, pip value ~$7.50 per lot
                pips = (avg_sell - avg_buy) * 100
                pip_value = 7.50
            elif sym.endswith('USD'):
                # USD quote: 1 pip = 0.0001, pip value = $10 per lot
                pips = (avg_sell - avg_buy) * 10000
                pip_value = 10.0
            else:
                # Cross pairs approximate
                pips = (avg_sell - avg_buy) * 10000
                pip_value = 7.50

            pnl = pips * pip_value * closed_qty

            print('  %s: BUY %.2f @ %.5f -> SELL @ %.5f' % (sym, closed_qty, avg_buy, avg_sell))
            print('    Result: %.1f pips | P/L: $%.2f' % (pips, pnl))

            if pnl > 0:
                wins += 1
                total_profit += pnl
            else:
                losses += 1
                total_loss += abs(pnl)

            closed_trades.append({
                'symbol': sym,
                'qty': closed_qty,
                'entry': avg_buy,
                'exit': avg_sell,
                'pips': pips,
                'pnl': pnl
            })
            print()

    if not closed_trades:
        print('  No fully closed trades found')
        print()

print()
print('=' * 70)
print('STATISTICS SUMMARY')
print('=' * 70)
print()

if closed_trades:
    total = wins + losses
    win_rate = (wins / total) * 100 if total > 0 else 0
    avg_win = total_profit / wins if wins > 0 else 0
    avg_loss = total_loss / losses if losses > 0 else 0
    risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)

    print('CLOSED TRADE STATS:')
    print('  Total Closed: %d trades' % total)
    print('  Wins: %d | Losses: %d' % (wins, losses))
    print('  Win Rate: %.1f%%' % win_rate)
    print('  Average Win: $%.2f' % avg_win)
    print('  Average Loss: $%.2f' % avg_loss)
    print('  Risk/Reward Ratio: %.2f' % risk_reward)
    print('  Expectancy: $%.2f per trade' % expectancy)
    print()
else:
    print('No closed trades to calculate statistics.')
    print('All current positions are still OPEN.')
    print()

# Show today's account stats
print('TODAY\'S ACCOUNT ACTIVITY:')
print('  Trades Executed: %d' % state['todayTradesCount'])
print('  Gross P/L (realized): $%.2f' % state['todayGross'])
print('  Net P/L (after fees): $%.2f' % state['todayNet'])
print('  Fees Paid: $%.2f' % state['todayFees'])
print()

# Current open positions
print('OPEN POSITIONS:')
positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    pos_summary = {}
    for _, pos in positions.iterrows():
        tid = str(pos.get('tradableInstrumentId', ''))
        symbol = id_to_name.get(int(tid), tid) if tid.isdigit() else tid
        qty = pos.get('qty', 0)
        entry = pos.get('avgPrice', 0)

        if symbol not in pos_summary:
            pos_summary[symbol] = {'qty': 0, 'entries': []}
        pos_summary[symbol]['qty'] += qty
        pos_summary[symbol]['entries'].append(entry)

    for sym, data in pos_summary.items():
        avg_entry = sum(data['entries']) / len(data['entries'])
        print('  %s LONG %.2f lots @ %.5f' % (sym, data['qty'], avg_entry))

print()

# Overall account performance
print('=' * 70)
print('OVERALL ACCOUNT PERFORMANCE')
print('=' * 70)
print('  Balance: $%.2f' % state['balance'])
print('  Open P/L: $%.2f' % state['openGrossPnL'])
print('  Equity: $%.2f' % (state['balance'] + state['openGrossPnL']))
print()
print('  Session Start: $336.50')
print('  Current Equity: $%.2f' % (state['balance'] + state['openGrossPnL']))
print('  Session Gain: $%.2f (+%.1f%%)' % (
    state['balance'] + state['openGrossPnL'] - 336.50,
    ((state['balance'] + state['openGrossPnL'] - 336.50) / 336.50) * 100
))
print('=' * 70)
