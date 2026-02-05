#!/usr/bin/env python
"""Calculate win rate and R:R from full account history"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from collections import defaultdict

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
id_to_name = {v: k for k, v in dict(zip(instruments['name'], instruments['tradableInstrumentId'])).items()}

# Get ALL orders history
orders = api.get_all_orders(history=True)
filled_orders = orders[orders['status'] == 'Filled']

print('=' * 70)
print('ACCOUNT STATISTICS SUMMARY')
print('=' * 70)
print()
print('Total Filled Orders: %d' % len(filled_orders))
print()

# Track trades by position ID
positions = defaultdict(list)
for _, order in filled_orders.iterrows():
    positions[order['positionId']].append({
        'symbol': id_to_name.get(order['tradableInstrumentId'], str(order['tradableInstrumentId'])),
        'side': order['side'],
        'qty': order['filledQty'],
        'price': order['avgPrice']
    })

# Analyze closed trades
wins = 0
losses = 0
breakeven = 0
total_profit = 0
total_loss = 0
all_wins = []
all_losses = []

for pos_id, trades in positions.items():
    buys = [t for t in trades if t['side'] == 'buy']
    sells = [t for t in trades if t['side'] == 'sell']

    if buys and sells:
        buy_qty = sum(t['qty'] for t in buys)
        sell_qty = sum(t['qty'] for t in sells)
        closed_qty = min(buy_qty, sell_qty)

        if closed_qty > 0.001:
            avg_buy = sum(t['qty'] * t['price'] for t in buys) / buy_qty
            avg_sell = sum(t['qty'] * t['price'] for t in sells) / sell_qty
            symbol = trades[0]['symbol']

            # Calculate P/L
            if 'JPY' in symbol:
                pips = (avg_sell - avg_buy) * 100
                pip_value = 7.50
            elif 'XAU' in symbol:
                pips = (avg_sell - avg_buy) * 10
                pip_value = 10.0
            elif 'ETH' in symbol or 'BTC' in symbol:
                pips = avg_sell - avg_buy
                pip_value = closed_qty * 10  # rough estimate
            elif 'NAS' in symbol or 'SPX' in symbol or 'RUS' in symbol:
                pips = avg_sell - avg_buy
                pip_value = closed_qty
            elif symbol.endswith('USD'):
                pips = (avg_sell - avg_buy) * 10000
                pip_value = 10.0
            else:
                pips = (avg_sell - avg_buy) * 10000
                pip_value = 7.50

            pnl = pips * pip_value * closed_qty

            # Simpler P/L calc for crypto/indices
            if 'ETH' in symbol or 'BTC' in symbol or 'NAS' in symbol or 'SPX' in symbol or 'RUS' in symbol:
                pnl = (avg_sell - avg_buy) * closed_qty * 10

            if pnl > 0.50:
                wins += 1
                total_profit += pnl
                all_wins.append(pnl)
            elif pnl < -0.50:
                losses += 1
                total_loss += abs(pnl)
                all_losses.append(abs(pnl))
            else:
                breakeven += 1

# Calculate statistics
total_closed = wins + losses
print('=' * 70)
print('CLOSED TRADE STATISTICS')
print('=' * 70)
print()
print('  Total Closed Trades: %d' % total_closed)
print('  Breakeven Trades: %d' % breakeven)
print()
print('  WINS: %d' % wins)
print('  LOSSES: %d' % losses)
print()

if total_closed > 0:
    win_rate = (wins / total_closed) * 100
    print('  WIN RATE: %.1f%%' % win_rate)
    print()

if wins > 0:
    avg_win = total_profit / wins
    print('  Total Profit: $%.2f' % total_profit)
    print('  Average Win: $%.2f' % avg_win)
    if all_wins:
        print('  Largest Win: $%.2f' % max(all_wins))
        print('  Smallest Win: $%.2f' % min(all_wins))
    print()

if losses > 0:
    avg_loss = total_loss / losses
    print('  Total Loss: $%.2f' % total_loss)
    print('  Average Loss: $%.2f' % avg_loss)
    if all_losses:
        print('  Largest Loss: $%.2f' % max(all_losses))
        print('  Smallest Loss: $%.2f' % min(all_losses))
    print()

if wins > 0 and losses > 0:
    avg_win = total_profit / wins
    avg_loss = total_loss / losses
    risk_reward = avg_win / avg_loss
    net_pnl = total_profit - total_loss
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    print('=' * 70)
    print('KEY METRICS')
    print('=' * 70)
    print()
    print('  RISK/REWARD RATIO: %.2f' % risk_reward)
    print('  PROFIT FACTOR: %.2f' % profit_factor)
    print('  NET P/L (realized): $%.2f' % net_pnl)
    print('  EXPECTANCY: $%.2f per trade' % expectancy)

print()
print('=' * 70)

# Current account status
state = api.get_account_state()
print('CURRENT ACCOUNT')
print('=' * 70)
print('  Balance: $%.2f' % state['balance'])
print('  Open P/L: $%.2f' % state['openGrossPnL'])
print('  Equity: $%.2f' % (state['balance'] + state['openGrossPnL']))
print('=' * 70)
