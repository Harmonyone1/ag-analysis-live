#!/usr/bin/env python
"""Get FULL account trade history using get_all_orders(history=True)"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime
from collections import defaultdict

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))
id_to_name = {v: k for k, v in name_to_id.items()}

print('=' * 70)
print('FULL ACCOUNT TRADE HISTORY')
print('=' * 70)
print()

# Get ALL orders history
orders = api.get_all_orders(history=True)

if orders is not None and len(orders) > 0:
    print('Total orders in history: %d' % len(orders))
    print()

    # Filter to only filled orders
    filled_orders = orders[orders['status'] == 'Filled']
    print('Filled orders: %d' % len(filled_orders))
    print()

    if len(filled_orders) > 0:
        print('ORDER HISTORY (Filled Orders):')
        print('-' * 70)

        # Track trades by position ID for accurate P/L calculation
        positions = defaultdict(list)

        for _, order in filled_orders.iterrows():
            tid = order['tradableInstrumentId']
            symbol = id_to_name.get(tid, str(tid))
            side = order['side']
            qty = order['filledQty']
            avg_price = order['avgPrice']
            pos_id = order['positionId']
            created = order['createdDate']

            dt = datetime.fromtimestamp(created / 1000) if created > 0 else None
            date_str = dt.strftime('%m/%d %H:%M') if dt else 'N/A'

            print('  %s | %s %s %.2f @ %.5f [Pos: %s]' % (
                date_str, symbol, side.upper(), qty, avg_price, str(pos_id)[-6:]))

            positions[pos_id].append({
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'price': avg_price,
                'date': date_str
            })

        print()
        print('-' * 70)
        print()

        # Analyze positions to find closed trades
        print('CLOSED TRADES ANALYSIS:')
        print('-' * 70)

        closed_trades = []
        open_positions = []

        for pos_id, trades in positions.items():
            buys = [t for t in trades if t['side'] == 'buy']
            sells = [t for t in trades if t['side'] == 'sell']

            buy_qty = sum(t['qty'] for t in buys)
            sell_qty = sum(t['qty'] for t in sells)

            symbol = trades[0]['symbol']

            if buys and sells:
                # Closed or partially closed position
                closed_qty = min(buy_qty, sell_qty)

                avg_buy = sum(t['qty'] * t['price'] for t in buys) / buy_qty
                avg_sell = sum(t['qty'] * t['price'] for t in sells) / sell_qty

                # Calculate P/L
                if 'JPY' in symbol:
                    pips = (avg_sell - avg_buy) * 100
                    pip_value = 7.50  # per lot for JPY pairs
                elif symbol.endswith('USD') or 'USD' in symbol:
                    pips = (avg_sell - avg_buy) * 10000
                    pip_value = 10.0  # per lot for USD pairs
                else:
                    pips = (avg_sell - avg_buy) * 10000
                    pip_value = 7.50  # approximate for cross pairs

                pnl = pips * pip_value * closed_qty

                closed_trades.append({
                    'symbol': symbol,
                    'qty': closed_qty,
                    'entry': avg_buy,
                    'exit': avg_sell,
                    'pips': pips,
                    'pnl': pnl,
                    'pos_id': pos_id
                })

                remaining = abs(buy_qty - sell_qty)
                if remaining > 0.001:
                    open_positions.append({
                        'symbol': symbol,
                        'qty': remaining,
                        'side': 'buy' if buy_qty > sell_qty else 'sell'
                    })

            else:
                # Still open position
                open_positions.append({
                    'symbol': symbol,
                    'qty': buy_qty if buys else sell_qty,
                    'side': 'buy' if buys else 'sell'
                })

        # Print closed trades
        if closed_trades:
            wins = 0
            losses = 0
            total_profit = 0
            total_loss = 0
            total_risk = 0
            total_reward = 0

            for trade in closed_trades:
                result = 'WIN' if trade['pnl'] > 0 else 'LOSS'
                print('  %s: %.2f lots | Entry: %.5f | Exit: %.5f' % (
                    trade['symbol'], trade['qty'], trade['entry'], trade['exit']))
                print('    Result: %.1f pips | P/L: $%.2f [%s]' % (trade['pips'], trade['pnl'], result))
                print()

                if trade['pnl'] > 0:
                    wins += 1
                    total_profit += trade['pnl']
                    total_reward += trade['pnl']
                elif trade['pnl'] < 0:
                    losses += 1
                    total_loss += abs(trade['pnl'])
                    total_risk += abs(trade['pnl'])

            # Calculate statistics
            print('=' * 70)
            print('ACCOUNT STATISTICS (Closed Trades)')
            print('=' * 70)
            print()

            total_trades = wins + losses
            if total_trades > 0:
                win_rate = (wins / total_trades) * 100
                avg_win = total_profit / wins if wins > 0 else 0
                avg_loss = total_loss / losses if losses > 0 else 0
                risk_reward = avg_win / avg_loss if avg_loss > 0 else float('inf')
                net_pnl = total_profit - total_loss
                expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)

                print('  TOTAL CLOSED TRADES: %d' % total_trades)
                print('  WINS: %d | LOSSES: %d' % (wins, losses))
                print()
                print('  WIN RATE: %.1f%%' % win_rate)
                print()
                print('  Total Profit (from wins): $%.2f' % total_profit)
                print('  Total Loss (from losses): $%.2f' % total_loss)
                print('  NET P/L (closed): $%.2f' % net_pnl)
                print()
                print('  Average Win: $%.2f' % avg_win)
                print('  Average Loss: $%.2f' % avg_loss)
                print('  RISK/REWARD RATIO: %.2f' % risk_reward)
                print()
                print('  Expectancy: $%.2f per trade' % expectancy)
            else:
                print('  No closed trades to analyze')

        else:
            print('  No closed trades found')
            print('  All positions are still OPEN')

        print()
        print('=' * 70)

else:
    print('No order history found')

# Account summary
state = api.get_account_state()
print('CURRENT ACCOUNT STATUS')
print('=' * 70)
print('  Balance: $%.2f' % state['balance'])
print('  Open P/L: $%.2f' % state['openGrossPnL'])
print('  Equity: $%.2f' % (state['balance'] + state['openGrossPnL']))
print('  Open Positions: %d' % state['positionsCount'])
print('=' * 70)
