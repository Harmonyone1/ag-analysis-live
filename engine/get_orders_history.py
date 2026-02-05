#!/usr/bin/env python
"""Get full orders history via direct API call"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import requests
import json

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))
id_to_name = {v: k for k, v in name_to_id.items()}

print('=' * 70)
print('FULL ACCOUNT HISTORY')
print('=' * 70)
print()

# Get access token and make direct API call
try:
    access_token = api.get_access_token()
    base_url = api.get_base_url()
    account_id = 592535

    print('API Access obtained')
    print('Base URL:', base_url)
    print()

    # Try to get orders history via direct API
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'accNum': str(account_id)
    }

    # Get orders history
    url = f'{base_url}/trade/accounts/{account_id}/ordersHistory'
    print('Fetching orders history from:', url)

    response = requests.get(url, headers=headers)
    print('Response status:', response.status_code)

    if response.status_code == 200:
        data = response.json()
        if 'd' in data and 'ordersHistory' in data['d']:
            orders = data['d']['ordersHistory']
            print('Total orders in history:', len(orders))
            print()

            if orders:
                # Analyze the orders
                print('ORDER HISTORY:')
                print('-' * 70)

                trades = {}  # Group by position ID

                for order in orders:
                    pos_id = order.get('positionId', '')
                    tid = order.get('tradableInstrumentId', 0)
                    symbol = id_to_name.get(tid, str(tid))
                    side = order.get('side', '')
                    qty = order.get('filledQty', 0)
                    avg_price = order.get('avgPrice', 0)
                    status = order.get('status', '')
                    order_type = order.get('type', '')
                    created = order.get('createdDate', 0)

                    if qty > 0:  # Only filled orders
                        from datetime import datetime
                        dt = datetime.fromtimestamp(created / 1000) if created > 0 else None
                        date_str = dt.strftime('%m/%d %H:%M') if dt else 'N/A'

                        print('  %s | %s %s %.2f @ %.5f [%s]' % (
                            date_str, symbol, side.upper(), qty, avg_price, status))

                        # Track for P/L calculation
                        if symbol not in trades:
                            trades[symbol] = {'buys': [], 'sells': []}
                        if side == 'buy':
                            trades[symbol]['buys'].append({'qty': qty, 'price': avg_price, 'date': date_str})
                        else:
                            trades[symbol]['sells'].append({'qty': qty, 'price': avg_price, 'date': date_str})

                print()
                print('-' * 70)
                print()

                # Calculate P/L for closed trades
                print('CLOSED TRADE ANALYSIS:')
                print('-' * 70)

                wins = 0
                losses = 0
                total_profit = 0
                total_loss = 0
                closed_trades = []

                for sym, trade_data in trades.items():
                    buys = trade_data['buys']
                    sells = trade_data['sells']

                    if buys and sells:
                        buy_qty = sum(t['qty'] for t in buys)
                        sell_qty = sum(t['qty'] for t in sells)
                        closed_qty = min(buy_qty, sell_qty)

                        if closed_qty > 0:
                            avg_buy = sum(t['qty'] * t['price'] for t in buys) / buy_qty
                            avg_sell = sum(t['qty'] * t['price'] for t in sells) / sell_qty

                            # Calculate P/L
                            if 'JPY' in sym:
                                pips = (avg_sell - avg_buy) * 100
                                pip_value = 7.50
                            elif sym.endswith('USD'):
                                pips = (avg_sell - avg_buy) * 10000
                                pip_value = 10.0
                            else:
                                pips = (avg_sell - avg_buy) * 10000
                                pip_value = 7.50

                            pnl = pips * pip_value * closed_qty

                            print('  %s: %.2f lots closed' % (sym, closed_qty))
                            print('    Avg Entry: %.5f | Avg Exit: %.5f' % (avg_buy, avg_sell))
                            print('    Result: %.1f pips | P/L: $%.2f' % (pips, pnl))
                            print()

                            if pnl > 0:
                                wins += 1
                                total_profit += pnl
                            elif pnl < 0:
                                losses += 1
                                total_loss += abs(pnl)

                            closed_trades.append({
                                'symbol': sym,
                                'qty': closed_qty,
                                'pnl': pnl
                            })

                # Summary
                print('=' * 70)
                print('ACCOUNT STATISTICS')
                print('=' * 70)
                print()

                total_closed = wins + losses
                if total_closed > 0:
                    win_rate = (wins / total_closed) * 100
                    avg_win = total_profit / wins if wins > 0 else 0
                    avg_loss = total_loss / losses if losses > 0 else 0
                    rr = avg_win / avg_loss if avg_loss > 0 else float('inf')
                    net_pnl = total_profit - total_loss

                    print('  Total Closed Trades: %d' % total_closed)
                    print('  Wins: %d | Losses: %d' % (wins, losses))
                    print('  Win Rate: %.1f%%' % win_rate)
                    print()
                    print('  Total Profit: $%.2f' % total_profit)
                    print('  Total Loss: $%.2f' % total_loss)
                    print('  Net P/L: $%.2f' % net_pnl)
                    print()
                    print('  Average Win: $%.2f' % avg_win)
                    print('  Average Loss: $%.2f' % avg_loss)
                    print('  Risk/Reward Ratio: %.2f' % rr)
                else:
                    print('  No closed trades found in history')

        else:
            print('No orders history data in response')
            print('Response:', json.dumps(data, indent=2)[:500])
    else:
        print('Error:', response.text[:500])

except Exception as e:
    print('Error:', str(e))
    import traceback
    traceback.print_exc()

print()
print('=' * 70)

# Also show account balance info
state = api.get_account_state()
print('CURRENT ACCOUNT:')
print('  Balance: $%.2f' % state['balance'])
print('  Equity: $%.2f' % (state['balance'] + state['openGrossPnL']))
print('  Open P/L: $%.2f' % state['openGrossPnL'])
print('=' * 70)
