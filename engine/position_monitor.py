#!/usr/bin/env python
"""Position Monitor - Track open positions"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
id_to_symbol = dict(zip(instruments['tradableInstrumentId'], instruments['name']))

print('=' * 60)
print('POSITION MONITOR - ICT+QUANT Framework')
print('=' * 60)

for cycle in range(30):  # 30 cycles x 30 sec = 15 min
    try:
        state = api.get_account_state()
        balance = state['balance']
        open_pnl = state['openGrossPnL']
        equity = balance + open_pnl

        positions = api.get_all_positions()

        now = datetime.now().strftime('%I:%M:%S %p')

        print()
        print('[%s] Equity: $%.2f | PnL: $%.2f' % (now, equity, open_pnl))

        if not positions.empty:
            for _, p in positions.iterrows():
                sym = id_to_symbol.get(p['tradableInstrumentId'], '???')
                side = 'BUY' if p['side'] == 'buy' else 'SELL'
                qty = p['qty']
                entry = p['avgPrice']
                pnl = p['unrealizedPl']

                # Get current price
                try:
                    current = api.get_latest_asking_price(p['tradableInstrumentId'])
                except:
                    current = entry

                # Calculate pips
                if 'JPY' in sym:
                    pip_size = 0.01
                elif sym in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
                    pip_size = 1.0
                else:
                    pip_size = 0.0001

                if side == 'BUY':
                    pips = (current - entry) / pip_size
                else:
                    pips = (entry - current) / pip_size

                # Status indicator
                if pnl > 0:
                    status = '++' if pnl > 1 else '+ '
                else:
                    status = '--' if pnl < -1 else '- '

                print('  %s %s %s %.2f @ %.5f | Now: %.5f | %+.1f pips | $%.2f' % (
                    status, sym, side, qty, entry, current, pips, pnl
                ))

                # Progress to targets
                if 'stopLoss' in p and 'takeProfit' in p:
                    sl = p['stopLoss']
                    tp = p['takeProfit']
                    if side == 'BUY':
                        total_range = tp - sl
                        progress = (current - sl) / total_range * 100 if total_range > 0 else 50
                    else:
                        total_range = sl - tp
                        progress = (sl - current) / total_range * 100 if total_range > 0 else 50
                    print('      SL: %.5f | TP: %.5f | Progress: %.0f%%' % (sl, tp, progress))
        else:
            print('  No open positions')
            # Check if there were any recent closed trades
            break

    except Exception as e:
        print('  Error: %s' % str(e))

    if cycle < 29:
        time.sleep(30)

print()
print('=' * 60)
print('Monitoring cycle complete')
print('=' * 60)
