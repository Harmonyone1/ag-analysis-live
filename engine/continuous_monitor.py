#!/usr/bin/env python
"""Continuous monitoring for positions and new setups"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Watchlist for new setups
WATCHLIST = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'EURJPY', 'GBPJPY',
             'EURGBP', 'AUDNZD', 'XAUUSD', 'US30', 'NAS100', 'BTCUSD']

def calc_rsi(closes):
    if len(closes) < 15: return 50
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-14:]]
    losses = [-d if d < 0 else 0 for d in deltas[-14:]]
    avg_gain = sum(gains) / 14
    avg_loss = sum(losses) / 14 if sum(losses) > 0 else 0.001
    return 100 - (100 / (1 + avg_gain / avg_loss))

print('=' * 70)
print('CONTINUOUS MONITOR | Started %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 70)
print()
print('Monitoring positions + scanning for new setups every 60 seconds')
print()

for cycle in range(30):  # 30 cycles x 60 sec = 30 minutes
    now = datetime.now().strftime('%I:%M:%S %p')

    try:
        # Account status
        state = api.get_account_state()
        balance = state['balance']
        equity = balance + state['openGrossPnL']
        pnl = state['openGrossPnL']

        print('=' * 70)
        print('[%s] Cycle %d/30 | Equity: $%.2f | P/L: $%.2f' % (now, cycle+1, equity, pnl))
        print('=' * 70)

        # Current positions
        positions = api.get_all_positions()
        if positions is not None and len(positions) > 0:
            print('POSITIONS:')
            nzdcad_pnl = 0
            nzdusd_pnl = 0
            for _, pos in positions.iterrows():
                symbol = str(pos.get('symbol', ''))
                pos_pnl = pos.get('unrealizedPnL', 0)
                if 'NZDCAD' in symbol:
                    nzdcad_pnl += pos_pnl
                elif 'NZDUSD' in symbol:
                    nzdusd_pnl += pos_pnl
            print('  NZDCAD: $%.2f | NZDUSD: $%.2f' % (nzdcad_pnl, nzdusd_pnl))

        # Quick scan for new setups
        new_setups = []
        for symbol in WATCHLIST[:6]:  # Limit to avoid timeout
            try:
                inst_id = name_to_id.get(symbol)
                if not inst_id: continue
                d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
                h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='14D')
                h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='7D')
                if d1 is None or h4 is None or h1 is None: continue

                price = api.get_latest_asking_price(inst_id)
                d1_b = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-5] else 'BEAR'
                h4_b = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-5] else 'BEAR'
                aligned = d1_b == h4_b
                rsi = calc_rsi(h1['c'].values)
                high = max(h4['h'].values[-20:])
                low = min(h4['l'].values[-20:])
                pct = (price - low) / (high - low) * 100 if high != low else 50
                zone = 'DISC' if pct < 30 else ('PREM' if pct > 70 else 'EQ')

                if aligned and d1_b == 'BULL' and zone == 'DISC' and rsi < 40:
                    new_setups.append('%s VALID LONG RSI:%.1f' % (symbol, rsi))
                elif aligned and d1_b == 'BEAR' and zone == 'PREM' and rsi > 60:
                    new_setups.append('%s VALID SHORT RSI:%.1f' % (symbol, rsi))

                time.sleep(0.2)
            except:
                continue

        if new_setups:
            print()
            print('!' * 50)
            print('NEW VALID SETUPS FOUND:')
            for s in new_setups:
                print('  >>> %s <<<' % s)
            print('!' * 50)
        else:
            print('  No new valid setups')

        print()

        if cycle < 29:
            time.sleep(60)

    except Exception as e:
        print('Error: %s' % str(e)[:50])
        time.sleep(60)

print('=' * 70)
print('Monitor session complete')
print('=' * 70)
