#!/usr/bin/env python
"""Monitor Position + Scan for More Opportunities"""

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
id_to_name = dict(zip(instruments['tradableInstrumentId'], instruments['name']))

SCAN_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'GBPJPY', 'AUDNZD',
              'NZDCAD', 'NZDCHF', 'XAUUSD', 'US30', 'NAS100']

def calc_rsi(closes):
    if len(closes) < 15:
        return 50
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-14:]]
    losses = [-d if d < 0 else 0 for d in deltas[-14:]]
    avg_gain = sum(gains) / 14
    avg_loss = sum(losses) / 14 if sum(losses) > 0 else 0.001
    return 100 - (100 / (1 + avg_gain / avg_loss))

def analyze(symbol):
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            return None

        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='7D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='3D')
        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')

        if h4 is None or h1 is None or d1 is None:
            return None

        price = api.get_latest_asking_price(inst_id)

        d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-5] else 'BEAR'
        h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-5] else 'BEAR'
        htf = d1_bias if d1_bias == h4_bias else 'MIX'

        rsi = calc_rsi(h1['c'].values)

        high = max(h4['h'].values[-20:])
        low = min(h4['l'].values[-20:])
        pct = (price - low) / (high - low) * 100 if high != low else 50

        zone = 'DISC' if pct < 30 else ('PREM' if pct > 70 else 'EQ')

        # Check for valid setup
        valid = None
        if htf == 'BULL' and zone == 'DISC' and rsi < 40:
            valid = 'LONG'
        elif htf == 'BEAR' and zone == 'PREM' and rsi > 60:
            valid = 'SHORT'

        return {'symbol': symbol, 'price': price, 'htf': htf, 'rsi': rsi, 'zone': zone, 'pct': pct, 'valid': valid}
    except:
        return None

print('=' * 70)
print('POSITION MONITOR + MARKET SCANNER')
print('=' * 70)

for cycle in range(30):
    now = datetime.now().strftime('%I:%M:%S %p')

    # Get account state
    try:
        state = api.get_account_state()
        balance = state['balance']
        equity = balance + state['openGrossPnL']
        pnl = state['openGrossPnL']
    except:
        balance = 340.65
        equity = balance
        pnl = 0

    # Get positions
    try:
        positions = api.get_all_positions()
        has_positions = not positions.empty
    except:
        has_positions = False
        positions = None

    print()
    print('[%s] Equity: $%.2f | PnL: $%.2f' % (now, equity, pnl))
    print('-' * 70)

    if has_positions:
        for _, p in positions.iterrows():
            sym = id_to_name.get(p['tradableInstrumentId'], '???')
            side = 'SHORT' if p['side'] == 'sell' else 'LONG'
            entry = p['avgPrice']
            pos_pnl = p['unrealizedPl']

            try:
                current = api.get_latest_asking_price(p['tradableInstrumentId'])
            except:
                current = entry

            pip_mult = 100 if 'JPY' in sym else 10000
            pips = (entry - current) * pip_mult if side == 'SHORT' else (current - entry) * pip_mult

            status = '++' if pos_pnl > 1 else ('+' if pos_pnl > 0 else ('-' if pos_pnl > -1 else '--'))

            print('%s %s %s %.2f @ %.3f | Now: %.3f | %+.1f pips | $%.2f' % (
                status, sym, side, p['qty'], entry, current, pips, pos_pnl))

            # Check SL/TP
            if 'stopLoss' in p and p['stopLoss']:
                sl_pips = abs(entry - p['stopLoss']) * pip_mult
                tp_pips = abs(p['takeProfit'] - entry) * pip_mult if 'takeProfit' in p and p['takeProfit'] else 0
                print('   SL: %.3f (%.0f pips) | TP: %.3f (%.0f pips)' % (
                    p['stopLoss'], sl_pips, p.get('takeProfit', 0), tp_pips))
    else:
        print('No open positions')

    # Quick scan for new opportunities
    print()
    print('Scanning for new setups...')

    valid_setups = []
    for sym in SCAN_PAIRS:
        r = analyze(sym)
        if r and r['valid']:
            valid_setups.append(r)
        time.sleep(0.2)

    if valid_setups:
        print()
        print('>>> NEW VALID SETUPS DETECTED:')
        for s in valid_setups:
            print('    %s %s | HTF: %s | RSI: %.1f | %s %.0f%%' % (
                s['symbol'], s['valid'], s['htf'], s['rsi'], s['zone'], s['pct']))
    else:
        print('No new valid setups')

    print()
    print('Next update in 60 seconds...')

    if cycle < 29:
        time.sleep(60)

print()
print('=' * 70)
print('Monitoring complete')
print('=' * 70)
