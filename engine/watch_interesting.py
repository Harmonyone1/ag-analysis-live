#!/usr/bin/env python
"""Monitor Interesting Setups"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Interesting setups to watch
WATCHLIST = [
    # Extreme overbought (watching for pullback/reversal)
    ('AUDNZD', 'OVERBOUGHT', 'Wait for pullback to LONG with trend'),
    ('USDJPY', 'OVERBOUGHT', 'Wait for pullback to LONG with trend'),
    ('GBPJPY', 'OVERBOUGHT', 'Wait for pullback to LONG with trend'),
    ('EURJPY', 'OVERBOUGHT', 'Wait for pullback to LONG with trend'),
    # Extreme oversold (watching for bounce/reversal)
    ('NZDUSD', 'OVERSOLD', 'Counter-trend bounce potential'),
    ('NZDCAD', 'OVERSOLD', 'Counter-trend bounce potential'),
    ('NZDCHF', 'OVERSOLD', 'Counter-trend bounce potential'),
    # Stocks
    ('PFE', 'OVERSOLD', 'Very oversold stock'),
    # Developing
    ('EURUSD', 'DEVELOPING', 'Approaching oversold'),
    ('AUDUSD', 'DEVELOPING', 'In discount zone'),
]

def analyze(symbol):
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            return None

        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='7D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='3D')

        if d1 is None or h4 is None or h1 is None:
            return None

        price = api.get_latest_asking_price(inst_id)

        d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-min(5, len(d1))] else 'BEAR'
        h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-min(5, len(h4))] else 'BEAR'
        htf_aligned = d1_bias == h4_bias
        htf_bias = d1_bias if htf_aligned else 'MIX'

        closes = h1['c'].values[-15:]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0.001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))

        high_20 = max(h4['h'].values[-min(20, len(h4)):])
        low_20 = min(h4['l'].values[-min(20, len(h4)):])
        range_pct = (price - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50

        zone = 'DISC' if range_pct < 30 else ('PREM' if range_pct > 70 else 'EQ')

        # Check for valid setup
        valid = False
        signal = None
        if htf_bias == 'BULL' and zone == 'DISC' and rsi < 40:
            valid = True
            signal = 'LONG'
        elif htf_bias == 'BEAR' and zone == 'PREM' and rsi > 60:
            valid = True
            signal = 'SHORT'

        return {
            'symbol': symbol,
            'price': price,
            'htf': htf_bias,
            'rsi': rsi,
            'zone': zone,
            'pct': range_pct,
            'valid': valid,
            'signal': signal
        }
    except:
        return None

state = api.get_account_state()
balance = state['balance']

print('=' * 75)
print('INTERESTING SETUPS MONITOR | %s | Balance: $%.2f' % (
    datetime.now().strftime('%I:%M:%S %p'), balance))
print('=' * 75)
print()

valid_setups = []

print('%-8s %10s %5s %6s %5s %5s %-8s %s' % (
    'PAIR', 'PRICE', 'HTF', 'RSI', 'ZONE', '%RNG', 'STATUS', 'NOTES'))
print('-' * 75)

for symbol, category, notes in WATCHLIST:
    r = analyze(symbol)
    if r:
        # Status indicator
        if r['valid']:
            status = '>>> VALID'
            valid_setups.append(r)
        elif r['rsi'] < 30:
            status = 'OVERSOLD'
        elif r['rsi'] > 70:
            status = 'OVERBOUGHT'
        elif r['zone'] == 'DISC':
            status = 'DISCOUNT'
        elif r['zone'] == 'PREM':
            status = 'PREMIUM'
        else:
            status = 'WATCHING'

        print('%-8s %10.5f %5s %6.1f %5s %5.0f %-8s %s' % (
            r['symbol'], r['price'], r['htf'], r['rsi'], r['zone'], r['pct'], status, notes))

print()
print('=' * 75)

if valid_setups:
    print()
    print('!!! VALID SETUPS DETECTED !!!')
    for s in valid_setups:
        print('  >>> %s %s | HTF: %s | RSI: %.1f | %s %.0f%%' % (
            s['symbol'], s['signal'], s['htf'], s['rsi'], s['zone'], s['pct']))
    print()
else:
    print('No valid setups yet. Continuing to monitor...')
    print()

# Summary of what we're waiting for
print('WAITING FOR:')
print('  - AUDNZD/USDJPY/GBPJPY/EURJPY to pullback to discount for LONG')
print('  - NZDUSD/NZDCAD/NZDCHF RSI to rise or HTF to flip for alignment')
print('  - EURUSD/AUDUSD RSI to drop below 40')
print()
print('=' * 75)
