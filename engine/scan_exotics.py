#!/usr/bin/env python
"""Scan Exotic Pairs"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

# Get all instruments to find exotics
instruments = api.get_all_instruments()
all_names = list(instruments['name'])

# Common exotic pairs to look for
exotics = [
    'USDMXN', 'USDZAR', 'USDTRY', 'USDSGD', 'USDHKD', 'USDPLN', 'USDSEK', 'USDNOK',
    'EURTRY', 'EURZAR', 'EURMXN', 'EURPLN', 'EURSEK', 'EURNOK', 'EURSGD', 'EURHUF',
    'GBPZAR', 'GBPMXN', 'GBPPLN', 'GBPSEK', 'GBPNOK', 'GBPSGD',
    'CHFPLN', 'CHFSEK', 'CHFZAR',
    'AUDSGD', 'AUDPLN', 'NZDSGD',
    'ZARJPY', 'MXNJPY', 'NOKJPY', 'SEKJPY', 'SGDJPY', 'TRYJPY',
    'USDCNH', 'EURCNH', 'USDRUB', 'EURRUB'
]

# Find available exotics
available = [p for p in exotics if p in all_names]
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

print('=' * 70)
print('EXOTIC PAIRS SCAN | %s | Balance: $340.65' % datetime.now().strftime('%I:%M %p'))
print('=' * 70)
print()
print('Available exotics: %d pairs' % len(available))
print()
print('%-8s %12s %6s %6s %5s %5s %s' % ('PAIR', 'PRICE', 'HTF', 'RSI', 'ZONE', '%RNG', 'STATUS'))
print('-' * 70)

for symbol in available:
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            continue

        # Get data
        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='7D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='3D')

        if d1 is None or h4 is None or h1 is None:
            continue
        if len(d1) < 10 or len(h4) < 20 or len(h1) < 24:
            continue

        price = api.get_latest_asking_price(inst_id)

        # HTF Bias
        d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-5] else 'BEAR'
        h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-5] else 'BEAR'
        htf_aligned = d1_bias == h4_bias
        htf_bias = d1_bias if htf_aligned else 'MIX'

        # RSI
        closes = h1['c'].values[-15:]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        rsi = 100 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))

        # Range
        high_20 = max(h4['h'].values[-20:])
        low_20 = min(h4['l'].values[-20:])
        range_pct = (price - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50

        zone = 'DISC' if range_pct < 30 else ('PREM' if range_pct > 70 else 'EQ')

        # Check setups - stricter for exotics due to wider spreads
        status = ''
        if htf_bias == 'BULL' and zone == 'DISC' and rsi < 40:
            status = '*** VALID LONG ***'
        elif htf_bias == 'BEAR' and zone == 'PREM' and rsi > 60:
            status = '*** VALID SHORT ***'
        elif htf_bias == 'BULL' and zone == 'DISC':
            status = 'LONG potential (RSI %.0f)' % rsi
        elif htf_bias == 'BEAR' and zone == 'PREM':
            status = 'SHORT potential (RSI %.0f)' % rsi
        elif htf_bias == 'MIX':
            status = 'HTF mixed'
        elif zone == 'EQ':
            status = 'Equilibrium'
        else:
            status = 'No edge'

        print('%-8s %12.5f %6s %6.1f %5s %5.0f %s' % (
            symbol, price, htf_bias, rsi, zone, range_pct, status
        ))

    except Exception as e:
        pass  # Skip errors silently

print()
print('Note: Exotics have wider spreads - use smaller position sizes')
print('=' * 70)
