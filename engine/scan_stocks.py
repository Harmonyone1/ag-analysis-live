#!/usr/bin/env python
"""Scan Stocks"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
all_names = list(instruments['name'])
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Major US stocks to look for
stocks = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC',
    'NFLX', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'SQ', 'SHOP', 'UBER', 'ABNB', 'SNAP',
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY',
    # Consumer
    'WMT', 'COST', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'DIS',
    # Energy
    'XOM', 'CVX', 'COP',
    # Other
    'BA', 'CAT', 'MMM', 'IBM', 'COIN', 'GME', 'AMC'
]

# Find available stocks
available = [s for s in stocks if s in all_names]

# Also search for stocks with common suffixes
for name in all_names:
    base = name.replace('.US', '').replace('.NYSE', '').replace('.NAS', '')
    if base in stocks and name not in available:
        available.append(name)

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
        if len(d1) < 10 or len(h4) < 10 or len(h1) < 14:
            return None

        price = api.get_latest_asking_price(inst_id)

        # HTF Bias
        d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-5] else 'BEAR'
        h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-3] else 'BEAR'
        htf_aligned = d1_bias == h4_bias
        htf_bias = d1_bias if htf_aligned else 'MIX'

        # RSI
        closes = h1['c'].values[-15:]
        if len(closes) < 5:
            return None
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        rsi = 100 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))

        # Range
        high_20 = max(h4['h'].values[-10:])
        low_20 = min(h4['l'].values[-10:])
        range_pct = (price - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50

        zone = 'DISC' if range_pct < 30 else ('PREM' if range_pct > 70 else 'EQ')

        # Check setups
        if htf_bias == 'BULL' and zone == 'DISC' and rsi < 40:
            status = '*** VALID LONG ***'
        elif htf_bias == 'BEAR' and zone == 'PREM' and rsi > 60:
            status = '*** VALID SHORT ***'
        elif htf_bias == 'BULL' and zone == 'DISC':
            status = 'LONG potential (RSI %.0f)' % rsi
        elif htf_bias == 'BEAR' and zone == 'PREM':
            status = 'SHORT potential (RSI %.0f)' % rsi
        elif htf_bias == 'BEAR' and zone == 'DISC' and rsi < 35:
            status = 'Oversold (counter-trend)'
        elif htf_bias == 'MIX':
            status = 'HTF mixed'
        elif zone == 'EQ':
            status = 'Equilibrium'
        else:
            status = ''

        return {
            'symbol': symbol,
            'price': price,
            'htf': htf_bias,
            'rsi': rsi,
            'zone': zone,
            'range_pct': range_pct,
            'status': status
        }
    except Exception as e:
        return None

print('=' * 70)
print('STOCKS SCAN | %s | Balance: $340.65' % datetime.now().strftime('%I:%M %p'))
print('=' * 70)
print()
print('Searching for available stock CFDs...')
print('Found: %d stocks' % len(available))
print()
print('%-10s %10s %6s %6s %5s %5s %s' % ('STOCK', 'PRICE', 'HTF', 'RSI', 'ZONE', '%RNG', 'STATUS'))
print('-' * 70)

count = 0
for symbol in available[:25]:  # Limit to 25
    result = analyze(symbol)
    if result:
        count += 1
        print('%-10s %10.2f %6s %6.1f %5s %5.0f %s' % (
            result['symbol'], result['price'], result['htf'],
            result['rsi'], result['zone'], result['range_pct'], result['status']
        ))

if count == 0:
    print('No stock CFDs found or market closed')
    print()
    print('Available instruments containing stock-like names:')
    for name in all_names[:50]:
        if any(c.isupper() and len(name) <= 6 for c in name):
            print('  %s' % name)

print()
print('Note: Stock CFDs trade during US market hours (9:30 AM - 4 PM EST)')
print('=' * 70)
