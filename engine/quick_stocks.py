#!/usr/bin/env python
"""Quick stock scan - top picks only"""

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

# Top 15 stocks only
STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
          'JPM', 'XOM', 'BAC', 'DIS', 'NFLX', 'AMD', 'COIN', 'BA']

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0.001
    return 100 - (100 / (1 + avg_gain / avg_loss))

print('=' * 75)
print('QUICK STOCK SCAN | %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 75)
print()

results = []

for symbol in STOCKS:
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            continue

        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='14D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='7D')

        if d1 is None or h4 is None or h1 is None:
            continue
        if len(d1) < 5 or len(h4) < 10 or len(h1) < 10:
            continue

        price = api.get_latest_asking_price(inst_id)

        d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-5] else 'BEAR'
        h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-5] else 'BEAR'
        htf_aligned = d1_bias == h4_bias

        rsi = calc_rsi(h1['c'].values)

        high_20 = max(h4['h'].values[-20:])
        low_20 = min(h4['l'].values[-20:])
        range_pct = (price - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50

        zone = 'DISC' if range_pct < 30 else ('PREM' if range_pct > 70 else 'EQ')

        # Check for setups
        valid_long = htf_aligned and d1_bias == 'BULL' and zone == 'DISC' and rsi < 40
        valid_short = htf_aligned and d1_bias == 'BEAR' and zone == 'PREM' and rsi > 60

        status = ''
        if valid_long:
            status = '>>> VALID LONG <<<'
        elif valid_short:
            status = '>>> VALID SHORT <<<'
        elif htf_aligned and d1_bias == 'BULL' and zone == 'DISC':
            status = 'LONG dev (RSI %.0f)' % rsi
        elif htf_aligned and d1_bias == 'BEAR' and zone == 'PREM':
            status = 'SHORT dev (RSI %.0f)' % rsi

        results.append({
            'symbol': symbol,
            'price': price,
            'd1': d1_bias[0],
            'h4': h4_bias[0],
            'aligned': 'Y' if htf_aligned else 'N',
            'rsi': rsi,
            'zone': zone,
            'pct': range_pct,
            'status': status
        })

        time.sleep(0.25)
    except:
        continue

print('%-6s %10s %3s %3s %5s %5s %5s %s' % (
    'STOCK', 'PRICE', 'D1', 'H4', 'RSI', 'ZONE', '%', 'STATUS'))
print('-' * 75)

for r in results:
    print('%-6s %10.2f  %s   %s  %5.1f %5s %5.0f %s' % (
        r['symbol'], r['price'], r['d1'], r['h4'],
        r['rsi'], r['zone'], r['pct'], r['status']))

print()
print('Scanned %d stocks' % len(results))
