#!/usr/bin/env python
"""Quick EURJPY Analysis for Trade Decision"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

symbol = 'EURJPY'
inst_id = name_to_id.get(symbol)

h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='14D')
h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='7D')

price = api.get_latest_asking_price(inst_id)

# Key levels
recent_high = max(h4['h'].values[-10:])
recent_low = min(h4['l'].values[-10:])
range_size = recent_high - recent_low

# RSI
closes = h1['c'].values[-15:]
deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
gains = [d if d > 0 else 0 for d in deltas]
losses = [-d if d < 0 else 0 for d in deltas]
avg_gain = sum(gains) / len(gains)
avg_loss = sum(losses) / len(losses) if sum(losses) > 0 else 0.001
rsi = 100 - (100 / (1 + avg_gain / avg_loss))

# Position in range
range_pct = (price - recent_low) / range_size * 100 if range_size > 0 else 50

print('EURJPY Analysis')
print('=' * 50)
print('Price: %.3f' % price)
print('RSI (1H): %.1f' % rsi)
print('Range: %.3f - %.3f' % (recent_low, recent_high))
print('Position: %.1f%% of range' % range_pct)
print()

# Trade setup for SHORT
sl_price = recent_high + 0.15  # 15 pips above high
tp1 = price - (range_size * 0.3)  # 30% pullback
tp2 = price - (range_size * 0.5)  # 50% pullback

sl_pips = (sl_price - price) * 100
tp1_pips = (price - tp1) * 100
tp2_pips = (price - tp2) * 100

print('SHORT Setup:')
print('  Entry: %.3f' % price)
print('  SL: %.3f (%.1f pips)' % (sl_price, sl_pips))
print('  TP1: %.3f (%.1f pips) R:R 1:%.1f' % (tp1, tp1_pips, tp1_pips/sl_pips))
print('  TP2: %.3f (%.1f pips) R:R 1:%.1f' % (tp2, tp2_pips, tp2_pips/sl_pips))
print()

# Confidence assessment
confidence = 50
if rsi > 85:
    confidence += 15
if range_pct > 95:
    confidence += 15
if rsi > 80 and range_pct > 90:
    confidence += 10

print('Confidence: %d%%' % confidence)
print('Note: Counter-trend trade (HTF is BULL)')
