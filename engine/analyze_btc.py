#!/usr/bin/env python
"""Analyze BTCUSD for trading opportunities"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_trend(prices):
    if len(prices) < 50:
        return 'NEUTRAL', 0, 0
    ma20 = np.mean(prices[-20:])
    ma50 = np.mean(prices[-50:])
    current = prices[-1]
    if current > ma20 > ma50:
        return 'BULL', ma20, ma50
    elif current < ma20 < ma50:
        return 'BEAR', ma20, ma50
    return 'NEUTRAL', ma20, ma50

print('=' * 70)
print('BTCUSD ANALYSIS | ICT+Quant Framework')
print('=' * 70)
print()

# Get account status
state = api.get_account_state()
print('Account: Balance $%.2f | Equity $%.2f' % (state['balance'], state['balance'] + state['openGrossPnL']))
print()

inst_id = name_to_id.get('BTCUSD')
if not inst_id:
    print('BTCUSD not found')
    exit()

# Get current price
current_price = api.get_latest_asking_price(inst_id)
print('Current Price: $%.2f' % current_price)
print()

print('MULTI-TIMEFRAME ANALYSIS:')
print('-' * 70)

# Daily
d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='6M')
if d1 is not None and len(d1) > 50:
    d1_trend, d1_ma20, d1_ma50 = get_trend(d1['c'].values)
    d1_rsi = calculate_rsi(d1['c'].values)
    d1_high = max(d1['h'].values)
    d1_low = min(d1['l'].values)
    print('Daily (D1):')
    print('  Trend: %s | MA20: $%.2f | MA50: $%.2f' % (d1_trend, d1_ma20, d1_ma50))
    print('  RSI: %.1f | Range: $%.2f - $%.2f' % (d1_rsi, d1_low, d1_high))
    print()

# 4H
h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='1M')
if h4 is not None and len(h4) > 50:
    h4_trend, h4_ma20, h4_ma50 = get_trend(h4['c'].values)
    h4_rsi = calculate_rsi(h4['c'].values)
    print('4-Hour (H4):')
    print('  Trend: %s | MA20: $%.2f | MA50: $%.2f' % (h4_trend, h4_ma20, h4_ma50))
    print('  RSI: %.1f' % h4_rsi)
    print()

# 1H
h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='2W')
if h1 is not None and len(h1) > 50:
    h1_trend, h1_ma20, h1_ma50 = get_trend(h1['c'].values)
    h1_rsi = calculate_rsi(h1['c'].values)
    print('1-Hour (H1):')
    print('  Trend: %s | MA20: $%.2f | MA50: $%.2f' % (h1_trend, h1_ma20, h1_ma50))
    print('  RSI: %.1f' % h1_rsi)
    print()

# Calculate discount/premium zone
if d1 is not None:
    high_52 = max(d1['h'].values[-260:]) if len(d1) >= 260 else max(d1['h'].values)
    low_52 = min(d1['l'].values[-260:]) if len(d1) >= 260 else min(d1['l'].values)
    range_pct = ((current_price - low_52) / (high_52 - low_52)) * 100 if high_52 != low_52 else 50

    if range_pct < 30:
        zone = 'DISCOUNT'
    elif range_pct > 70:
        zone = 'PREMIUM'
    else:
        zone = 'EQUILIBRIUM'

    print('-' * 70)
    print('PRICE ZONE:')
    print('  52-Week Range: $%.2f - $%.2f' % (low_52, high_52))
    print('  Current Position: %.1f%% [%s ZONE]' % (range_pct, zone))
    print()

# Framework check
print('=' * 70)
print('ICT+QUANT FRAMEWORK CHECK')
print('=' * 70)
print()

htf_aligned_bull = d1_trend == 'BULL' and h4_trend == 'BULL'
htf_aligned_bear = d1_trend == 'BEAR' and h4_trend == 'BEAR'

print('HTF Alignment:')
print('  D1: %s | H4: %s' % (d1_trend, h4_trend))
if htf_aligned_bull:
    print('  Status: BULL ALIGNED')
elif htf_aligned_bear:
    print('  Status: BEAR ALIGNED')
else:
    print('  Status: MIXED (No alignment)')
print()

print('Zone Check:')
print('  Current: %s (%.1f%%)' % (zone, range_pct))
print('  For LONG: Need DISCOUNT (<30%%)')
print('  For SHORT: Need PREMIUM (>70%%)')
print()

print('RSI Check:')
print('  H4 RSI: %.1f' % h4_rsi)
print('  For LONG: Need RSI < 40')
print('  For SHORT: Need RSI > 60')
print()

# Final verdict
print('=' * 70)
print('VERDICT')
print('=' * 70)
print()

valid_long = htf_aligned_bull and zone == 'DISCOUNT' and h4_rsi < 40
valid_short = htf_aligned_bear and zone == 'PREMIUM' and h4_rsi > 60

if valid_long:
    print('>>> VALID LONG SETUP <<<')
    print()
    print('Entry: $%.2f' % current_price)
    sl = current_price * 0.97
    tp = current_price * 1.06
    print('Suggested SL: $%.2f (3%%)' % sl)
    print('Suggested TP: $%.2f (6%%)' % tp)
    print('R:R = 1:2')
elif valid_short:
    print('>>> VALID SHORT SETUP <<<')
    print()
    print('Entry: $%.2f' % current_price)
    sl = current_price * 1.03
    tp = current_price * 0.94
    print('Suggested SL: $%.2f (3%%)' % sl)
    print('Suggested TP: $%.2f (6%%)' % tp)
    print('R:R = 1:2')
else:
    print('NO VALID SETUP')
    print()
    reasons = []
    if not htf_aligned_bull and not htf_aligned_bear:
        reasons.append('HTF not aligned (D1: %s, H4: %s)' % (d1_trend, h4_trend))
    if zone == 'EQUILIBRIUM':
        reasons.append('Price in equilibrium zone (%.1f%%)' % range_pct)
    if zone == 'PREMIUM' and not htf_aligned_bear:
        reasons.append('In premium but HTF not bearish')
    if zone == 'DISCOUNT' and not htf_aligned_bull:
        reasons.append('In discount but HTF not bullish')
    if 40 <= h4_rsi <= 60:
        reasons.append('RSI neutral (%.1f)' % h4_rsi)

    if reasons:
        print('Reasons:')
        for r in reasons:
            print('  - %s' % r)

print()
print('=' * 70)
