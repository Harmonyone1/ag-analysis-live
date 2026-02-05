#!/usr/bin/env python
"""Scan Forex Minors with Volume Analysis"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Forex Minors (crosses without USD)
MINORS = [
    # EUR crosses
    'EURGBP', 'EURCHF', 'EURAUD', 'EURNZD', 'EURCAD',
    # GBP crosses
    'GBPCHF', 'GBPAUD', 'GBPNZD', 'GBPCAD',
    # AUD crosses
    'AUDNZD', 'AUDCAD', 'AUDCHF',
    # NZD crosses
    'NZDCHF', 'NZDCAD',
    # CAD crosses
    'CADCHF',
    # CHF cross
    'CHFJPY',
]

# All pairs for volume comparison
ALL_PAIRS = [
    # Majors
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
    # JPY crosses
    'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY',
    # Minors
    'EURGBP', 'EURCHF', 'EURAUD', 'EURNZD', 'EURCAD',
    'GBPCHF', 'GBPAUD', 'GBPNZD', 'GBPCAD',
    'AUDNZD', 'AUDCAD', 'AUDCHF',
    'NZDCHF', 'NZDCAD', 'CADCHF',
]

def analyze_with_volume(symbol):
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            return None

        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='7D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='3D')

        if d1 is None or h4 is None or h1 is None:
            return None
        if len(d1) < 5 or len(h4) < 5 or len(h1) < 10:
            return None

        price = api.get_latest_asking_price(inst_id)

        # HTF Bias
        d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-min(5, len(d1))] else 'BEAR'
        h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-min(5, len(h4))] else 'BEAR'
        htf_aligned = d1_bias == h4_bias
        htf_bias = d1_bias if htf_aligned else 'MIX'

        # RSI
        closes = h1['c'].values[-15:]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0.001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))

        # Range
        high_20 = max(h4['h'].values[-min(20, len(h4)):])
        low_20 = min(h4['l'].values[-min(20, len(h4)):])
        range_pct = (price - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50

        zone = 'DISC' if range_pct < 30 else ('PREM' if range_pct > 70 else 'EQ')

        # Volume analysis (using candle range as proxy for activity)
        # Higher range = more volatility/activity
        h1_ranges = [(h1['h'].iloc[i] - h1['l'].iloc[i]) for i in range(-10, 0)]
        avg_range = sum(h1_ranges) / len(h1_ranges)
        recent_range = sum(h1_ranges[-3:]) / 3
        volume_increase = recent_range / avg_range if avg_range > 0 else 1.0

        # Daily volume proxy
        if 'v' in d1.columns:
            daily_vol = d1['v'].iloc[-1]
            avg_daily_vol = d1['v'].mean()
            vol_ratio = daily_vol / avg_daily_vol if avg_daily_vol > 0 else 1.0
        else:
            daily_vol = 0
            vol_ratio = volume_increase  # Use range as proxy

        # Check setup
        signal = None
        alert = 0

        if htf_bias == 'BULL' and zone == 'DISC' and rsi < 40:
            signal = 'LONG'
            alert = 3
        elif htf_bias == 'BEAR' and zone == 'PREM' and rsi > 60:
            signal = 'SHORT'
            alert = 3
        elif htf_bias == 'BULL' and zone == 'DISC':
            signal = 'LONG'
            alert = 2
        elif htf_bias == 'BEAR' and zone == 'PREM':
            signal = 'SHORT'
            alert = 2

        return {
            'symbol': symbol,
            'price': price,
            'htf': htf_bias,
            'rsi': rsi,
            'zone': zone,
            'pct': range_pct,
            'signal': signal,
            'alert': alert,
            'vol_ratio': vol_ratio,
            'activity': volume_increase
        }
    except:
        return None

print('=' * 80)
print('FOREX MINORS + VOLUME ANALYSIS | %s' % datetime.now().strftime('%I:%M %p'))
print('=' * 80)
print()

# First, get volume/activity for all pairs
print('=== VOLUME/ACTIVITY RANKING (All Forex) ===')
print('Higher activity ratio = more movement recently')
print()

all_results = []
for sym in ALL_PAIRS:
    r = analyze_with_volume(sym)
    if r:
        all_results.append(r)

# Sort by activity (volatility proxy)
all_results.sort(key=lambda x: x['activity'], reverse=True)

print('%-8s %10s %6s %5s %6s %8s' % ('PAIR', 'PRICE', 'HTF', 'RSI', 'ZONE', 'ACTIVITY'))
print('-' * 55)
for r in all_results[:15]:
    activity_bar = '*' * min(int(r['activity'] * 3), 10)
    print('%-8s %10.5f %6s %5.1f %6s %6.2fx %s' % (
        r['symbol'], r['price'], r['htf'], r['rsi'], r['zone'], r['activity'], activity_bar
    ))

print()
print('=' * 80)
print('=== FOREX MINORS DETAILED ANALYSIS ===')
print('=' * 80)
print()
print('[3]=VALID | [2]=DEVELOPING | RSI: LONG<40, SHORT>60')
print()
print('%-8s %10s %6s %6s %5s %5s %8s %s' % (
    'PAIR', 'PRICE', 'HTF', 'RSI', 'ZONE', '%RNG', 'ACTIVITY', 'STATUS'))
print('-' * 80)

minor_results = []
for sym in MINORS:
    r = analyze_with_volume(sym)
    if r:
        minor_results.append(r)

        lvl = '[%d]' % r['alert'] if r['alert'] > 0 else '   '
        sig = r['signal'] if r['signal'] else ''

        # Highlight high activity
        act_str = '%.2fx' % r['activity']
        if r['activity'] > 1.5:
            act_str = '**%.2fx**' % r['activity']

        print('%s %-8s %10.5f %6s %6.1f %5s %5.0f %8s %s' % (
            lvl, r['symbol'], r['price'], r['htf'], r['rsi'], r['zone'], r['pct'], act_str, sig
        ))

print()
print('=' * 80)
print('SUMMARY')
print('=' * 80)

# Valid setups
valid = [r for r in minor_results if r['alert'] == 3]
developing = [r for r in minor_results if r['alert'] == 2]
high_activity = [r for r in minor_results if r['activity'] > 1.3]

if valid:
    print()
    print('*** VALID SETUPS IN MINORS ***')
    for r in valid:
        print('  >>> %s %s | RSI: %.1f | %s %.0f%% | Activity: %.2fx' % (
            r['symbol'], r['signal'], r['rsi'], r['zone'], r['pct'], r['activity']))

if developing:
    print()
    print('DEVELOPING SETUPS:')
    for r in developing:
        need = '<40' if r['signal'] == 'LONG' else '>60'
        print('  [2] %s %s | RSI: %.1f (need %s) | Activity: %.2fx' % (
            r['symbol'], r['signal'], r['rsi'], need, r['activity']))

if high_activity:
    print()
    print('HIGH ACTIVITY PAIRS (worth watching):')
    for r in high_activity:
        print('  %s | Activity: %.2fx | HTF: %s | RSI: %.1f' % (
            r['symbol'], r['activity'], r['htf'], r['rsi']))

if not valid and not developing:
    print()
    print('No actionable setups in minors currently.')

print()
print('=' * 80)
