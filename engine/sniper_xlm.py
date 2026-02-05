"""XLMUSD Sniper Entry Analysis"""
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

def get_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.0001
    return 100 - (100 / (1 + avg_gain/avg_loss))

def find_obs(opens, closes, highs, lows, lookback=50):
    bearish_obs = []
    bullish_obs = []

    o, c, h, l = opens[-lookback:], closes[-lookback:], highs[-lookback:], lows[-lookback:]

    for i in range(2, len(o)-3):
        if c[i] > o[i]:
            if c[i+1] < o[i+1] and c[i+2] < o[i+2]:
                if l[i+2] < l[i]:
                    bearish_obs.append({'high': h[i], 'low': l[i], 'idx': i})
        if c[i] < o[i]:
            if c[i+1] > o[i+1] and c[i+2] > o[i+2]:
                if h[i+2] > h[i]:
                    bullish_obs.append({'high': h[i], 'low': l[i], 'idx': i})

    return bearish_obs, bullish_obs

def find_fvg(highs, lows, lookback=30):
    fvgs = []
    h, l = highs[-lookback:], lows[-lookback:]

    for i in range(1, len(h)-1):
        if l[i+1] > h[i-1]:
            fvgs.append({'type': 'bullish', 'top': l[i+1], 'bottom': h[i-1], 'idx': i})
        if h[i+1] < l[i-1]:
            fvgs.append({'type': 'bearish', 'top': l[i-1], 'bottom': h[i+1], 'idx': i})

    return fvgs

inst_id = symbol_to_id['XLMUSD']

mn = api.get_price_history(inst_id, resolution='1M', lookback_period='365D')
w1 = api.get_price_history(inst_id, resolution='1W', lookback_period='180D')
d1 = api.get_price_history(inst_id, resolution='1D', lookback_period='60D')
h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='21D')
h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='7D')
m15 = api.get_price_history(inst_id, resolution='15M', lookback_period='3D')

price = h1['c'].values[-1]

print('='*70)
print('XLMUSD - SNIPER ENTRY ANALYSIS')
print('='*70)
print()
print('CURRENT PRICE: $%.4f' % price)
print()

# HTF Analysis
print('HTF BIAS:')
print('-'*70)
mn_ma = np.mean(mn['c'].values[-6:])
w1_ma = np.mean(w1['c'].values[-10:])
d1_ma = np.mean(d1['c'].values[-20:])

mn_trend = 'BULL' if price > mn_ma else 'BEAR'
w1_trend = 'BULL' if price > w1_ma else 'BEAR'
d1_trend = 'BULL' if price > d1_ma else 'BEAR'

print('  Monthly: %s (MA: $%.4f)' % (mn_trend, mn_ma))
print('  Weekly:  %s (MA: $%.4f)' % (w1_trend, w1_ma))
print('  Daily:   %s (MA: $%.4f)' % (d1_trend, d1_ma))
print()

# RSI
print('RSI ANALYSIS:')
print('-'*70)
d1_rsi = get_rsi(d1['c'].values)
h4_rsi = get_rsi(h4['c'].values)
h1_rsi = get_rsi(h1['c'].values)
m15_rsi = get_rsi(m15['c'].values)

d1_sig = ' [OVERSOLD]' if d1_rsi < 30 else ''
h4_sig = ' [OVERSOLD]' if h4_rsi < 30 else ''
h1_sig = ' [OVERSOLD]' if h1_rsi < 30 else ''
m15_sig = ' [OVERSOLD]' if m15_rsi < 30 else ''

print('  Daily:  %.1f%s' % (d1_rsi, d1_sig))
print('  H4:     %.1f%s' % (h4_rsi, h4_sig))
print('  H1:     %.1f%s' % (h1_rsi, h1_sig))
print('  M15:    %.1f%s' % (m15_rsi, m15_sig))
print()

# Key levels
print('KEY LEVELS:')
print('-'*70)
h4_high = h4['h'].values.max()
h4_low = h4['l'].values.min()
d1_high = d1['h'].values[-20:].max()
d1_low = d1['l'].values[-20:].min()
h1_high = h1['h'].values[-48:].max()
h1_low = h1['l'].values[-48:].min()

print('  D1 Range (20d):  $%.4f - $%.4f' % (d1_low, d1_high))
print('  H4 Range:        $%.4f - $%.4f' % (h4_low, h4_high))
print('  H1 Range (48h):  $%.4f - $%.4f' % (h1_low, h1_high))
print()

# Order Blocks H4
print('ORDER BLOCKS (H4):')
print('-'*70)
minus_obs, plus_obs = find_obs(h4['o'].values, h4['c'].values, h4['h'].values, h4['l'].values)

latest_plus = None
if plus_obs:
    latest_plus = plus_obs[-1]
    print('  +OB (Bullish): $%.4f - $%.4f' % (latest_plus['low'], latest_plus['high']))
    if price > latest_plus['high']:
        print('    Status: MITIGATED')
    elif price >= latest_plus['low']:
        print('    Status: TESTING NOW')
    else:
        print('    Status: Below - potential support')
else:
    print('  +OB: None found')

latest_minus = None
if minus_obs:
    latest_minus = minus_obs[-1]
    print('  -OB (Bearish): $%.4f - $%.4f' % (latest_minus['low'], latest_minus['high']))
    if price < latest_minus['low']:
        print('    Status: MITIGATED')
    else:
        print('    Status: Target above')
else:
    print('  -OB: None found')
print()

# Order Blocks H1
print('ORDER BLOCKS (H1 - Precision):')
print('-'*70)
h1_minus, h1_plus = find_obs(h1['o'].values, h1['c'].values, h1['h'].values, h1['l'].values, lookback=80)

if h1_plus:
    for ob in h1_plus[-3:]:
        if price > ob['high']:
            status = 'MITIGATED'
        elif price >= ob['low']:
            status = 'TESTING'
        else:
            status = 'BELOW'
        print('  +OB: $%.4f - $%.4f [%s]' % (ob['low'], ob['high'], status))
else:
    print('  +OB: None')

if h1_minus:
    for ob in h1_minus[-2:]:
        status = 'MITIGATED' if price < ob['low'] else 'TARGET'
        print('  -OB: $%.4f - $%.4f [%s]' % (ob['low'], ob['high'], status))
print()

# FVGs
print('FAIR VALUE GAPS (H1):')
print('-'*70)
fvgs = find_fvg(h1['h'].values, h1['l'].values, lookback=50)
bullish_fvgs = [f for f in fvgs if f['type'] == 'bullish' and f['bottom'] < price]
bearish_fvgs = [f for f in fvgs if f['type'] == 'bearish' and f['top'] > price]

if bullish_fvgs:
    for fvg in bullish_fvgs[-2:]:
        print('  Bullish FVG: $%.4f - $%.4f' % (fvg['bottom'], fvg['top']))
else:
    print('  No unfilled bullish FVGs below')

if bearish_fvgs:
    for fvg in bearish_fvgs[-2:]:
        print('  Bearish FVG: $%.4f - $%.4f' % (fvg['bottom'], fvg['top']))
print()

# Liquidity
print('LIQUIDITY TARGETS:')
print('-'*70)
swing_lows = []
for i in range(2, len(h1['l'].values)-2):
    if h1['l'].values[i] < h1['l'].values[i-1] and h1['l'].values[i] < h1['l'].values[i-2]:
        if h1['l'].values[i] < h1['l'].values[i+1] and h1['l'].values[i] < h1['l'].values[i+2]:
            swing_lows.append(h1['l'].values[i])

swing_highs = []
for i in range(2, len(h1['h'].values)-2):
    if h1['h'].values[i] > h1['h'].values[i-1] and h1['h'].values[i] > h1['h'].values[i-2]:
        if h1['h'].values[i] > h1['h'].values[i+1] and h1['h'].values[i] > h1['h'].values[i+2]:
            swing_highs.append(h1['h'].values[i])

ssl_below = [s for s in swing_lows if s < price]
bsl_above = [s for s in swing_highs if s > price]

if ssl_below:
    print('  Sell-side (below): $%.4f' % min(ssl_below))
if bsl_above:
    print('  Buy-side (above):  $%.4f' % max(bsl_above))
print()

# Sniper Entry
print('='*70)
print('SNIPER ENTRY SETUP')
print('='*70)
print()

entry_zones = []
if h1_plus:
    for ob in h1_plus:
        if ob['low'] < price and ob['high'] >= price * 0.98:
            entry_zones.append({'type': '+OB', 'low': ob['low'], 'high': ob['high']})

if bullish_fvgs:
    for fvg in bullish_fvgs:
        entry_zones.append({'type': 'FVG', 'low': fvg['bottom'], 'high': fvg['top']})

# Best entry
if entry_zones:
    best = max(entry_zones, key=lambda x: x['low'])
    entry = best['high']
    sl = best['low'] * 0.99
    entry_type = best['type']
else:
    entry = price
    sl = h1_low * 0.99
    entry_type = 'Market'
    best = None

# TPs
tp1 = h1_high
tp2 = d1_high
if bsl_above:
    tp1 = max(bsl_above)

# R:R
risk = entry - sl
reward1 = tp1 - entry
reward2 = tp2 - entry
rr1 = reward1 / risk if risk > 0 else 0
rr2 = reward2 / risk if risk > 0 else 0

print('DIRECTION: LONG')
print()
print('ENTRY: $%.4f' % entry)
if entry != price:
    print('  (Limit order - wait for price to reach)')
print('  Entry Type: %s' % entry_type)
print()
print('STOP LOSS: $%.4f' % sl)
print('  Risk: %.2f%%' % ((entry - sl) / entry * 100))
print()
print('TAKE PROFIT 1: $%.4f (R:R = %.1f:1)' % (tp1, rr1))
print('TAKE PROFIT 2: $%.4f (R:R = %.1f:1)' % (tp2, rr2))
print()

# Position sizing
account = 300
risk_pct = 0.02
risk_amount = account * risk_pct
pip_risk = abs(entry - sl)
lots = risk_amount / (pip_risk * 10000) if pip_risk > 0 else 0.01
lots = min(lots, 0.5)
lots = max(lots, 0.01)

print('POSITION SIZE (2%% risk on $%d):' % account)
print('  Suggested: %.2f lots' % lots)
print()

print('ENTRY TRIGGERS:')
print('-'*70)
print('  1. H1 bullish engulfing or hammer at support')
print('  2. M15 break of structure (higher high)')
print('  3. RSI divergence on M15/H1')
print()

print('CONFLUENCE SCORE:')
print('-'*70)
score = 0
factors = []

if d1_rsi < 30:
    score += 2
    factors.append('D1 RSI oversold (%.1f)' % d1_rsi)
if h4_rsi < 30:
    score += 2
    factors.append('H4 RSI oversold (%.1f)' % h4_rsi)
if h1_rsi < 35:
    score += 1
    factors.append('H1 RSI low (%.1f)' % h1_rsi)
if price < d1_ma:
    score += 1
    factors.append('Below D1 MA (discount)')
if entry_zones:
    score += 2
    factors.append('Entry at %s zone' % entry_type)
if rr1 >= 2:
    score += 1
    factors.append('R:R >= 2:1')

print('  Score: %d/9' % score)
for f in factors:
    print('  [+] %s' % f)
print()

if score >= 6:
    print('>>> HIGH PROBABILITY SETUP <<<')
elif score >= 4:
    print('>>> MODERATE SETUP - Wait for confirmation <<<')
else:
    print('>>> LOW PROBABILITY - Consider passing <<<')
