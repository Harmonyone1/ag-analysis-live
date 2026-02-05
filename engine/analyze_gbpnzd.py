"""Comprehensive GBPNZD Analysis"""
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time

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

def get_trend(closes, ma_period):
    if len(closes) < ma_period:
        return 'UNKNOWN', 0, 0
    ma = np.mean(closes[-ma_period:])
    price = closes[-1]
    pct = (price - ma) / ma * 100
    return ('BULL' if price > ma else 'BEAR'), pct, ma

def count_candles(o, c, n):
    bulls = sum(1 for i in range(-n, 0) if c[i] > o[i])
    return bulls, n - bulls

print('='*70)
print('GBPNZD - COMPREHENSIVE ANALYSIS')
print('='*70)
print()

inst_id = symbol_to_id['GBPNZD']

mn = api.get_price_history(inst_id, resolution='1M', lookback_period='730D')
time.sleep(0.3)
w1 = api.get_price_history(inst_id, resolution='1W', lookback_period='365D')
time.sleep(0.3)
d1 = api.get_price_history(inst_id, resolution='1D', lookback_period='90D')
time.sleep(0.3)
h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='30D')
time.sleep(0.3)
h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='7D')

price = h1['c'].values[-1]

# Calculate all metrics
mn_rsi = get_rsi(mn['c'].values)
mn_trend, mn_pct, mn_ma = get_trend(mn['c'].values, 6)
mn_high = mn['h'].values[-12:].max()
mn_low = mn['l'].values[-12:].min()

w1_rsi = get_rsi(w1['c'].values)
w1_trend, w1_pct, w1_ma = get_trend(w1['c'].values, 10)
w1_high = w1['h'].values[-8:].max()
w1_low = w1['l'].values[-8:].min()

d1_rsi = get_rsi(d1['c'].values)
d1_trend, d1_pct, d1_ma = get_trend(d1['c'].values, 20)
d1_high = d1['h'].values[-20:].max()
d1_low = d1['l'].values[-20:].min()

h4_rsi = get_rsi(h4['c'].values)
h4_trend, h4_pct, h4_ma = get_trend(h4['c'].values, 20)
h4_high = h4['h'].values[-30:].max()
h4_low = h4['l'].values[-30:].min()

h1_rsi = get_rsi(h1['c'].values)
h1_trend, h1_pct, h1_ma = get_trend(h1['c'].values, 20)
h1_high = h1['h'].values[-24:].max()
h1_low = h1['l'].values[-24:].min()

# Range position
h4_range_pct = (price - h4_low) / (h4_high - h4_low) * 100
d1_range_pct = (price - d1_low) / (d1_high - d1_low) * 100

mn_bulls, mn_bears = count_candles(mn['o'].values, mn['c'].values, 3)
w1_bulls, w1_bears = count_candles(w1['o'].values, w1['c'].values, 4)
d1_bulls, d1_bears = count_candles(d1['o'].values, d1['c'].values, 5)
h4_bulls, h4_bears = count_candles(h4['o'].values, h4['c'].values, 6)

print('CURRENT PRICE: %.5f' % price)
print()

print('='*70)
print('TIMEFRAME ANALYSIS')
print('='*70)
print()

print('MONTHLY')
print('-'*70)
print('  Trend: %s (%.2f%% from 6-MA at %.5f)' % (mn_trend, mn_pct, mn_ma))
print('  RSI: %.1f' % mn_rsi)
print('  Candles: %d BULL / %d BEAR (last 3)' % (mn_bulls, mn_bears))
print('  Range: %.5f - %.5f' % (mn_low, mn_high))
print('  BIAS: LONG ONLY')
print()

print('WEEKLY')
print('-'*70)
print('  Trend: %s (%.2f%% from 10-MA at %.5f)' % (w1_trend, w1_pct, w1_ma))
print('  RSI: %.1f' % w1_rsi)
print('  Candles: %d BULL / %d BEAR (last 4)' % (w1_bulls, w1_bears))
print('  Range: %.5f - %.5f' % (w1_low, w1_high))
if w1_rsi > 55:
    print('  BIAS: LONG')
elif w1_rsi < 45:
    print('  BIAS: SHORT')
else:
    print('  BIAS: NEUTRAL (RSI 45-55)')
print()

print('DAILY')
print('-'*70)
print('  Trend: %s (%.2f%% from 20-MA at %.5f)' % (d1_trend, d1_pct, d1_ma))
rsi_note = ''
if d1_rsi < 30: rsi_note = ' [OVERSOLD]'
elif d1_rsi > 70: rsi_note = ' [OVERBOUGHT]'
print('  RSI: %.1f%s' % (d1_rsi, rsi_note))
print('  Candles: %d BULL / %d BEAR (last 5)' % (d1_bulls, d1_bears))
print('  Range: %.5f - %.5f' % (d1_low, d1_high))
print('  Position in Range: %.0f%%' % d1_range_pct)
print()

print('H4 (Entry TF)')
print('-'*70)
print('  Trend: %s (%.2f%% from 20-MA at %.5f)' % (h4_trend, h4_pct, h4_ma))
rsi_note = ''
if h4_rsi < 30: rsi_note = ' [OVERSOLD] <<<'
elif h4_rsi > 70: rsi_note = ' [OVERBOUGHT]'
print('  RSI: %.1f%s' % (h4_rsi, rsi_note))
print('  Candles: %d BULL / %d BEAR (last 6)' % (h4_bulls, h4_bears))
print('  Range: %.5f - %.5f' % (h4_low, h4_high))
print('  Position in Range: %.0f%%' % h4_range_pct)
if h4_range_pct < 30:
    print('  ZONE: DISCOUNT (good for longs)')
elif h4_range_pct > 70:
    print('  ZONE: PREMIUM (good for shorts)')
else:
    print('  ZONE: EQUILIBRIUM')
print()

print('H1 (Timing)')
print('-'*70)
print('  Trend: %s' % h1_trend)
rsi_note = ''
if h1_rsi < 30: rsi_note = ' [OVERSOLD]'
elif h1_rsi > 70: rsi_note = ' [OVERBOUGHT]'
print('  RSI: %.1f%s' % (h1_rsi, rsi_note))
print('  24H Range: %.5f - %.5f' % (h1_low, h1_high))
print()

print('='*70)
print('KEY LEVELS')
print('='*70)
print()
print('RESISTANCE:')
print('  Monthly High:  %.5f' % mn_high)
print('  Weekly High:   %.5f' % w1_high)
print('  Daily High:    %.5f' % d1_high)
print('  H4 High:       %.5f' % h4_high)
print()
print('SUPPORT:')
print('  H4 Low:        %.5f' % h4_low)
print('  Daily Low:     %.5f' % d1_low)
print('  Weekly Low:    %.5f' % w1_low)
print('  Monthly Low:   %.5f' % mn_low)
print()
print('MOVING AVERAGES:')
print('  Monthly 6-MA:  %.5f' % mn_ma)
print('  Weekly 10-MA:  %.5f' % w1_ma)
print('  Daily 20-MA:   %.5f' % d1_ma)
print('  H4 20-MA:      %.5f' % h4_ma)
print()

print('='*70)
print('SETUP SCORING')
print('='*70)
print()

score = 0
print('HTF Alignment:')
if mn_trend == 'BULL':
    score += 3
    print('  [+3] Monthly BULLISH')
if w1_trend == 'BULL':
    score += 2
    print('  [+2] Weekly BULLISH')
elif w1_rsi > 45:
    score += 1
    print('  [+1] Weekly RSI > 45 (supportive)')
if d1_trend == 'BULL':
    score += 1
    print('  [+1] Daily BULLISH')

print()
print('Entry Signals:')
if h4_rsi < 30:
    score += 3
    print('  [+3] H4 RSI < 30 (oversold)')
elif h4_rsi < 40:
    score += 2
    print('  [+2] H4 RSI < 40')
if h1_rsi < 30:
    score += 2
    print('  [+2] H1 RSI < 30 (oversold)')
elif h1_rsi < 40:
    score += 1
    print('  [+1] H1 RSI < 40')

print()
print('Zone:')
if h4_range_pct < 35:
    score += 1
    print('  [+1] Price in discount zone (%.0f%%)' % h4_range_pct)
else:
    print('  [+0] Price at %.0f%% of range' % h4_range_pct)

print()
print('TOTAL SCORE: %d / 13' % score)
print()

print('='*70)
print('TRADE ASSESSMENT')
print('='*70)
print()

if score >= 10:
    rating = 'STRONG'
elif score >= 7:
    rating = 'VALID'
elif score >= 5:
    rating = 'MODERATE'
else:
    rating = 'WEAK'

print('Setup Rating: %s (%d/13)' % (rating, score))
print()
print('Confluence:')
yn_mn = 'Y' if mn_trend == 'BULL' else 'N'
yn_w1 = 'Y' if w1_trend == 'BULL' else 'N'
yn_h4 = 'Y' if h4_rsi < 35 else 'N'
yn_h1 = 'Y' if h1_rsi < 35 else 'N'
yn_disc = 'Y' if h4_range_pct < 35 else 'N'

print('  [%s] Monthly trend BULLISH' % yn_mn)
print('  [%s] Weekly trend BULLISH' % yn_w1)
print('  [%s] H4 RSI oversold (%.1f)' % (yn_h4, h4_rsi))
print('  [%s] H1 RSI oversold (%.1f)' % (yn_h1, h1_rsi))
print('  [%s] Price in discount (%.0f%%)' % (yn_disc, h4_range_pct))
print()

print('Current Position: LONG 0.50 lots')
print('Risk Management:')
print('  Stop Loss: 2.30754 (below H4 swing low)')
print('  Take Profit: 2.33388 (2:1 R:R)')
risk_pips = (2.31665 - 2.30754) * 10000
reward_pips = (2.33388 - 2.31665) * 10000
print('  Risk: ~%.0f pips | Reward: ~%.0f pips' % (risk_pips, reward_pips))
