"""Comprehensive ETHUSD Analysis"""
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
print('ETHUSD - COMPREHENSIVE ANALYSIS')
print('='*70)
print()

inst_id = symbol_to_id['ETHUSD']

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

h4_range_pct = (price - h4_low) / (h4_high - h4_low) * 100
d1_range_pct = (price - d1_low) / (d1_high - d1_low) * 100
w1_range_pct = (price - w1_low) / (w1_high - w1_low) * 100

mn_bulls, mn_bears = count_candles(mn['o'].values, mn['c'].values, 3)
w1_bulls, w1_bears = count_candles(w1['o'].values, w1['c'].values, 4)
d1_bulls, d1_bears = count_candles(d1['o'].values, d1['c'].values, 5)
h4_bulls, h4_bears = count_candles(h4['o'].values, h4['c'].values, 6)

print('CURRENT PRICE: $%.2f' % price)
print()

print('='*70)
print('TIMEFRAME ANALYSIS')
print('='*70)
print()

print('MONTHLY')
print('-'*70)
print('  Trend: %s (%.2f%% from 6-MA at $%.2f)' % (mn_trend, mn_pct, mn_ma))
print('  RSI: %.1f' % mn_rsi)
print('  Candles: %d BULL / %d BEAR (last 3)' % (mn_bulls, mn_bears))
print('  12M Range: $%.2f - $%.2f' % (mn_low, mn_high))
if mn_trend == 'BULL':
    print('  BIAS: LONG ONLY')
else:
    print('  BIAS: SHORT ONLY')
print()

print('WEEKLY')
print('-'*70)
print('  Trend: %s (%.2f%% from 10-MA at $%.2f)' % (w1_trend, w1_pct, w1_ma))
print('  RSI: %.1f' % w1_rsi)
print('  Candles: %d BULL / %d BEAR (last 4)' % (w1_bulls, w1_bears))
print('  8W Range: $%.2f - $%.2f' % (w1_low, w1_high))
print('  Position in Range: %.0f%%' % w1_range_pct)
if w1_rsi > 55:
    print('  BIAS: LONG')
elif w1_rsi < 45:
    print('  BIAS: SHORT')
else:
    print('  BIAS: NEUTRAL')
print()

print('DAILY')
print('-'*70)
print('  Trend: %s (%.2f%% from 20-MA at $%.2f)' % (d1_trend, d1_pct, d1_ma))
rsi_note = ''
if d1_rsi < 30: rsi_note = ' [OVERSOLD]'
elif d1_rsi > 70: rsi_note = ' [OVERBOUGHT]'
print('  RSI: %.1f%s' % (d1_rsi, rsi_note))
print('  Candles: %d BULL / %d BEAR (last 5)' % (d1_bulls, d1_bears))
print('  20D Range: $%.2f - $%.2f' % (d1_low, d1_high))
print('  Position in Range: %.0f%%' % d1_range_pct)
print()

print('H4 (Entry TF)')
print('-'*70)
print('  Trend: %s (%.2f%% from 20-MA at $%.2f)' % (h4_trend, h4_pct, h4_ma))
rsi_note = ''
if h4_rsi < 30: rsi_note = ' [OVERSOLD] <<<'
elif h4_rsi > 70: rsi_note = ' [OVERBOUGHT] <<<'
print('  RSI: %.1f%s' % (h4_rsi, rsi_note))
print('  Candles: %d BULL / %d BEAR (last 6)' % (h4_bulls, h4_bears))
print('  Range: $%.2f - $%.2f' % (h4_low, h4_high))
print('  Position in Range: %.0f%%' % h4_range_pct)
if h4_range_pct < 30:
    print('  ZONE: DISCOUNT')
elif h4_range_pct > 70:
    print('  ZONE: PREMIUM')
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
print('  24H Range: $%.2f - $%.2f' % (h1_low, h1_high))
print()

print('='*70)
print('KEY LEVELS')
print('='*70)
print()
print('RESISTANCE:')
print('  Monthly High:  $%.2f' % mn_high)
print('  Weekly High:   $%.2f' % w1_high)
print('  Daily High:    $%.2f' % d1_high)
print('  H4 High:       $%.2f' % h4_high)
print()
print('SUPPORT:')
print('  H4 Low:        $%.2f' % h4_low)
print('  Daily Low:     $%.2f' % d1_low)
print('  Weekly Low:    $%.2f' % w1_low)
print('  Monthly Low:   $%.2f' % mn_low)
print()
print('MOVING AVERAGES:')
print('  Monthly 6-MA:  $%.2f' % mn_ma)
print('  Weekly 10-MA:  $%.2f' % w1_ma)
print('  Daily 20-MA:   $%.2f' % d1_ma)
print('  H4 20-MA:      $%.2f' % h4_ma)
print()

print('='*70)
print('SETUP SCORING - LONG')
print('='*70)
print()

long_score = 0
print('HTF Alignment:')
if mn_trend == 'BULL':
    long_score += 3
    print('  [+3] Monthly BULLISH')
else:
    print('  [+0] Monthly BEARISH')
if w1_trend == 'BULL':
    long_score += 2
    print('  [+2] Weekly BULLISH')
elif w1_rsi > 45:
    long_score += 1
    print('  [+1] Weekly RSI > 45')
else:
    print('  [+0] Weekly not supportive')
if d1_trend == 'BULL':
    long_score += 1
    print('  [+1] Daily BULLISH')
else:
    print('  [+0] Daily BEARISH')

print()
print('Entry Signals:')
if h4_rsi < 30:
    long_score += 3
    print('  [+3] H4 RSI < 30 (oversold)')
elif h4_rsi < 40:
    long_score += 2
    print('  [+2] H4 RSI < 40')
else:
    print('  [+0] H4 RSI not oversold (%.1f)' % h4_rsi)
if h1_rsi < 30:
    long_score += 2
    print('  [+2] H1 RSI < 30 (oversold)')
elif h1_rsi < 40:
    long_score += 1
    print('  [+1] H1 RSI < 40')
else:
    print('  [+0] H1 RSI not oversold (%.1f)' % h1_rsi)

print()
print('Zone:')
if h4_range_pct < 35:
    long_score += 1
    print('  [+1] Price in discount (%.0f%%)' % h4_range_pct)
else:
    print('  [+0] Price at %.0f%% of range' % h4_range_pct)

print()
print('LONG SCORE: %d / 13' % long_score)

print()
print('='*70)
print('SETUP SCORING - SHORT')
print('='*70)
print()

short_score = 0
print('HTF Alignment:')
if mn_trend == 'BEAR':
    short_score += 3
    print('  [+3] Monthly BEARISH')
else:
    print('  [+0] Monthly BULLISH')
if w1_trend == 'BEAR':
    short_score += 2
    print('  [+2] Weekly BEARISH')
elif w1_rsi < 55:
    short_score += 1
    print('  [+1] Weekly RSI < 55')
else:
    print('  [+0] Weekly not supportive')
if d1_trend == 'BEAR':
    short_score += 1
    print('  [+1] Daily BEARISH')
else:
    print('  [+0] Daily BULLISH')

print()
print('Entry Signals:')
if h4_rsi > 70:
    short_score += 3
    print('  [+3] H4 RSI > 70 (overbought)')
elif h4_rsi > 60:
    short_score += 2
    print('  [+2] H4 RSI > 60')
else:
    print('  [+0] H4 RSI not overbought (%.1f)' % h4_rsi)
if h1_rsi > 70:
    short_score += 2
    print('  [+2] H1 RSI > 70 (overbought)')
elif h1_rsi > 60:
    short_score += 1
    print('  [+1] H1 RSI > 60')
else:
    print('  [+0] H1 RSI not overbought (%.1f)' % h1_rsi)

print()
print('Zone:')
if h4_range_pct > 65:
    short_score += 1
    print('  [+1] Price in premium (%.0f%%)' % h4_range_pct)
else:
    print('  [+0] Price at %.0f%% of range' % h4_range_pct)

print()
print('SHORT SCORE: %d / 13' % short_score)

print()
print('='*70)
print('TRADE DECISION')
print('='*70)
print()

if long_score >= 10:
    print('LONG: STRONG SETUP (%d/13)' % long_score)
elif long_score >= 7:
    print('LONG: VALID SETUP (%d/13)' % long_score)
elif long_score >= 5:
    print('LONG: MODERATE (%d/13)' % long_score)
else:
    print('LONG: WEAK (%d/13) - NOT RECOMMENDED' % long_score)

if short_score >= 10:
    print('SHORT: STRONG SETUP (%d/13)' % short_score)
elif short_score >= 7:
    print('SHORT: VALID SETUP (%d/13)' % short_score)
elif short_score >= 5:
    print('SHORT: MODERATE (%d/13)' % short_score)
else:
    print('SHORT: WEAK (%d/13) - NOT RECOMMENDED' % short_score)

print()
if long_score > short_score and long_score >= 7:
    print('RECOMMENDATION: LONG')
    print('  Wait for H4/H1 RSI < 30 for optimal entry')
    sl = h4_low - 50
    tp = price + (price - sl) * 2
    print('  Suggested SL: $%.2f (below H4 low)' % sl)
    print('  Suggested TP: $%.2f (2:1 R:R)' % tp)
elif short_score > long_score and short_score >= 7:
    print('RECOMMENDATION: SHORT')
    print('  Wait for H4/H1 RSI > 70 for optimal entry')
    sl = h4_high + 50
    tp = price - (sl - price) * 2
    print('  Suggested SL: $%.2f (above H4 high)' % sl)
    print('  Suggested TP: $%.2f (2:1 R:R)' % tp)
else:
    print('RECOMMENDATION: WAIT')
    print('  No clear setup - wait for better alignment')
    if mn_trend == 'BULL':
        print('  Monthly is BULL - look for pullback to enter LONG')
        print('  Wait for H4 RSI < 35')
    else:
        print('  Monthly is BEAR - look for bounce to enter SHORT')
        print('  Wait for H4 RSI > 65')
