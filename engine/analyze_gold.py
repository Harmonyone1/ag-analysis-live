#!/usr/bin/env python
"""XAUUSD Gold Analysis - ICT+QUANT Framework"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

inst_id = symbol_to_id.get('XAUUSD')

h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='1W')
h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='3D')
d1 = api.get_price_history(inst_id, resolution='1D', lookback_period='2W')

current = h1['c'].values[-1]
h4c, h4h, h4l = h4['c'].values, h4['h'].values, h4['l'].values
h1c, h1h, h1l = h1['c'].values, h1['h'].values, h1['l'].values
d1c = d1['c'].values

daily_bias = 'BULLISH' if d1c[-1] > d1c[-5] else 'BEARISH'
htf_bias = 'BULLISH' if h4c[-1] > h4c[-5] else 'BEARISH'

# RSI
deltas = np.diff(h1c[-15:])
gains = np.where(deltas > 0, deltas, 0)
losses = np.where(deltas < 0, -deltas, 0)
avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.0001
rsi = 100 - (100 / (1 + np.mean(gains)/avg_loss))

# ATR
tr_list = [max(h1h[i] - h1l[i], abs(h1h[i] - h1c[i-1]), abs(h1l[i] - h1c[i-1])) for i in range(-14, 0)]
atr = np.mean(tr_list)

# Ranges
week_high, week_low = max(h4h), min(h4l)
equilibrium = (week_high + week_low) / 2
price_pos = (current - week_low) / (week_high - week_low)
zone = 'PREMIUM' if current > equilibrium else 'DISCOUNT'

swing_high, swing_low = max(h1h[-24:]), min(h1l[-24:])

# OTE Zone
if htf_bias == 'BULLISH':
    ote_top = swing_low + (swing_high - swing_low) * 0.38
    ote_bottom = swing_low + (swing_high - swing_low) * 0.21
else:
    ote_bottom = swing_high - (swing_high - swing_low) * 0.38
    ote_top = swing_high - (swing_high - swing_low) * 0.21
in_ote = current >= ote_bottom and current <= ote_top

print('=' * 70)
print('XAUUSD (GOLD) - ICT+QUANT Analysis')
print('Time:', datetime.now().strftime('%I:%M %p EST'))
print('=' * 70)
print()
print('BIAS ALIGNMENT:')
print('  Daily:  ' + daily_bias)
print('  4H:     ' + htf_bias)
aligned = daily_bias == htf_bias
print('  Status: ' + ('ALIGNED' if aligned else 'MIXED'))
print()
print('PRICE LEVELS:')
print('  Current:     $%.2f' % current)
print('  Week High:   $%.2f' % week_high)
print('  Week Low:    $%.2f' % week_low)
print('  Equilibrium: $%.2f' % equilibrium)
print('  Zone:        %s (%.0f%%)' % (zone, price_pos * 100))
print()
print('SWING POINTS (24H):')
print('  High: $%.2f (buy-side liquidity)' % swing_high)
print('  Low:  $%.2f (sell-side liquidity)' % swing_low)
print()
print('INDICATORS:')
if rsi < 30:
    rsi_status = 'OVERSOLD'
elif rsi > 70:
    rsi_status = 'OVERBOUGHT'
elif rsi < 40:
    rsi_status = 'oversold zone'
elif rsi > 60:
    rsi_status = 'elevated'
else:
    rsi_status = 'neutral'
print('  RSI(14): %.1f [%s]' % (rsi, rsi_status))
print('  ATR(14): $%.2f' % atr)
print('  2x ATR:  $%.2f (suggested SL distance)' % (atr * 2))
print()
print('OTE ZONE (62-79%% Fib):')
print('  Range: $%.2f - $%.2f' % (ote_bottom, ote_top))
print('  Status: ' + ('** IN OTE **' if in_ote else 'Outside OTE'))
print()
print('=' * 70)
print('CONFLUENCE CHECK:')
print('=' * 70)

conf = 0
if htf_bias == 'BULLISH':
    print('Direction: LONG (HTF Bullish)')
    if aligned:
        conf += 1
        print('  [x] Daily + 4H aligned BULLISH')
    else:
        print('  [ ] Daily + 4H NOT aligned')
    if rsi < 40:
        conf += 1
        print('  [x] RSI in oversold zone')
    else:
        print('  [ ] RSI not oversold (%.1f)' % rsi)
    if zone == 'DISCOUNT':
        conf += 1
        print('  [x] Price in DISCOUNT zone')
    else:
        print('  [ ] Price in PREMIUM')
    if in_ote:
        conf += 1
        print('  [x] Price in OTE zone')
    else:
        print('  [ ] Not in OTE')
    if price_pos < 0.35:
        conf += 1
        print('  [x] Deep discount (<35%%)')
    else:
        print('  [ ] Not deep discount (%.0f%%)' % (price_pos * 100))
else:
    print('Direction: SHORT (HTF Bearish)')
    if aligned:
        conf += 1
        print('  [x] Daily + 4H aligned BEARISH')
    else:
        print('  [ ] Daily + 4H NOT aligned')
    if rsi > 60:
        conf += 1
        print('  [x] RSI elevated')
    else:
        print('  [ ] RSI not elevated (%.1f)' % rsi)
    if zone == 'PREMIUM':
        conf += 1
        print('  [x] Price in PREMIUM zone')
    else:
        print('  [ ] Price in DISCOUNT')
    if in_ote:
        conf += 1
        print('  [x] Price in OTE zone')
    else:
        print('  [ ] Not in OTE')
    if price_pos > 0.65:
        conf += 1
        print('  [x] Deep premium (>65%%)')
    else:
        print('  [ ] Not deep premium (%.0f%%)' % (price_pos * 100))

print()
print('TOTAL CONFLUENCES: %d/5' % conf)
if conf >= 3:
    print('>>> HIGH PROBABILITY SETUP <<<')
elif conf >= 2:
    print('>> Developing setup - need more confluence')
else:
    print('No clear setup - wait for better conditions')

print()
print('TRADE LEVELS (if entering):')
if htf_bias == 'BULLISH':
    sl = swing_low - atr * 0.5
    tp = swing_high
    risk = current - sl
    reward = tp - current
    rr = reward / risk if risk > 0 else 0
    print('  Entry: $%.2f' % current)
    print('  SL:    $%.2f (below swing low)' % sl)
    print('  TP:    $%.2f (swing high)' % tp)
    print('  Risk:  $%.2f | Reward: $%.2f' % (risk, reward))
    print('  R:R:   1:%.1f' % rr)
else:
    sl = swing_high + atr * 0.5
    tp = swing_low
    risk = sl - current
    reward = current - tp
    rr = reward / risk if risk > 0 else 0
    print('  Entry: $%.2f' % current)
    print('  SL:    $%.2f (above swing high)' % sl)
    print('  TP:    $%.2f (swing low)' % tp)
    print('  Risk:  $%.2f | Reward: $%.2f' % (risk, reward))
    print('  R:R:   1:%.1f' % rr)

print('=' * 70)
