#!/usr/bin/env python
"""Deep Analysis - AUDNZD"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

symbol = 'AUDNZD'
inst_id = name_to_id.get(symbol)

print('=' * 70)
print('DEEP ANALYSIS: AUDNZD | %s' % datetime.now().strftime('%I:%M %p'))
print('=' * 70)

# Get all timeframe data
d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='60D')
h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='14D')
h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='7D')
m15 = api.get_price_history(inst_id, resolution='15m', start_timestamp=0, end_timestamp=0, lookback_period='3D')

price = api.get_latest_asking_price(inst_id)
spread = 1.5  # Typical spread for AUDNZD

print()
print('CURRENT PRICE: %.5f | Spread: %.1f pips' % (price, spread))
print()

# ============================================================
# MULTI-TIMEFRAME BIAS
# ============================================================
print('=' * 70)
print('MULTI-TIMEFRAME ANALYSIS')
print('=' * 70)

# Daily bias
d1_high = max(d1['h'].values[-20:])
d1_low = min(d1['l'].values[-20:])
d1_close = d1['c'].iloc[-1]
d1_open_5 = d1['o'].iloc[-5]
d1_bias = 'BULLISH' if d1_close > d1_open_5 else 'BEARISH'
d1_range_pct = (price - d1_low) / (d1_high - d1_low) * 100

print()
print('DAILY (D1):')
print('  Bias: %s (close %.5f > open_5d %.5f)' % (d1_bias, d1_close, d1_open_5))
print('  20-Day Range: %.5f - %.5f' % (d1_low, d1_high))
print('  Position in Range: %.1f%%' % d1_range_pct)

# 4H bias
h4_high = max(h4['h'].values[-20:])
h4_low = min(h4['l'].values[-20:])
h4_close = h4['c'].iloc[-1]
h4_open_5 = h4['o'].iloc[-5]
h4_bias = 'BULLISH' if h4_close > h4_open_5 else 'BEARISH'
h4_range_pct = (price - h4_low) / (h4_high - h4_low) * 100

print()
print('4-HOUR (H4):')
print('  Bias: %s' % h4_bias)
print('  Range: %.5f - %.5f' % (h4_low, h4_high))
print('  Position in Range: %.1f%%' % h4_range_pct)

# 1H bias
h1_high = max(h1['h'].values[-24:])
h1_low = min(h1['l'].values[-24:])
h1_close = h1['c'].iloc[-1]
h1_open_5 = h1['o'].iloc[-5]
h1_bias = 'BULLISH' if h1_close > h1_open_5 else 'BEARISH'

print()
print('1-HOUR (H1):')
print('  Bias: %s' % h1_bias)
print('  24H Range: %.5f - %.5f' % (h1_low, h1_high))

# HTF Alignment
htf_aligned = d1_bias == h4_bias
print()
print('HTF ALIGNMENT: %s' % ('YES - %s' % d1_bias if htf_aligned else 'NO - MIXED'))

# ============================================================
# RSI ANALYSIS
# ============================================================
print()
print('=' * 70)
print('RSI ANALYSIS (Multiple Timeframes)')
print('=' * 70)

def calc_rsi(closes, period=14):
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0.001
    return 100 - (100 / (1 + avg_gain / avg_loss))

rsi_d1 = calc_rsi(d1['c'].values)
rsi_h4 = calc_rsi(h4['c'].values)
rsi_h1 = calc_rsi(h1['c'].values)
rsi_m15 = calc_rsi(m15['c'].values)

print()
print('  Daily RSI:  %.1f %s' % (rsi_d1, '(OVERBOUGHT)' if rsi_d1 > 70 else '(OVERSOLD)' if rsi_d1 < 30 else ''))
print('  4H RSI:     %.1f %s' % (rsi_h4, '(OVERBOUGHT)' if rsi_h4 > 70 else '(OVERSOLD)' if rsi_h4 < 30 else ''))
print('  1H RSI:     %.1f %s' % (rsi_h1, '(OVERBOUGHT)' if rsi_h1 > 70 else '(OVERSOLD)' if rsi_h1 < 30 else ''))
print('  15M RSI:    %.1f %s' % (rsi_m15, '(OVERBOUGHT)' if rsi_m15 > 70 else '(OVERSOLD)' if rsi_m15 < 30 else ''))

# ============================================================
# KEY LEVELS
# ============================================================
print()
print('=' * 70)
print('KEY LEVELS')
print('=' * 70)

# Recent swing highs/lows
recent_high = max(h4['h'].values[-10:])
recent_low = min(h4['l'].values[-10:])
weekly_high = max(d1['h'].values[-5:])
weekly_low = min(d1['l'].values[-5:])
monthly_high = max(d1['h'].values[-20:])
monthly_low = min(d1['l'].values[-20:])

print()
print('  Current Price:  %.5f' % price)
print()
print('  Recent High:    %.5f (%.1f pips away)' % (recent_high, (recent_high - price) * 10000))
print('  Recent Low:     %.5f (%.1f pips away)' % (recent_low, (price - recent_low) * 10000))
print()
print('  Weekly High:    %.5f' % weekly_high)
print('  Weekly Low:     %.5f' % weekly_low)
print()
print('  Monthly High:   %.5f' % monthly_high)
print('  Monthly Low:    %.5f' % monthly_low)

# ============================================================
# TRADE ANALYSIS
# ============================================================
print()
print('=' * 70)
print('TRADE ANALYSIS')
print('=' * 70)

# Framework assessment
print()
print('ICT+QUANT FRAMEWORK ASSESSMENT:')
print('-' * 50)
print('  HTF Bias:     %s' % d1_bias)
print('  Zone:         PREMIUM (%.1f%%)' % d1_range_pct)
print('  RSI (1H):     %.1f' % rsi_h1)
print()

if d1_bias == 'BULLISH':
    print('  Framework says: LONG only in DISCOUNT with RSI < 40')
    print('  Current:        Price at PREMIUM (%.1f%%), RSI %.1f' % (d1_range_pct, rsi_h1))
    print('  Status:         NOT A VALID LONG - too extended')
    print()
    print('  For SHORT (counter-trend):')
    print('    - HTF is BULLISH (against SHORT)')
    print('    - RSI %.1f is extremely overbought' % rsi_h1)
    print('    - At %.1f%% of range (near top)' % d1_range_pct)

# Risk analysis for counter-trend SHORT
print()
print('-' * 50)
print('COUNTER-TREND SHORT ANALYSIS:')
print('-' * 50)

# SL above recent high, TP at recent support
sl_price = recent_high + 0.0010  # 10 pips above recent high
sl_pips = (sl_price - price) * 10000

# Targets
tp1 = h4_low + (h4_high - h4_low) * 0.5  # 50% retracement
tp2 = h4_low + (h4_high - h4_low) * 0.382  # 38.2% Fib
tp3 = h4_low  # Full retracement

tp1_pips = (price - tp1) * 10000
tp2_pips = (price - tp2) * 10000
tp3_pips = (price - tp3) * 10000

print()
print('  Entry:      %.5f (current)' % price)
print('  Stop Loss:  %.5f (%.1f pips above recent high)' % (sl_price, sl_pips))
print()
print('  Target 1:   %.5f (50%% retrace) = %.1f pips | R:R = 1:%.1f' % (tp1, tp1_pips, tp1_pips/sl_pips if sl_pips > 0 else 0))
print('  Target 2:   %.5f (38.2%% Fib) = %.1f pips | R:R = 1:%.1f' % (tp2, tp2_pips, tp2_pips/sl_pips if sl_pips > 0 else 0))
print('  Target 3:   %.5f (range low) = %.1f pips | R:R = 1:%.1f' % (tp3, tp3_pips, tp3_pips/sl_pips if sl_pips > 0 else 0))

print()
print('=' * 70)
print('RECOMMENDATION')
print('=' * 70)
print()

if rsi_h1 > 85 and d1_range_pct > 90:
    print('AUDNZD is at EXTREME levels:')
    print('  - RSI %.1f (very overbought)' % rsi_h1)
    print('  - %.1f%% of range (near absolute top)' % d1_range_pct)
    print()
    print('OPTIONS:')
    print('  1. WAIT for pullback to discount, then LONG with trend')
    print('     (Framework-compliant, safest)')
    print()
    print('  2. COUNTER-TREND SHORT with reduced size')
    print('     - Use 0.01-0.02 lots MAX')
    print('     - Tight SL above %.5f' % recent_high)
    print('     - Target 50%% retracement first')
    print('     - HIGH RISK: Trading against HTF bias')
    print()
    print('  3. NO TRADE - wait for cleaner setup')
    print()
    print('RISK WARNING: Counter-trend trades have lower win rate.')
    print('The trend can stay overbought longer than expected.')
else:
    print('Setup not at extreme levels. Continue monitoring.')

print()
print('=' * 70)
