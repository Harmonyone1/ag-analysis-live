#!/usr/bin/env python
"""Scan for near-valid and watch-worthy setups"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time
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
        return 'NEUTRAL'
    ma20 = np.mean(prices[-20:])
    ma50 = np.mean(prices[-50:])
    current = prices[-1]
    if current > ma20 > ma50:
        return 'BULL'
    elif current < ma20 < ma50:
        return 'BEAR'
    return 'NEUTRAL'

def analyze(symbol):
    inst_id = name_to_id.get(symbol)
    if not inst_id:
        return None
    try:
        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='6M')
        time.sleep(0.1)
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='1M')
        time.sleep(0.1)

        if d1 is None or h4 is None or len(d1) < 50 or len(h4) < 50:
            return None

        d1_trend = get_trend(d1['c'].values)
        h4_trend = get_trend(h4['c'].values)
        rsi = calculate_rsi(h4['c'].values)

        high_52 = max(d1['h'].values[-260:]) if len(d1) >= 260 else max(d1['h'].values)
        low_52 = min(d1['l'].values[-260:]) if len(d1) >= 260 else min(d1['l'].values)
        current = d1['c'].values[-1]
        range_pct = ((current - low_52) / (high_52 - low_52)) * 100 if high_52 != low_52 else 50

        return {
            'symbol': symbol,
            'd1': d1_trend,
            'h4': h4_trend,
            'rsi': rsi,
            'range': range_pct,
            'price': current
        }
    except:
        return None

symbols = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF',
    'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'EURAUD', 'GBPAUD',
    'AUDNZD', 'AUDCAD', 'NZDCAD', 'EURCAD', 'GBPCAD', 'CHFJPY', 'CADJPY',
    'US30', 'US500', 'US100', 'GER40', 'UK100',
    'XAUUSD', 'XAGUSD', 'XTIUSD',
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
    'JPM', 'BAC', 'JNJ', 'PFE', 'XOM', 'CVX', 'DIS', 'NKE', 'KO',
    'BTCUSD', 'ETHUSD',
]

print('=' * 70)
print('MARKET SCAN | Looking for setups and watch candidates')
print('=' * 70)
print()

# Get account status
state = api.get_account_state()
print('Account: Equity $%.2f | P/L $%.2f' % (state['balance'] + state['openGrossPnL'], state['openGrossPnL']))
print()

valid_longs = []
valid_shorts = []
watch_longs = []
watch_shorts = []
extreme_rsi = []

print('Scanning %d instruments...' % len(symbols))
print()

for sym in symbols:
    result = analyze(sym)
    if not result:
        continue

    d1 = result['d1']
    h4 = result['h4']
    rsi = result['rsi']
    rng = result['range']

    # Valid LONG: HTF BULL + DISC (<30%) + RSI < 40
    if d1 == 'BULL' and h4 == 'BULL' and rng < 30 and rsi < 40:
        valid_longs.append(result)
    # Valid SHORT: HTF BEAR + PREM (>70%) + RSI > 60
    elif d1 == 'BEAR' and h4 == 'BEAR' and rng > 70 and rsi > 60:
        valid_shorts.append(result)
    # Watch LONG: HTF BULL + close to discount + RSI approaching oversold
    elif d1 == 'BULL' and h4 == 'BULL' and rng < 40 and rsi < 45:
        watch_longs.append(result)
    # Watch SHORT: HTF BEAR + close to premium + RSI approaching overbought
    elif d1 == 'BEAR' and h4 == 'BEAR' and rng > 60 and rsi > 55:
        watch_shorts.append(result)
    # Extreme RSI (potential reversal)
    elif rsi < 25 or rsi > 75:
        extreme_rsi.append(result)

print('=' * 70)
print('SCAN RESULTS')
print('=' * 70)
print()

if valid_longs:
    print('VALID LONG SETUPS:')
    for s in valid_longs:
        print('  >>> %s LONG | HTF: %s/%s | Zone: %.0f%% | RSI: %.1f' % (
            s['symbol'], s['d1'], s['h4'], s['range'], s['rsi']))
    print()
else:
    print('No valid LONG setups')
    print()

if valid_shorts:
    print('VALID SHORT SETUPS:')
    for s in valid_shorts:
        print('  >>> %s SHORT | HTF: %s/%s | Zone: %.0f%% | RSI: %.1f' % (
            s['symbol'], s['d1'], s['h4'], s['range'], s['rsi']))
    print()
else:
    print('No valid SHORT setups')
    print()

if watch_longs:
    print('WATCH LIST - LONGS (close to valid):')
    for s in watch_longs:
        print('  %s | HTF: %s/%s | Zone: %.0f%% | RSI: %.1f' % (
            s['symbol'], s['d1'], s['h4'], s['range'], s['rsi']))
    print()

if watch_shorts:
    print('WATCH LIST - SHORTS (close to valid):')
    for s in watch_shorts:
        print('  %s | HTF: %s/%s | Zone: %.0f%% | RSI: %.1f' % (
            s['symbol'], s['d1'], s['h4'], s['range'], s['rsi']))
    print()

if extreme_rsi:
    print('EXTREME RSI (potential reversal zones):')
    for s in extreme_rsi:
        direction = 'OVERSOLD' if s['rsi'] < 30 else 'OVERBOUGHT'
        print('  %s | RSI: %.1f [%s] | HTF: %s/%s' % (
            s['symbol'], s['rsi'], direction, s['d1'], s['h4']))
    print()

print('=' * 70)
