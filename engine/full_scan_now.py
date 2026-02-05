#!/usr/bin/env python
"""Full market scan for valid ICT+Quant setups"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time
import pandas as pd
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

def analyze_symbol(symbol, inst_id):
    try:
        # Get D1 data
        d1_bars = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='6M')
        time.sleep(0.1)
        if d1_bars is None or len(d1_bars) < 50:
            return None
        d1_close = d1_bars['c'].values
        d1_high = d1_bars['h'].values
        d1_low = d1_bars['l'].values

        # Get H4 data
        h4_bars = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='1M')
        time.sleep(0.1)
        if h4_bars is None or len(h4_bars) < 50:
            return None
        h4_close = h4_bars['c'].values

        d1_trend = get_trend(d1_close)
        h4_trend = get_trend(h4_close)

        # Calculate RSI
        rsi = calculate_rsi(h4_close)

        # Calculate discount/premium zone
        high_52 = max(d1_high[-260:]) if len(d1_high) >= 260 else max(d1_high)
        low_52 = min(d1_low[-260:]) if len(d1_low) >= 260 else min(d1_low)
        current = d1_close[-1]
        range_pct = ((current - low_52) / (high_52 - low_52)) * 100 if high_52 != low_52 else 50

        zone = 'DISC' if range_pct < 30 else 'PREM' if range_pct > 70 else 'EQ'

        # Check for valid setups
        # LONG: HTF BULL aligned + Discount + RSI < 40
        # SHORT: HTF BEAR aligned + Premium + RSI > 60

        valid_long = (d1_trend == 'BULL' and h4_trend == 'BULL' and zone == 'DISC' and rsi < 40)
        valid_short = (d1_trend == 'BEAR' and h4_trend == 'BEAR' and zone == 'PREM' and rsi > 60)

        if valid_long or valid_short:
            return {
                'symbol': symbol,
                'signal': 'LONG' if valid_long else 'SHORT',
                'd1_trend': d1_trend,
                'h4_trend': h4_trend,
                'zone': zone,
                'range_pct': range_pct,
                'rsi': rsi,
                'price': current,
                'confidence': 'HIGH' if (rsi < 35 if valid_long else rsi > 65) else 'MEDIUM'
            }

        # Also return "watch" candidates (almost valid)
        watch_long = (d1_trend == 'BULL' and h4_trend == 'BULL' and range_pct < 40 and rsi < 45)
        watch_short = (d1_trend == 'BEAR' and h4_trend == 'BEAR' and range_pct > 60 and rsi > 55)

        if watch_long or watch_short:
            return {
                'symbol': symbol,
                'signal': 'WATCH LONG' if watch_long else 'WATCH SHORT',
                'd1_trend': d1_trend,
                'h4_trend': h4_trend,
                'zone': zone,
                'range_pct': range_pct,
                'rsi': rsi,
                'price': current,
                'confidence': 'WATCH'
            }

        return None

    except Exception as e:
        return None

print('=' * 70)
print('FULL MARKET SCAN | Checking for ICT+Quant setups')
print('=' * 70)
print()

# Get current account state first
state = api.get_account_state()
print('ACCOUNT: Equity $%.2f | P/L $%.2f' % (state['balance'] + state['openGrossPnL'], state['openGrossPnL']))
print()

# Show current positions
positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    print('CURRENT POSITIONS:')
    for _, pos in positions.iterrows():
        sym = str(pos.get('tradableInstrumentId', ''))
        qty = pos.get('qty', 0)
        pnl = pos.get('unrealizedPnL', 0)
        side = 'LONG' if pos.get('side', '') == 'buy' else 'SHORT'
        # Get symbol name
        for name, tid in name_to_id.items():
            if str(tid) == sym:
                print('  %s %s %.2f lots: P/L $%.2f' % (name, side, qty, pnl))
                break
    print()

# Scan all markets
symbols_to_scan = [
    # Major Forex
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    # Minor Forex
    'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY',
    'EURAUD', 'EURNZD', 'EURCAD', 'GBPAUD', 'GBPNZD', 'GBPCAD',
    'AUDNZD', 'AUDCAD', 'NZDCAD',
    # Indices
    'US30', 'US500', 'US100', 'GER40', 'UK100', 'JPN225',
    # Commodities
    'XAUUSD', 'XAGUSD', 'XTIUSD', 'XBRUSD',
    # Stocks
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
    'JPM', 'BAC', 'WFC', 'GS', 'MS',
    'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV',
    'XOM', 'CVX', 'COP', 'SLB',
    'DIS', 'SBUX', 'NKE', 'MCD', 'KO', 'PEP',
    # Crypto
    'BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD',
]

valid_setups = []
watch_setups = []
scanned = 0

print('Scanning %d instruments...' % len(symbols_to_scan))
print()

for symbol in symbols_to_scan:
    inst_id = name_to_id.get(symbol)
    if not inst_id:
        continue

    result = analyze_symbol(symbol, inst_id)
    scanned += 1

    if result:
        if 'WATCH' in result['signal']:
            watch_setups.append(result)
        else:
            valid_setups.append(result)

    # Progress indicator
    if scanned % 10 == 0:
        print('  Scanned %d/%d...' % (scanned, len(symbols_to_scan)))

print()
print('=' * 70)
print('SCAN RESULTS')
print('=' * 70)
print()

if valid_setups:
    print('VALID SETUPS (Framework rules met):')
    for s in valid_setups:
        print('  %s %s [%s]' % (s['symbol'], s['signal'], s['confidence']))
        print('    HTF: %s/%s | Zone: %s (%.0f%%) | RSI: %.1f' % (
            s['d1_trend'], s['h4_trend'], s['zone'], s['range_pct'], s['rsi']))
        print()
else:
    print('NO VALID SETUPS FOUND')
    print()

if watch_setups:
    print('WATCH LIST (Close to valid):')
    for s in watch_setups:
        print('  %s %s' % (s['symbol'], s['signal']))
        print('    HTF: %s/%s | Zone: %s (%.0f%%) | RSI: %.1f' % (
            s['d1_trend'], s['h4_trend'], s['zone'], s['range_pct'], s['rsi']))
        print()

print('=' * 70)
print('Scan complete. Checked %d instruments.' % scanned)
print('=' * 70)
