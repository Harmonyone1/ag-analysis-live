#!/usr/bin/env python
"""Full multi-timeframe analysis for ETHUSD and BTCUSD including Weekly and Monthly"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time

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
        if len(prices) < 20:
            return 'NEUTRAL', 0, 0
        ma20 = np.mean(prices[-20:])
        current = prices[-1]
        if current > ma20:
            return 'BULL', ma20, 0
        elif current < ma20:
            return 'BEAR', ma20, 0
        return 'NEUTRAL', ma20, 0

    ma20 = np.mean(prices[-20:])
    ma50 = np.mean(prices[-50:])
    current = prices[-1]
    if current > ma20 > ma50:
        return 'BULL', ma20, ma50
    elif current < ma20 < ma50:
        return 'BEAR', ma20, ma50
    return 'NEUTRAL', ma20, ma50

def analyze_symbol(symbol):
    inst_id = name_to_id.get(symbol)
    if not inst_id:
        return None

    print('=' * 70)
    print('%s FULL MULTI-TIMEFRAME ANALYSIS' % symbol)
    print('=' * 70)
    print()

    # Get current price
    current_price = api.get_latest_asking_price(inst_id)
    print('Current Price: $%.2f' % current_price)
    print()

    results = {}

    # Monthly (MN)
    print('Fetching Monthly data...')
    try:
        mn = api.get_price_history(inst_id, resolution='1M', start_timestamp=0, end_timestamp=0, lookback_period='2Y')
        time.sleep(0.15)
        if mn is not None and len(mn) > 5:
            mn_trend, mn_ma20, mn_ma50 = get_trend(mn['c'].values)
            mn_rsi = calculate_rsi(mn['c'].values)
            mn_high = max(mn['h'].values)
            mn_low = min(mn['l'].values)
            results['MN'] = {'trend': mn_trend, 'rsi': mn_rsi, 'high': mn_high, 'low': mn_low}
            print('Monthly (MN): Trend: %s | RSI: %.1f' % (mn_trend, mn_rsi))
        else:
            results['MN'] = {'trend': 'N/A', 'rsi': 0}
            print('Monthly (MN): Insufficient data')
    except Exception as e:
        results['MN'] = {'trend': 'N/A', 'rsi': 0}
        print('Monthly (MN): Error - %s' % str(e))

    # Weekly (W1)
    print('Fetching Weekly data...')
    try:
        w1 = api.get_price_history(inst_id, resolution='1W', start_timestamp=0, end_timestamp=0, lookback_period='1Y')
        time.sleep(0.15)
        if w1 is not None and len(w1) > 10:
            w1_trend, w1_ma20, w1_ma50 = get_trend(w1['c'].values)
            w1_rsi = calculate_rsi(w1['c'].values)
            results['W1'] = {'trend': w1_trend, 'rsi': w1_rsi}
            print('Weekly (W1): Trend: %s | RSI: %.1f' % (w1_trend, w1_rsi))
        else:
            results['W1'] = {'trend': 'N/A', 'rsi': 0}
            print('Weekly (W1): Insufficient data')
    except Exception as e:
        results['W1'] = {'trend': 'N/A', 'rsi': 0}
        print('Weekly (W1): Error - %s' % str(e))

    # Daily (D1)
    print('Fetching Daily data...')
    try:
        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='6M')
        time.sleep(0.15)
        if d1 is not None and len(d1) > 50:
            d1_trend, d1_ma20, d1_ma50 = get_trend(d1['c'].values)
            d1_rsi = calculate_rsi(d1['c'].values)
            d1_high = max(d1['h'].values)
            d1_low = min(d1['l'].values)
            results['D1'] = {'trend': d1_trend, 'rsi': d1_rsi, 'high': d1_high, 'low': d1_low}
            print('Daily (D1): Trend: %s | RSI: %.1f' % (d1_trend, d1_rsi))
        else:
            results['D1'] = {'trend': 'N/A', 'rsi': 0}
            print('Daily (D1): Insufficient data')
    except Exception as e:
        results['D1'] = {'trend': 'N/A', 'rsi': 0}
        print('Daily (D1): Error - %s' % str(e))

    # 4-Hour (H4)
    print('Fetching 4H data...')
    try:
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='1M')
        time.sleep(0.15)
        if h4 is not None and len(h4) > 50:
            h4_trend, h4_ma20, h4_ma50 = get_trend(h4['c'].values)
            h4_rsi = calculate_rsi(h4['c'].values)
            results['H4'] = {'trend': h4_trend, 'rsi': h4_rsi}
            print('4-Hour (H4): Trend: %s | RSI: %.1f' % (h4_trend, h4_rsi))
        else:
            results['H4'] = {'trend': 'N/A', 'rsi': 0}
            print('4-Hour (H4): Insufficient data')
    except Exception as e:
        results['H4'] = {'trend': 'N/A', 'rsi': 0}
        print('4-Hour (H4): Error - %s' % str(e))

    # 1-Hour (H1)
    print('Fetching 1H data...')
    try:
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='2W')
        time.sleep(0.15)
        if h1 is not None and len(h1) > 50:
            h1_trend, h1_ma20, h1_ma50 = get_trend(h1['c'].values)
            h1_rsi = calculate_rsi(h1['c'].values)
            results['H1'] = {'trend': h1_trend, 'rsi': h1_rsi}
            print('1-Hour (H1): Trend: %s | RSI: %.1f' % (h1_trend, h1_rsi))
        else:
            results['H1'] = {'trend': 'N/A', 'rsi': 0}
            print('1-Hour (H1): Insufficient data')
    except Exception as e:
        results['H1'] = {'trend': 'N/A', 'rsi': 0}
        print('1-Hour (H1): Error - %s' % str(e))

    print()

    # Calculate price zone using D1 data
    if 'D1' in results and 'high' in results['D1']:
        high_range = results['D1']['high']
        low_range = results['D1']['low']
        range_pct = ((current_price - low_range) / (high_range - low_range)) * 100 if high_range != low_range else 50

        if range_pct < 30:
            zone = 'DISCOUNT'
        elif range_pct > 70:
            zone = 'PREMIUM'
        else:
            zone = 'EQUILIBRIUM'

        results['zone'] = zone
        results['range_pct'] = range_pct
        results['high'] = high_range
        results['low'] = low_range

        print('PRICE ZONE:')
        print('  Range: $%.2f - $%.2f' % (low_range, high_range))
        print('  Current: %.1f%% [%s]' % (range_pct, zone))
        print()

    # HTF Analysis Summary
    print('-' * 70)
    print('HIGHER TIMEFRAME SUMMARY')
    print('-' * 70)
    print()
    print('  Timeframe | Trend    | RSI')
    print('  ----------|----------|------')
    for tf in ['MN', 'W1', 'D1', 'H4', 'H1']:
        if tf in results:
            print('  %-9s | %-8s | %.1f' % (tf, results[tf]['trend'], results[tf]['rsi']))

    print()

    # Determine HTF bias
    htf_trends = [results[tf]['trend'] for tf in ['MN', 'W1', 'D1'] if tf in results and results[tf]['trend'] != 'N/A']
    bull_count = htf_trends.count('BULL')
    bear_count = htf_trends.count('BEAR')

    if bull_count == len(htf_trends) and len(htf_trends) >= 2:
        htf_bias = 'STRONG BULL'
    elif bear_count == len(htf_trends) and len(htf_trends) >= 2:
        htf_bias = 'STRONG BEAR'
    elif bull_count > bear_count:
        htf_bias = 'BULL BIAS'
    elif bear_count > bull_count:
        htf_bias = 'BEAR BIAS'
    else:
        htf_bias = 'MIXED/NEUTRAL'

    print('HTF BIAS: %s (MN/W1/D1: %d BULL, %d BEAR)' % (htf_bias, bull_count, bear_count))
    print()

    # LTF alignment check
    ltf_trends = [results[tf]['trend'] for tf in ['D1', 'H4', 'H1'] if tf in results and results[tf]['trend'] != 'N/A']
    ltf_bull = ltf_trends.count('BULL')
    ltf_bear = ltf_trends.count('BEAR')

    if ltf_bull == len(ltf_trends):
        ltf_alignment = 'BULL ALIGNED'
    elif ltf_bear == len(ltf_trends):
        ltf_alignment = 'BEAR ALIGNED'
    else:
        ltf_alignment = 'MIXED'

    print('LTF ALIGNMENT: %s (D1/H4/H1: %d BULL, %d BEAR)' % (ltf_alignment, ltf_bull, ltf_bear))
    print()

    # Framework verdict
    print('=' * 70)
    print('ICT+QUANT FRAMEWORK VERDICT')
    print('=' * 70)
    print()

    h4_rsi = results.get('H4', {}).get('rsi', 50)
    zone = results.get('zone', 'EQUILIBRIUM')
    range_pct = results.get('range_pct', 50)

    # Full alignment check: MN + W1 + D1 + H4 all same direction
    full_bull = all(results.get(tf, {}).get('trend') == 'BULL' for tf in ['MN', 'W1', 'D1', 'H4'] if results.get(tf, {}).get('trend') != 'N/A')
    full_bear = all(results.get(tf, {}).get('trend') == 'BEAR' for tf in ['MN', 'W1', 'D1', 'H4'] if results.get(tf, {}).get('trend') != 'N/A')

    # Valid setups
    valid_long = full_bull and zone == 'DISCOUNT' and h4_rsi < 40
    valid_short = full_bear and zone == 'PREMIUM' and h4_rsi > 60

    print('Checklist for LONG:')
    print('  [%s] HTF BULL (MN+W1+D1+H4 aligned)' % ('X' if full_bull else ' '))
    print('  [%s] DISCOUNT zone (<30%%)' % ('X' if zone == 'DISCOUNT' else ' '))
    print('  [%s] H4 RSI < 40 (current: %.1f)' % ('X' if h4_rsi < 40 else ' ', h4_rsi))
    print()

    print('Checklist for SHORT:')
    print('  [%s] HTF BEAR (MN+W1+D1+H4 aligned)' % ('X' if full_bear else ' '))
    print('  [%s] PREMIUM zone (>70%%)' % ('X' if zone == 'PREMIUM' else ' '))
    print('  [%s] H4 RSI > 60 (current: %.1f)' % ('X' if h4_rsi > 60 else ' ', h4_rsi))
    print()

    if valid_long:
        print('>>> VALID LONG SETUP <<<')
        print('Entry: $%.2f' % current_price)
        print('SL: $%.2f (3%%)' % (current_price * 0.97))
        print('TP: $%.2f (6%%)' % (current_price * 1.06))
    elif valid_short:
        print('>>> VALID SHORT SETUP <<<')
        print('Entry: $%.2f' % current_price)
        print('SL: $%.2f (3%%)' % (current_price * 1.03))
        print('TP: $%.2f (6%%)' % (current_price * 0.94))
    else:
        print('NO VALID SETUP')
        print()
        print('What to watch for:')
        if htf_bias in ['STRONG BULL', 'BULL BIAS']:
            print('  - Wait for pullback to DISCOUNT zone')
            print('  - Wait for H4 RSI to drop below 40')
            print('  - LTF needs to align BULL')
        elif htf_bias in ['STRONG BEAR', 'BEAR BIAS']:
            print('  - Wait for rally to PREMIUM zone')
            print('  - Wait for H4 RSI to rise above 60')
            print('  - LTF needs to align BEAR')
        else:
            print('  - Wait for HTF to establish clear direction')
            print('  - No trade until MN/W1/D1 align')

    print()
    print('=' * 70)

    return results

# Main execution
print()
state = api.get_account_state()
print('Account: Balance $%.2f | Equity $%.2f' % (state['balance'], state['balance'] + state['openGrossPnL']))
print()

# Analyze both
eth_results = analyze_symbol('ETHUSD')
print()
print()
btc_results = analyze_symbol('BTCUSD')
