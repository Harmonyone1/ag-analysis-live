#!/usr/bin/env python
"""Analyze market structure for ETHUSD and BTCUSD"""

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

def find_swing_points(highs, lows, closes, lookback=5):
    """Find swing highs and swing lows"""
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(highs) - lookback):
        # Swing high: highest point in lookback window
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            swing_highs.append((i, highs[i]))

        # Swing low: lowest point in lookback window
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows

def analyze_structure(swing_highs, swing_lows):
    """Determine market structure from swing points"""
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 'UNDEFINED', []

    # Get last few swing points
    recent_highs = swing_highs[-4:] if len(swing_highs) >= 4 else swing_highs
    recent_lows = swing_lows[-4:] if len(swing_lows) >= 4 else swing_lows

    structure_points = []

    # Check for Higher Highs / Lower Highs
    if len(recent_highs) >= 2:
        for i in range(1, len(recent_highs)):
            if recent_highs[i][1] > recent_highs[i-1][1]:
                structure_points.append('HH')  # Higher High
            else:
                structure_points.append('LH')  # Lower High

    # Check for Higher Lows / Lower Lows
    if len(recent_lows) >= 2:
        for i in range(1, len(recent_lows)):
            if recent_lows[i][1] > recent_lows[i-1][1]:
                structure_points.append('HL')  # Higher Low
            else:
                structure_points.append('LL')  # Lower Low

    # Determine overall structure
    hh_count = structure_points.count('HH')
    hl_count = structure_points.count('HL')
    lh_count = structure_points.count('LH')
    ll_count = structure_points.count('LL')

    bullish_signals = hh_count + hl_count
    bearish_signals = lh_count + ll_count

    if bullish_signals > bearish_signals and hh_count > 0 and hl_count > 0:
        structure = 'BULLISH (HH + HL)'
    elif bearish_signals > bullish_signals and lh_count > 0 and ll_count > 0:
        structure = 'BEARISH (LH + LL)'
    elif hh_count > 0 and ll_count > 0:
        structure = 'TRANSITIONAL'
    elif lh_count > 0 and hl_count > 0:
        structure = 'RANGING'
    else:
        structure = 'UNCLEAR'

    return structure, structure_points, recent_highs, recent_lows

def analyze_symbol_structure(symbol):
    inst_id = name_to_id.get(symbol)
    if not inst_id:
        return

    print('=' * 70)
    print('%s MARKET STRUCTURE ANALYSIS' % symbol)
    print('=' * 70)
    print()

    current_price = api.get_latest_asking_price(inst_id)
    print('Current Price: $%.2f' % current_price)
    print()

    timeframes = [
        ('1M', 'Monthly', '2Y', 3),
        ('1W', 'Weekly', '1Y', 3),
        ('1D', 'Daily', '6M', 5),
        ('4H', '4-Hour', '1M', 5),
        ('1H', '1-Hour', '2W', 5),
    ]

    all_structures = {}

    for resolution, name, lookback, swing_lookback in timeframes:
        try:
            data = api.get_price_history(inst_id, resolution=resolution, start_timestamp=0, end_timestamp=0, lookback_period=lookback)
            time.sleep(0.15)

            if data is None or len(data) < 20:
                print('%s: Insufficient data' % name)
                continue

            highs = data['h'].values
            lows = data['l'].values
            closes = data['c'].values

            swing_highs, swing_lows = find_swing_points(highs, lows, closes, swing_lookback)

            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                structure, points, recent_highs, recent_lows = analyze_structure(swing_highs, swing_lows)
                all_structures[name] = structure

                # Get the most recent swing points for display
                last_sh = swing_highs[-1] if swing_highs else None
                last_sl = swing_lows[-1] if swing_lows else None
                prev_sh = swing_highs[-2] if len(swing_highs) >= 2 else None
                prev_sl = swing_lows[-2] if len(swing_lows) >= 2 else None

                print('%s Structure: %s' % (name, structure))

                if prev_sh and last_sh:
                    sh_change = 'HH' if last_sh[1] > prev_sh[1] else 'LH'
                    print('  Swing Highs: $%.2f -> $%.2f [%s]' % (prev_sh[1], last_sh[1], sh_change))

                if prev_sl and last_sl:
                    sl_change = 'HL' if last_sl[1] > prev_sl[1] else 'LL'
                    print('  Swing Lows: $%.2f -> $%.2f [%s]' % (prev_sl[1], last_sl[1], sl_change))

                # Key levels
                if last_sh:
                    dist_to_high = ((last_sh[1] - current_price) / current_price) * 100
                    print('  Last Swing High: $%.2f (%.1f%% away)' % (last_sh[1], dist_to_high))
                if last_sl:
                    dist_to_low = ((current_price - last_sl[1]) / current_price) * 100
                    print('  Last Swing Low: $%.2f (%.1f%% away)' % (last_sl[1], dist_to_low))

                print()
            else:
                print('%s: Not enough swing points' % name)
                print()

        except Exception as e:
            print('%s: Error - %s' % (name, str(e)))
            print()

    # Summary
    print('-' * 70)
    print('STRUCTURE SUMMARY')
    print('-' * 70)
    print()

    for tf, struct in all_structures.items():
        print('  %s: %s' % (tf, struct))

    # Overall bias
    bullish_count = sum(1 for s in all_structures.values() if 'BULLISH' in s)
    bearish_count = sum(1 for s in all_structures.values() if 'BEARISH' in s)

    print()
    if bullish_count > bearish_count:
        print('OVERALL STRUCTURE: BULLISH (%d/%d timeframes)' % (bullish_count, len(all_structures)))
    elif bearish_count > bullish_count:
        print('OVERALL STRUCTURE: BEARISH (%d/%d timeframes)' % (bearish_count, len(all_structures)))
    else:
        print('OVERALL STRUCTURE: MIXED/TRANSITIONAL')

    # Key levels to watch
    print()
    print('KEY LEVELS TO WATCH:')

    # Get D1 data for key levels
    d1_data = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='3M')
    if d1_data is not None and len(d1_data) > 20:
        recent_high = max(d1_data['h'].values[-20:])
        recent_low = min(d1_data['l'].values[-20:])

        print('  20-Day High: $%.2f' % recent_high)
        print('  20-Day Low: $%.2f' % recent_low)
        print('  Current: $%.2f' % current_price)

        # Determine position
        range_size = recent_high - recent_low
        position = (current_price - recent_low) / range_size * 100 if range_size > 0 else 50

        if position < 30:
            print('  Position: NEAR LOWS (%.0f%% of range)' % position)
        elif position > 70:
            print('  Position: NEAR HIGHS (%.0f%% of range)' % position)
        else:
            print('  Position: MID-RANGE (%.0f%% of range)' % position)

    print()
    print('=' * 70)

    return all_structures

# Main
print()
state = api.get_account_state()
print('Account: Balance $%.2f' % state['balance'])
print()

eth_structure = analyze_symbol_structure('ETHUSD')
print()
print()
btc_structure = analyze_symbol_structure('BTCUSD')
