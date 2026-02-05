#!/usr/bin/env python
"""Liquidity Analysis - BSL/SSL Levels"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Pairs to analyze
PAIRS = [
    'AUDNZD', 'USDJPY', 'GBPJPY', 'EURJPY',  # Overbought
    'NZDUSD', 'NZDCAD', 'NZDCHF',            # Oversold
    'EURUSD', 'AUDUSD', 'GBPUSD',            # Developing
    'XAUUSD',                                  # Gold
]

def find_swing_points(highs, lows, lookback=20):
    """Find swing highs and lows (liquidity pools)"""
    swing_highs = []
    swing_lows = []

    for i in range(2, min(lookback, len(highs) - 2)):
        # Swing high: higher than 2 candles before and after
        if highs[-i] > highs[-i-1] and highs[-i] > highs[-i-2] and \
           highs[-i] > highs[-i+1] and highs[-i] > highs[-i+2]:
            swing_highs.append(highs[-i])

        # Swing low: lower than 2 candles before and after
        if lows[-i] < lows[-i-1] and lows[-i] < lows[-i-2] and \
           lows[-i] < lows[-i+1] and lows[-i] < lows[-i+2]:
            swing_lows.append(lows[-i])

    return swing_highs, swing_lows

def find_equal_highs_lows(highs, lows, tolerance=0.0003):
    """Find equal highs/lows (strong liquidity pools)"""
    equal_highs = []
    equal_lows = []

    # Check for equal highs
    for i in range(len(highs) - 1):
        for j in range(i + 1, min(i + 10, len(highs))):
            if abs(highs[i] - highs[j]) / highs[i] < tolerance:
                equal_highs.append((highs[i] + highs[j]) / 2)

    # Check for equal lows
    for i in range(len(lows) - 1):
        for j in range(i + 1, min(i + 10, len(lows))):
            if abs(lows[i] - lows[j]) / lows[i] < tolerance:
                equal_lows.append((lows[i] + lows[j]) / 2)

    return list(set(equal_highs)), list(set(equal_lows))

def analyze_liquidity(symbol):
    """Full liquidity analysis for a symbol"""
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            return None

        # Get multiple timeframes
        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='14D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='7D')

        if d1 is None or h4 is None or h1 is None:
            return None

        price = api.get_latest_asking_price(inst_id)

        # Get highs and lows
        d1_highs = d1['h'].values
        d1_lows = d1['l'].values
        h4_highs = h4['h'].values
        h4_lows = h4['l'].values
        h1_highs = h1['h'].values
        h1_lows = h1['l'].values

        # Find key levels
        daily_high = max(d1_highs[-5:])
        daily_low = min(d1_lows[-5:])
        weekly_high = max(d1_highs[-10:])
        weekly_low = min(d1_lows[-10:])

        # Find swing points on H4
        h4_swing_highs, h4_swing_lows = find_swing_points(h4_highs, h4_lows, 30)

        # Find equal highs/lows (strong liquidity)
        h4_equal_highs, h4_equal_lows = find_equal_highs_lows(h4_highs[-30:], h4_lows[-30:])

        # BSL (Buy-Side Liquidity) = stops above highs
        bsl_levels = []
        bsl_levels.append(('Daily High', daily_high))
        bsl_levels.append(('Weekly High', weekly_high))
        for i, sh in enumerate(sorted(h4_swing_highs, reverse=True)[:3]):
            if sh > price:
                bsl_levels.append(('Swing High %d' % (i+1), sh))
        for eh in sorted(h4_equal_highs, reverse=True)[:2]:
            if eh > price:
                bsl_levels.append(('Equal Highs', eh))

        # SSL (Sell-Side Liquidity) = stops below lows
        ssl_levels = []
        ssl_levels.append(('Daily Low', daily_low))
        ssl_levels.append(('Weekly Low', weekly_low))
        for i, sl in enumerate(sorted(h4_swing_lows)[:3]):
            if sl < price:
                ssl_levels.append(('Swing Low %d' % (i+1), sl))
        for el in sorted(h4_equal_lows)[:2]:
            if el < price:
                ssl_levels.append(('Equal Lows', el))

        # Calculate distances
        bsl_above = [(name, level, (level - price) * 10000 if 'JPY' not in symbol else (level - price) * 100)
                     for name, level in bsl_levels if level > price]
        ssl_below = [(name, level, (price - level) * 10000 if 'JPY' not in symbol else (price - level) * 100)
                     for name, level in ssl_levels if level < price]

        # Determine pip multiplier
        if 'JPY' in symbol:
            pip_mult = 100
        elif symbol in ['XAUUSD']:
            pip_mult = 10
        else:
            pip_mult = 10000

        # Nearest liquidity
        nearest_bsl = min(bsl_above, key=lambda x: x[2]) if bsl_above else None
        nearest_ssl = min(ssl_below, key=lambda x: x[2]) if ssl_below else None

        # Liquidity bias (which side is closer/more attractive to sweep)
        if nearest_bsl and nearest_ssl:
            if nearest_bsl[2] < nearest_ssl[2]:
                liq_bias = 'BSL_CLOSER'
            else:
                liq_bias = 'SSL_CLOSER'
        elif nearest_bsl:
            liq_bias = 'BSL_ONLY'
        elif nearest_ssl:
            liq_bias = 'SSL_ONLY'
        else:
            liq_bias = 'NEUTRAL'

        return {
            'symbol': symbol,
            'price': price,
            'bsl_above': sorted(bsl_above, key=lambda x: x[2])[:4],
            'ssl_below': sorted(ssl_below, key=lambda x: x[2])[:4],
            'nearest_bsl': nearest_bsl,
            'nearest_ssl': nearest_ssl,
            'liq_bias': liq_bias,
            'daily_high': daily_high,
            'daily_low': daily_low,
            'weekly_high': weekly_high,
            'weekly_low': weekly_low,
        }
    except Exception as e:
        return None

print('=' * 80)
print('LIQUIDITY ANALYSIS - BSL/SSL Levels | %s' % datetime.now().strftime('%I:%M %p'))
print('=' * 80)
print()
print('BSL (Buy-Side Liquidity) = Stop losses ABOVE price (shorts get stopped)')
print('SSL (Sell-Side Liquidity) = Stop losses BELOW price (longs get stopped)')
print()
print('Price tends to sweep liquidity before reversing.')
print()

for symbol in PAIRS:
    result = analyze_liquidity(symbol)
    if not result:
        continue

    print('=' * 80)
    print('%s | Price: %.5f | Bias: %s' % (result['symbol'], result['price'], result['liq_bias']))
    print('=' * 80)

    # BSL above
    print()
    print('  BSL (Above - targets for longs / stops for shorts):')
    if result['bsl_above']:
        for name, level, pips in result['bsl_above']:
            marker = '<<<' if result['nearest_bsl'] and level == result['nearest_bsl'][1] else ''
            print('    %.5f  %+.1f pips  %-15s %s' % (level, pips, name, marker))
    else:
        print('    None above current price')

    # Current price
    print()
    print('  >>> PRICE: %.5f <<<' % result['price'])

    # SSL below
    print()
    print('  SSL (Below - targets for shorts / stops for longs):')
    if result['ssl_below']:
        for name, level, pips in result['ssl_below']:
            marker = '<<<' if result['nearest_ssl'] and level == result['nearest_ssl'][1] else ''
            print('    %.5f  %+.1f pips  %-15s %s' % (level, -pips, name, marker))
    else:
        print('    None below current price')

    # Analysis
    print()
    if result['liq_bias'] == 'BSL_CLOSER':
        print('  ANALYSIS: BSL is closer - price may sweep highs before dropping')
    elif result['liq_bias'] == 'SSL_CLOSER':
        print('  ANALYSIS: SSL is closer - price may sweep lows before rising')

    print()

print('=' * 80)
print('TRADE IMPLICATIONS')
print('=' * 80)
print()
print('For LONGS:')
print('  - Enter AFTER SSL sweep (liquidity grab below lows)')
print('  - Target BSL levels above')
print('  - Place SL below the swept low')
print()
print('For SHORTS:')
print('  - Enter AFTER BSL sweep (liquidity grab above highs)')
print('  - Target SSL levels below')
print('  - Place SL above the swept high')
print()
print('=' * 80)
