#!/usr/bin/env python
"""ICT Structure Analysis - Order Blocks, FVGs, Imbalances"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'NZDUSD', 'EURJPY', 'GBPJPY', 'NZDCAD', 'XAUUSD', 'ETHUSD']

def find_order_blocks(data, lookback=30):
    """Find Order Blocks - last opposing candle before strong move"""
    bullish_obs = []  # Last bearish candle before bullish move
    bearish_obs = []  # Last bullish candle before bearish move

    opens = data['o'].values
    closes = data['c'].values
    highs = data['h'].values
    lows = data['l'].values

    for i in range(2, min(lookback, len(data) - 1)):
        idx = -i

        # Check for Bullish Order Block
        # Current candle is bearish, next candle(s) are strongly bullish
        if closes[idx] < opens[idx]:  # Bearish candle
            # Check if followed by strong bullish move
            if idx + 1 < 0:
                next_close = closes[idx + 1]
                next_open = opens[idx + 1]
                if next_close > next_open:  # Next is bullish
                    move_size = next_close - next_open
                    candle_size = opens[idx] - closes[idx]
                    if move_size > candle_size * 1.5:  # Strong move
                        bullish_obs.append({
                            'high': highs[idx],
                            'low': lows[idx],
                            'type': 'BULLISH_OB',
                            'candles_ago': i
                        })

        # Check for Bearish Order Block
        # Current candle is bullish, next candle(s) are strongly bearish
        if closes[idx] > opens[idx]:  # Bullish candle
            if idx + 1 < 0:
                next_close = closes[idx + 1]
                next_open = opens[idx + 1]
                if next_close < next_open:  # Next is bearish
                    move_size = next_open - next_close
                    candle_size = closes[idx] - opens[idx]
                    if move_size > candle_size * 1.5:  # Strong move
                        bearish_obs.append({
                            'high': highs[idx],
                            'low': lows[idx],
                            'type': 'BEARISH_OB',
                            'candles_ago': i
                        })

    return bullish_obs[:3], bearish_obs[:3]  # Return top 3 of each

def find_fvgs(data, lookback=30):
    """Find Fair Value Gaps - 3 candle pattern with gap"""
    bullish_fvgs = []  # Gap up (candle 1 high < candle 3 low)
    bearish_fvgs = []  # Gap down (candle 1 low > candle 3 high)

    highs = data['h'].values
    lows = data['l'].values

    for i in range(2, min(lookback, len(data) - 2)):
        idx = -i

        # Bullish FVG: Candle 1 high < Candle 3 low (gap up)
        if highs[idx - 2] < lows[idx]:
            gap_size = lows[idx] - highs[idx - 2]
            bullish_fvgs.append({
                'top': lows[idx],
                'bottom': highs[idx - 2],
                'size': gap_size,
                'type': 'BULLISH_FVG',
                'candles_ago': i
            })

        # Bearish FVG: Candle 1 low > Candle 3 high (gap down)
        if lows[idx - 2] > highs[idx]:
            gap_size = lows[idx - 2] - highs[idx]
            bearish_fvgs.append({
                'top': lows[idx - 2],
                'bottom': highs[idx],
                'size': gap_size,
                'type': 'BEARISH_FVG',
                'candles_ago': i
            })

    return bullish_fvgs[:3], bearish_fvgs[:3]

def find_imbalances(data, lookback=30):
    """Find Imbalances - large candles with no overlap"""
    imbalances = []

    opens = data['o'].values
    closes = data['c'].values
    highs = data['h'].values
    lows = data['l'].values

    for i in range(1, min(lookback, len(data) - 1)):
        idx = -i

        body_size = abs(closes[idx] - opens[idx])
        candle_range = highs[idx] - lows[idx]

        # Large body candle (body > 70% of range)
        if candle_range > 0 and body_size / candle_range > 0.7:
            # Check if previous candle doesn't overlap
            if idx - 1 >= -len(data):
                prev_high = highs[idx - 1]
                prev_low = lows[idx - 1]

                # Bullish imbalance
                if closes[idx] > opens[idx] and lows[idx] > prev_high:
                    imbalances.append({
                        'top': lows[idx],
                        'bottom': prev_high,
                        'type': 'BULLISH_IMB',
                        'candles_ago': i
                    })

                # Bearish imbalance
                if closes[idx] < opens[idx] and highs[idx] < prev_low:
                    imbalances.append({
                        'top': prev_low,
                        'bottom': highs[idx],
                        'type': 'BEARISH_IMB',
                        'candles_ago': i
                    })

    return imbalances[:5]

def analyze_pair(symbol):
    """Full ICT structure analysis"""
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            return None

        # Get H1 and H4 data
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='14D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='7D')

        if h4 is None or h1 is None:
            return None

        price = api.get_latest_asking_price(inst_id)

        # Find structures on H4
        h4_bull_ob, h4_bear_ob = find_order_blocks(h4)
        h4_bull_fvg, h4_bear_fvg = find_fvgs(h4)

        # Find structures on H1
        h1_bull_ob, h1_bear_ob = find_order_blocks(h1)
        h1_bull_fvg, h1_bear_fvg = find_fvgs(h1)
        h1_imbalances = find_imbalances(h1)

        # Determine pip multiplier
        if 'JPY' in symbol:
            pip_mult = 100
        elif symbol in ['XAUUSD', 'ETHUSD', 'BTCUSD']:
            pip_mult = 10
        else:
            pip_mult = 10000

        return {
            'symbol': symbol,
            'price': price,
            'pip_mult': pip_mult,
            'h4_bull_ob': h4_bull_ob,
            'h4_bear_ob': h4_bear_ob,
            'h4_bull_fvg': h4_bull_fvg,
            'h4_bear_fvg': h4_bear_fvg,
            'h1_bull_ob': h1_bull_ob,
            'h1_bear_ob': h1_bear_ob,
            'h1_bull_fvg': h1_bull_fvg,
            'h1_bear_fvg': h1_bear_fvg,
            'h1_imbalances': h1_imbalances,
        }
    except Exception as e:
        return None

print('=' * 80)
print('ICT STRUCTURE ANALYSIS | %s' % datetime.now().strftime('%I:%M %p'))
print('=' * 80)
print()
print('Order Blocks (OB): Last opposing candle before strong move')
print('Fair Value Gaps (FVG): 3-candle gap pattern - price tends to fill')
print('Imbalances (IMB): Large move with no candle overlap')
print()

for symbol in PAIRS:
    result = analyze_pair(symbol)
    if not result:
        continue

    price = result['price']
    pip_mult = result['pip_mult']

    print('=' * 80)
    print('%s | Price: %.5f' % (symbol, price))
    print('=' * 80)

    # Bullish structures (support - below price)
    print()
    print('BULLISH STRUCTURES (Support - potential LONG entries):')
    print('-' * 60)

    structures_below = []

    for ob in result['h4_bull_ob']:
        if ob['low'] < price:
            dist = (price - ob['low']) * pip_mult
            structures_below.append(('H4 Bullish OB', ob['low'], ob['high'], dist))

    for ob in result['h1_bull_ob']:
        if ob['low'] < price:
            dist = (price - ob['low']) * pip_mult
            structures_below.append(('H1 Bullish OB', ob['low'], ob['high'], dist))

    for fvg in result['h4_bull_fvg']:
        if fvg['bottom'] < price:
            dist = (price - fvg['top']) * pip_mult
            structures_below.append(('H4 Bullish FVG', fvg['bottom'], fvg['top'], dist))

    for fvg in result['h1_bull_fvg']:
        if fvg['bottom'] < price:
            dist = (price - fvg['top']) * pip_mult
            structures_below.append(('H1 Bullish FVG', fvg['bottom'], fvg['top'], dist))

    for imb in result['h1_imbalances']:
        if imb['type'] == 'BULLISH_IMB' and imb['bottom'] < price:
            dist = (price - imb['top']) * pip_mult
            structures_below.append(('H1 Bullish IMB', imb['bottom'], imb['top'], dist))

    # Sort by distance
    structures_below.sort(key=lambda x: x[3])

    if structures_below:
        for struct in structures_below[:4]:
            print('  %s: %.5f - %.5f (%.1f pips away)' % (struct[0], struct[1], struct[2], struct[3]))
    else:
        print('  No bullish structures found below price')

    # Bearish structures (resistance - above price)
    print()
    print('BEARISH STRUCTURES (Resistance - potential SHORT entries):')
    print('-' * 60)

    structures_above = []

    for ob in result['h4_bear_ob']:
        if ob['high'] > price:
            dist = (ob['high'] - price) * pip_mult
            structures_above.append(('H4 Bearish OB', ob['low'], ob['high'], dist))

    for ob in result['h1_bear_ob']:
        if ob['high'] > price:
            dist = (ob['high'] - price) * pip_mult
            structures_above.append(('H1 Bearish OB', ob['low'], ob['high'], dist))

    for fvg in result['h4_bear_fvg']:
        if fvg['top'] > price:
            dist = (fvg['bottom'] - price) * pip_mult
            structures_above.append(('H4 Bearish FVG', fvg['bottom'], fvg['top'], dist))

    for fvg in result['h1_bear_fvg']:
        if fvg['top'] > price:
            dist = (fvg['bottom'] - price) * pip_mult
            structures_above.append(('H1 Bearish FVG', fvg['bottom'], fvg['top'], dist))

    for imb in result['h1_imbalances']:
        if imb['type'] == 'BEARISH_IMB' and imb['top'] > price:
            dist = (imb['bottom'] - price) * pip_mult
            structures_above.append(('H1 Bearish IMB', imb['bottom'], imb['top'], dist))

    # Sort by distance
    structures_above.sort(key=lambda x: x[3])

    if structures_above:
        for struct in structures_above[:4]:
            print('  %s: %.5f - %.5f (%.1f pips away)' % (struct[0], struct[1], struct[2], struct[3]))
    else:
        print('  No bearish structures found above price')

    print()

print('=' * 80)
print('TRADING NOTES:')
print('- Price tends to return to FVGs to "fill" them')
print('- Order Blocks act as support/resistance zones')
print('- Look for price to tap OB/FVG + rejection for entry')
print('=' * 80)
