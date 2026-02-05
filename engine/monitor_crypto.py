#!/usr/bin/env python
"""Monitor ETHUSD and BTCUSD for valid ICT+Quant setups"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time
from datetime import datetime

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

def analyze_symbol(symbol):
    inst_id = name_to_id.get(symbol)
    if not inst_id:
        return None

    try:
        price = api.get_latest_asking_price(inst_id)
        time.sleep(0.1)

        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='6M')
        time.sleep(0.1)

        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='1M')
        time.sleep(0.1)

        if d1 is None or h4 is None or len(d1) < 50 or len(h4) < 50:
            return None

        d1_trend = get_trend(d1['c'].values)
        h4_trend = get_trend(h4['c'].values)
        h4_rsi = calculate_rsi(h4['c'].values)

        high_52 = max(d1['h'].values[-260:]) if len(d1) >= 260 else max(d1['h'].values)
        low_52 = min(d1['l'].values[-260:]) if len(d1) >= 260 else min(d1['l'].values)
        range_pct = ((price - low_52) / (high_52 - low_52)) * 100 if high_52 != low_52 else 50

        if range_pct < 30:
            zone = 'DISC'
        elif range_pct > 70:
            zone = 'PREM'
        else:
            zone = 'EQ'

        valid_long = d1_trend == 'BULL' and h4_trend == 'BULL' and zone == 'DISC' and h4_rsi < 40
        valid_short = d1_trend == 'BEAR' and h4_trend == 'BEAR' and zone == 'PREM' and h4_rsi > 60

        return {
            'symbol': symbol,
            'price': price,
            'd1': d1_trend,
            'h4': h4_trend,
            'rsi': h4_rsi,
            'zone': zone,
            'range_pct': range_pct,
            'valid_long': valid_long,
            'valid_short': valid_short
        }
    except Exception as e:
        return None

print('=' * 70)
print('CRYPTO MONITOR | ETHUSD + BTCUSD')
print('Started %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 70)
print()
print('Checking every 60 seconds for valid setups')
print('LONG: HTF BULL + DISC zone + RSI < 40')
print('SHORT: HTF BEAR + PREM zone + RSI > 60')
print()

symbols = ['ETHUSD', 'BTCUSD']

for cycle in range(1, 31):
    now = datetime.now().strftime('%I:%M:%S %p')

    print('=' * 70)
    print('[%s] Cycle %d/30' % (now, cycle))
    print('=' * 70)

    for symbol in symbols:
        result = analyze_symbol(symbol)

        if result:
            print('%s: $%.2f | D1:%s H4:%s | %s (%.0f%%) | RSI:%.1f' % (
                result['symbol'], result['price'], result['d1'], result['h4'],
                result['zone'], result['range_pct'], result['rsi']))

            if result['valid_long']:
                print()
                print('!' * 70)
                print('>>> %s VALID LONG SETUP <<<' % symbol)
                print('!' * 70)
                print('Entry: $%.2f' % result['price'])
                print('SL: $%.2f (3%%)' % (result['price'] * 0.97))
                print('TP: $%.2f (6%%)' % (result['price'] * 1.06))
                print('!' * 70)
                print()
            elif result['valid_short']:
                print()
                print('!' * 70)
                print('>>> %s VALID SHORT SETUP <<<' % symbol)
                print('!' * 70)
                print('Entry: $%.2f' % result['price'])
                print('SL: $%.2f (3%%)' % (result['price'] * 1.03))
                print('TP: $%.2f (6%%)' % (result['price'] * 0.94))
                print('!' * 70)
                print()
            else:
                missing = []
                if result['d1'] != result['h4']:
                    missing.append('HTF mixed')
                if result['zone'] == 'EQ':
                    missing.append('EQ zone')
                if 40 <= result['rsi'] <= 60:
                    missing.append('RSI neutral')
                if missing:
                    print('  -> Missing: %s' % ', '.join(missing))
        else:
            print('%s: Error fetching data' % symbol)

    print()

    if cycle < 30:
        time.sleep(60)

print('=' * 70)
print('Monitor session complete')
print('=' * 70)
