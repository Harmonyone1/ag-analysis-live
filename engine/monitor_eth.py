#!/usr/bin/env python
"""Monitor ETHUSD for valid ICT+Quant setup"""

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
inst_id = name_to_id.get('ETHUSD')

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

print('=' * 70)
print('ETHUSD MONITOR | Started %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 70)
print()
print('Monitoring for valid setup (checking every 60 seconds)')
print('LONG: HTF BULL aligned + DISC zone + RSI < 40')
print('SHORT: HTF BEAR aligned + PREM zone + RSI > 60')
print()

for cycle in range(1, 31):  # 30 cycles = ~30 minutes
    now = datetime.now().strftime('%I:%M:%S %p')

    try:
        # Get current price
        price = api.get_latest_asking_price(inst_id)

        # Get D1 data
        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='6M')
        time.sleep(0.1)

        # Get H4 data
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='1M')
        time.sleep(0.1)

        if d1 is None or h4 is None or len(d1) < 50 or len(h4) < 50:
            print('[%s] Error getting data' % now)
            time.sleep(60)
            continue

        d1_trend = get_trend(d1['c'].values)
        h4_trend = get_trend(h4['c'].values)
        h4_rsi = calculate_rsi(h4['c'].values)

        # Calculate zone
        high_52 = max(d1['h'].values[-260:]) if len(d1) >= 260 else max(d1['h'].values)
        low_52 = min(d1['l'].values[-260:]) if len(d1) >= 260 else min(d1['l'].values)
        range_pct = ((price - low_52) / (high_52 - low_52)) * 100 if high_52 != low_52 else 50

        if range_pct < 30:
            zone = 'DISC'
        elif range_pct > 70:
            zone = 'PREM'
        else:
            zone = 'EQ'

        # Check for valid setups
        valid_long = d1_trend == 'BULL' and h4_trend == 'BULL' and zone == 'DISC' and h4_rsi < 40
        valid_short = d1_trend == 'BEAR' and h4_trend == 'BEAR' and zone == 'PREM' and h4_rsi > 60

        print('[%s] Cycle %d/30 | $%.2f | D1:%s H4:%s | Zone:%s (%.0f%%) | RSI:%.1f' % (
            now, cycle, price, d1_trend, h4_trend, zone, range_pct, h4_rsi))

        if valid_long:
            print()
            print('!' * 70)
            print('>>> VALID LONG SETUP DETECTED <<<')
            print('!' * 70)
            print()
            print('Entry: $%.2f' % price)
            print('SL: $%.2f (3%%)' % (price * 0.97))
            print('TP: $%.2f (6%%)' % (price * 1.06))
            print()
        elif valid_short:
            print()
            print('!' * 70)
            print('>>> VALID SHORT SETUP DETECTED <<<')
            print('!' * 70)
            print()
            print('Entry: $%.2f' % price)
            print('SL: $%.2f (3%%)' % (price * 1.03))
            print('TP: $%.2f (6%%)' % (price * 0.94))
            print()
        else:
            # Show what's missing
            missing = []
            if d1_trend != h4_trend:
                missing.append('HTF mixed')
            if zone == 'EQ':
                missing.append('in EQ zone')
            if 40 <= h4_rsi <= 60:
                missing.append('RSI neutral')
            if missing:
                print('  Missing: %s' % ', '.join(missing))

    except Exception as e:
        print('[%s] Error: %s' % (now, str(e)))

    if cycle < 30:
        time.sleep(60)

print()
print('=' * 70)
print('Monitor session complete')
print('=' * 70)
