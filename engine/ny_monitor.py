#!/usr/bin/env python
"""NY Session Monitor - Strict ICT+Quant Framework"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time
from datetime import datetime

from config import get_api
api = get_api()

pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF',
         'GBPJPY', 'EURJPY', 'AUDJPY', 'XAUUSD']

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

def get_session():
    hour = datetime.now().hour
    if 19 <= hour or hour < 22:
        return "ASIAN"
    elif 2 <= hour < 5:
        return "LONDON"
    elif 8 <= hour < 11:
        return "NY"
    else:
        return "OFF-PEAK"

def analyze_pair(symbol):
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            return None

        # Get price data
        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='7D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='3D')

        if d1 is None or h4 is None or h1 is None:
            return None
        if len(d1) < 10 or len(h4) < 20 or len(h1) < 24:
            return None

        # Current price
        price = api.get_latest_asking_price(inst_id)

        # HTF Bias
        d1_close = d1['c'].iloc[-1]
        d1_open = d1['o'].iloc[-5]
        d1_bias = 'BULL' if d1_close > d1_open else 'BEAR'

        h4_close = h4['c'].iloc[-1]
        h4_open = h4['o'].iloc[-5]
        h4_bias = 'BULL' if h4_close > h4_open else 'BEAR'

        htf_aligned = d1_bias == h4_bias
        htf_bias = d1_bias if htf_aligned else 'MIXED'

        # RSI calculation
        closes = h1['c'].values[-15:]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Range position
        high_20 = max(h4['h'].values[-20:])
        low_20 = min(h4['l'].values[-20:])
        range_pct = (price - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50

        # Zone
        if range_pct < 30:
            zone = 'DISC'
        elif range_pct > 70:
            zone = 'PREM'
        else:
            zone = 'EQ'

        # OTE check (62-79% retracement)
        swing_high = max(h4['h'].values[-10:])
        swing_low = min(h4['l'].values[-10:])
        swing_range = swing_high - swing_low
        if swing_range > 0:
            retrace = (swing_high - price) / swing_range * 100
            in_ote = 62 <= retrace <= 79
        else:
            in_ote = False

        # Count confluences for LONG
        long_conf = 0
        if range_pct < 30:
            long_conf += 1
        if range_pct < 20:
            long_conf += 1
        if in_ote:
            long_conf += 1
        if zone == 'DISC':
            long_conf += 1

        # Count confluences for SHORT
        short_conf = 0
        if range_pct > 70:
            short_conf += 1
        if range_pct > 80:
            short_conf += 1
        if in_ote:
            short_conf += 1
        if zone == 'PREM':
            short_conf += 1

        # Determine best direction
        if htf_bias == 'BULL' and long_conf >= 2:
            direction = 'LONG'
            confluences = long_conf
            rsi_ok = rsi < 40
        elif htf_bias == 'BEAR' and short_conf >= 2:
            direction = 'SHORT'
            confluences = short_conf
            rsi_ok = rsi > 60
        else:
            direction = None
            confluences = 0
            rsi_ok = False

        return {
            'symbol': symbol,
            'price': price,
            'htf_bias': htf_bias,
            'rsi': rsi,
            'zone': zone,
            'range_pct': range_pct,
            'direction': direction,
            'confluences': confluences,
            'rsi_ok': rsi_ok,
            'htf_aligned': htf_aligned
        }
    except Exception as e:
        return None

print('=' * 70)
print('MONITORING - Strict ICT+Quant (HTF aligned + RSI confirmed)')
print('Balance: $340.65 | Waiting for valid setups')
print('=' * 70)
print()
print('Rule: LONG = HTF Bull + RSI<40 + Discount')
print('Rule: SHORT = HTF Bear + RSI>60 + Premium')
print()

for cycle in range(20):  # 20 cycles x 3 min = 60 minutes
    session = get_session()
    now = datetime.now().strftime('%I:%M %p')

    print('=' * 70)
    print('%s SESSION | Time: %s | Cycle %d/20' % (session, now, cycle + 1))
    print('=' * 70)
    print()
    print('%-8s %10s %6s %6s %5s %5s %4s %s' % (
        'PAIR', 'PRICE', 'HTF', 'RSI', 'ZONE', '%RNG', 'CONF', 'STATUS'
    ))
    print('-' * 70)

    valid_setups = []

    for symbol in pairs:
        result = analyze_pair(symbol)
        if result is None:
            continue

        # Determine status
        if result['direction'] and result['confluences'] >= 3:
            if result['rsi_ok']:
                status = '*** VALID %s ***' % result['direction']
                valid_setups.append(result)
            else:
                rsi_need = '<40' if result['direction'] == 'LONG' else '>60'
                status = '%s blocked (RSI %.0f, need %s)' % (result['direction'], result['rsi'], rsi_need)
        elif result['direction'] and result['confluences'] >= 2:
            status = '%s developing (%d conf)' % (result['direction'], result['confluences'])
        elif not result['htf_aligned']:
            status = 'HTF mixed'
        else:
            status = ''

        print('%-8s %10.5f %6s %6.1f %5s %5.0f %4d %s' % (
            result['symbol'],
            result['price'],
            result['htf_bias'],
            result['rsi'],
            result['zone'],
            result['range_pct'],
            result['confluences'],
            status
        ))

        time.sleep(0.3)  # Rate limiting

    print()

    if valid_setups:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('VALID SETUP DETECTED!')
        for s in valid_setups:
            print('  %s %s | RSI: %.1f | Confluences: %d' % (
                s['symbol'], s['direction'], s['rsi'], s['confluences']
            ))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    else:
        print('No valid setups. Monitoring continues...')

    print()
    print('Next scan in 3 minutes...')
    print()

    if cycle < 19:
        time.sleep(180)  # 3 minute intervals

print()
print('=' * 70)
print('Monitoring complete. Check for opportunities.')
print('=' * 70)
