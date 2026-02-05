#!/usr/bin/env python
"""Market Monitor - ICT+QUANT Framework"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np
from datetime import datetime
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Priority pairs to monitor
PAIRS = ['GBPUSD', 'EURUSD', 'XAUUSD', 'ETHUSD', 'USDJPY', 'EURGBP', 'AUDUSD', 'NZDUSD']

def analyze_pair(symbol):
    inst_id = symbol_to_id.get(symbol)
    if not inst_id:
        return None
    try:
        h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='1W')
        h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='3D')
        d1 = api.get_price_history(inst_id, resolution='1D', lookback_period='2W')

        if h4 is None or h1 is None or len(h4) < 10 or len(h1) < 20:
            return None

        current = h1['c'].values[-1]
        h4c, h4h, h4l = h4['c'].values, h4['h'].values, h4['l'].values
        h1c, h1h, h1l = h1['c'].values, h1['h'].values, h1['l'].values
        d1c = d1['c'].values if d1 is not None and len(d1) >= 5 else h4c

        daily_bias = 'BULLISH' if d1c[-1] > d1c[-5] else 'BEARISH'
        htf_bias = 'BULLISH' if h4c[-1] > h4c[-5] else 'BEARISH'
        aligned = daily_bias == htf_bias

        # RSI
        deltas = np.diff(h1c[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.0001
        rsi = 100 - (100 / (1 + np.mean(gains)/avg_loss))

        # ATR
        tr_list = [max(h1h[i] - h1l[i], abs(h1h[i] - h1c[i-1]), abs(h1l[i] - h1c[i-1])) for i in range(-14, 0)]
        atr = np.mean(tr_list)

        # Ranges
        week_high, week_low = max(h4h), min(h4l)
        equilibrium = (week_high + week_low) / 2
        price_pos = (current - week_low) / (week_high - week_low) if week_high != week_low else 0.5
        zone = 'PREMIUM' if current > equilibrium else 'DISCOUNT'

        swing_high, swing_low = max(h1h[-24:]), min(h1l[-24:])

        # OTE
        if htf_bias == 'BULLISH':
            ote_top = swing_low + (swing_high - swing_low) * 0.38
            ote_bottom = swing_low + (swing_high - swing_low) * 0.21
        else:
            ote_bottom = swing_high - (swing_high - swing_low) * 0.38
            ote_top = swing_high - (swing_high - swing_low) * 0.21
        in_ote = current >= ote_bottom and current <= ote_top

        # Count confluences
        conf = 0
        conf_list = []

        if htf_bias == 'BULLISH':
            direction = 'LONG'
            if aligned:
                conf += 1
                conf_list.append('Aligned')
            if rsi < 40:
                conf += 1
                conf_list.append('RSI<40')
            if zone == 'DISCOUNT':
                conf += 1
                conf_list.append('Discount')
            if in_ote:
                conf += 1
                conf_list.append('OTE')
            if price_pos < 0.35:
                conf += 1
                conf_list.append('Deep')
        else:
            direction = 'SHORT'
            if aligned:
                conf += 1
                conf_list.append('Aligned')
            if rsi > 60:
                conf += 1
                conf_list.append('RSI>60')
            if zone == 'PREMIUM':
                conf += 1
                conf_list.append('Premium')
            if in_ote:
                conf += 1
                conf_list.append('OTE')
            if price_pos > 0.65:
                conf += 1
                conf_list.append('Deep')

        return {
            'symbol': symbol,
            'current': current,
            'htf_bias': htf_bias,
            'direction': direction,
            'rsi': rsi,
            'zone': zone,
            'price_pos': price_pos,
            'in_ote': in_ote,
            'conf': conf,
            'conf_list': conf_list,
            'atr': atr,
            'swing_high': swing_high,
            'swing_low': swing_low
        }
    except Exception as e:
        return None

def get_session():
    hour = datetime.now().hour
    if 19 <= hour or hour < 22:
        return 'ASIAN'
    elif 2 <= hour < 5:
        return 'LONDON'
    elif 8 <= hour < 11:
        return 'NEW YORK'
    elif 11 <= hour < 13:
        return 'LONDON CLOSE'
    else:
        return 'OFF-HOURS'

print('=' * 70)
print('MARKET MONITOR - ICT+QUANT Framework')
print('Monitoring: ' + ', '.join(PAIRS))
print('=' * 70)

for cycle in range(20):  # 20 cycles x 60 sec = 20 min
    now = datetime.now().strftime('%I:%M:%S %p')
    session = get_session()

    # Get account status
    try:
        state = api.get_account_state()
        equity = state['balance'] + state['openGrossPnL']
    except:
        equity = 0

    print()
    print('[%s] Session: %s | Equity: $%.2f' % (now, session, equity))
    print('-' * 70)

    setups = []

    for symbol in PAIRS:
        r = analyze_pair(symbol)
        if r:
            # Format price based on instrument
            if symbol in ['XAUUSD', 'BTCUSD']:
                price_fmt = '%.2f' % r['current']
            elif symbol == 'ETHUSD':
                price_fmt = '%.2f' % r['current']
            elif 'JPY' in symbol:
                price_fmt = '%.3f' % r['current']
            else:
                price_fmt = '%.5f' % r['current']

            status = ''
            if r['conf'] >= 3:
                status = ' *** SETUP: %s %s ***' % (r['direction'], '+'.join(r['conf_list']))
                setups.append(r)
            elif r['conf'] >= 2:
                status = ' >> %s (%d conf)' % (r['direction'], r['conf'])

            print('  %s: %s | %s | RSI:%.1f | %s %.0f%% | Conf:%d%s' % (
                symbol, price_fmt, r['htf_bias'], r['rsi'],
                r['zone'], r['price_pos']*100, r['conf'], status
            ))

    if setups:
        print()
        print('!' * 70)
        print('HIGH PROBABILITY SETUPS DETECTED:')
        for s in setups:
            print('  >>> %s %s - %d confluences: %s' % (
                s['symbol'], s['direction'], s['conf'], ', '.join(s['conf_list'])
            ))
            if s['direction'] == 'LONG':
                sl = s['swing_low'] - s['atr'] * 0.5
                tp = s['swing_high']
            else:
                sl = s['swing_high'] + s['atr'] * 0.5
                tp = s['swing_low']
            print('      Entry: %s | SL: %.5f | TP: %.5f' % (
                '%.5f' % s['current'] if s['current'] < 100 else '%.2f' % s['current'],
                sl, tp
            ))
        print('!' * 70)

    print()
    if cycle < 19:
        time.sleep(60)  # Wait 60 seconds between scans

print()
print('Monitoring complete - 20 cycles finished')
print('=' * 70)
