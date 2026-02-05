#!/usr/bin/env python
"""Asian Session Monitor - STRICT ICT+QUANT Framework"""

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

# Asian session pairs + key majors
PAIRS = ['USDJPY', 'AUDUSD', 'NZDUSD', 'AUDJPY', 'NZDJPY', 'AUDNZD', 'GBPUSD', 'EURUSD', 'XAUUSD', 'CADJPY', 'ETHUSD']

def analyze(symbol):
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

        daily_bias = 'BULL' if d1c[-1] > d1c[-5] else 'BEAR'
        htf_bias = 'BULL' if h4c[-1] > h4c[-5] else 'BEAR'
        aligned = daily_bias == htf_bias

        deltas = np.diff(h1c[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        rsi = 100 - (100 / (1 + np.mean(gains)/(np.mean(losses) if np.mean(losses) > 0 else 0.0001)))

        week_high, week_low = max(h4h), min(h4l)
        equilibrium = (week_high + week_low) / 2
        price_pos = (current - week_low) / (week_high - week_low) if week_high != week_low else 0.5
        zone = 'PREM' if current > equilibrium else 'DISC'

        swing_high, swing_low = max(h1h[-24:]), min(h1l[-24:])

        if htf_bias == 'BULL':
            ote_top = swing_low + (swing_high - swing_low) * 0.38
            ote_bottom = swing_low + (swing_high - swing_low) * 0.21
        else:
            ote_bottom = swing_high - (swing_high - swing_low) * 0.38
            ote_top = swing_high - (swing_high - swing_low) * 0.21
        in_ote = current >= ote_bottom and current <= ote_top

        # STRICT confluence counting per framework
        conf = 0
        conf_list = []
        direction = 'LONG' if htf_bias == 'BULL' else 'SHORT'

        if htf_bias == 'BULL':
            if aligned:
                conf += 1
                conf_list.append('Aligned')
            if rsi < 40:
                conf += 1
                conf_list.append('RSI<40')
            if zone == 'DISC':
                conf += 1
                conf_list.append('Discount')
            if in_ote:
                conf += 1
                conf_list.append('OTE')
            if price_pos < 0.35:
                conf += 1
                conf_list.append('Deep')
        else:
            if aligned:
                conf += 1
                conf_list.append('Aligned')
            if rsi > 60:
                conf += 1
                conf_list.append('RSI>60')
            if zone == 'PREM':
                conf += 1
                conf_list.append('Premium')
            if in_ote:
                conf += 1
                conf_list.append('OTE')
            if price_pos > 0.65:
                conf += 1
                conf_list.append('Deep')

        # Calculate trade levels
        tr_list = [max(h1h[i] - h1l[i], abs(h1h[i] - h1c[i-1]), abs(h1l[i] - h1c[i-1])) for i in range(-14, 0)]
        atr = np.mean(tr_list)

        if htf_bias == 'BULL':
            sl = swing_low - atr * 0.5
            tp = swing_high
            risk = current - sl
            reward = tp - current
        else:
            sl = swing_high + atr * 0.5
            tp = swing_low
            risk = sl - current
            reward = current - tp
        rr = reward / risk if risk > 0 else 0

        return {
            'symbol': symbol,
            'current': current,
            'htf_bias': htf_bias,
            'aligned': aligned,
            'direction': direction,
            'rsi': rsi,
            'zone': zone,
            'price_pos': price_pos,
            'in_ote': in_ote,
            'conf': conf,
            'conf_list': conf_list,
            'sl': sl,
            'tp': tp,
            'rr': rr
        }
    except:
        return None

# Run monitoring loop
for cycle in range(15):
    try:
        state = api.get_account_state()
        equity = state['balance'] + state['openGrossPnL']
    except:
        equity = 0

    print('=' * 70)
    print('ASIAN SESSION - STRICT ICT+QUANT (RSI must align)')
    print('Time: %s | Equity: $%.2f | Cycle %d/15' % (datetime.now().strftime('%I:%M %p'), equity, cycle+1))
    print('=' * 70)
    print()
    print('Rule: LONG needs RSI<40 | SHORT needs RSI>60')
    print()
    print('%-8s %-10s %-5s %5s %-5s %4s %4s %s' % ('PAIR', 'PRICE', 'BIAS', 'RSI', 'ZONE', 'POS', 'CONF', 'STATUS'))
    print('-' * 70)

    valid_setups = []

    for symbol in PAIRS:
        r = analyze(symbol)
        if r:
            if r['current'] > 1000:
                price = '%.2f' % r['current']
            elif r['current'] > 100:
                price = '%.2f' % r['current']
            elif 'JPY' in symbol:
                price = '%.3f' % r['current']
            else:
                price = '%.5f' % r['current']

            status = ''

            # Check if RSI aligns with direction
            rsi_ok = (r['direction'] == 'LONG' and r['rsi'] < 40) or (r['direction'] == 'SHORT' and r['rsi'] > 60)

            if r['conf'] >= 3 and rsi_ok:
                status = '*** VALID %s (R:R 1:%.1f) ***' % (r['direction'], r['rr'])
                valid_setups.append(r)
            elif r['conf'] >= 3:
                status = '%s blocked (RSI %.0f)' % (r['direction'], r['rsi'])
            elif r['conf'] >= 2:
                status = '%s developing' % r['direction']
            elif r['in_ote']:
                status = 'In OTE'

            print('%-8s %-10s %-5s %5.1f %-5s %3.0f%% %4d %s' % (
                symbol, price, r['htf_bias'], r['rsi'], r['zone'], r['price_pos']*100, r['conf'], status
            ))

    print('-' * 70)

    if valid_setups:
        print()
        print('!' * 70)
        print('VALID SETUPS READY FOR ENTRY:')
        for s in valid_setups:
            print()
            print('>>> %s %s <<<' % (s['symbol'], s['direction']))
            print('    Confluences: %s' % ', '.join(s['conf_list']))
            if s['current'] > 100:
                print('    Entry: %.2f | SL: %.2f | TP: %.2f' % (s['current'], s['sl'], s['tp']))
            else:
                print('    Entry: %.5f | SL: %.5f | TP: %.5f' % (s['current'], s['sl'], s['tp']))
            print('    R:R = 1:%.1f' % s['rr'])
        print()
        print('!' * 70)
    else:
        print()
        print('No valid setups. USDJPY has structure but RSI too low for SHORT.')
        print('Waiting for RSI alignment or new opportunities...')

    print()

    if cycle < 14:
        print('Next scan in 60 seconds...')
        print()
        time.sleep(60)

print('=' * 70)
print('Monitoring complete - 15 cycles finished')
print('=' * 70)
