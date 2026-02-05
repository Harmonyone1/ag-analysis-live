#!/usr/bin/env python
"""Analyze TSLA for potential setup"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0.001
    return 100 - (100 / (1 + avg_gain / avg_loss))

symbol = 'TSLA'
inst_id = name_to_id.get(symbol)

if not inst_id:
    print('TSLA not found in instruments')
else:
    try:
        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='14D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='7D')

        price = api.get_latest_asking_price(inst_id)

        print('=' * 60)
        print('TSLA ANALYSIS | %s' % datetime.now().strftime('%I:%M:%S %p'))
        print('=' * 60)
        print()
        print('Current Price: $%.2f' % price)
        print()

        # HTF Bias
        if d1 is not None and len(d1) > 5:
            d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-5] else 'BEAR'
            print('Daily Bias: %s' % d1_bias)
        else:
            d1_bias = 'N/A'
            print('Daily Bias: N/A')

        if h4 is not None and len(h4) > 5:
            h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-5] else 'BEAR'
            print('4H Bias: %s' % h4_bias)
        else:
            h4_bias = 'N/A'
            print('4H Bias: N/A')

        htf_aligned = d1_bias == h4_bias
        print('HTF Aligned: %s' % ('YES' if htf_aligned else 'NO'))
        print()

        # RSI
        if h1 is not None:
            rsi = calc_rsi(h1['c'].values)
            print('RSI (H1): %.1f' % rsi)
            if rsi < 30:
                print('  Status: OVERSOLD')
            elif rsi > 70:
                print('  Status: OVERBOUGHT')
            else:
                print('  Status: NEUTRAL')
        print()

        # Zone
        if h4 is not None and len(h4) > 20:
            high_20 = max(h4['h'].values[-20:])
            low_20 = min(h4['l'].values[-20:])
            range_pct = (price - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50

            if range_pct < 30:
                zone = 'DISCOUNT'
            elif range_pct > 70:
                zone = 'PREMIUM'
            else:
                zone = 'EQUILIBRIUM'

            print('Zone: %s (%.0f%%)' % (zone, range_pct))
            print('20-period Range: $%.2f - $%.2f' % (low_20, high_20))
        print()

        # Framework Check
        print('FRAMEWORK CHECK:')
        print('-' * 40)

        valid_long = htf_aligned and d1_bias == 'BULL' and range_pct < 30 and rsi < 40
        valid_short = htf_aligned and d1_bias == 'BEAR' and range_pct > 70 and rsi > 60

        if valid_long:
            print('>>> VALID LONG SETUP <<<')
        elif valid_short:
            print('>>> VALID SHORT SETUP <<<')
        else:
            print('Not a valid framework setup')
            if not htf_aligned:
                print('  - HTF not aligned')
            if range_pct >= 30 and range_pct <= 70:
                print('  - In equilibrium zone (need DISC or PREM)')
            if rsi >= 40 and rsi <= 60:
                print('  - RSI neutral (need <40 for LONG or >60 for SHORT)')

    except Exception as e:
        print('Error: %s' % str(e))
