#!/usr/bin/env python
"""Quick Market Scan - Single Pass"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'GBPJPY', 'EURJPY', 'XAUUSD']

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

# Get account state
state = api.get_account_state()
balance = state['balance']
equity = balance + state['openGrossPnL']

session = get_session()
now = datetime.now().strftime('%I:%M %p')

print('=' * 70)
print('QUICK SCAN - %s SESSION | %s | Balance: $%.2f' % (session, now, balance))
print('=' * 70)
print()
print('Rule: LONG = HTF Bull + RSI<40 + Discount zone')
print('Rule: SHORT = HTF Bear + RSI>60 + Premium zone')
print()
print('%-8s %10s %6s %6s %5s %5s %s' % ('PAIR', 'PRICE', 'HTF', 'RSI', 'ZONE', '%RNG', 'STATUS'))
print('-' * 70)

for symbol in pairs:
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            continue

        # Get data
        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='7D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='3D')

        if d1 is None or h4 is None or h1 is None:
            continue
        if len(d1) < 10 or len(h4) < 20 or len(h1) < 24:
            continue

        price = api.get_latest_asking_price(inst_id)

        # HTF Bias
        d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-5] else 'BEAR'
        h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-5] else 'BEAR'
        htf_aligned = d1_bias == h4_bias
        htf_bias = d1_bias if htf_aligned else 'MIX'

        # RSI
        closes = h1['c'].values[-15:]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        rsi = 100 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))

        # Range
        high_20 = max(h4['h'].values[-20:])
        low_20 = min(h4['l'].values[-20:])
        range_pct = (price - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50

        zone = 'DISC' if range_pct < 30 else ('PREM' if range_pct > 70 else 'EQ')

        # Check setups
        status = ''
        if htf_bias == 'BULL' and zone == 'DISC':
            if rsi < 40:
                status = '*** VALID LONG ***'
            else:
                status = 'LONG blocked (RSI %.0f, need <40)' % rsi
        elif htf_bias == 'BEAR' and zone == 'PREM':
            if rsi > 60:
                status = '*** VALID SHORT ***'
            else:
                status = 'SHORT blocked (RSI %.0f, need >60)' % rsi
        elif htf_bias == 'MIX':
            status = 'HTF not aligned'
        elif zone == 'EQ':
            status = 'In equilibrium'
        else:
            # Zone doesn't match bias
            if htf_bias == 'BULL':
                status = 'Waiting for pullback'
            else:
                status = 'Waiting for rally'

        print('%-8s %10.5f %6s %6.1f %5s %5.0f %s' % (
            symbol, price, htf_bias, rsi, zone, range_pct, status
        ))

    except Exception as e:
        print('%-8s Error: %s' % (symbol, str(e)[:30]))

print()
print('NY Session starts at 8 AM EST (~2.5 hours)')
print('=' * 70)
