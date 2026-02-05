#!/usr/bin/env python
"""Full Market Scan - All Assets"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

WATCHLIST = {
    'FOREX': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'GBPJPY', 'EURJPY', 'EURNOK'],
    'INDICES': ['US30', 'NAS100', 'SPX500', 'DE40', 'UK100'],
    'STOCKS': ['ORCL', 'JNJ', 'JPM', 'AXP', 'PFE', 'NFLX', 'AAPL', 'MSFT', 'TSLA'],
    'CRYPTO': ['BTCUSD', 'ETHUSD'],
    'COMMODITIES': ['XAUUSD', 'XAGUSD', 'USOIL']
}

def analyze(symbol):
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            return None

        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='7D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='3D')

        if d1 is None or h4 is None or h1 is None:
            return None
        if len(d1) < 5 or len(h4) < 5 or len(h1) < 10:
            return None

        price = api.get_latest_asking_price(inst_id)

        d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-min(5, len(d1))] else 'BEAR'
        h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-min(5, len(h4))] else 'BEAR'
        htf_aligned = d1_bias == h4_bias
        htf_bias = d1_bias if htf_aligned else 'MIX'

        closes = h1['c'].values[-15:]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0.001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))

        high_20 = max(h4['h'].values[-min(20, len(h4)):])
        low_20 = min(h4['l'].values[-min(20, len(h4)):])
        range_pct = (price - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50

        zone = 'DISC' if range_pct < 30 else ('PREM' if range_pct > 70 else 'EQ')

        signal = None
        alert = 0

        if htf_bias == 'BULL' and zone == 'DISC' and rsi < 40:
            signal = 'LONG'
            alert = 3
        elif htf_bias == 'BEAR' and zone == 'PREM' and rsi > 60:
            signal = 'SHORT'
            alert = 3
        elif htf_bias == 'BULL' and zone == 'DISC':
            signal = 'LONG'
            alert = 2
        elif htf_bias == 'BEAR' and zone == 'PREM':
            signal = 'SHORT'
            alert = 2
        elif rsi < 30 or rsi > 70:
            signal = 'EXTREME'
            alert = 1

        return {'symbol': symbol, 'price': price, 'htf': htf_bias, 'rsi': rsi,
                'zone': zone, 'pct': range_pct, 'signal': signal, 'alert': alert}
    except:
        return None

state = api.get_account_state()
balance = state['balance']

print('=' * 75)
print('FULL MARKET SCAN | %s | Balance: $%.2f' % (datetime.now().strftime('%I:%M %p'), balance))
print('=' * 75)
print()
print('[3]=VALID | [2]=DEVELOPING | [1]=WATCHING | RSI: LONG<40, SHORT>60')
print()

valid = []
developing = []
watching = []

for cat, symbols in WATCHLIST.items():
    print('--- %s ---' % cat)
    for sym in symbols:
        r = analyze(sym)
        if r:
            lvl = '[%d]' % r['alert'] if r['alert'] > 0 else '   '
            sig = r['signal'] if r['signal'] else ''
            print('%s %-8s %12.5f %5s RSI:%5.1f %5s %4.0f%% %s' % (
                lvl, r['symbol'], r['price'], r['htf'], r['rsi'], r['zone'], r['pct'], sig
            ))
            if r['alert'] == 3:
                valid.append((cat, r))
            elif r['alert'] == 2:
                developing.append((cat, r))
            elif r['alert'] == 1:
                watching.append((cat, r))
    print()

print('=' * 75)
print('SUMMARY')
print('=' * 75)

if valid:
    print()
    print('*** VALID SETUPS - READY TO TRADE ***')
    for cat, s in valid:
        print('  >>> %s %s | HTF: %s | RSI: %.1f | %s (%.0f%%)' % (
            s['symbol'], s['signal'], s['htf'], s['rsi'], s['zone'], s['pct']))

if developing:
    print()
    print('DEVELOPING (waiting for RSI):')
    for cat, s in developing:
        need = '<40' if s['signal'] == 'LONG' else '>60'
        print('  [2] %-8s %s | RSI: %.1f (need %s) | %s' % (
            s['symbol'], s['signal'], s['rsi'], need, cat))

if watching:
    print()
    print('WATCHING (extreme RSI):')
    for cat, s in watching:
        print('  [1] %-8s RSI: %.1f | %s %.0f%% | %s' % (
            s['symbol'], s['rsi'], s['zone'], s['pct'], cat))

if not valid and not developing and not watching:
    print('No alerts at this time.')

print()
print('=' * 75)
