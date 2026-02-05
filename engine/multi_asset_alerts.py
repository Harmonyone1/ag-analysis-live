#!/usr/bin/env python
"""Multi-Asset Alert System - ICT+Quant Framework"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# ============================================================
# WATCHLIST - All interesting setups from our scans
# ============================================================

WATCHLIST = {
    # FOREX - Majors & Crosses
    'FOREX': [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD',
        'GBPJPY', 'EURJPY', 'AUDJPY', 'EURNOK'
    ],
    # INDICES
    'INDICES': [
        'US30', 'NAS100', 'SPX500', 'DE40', 'UK100', 'JP225'
    ],
    # STOCKS - Oversold/Interesting
    'STOCKS': [
        'ORCL', 'JNJ', 'JPM', 'AXP', 'PFE', 'NFLX', 'AAPL', 'MSFT', 'TSLA', 'INTC'
    ],
    # CRYPTO
    'CRYPTO': [
        'BTCUSD', 'ETHUSD'
    ],
    # COMMODITIES
    'COMMODITIES': [
        'XAUUSD', 'XAGUSD', 'USOIL'
    ]
}

def analyze(symbol):
    """Full ICT+Quant analysis"""
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

        # HTF Bias
        d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-min(5, len(d1))] else 'BEAR'
        h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-min(5, len(h4))] else 'BEAR'
        htf_aligned = d1_bias == h4_bias
        htf_bias = d1_bias if htf_aligned else 'MIX'

        # RSI
        closes = h1['c'].values[-15:]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0.001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))

        # Range
        high_20 = max(h4['h'].values[-min(20, len(h4)):])
        low_20 = min(h4['l'].values[-min(20, len(h4)):])
        range_pct = (price - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50

        zone = 'DISC' if range_pct < 30 else ('PREM' if range_pct > 70 else 'EQ')

        # Determine signal
        signal = None
        alert_level = 0  # 0=none, 1=watching, 2=developing, 3=valid

        # VALID LONG: HTF Bull + Discount + RSI < 40
        if htf_bias == 'BULL' and zone == 'DISC' and rsi < 40:
            signal = 'LONG'
            alert_level = 3
        # VALID SHORT: HTF Bear + Premium + RSI > 60
        elif htf_bias == 'BEAR' and zone == 'PREM' and rsi > 60:
            signal = 'SHORT'
            alert_level = 3
        # DEVELOPING LONG: HTF Bull + Discount (RSI not confirmed)
        elif htf_bias == 'BULL' and zone == 'DISC':
            signal = 'LONG'
            alert_level = 2 if rsi < 50 else 1
        # DEVELOPING SHORT: HTF Bear + Premium (RSI not confirmed)
        elif htf_bias == 'BEAR' and zone == 'PREM':
            signal = 'SHORT'
            alert_level = 2 if rsi > 50 else 1
        # COUNTER-TREND ALERT: Extreme RSI
        elif rsi < 30 or rsi > 70:
            signal = 'COUNTER'
            alert_level = 1

        return {
            'symbol': symbol,
            'price': price,
            'htf': htf_bias,
            'rsi': rsi,
            'zone': zone,
            'range_pct': range_pct,
            'signal': signal,
            'alert_level': alert_level
        }
    except:
        return None

def get_session():
    hour = datetime.now().hour
    if 19 <= hour or hour < 2:
        return "ASIAN"
    elif 2 <= hour < 5:
        return "LONDON"
    elif 8 <= hour < 11:
        return "NY"
    else:
        return "OFF-PEAK"

# ============================================================
# MAIN MONITORING LOOP
# ============================================================

print('=' * 75)
print('MULTI-ASSET ALERT SYSTEM | ICT+Quant Framework')
print('=' * 75)
print()
print('Monitoring %d assets across 5 categories' % sum(len(v) for v in WATCHLIST.values()))
print('Alert Levels: [3]=VALID SETUP | [2]=DEVELOPING | [1]=WATCHING')
print()
print('Rules:')
print('  LONG  = HTF Bull + Discount Zone + RSI < 40')
print('  SHORT = HTF Bear + Premium Zone + RSI > 60')
print()

for cycle in range(30):  # 30 cycles x 2 min = 1 hour
    session = get_session()
    now = datetime.now().strftime('%I:%M:%S %p')

    # Get account state
    try:
        state = api.get_account_state()
        balance = state['balance']
    except:
        balance = 340.65

    print('=' * 75)
    print('[%s] %s SESSION | Balance: $%.2f | Cycle %d/30' % (now, session, balance, cycle + 1))
    print('=' * 75)

    valid_setups = []
    developing = []
    watching = []

    for category, symbols in WATCHLIST.items():
        results = []
        for symbol in symbols:
            r = analyze(symbol)
            if r:
                results.append(r)
                if r['alert_level'] == 3:
                    valid_setups.append((category, r))
                elif r['alert_level'] == 2:
                    developing.append((category, r))
                elif r['alert_level'] == 1:
                    watching.append((category, r))
            time.sleep(0.2)  # Rate limiting

        if results:
            print()
            print('--- %s ---' % category)
            for r in results:
                level_str = '[%d]' % r['alert_level'] if r['alert_level'] > 0 else '   '
                sig_str = r['signal'] if r['signal'] else ''
                print('%s %-8s %12.5f %5s RSI:%5.1f %5s %4.0f%% %s' % (
                    level_str, r['symbol'], r['price'], r['htf'],
                    r['rsi'], r['zone'], r['range_pct'], sig_str
                ))

    # ALERTS SUMMARY
    print()
    print('-' * 75)

    if valid_setups:
        print()
        print('!!! VALID SETUPS - READY TO TRADE !!!')
        for cat, s in valid_setups:
            print('  >>> %s %s %s | RSI: %.1f | Zone: %s (%.0f%%)' % (
                s['symbol'], s['signal'], s['htf'], s['rsi'], s['zone'], s['range_pct']
            ))
        print()

    if developing:
        print()
        print('DEVELOPING (waiting for RSI confirmation):')
        for cat, s in developing:
            need = '<40' if s['signal'] == 'LONG' else '>60'
            print('  [2] %s %s | RSI: %.1f (need %s) | %s' % (
                s['symbol'], s['signal'], s['rsi'], need, cat
            ))

    if watching:
        print()
        print('WATCHING (extreme RSI or early stage):')
        for cat, s in watching[:5]:  # Limit to top 5
            print('  [1] %s | RSI: %.1f | %s %.0f%%' % (
                s['symbol'], s['rsi'], s['zone'], s['range_pct']
            ))

    if not valid_setups and not developing:
        print('No actionable setups. Continuing to monitor...')

    print()
    print('Next scan in 2 minutes...')
    print()

    if cycle < 29:
        time.sleep(120)

print('=' * 75)
print('Alert monitoring complete')
print('=' * 75)
