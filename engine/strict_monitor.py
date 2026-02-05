#!/usr/bin/env python
"""STRICT Framework Monitor - Only Valid Aligned Setups"""

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

# Full watchlist
ALL_PAIRS = [
    # Majors
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF',
    # JPY crosses
    'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY',
    # EUR crosses
    'EURGBP', 'EURAUD', 'EURNZD', 'EURCAD', 'EURCHF',
    # GBP crosses
    'GBPAUD', 'GBPNZD', 'GBPCAD', 'GBPCHF',
    # AUD/NZD crosses
    'AUDNZD', 'AUDCAD', 'AUDCHF', 'NZDCAD', 'NZDCHF',
    # Commodities
    'XAUUSD', 'XAGUSD', 'USOIL',
    # Indices
    'US30', 'NAS100', 'SPX500', 'DE40', 'UK100',
    # Crypto
    'BTCUSD', 'ETHUSD',
]

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0.001
    return 100 - (100 / (1 + avg_gain / avg_loss))

def analyze_strict(symbol):
    """Strict framework analysis"""
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            return None

        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='14D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='7D')

        if d1 is None or h4 is None or h1 is None:
            return None
        if len(d1) < 10 or len(h4) < 10 or len(h1) < 15:
            return None

        price = api.get_latest_asking_price(inst_id)

        # HTF Bias - MUST be aligned
        d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-5] else 'BEAR'
        h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-5] else 'BEAR'
        htf_aligned = d1_bias == h4_bias

        if not htf_aligned:
            return {'symbol': symbol, 'status': 'HTF_MIXED', 'valid': False}

        htf_bias = d1_bias

        # RSI
        rsi = calc_rsi(h1['c'].values)

        # Zone
        high_20 = max(h4['h'].values[-20:])
        low_20 = min(h4['l'].values[-20:])
        range_pct = (price - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50

        if range_pct < 30:
            zone = 'DISC'
        elif range_pct > 70:
            zone = 'PREM'
        else:
            zone = 'EQ'

        # STRICT VALID SETUP CHECK
        valid_long = htf_bias == 'BULL' and zone == 'DISC' and rsi < 40
        valid_short = htf_bias == 'BEAR' and zone == 'PREM' and rsi > 60

        # Developing check
        dev_long = htf_bias == 'BULL' and zone == 'DISC' and rsi < 50
        dev_short = htf_bias == 'BEAR' and zone == 'PREM' and rsi > 50

        if valid_long:
            return {
                'symbol': symbol, 'price': price, 'htf': htf_bias, 'rsi': rsi,
                'zone': zone, 'pct': range_pct, 'signal': 'LONG', 'valid': True
            }
        elif valid_short:
            return {
                'symbol': symbol, 'price': price, 'htf': htf_bias, 'rsi': rsi,
                'zone': zone, 'pct': range_pct, 'signal': 'SHORT', 'valid': True
            }
        elif dev_long:
            return {
                'symbol': symbol, 'price': price, 'htf': htf_bias, 'rsi': rsi,
                'zone': zone, 'pct': range_pct, 'signal': 'LONG_DEV', 'valid': False,
                'status': 'RSI %.1f (need <40)' % rsi
            }
        elif dev_short:
            return {
                'symbol': symbol, 'price': price, 'htf': htf_bias, 'rsi': rsi,
                'zone': zone, 'pct': range_pct, 'signal': 'SHORT_DEV', 'valid': False,
                'status': 'RSI %.1f (need >60)' % rsi
            }
        else:
            return {'symbol': symbol, 'status': 'NO_SETUP', 'valid': False, 'htf': htf_bias, 'zone': zone, 'rsi': rsi}

    except:
        return None

# Get account state
state = api.get_account_state()
balance = state['balance']

print('=' * 75)
print('STRICT FRAMEWORK MONITOR | Balance: $%.2f' % balance)
print('=' * 75)
print()
print('RULES (NO EXCEPTIONS):')
print('  LONG  = HTF BULL (D1+H4) + DISCOUNT zone + RSI < 40')
print('  SHORT = HTF BEAR (D1+H4) + PREMIUM zone + RSI > 60')
print()
print('Scanning %d instruments every 90 seconds...' % len(ALL_PAIRS))
print()

for cycle in range(40):  # 40 cycles x 90 sec = 1 hour
    now = datetime.now().strftime('%I:%M:%S %p')

    # Update account
    try:
        state = api.get_account_state()
        balance = state['balance']
        equity = balance + state['openGrossPnL']
    except:
        equity = balance

    print('=' * 75)
    print('[%s] Cycle %d/40 | Balance: $%.2f | Equity: $%.2f' % (
        now, cycle + 1, balance, equity))
    print('=' * 75)

    valid_setups = []
    developing = []

    for symbol in ALL_PAIRS:
        result = analyze_strict(symbol)
        if result:
            if result.get('valid'):
                valid_setups.append(result)
            elif result.get('signal') and 'DEV' in result.get('signal', ''):
                developing.append(result)
        time.sleep(0.15)

    # VALID SETUPS
    if valid_setups:
        print()
        print('!' * 75)
        print('>>> VALID FRAMEWORK SETUPS - READY TO TRADE <<<')
        print('!' * 75)
        for s in valid_setups:
            print()
            print('  %s %s' % (s['symbol'], s['signal']))
            print('  Price: %.5f | HTF: %s | RSI: %.1f | Zone: %s (%.0f%%)' % (
                s['price'], s['htf'], s['rsi'], s['zone'], s['pct']))
        print()
        print('!' * 75)
    else:
        print()
        print('No valid setups this scan.')

    # DEVELOPING
    if developing:
        print()
        print('DEVELOPING (waiting for RSI):')
        for d in developing[:5]:
            print('  %s %s | HTF: %s | RSI: %.1f | %s %.0f%% | %s' % (
                d['symbol'], d['signal'].replace('_DEV', ''),
                d['htf'], d['rsi'], d['zone'], d['pct'], d.get('status', '')))

    print()
    print('Next scan in 90 seconds...')
    print()

    if cycle < 39:
        time.sleep(90)

print('=' * 75)
print('Monitoring session complete')
print('=' * 75)
