#!/usr/bin/env python
"""Full Market Scan - Forex, Indices, Stocks"""

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

# All instruments to scan
FOREX_MAJORS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF']
FOREX_CROSSES = ['EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY',
                 'EURGBP', 'EURAUD', 'EURNZD', 'EURCAD', 'EURCHF',
                 'GBPAUD', 'GBPNZD', 'GBPCAD', 'GBPCHF',
                 'AUDNZD', 'AUDCAD', 'AUDCHF', 'NZDCAD', 'NZDCHF', 'CADCHF']
INDICES = ['US30', 'NAS100', 'SPX500', 'DE40', 'UK100', 'JP225', 'AUS200']
COMMODITIES = ['XAUUSD', 'XAGUSD', 'USOIL', 'UKOIL']
CRYPTO = ['BTCUSD', 'ETHUSD']
STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'BAC',
          'XOM', 'DIS', 'NFLX', 'AMD', 'BA', 'COIN', 'V', 'MA', 'WMT', 'KO', 'PFE']

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0.001
    return 100 - (100 / (1 + avg_gain / avg_loss))

def analyze(symbol):
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            return None

        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='14D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='7D')

        if d1 is None or h4 is None or h1 is None:
            return None
        if len(d1) < 5 or len(h4) < 10 or len(h1) < 10:
            return None

        price = api.get_latest_asking_price(inst_id)

        d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-5] else 'BEAR'
        h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-5] else 'BEAR'
        htf_aligned = d1_bias == h4_bias

        rsi = calc_rsi(h1['c'].values)

        high_20 = max(h4['h'].values[-20:])
        low_20 = min(h4['l'].values[-20:])
        range_pct = (price - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50

        zone = 'DISC' if range_pct < 30 else ('PREM' if range_pct > 70 else 'EQ')

        return {
            'symbol': symbol,
            'price': price,
            'd1': d1_bias,
            'h4': h4_bias,
            'aligned': htf_aligned,
            'rsi': rsi,
            'zone': zone,
            'pct': range_pct
        }
    except:
        return None

print('=' * 80)
print('FULL MARKET SCAN | %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 80)
print()
print('Framework: LONG = HTF BULL + DISC + RSI<40 | SHORT = HTF BEAR + PREM + RSI>60')
print()

valid_setups = []
developing = []

def scan_category(name, symbols):
    global valid_setups, developing
    results = []
    for symbol in symbols:
        r = analyze(symbol)
        if r:
            results.append(r)
            # Check for valid/developing
            if r['aligned'] and r['d1'] == 'BULL' and r['zone'] == 'DISC' and r['rsi'] < 40:
                valid_setups.append((symbol, 'LONG', r['rsi'], r['pct'], name))
            elif r['aligned'] and r['d1'] == 'BEAR' and r['zone'] == 'PREM' and r['rsi'] > 60:
                valid_setups.append((symbol, 'SHORT', r['rsi'], r['pct'], name))
            elif r['aligned'] and r['d1'] == 'BULL' and r['zone'] == 'DISC' and r['rsi'] < 50:
                developing.append((symbol, 'LONG', r['rsi'], r['pct'], name, 'RSI<40'))
            elif r['aligned'] and r['d1'] == 'BEAR' and r['zone'] == 'PREM' and r['rsi'] > 50:
                developing.append((symbol, 'SHORT', r['rsi'], r['pct'], name, 'RSI>60'))
        time.sleep(0.15)
    return results

# Scan each category
print('Scanning Forex Majors...')
forex_m = scan_category('FX Major', FOREX_MAJORS)

print('Scanning Forex Crosses...')
forex_c = scan_category('FX Cross', FOREX_CROSSES)

print('Scanning Indices...')
indices = scan_category('Index', INDICES)

print('Scanning Commodities...')
commodities = scan_category('Commodity', COMMODITIES)

print('Scanning Crypto...')
crypto = scan_category('Crypto', CRYPTO)

print('Scanning Stocks...')
stocks = scan_category('Stock', STOCKS)

print()
print('=' * 80)

# Display valid setups
if valid_setups:
    print()
    print('!' * 80)
    print('VALID SETUPS - READY TO TRADE')
    print('!' * 80)
    for v in valid_setups:
        print('  >>> %s %s | RSI: %.1f | Zone: %.0f%% | Category: %s' % (v[0], v[1], v[2], v[3], v[4]))
    print()
else:
    print()
    print('No valid setups found.')
    print()

# Display developing
if developing:
    print('DEVELOPING SETUPS (close to valid):')
    print('-' * 60)
    for d in developing[:10]:
        print('  %s %s | RSI: %.1f (need %s) | Zone: %.0f%% | %s' % (d[0], d[1], d[2], d[5], d[3], d[4]))
    print()

# Summary tables
def print_table(name, results):
    if not results:
        return
    print('%s:' % name)
    print('-' * 70)
    for r in results:
        align = 'Y' if r['aligned'] else 'N'
        status = ''
        if r['aligned'] and r['d1'] == 'BULL' and r['zone'] == 'DISC':
            status = 'LONG potential'
        elif r['aligned'] and r['d1'] == 'BEAR' and r['zone'] == 'PREM':
            status = 'SHORT potential'
        print('  %-8s | D1:%s H4:%s | RSI:%5.1f | %4s %3.0f%% | %s' % (
            r['symbol'], r['d1'][0], r['h4'][0], r['rsi'], r['zone'], r['pct'], status))
    print()

print_table('FOREX MAJORS', forex_m)
print_table('FOREX CROSSES', [r for r in forex_c if r['zone'] != 'EQ'])  # Only show non-EQ
print_table('INDICES', indices)
print_table('COMMODITIES', commodities)
print_table('CRYPTO', crypto)
print_table('STOCKS', [r for r in stocks if r['zone'] != 'EQ'])  # Only show non-EQ

print('=' * 80)
print('Scan complete. Total instruments: %d' % (len(forex_m) + len(forex_c) + len(indices) + len(commodities) + len(crypto) + len(stocks)))
print('=' * 80)
