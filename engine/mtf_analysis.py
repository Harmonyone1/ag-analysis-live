#!/usr/bin/env python
"""Multi-Timeframe Analysis - Daily, Weekly, Monthly"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Key pairs to analyze
PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'AUDNZD', 'NZDCAD',
    'XAUUSD', 'ETHUSD', 'BTCUSD',
    'US30', 'NAS100', 'SPX500'
]

def get_bias(data, lookback=5):
    """Determine bias from price data"""
    if data is None or len(data) < lookback:
        return 'N/A'
    close = data['c'].iloc[-1]
    open_lb = data['o'].iloc[-min(lookback, len(data))]
    return 'BULL' if close > open_lb else 'BEAR'

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0.001
    return 100 - (100 / (1 + avg_gain / avg_loss))

print('=' * 85)
print('MULTI-TIMEFRAME ANALYSIS | %s' % datetime.now().strftime('%I:%M %p'))
print('=' * 85)
print()
print('Timeframes: Monthly (M) | Weekly (W) | Daily (D) | 4-Hour (4H) | 1-Hour (1H)')
print()
print('%-8s %10s | %4s %4s %4s %4s | %5s %5s | %s' % (
    'PAIR', 'PRICE', 'M', 'W', 'D', '4H', 'RSI', 'ZONE', 'ALIGNMENT'))
print('-' * 85)

for symbol in PAIRS:
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            continue

        # Get all timeframes
        m1 = api.get_price_history(inst_id, resolution='1M', start_timestamp=0, end_timestamp=0, lookback_period='365D')
        w1 = api.get_price_history(inst_id, resolution='1W', start_timestamp=0, end_timestamp=0, lookback_period='180D')
        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='60D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='14D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='7D')

        price = api.get_latest_asking_price(inst_id)

        # Get bias for each timeframe
        m_bias = get_bias(m1, 3) if m1 is not None and len(m1) > 2 else 'N/A'
        w_bias = get_bias(w1, 4) if w1 is not None and len(w1) > 3 else 'N/A'
        d_bias = get_bias(d1, 5) if d1 is not None and len(d1) > 4 else 'N/A'
        h4_bias = get_bias(h4, 5) if h4 is not None and len(h4) > 4 else 'N/A'

        # RSI
        rsi = calc_rsi(h1['c'].values) if h1 is not None else 50

        # Zone from H4
        if h4 is not None and len(h4) > 10:
            high = max(h4['h'].values[-20:])
            low = min(h4['l'].values[-20:])
            pct = (price - low) / (high - low) * 100 if high != low else 50
            zone = 'DISC' if pct < 30 else ('PREM' if pct > 70 else 'EQ')
        else:
            pct = 50
            zone = 'N/A'

        # Check alignment
        biases = [m_bias, w_bias, d_bias, h4_bias]
        bull_count = biases.count('BULL')
        bear_count = biases.count('BEAR')

        if bull_count == 4:
            alignment = 'ALL BULL ****'
        elif bear_count == 4:
            alignment = 'ALL BEAR ****'
        elif bull_count >= 3:
            alignment = 'MOSTLY BULL'
        elif bear_count >= 3:
            alignment = 'MOSTLY BEAR'
        else:
            alignment = 'MIXED'

        # Format biases
        m_str = 'B' if m_bias == 'BULL' else ('S' if m_bias == 'BEAR' else '-')
        w_str = 'B' if w_bias == 'BULL' else ('S' if w_bias == 'BEAR' else '-')
        d_str = 'B' if d_bias == 'BULL' else ('S' if d_bias == 'BEAR' else '-')
        h4_str = 'B' if h4_bias == 'BULL' else ('S' if h4_bias == 'BEAR' else '-')

        print('%-8s %10.5f |  %s    %s    %s    %s  | %5.1f %5s | %s' % (
            symbol, price, m_str, w_str, d_str, h4_str, rsi, zone, alignment))

    except Exception as e:
        print('%-8s Error: %s' % (symbol, str(e)[:30]))

print()
print('=' * 85)
print('LEGEND: B=BULL, S=BEAR | RSI: <40 oversold, >60 overbought')
print()
print('VALID LONG  = All TF BULL + DISC zone + RSI < 40')
print('VALID SHORT = All TF BEAR + PREM zone + RSI > 60')
print('=' * 85)
