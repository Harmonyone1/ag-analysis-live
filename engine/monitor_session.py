#!/usr/bin/env python
"""Monitor positions and scan for new setups"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time
import numpy as np
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_trend(prices):
    if len(prices) < 50:
        return 'NEUTRAL'
    ma20 = np.mean(prices[-20:])
    ma50 = np.mean(prices[-50:])
    current = prices[-1]
    if current > ma20 > ma50:
        return 'BULL'
    elif current < ma20 < ma50:
        return 'BEAR'
    return 'NEUTRAL'

def quick_scan(symbol):
    inst_id = name_to_id.get(symbol)
    if not inst_id:
        return None
    try:
        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='6M')
        time.sleep(0.08)
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='1M')
        time.sleep(0.08)

        if d1 is None or h4 is None or len(d1) < 50 or len(h4) < 50:
            return None

        d1_trend = get_trend(d1['c'].values)
        h4_trend = get_trend(h4['c'].values)
        rsi = calculate_rsi(h4['c'].values)

        high_52 = max(d1['h'].values[-260:]) if len(d1) >= 260 else max(d1['h'].values)
        low_52 = min(d1['l'].values[-260:]) if len(d1) >= 260 else min(d1['l'].values)
        current = d1['c'].values[-1]
        range_pct = ((current - low_52) / (high_52 - low_52)) * 100 if high_52 != low_52 else 50

        zone = 'DISC' if range_pct < 30 else 'PREM' if range_pct > 70 else 'EQ'

        valid_long = (d1_trend == 'BULL' and h4_trend == 'BULL' and zone == 'DISC' and rsi < 40)
        valid_short = (d1_trend == 'BEAR' and h4_trend == 'BEAR' and zone == 'PREM' and rsi > 60)

        if valid_long:
            return {'symbol': symbol, 'signal': 'LONG', 'rsi': rsi, 'range_pct': range_pct}
        if valid_short:
            return {'symbol': symbol, 'signal': 'SHORT', 'rsi': rsi, 'range_pct': range_pct}
        return None
    except:
        return None

symbols = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF',
    'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'EURAUD', 'GBPAUD',
    'AUDNZD', 'AUDCAD', 'NZDCAD', 'EURCAD', 'GBPCAD',
    'US30', 'US500', 'US100', 'GER40',
    'XAUUSD', 'XAGUSD', 'XTIUSD',
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
    'JPM', 'BAC', 'JNJ', 'PFE', 'XOM', 'CVX',
    'BTCUSD', 'ETHUSD',
]

print('=' * 70)
print('CONTINUOUS MONITOR | Started %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 70)
print()

for cycle in range(1, 21):
    now = datetime.now().strftime('%I:%M:%S %p')

    # Get account state
    state = api.get_account_state()
    equity = state['balance'] + state['openGrossPnL']
    pnl = state['openGrossPnL']

    print('=' * 70)
    print('[%s] Cycle %d/20 | Equity: $%.2f | P/L: $%.2f' % (now, cycle, equity, pnl))
    print('=' * 70)

    # Get positions
    positions = api.get_all_positions()
    if positions is not None and len(positions) > 0:
        print('POSITIONS:')
        pos_summary = {}
        for _, pos in positions.iterrows():
            tid = str(pos.get('tradableInstrumentId', ''))
            qty = pos.get('qty', 0)
            upnl = pos.get('unrealizedPnL', 0)
            for name, id_val in name_to_id.items():
                if str(id_val) == tid:
                    if name not in pos_summary:
                        pos_summary[name] = {'qty': 0, 'pnl': 0}
                    pos_summary[name]['qty'] += qty
                    pos_summary[name]['pnl'] += upnl
                    break

        for sym, data in pos_summary.items():
            print('  %s: %.2f lots | P/L: $%.2f' % (sym, data['qty'], data['pnl']))

    # Quick scan for new setups
    print()
    print('Scanning for new setups...')
    found = []
    for sym in symbols:
        result = quick_scan(sym)
        if result:
            found.append(result)

    if found:
        print('NEW SETUPS FOUND:')
        for f in found:
            print('  >>> %s %s | RSI: %.1f | Zone: %.0f%%' % (
                f['symbol'], f['signal'], f['rsi'], f['range_pct']))
    else:
        print('  No new valid setups')

    print()

    if cycle < 20:
        time.sleep(90)

print('=' * 70)
print('Monitor session complete')
print('=' * 70)
