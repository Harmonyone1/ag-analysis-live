#!/usr/bin/env python
"""NY Session Monitor - Full Session Coverage with Trade Execution Alerts"""

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

# Full watchlist across all asset classes
WATCHLIST = {
    'FOREX': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'GBPJPY', 'EURJPY', 'EURNOK'],
    'INDICES': ['US30', 'NAS100', 'SPX500', 'DE40', 'UK100'],
    'STOCKS': ['ORCL', 'JNJ', 'JPM', 'AXP', 'PFE', 'NFLX', 'AAPL', 'MSFT', 'TSLA', 'INTC'],
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

        # Calculate suggested SL/TP
        if 'JPY' in symbol:
            pip = 0.01
        elif symbol in ['XAUUSD', 'BTCUSD', 'ETHUSD']:
            pip = 1.0
        elif symbol in ['US30', 'NAS100', 'SPX500', 'DE40', 'UK100', 'JP225']:
            pip = 1.0
        else:
            pip = 0.0001

        atr = sum(abs(h1['h'].iloc[i] - h1['l'].iloc[i]) for i in range(-5, 0)) / 5
        sl_pips = round(atr / pip * 1.5, 1)
        tp_pips = round(sl_pips * 2, 1)

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
            alert = 2 if rsi < 50 else 1
        elif htf_bias == 'BEAR' and zone == 'PREM':
            signal = 'SHORT'
            alert = 2 if rsi > 50 else 1

        return {
            'symbol': symbol, 'price': price, 'htf': htf_bias, 'rsi': rsi,
            'zone': zone, 'pct': range_pct, 'signal': signal, 'alert': alert,
            'sl_pips': sl_pips, 'tp_pips': tp_pips
        }
    except:
        return None

print('=' * 80)
print('NY SESSION MONITOR | ICT+Quant Framework')
print('=' * 80)
print()
print('Session: 8:00 AM - 11:00 AM EST (3 hours)')
print('Scanning every 3 minutes for valid setups')
print('Will alert on: [3]=VALID SETUP | [2]=DEVELOPING | [1]=WATCHING')
print()

# NY session is ~3 hours = 180 min / 3 = 60 cycles
for cycle in range(60):
    now = datetime.now()
    time_str = now.strftime('%I:%M:%S %p')

    try:
        state = api.get_account_state()
        balance = state['balance']
        positions = api.get_all_positions()
        open_count = len(positions) if not positions.empty else 0
    except:
        balance = 340.65
        open_count = 0

    print('=' * 80)
    print('[%s] NY SESSION | Cycle %d/60 | Balance: $%.2f | Positions: %d' % (
        time_str, cycle + 1, balance, open_count))
    print('=' * 80)

    valid = []
    developing = []

    for cat, symbols in WATCHLIST.items():
        for sym in symbols:
            r = analyze(sym)
            if r and r['alert'] >= 2:
                if r['alert'] == 3:
                    valid.append((cat, r))
                else:
                    developing.append((cat, r))
            time.sleep(0.15)

    # Show valid setups prominently
    if valid:
        print()
        print('!' * 80)
        print('!!! VALID SETUP DETECTED - READY TO EXECUTE !!!')
        print('!' * 80)
        for cat, s in valid:
            print()
            print('  SYMBOL: %s (%s)' % (s['symbol'], cat))
            print('  SIGNAL: %s' % s['signal'])
            print('  PRICE:  %.5f' % s['price'])
            print('  HTF:    %s (aligned)' % s['htf'])
            print('  RSI:    %.1f' % s['rsi'])
            print('  ZONE:   %s (%.0f%%)' % (s['zone'], s['pct']))
            print('  SL:     %.1f pips | TP: %.1f pips (1:2 R:R)' % (s['sl_pips'], s['tp_pips']))
            print()
        print('!' * 80)

    # Show developing setups
    if developing:
        print()
        print('DEVELOPING SETUPS (waiting for RSI confirmation):')
        for cat, s in developing:
            need = '<40' if s['signal'] == 'LONG' else '>60'
            print('  [2] %-8s %6s | RSI: %5.1f (need %s) | %s %.0f%% | %s' % (
                s['symbol'], s['signal'], s['rsi'], need, s['zone'], s['pct'], cat))

    if not valid and not developing:
        print()
        print('No actionable setups this scan. Continuing to monitor...')

    print()

    # Check if we should continue (within NY session hours roughly 8-11 AM)
    hour = now.hour
    if hour >= 11 and cycle > 20:
        print('NY session ending. Wrapping up monitoring.')
        break

    print('Next scan in 3 minutes...')
    print()

    if cycle < 59:
        time.sleep(180)

print()
print('=' * 80)
print('NY Session monitoring complete')
print('=' * 80)
