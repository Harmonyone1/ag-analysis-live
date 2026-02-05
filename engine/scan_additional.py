#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('../.env', override=True)

from src.config import load_config
from src.adapters import TradeLockerAdapter
import pandas as pd
from src.smc import MultiTimeframeAnalyzer

config = load_config('../.env')
broker = TradeLockerAdapter(
    environment=config.tradelocker.environment,
    email=config.tradelocker.email,
    password=config.tradelocker.password,
    server=config.tradelocker.server,
    acc_num=0,
    account_id=config.tradelocker.acc_num,
)
broker.connect()

def candles_to_df(candles):
    return pd.DataFrame({
        'open': [float(c.open) for c in candles],
        'high': [float(c.high) for c in candles],
        'low': [float(c.low) for c in candles],
        'close': [float(c.close) for c in candles],
    })

def analyze(symbol):
    htf = broker.get_candles(symbol, '4H', limit=100)
    mtf = broker.get_candles(symbol, '1H', limit=100)
    ltf = broker.get_candles(symbol, '15m', limit=100)

    if not all([htf, mtf, ltf]):
        return None

    htf_df = candles_to_df(htf)
    mtf_df = candles_to_df(mtf)
    ltf_df = candles_to_df(ltf)

    analyzer = MultiTimeframeAnalyzer()
    result = analyzer.full_analysis(htf_df, mtf_df, ltf_df, '4H', '1H', '15m')

    current = ltf_df['close'].iloc[-1]
    alignment = result['summary']['alignment']
    strength = result['summary']['alignment_strength']

    mtf_analysis = result['mtf']
    bull_fvgs = [f for f in mtf_analysis.fvgs if f.fvg_type == 'bullish' and not f.is_filled]
    bear_fvgs = [f for f in mtf_analysis.fvgs if f.fvg_type == 'bearish' and not f.is_filled]

    return {
        'symbol': symbol,
        'price': current,
        'htf': result['htf'].bias,
        'mtf': result['mtf'].bias,
        'ltf': result['ltf'].bias,
        'alignment': alignment,
        'strength': strength,
        'bull_fvgs': bull_fvgs,
        'bear_fvgs': bear_fvgs,
    }

symbols = ['BTCUSD', 'XAUUSD', 'US30', 'NAS100', 'SPX500', 'NZDCAD', 'AUDCAD', 'CADCHF', 'CADJPY', 'CHFJPY']

print('ADDITIONAL INSTRUMENTS SCAN')
print('='*60)

aligned = []

for sym in symbols:
    try:
        r = analyze(sym)
        if r:
            status = ''
            if r['alignment'] != 'neutral' and r['strength'] >= 0.67:
                status = ' ** ALIGNED **'
                aligned.append(r)

            if sym in ['BTCUSD', 'XAUUSD']:
                print(f"{r['symbol']}: {r['price']:.2f} | {r['alignment'].upper()} ({r['strength']:.0%}) | HTF:{r['htf']} MTF:{r['mtf']} LTF:{r['ltf']} | FVGs: B{len(r['bull_fvgs'])}/S{len(r['bear_fvgs'])}{status}")
            elif sym in ['US30', 'NAS100', 'SPX500']:
                print(f"{r['symbol']}: {r['price']:.2f} | {r['alignment'].upper()} ({r['strength']:.0%}) | HTF:{r['htf']} MTF:{r['mtf']} LTF:{r['ltf']} | FVGs: B{len(r['bull_fvgs'])}/S{len(r['bear_fvgs'])}{status}")
            else:
                print(f"{r['symbol']}: {r['price']:.5f} | {r['alignment'].upper()} ({r['strength']:.0%}) | HTF:{r['htf']} MTF:{r['mtf']} LTF:{r['ltf']} | FVGs: B{len(r['bull_fvgs'])}/S{len(r['bear_fvgs'])}{status}")
    except Exception as e:
        print(f'{sym}: Error - {str(e)[:40]}')

if aligned:
    print('\n' + '='*60)
    print('ALIGNED SETUPS FOUND:')
    print('='*60)
    for r in aligned:
        print(f"\n{r['symbol']} - {r['alignment'].upper()} ({r['strength']:.0%})")
        if r['alignment'] == 'bullish' and r['bull_fvgs']:
            below = [f for f in r['bull_fvgs'] if f.high < r['price']]
            if below:
                nearest = max(below, key=lambda x: x.high)
                dist = (r['price'] - nearest.high) * 10000 if r['symbol'] not in ['BTCUSD', 'XAUUSD', 'US30', 'NAS100', 'SPX500'] else r['price'] - nearest.high
                print(f"  Entry Zone (FVG): {nearest.low:.5f} - {nearest.high:.5f}")
                print(f"  Distance: {dist:.1f} {'pips' if r['symbol'] not in ['BTCUSD', 'XAUUSD', 'US30', 'NAS100', 'SPX500'] else 'points'}")
        elif r['alignment'] == 'bearish' and r['bear_fvgs']:
            above = [f for f in r['bear_fvgs'] if f.low > r['price']]
            if above:
                nearest = min(above, key=lambda x: x.low)
                dist = (nearest.low - r['price']) * 10000 if r['symbol'] not in ['BTCUSD', 'XAUUSD', 'US30', 'NAS100', 'SPX500'] else nearest.low - r['price']
                print(f"  Entry Zone (FVG): {nearest.low:.5f} - {nearest.high:.5f}")
                print(f"  Distance: {dist:.1f} {'pips' if r['symbol'] not in ['BTCUSD', 'XAUUSD', 'US30', 'NAS100', 'SPX500'] else 'points'}")

broker.disconnect()
