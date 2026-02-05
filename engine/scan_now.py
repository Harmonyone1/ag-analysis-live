import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

def get_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.0001
    return 100 - (100 / (1 + avg_gain/avg_loss))

watchlist = ['BTCUSD', 'ETHUSD', 'XAUUSD', 'US30', 'NAS100', 'EURUSD', 'GBPUSD']

print('='*70)
print('MARKET SCAN - LONDON SESSION')
print('='*70)

for sym in watchlist:
    inst_id = symbol_to_id.get(sym)
    if not inst_id:
        continue

    time.sleep(0.8)
    try:
        h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='3D')
        m15 = api.get_price_history(inst_id, resolution='15m', lookback_period='1D')

        if h1 is None or m15 is None:
            continue

        price = m15['c'].values[-1]
        m15_rsi = get_rsi(m15['c'].values)
        h1_rsi = get_rsi(h1['c'].values)

        # Trend
        h1_trend = 'UP' if h1['c'].values[-1] > h1['c'].values[-6] else 'DOWN'

        # Momentum (last 3 M15 candles)
        last3 = m15['c'].values[-3:]
        if last3[-1] > last3[-2] > last3[-3]:
            momentum = 'STRONG UP'
        elif last3[-1] < last3[-2] < last3[-3]:
            momentum = 'STRONG DOWN'
        elif last3[-1] > last3[0]:
            momentum = 'UP'
        else:
            momentum = 'DOWN'

        # Range position
        h1_high = max(h1['h'].values[-24:])
        h1_low = min(h1['l'].values[-24:])
        range_pos = (price - h1_low) / (h1_high - h1_low) * 100 if h1_high > h1_low else 50

        # Signal
        signal = ''
        if m15_rsi < 25:
            signal = '>>> OVERSOLD - watch for reversal'
        elif m15_rsi > 75:
            signal = '>>> OVERBOUGHT - watch for rejection'
        elif m15_rsi < 35 and momentum in ['UP', 'STRONG UP']:
            signal = '>>> REVERSAL? RSI low + momentum turning'
        elif m15_rsi > 65 and momentum in ['DOWN', 'STRONG DOWN']:
            signal = '>>> REVERSAL? RSI high + momentum turning'
        elif range_pos < 20 and momentum == 'STRONG UP':
            signal = '>>> BOUNCE from lows'
        elif range_pos > 80 and momentum == 'STRONG DOWN':
            signal = '>>> REJECTION from highs'

        print(f'\n{sym}: ${price:.2f}')
        print(f'  RSI: M15={m15_rsi:.1f} H1={h1_rsi:.1f} | Trend: {h1_trend} | Mom: {momentum}')
        print(f'  Range: {range_pos:.0f}% (${h1_low:.2f} - ${h1_high:.2f})')
        if signal:
            print(f'  {signal}')

    except Exception as e:
        print(f'{sym}: Error - {e}')

print('\n' + '='*70)
