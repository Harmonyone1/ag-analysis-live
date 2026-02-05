import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time
from datetime import datetime

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

def get_momentum(closes):
    if len(closes) < 5:
        return 'FLAT', 0
    change = (closes[-1] - closes[-5]) / closes[-5] * 100
    if change > 0.5: return 'STRONG UP', change
    elif change > 0.1: return 'UP', change
    elif change < -0.5: return 'STRONG DOWN', change
    elif change < -0.1: return 'DOWN', change
    return 'FLAT', change

# Watchlist with direction bias
WATCHLIST = [
    {'sym': 'XAUUSD', 'bias': 'LONG', 'reason': 'Pullback entry on gold'},
    {'sym': 'XAGUSD', 'bias': 'SHORT', 'reason': 'Re-entry on exhaustion'},
    {'sym': 'BTCUSD', 'bias': 'LONG', 'reason': 'Pullback to support'},
    {'sym': 'ETHUSD', 'bias': 'LONG', 'reason': 'Pullback to support'},
]

print('='*65)
print('WATCHING FOR ENTRY SIGNALS')
print(f'Started: {datetime.now().strftime("%H:%M:%S")}')
print('='*65)
for w in WATCHLIST:
    print(f'  {w["sym"]}: {w["bias"]} - {w["reason"]}')
print('='*65)
print()

scan = 0
while True:
    scan += 1
    now = datetime.now().strftime("%H:%M:%S")

    print(f'[{now}] Scan #{scan}')
    print('-'*65)

    for w in WATCHLIST:
        sym = w['sym']
        bias = w['bias']

        if sym not in symbol_to_id:
            print(f'  {sym}: Not available')
            continue

        try:
            time.sleep(0.5)
            inst_id = symbol_to_id[sym]

            # Get multiple timeframes
            h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='7D')
            time.sleep(0.3)
            m15 = api.get_price_history(inst_id, resolution='15m', lookback_period='3D')

            if h1 is None or m15 is None or len(h1) < 20 or len(m15) < 10:
                print(f'  {sym}: Insufficient data')
                continue

            price = m15['c'].values[-1]

            # RSI
            h1_rsi = get_rsi(h1['c'].values)
            m15_rsi = get_rsi(m15['c'].values)

            # Momentum
            h1_mom, h1_chg = get_momentum(h1['c'].values)
            m15_mom, m15_chg = get_momentum(m15['c'].values)

            # Candle analysis (last 3 M15 candles)
            m15_o = m15['o'].values
            m15_h = m15['h'].values
            m15_l = m15['l'].values
            m15_c = m15['c'].values

            # Check for reversal candle
            last_candle_bull = m15_c[-1] > m15_o[-1]
            last_candle_bear = m15_c[-1] < m15_o[-1]
            prev_candle_bull = m15_c[-2] > m15_o[-2]
            prev_candle_bear = m15_c[-2] < m15_o[-2]

            # Engulfing pattern
            bull_engulf = last_candle_bull and prev_candle_bear and m15_c[-1] > m15_o[-2]
            bear_engulf = last_candle_bear and prev_candle_bull and m15_c[-1] < m15_o[-2]

            # Higher low / Lower high
            higher_low = m15_l[-1] > m15_l[-2] and m15_l[-2] > m15_l[-3]
            lower_high = m15_h[-1] < m15_h[-2] and m15_h[-2] < m15_h[-3]

            # Entry signals
            signals = []
            entry_ready = False

            if bias == 'LONG':
                if 'UP' in m15_mom:
                    signals.append('M15 momentum UP')
                if bull_engulf:
                    signals.append('BULLISH ENGULFING')
                if higher_low:
                    signals.append('Higher lows forming')
                if m15_rsi > 30 and h1_rsi < 35:
                    signals.append('RSI divergence (M15 rising)')
                if last_candle_bull and m15_rsi < 40:
                    signals.append('Green candle at low RSI')

                # Entry ready if multiple confirmations
                if len(signals) >= 2:
                    entry_ready = True

            elif bias == 'SHORT':
                if 'DOWN' in m15_mom:
                    signals.append('M15 momentum DOWN')
                if bear_engulf:
                    signals.append('BEARISH ENGULFING')
                if lower_high:
                    signals.append('Lower highs forming')
                if m15_rsi < 70 and h1_rsi > 65:
                    signals.append('RSI divergence (M15 falling)')
                if last_candle_bear and m15_rsi > 60:
                    signals.append('Red candle at high RSI')

                if len(signals) >= 2:
                    entry_ready = True

            # Display
            if price > 100:
                price_str = f'${price:.2f}'
            elif price > 1:
                price_str = f'{price:.4f}'
            else:
                price_str = f'{price:.5f}'

            status = f'  {sym} {price_str} | H1 RSI:{h1_rsi:.0f} M15 RSI:{m15_rsi:.0f} | {m15_mom}'

            if entry_ready:
                print(f'{status}')
                print(f'  >>> ENTRY SIGNAL [{bias}]: {", ".join(signals)}')
                print(f'  >>> READY TO ENTER!')
            elif signals:
                print(f'{status}')
                print(f'      Developing: {", ".join(signals)}')
            else:
                print(f'{status} | Waiting...')

        except Exception as e:
            print(f'  {sym}: Error - {str(e)[:40]}')

    print()
    time.sleep(45)
