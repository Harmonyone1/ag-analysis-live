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
eth_id = symbol_to_id.get('ETHUSD')

# Key levels from analysis
BEARISH_OB_TOP = 2996.30
BEARISH_OB_BOTTOM = 2852.56
NEXT_BEARISH_OB = 3124.00
TARGET_SIZE = 1.0  # 1 lot as requested

print('=' * 60)
print('ETHUSD ENTRY WATCH - 1 LOT')
print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('=' * 60)
print()
print('KEY LEVELS:')
print(f'  Bearish OB Zone: ${BEARISH_OB_BOTTOM:.2f} - ${BEARISH_OB_TOP:.2f}')
print(f'  Next Bearish OB: ${NEXT_BEARISH_OB:.2f}')
print()
print('ENTRY TRIGGERS:')
print('  SHORT: Rejection candle at OB top ($2996) OR break below $2935')
print('  LONG:  Break and hold above $2996')
print()
print('-' * 60)

def get_rsi(closes, period=14):
    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.0001
    return 100 - (100 / (1 + avg_gain/avg_loss))

def check_rejection_candle(opens, highs, lows, closes, level, direction='bearish'):
    """Check if last candle is a rejection at a level"""
    last_o = opens[-1]
    last_h = highs[-1]
    last_l = lows[-1]
    last_c = closes[-1]

    body = abs(last_c - last_o)
    upper_wick = last_h - max(last_o, last_c)
    lower_wick = min(last_o, last_c) - last_l

    if direction == 'bearish':
        # Bearish rejection: long upper wick, price touched level from below
        if last_h >= level * 0.998 and upper_wick > body * 1.5 and last_c < last_o:
            return True
    else:
        # Bullish rejection: long lower wick, price touched level from above
        if last_l <= level * 1.002 and lower_wick > body * 1.5 and last_c > last_o:
            return True
    return False

scan_count = 0
while True:
    scan_count += 1
    now = datetime.now().strftime("%H:%M:%S")

    try:
        m15 = api.get_price_history(eth_id, resolution='15m', lookback_period='1D')
        m5 = api.get_price_history(eth_id, resolution='5m', lookback_period='6H')

        m15_o = m15['o'].values
        m15_h = m15['h'].values
        m15_l = m15['l'].values
        m15_c = m15['c'].values

        m5_o = m5['o'].values
        m5_h = m5['h'].values
        m5_l = m5['l'].values
        m5_c = m5['c'].values

        current = m5_c[-1]
        m15_rsi = get_rsi(m15_c)
        m5_rsi = get_rsi(m5_c)

        # Check position in OB
        in_ob = current >= BEARISH_OB_BOTTOM and current <= BEARISH_OB_TOP
        dist_to_top = BEARISH_OB_TOP - current
        dist_to_bottom = current - BEARISH_OB_BOTTOM

        # Check for rejection candle at OB top
        rejection_at_top = check_rejection_candle(m5_o, m5_h, m5_l, m5_c, BEARISH_OB_TOP, 'bearish')

        # Check for break above OB
        break_above = current > BEARISH_OB_TOP and m5_c[-2] < BEARISH_OB_TOP

        # Check for break below key support
        break_below = current < 2935 and m5_c[-2] > 2935

        # Entry signals
        entry_signal = None

        if rejection_at_top and m5_rsi > 60:
            entry_signal = 'SHORT - Rejection at OB top with elevated RSI'
        elif break_above and m5_c[-1] > m5_o[-1]:  # Bullish candle break
            entry_signal = 'LONG - Break above OB with bullish candle'
        elif break_below:
            entry_signal = 'SHORT - Break below $2935 support'
        elif dist_to_top < 5 and m5_rsi > 65:
            entry_signal = 'SHORT ALERT - Near OB top with RSI > 65'

        # Status output
        status = 'IN_OB' if in_ob else ('ABOVE_OB' if current > BEARISH_OB_TOP else 'BELOW_OB')

        print(f'[{now}] #{scan_count} | ${current:.2f} | {status} | RSI M5:{m5_rsi:.1f} M15:{m15_rsi:.1f} | To top: ${dist_to_top:.2f}')

        if entry_signal:
            print()
            print('!' * 60)
            print(f'>>> ENTRY SIGNAL: {entry_signal}')
            print('!' * 60)
            print(f'Price: ${current:.2f}')
            print(f'Size: {TARGET_SIZE} lot')
            if 'SHORT' in entry_signal:
                print(f'Stop Loss: ${BEARISH_OB_TOP + 20:.2f}')
                print(f'Take Profit: ${BEARISH_OB_BOTTOM:.2f}')
                print(f'R:R = 1:{(current - BEARISH_OB_BOTTOM) / 20:.1f}')
            else:
                print(f'Stop Loss: ${current - 30:.2f}')
                print(f'Take Profit: ${NEXT_BEARISH_OB:.2f}')
            print()

        # Check for M15 candle close (more significant)
        if scan_count % 3 == 0:  # Every 3rd scan (approx 1.5 min)
            # Check M15 structure
            if m15_c[-1] < m15_o[-1] and m15_h[-1] > BEARISH_OB_TOP * 0.998:
                print(f'  ** M15 bearish candle touching OB top - watch for confirmation')
            elif m15_c[-1] > m15_o[-1] and m15_c[-1] > BEARISH_OB_TOP:
                print(f'  ** M15 bullish close above OB - potential breakout')

    except Exception as e:
        print(f'[{now}] Error: {e}')

    time.sleep(30)  # Check every 30 seconds for entry
