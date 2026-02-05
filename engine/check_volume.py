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

for sym in ['BTCUSD', 'ETHUSD']:
    inst_id = symbol_to_id[sym]

    time.sleep(0.5)
    h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='1D')
    m15 = api.get_price_history(inst_id, resolution='15m', lookback_period='6H')

    print(f'{sym}:')
    print('-'*50)

    # H1 Volume
    h1_vol = h1['v'].values
    avg_h1_vol = np.mean(h1_vol[:-1])  # Exclude current candle
    curr_h1_vol = h1_vol[-1]
    h1_ratio = curr_h1_vol / avg_h1_vol if avg_h1_vol > 0 else 0

    print(f'H1 Volume:')
    print(f'  Current: {curr_h1_vol:,.0f}')
    print(f'  Average: {avg_h1_vol:,.0f}')
    print(f'  Ratio: {h1_ratio:.2f}x avg')

    # M15 Volume
    m15_vol = m15['v'].values
    avg_m15_vol = np.mean(m15_vol[:-1])
    curr_m15_vol = m15_vol[-1]
    m15_ratio = curr_m15_vol / avg_m15_vol if avg_m15_vol > 0 else 0

    print(f'M15 Volume:')
    print(f'  Current: {curr_m15_vol:,.0f}')
    print(f'  Average: {avg_m15_vol:,.0f}')
    print(f'  Ratio: {m15_ratio:.2f}x avg')

    # Volume trend (last 3 candles)
    last3_vol = m15_vol[-3:]
    if last3_vol[-1] > last3_vol[-2] > last3_vol[-3]:
        vol_trend = 'INCREASING'
    elif last3_vol[-1] < last3_vol[-2] < last3_vol[-3]:
        vol_trend = 'DECREASING'
    else:
        vol_trend = 'MIXED'

    print(f'M15 Volume Trend: {vol_trend}')
    print()
