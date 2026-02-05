import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

eth_id = symbol_to_id['ETHUSD']
time.sleep(0.5)
m15 = api.get_price_history(eth_id, resolution='15m', lookback_period='6H')

print('ETH M15 Last 6 Candles:')
print('-'*60)
for i in range(-6, 0):
    o = m15['o'].values[i]
    h = m15['h'].values[i]
    l = m15['l'].values[i]
    c = m15['c'].values[i]
    candle = 'GREEN' if c > o else 'RED'
    body = abs(c - o)
    print(f'  O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f} | {candle} | Body:{body:.2f}')

# Check for higher low
lows = m15['l'].values[-6:]
if lows[-1] > min(lows[:-1]):
    print('\n>>> Recent low is HIGHER than prior lows')

# Current
price = api.get_latest_bid_price(eth_id)
ask = api.get_latest_asking_price(eth_id)
print(f'\nCurrent: Bid ${price:.2f} | Ask ${ask:.2f}')

# Entry setup
h1_low = 2942.38
stop = h1_low - 5
target = 3000
risk = ask - stop
reward = target - ask
print(f'\nPotential LONG Entry:')
print(f'  Entry: ${ask:.2f}')
print(f'  Stop: ${stop:.2f} (below session low)')
print(f'  Target: ${target:.2f}')
print(f'  R:R = 1:{reward/risk:.1f}')

# Risk calc
balance = 305.68
risk_pct = 0.06
risk_amt = balance * risk_pct
lot_size = risk_amt / risk
lot_size = max(0.1, round(lot_size / 0.1) * 0.1)
print(f'  Size: {lot_size} lots (6% risk = ${risk_amt:.2f})')
