import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

btc_id = symbol_to_id.get('BTCUSD')
eth_id = symbol_to_id.get('ETHUSD')

# Get H1 data
btc_h1 = api.get_price_history(btc_id, resolution='1H', lookback_period='1D')
eth_h1 = api.get_price_history(eth_id, resolution='1H', lookback_period='1D')

# Get M15 data for more granular view
btc_m15 = api.get_price_history(btc_id, resolution='15m', lookback_period='6H')
eth_m15 = api.get_price_history(eth_id, resolution='15m', lookback_period='6H')

print('=' * 80)
print('BTCUSD - CANDLE VOLUME DATA')
print('=' * 80)
print(f'Available columns: {list(btc_h1.columns)}')
print()

if 'v' in btc_h1.columns:
    print('H1 Candles (last 10):')
    print('-' * 80)
    for i in range(-10, 0):
        try:
            ts = btc_h1.index[i]
            o = btc_h1['o'].iloc[i]
            h = btc_h1['h'].iloc[i]
            l = btc_h1['l'].iloc[i]
            c = btc_h1['c'].iloc[i]
            v = btc_h1['v'].iloc[i]
            candle = 'GREEN' if c > o else 'RED'
            print(f'{str(ts)[:16]} | O:{o:>10.2f} H:{h:>10.2f} L:{l:>10.2f} C:{c:>10.2f} | Vol:{v:>12.2f} | {candle}')
        except:
            pass

    print()
    print('M15 Candles (last 12):')
    print('-' * 80)
    for i in range(-12, 0):
        try:
            ts = btc_m15.index[i]
            o = btc_m15['o'].iloc[i]
            h = btc_m15['h'].iloc[i]
            l = btc_m15['l'].iloc[i]
            c = btc_m15['c'].iloc[i]
            v = btc_m15['v'].iloc[i]
            candle = 'GREEN' if c > o else 'RED'
            print(f'{str(ts)[:16]} | O:{o:>10.2f} H:{h:>10.2f} L:{l:>10.2f} C:{c:>10.2f} | Vol:{v:>12.2f} | {candle}')
        except:
            pass
else:
    print('Volume column (v) not found in data')
    print(btc_h1.tail(5))

print()
print('=' * 80)
print('ETHUSD - CANDLE VOLUME DATA')
print('=' * 80)

if 'v' in eth_h1.columns:
    print('H1 Candles (last 10):')
    print('-' * 80)
    for i in range(-10, 0):
        try:
            ts = eth_h1.index[i]
            o = eth_h1['o'].iloc[i]
            h = eth_h1['h'].iloc[i]
            l = eth_h1['l'].iloc[i]
            c = eth_h1['c'].iloc[i]
            v = eth_h1['v'].iloc[i]
            candle = 'GREEN' if c > o else 'RED'
            print(f'{str(ts)[:16]} | O:{o:>10.2f} H:{h:>10.2f} L:{l:>10.2f} C:{c:>10.2f} | Vol:{v:>12.2f} | {candle}')
        except:
            pass

    print()
    print('M15 Candles (last 12):')
    print('-' * 80)
    for i in range(-12, 0):
        try:
            ts = eth_m15.index[i]
            o = eth_m15['o'].iloc[i]
            h = eth_m15['h'].iloc[i]
            l = eth_m15['l'].iloc[i]
            c = eth_m15['c'].iloc[i]
            v = eth_m15['v'].iloc[i]
            candle = 'GREEN' if c > o else 'RED'
            print(f'{str(ts)[:16]} | O:{o:>10.2f} H:{h:>10.2f} L:{l:>10.2f} C:{c:>10.2f} | Vol:{v:>12.2f} | {candle}')
        except:
            pass
else:
    print('Volume column (v) not found in data')
    print(eth_h1.tail(5))

# Check for DOM / Order Book data
print()
print('=' * 80)
print('CHECKING FOR DEPTH OF MARKET / ORDER BOOK')
print('=' * 80)

# Check API methods
api_methods = [m for m in dir(api) if not m.startswith('_')]
dom_methods = [m for m in api_methods if any(x in m.lower() for x in ['depth', 'book', 'order', 'dom', 'level', 'quote'])]
print(f'Potential DOM-related API methods: {dom_methods}')

# Try to get quotes/depth if available
try:
    quote = api.get_instrument_details(btc_id)
    print(f'\nBTCUSD Instrument Details:')
    print(quote)
except Exception as e:
    print(f'get_instrument_details error: {e}')

# List all API methods for reference
print(f'\nAll available API methods:')
for m in sorted(api_methods):
    print(f'  - {m}')
