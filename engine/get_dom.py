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

print('='*80)
print('DEPTH OF MARKET / ORDER BOOK')
print('='*80)

# Try get_market_depth
print('\nBTCUSD Market Depth:')
print('-'*80)
try:
    depth = api.get_market_depth(btc_id)
    print(depth)
except Exception as e:
    print(f'Error: {e}')

print()
print('\nETHUSD Market Depth:')
print('-'*80)
try:
    depth = api.get_market_depth(eth_id)
    print(depth)
except Exception as e:
    print(f'Error: {e}')

# Try get_quotes
print()
print('='*80)
print('QUOTES (BID/ASK)')
print('='*80)

print('\nBTCUSD Quotes:')
try:
    quotes = api.get_quotes(btc_id)
    print(quotes)
except Exception as e:
    print(f'Error: {e}')

print('\nETHUSD Quotes:')
try:
    quotes = api.get_quotes(eth_id)
    print(quotes)
except Exception as e:
    print(f'Error: {e}')

# Get latest bid/ask
print()
print('='*80)
print('LATEST BID/ASK PRICES')
print('='*80)

try:
    btc_bid = api.get_latest_bid_price(btc_id)
    btc_ask = api.get_latest_asking_price(btc_id)
    print(f'BTCUSD: Bid ${btc_bid:.2f} | Ask ${btc_ask:.2f} | Spread ${btc_ask - btc_bid:.2f}')
except Exception as e:
    print(f'BTC error: {e}')

try:
    eth_bid = api.get_latest_bid_price(eth_id)
    eth_ask = api.get_latest_asking_price(eth_id)
    print(f'ETHUSD: Bid ${eth_bid:.2f} | Ask ${eth_ask:.2f} | Spread ${eth_ask - eth_bid:.2f}')
except Exception as e:
    print(f'ETH error: {e}')
