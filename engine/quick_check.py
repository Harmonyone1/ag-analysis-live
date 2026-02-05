import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

eth_price = api.get_latest_bid_price(symbol_to_id['ETHUSD'])
btc_price = api.get_latest_bid_price(symbol_to_id['BTCUSD'])

positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    entry = positions['avgPrice'].values[0]
    pnl = positions['unrealizedPl'].values[0]
    qty = positions['qty'].values[0]
    print(f'ETH LONG {qty} @ ${entry:.2f}')
    print(f'Current: ${eth_price:.2f} | P/L: ${pnl:.2f}')
    print(f'To TP ($3,050): ${3050 - eth_price:.2f}')
    print(f'To SL ($2,960): ${eth_price - 2960:.2f}')
else:
    print('No position')

print(f'\nBTC: ${btc_price:.2f}')
