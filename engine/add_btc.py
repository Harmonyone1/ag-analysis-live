import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))
btc_id = symbol_to_id['BTCUSD']

# Current price
price = api.get_latest_bid_price(btc_id)
print(f'BTC: ${price:.2f}')

# Smaller size since we have ETH exposure
# Stop below recent low, target round number
stop = 87550  # Below session low
target = 89000

print(f'\nExecuting BTC LONG...')

order = api.create_order(
    instrument_id=btc_id,
    quantity=0.03,
    side='buy',
    type_='market',
    stop_loss=stop,
    stop_loss_type='absolute',
    take_profit=target,
    take_profit_type='absolute'
)

print(f'Order ID: {order}')

time.sleep(1.5)

positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    print(f'\nOPEN POSITIONS ({len(positions)}):')
    for _, pos in positions.iterrows():
        inst_id = pos['tradableInstrumentId']
        if inst_id == 3378:
            sym = 'BTC'
        elif inst_id == 3379:
            sym = 'ETH'
        else:
            sym = str(inst_id)
        print(f'  {sym} {pos["side"].upper()} {pos["qty"]} @ ${pos["avgPrice"]:.2f} | P/L: ${pos["unrealizedPl"]:.2f}')

state = api.get_account_state()
print(f'\nBalance: ${float(state.get("balance", 0)):.2f}')
