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

print('Executing BTC LONG...')

order = api.create_order(
    instrument_id=btc_id,
    quantity=0.04,
    side='buy',
    type_='market',
    stop_loss=87805,
    stop_loss_type='absolute',
    take_profit=89000,
    take_profit_type='absolute'
)

print(f'Order ID: {order}')

time.sleep(1.5)

positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    print('\nOPEN POSITIONS:')
    for _, pos in positions.iterrows():
        inst = pos.get('tradableInstrumentId', '')
        sym = 'BTC' if inst == 3378 else 'ETH' if inst == 3379 else str(inst)
        print(f'  {sym} {pos["side"].upper()} {pos["qty"]} @ ${pos["avgPrice"]:.2f} | P/L: ${pos["unrealizedPl"]:.2f}')

state = api.get_account_state()
print(f'\nBalance: ${float(state.get("balance", 0)):.2f}')
