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

print('Executing ETH LONG...')

order = api.create_order(
    instrument_id=eth_id,
    quantity=0.8,
    side='buy',
    type_='market',
    stop_loss=2937,
    stop_loss_type='absolute',
    take_profit=3000,
    take_profit_type='absolute'
)

print(f'Order ID: {order}')

time.sleep(1.5)

positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    print('\nPOSITION OPENED:')
    for _, pos in positions.iterrows():
        inst_id = pos['tradableInstrumentId']
        sym = 'ETH' if inst_id == 3379 else 'BTC' if inst_id == 3378 else str(inst_id)
        print(f'  {sym} {pos["side"].upper()} {pos["qty"]} @ ${pos["avgPrice"]:.2f}')
        print(f'  P/L: ${pos["unrealizedPl"]:.2f}')
else:
    print('No position - check order')

state = api.get_account_state()
print(f'\nBalance: ${float(state.get("balance", 0)):.2f}')
