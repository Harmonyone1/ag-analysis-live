import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

sym = 'JNJ'
inst_id = symbol_to_id[sym]

m15 = api.get_price_history(inst_id, resolution='15m', lookback_period='1D')
price = m15['c'].values[-1]

qty = 0.02  # lot size
sl = 205.29
tp = 208.17

print(f'JNJ LONG: {qty} lots @ ${price:.2f}')
print(f'SL: ${sl} | TP: ${tp}')
print()

result = api.create_order(
    instrument_id=inst_id,
    quantity=qty,
    side='buy',
    type_='market',
    stop_loss=sl,
    stop_loss_type='absolute',
    take_profit=tp,
    take_profit_type='absolute'
)
print(f'Result: {result}')

time.sleep(1)
positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    print()
    print('POSITIONS:')
    for _, pos in positions.iterrows():
        pid = pos['tradableInstrumentId']
        sym_map = dict(zip(instruments['tradableInstrumentId'], instruments['name']))
        psym = sym_map.get(pid, str(pid))
        print(f'  {psym}: {pos["side"]} {pos["qty"]} @ ${pos["avgPrice"]:.2f} | PnL: ${pos["unrealizedPl"]:.2f}')

state = api.get_account_state()
print()
print(f'Balance: ${state["balance"]}')
