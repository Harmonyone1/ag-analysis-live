import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

inst_id = symbol_to_id['UK100']
price = api.get_latest_asking_price(inst_id)

time.sleep(0.3)
h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='3D')

h1_high = max(h1['h'].values[-48:])
h1_low = min(h1['l'].values[-48:])

sl = h1_low - 20
tp = h1_high - 10

print(f'UK100 LONG')
print(f'Price: {price:.2f}')
print(f'SL: {sl:.2f}')
print(f'TP: {tp:.2f}')
print(f'R:R: {(tp - price) / (price - sl):.1f}')
print()

state = api.get_account_state()
balance = float(state['balance'])

lot_size = 0.02  # Small size due to weak R:R

print(f'Balance: ${balance:.2f}')
print(f'Lot size: {lot_size}')
print()

print('Executing LONG...')
order = api.create_order(
    instrument_id=inst_id,
    quantity=lot_size,
    side='buy',
    type_='market',
    stop_loss=sl,
    stop_loss_type='absolute',
    take_profit=tp,
    take_profit_type='absolute'
)
print(f'Order ID: {order}')

time.sleep(1)
positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    print()
    print('OPEN POSITIONS:')
    for _, pos in positions.iterrows():
        sym = pos.get('tradableInstrumentId', 'Unknown')
        side = pos['side']
        qty = pos['qty']
        entry = pos['avgPrice']
        pnl = pos['unrealizedPl']
        print(f'  {sym}: {side} {qty} @ {entry:.2f} | P/L: ${pnl:.2f}')
