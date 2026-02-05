import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

inst_id = symbol_to_id['XAGUSD']
price = api.get_latest_asking_price(inst_id)

time.sleep(0.3)
h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='3D')

h1_high = max(h1['h'].values[-24:])
h1_low = min(h1['l'].values[-24:])

# Wider SL this time
sl = h1_high + 0.30
tp = h1_low + 0.20

print(f'XAGUSD Re-entry SHORT')
print(f'Price: ${price:.4f}')
print(f'SL: ${sl:.4f}')
print(f'TP: ${tp:.4f}')
print(f'R:R: {(price - tp) / (sl - price):.1f}')
print()

state = api.get_account_state()
balance = float(state['balance'])

lot_size = 0.03  # Smaller size for re-entry

print(f'Balance: ${balance:.2f}')
print(f'Lot size: {lot_size}')
print()

print('Executing SHORT...')
order = api.create_order(
    instrument_id=inst_id,
    quantity=lot_size,
    side='sell',
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
        print(f'  {sym}: {side} {qty} @ ${entry:.4f} | P/L: ${pnl:.2f}')
