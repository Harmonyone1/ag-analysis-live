import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

inst_id = symbol_to_id['USOIL']
price = api.get_latest_asking_price(inst_id)

time.sleep(0.3)
h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='3D')

highs = h1['h'].values
lows = h1['l'].values

recent_high = max(highs[-12:])
recent_low = min(lows[-24:])

print(f'USOIL Current: ${price:.3f}')
print(f'Recent High: ${recent_high:.3f}')
print(f'Recent Low: ${recent_low:.3f}')
print()

# SL above recent high, TP at recent low
sl = recent_high + 0.20
tp = recent_low + 0.15

print(f'SHORT Setup:')
print(f'  Entry: ~${price:.3f}')
print(f'  SL: ${sl:.3f} (risk: ${sl - price:.3f})')
print(f'  TP: ${tp:.3f} (reward: ${price - tp:.3f})')
rr = (price - tp) / (sl - price) if sl > price else 0
print(f'  R:R = {rr:.1f}')
print()

state = api.get_account_state()
balance = float(state['balance'])

lot_size = 0.05

print(f'Balance: ${balance:.2f}')
print(f'Lot size: {lot_size}')
print()

print('Executing USOIL SHORT...')
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
print(f'Order result: {order}')

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
