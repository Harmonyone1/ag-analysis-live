import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Get current price and recent levels
inst_id = symbol_to_id['XAGUSD']
price = api.get_latest_asking_price(inst_id)

time.sleep(0.3)
h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='3D')

highs = h1['h'].values
lows = h1['l'].values

recent_high = max(highs[-10:])
recent_low = min(lows[-10:])

print(f'XAGUSD Current: {price:.4f}')
print(f'Recent High: {recent_high:.4f}')
print(f'Recent Low: {recent_low:.4f}')
print()

# SL above recent high, TP at recent low
sl = recent_high + 0.15  # Buffer above high
tp = recent_low + 0.10   # Near recent low

print(f'Proposed SHORT:')
print(f'  Entry: ~{price:.4f}')
print(f'  SL: {sl:.4f} (risk: {sl - price:.4f})')
print(f'  TP: {tp:.4f} (reward: {price - tp:.4f})')
print(f'  R:R = {(price - tp) / (sl - price):.1f}')
print()

# Calculate lot size
state = api.get_account_state()
balance = float(state['balance'])
risk_amount = balance * 0.08

lot_size = 0.05  # Conservative for silver

print(f'Balance: ${balance:.2f}')
print(f'Risk 8%: ${risk_amount:.2f}')
print(f'Lot size: {lot_size}')
print()

# Execute trade
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
print(f'Order result: {order}')

time.sleep(0.5)
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
        print(f'  {sym}: {side} {qty} @ {entry:.4f} | P/L: ${pnl:.2f}')
