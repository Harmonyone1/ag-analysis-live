import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Get current prices
btc_id = symbol_to_id['BTCUSD']
eth_id = symbol_to_id['ETHUSD']

btc_data = api.get_price_history(btc_id, resolution='15m', lookback_period='1D')
time.sleep(0.3)
eth_data = api.get_price_history(eth_id, resolution='15m', lookback_period='1D')

btc_price = btc_data['c'].values[-1]
eth_price = eth_data['c'].values[-1]

# BTC setup - using recent swing low for SL
btc_low = btc_data['l'].values[-10:].min()
btc_sl = btc_low - 100  # 100 below recent low
btc_tp = btc_price + (btc_price - btc_sl) * 2  # 2:1 R:R

# ETH setup
eth_low = eth_data['l'].values[-10:].min()
eth_sl = eth_low - 10  # 10 below recent low
eth_tp = eth_price + (eth_price - eth_sl) * 2  # 2:1 R:R

print('='*55)
print('ENTERING BTC & ETH LONG')
print('='*55)

# Enter BTC
print(f'\nBTCUSD LONG:')
print(f'  Price: ${btc_price:.2f}')
print(f'  SL: ${btc_sl:.2f}')
print(f'  TP: ${btc_tp:.2f}')

btc_result = api.create_order(
    instrument_id=btc_id,
    quantity=0.01,
    side='buy',
    type_='market',
    stop_loss=btc_sl,
    stop_loss_type='absolute',
    take_profit=btc_tp,
    take_profit_type='absolute'
)
print(f'  Result: {btc_result}')

time.sleep(0.5)

# Enter ETH
print(f'\nETHUSD LONG:')
print(f'  Price: ${eth_price:.2f}')
print(f'  SL: ${eth_sl:.2f}')
print(f'  TP: ${eth_tp:.2f}')

eth_result = api.create_order(
    instrument_id=eth_id,
    quantity=0.5,
    side='buy',
    type_='market',
    stop_loss=eth_sl,
    stop_loss_type='absolute',
    take_profit=eth_tp,
    take_profit_type='absolute'
)
print(f'  Result: {eth_result}')

time.sleep(1)

# Show all positions
print('\n' + '='*55)
print('ALL POSITIONS:')
print('='*55)
positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    sym_map = dict(zip(instruments['tradableInstrumentId'], instruments['name']))
    for _, pos in positions.iterrows():
        psym = sym_map.get(pos['tradableInstrumentId'], str(pos['tradableInstrumentId']))
        print(f'  {psym}: {pos["side"]} {pos["qty"]} @ ${pos["avgPrice"]:.2f} | P/L: ${pos["unrealizedPl"]:.2f}')

state = api.get_account_state()
print(f'\nBalance: ${state["balance"]}')
