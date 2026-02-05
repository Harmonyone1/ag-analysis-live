"""Enter BTCUSD and add to ETHUSD positions"""
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Get account state
state = api.get_account_state()
balance = float(state.get('accountBalance', 0))
equity = float(state.get('accountEquity', 0))
print(f'Account Balance: ${balance:.2f}')
print(f'Account Equity: ${equity:.2f}')
print()

# Current positions
positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    print('Current Positions:')
    for _, p in positions.iterrows():
        pnl = float(p.get('unrealizedPL', 0))
        print(f'  {p["symbol"]}: {p["side"]} {p["qty"]} @ ${float(p["avgPrice"]):.2f} | P/L: ${pnl:.2f}')
    print()

time.sleep(1)

# BTCUSD Entry
print('='*50)
print('BTCUSD LONG Entry')
print('='*50)
btc_id = symbol_to_id['BTCUSD']
btc_bars = api.get_price_history(btc_id, resolution='15m', lookback_period='1D')
btc_price = btc_bars['c'].values[-1]
btc_low = btc_bars['l'].values[-10:].min()

# Set SL below recent low, TP for 2:1 R:R
btc_sl = btc_low - 200  # Below recent low
btc_risk = btc_price - btc_sl
btc_tp = btc_price + (btc_risk * 2)  # 2:1 R:R

print(f'Price: ${btc_price:.2f}')
print(f'Stop Loss: ${btc_sl:.2f}')
print(f'Take Profit: ${btc_tp:.2f}')
print(f'Risk: ${btc_risk:.2f} | Reward: ${btc_risk*2:.2f}')

time.sleep(1)

btc_order = api.create_order(
    instrument_id=btc_id,
    quantity=0.01,
    side='buy',
    type_='market',
    stop_loss=btc_sl,
    take_profit=btc_tp,
    stop_loss_type='absolute',
    take_profit_type='absolute'
)
print(f'BTC Order Result: {btc_order}')
print()

time.sleep(2)

# ETHUSD Add Position
print('='*50)
print('ETHUSD LONG Add Position')
print('='*50)
eth_id = symbol_to_id['ETHUSD']
eth_bars = api.get_price_history(eth_id, resolution='15m', lookback_period='1D')
eth_price = eth_bars['c'].values[-1]
eth_low = eth_bars['l'].values[-10:].min()

# Set SL below recent low, TP for 2:1 R:R
eth_sl = eth_low - 30  # Below recent low
eth_risk = eth_price - eth_sl
eth_tp = eth_price + (eth_risk * 2)  # 2:1 R:R

print(f'Price: ${eth_price:.2f}')
print(f'Stop Loss: ${eth_sl:.2f}')
print(f'Take Profit: ${eth_tp:.2f}')
print(f'Risk: ${eth_risk:.2f} | Reward: ${eth_risk*2:.2f}')

time.sleep(1)

eth_order = api.create_order(
    instrument_id=eth_id,
    quantity=0.5,
    side='buy',
    type_='market',
    stop_loss=eth_sl,
    take_profit=eth_tp,
    stop_loss_type='absolute',
    take_profit_type='absolute'
)
print(f'ETH Order Result: {eth_order}')
print()

time.sleep(2)

# Updated positions
print('='*50)
print('UPDATED POSITIONS')
print('='*50)
positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    for _, p in positions.iterrows():
        pnl = float(p.get('unrealizedPL', 0))
        print(f'{p["symbol"]}: {p["side"]} {p["qty"]} @ ${float(p["avgPrice"]):.2f} | P/L: ${pnl:.2f}')

state = api.get_account_state()
balance = float(state.get('accountBalance', 0))
equity = float(state.get('accountEquity', 0))
print()
print(f'Final Balance: ${balance:.2f}')
print(f'Final Equity: ${equity:.2f}')
