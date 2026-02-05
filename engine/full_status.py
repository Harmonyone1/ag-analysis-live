import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

btc_price = api.get_latest_bid_price(symbol_to_id['BTCUSD'])
eth_price = api.get_latest_bid_price(symbol_to_id['ETHUSD'])

print(f'BTC: ${btc_price:.2f} | ETH: ${eth_price:.2f}')
print()

time.sleep(0.5)
state = api.get_account_state()
balance = float(state.get('balance', 0))
equity = float(state.get('equity', 0))
print(f'Balance: ${balance:.2f}')
print(f'Equity: ${equity:.2f}')
print()

positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    print(f'OPEN POSITIONS ({len(positions)}):')
    for _, pos in positions.iterrows():
        inst_id = pos['tradableInstrumentId']
        if inst_id == 3378:
            sym = 'BTCUSD'
        elif inst_id == 3379:
            sym = 'ETHUSD'
        else:
            sym = str(inst_id)
        side = pos['side'].upper()
        qty = pos['qty']
        entry = pos['avgPrice']
        pnl = pos['unrealizedPl']
        print(f'  {sym}: {side} {qty} @ ${entry:.2f} | P/L: ${pnl:.2f}')
else:
    print('No open positions')
