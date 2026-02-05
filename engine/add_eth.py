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

price = api.get_latest_bid_price(eth_id)
print(f'ETH: ${price:.2f}')

print('\nAdding 0.4 lots to ETH LONG...')

order = api.create_order(
    instrument_id=eth_id,
    quantity=0.4,
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
    print(f'\nOPEN POSITIONS ({len(positions)}):')
    total_pnl = 0
    for _, pos in positions.iterrows():
        inst_id = pos['tradableInstrumentId']
        if inst_id == 3378:
            sym = 'BTC'
        elif inst_id == 3379:
            sym = 'ETH'
        else:
            sym = str(inst_id)
        pnl = pos['unrealizedPl']
        total_pnl += pnl
        print(f'  {sym} {pos["side"].upper()} {pos["qty"]} @ ${pos["avgPrice"]:.2f} | P/L: ${pnl:.2f}')
    print(f'\nTotal P/L: ${total_pnl:.2f}')

state = api.get_account_state()
print(f'Balance: ${float(state.get("balance", 0)):.2f}')
