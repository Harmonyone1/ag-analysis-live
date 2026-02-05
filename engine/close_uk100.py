import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time

from config import get_api
api = get_api()

positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    for _, pos in positions.iterrows():
        inst_id = pos['tradableInstrumentId']
        # UK100 instrument ID is 3889
        if inst_id == 3889:
            pos_id = int(pos['id'])
            qty = pos['qty']
            side = pos['side']
            pnl = pos['unrealizedPl']
            print(f'Closing UK100 position {pos_id}: {side} {qty} | P/L: ${pnl:.2f}')

            result = api.close_position(pos_id)
            print(f'Result: {result}')
            time.sleep(0.5)

time.sleep(1)
state = api.get_account_state()
balance = float(state['balance'])
print(f'Balance: ${balance:.2f}')

positions = api.get_all_positions()
print()
print('Remaining positions:')
if positions is not None and len(positions) > 0:
    for _, pos in positions.iterrows():
        sym = pos.get('tradableInstrumentId', 'Unknown')
        side = pos['side']
        qty = pos['qty']
        entry = pos['avgPrice']
        pnl = pos['unrealizedPl']
        print(f'  {sym}: {side} {qty} @ {entry:.2f} | P/L: ${pnl:.2f}')
else:
    print('  No positions')
