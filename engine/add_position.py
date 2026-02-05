#!/usr/bin/env python
"""Add to NZDCAD LONG position for growth"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Get current balance
state = api.get_account_state()
balance = state['balance']

print('=' * 60)
print('ADD TO NZDCAD POSITION | %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 60)
print('Balance: $%.2f' % balance)
print()

symbol = 'NZDCAD'
inst_id = name_to_id.get(symbol)
price = api.get_latest_asking_price(inst_id)

# For growth - larger position size
# Adding 0.08 lots (total will be 0.10)
additional_lots = 0.08
sl = 0.7907
tp = 0.7963

print('Adding %.2f lots to existing 0.02 position' % additional_lots)
print('Entry: %.5f | SL: %.5f | TP: %.5f' % (price, sl, tp))
print()

try:
    order = api.create_order(
        instrument_id=inst_id,
        quantity=additional_lots,
        side='buy',
        type_='market',
        stop_loss=sl,
        stop_loss_type='absolute',
        take_profit=tp,
        take_profit_type='absolute'
    )
    print('ADDED! Order ID: %s' % order)
except Exception as e:
    print('Error: %s' % str(e))

print()

# Verify
import time
time.sleep(1)
positions = api.get_all_positions()
print('Current Positions:')
if positions is not None and len(positions) > 0:
    total_lots = 0
    for _, pos in positions.iterrows():
        qty = pos.get('qty', 0)
        total_lots += qty
        print('  %s %s %.2f lots | P/L: $%.2f' % (
            pos.get('symbol', 'N/A'),
            pos.get('side', 'N/A'),
            qty,
            pos.get('unrealizedPnL', 0)))
    print()
    print('Total NZDCAD: %.2f lots' % total_lots)
