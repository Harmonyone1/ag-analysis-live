#!/usr/bin/env python
"""Add to NZDCAD and NZDUSD positions"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

print('=' * 60)
print('ADDING TO POSITIONS | %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 60)
print()

# Add to NZDCAD
symbol = 'NZDCAD'
inst_id = name_to_id.get(symbol)
price = api.get_latest_asking_price(inst_id)
lots = 0.10
sl = 0.7907
tp = 0.7963

print('NZDCAD LONG - Adding 0.10 lots')
print('  Entry: %.5f | SL: %.5f | TP: %.5f' % (price, sl, tp))
try:
    order = api.create_order(
        instrument_id=inst_id,
        quantity=lots,
        side='buy',
        type_='market',
        stop_loss=sl,
        stop_loss_type='absolute',
        take_profit=tp,
        take_profit_type='absolute'
    )
    print('  SUCCESS! Order ID: %s' % order)
except Exception as e:
    print('  Error: %s' % str(e))

print()

# Add to NZDUSD
symbol = 'NZDUSD'
inst_id = name_to_id.get(symbol)
price = api.get_latest_asking_price(inst_id)
lots = 0.20
sl = 0.5730
tp = 0.5805

print('NZDUSD LONG - Adding 0.20 lots')
print('  Entry: %.5f | SL: %.5f | TP: %.5f' % (price, sl, tp))
try:
    order = api.create_order(
        instrument_id=inst_id,
        quantity=lots,
        side='buy',
        type_='market',
        stop_loss=sl,
        stop_loss_type='absolute',
        take_profit=tp,
        take_profit_type='absolute'
    )
    print('  SUCCESS! Order ID: %s' % order)
except Exception as e:
    print('  Error: %s' % str(e))

print()

# Verify
import time
time.sleep(1)
state = api.get_account_state()
print('=' * 60)
print('Account: Equity $%.2f | Open P/L: $%.2f' % (
    state['balance'] + state['openGrossPnL'], state['openGrossPnL']))

positions = api.get_all_positions()
total_nzdcad = 0
total_nzdusd = 0
if positions is not None and len(positions) > 0:
    for _, pos in positions.iterrows():
        qty = pos.get('qty', 0)
        sym = str(pos.get('symbol', ''))
        if 'NZDCAD' in sym:
            total_nzdcad += qty
        elif 'NZDUSD' in sym:
            total_nzdusd += qty

print('Total NZDCAD: %.2f lots' % total_nzdcad)
print('Total NZDUSD: %.2f lots' % total_nzdusd)
print('=' * 60)
