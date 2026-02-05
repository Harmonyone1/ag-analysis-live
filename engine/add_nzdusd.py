#!/usr/bin/env python
"""Add to NZDUSD LONG position"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

symbol = 'NZDUSD'
inst_id = name_to_id.get(symbol)
price = api.get_latest_asking_price(inst_id)

# Adding 0.20 more lots (will be 0.40 total)
additional_lots = 0.20
sl = 0.5730
tp = 0.5805

print('=' * 60)
print('ADD TO NZDUSD LONG | %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 60)
print()
print('CONFIDENCE: HIGH')
print('  - HTF BULL aligned (D1+H4)')
print('  - DISCOUNT zone (27%%)')
print('  - RSI < 40 (oversold)')
print('  - At H4 Bullish Order Block')
print()
print('Adding %.2f lots at %.5f' % (additional_lots, price))
print('SL: %.5f | TP: %.5f' % (sl, tp))
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

# Verify positions
import time
time.sleep(1)
state = api.get_account_state()
print('Equity: $%.2f | Open P/L: $%.2f' % (
    state['balance'] + state['openGrossPnL'], state['openGrossPnL']))

positions = api.get_all_positions()
total_nzdusd = 0
if positions is not None and len(positions) > 0:
    print()
    print('All Positions:')
    for _, pos in positions.iterrows():
        qty = pos.get('qty', 0)
        symbol_pos = str(pos.get('symbol', 'N/A'))
        if 'NZDUSD' in symbol_pos:
            total_nzdusd += qty
        print('  %s %s %.2f lots | P/L: $%.2f' % (
            symbol_pos, pos.get('side', 'N/A'), qty, pos.get('unrealizedPnL', 0)))
    print()
    print('Total NZDUSD: %.2f lots' % total_nzdusd)
