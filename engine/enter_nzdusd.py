#!/usr/bin/env python
"""Enter NZDUSD LONG - Valid Framework Setup at OB"""

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

# Trade parameters - Framework Valid + at OB
# HTF BULL aligned, DISC 27%, RSI 35.6
sl = 0.5730    # Below swing low (25 pips)
tp = 0.5805    # 2R target (50 pips)
lots = 0.20    # For growth

print('=' * 60)
print('NZDUSD LONG | %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 60)
print()
print('SETUP:')
print('  HTF: BULL (D1+H4 aligned)')
print('  Zone: DISCOUNT (27%)')
print('  RSI: 35.6 (< 40)')
print('  ICT: At H4 Bullish OB')
print()
print('TRADE:')
print('  Entry: %.5f' % price)
print('  SL: %.5f (%.1f pips)' % (sl, (price - sl) * 10000))
print('  TP: %.5f (%.1f pips)' % (tp, (tp - price) * 10000))
print('  R:R = 1:%.1f' % ((tp - price) / (price - sl)))
print('  Lots: %.2f' % lots)
print()

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
    print('ORDER EXECUTED! ID: %s' % order)
except Exception as e:
    print('Error: %s' % str(e))

print()

# Verify
import time
time.sleep(1)
state = api.get_account_state()
print('Equity: $%.2f | Open P/L: $%.2f' % (
    state['balance'] + state['openGrossPnL'],
    state['openGrossPnL']))

positions = api.get_all_positions()
print()
print('All Positions:')
if positions is not None and len(positions) > 0:
    for _, pos in positions.iterrows():
        print('  %s %s %.2f lots | P/L: $%.2f' % (
            pos.get('symbol', 'N/A'),
            pos.get('side', 'N/A'),
            pos.get('qty', 0),
            pos.get('unrealizedPnL', 0)))
