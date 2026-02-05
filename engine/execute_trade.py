#!/usr/bin/env python
"""Execute NZDCAD LONG Trade"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

symbol = 'NZDCAD'
inst_id = name_to_id.get(symbol)
price = api.get_latest_asking_price(inst_id)

# Trade parameters - Framework + ICT Confluence
# HTF BULL aligned, DISC 21%, RSI 23.5, at H1 Bullish OB
entry = price
sl = 0.7907   # Below swing low (~18 pips)
tp = 0.7963   # 2R target (~38 pips)
lots = 0.02   # Conservative size

print('=' * 60)
print('NZDCAD LONG | %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 60)
print()
print('SETUP CONFIRMATION:')
print('  HTF: BULL (D1+H4 aligned)')
print('  Zone: DISCOUNT (21%)')
print('  RSI: 23.5 (< 40 - oversold)')
print('  ICT: At H1 Bullish Order Block')
print()
print('TRADE PARAMETERS:')
print('  Entry: %.5f (market)' % entry)
print('  Stop Loss: %.5f (%.1f pips)' % (sl, (entry - sl) * 10000))
print('  Take Profit: %.5f (%.1f pips)' % (tp, (tp - entry) * 10000))
print('  R:R = 1:%.1f' % ((tp - entry) / (entry - sl)))
print('  Lots: %.2f' % lots)
print()

# Execute
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
    print('ORDER EXECUTED!')
    print('Result: %s' % order)
except Exception as e:
    print('ORDER ERROR: %s' % str(e))

print()
print('=' * 60)

# Verify position
import time
time.sleep(1)
positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    print('OPEN POSITIONS:')
    for _, pos in positions.iterrows():
        print('  %s %s %.2f lots' % (pos.get('symbol', symbol), pos.get('side', 'N/A'), pos.get('qty', 0)))
else:
    print('No positions (order may have failed)')
