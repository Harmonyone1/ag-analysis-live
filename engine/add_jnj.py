#!/usr/bin/env python
"""Add to JNJ position"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

symbol = 'JNJ'
inst_id = name_to_id.get(symbol)
price = api.get_latest_asking_price(inst_id)
sl = round(price * 0.97, 2)
tp = round(price * 1.06, 2)
lots = 0.01

print('JNJ - Adding 0.01 lots')
print('  Confidence: HIGH (HTF BULL + DISC 18%% + RSI 37.1)')
print('  Entry: $%.2f | SL: $%.2f | TP: $%.2f' % (price, sl, tp))

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
