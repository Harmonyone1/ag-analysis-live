#!/usr/bin/env python
"""Enter JNJ and CVX LONG positions"""

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
print('ENTERING STOCK POSITIONS | %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 60)
print()

# JNJ LONG
symbol = 'JNJ'
inst_id = name_to_id.get(symbol)
if inst_id:
    price = api.get_latest_asking_price(inst_id)
    # Set SL 3% below, TP 6% above for 2R
    sl = round(price * 0.97, 2)
    tp = round(price * 1.06, 2)
    lots = 0.01  # Minimum size for margin

    print('JNJ LONG')
    print('  Setup: HTF BULL + DISC 18%% + RSI 37.1')
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
    print()

# CVX LONG
symbol = 'CVX'
inst_id = name_to_id.get(symbol)
if inst_id:
    price = api.get_latest_asking_price(inst_id)
    sl = round(price * 0.97, 2)
    tp = round(price * 1.06, 2)
    lots = 0.01  # Minimum size for margin

    print('CVX LONG')
    print('  Setup: HTF BULL + DISC 27%% + RSI 37.7')
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
    print()

# Verify
import time
time.sleep(1)
state = api.get_account_state()
print('=' * 60)
print('Account: Equity $%.2f | Open P/L: $%.2f' % (
    state['balance'] + state['openGrossPnL'], state['openGrossPnL']))
print('=' * 60)
