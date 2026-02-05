#!/usr/bin/env python
"""Quick status check"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
id_to_name = {v: k for k, v in dict(zip(instruments['name'], instruments['tradableInstrumentId'])).items()}

now = datetime.now()
print('Current Time: %s' % now.strftime('%I:%M:%S %p'))
print()

state = api.get_account_state()
print('Account: Equity $%.2f | P/L $%.2f' % (state['balance'] + state['openGrossPnL'], state['openGrossPnL']))
print()

positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    print('OPEN POSITIONS:')
    pos_summary = {}
    for _, pos in positions.iterrows():
        tid = str(pos.get('tradableInstrumentId', ''))
        symbol = id_to_name.get(int(tid), tid) if tid.isdigit() else tid
        qty = pos.get('qty', 0)
        if symbol not in pos_summary:
            pos_summary[symbol] = 0
        pos_summary[symbol] += qty

    for sym, qty in pos_summary.items():
        print('  %s: %.2f lots' % (sym, qty))
