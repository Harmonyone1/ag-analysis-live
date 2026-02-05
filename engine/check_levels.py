#!/usr/bin/env python
"""Check position levels"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Current prices
nzdcad_price = api.get_latest_asking_price(name_to_id.get('NZDCAD'))
nzdusd_price = api.get_latest_asking_price(name_to_id.get('NZDUSD'))

# Entry prices and levels
nzdcad_entry = 0.7924
nzdcad_sl = 0.7907
nzdcad_tp = 0.7963

nzdusd_entry = 0.5753
nzdusd_sl = 0.5730
nzdusd_tp = 0.5805

print('POSITION STATUS')
print('=' * 60)
print()
print('NZDCAD LONG 0.10 lots')
print('  Entry: %.5f | Current: %.5f' % (nzdcad_entry, nzdcad_price))
print('  SL: %.5f (%.1f pips to SL)' % (nzdcad_sl, (nzdcad_price - nzdcad_sl) * 10000))
print('  TP: %.5f (%.1f pips to TP)' % (nzdcad_tp, (nzdcad_tp - nzdcad_price) * 10000))
print('  P/L: %.1f pips' % ((nzdcad_price - nzdcad_entry) * 10000))
print()
print('NZDUSD LONG 0.20 lots')
print('  Entry: %.5f | Current: %.5f' % (nzdusd_entry, nzdusd_price))
print('  SL: %.5f (%.1f pips to SL)' % (nzdusd_sl, (nzdusd_price - nzdusd_sl) * 10000))
print('  TP: %.5f (%.1f pips to TP)' % (nzdusd_tp, (nzdusd_tp - nzdusd_price) * 10000))
print('  P/L: %.1f pips' % ((nzdusd_price - nzdusd_entry) * 10000))
print()

# Account
state = api.get_account_state()
print('Account: Equity $%.2f | Open P/L: $%.2f' % (
    state['balance'] + state['openGrossPnL'], state['openGrossPnL']))
