#!/usr/bin/env python
"""Get account trade history - closed positions, win rate, R:R"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))
id_to_name = {v: k for k, v in name_to_id.items()}

print('=' * 70)
print('ACCOUNT TRADE HISTORY')
print('=' * 70)
print()

# Check available methods
print('Checking API methods for history...')
methods = [m for m in dir(api) if not m.startswith('_')]
history_methods = [m for m in methods if 'history' in m.lower() or 'closed' in m.lower() or 'order' in m.lower() or 'trade' in m.lower()]
print('Available history methods:', history_methods)
print()

# Try different methods to get trade history
print('Attempting to retrieve trade history...')
print()

# Try get_positions_history
try:
    pos_hist = api.get_positions_history()
    print('get_positions_history():')
    if pos_hist is not None:
        print('  Type:', type(pos_hist))
        print('  Length:', len(pos_hist) if hasattr(pos_hist, '__len__') else 'N/A')
        if hasattr(pos_hist, 'columns'):
            print('  Columns:', list(pos_hist.columns))
        if hasattr(pos_hist, 'head'):
            print('  Sample:')
            print(pos_hist.head(10).to_string())
    else:
        print('  None returned')
except Exception as e:
    print('  Error:', str(e))

print()

# Try get_all_orders
try:
    orders = api.get_all_orders()
    print('get_all_orders():')
    if orders is not None:
        print('  Type:', type(orders))
        print('  Length:', len(orders) if hasattr(orders, '__len__') else 'N/A')
        if hasattr(orders, 'columns'):
            print('  Columns:', list(orders.columns))
        if hasattr(orders, 'head') and len(orders) > 0:
            print('  Sample:')
            print(orders.head(10).to_string())
    else:
        print('  None returned')
except Exception as e:
    print('  Error:', str(e))

print()

# Check account state for any history info
try:
    state = api.get_account_state()
    print('Account State Keys:', list(state.keys()) if isinstance(state, dict) else 'Not a dict')
except Exception as e:
    print('Account state error:', str(e))

print()
print('=' * 70)
