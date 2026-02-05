#!/usr/bin/env python
"""Get full account statistics including today's trades"""

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
print('FULL ACCOUNT STATISTICS')
print('=' * 70)
print()

# Get full account state
state = api.get_account_state()

print('ACCOUNT OVERVIEW:')
print('  Balance: $%.2f' % state['balance'])
print('  Projected Balance: $%.2f' % state['projectedBalance'])
print('  Available Funds: $%.2f' % state['availableFunds'])
print('  Blocked Balance: $%.2f' % state['blockedBalance'])
print()

print('MARGIN INFO:')
print('  Initial Margin Req: $%.2f' % state['initialMarginReq'])
print('  Maint Margin Req: $%.2f' % state['maintMarginReq'])
print('  Stop Out Level: %.2f%%' % state['stopOutLevel'])
print()

print('TODAY\'S STATISTICS:')
print('  Today Gross P/L: $%.2f' % state['todayGross'])
print('  Today Net P/L: $%.2f' % state['todayNet'])
print('  Today Fees: $%.2f' % state['todayFees'])
print('  Today Volume: %.2f' % state['todayVolume'])
print('  Today Trades Count: %d' % state['todayTradesCount'])
print()

print('OPEN POSITIONS:')
print('  Open Gross P/L: $%.2f' % state['openGrossPnL'])
print('  Open Net P/L: $%.2f' % state['openNetPnL'])
print('  Positions Count: %d' % state['positionsCount'])
print('  Orders Count: %d' % state['ordersCount'])
print()

# Calculate equity
equity = state['balance'] + state['openGrossPnL']
print('CURRENT EQUITY: $%.2f' % equity)
print()

# Check for any execution/fill data in orders
orders = api.get_all_orders()
if orders is not None and len(orders) > 0:
    filled_orders = orders[orders['filledQty'] > 0]
    if len(filled_orders) > 0:
        print('FILLED ORDERS:')
        for _, order in filled_orders.iterrows():
            tid = order['tradableInstrumentId']
            symbol = id_to_name.get(tid, str(tid))
            side = order['side']
            qty = order['filledQty']
            avg_price = order['avgPrice']
            print('  %s %s %.2f @ %.5f' % (symbol, side, qty, avg_price))
        print()

# Try to access trade history via different approach
print('=' * 70)
print('Checking for historical trade data...')

# List all API methods
all_methods = [m for m in dir(api) if not m.startswith('_') and callable(getattr(api, m, None))]
print('All callable API methods:')
for m in sorted(all_methods):
    print('  - %s' % m)

print()
print('=' * 70)
