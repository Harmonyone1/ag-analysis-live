#!/usr/bin/env python
"""Quick status check using direct API."""
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)


from config import get_api
api = get_api()

# Get instruments for mapping
instruments = api.get_all_instruments()
id_to_symbol = dict(zip(instruments['tradableInstrumentId'], instruments['name']))

# Account state
state = api.get_account_state()
print(f"Balance: ${state['balance']:.2f} | Equity: ${state['balance'] + state['openGrossPnL']:.2f} | Open PnL: ${state['openGrossPnL']:.2f}")

# Positions
print('\n=== OPEN POSITIONS ===')
positions = api.get_all_positions()
if not positions.empty:
    for _, p in positions.iterrows():
        sym = id_to_symbol.get(p['tradableInstrumentId'], '?')
        pnl = p['unrealizedPl']
        print(f"{sym} {p['side']} {p['qty']} @ {p['avgPrice']} | PnL: ${pnl:.2f}")
else:
    print('No open positions')

# Pending orders with distances
print('\n=== PENDING LIMIT ORDERS ===')
orders = api.get_all_orders()
if not orders.empty:
    limit_orders = orders[(orders['type'] == 'limit') & (orders['positionId'] == 0)]
    for _, o in limit_orders.iterrows():
        inst_id = o['tradableInstrumentId']
        sym = id_to_symbol.get(inst_id, '?')
        entry = o['price']
        side = o['side']

        try:
            current = api.get_latest_asking_price(inst_id)
            if sym in ['BTCUSD', 'XAUUSD', 'ETHUSD', 'US30', 'NAS100', 'SPX500']:
                dist = abs(current - entry)
                unit = 'pts'
                print(f"{sym} {side} @ {entry:.2f} | Now: {current:.2f} | {dist:.1f} {unit}")
            else:
                dist = abs(current - entry) * 10000
                unit = 'pips'
                print(f"{sym} {side} @ {entry:.5f} | Now: {current:.5f} | {dist:.1f} {unit}")
        except Exception as e:
            print(f"{sym} {side} @ {entry} | Error: {e}")
else:
    print('No pending orders')
