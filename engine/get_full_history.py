#!/usr/bin/env python
"""Get full orders history using tradelocker internal methods"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import requests

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))
id_to_name = {v: k for k, v in name_to_id.items()}

print('=' * 70)
print('FULL ACCOUNT TRADE HISTORY')
print('=' * 70)
print()

# Access internal session and headers
try:
    # The TLAPI uses internal _session for requests
    session = api._session if hasattr(api, '_session') else None
    headers = api._headers if hasattr(api, '_headers') else None

    print('Checking internal API structure...')
    internal_attrs = [a for a in dir(api) if a.startswith('_') and not a.startswith('__')]
    print('Internal attributes:', internal_attrs[:10])

    # Try to find the request method
    if hasattr(api, '_make_request'):
        print('Has _make_request method')
    if hasattr(api, '_get'):
        print('Has _get method')
    if hasattr(api, '_post'):
        print('Has _post method')

    # Check the source of get_all_executions
    import inspect
    source = inspect.getsourcefile(api.get_all_executions)
    print('Source file:', source)

except Exception as e:
    print('Error exploring API:', e)

print()

# Try using the library's built-in session
try:
    # Access token and make authenticated request
    access_token = api.get_access_token()
    base_url = 'https://live.tradelocker.com/backend-api'
    account_id = 592535

    # Get the route ID from an instrument
    inst_id = name_to_id.get('EURUSD')
    route_id = api.get_trade_route_id(inst_id)

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'accNum': str(account_id),
        'routeId': str(route_id)
    }

    # Try orders history endpoint
    url = f'{base_url}/trade/accounts/{account_id}/ordersHistory'
    response = requests.get(url, headers=headers)
    print('Orders History Response:', response.status_code)

    if response.status_code == 200:
        data = response.json()
        print('Data keys:', data.keys() if isinstance(data, dict) else type(data))
        if 'd' in data:
            print('d keys:', data['d'].keys() if isinstance(data['d'], dict) else data['d'])
    else:
        # Try with different header format
        headers2 = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
        }
        params = {'accNum': account_id}
        response2 = requests.get(url, headers=headers2, params=params)
        print('Second attempt:', response2.status_code)

        if response2.status_code != 200:
            print('Response:', response2.text[:300])

except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()

print()
print('=' * 70)

# Fallback: Use what we have - the balance change tells the story
state = api.get_account_state()
balance = state['balance']

print('ACCOUNT ANALYSIS (from balance)')
print('=' * 70)
print()

# The account started at some point and now has this balance
# Balance = Starting capital + realized P/L
# Current balance is $336.50

print('Current Balance: $%.2f' % balance)
print('Open P/L: $%.2f' % state['openGrossPnL'])
print('Equity: $%.2f' % (balance + state['openGrossPnL']))
print()

# If we know the starting capital, we can calculate total realized P/L
# Common demo starting amounts: $100, $500, $1000, $5000, $10000, $50000, $100000
starting_options = [100, 500, 1000, 5000, 10000, 50000, 100000]

print('Possible starting capitals and realized P/L:')
for start in starting_options:
    realized = balance - start
    if -1000 < realized < 1000:  # Reasonable range
        pct = (realized / start) * 100
        print('  If started at $%d: Realized P/L = $%.2f (%.1f%%)' % (start, realized, pct))

print()
print('=' * 70)
