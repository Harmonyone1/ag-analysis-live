#!/usr/bin/env python
"""Get FULL account trade history - all time"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import inspect

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))
id_to_name = {v: k for k, v in name_to_id.items()}

print('=' * 70)
print('CHECKING API FOR FULL HISTORY ACCESS')
print('=' * 70)
print()

# Check get_all_executions signature
print('get_all_executions signature:')
try:
    sig = inspect.signature(api.get_all_executions)
    print('  Parameters:', sig)
except Exception as e:
    print('  Error:', e)

print()

# Try calling with different parameters
print('Trying get_all_executions with lookback...')
try:
    # Try with a long lookback period
    execs = api.get_all_executions()
    print('  Default call returned %d executions' % (len(execs) if execs is not None else 0))

    # Check the date range of returned data
    if execs is not None and len(execs) > 0:
        from datetime import datetime
        dates = execs['createdDate'].values
        min_date = datetime.fromtimestamp(min(dates) / 1000)
        max_date = datetime.fromtimestamp(max(dates) / 1000)
        print('  Date range: %s to %s' % (min_date.strftime('%Y-%m-%d %H:%M'), max_date.strftime('%Y-%m-%d %H:%M')))
except Exception as e:
    print('  Error:', e)

print()

# Check session details for account creation date
print('Session/Account details:')
try:
    session = api.get_session_details()
    print('  Session details:', session)
except Exception as e:
    print('  Error:', e)

print()

# Check all accounts
print('All trade accounts:')
try:
    accounts = api.get_trade_accounts()
    print('  Accounts:', accounts)
    if accounts is not None:
        for acc in accounts if isinstance(accounts, list) else [accounts]:
            print('  Account data:', acc)
except Exception as e:
    print('  Error:', e)

print()

# Try to access the underlying API directly
print('Checking for raw API access...')
try:
    # Check if there's a way to make direct API calls
    config = api.get_config()
    print('  Config:', config)
except Exception as e:
    print('  Error:', e)

print()

# Check account state for historical info
state = api.get_account_state()
print('Account State (all keys):')
for key, value in state.items():
    print('  %s: %s' % (key, value))

print()
print('=' * 70)
