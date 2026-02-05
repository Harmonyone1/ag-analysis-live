#!/usr/bin/env python
"""Monitor active positions and potential entries"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Key levels to watch
NZDUSD_OB_TOP = 0.57531
NZDUSD_OB_BOTTOM = 0.57437

print('=' * 70)
print('POSITION & SETUP MONITOR | Started %s' % datetime.now().strftime('%I:%M:%S %p'))
print('=' * 70)
print()
print('Active: NZDCAD LONG @ 0.79249 | SL: 0.7907 | TP: 0.7963')
print('Watching: NZDUSD Bullish OB at 0.57437-0.57531')
print()

for cycle in range(20):  # 20 cycles x 30 sec = 10 minutes
    try:
        now = datetime.now().strftime('%I:%M:%S %p')

        # Get account state
        state = api.get_account_state()
        equity = state['balance'] + state['openGrossPnL']
        pnl = state['openGrossPnL']

        # Get positions
        positions = api.get_all_positions()

        # Get NZDUSD price
        nzdusd_id = name_to_id.get('NZDUSD')
        nzdusd_price = api.get_latest_asking_price(nzdusd_id)
        dist_to_ob = (nzdusd_price - NZDUSD_OB_TOP) * 10000

        print('[%s] Equity: $%.2f | Open P/L: $%.2f' % (now, equity, pnl))

        if positions is not None and len(positions) > 0:
            for _, pos in positions.iterrows():
                symbol = pos.get('symbol', 'N/A')
                side = pos.get('side', 'N/A')
                pnl_pos = pos.get('unrealizedPnL', 0)
                print('  %s %s | P/L: $%.2f' % (symbol, side, pnl_pos))

        # NZDUSD check
        if nzdusd_price <= NZDUSD_OB_TOP:
            print('  >>> NZDUSD AT OB %.5f - ENTRY ZONE! <<<' % nzdusd_price)
        else:
            print('  NZDUSD: %.5f (%.1f pips above OB)' % (nzdusd_price, dist_to_ob))

        print()

        if cycle < 19:
            time.sleep(30)

    except Exception as e:
        print('Error: %s' % str(e)[:50])
        time.sleep(30)

print('=' * 70)
print('Monitor ended')
print('=' * 70)
