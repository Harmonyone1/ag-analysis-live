import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time
from datetime import datetime

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Entry zones
ETH_ENTRY_LOW = 2980
ETH_ENTRY_HIGH = 2992
BTC_ENTRY_LOW = 88266
BTC_ENTRY_HIGH = 88400

print('='*60)
print('MONITORING FOR RE-ENTRY')
print(f'Started: {datetime.now().strftime("%H:%M:%S")}')
print('='*60)
print(f'ETH Entry Zone: ${ETH_ENTRY_LOW} - ${ETH_ENTRY_HIGH}')
print(f'BTC Entry Zone: ${BTC_ENTRY_LOW} - ${BTC_ENTRY_HIGH}')
print('='*60)

scan = 0
while True:
    scan += 1
    now = datetime.now().strftime("%H:%M:%S")

    try:
        btc_price = api.get_latest_bid_price(symbol_to_id['BTCUSD'])
        eth_price = api.get_latest_bid_price(symbol_to_id['ETHUSD'])

        time.sleep(0.3)
        positions = api.get_all_positions()
        pos_count = len(positions) if positions is not None else 0

        time.sleep(0.3)
        state = api.get_account_state()
        balance = float(state.get('balance', 0))

        # Check positions
        btc_pnl = 0
        has_btc = False
        has_eth = False
        if positions is not None and len(positions) > 0:
            for _, pos in positions.iterrows():
                if pos['tradableInstrumentId'] == 3378:
                    has_btc = True
                    btc_pnl = pos['unrealizedPl']
                if pos['tradableInstrumentId'] == 3379:
                    has_eth = True

        print(f'[{now}] #{scan} | BTC: ${btc_price:.2f} | ETH: ${eth_price:.2f} | Bal: ${balance:.2f}')

        # Check BTC TP
        if has_btc and btc_price >= 89000:
            print('>>> BTC HIT $89,000 TARGET!')

        if has_btc:
            print(f'  BTC LONG P/L: ${btc_pnl:.2f} | To TP: ${89000 - btc_price:.2f}')

        # Check ETH entry zone
        if not has_eth and eth_price >= ETH_ENTRY_LOW and eth_price <= ETH_ENTRY_HIGH:
            print(f'  >>> ETH IN ENTRY ZONE! ${eth_price:.2f}')
            print(f'  >>> ENTERING ETH LONG...')

            order = api.create_order(
                instrument_id=symbol_to_id['ETHUSD'],
                quantity=0.6,
                side='buy',
                type_='market',
                stop_loss=2950,
                stop_loss_type='absolute',
                take_profit=3050,
                take_profit_type='absolute'
            )
            print(f'  >>> Order ID: {order}')

        # Check BTC entry zone (if BTC TP hit and we want to re-enter)
        if not has_btc and btc_price >= BTC_ENTRY_LOW and btc_price <= BTC_ENTRY_HIGH:
            print(f'  >>> BTC IN ENTRY ZONE! ${btc_price:.2f}')

    except Exception as e:
        print(f'[{now}] Error: {e}')

    time.sleep(30)
