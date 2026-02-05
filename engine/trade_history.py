#!/usr/bin/env python
"""Get trade execution history"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import pandas as pd

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))
id_to_name = {v: k for k, v in name_to_id.items()}

print('=' * 70)
print('TRADE EXECUTION HISTORY')
print('=' * 70)
print()

# Get all executions
try:
    executions = api.get_all_executions()
    print('Executions retrieved:')
    if executions is not None:
        print('  Type:', type(executions))
        print('  Length:', len(executions) if hasattr(executions, '__len__') else 'N/A')
        if hasattr(executions, 'columns'):
            print('  Columns:', list(executions.columns))
            print()

            # Show all executions
            if len(executions) > 0:
                print('ALL EXECUTIONS:')
                print('-' * 70)

                # Group by position to calculate P/L
                for _, exe in executions.iterrows():
                    tid = exe.get('tradableInstrumentId', '')
                    symbol = id_to_name.get(tid, str(tid))
                    side = exe.get('side', '')
                    qty = exe.get('qty', 0)
                    price = exe.get('price', 0)
                    exec_id = exe.get('id', '')
                    pos_id = exe.get('positionId', '')
                    created = exe.get('createdDate', 0)

                    # Convert timestamp to readable format
                    from datetime import datetime
                    if created > 0:
                        dt = datetime.fromtimestamp(created / 1000)
                        date_str = dt.strftime('%m/%d %H:%M')
                    else:
                        date_str = 'N/A'

                    print('  %s | %s %s %.2f @ %.5f' % (date_str, symbol, side.upper(), qty, price))

                print()
                print('-' * 70)

                # Calculate statistics
                # Group trades by symbol and calculate realized P/L
                trades_by_symbol = {}
                for _, exe in executions.iterrows():
                    tid = exe.get('tradableInstrumentId', '')
                    symbol = id_to_name.get(tid, str(tid))
                    side = exe.get('side', '')
                    qty = exe.get('qty', 0)
                    price = exe.get('price', 0)

                    if symbol not in trades_by_symbol:
                        trades_by_symbol[symbol] = []
                    trades_by_symbol[symbol].append({
                        'side': side,
                        'qty': qty,
                        'price': price
                    })

                print()
                print('TRADES BY SYMBOL:')
                for sym, trades in trades_by_symbol.items():
                    buys = [t for t in trades if t['side'] == 'buy']
                    sells = [t for t in trades if t['side'] == 'sell']

                    buy_qty = sum(t['qty'] for t in buys)
                    sell_qty = sum(t['qty'] for t in sells)

                    print('  %s: %d buys (%.2f lots), %d sells (%.2f lots)' % (
                        sym, len(buys), buy_qty, len(sells), sell_qty))

                    # If there are matching buys and sells, calculate P/L
                    if buys and sells:
                        avg_buy = sum(t['qty'] * t['price'] for t in buys) / buy_qty if buy_qty > 0 else 0
                        avg_sell = sum(t['qty'] * t['price'] for t in sells) / sell_qty if sell_qty > 0 else 0
                        closed_qty = min(buy_qty, sell_qty)

                        # Simplified P/L calculation
                        if 'USD' in sym:
                            pip_value = 10  # per standard lot
                        else:
                            pip_value = 7.5  # approximate for cross pairs

                        pips = (avg_sell - avg_buy) * 10000
                        estimated_pnl = pips * pip_value * closed_qty / 100000
                        print('    Closed: %.2f lots | Avg Buy: %.5f | Avg Sell: %.5f' % (
                            closed_qty, avg_buy, avg_sell))
                        print('    Estimated P/L: $%.2f' % estimated_pnl)

        else:
            print('  Data:', executions)
    else:
        print('  None returned')
except Exception as e:
    print('  Error:', str(e))
    import traceback
    traceback.print_exc()

print()

# Get account state for summary
state = api.get_account_state()
print('=' * 70)
print('ACCOUNT SUMMARY')
print('=' * 70)
print('  Balance: $%.2f' % state['balance'])
print('  Open P/L: $%.2f' % state['openGrossPnL'])
print('  Equity: $%.2f' % (state['balance'] + state['openGrossPnL']))
print()
print('  Today Trades: %d' % state['todayTradesCount'])
print('  Today Gross: $%.2f' % state['todayGross'])
print('  Today Net: $%.2f' % state['todayNet'])
print('  Today Fees: $%.2f' % state['todayFees'])
print('=' * 70)
