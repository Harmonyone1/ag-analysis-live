"""
Pending Entry Monitor - USDCAD SHORT and NZDUSD LONG
Monitors for pullback entries with HTF alignment confirmed
"""
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time
import sys

from config import get_api

# Entry Plans (HTF Aligned)
ENTRIES = {
    'USDCAD': {
        'direction': 'SHORT',
        'entry_low': 1.3775,
        'entry_high': 1.3790,
        'entry_target': 1.3780,
        'stop_loss': 1.3825,
        'take_profit': 1.3685,
        'lot_size': 0.25,
        'htf_bias': 'BEARISH - Monthly/Weekly downtrend',
        'trigger': 'Rally to OTE zone + bearish rejection'
    },
    'NZDUSD': {
        'direction': 'LONG',
        'entry_low': 0.5735,
        'entry_high': 0.5770,
        'entry_target': 0.5755,
        'stop_loss': 0.5715,
        'take_profit': 0.5850,
        'lot_size': 0.25,
        'htf_bias': 'BULLISH - Monthly/Weekly uptrend',
        'trigger': 'Pullback to discount + bullish FVG'
    }
}

def check_entries(execute=False):
    """Check if price is in entry zone for pending trades"""
    api = get_api()
    instruments = api.get_all_instruments()
    symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

    print('=' * 70)
    print('PENDING ENTRY MONITOR')
    print('=' * 70)
    print()

    for symbol, plan in ENTRIES.items():
        inst_id = symbol_to_id[symbol]
        bars = api.get_price_history(inst_id, resolution='15m', lookback_period='1D')
        price = bars['c'].values[-1]
        h1_bars = api.get_price_history(inst_id, resolution='1H', lookback_period='2D')

        # Calculate RSI
        closes = h1_bars['c'].values
        if len(closes) > 14:
            deltas = closes[-15:] - closes[-16:-1]
            gains = sum([d for d in deltas if d > 0])
            losses = sum([-d for d in deltas if d < 0])
            if losses > 0:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
        else:
            rsi = 50

        print(f'{symbol} - {plan["direction"]}')
        print('-' * 70)
        print(f'  HTF Bias: {plan["htf_bias"]}')
        print(f'  Current Price: {price:.5f}')
        print(f'  H1 RSI: {rsi:.1f}')
        print(f'  Entry Zone: {plan["entry_low"]:.5f} - {plan["entry_high"]:.5f}')
        print(f'  Target Entry: {plan["entry_target"]:.5f}')
        print(f'  Stop Loss: {plan["stop_loss"]:.5f}')
        print(f'  Take Profit: {plan["take_profit"]:.5f}')

        # Check if in zone
        in_zone = False
        if plan['direction'] == 'SHORT':
            # For short, we want price to rally INTO zone
            if price >= plan['entry_low'] and price <= plan['entry_high']:
                in_zone = True
                print(f'  >>> IN ENTRY ZONE! <<<')
                if rsi > 65:
                    print(f'  >>> RSI OVERBOUGHT ({rsi:.1f}) - ENTRY SIGNAL! <<<')
            else:
                dist = (plan['entry_low'] - price) * 10000
                print(f'  Distance to zone: {dist:.1f} pips (need rally)')
        else:
            # For long, we want price to pullback INTO zone
            if price >= plan['entry_low'] and price <= plan['entry_high']:
                in_zone = True
                print(f'  >>> IN ENTRY ZONE! <<<')
                if rsi < 35:
                    print(f'  >>> RSI OVERSOLD ({rsi:.1f}) - ENTRY SIGNAL! <<<')
            else:
                dist = (price - plan['entry_high']) * 10000
                print(f'  Distance to zone: {dist:.1f} pips (need pullback)')

        # Execute if requested and in zone
        if execute and in_zone:
            print()
            print(f'  EXECUTING {plan["direction"]} ORDER...')
            side = 'sell' if plan['direction'] == 'SHORT' else 'buy'
            try:
                order = api.create_order(
                    instrument_id=inst_id,
                    quantity=plan['lot_size'],
                    side=side,
                    type_='market',
                    stop_loss=plan['stop_loss'],
                    take_profit=plan['take_profit'],
                    stop_loss_type='absolute',
                    take_profit_type='absolute'
                )
                print(f'  Order Result: {order}')
            except Exception as e:
                print(f'  Order Error: {e}')

        print()
        time.sleep(0.5)

    # Show current positions
    print('=' * 70)
    print('CURRENT POSITIONS')
    print('=' * 70)
    positions = api.get_all_positions()
    if positions is not None and len(positions) > 0:
        cols = positions.columns.tolist()
        for _, p in positions.iterrows():
            sym = p.get('symbol', p.get('instrumentId', 'Unknown'))
            side = p.get('side', p.get('tradeSide', 'Unknown'))
            qty = p.get('qty', p.get('quantity', 0))
            avg = float(p.get('avgPrice', p.get('averagePrice', 0)))
            pnl = float(p.get('unrealizedPL', p.get('unrealizedPnl', 0)))
            print(f'  {sym}: {side} {qty} @ {avg:.5f} | P/L: ${pnl:.2f}')
    else:
        print('  No open positions')

    print()
    state = api.get_account_state()
    balance = float(state.get('accountBalance', 0))
    equity = float(state.get('accountEquity', 0))
    print(f'Balance: ${balance:.2f} | Equity: ${equity:.2f}')

def continuous_monitor(interval=60):
    """Continuously monitor for entry signals"""
    print('Starting continuous monitor...')
    print(f'Checking every {interval} seconds')
    print('Press Ctrl+C to stop')
    print()

    while True:
        try:
            check_entries(execute=False)
            print(f'\nNext check in {interval} seconds...\n')
            time.sleep(interval)
        except KeyboardInterrupt:
            print('\nMonitor stopped.')
            break
        except Exception as e:
            print(f'Error: {e}')
            time.sleep(30)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--execute':
            check_entries(execute=True)
        elif sys.argv[1] == '--monitor':
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            continuous_monitor(interval)
    else:
        check_entries(execute=False)
