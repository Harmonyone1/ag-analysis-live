import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

def get_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.0001
    return 100 - (100 / (1 + avg_gain/avg_loss))

print('='*70)
print('NEXT SETUP SCAN')
print('='*70)

# Current positions
positions = api.get_all_positions()
if positions is not None and len(positions) > 0:
    print('CURRENT POSITIONS:')
    total_pnl = 0
    for _, pos in positions.iterrows():
        pnl = pos['unrealizedPl']
        total_pnl += pnl
    print(f'  Total P/L: ${total_pnl:.2f}')
    print()

for sym in ['BTCUSD', 'ETHUSD']:
    inst_id = symbol_to_id[sym]
    time.sleep(0.8)

    h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='1W')
    time.sleep(0.3)
    h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='3D')
    time.sleep(0.3)
    m15 = api.get_price_history(inst_id, resolution='15m', lookback_period='1D')

    price = m15['c'].values[-1]

    print(f'{sym} @ ${price:.2f}')
    print('-'*70)

    # RSI across timeframes
    h4_rsi = get_rsi(h4['c'].values)
    h1_rsi = get_rsi(h1['c'].values)
    m15_rsi = get_rsi(m15['c'].values)
    print(f'RSI: H4={h4_rsi:.1f} | H1={h1_rsi:.1f} | M15={m15_rsi:.1f}')

    # Key levels above (resistance / bearish OBs)
    print(f'\nRESISTANCE LEVELS ABOVE:')
    h1_o, h1_h, h1_l, h1_c = h1['o'].values, h1['h'].values, h1['l'].values, h1['c'].values

    bearish_obs = []
    for i in range(2, 40):
        idx = len(h1_c) - i - 1
        if idx < 1:
            break
        if h1_c[idx] > h1_o[idx] and h1_c[idx+1] < h1_o[idx+1]:
            ob_range = h1_h[idx] - h1_l[idx]
            move = h1_o[idx+1] - h1_c[idx+1]
            if move > ob_range * 0.8 and h1_l[idx] > price:
                bearish_obs.append({'low': h1_l[idx], 'high': h1_h[idx], 'strength': move/ob_range})

    if bearish_obs:
        for ob in sorted(bearish_obs, key=lambda x: x['low'])[:3]:
            print(f'  BEARISH OB: ${ob["low"]:.2f} - ${ob["high"]:.2f} (strength: {ob["strength"]:.1f}x)')
    else:
        print('  No bearish OBs found above')

    # Support levels below (bullish OBs)
    print(f'\nSUPPORT LEVELS BELOW:')
    bullish_obs = []
    for i in range(2, 40):
        idx = len(h1_c) - i - 1
        if idx < 1:
            break
        if h1_c[idx] < h1_o[idx] and h1_c[idx+1] > h1_o[idx+1]:
            ob_range = h1_h[idx] - h1_l[idx]
            move = h1_c[idx+1] - h1_o[idx+1]
            if move > ob_range * 0.5 and h1_h[idx] < price:
                bullish_obs.append({'low': h1_l[idx], 'high': h1_h[idx], 'strength': move/ob_range})

    if bullish_obs:
        for ob in sorted(bullish_obs, key=lambda x: x['high'], reverse=True)[:3]:
            print(f'  BULLISH OB: ${ob["low"]:.2f} - ${ob["high"]:.2f} (strength: {ob["strength"]:.1f}x)')
    else:
        print('  No bullish OBs found below')

    # Next setup recommendation
    print(f'\nNEXT SETUP:')
    if m15_rsi > 65:
        print(f'  RSI getting elevated ({m15_rsi:.1f}) - watch for exhaustion')
        print(f'  POTENTIAL SHORT if price hits resistance with rejection')
    elif m15_rsi < 35:
        print(f'  RSI still low ({m15_rsi:.1f}) - continuation long possible')
    else:
        print(f'  RSI neutral ({m15_rsi:.1f})')
        if bearish_obs:
            print(f'  Watch ${bearish_obs[0]["low"]:.2f} for SHORT setup on rejection')
        if bullish_obs:
            print(f'  Watch ${bullish_obs[0]["high"]:.2f} for LONG setup on pullback')

    print()

print('='*70)
print('SUMMARY - NEXT PLAYS:')
print('='*70)
print('1. After ETH TP ($3,000):')
print('   - If momentum continues: look for pullback to $2,970-$2,980 to re-long')
print('   - If rejection at $3,000+: watch for SHORT setup')
print()
print('2. BTC approaching $89,000:')
print('   - Let it run to TP')
print('   - Watch $89,000-$90,000 zone for potential reversal')
print()
print('3. Continuation scenario:')
print('   - Both break higher = strong bullish momentum')
print('   - Add on pullbacks to prior resistance (now support)')
