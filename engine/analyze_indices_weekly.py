import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

def detect_fvgs(highs, lows, closes):
    bullish_fvgs = []
    bearish_fvgs = []
    current = closes[-1]
    for i in range(2, min(20, len(closes)-1)):
        idx = len(closes) - i - 1
        if idx < 1:
            break
        if lows[idx+1] > highs[idx-1]:
            gap_low = highs[idx-1]
            gap_high = lows[idx+1]
            if current < gap_high:
                bullish_fvgs.append({'low': gap_low, 'high': gap_high, 'age': i})
        if highs[idx+1] < lows[idx-1]:
            gap_high = lows[idx-1]
            gap_low = highs[idx+1]
            if current > gap_low:
                bearish_fvgs.append({'low': gap_low, 'high': gap_high, 'age': i})
    return bullish_fvgs[:3], bearish_fvgs[:3]

def detect_order_blocks(opens, highs, lows, closes, lookback=30):
    order_blocks = []
    current = closes[-1]
    for i in range(2, min(lookback, len(closes)-1)):
        idx = len(closes) - i - 1
        if idx < 1:
            break
        is_bullish = closes[idx] > opens[idx]
        next_bearish = closes[idx+1] < opens[idx+1]
        if is_bullish and next_bearish:
            ob_range = highs[idx] - lows[idx]
            move = opens[idx+1] - closes[idx+1]
            if ob_range > 0 and move > ob_range * 1.2:
                ob_high = highs[idx]
                ob_low = lows[idx]
                if current < ob_low:
                    status = 'ACTIVE'
                elif current >= ob_low and current <= ob_high:
                    status = 'IN_ZONE'
                else:
                    status = 'BROKEN'
                if status != 'BROKEN':
                    order_blocks.append({'type': 'BEARISH', 'high': ob_high, 'low': ob_low, 'status': status, 'strength': move/ob_range})
        is_bearish = closes[idx] < opens[idx]
        next_bullish = closes[idx+1] > opens[idx+1]
        if is_bearish and next_bullish:
            ob_range = highs[idx] - lows[idx]
            move = closes[idx+1] - opens[idx+1]
            if ob_range > 0 and move > ob_range * 1.2:
                ob_high = highs[idx]
                ob_low = lows[idx]
                if current > ob_high:
                    status = 'ACTIVE'
                elif current >= ob_low and current <= ob_high:
                    status = 'IN_ZONE'
                else:
                    status = 'BROKEN'
                if status != 'BROKEN':
                    order_blocks.append({'type': 'BULLISH', 'high': ob_high, 'low': ob_low, 'status': status, 'strength': move/ob_range})
    return order_blocks[:5]

def get_rsi(closes, period=14):
    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.0001
    return 100 - (100 / (1 + avg_gain/avg_loss))

def analyze_index(sym):
    inst_id = symbol_to_id.get(sym)
    if not inst_id:
        return None
    try:
        d1 = api.get_price_history(inst_id, resolution='1D', lookback_period='1M')
        h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='2W')
        if d1 is None or h4 is None or len(d1) < 10 or len(h4) < 10:
            return None
        d_closes = d1['c'].values
        d_highs = d1['h'].values
        d_lows = d1['l'].values
        d_opens = d1['o'].values
        h4_closes = h4['c'].values
        h4_highs = h4['h'].values
        h4_lows = h4['l'].values
        h4_opens = h4['o'].values
        current = d_closes[-1]
        d1_trend = 'BULL' if d_closes[-1] > d_closes[-5] else 'BEAR'
        h4_trend = 'BULL' if h4_closes[-1] > h4_closes[-6] else 'BEAR'
        d1_rsi = get_rsi(d_closes)
        h4_rsi = get_rsi(h4_closes)
        week_high = max(d_highs[-5:])
        week_low = min(d_lows[-5:])
        range_pct = (current - week_low) / (week_high - week_low) * 100 if week_high > week_low else 50
        zone = 'PREMIUM' if range_pct > 50 else 'DISCOUNT'
        bull_fvgs, bear_fvgs = detect_fvgs(h4_highs, h4_lows, h4_closes)
        obs = detect_order_blocks(h4_opens, h4_highs, h4_lows, h4_closes)
        swing_high = max(d_highs[-10:])
        swing_low = min(d_lows[-10:])
        if d_closes[-1] > d_closes[-2] > d_closes[-3]:
            structure = 'HH+HL (Bullish)'
        elif d_closes[-1] < d_closes[-2] < d_closes[-3]:
            structure = 'LL+LH (Bearish)'
        else:
            structure = 'Consolidating'
        return {
            'sym': sym, 'price': current, 'd1_trend': d1_trend, 'h4_trend': h4_trend,
            'd1_rsi': d1_rsi, 'h4_rsi': h4_rsi, 'zone': zone, 'range_pct': range_pct,
            'bull_fvgs': bull_fvgs, 'bear_fvgs': bear_fvgs, 'obs': obs,
            'swing_high': swing_high, 'swing_low': swing_low, 'structure': structure,
            'week_high': week_high, 'week_low': week_low
        }
    except:
        return None

indices = ['US30', 'NAS100', 'SPX500', 'UK100', 'JP225', 'AUS200', 'HK50']

for sym in indices:
    r = analyze_index(sym)
    if r:
        print('=' * 60)
        print(f'{r["sym"]} - ICT ANALYSIS')
        print('=' * 60)
        print(f'Price: {r["price"]:.2f}')
        print(f'D1 Trend: {r["d1_trend"]} | H4 Trend: {r["h4_trend"]}')
        print(f'RSI: D1={r["d1_rsi"]:.1f} | H4={r["h4_rsi"]:.1f}')
        print(f'Zone: {r["zone"]} ({r["range_pct"]:.1f}% of weekly range)')
        print(f'Structure: {r["structure"]}')
        print(f'Week Range: {r["week_low"]:.2f} - {r["week_high"]:.2f}')
        print(f'Swing High: {r["swing_high"]:.2f} | Swing Low: {r["swing_low"]:.2f}')
        print(f'\nFVGs: {len(r["bull_fvgs"])} bullish, {len(r["bear_fvgs"])} bearish')
        for fvg in r['bull_fvgs'][:2]:
            print(f'  Bull FVG: {fvg["low"]:.2f} - {fvg["high"]:.2f}')
        for fvg in r['bear_fvgs'][:2]:
            print(f'  Bear FVG: {fvg["low"]:.2f} - {fvg["high"]:.2f}')
        print('\nOrder Blocks:')
        if not r['obs']:
            print('  None detected')
        for ob in r['obs'][:3]:
            print(f'  {ob["type"]} OB: {ob["low"]:.2f} - {ob["high"]:.2f} [{ob["status"]}] ({ob["strength"]:.1f}x)')

        # Bias
        if r['d1_trend'] == 'BULL' and r['zone'] == 'DISCOUNT' and r['d1_rsi'] < 50:
            bias = 'BULLISH - Look for longs'
        elif r['d1_trend'] == 'BEAR' and r['zone'] == 'PREMIUM' and r['d1_rsi'] > 50:
            bias = 'BEARISH - Look for shorts'
        elif r['d1_trend'] == 'BULL' and r['zone'] == 'PREMIUM':
            bias = 'CAUTION - Extended'
        elif r['d1_trend'] == 'BEAR' and r['zone'] == 'DISCOUNT':
            bias = 'WATCH - Potential reversal'
        else:
            bias = 'NEUTRAL'
        print(f'\nBIAS: {bias}')
        print()
