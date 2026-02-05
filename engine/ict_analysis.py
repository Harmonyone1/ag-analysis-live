"""
ICT/SMC Analysis Framework
Must check BEFORE any entry:
1. HTF Bias (Monthly/Weekly)
2. Order Blocks (+OB/-OB)
3. Liquidity targets (buy-side/sell-side)
4. Premium/Discount zones
5. OB mitigation status
6. Entry confirmation
"""
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time
import sys

from config import get_api

def get_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.0001
    return 100 - (100 / (1 + avg_gain/avg_loss))

def get_trend(closes, ma_period):
    if len(closes) < ma_period:
        return 'UNKNOWN', 0
    ma = np.mean(closes[-ma_period:])
    price = closes[-1]
    pct = (price - ma) / ma * 100
    return ('BULL' if price > ma else 'BEAR'), pct

def find_swing_points(highs, lows, lookback=30):
    """Find swing highs and swing lows"""
    swing_highs = []
    swing_lows = []

    h = highs[-lookback:]
    l = lows[-lookback:]

    for i in range(2, len(h)-2):
        # Swing high
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            swing_highs.append({'idx': i - lookback, 'price': h[i]})
        # Swing low
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            swing_lows.append({'idx': i - lookback, 'price': l[i]})

    return swing_highs, swing_lows

def find_order_blocks(opens, closes, highs, lows, lookback=50):
    """
    Find order blocks:
    -OB (Bearish): Last bullish candle before a bearish impulse
    +OB (Bullish): Last bearish candle before a bullish impulse
    """
    bearish_obs = []  # -OB
    bullish_obs = []  # +OB

    o = opens[-lookback:]
    c = closes[-lookback:]
    h = highs[-lookback:]
    l = lows[-lookback:]

    for i in range(2, len(o)-3):
        # -OB: Bullish candle followed by strong bearish move
        if c[i] > o[i]:  # Bullish candle
            # Check if followed by bearish impulse (2+ bearish candles breaking low)
            if c[i+1] < o[i+1] and c[i+2] < o[i+2]:
                if l[i+2] < l[i]:  # Broke below the bullish candle
                    bearish_obs.append({
                        'idx': i - lookback,
                        'high': h[i],
                        'low': l[i],
                        'open': o[i],
                        'close': c[i],
                        'type': '-OB'
                    })

        # +OB: Bearish candle followed by strong bullish move
        if c[i] < o[i]:  # Bearish candle
            # Check if followed by bullish impulse (2+ bullish candles breaking high)
            if c[i+1] > o[i+1] and c[i+2] > o[i+2]:
                if h[i+2] > h[i]:  # Broke above the bearish candle
                    bullish_obs.append({
                        'idx': i - lookback,
                        'high': h[i],
                        'low': l[i],
                        'open': o[i],
                        'close': c[i],
                        'type': '+OB'
                    })

    return bearish_obs, bullish_obs

def find_fvg(opens, closes, highs, lows, lookback=30):
    """Find Fair Value Gaps (imbalances)"""
    fvgs = []

    h = highs[-lookback:]
    l = lows[-lookback:]
    c = closes[-lookback:]

    for i in range(1, len(h)-1):
        # Bullish FVG: Gap between candle 1 high and candle 3 low
        if l[i+1] > h[i-1]:
            fvgs.append({
                'idx': i - lookback,
                'top': l[i+1],
                'bottom': h[i-1],
                'type': 'bullish'
            })
        # Bearish FVG: Gap between candle 1 low and candle 3 high
        if h[i+1] < l[i-1]:
            fvgs.append({
                'idx': i - lookback,
                'top': l[i-1],
                'bottom': h[i+1],
                'type': 'bearish'
            })

    return fvgs

def analyze_instrument(symbol):
    api = get_api()
    instruments = api.get_all_instruments()
    symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

    if symbol not in symbol_to_id:
        print('Symbol %s not found' % symbol)
        return None

    inst_id = symbol_to_id[symbol]

    # Determine price format
    if symbol in ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY']:
        pformat = '%.3f'
    elif symbol in ['BTCUSD', 'ETHUSD', 'XAUUSD', 'US500', 'US30', 'NAS100', 'TSLA', 'AAPL']:
        pformat = '%.2f'
    else:
        pformat = '%.5f'

    print('='*70)
    print('%s - ICT/SMC ANALYSIS' % symbol)
    print('='*70)
    print()

    # Fetch data
    mn = api.get_price_history(inst_id, resolution='1M', lookback_period='365D')
    time.sleep(0.3)
    w1 = api.get_price_history(inst_id, resolution='1W', lookback_period='180D')
    time.sleep(0.3)
    d1 = api.get_price_history(inst_id, resolution='1D', lookback_period='60D')
    time.sleep(0.3)
    h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='30D')
    time.sleep(0.3)
    h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='7D')

    price = h1['c'].values[-1]

    print('CURRENT PRICE: %s' % (pformat % price))
    print()

    # ============ 1. HTF BIAS ============
    print('1. HTF BIAS')
    print('-'*70)

    mn_trend, mn_pct = get_trend(mn['c'].values, 6)
    w1_trend, w1_pct = get_trend(w1['c'].values, 10)
    d1_trend, d1_pct = get_trend(d1['c'].values, 20)

    print('   Monthly: %s (%.2f%% from MA)' % (mn_trend, mn_pct))
    print('   Weekly:  %s (%.2f%% from MA)' % (w1_trend, w1_pct))
    print('   Daily:   %s (%.2f%% from MA)' % (d1_trend, d1_pct))

    if mn_trend == 'BULL':
        htf_bias = 'LONG'
        print('   >>> HTF BIAS: LONG ONLY <<<')
    else:
        htf_bias = 'SHORT'
        print('   >>> HTF BIAS: SHORT ONLY <<<')
    print()

    # ============ 2. ORDER BLOCKS ============
    print('2. ORDER BLOCKS (H4)')
    print('-'*70)

    h4_o = h4['o'].values
    h4_c = h4['c'].values
    h4_h = h4['h'].values
    h4_l = h4['l'].values

    bearish_obs, bullish_obs = find_order_blocks(h4_o, h4_c, h4_h, h4_l, 50)

    # Get most recent OBs
    recent_minus_ob = bearish_obs[-1] if bearish_obs else None
    recent_plus_ob = bullish_obs[-1] if bullish_obs else None

    if recent_minus_ob:
        print('   -OB (Bearish):')
        print('      Zone: %s - %s' % (pformat % recent_minus_ob['low'], pformat % recent_minus_ob['high']))
        if price > recent_minus_ob['high']:
            print('      Status: Price ABOVE -OB (not yet tested)')
            minus_ob_status = 'above'
        elif price > recent_minus_ob['low']:
            print('      Status: Price IN -OB zone (testing)')
            minus_ob_status = 'in_zone'
        else:
            print('      Status: Price BELOW -OB (mitigated - expect continuation DOWN)')
            minus_ob_status = 'mitigated'
    else:
        print('   -OB: None found')
        minus_ob_status = None

    if recent_plus_ob:
        print('   +OB (Bullish):')
        print('      Zone: %s - %s' % (pformat % recent_plus_ob['low'], pformat % recent_plus_ob['high']))
        if price < recent_plus_ob['low']:
            print('      Status: Price BELOW +OB (not yet tested)')
            plus_ob_status = 'below'
        elif price < recent_plus_ob['high']:
            print('      Status: Price IN +OB zone (testing)')
            plus_ob_status = 'in_zone'
        else:
            print('      Status: Price ABOVE +OB (mitigated - expect continuation UP)')
            plus_ob_status = 'mitigated'
    else:
        print('   +OB: None found')
        plus_ob_status = None
    print()

    # ============ 3. LIQUIDITY ============
    print('3. LIQUIDITY TARGETS')
    print('-'*70)

    swing_highs, swing_lows = find_swing_points(h4_h, h4_l, 40)

    # Buy-side liquidity (above swing highs)
    if swing_highs:
        bsl = max(sh['price'] for sh in swing_highs)
        print('   Buy-side (above): %s' % (pformat % bsl))
        bsl_dist = bsl - price
    else:
        bsl = h4_h.max()
        print('   Buy-side (above): %s' % (pformat % bsl))
        bsl_dist = bsl - price

    # Sell-side liquidity (below swing lows)
    if swing_lows:
        ssl = min(sl['price'] for sl in swing_lows)
        print('   Sell-side (below): %s' % (pformat % ssl))
        ssl_dist = price - ssl
    else:
        ssl = h4_l.min()
        print('   Sell-side (below): %s' % (pformat % ssl))
        ssl_dist = price - ssl

    # Which is closer?
    if bsl_dist < ssl_dist:
        print('   >>> Price reaching for BUY-SIDE liquidity <<<')
        liquidity_target = 'buy_side'
    else:
        print('   >>> Price reaching for SELL-SIDE liquidity <<<')
        liquidity_target = 'sell_side'
    print()

    # ============ 4. PREMIUM/DISCOUNT ============
    print('4. PREMIUM/DISCOUNT ZONE')
    print('-'*70)

    range_high = h4_h[-30:].max()
    range_low = h4_l[-30:].min()
    range_mid = (range_high + range_low) / 2
    range_pct = (price - range_low) / (range_high - range_low) * 100

    print('   Range: %s - %s' % (pformat % range_low, pformat % range_high))
    print('   Equilibrium: %s' % (pformat % range_mid))
    print('   Current Position: %.0f%%' % range_pct)

    if range_pct > 70:
        zone = 'PREMIUM'
        print('   >>> PREMIUM ZONE (good for shorts) <<<')
    elif range_pct < 30:
        zone = 'DISCOUNT'
        print('   >>> DISCOUNT ZONE (good for longs) <<<')
    else:
        zone = 'EQUILIBRIUM'
        print('   >>> EQUILIBRIUM (wait for better level) <<<')
    print()

    # ============ 5. RSI CHECK ============
    print('5. RSI STATUS')
    print('-'*70)

    h4_rsi = get_rsi(h4['c'].values)
    h1_rsi = get_rsi(h1['c'].values)

    print('   H4 RSI: %.1f' % h4_rsi, end='')
    if h4_rsi < 30:
        print(' [OVERSOLD]')
    elif h4_rsi > 70:
        print(' [OVERBOUGHT]')
    else:
        print()

    print('   H1 RSI: %.1f' % h1_rsi, end='')
    if h1_rsi < 30:
        print(' [OVERSOLD]')
    elif h1_rsi > 70:
        print(' [OVERBOUGHT]')
    else:
        print()
    print()

    # ============ 6. TRADE DECISION ============
    print('='*70)
    print('TRADE DECISION')
    print('='*70)
    print()

    # Scoring
    long_valid = True
    short_valid = True
    long_reasons = []
    short_reasons = []
    warnings = []

    # Check HTF bias
    if htf_bias == 'LONG':
        short_valid = False
        short_reasons.append('HTF bias is LONG')
        long_reasons.append('[OK] HTF bias LONG')
    else:
        long_valid = False
        long_reasons.append('HTF bias is SHORT')
        short_reasons.append('[OK] HTF bias SHORT')

    # Check OB status for longs
    if htf_bias == 'LONG':
        if minus_ob_status == 'mitigated':
            warnings.append('WARNING: -OB mitigated - expect continuation DOWN')
            long_valid = False
        if plus_ob_status == 'mitigated':
            long_reasons.append('[OK] +OB mitigated - continuation UP expected')
        if zone == 'PREMIUM':
            warnings.append('WARNING: Price in PREMIUM - not ideal for longs')
        if zone == 'DISCOUNT':
            long_reasons.append('[OK] Price in DISCOUNT zone')
        if h4_rsi < 35:
            long_reasons.append('[OK] H4 RSI oversold (%.1f)' % h4_rsi)
        if liquidity_target == 'sell_side':
            warnings.append('WARNING: Price targeting SELL-SIDE liquidity')

    # Check OB status for shorts
    if htf_bias == 'SHORT':
        if plus_ob_status == 'mitigated':
            warnings.append('WARNING: +OB mitigated - expect continuation UP')
            short_valid = False
        if minus_ob_status == 'mitigated':
            short_reasons.append('[OK] -OB mitigated - continuation DOWN expected')
        if zone == 'DISCOUNT':
            warnings.append('WARNING: Price in DISCOUNT - not ideal for shorts')
        if zone == 'PREMIUM':
            short_reasons.append('[OK] Price in PREMIUM zone')
        if h4_rsi > 65:
            short_reasons.append('[OK] H4 RSI overbought (%.1f)' % h4_rsi)
        if liquidity_target == 'buy_side':
            warnings.append('WARNING: Price targeting BUY-SIDE liquidity')

    # Print decision
    if htf_bias == 'LONG':
        print('LONG ANALYSIS:')
        for r in long_reasons:
            print('   %s' % r)
        for w in warnings:
            print('   %s' % w)
        print()
        if long_valid and len(warnings) == 0:
            print('   VERDICT: LONG VALID')
            print('   Entry: Look for +OB or bullish FVG in discount')
        elif long_valid and len(warnings) > 0:
            print('   VERDICT: LONG RISKY - warnings present')
        else:
            print('   VERDICT: NO LONG - wait for better setup')
    else:
        print('SHORT ANALYSIS:')
        for r in short_reasons:
            print('   %s' % r)
        for w in warnings:
            print('   %s' % w)
        print()
        if short_valid and len(warnings) == 0:
            print('   VERDICT: SHORT VALID')
            print('   Entry: Look for -OB or bearish FVG in premium')
        elif short_valid and len(warnings) > 0:
            print('   VERDICT: SHORT RISKY - warnings present')
        else:
            print('   VERDICT: NO SHORT - wait for better setup')

    print()
    print('='*70)
    print('PRE-ENTRY CHECKLIST')
    print('='*70)
    print()
    print('Before entering, confirm:')
    print('[ ] 1. HTF bias matches trade direction')
    print('[ ] 2. Not entering after OB mitigation (wrong side)')
    print('[ ] 3. Price in correct zone (discount for longs, premium for shorts)')
    print('[ ] 4. Clear OB or FVG for entry')
    print('[ ] 5. Liquidity target supports trade direction')
    print('[ ] 6. RSI confirms (oversold for longs, overbought for shorts)')
    print('[ ] 7. Stop loss below/above structure (not arbitrary)')

    return {
        'symbol': symbol,
        'price': price,
        'htf_bias': htf_bias,
        'zone': zone,
        'liquidity_target': liquidity_target,
        'h4_rsi': h4_rsi,
        'minus_ob': recent_minus_ob,
        'plus_ob': recent_plus_ob,
        'minus_ob_status': minus_ob_status,
        'plus_ob_status': plus_ob_status
    }

if __name__ == '__main__':
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = 'GBPNZD'

    analyze_instrument(symbol)
