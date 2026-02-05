#!/usr/bin/env python
"""Continuous monitoring and trading for ETHUSD and BTCUSD"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import time
import sys
import os

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
from datetime import datetime, timezone
from decimal import Decimal
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('../.env', override=True)

from tradelocker import TLAPI

# Configuration
SYMBOLS = ['BTCUSD', 'ETHUSD']
CHECK_INTERVAL = 60  # seconds between checks
BASE_LOT_SIZE = 0.10          # Standard setup
CONFIDENT_LOT_SIZE = 0.20     # 75%+ confidence
HIGH_CONFIDENCE_LOT_SIZE = 0.50  # 85%+ confidence
MAX_LOT_SIZE = 1.00           # Maximum when extremely confident

def connect():
    """Connect to TradeLocker using env credentials."""
    api = TLAPI(
        environment=os.getenv('TL_ENVIRONMENT'),
        username=os.getenv('TL_EMAIL'),
        password=os.getenv('TL_PASSWORD'),
        server=os.getenv('TL_SERVER'),
        account_id=int(os.getenv('TL_ACC_NUM')),
    )
    return api

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_trend(prices):
    if len(prices) < 50:
        return 'NEUTRAL', 0, 0
    ma20 = np.mean(prices[-20:])
    ma50 = np.mean(prices[-50:])
    current = prices[-1]
    if current > ma20 > ma50:
        return 'BULL', ma20, ma50
    elif current < ma20 < ma50:
        return 'BEAR', ma20, ma50
    return 'NEUTRAL', ma20, ma50

def get_swing_points(highs, lows, lookback=5):
    """Detect swing highs and swing lows."""
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(highs) - lookback):
        # Swing high: higher than lookback candles on both sides
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            swing_highs.append((i, highs[i]))
        # Swing low: lower than lookback candles on both sides
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows

def get_market_structure(highs, lows):
    """Determine market structure from swing points.
    Returns: 'BULLISH' (HH+HL), 'BEARISH' (LL+LH), or 'NEUTRAL'
    """
    swing_highs, swing_lows = get_swing_points(highs, lows)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 'NEUTRAL', None, None

    # Get last 2 swing highs and lows
    last_highs = [sh[1] for sh in swing_highs[-2:]]
    last_lows = [sl[1] for sl in swing_lows[-2:]]

    # Check for Higher Highs and Higher Lows (Bullish)
    hh = last_highs[-1] > last_highs[-2]  # Higher High
    hl = last_lows[-1] > last_lows[-2]    # Higher Low

    # Check for Lower Lows and Lower Highs (Bearish)
    ll = last_lows[-1] < last_lows[-2]    # Lower Low
    lh = last_highs[-1] < last_highs[-2]  # Lower High

    if hh and hl:
        return 'BULLISH', last_highs[-1], last_lows[-1]
    elif ll and lh:
        return 'BEARISH', last_highs[-1], last_lows[-1]
    else:
        return 'NEUTRAL', last_highs[-1] if last_highs else None, last_lows[-1] if last_lows else None


def detect_fair_value_gaps(opens, highs, lows, closes, lookback=20):
    """Detect Fair Value Gaps (FVGs) - imbalances in price.
    Bullish FVG: Gap up where candle 1 high < candle 3 low
    Bearish FVG: Gap down where candle 1 low > candle 3 high
    Returns list of unfilled FVGs with their zones.
    """
    fvgs = []
    current_price = closes[-1]

    for i in range(2, min(lookback, len(highs))):
        idx = len(highs) - i - 1
        if idx < 2:
            break

        # Candle indices: idx-2 (first), idx-1 (middle), idx (third)
        c1_high = highs[idx - 2]
        c1_low = lows[idx - 2]
        c3_high = highs[idx]
        c3_low = lows[idx]

        # Bullish FVG: Gap between candle 1 high and candle 3 low
        if c3_low > c1_high:
            fvg_top = c3_low
            fvg_bottom = c1_high
            # Check if unfilled (price hasn't returned to fill it)
            if current_price > fvg_top:  # Above the gap
                fvgs.append({
                    'type': 'BULLISH',
                    'top': fvg_top,
                    'bottom': fvg_bottom,
                    'filled': False,
                    'age': i
                })

        # Bearish FVG: Gap between candle 1 low and candle 3 high
        elif c1_low > c3_high:
            fvg_top = c1_low
            fvg_bottom = c3_high
            # Check if unfilled
            if current_price < fvg_bottom:  # Below the gap
                fvgs.append({
                    'type': 'BEARISH',
                    'top': fvg_top,
                    'bottom': fvg_bottom,
                    'filled': False,
                    'age': i
                })

    return fvgs


def detect_order_blocks(opens, highs, lows, closes, lookback=50):
    """Detect Order Blocks - last opposing candle before a strong move.
    Bullish OB: Last bearish candle before strong bullish move
    Bearish OB: Last bullish candle before strong bearish move

    Returns OBs with status: ACTIVE (price outside), IN_ZONE (price inside), BROKEN (invalidated)
    """
    order_blocks = []
    current_price = closes[-1]

    for i in range(2, min(lookback, len(closes) - 1)):
        idx = len(closes) - i - 1
        if idx < 1:
            break

        # Check for bullish order block (last red candle before big green move)
        is_bearish_candle = closes[idx] < opens[idx]
        next_is_bullish = closes[idx + 1] > opens[idx + 1]

        if is_bearish_candle and next_is_bullish:
            ob_range = highs[idx] - lows[idx]
            move_size = closes[idx + 1] - opens[idx + 1]

            # Significant move (> 1.2x the OB candle range)
            if ob_range > 0 and move_size > ob_range * 1.2:
                ob_high = highs[idx]
                ob_low = lows[idx]

                # Determine OB status
                if current_price > ob_high:
                    status = 'ACTIVE'  # Price above OB, OB is support
                elif current_price >= ob_low and current_price <= ob_high:
                    status = 'IN_ZONE'  # Price inside OB
                else:
                    status = 'BROKEN'  # Price below OB, invalidated

                if status != 'BROKEN':
                    order_blocks.append({
                        'type': 'BULLISH',
                        'high': ob_high,
                        'low': ob_low,
                        'mid': (ob_high + ob_low) / 2,
                        'age': i,
                        'status': status,
                        'strength': move_size / ob_range  # How strong was the move
                    })

        # Check for bearish order block (last green candle before big red move)
        is_bullish_candle = closes[idx] > opens[idx]
        next_is_bearish = closes[idx + 1] < opens[idx + 1]

        if is_bullish_candle and next_is_bearish:
            ob_range = highs[idx] - lows[idx]
            move_size = opens[idx + 1] - closes[idx + 1]

            # Significant move (> 1.2x the OB candle range)
            if ob_range > 0 and move_size > ob_range * 1.2:
                ob_high = highs[idx]
                ob_low = lows[idx]

                # Determine OB status
                if current_price < ob_low:
                    status = 'ACTIVE'  # Price below OB, OB is resistance
                elif current_price >= ob_low and current_price <= ob_high:
                    status = 'IN_ZONE'  # Price inside OB - CRITICAL!
                else:
                    status = 'BROKEN'  # Price above OB, invalidated

                if status != 'BROKEN':
                    order_blocks.append({
                        'type': 'BEARISH',
                        'high': ob_high,
                        'low': ob_low,
                        'mid': (ob_high + ob_low) / 2,
                        'age': i,
                        'status': status,
                        'strength': move_size / ob_range
                    })

    # Sort by strength and return most significant OBs
    bullish_obs = sorted([ob for ob in order_blocks if ob['type'] == 'BULLISH'],
                         key=lambda x: x['strength'], reverse=True)[:3]
    bearish_obs = sorted([ob for ob in order_blocks if ob['type'] == 'BEARISH'],
                         key=lambda x: x['strength'], reverse=True)[:3]

    return bullish_obs + bearish_obs


def analyze_volume(volumes, lookback=20):
    """Analyze volume patterns.
    Returns volume analysis with relative strength and trend.
    """
    if len(volumes) < lookback:
        return {'relative': 1.0, 'trend': 'NEUTRAL', 'spike': False}

    recent_vol = volumes[-1]
    avg_vol = np.mean(volumes[-lookback:])
    prev_avg = np.mean(volumes[-lookback*2:-lookback]) if len(volumes) >= lookback*2 else avg_vol

    relative = recent_vol / avg_vol if avg_vol > 0 else 1.0

    # Volume trend
    if avg_vol > prev_avg * 1.2:
        vol_trend = 'INCREASING'
    elif avg_vol < prev_avg * 0.8:
        vol_trend = 'DECREASING'
    else:
        vol_trend = 'NEUTRAL'

    # Volume spike detection
    spike = relative > 2.0

    return {
        'current': recent_vol,
        'average': avg_vol,
        'relative': relative,
        'trend': vol_trend,
        'spike': spike
    }


def detect_liquidity_sweeps(highs, lows, closes, lookback=50):
    """Detect liquidity sweeps - price taking out equal highs/lows then reversing.
    These indicate smart money grabbing liquidity before the real move.
    """
    sweeps = []
    current_price = closes[-1]

    # Find equal highs (potential buy-side liquidity)
    equal_highs = []
    for i in range(lookback - 5, 5, -1):
        idx = len(highs) - i
        if idx < 5:
            continue
        # Look for highs within 0.1% of each other
        for j in range(i - 5, max(0, i - 20), -1):
            jdx = len(highs) - j
            if abs(highs[idx] - highs[jdx]) / highs[idx] < 0.001:
                equal_highs.append({
                    'level': max(highs[idx], highs[jdx]),
                    'idx1': idx,
                    'idx2': jdx
                })
                break

    # Find equal lows (potential sell-side liquidity)
    equal_lows = []
    for i in range(lookback - 5, 5, -1):
        idx = len(lows) - i
        if idx < 5:
            continue
        for j in range(i - 5, max(0, i - 20), -1):
            jdx = len(lows) - j
            if abs(lows[idx] - lows[jdx]) / lows[idx] < 0.001:
                equal_lows.append({
                    'level': min(lows[idx], lows[jdx]),
                    'idx1': idx,
                    'idx2': jdx
                })
                break

    # Check for recent sweeps (price broke level then reversed)
    for eh in equal_highs[-5:]:  # Check last 5 equal highs
        level = eh['level']
        # Check if price swept above then came back down
        recent_highs = highs[-10:]
        if any(h > level for h in recent_highs) and current_price < level:
            sweeps.append({
                'type': 'BUYSIDE_SWEPT',
                'level': level,
                'direction': 'BEARISH'  # Swept buy-side = bearish
            })

    for el in equal_lows[-5:]:
        level = el['level']
        recent_lows = lows[-10:]
        if any(l < level for l in recent_lows) and current_price > level:
            sweeps.append({
                'type': 'SELLSIDE_SWEPT',
                'level': level,
                'direction': 'BULLISH'  # Swept sell-side = bullish
            })

    # Also return pending liquidity levels (not yet swept)
    pending_buyside = [eh['level'] for eh in equal_highs if eh['level'] > current_price][-3:]
    pending_sellside = [el['level'] for el in equal_lows if el['level'] < current_price][-3:]

    return {
        'sweeps': sweeps,
        'pending_buyside': pending_buyside,  # Equal highs above price
        'pending_sellside': pending_sellside  # Equal lows below price
    }


def get_trading_session():
    """Determine current trading session based on UTC time.
    Returns session name and characteristics.
    """
    now = datetime.now(timezone.utc)
    hour = now.hour

    # Session times (UTC)
    # Asian: 00:00 - 08:00 UTC (Tokyo/Sydney)
    # London: 08:00 - 16:00 UTC
    # New York: 13:00 - 21:00 UTC
    # Note: London/NY overlap is 13:00-16:00 UTC

    if 0 <= hour < 8:
        session = 'ASIAN'
        volatility = 'LOW'
        description = 'Range-bound, accumulation'
    elif 8 <= hour < 13:
        session = 'LONDON'
        volatility = 'HIGH'
        description = 'Breakouts, trend starts'
    elif 13 <= hour < 16:
        session = 'LONDON_NY_OVERLAP'
        volatility = 'HIGHEST'
        description = 'Peak liquidity, big moves'
    elif 16 <= hour < 21:
        session = 'NEW_YORK'
        volatility = 'HIGH'
        description = 'Continuation or reversal'
    else:
        session = 'OFF_HOURS'
        volatility = 'LOW'
        description = 'Low liquidity'

    return {
        'session': session,
        'volatility': volatility,
        'description': description,
        'utc_hour': hour
    }


def calculate_correlation(prices1, prices2, lookback=20):
    """Calculate correlation between two price series.
    Returns correlation coefficient and divergence status.
    """
    if len(prices1) < lookback or len(prices2) < lookback:
        return {'correlation': 0, 'diverging': False, 'status': 'UNKNOWN'}

    # Get returns
    returns1 = np.diff(prices1[-lookback:]) / prices1[-lookback:-1]
    returns2 = np.diff(prices2[-lookback:]) / prices2[-lookback:-1]

    # Calculate correlation
    if len(returns1) > 0 and len(returns2) > 0:
        correlation = np.corrcoef(returns1, returns2)[0, 1]
    else:
        correlation = 0

    # Check for divergence (one up, one down recently)
    recent1 = (prices1[-1] - prices1[-5]) / prices1[-5] if len(prices1) >= 5 else 0
    recent2 = (prices2[-1] - prices2[-5]) / prices2[-5] if len(prices2) >= 5 else 0

    diverging = (recent1 > 0.01 and recent2 < -0.01) or (recent1 < -0.01 and recent2 > 0.01)

    if correlation > 0.7:
        status = 'STRONG_POSITIVE'
    elif correlation > 0.3:
        status = 'MODERATE_POSITIVE'
    elif correlation < -0.3:
        status = 'NEGATIVE'
    else:
        status = 'WEAK'

    return {
        'correlation': correlation,
        'diverging': diverging,
        'btc_change': recent1 * 100,
        'eth_change': recent2 * 100,
        'status': status
    }


def analyze_symbol(api, inst_id, symbol):
    """Full multi-timeframe analysis."""
    current_price = api.get_latest_asking_price(inst_id)

    # Get timeframe data
    d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='6M')
    h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='1M')
    h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='2W')
    m15 = api.get_price_history(inst_id, resolution='15m', start_timestamp=0, end_timestamp=0, lookback_period='1W')

    if d1 is None or h4 is None or len(d1) < 50 or len(h4) < 50:
        return None

    # Calculate indicators for each timeframe
    d1_trend, d1_ma20, d1_ma50 = get_trend(d1['c'].values)
    d1_rsi = calculate_rsi(d1['c'].values)

    h4_trend, h4_ma20, h4_ma50 = get_trend(h4['c'].values)
    h4_rsi = calculate_rsi(h4['c'].values)

    h1_trend, h1_ma20, h1_ma50 = get_trend(h1['c'].values) if h1 is not None and len(h1) > 50 else ('NEUTRAL', 0, 0)
    h1_rsi = calculate_rsi(h1['c'].values) if h1 is not None else 50

    m15_trend, m15_ma20, m15_ma50 = get_trend(m15['c'].values) if m15 is not None and len(m15) > 50 else ('NEUTRAL', 0, 0)
    m15_rsi = calculate_rsi(m15['c'].values) if m15 is not None else 50

    # Price zone
    d1_high = max(d1['h'].values)
    d1_low = min(d1['l'].values)
    range_pct = ((current_price - d1_low) / (d1_high - d1_low)) * 100 if d1_high != d1_low else 50

    if range_pct < 30:
        zone = 'DISCOUNT'
    elif range_pct > 70:
        zone = 'PREMIUM'
    else:
        zone = 'EQUILIBRIUM'

    # Recent momentum - count green/red candles
    h4_closes = h4['c'].values[-10:]
    h4_opens = h4['o'].values[-10:]
    green_candles = sum(1 for i in range(len(h4_closes)) if h4_closes[i] > h4_opens[i])

    # ATR for stop loss calculation
    h4_atr = np.mean([h4['h'].values[i] - h4['l'].values[i] for i in range(-14, 0)])

    # Market structure analysis on H4
    h4_structure, h4_last_high, h4_last_low = get_market_structure(h4['h'].values, h4['l'].values)

    # NEW: Fair Value Gaps on H4
    h4_fvgs = detect_fair_value_gaps(h4['o'].values, h4['h'].values, h4['l'].values, h4['c'].values)
    bullish_fvgs = [f for f in h4_fvgs if f['type'] == 'BULLISH']
    bearish_fvgs = [f for f in h4_fvgs if f['type'] == 'BEARISH']

    # NEW: Order Blocks on H4
    h4_order_blocks = detect_order_blocks(h4['o'].values, h4['h'].values, h4['l'].values, h4['c'].values)
    bullish_obs = [ob for ob in h4_order_blocks if ob['type'] == 'BULLISH']
    bearish_obs = [ob for ob in h4_order_blocks if ob['type'] == 'BEARISH']

    # NEW: Volume analysis (if available)
    volume_analysis = {'relative': 1.0, 'trend': 'N/A', 'spike': False}
    if 'v' in h4.columns:
        volume_analysis = analyze_volume(h4['v'].values)

    # NEW: Liquidity sweeps
    liquidity = detect_liquidity_sweeps(h4['h'].values, h4['l'].values, h4['c'].values)

    # Store raw H4 data for correlation
    h4_closes = h4['c'].values

    return {
        'symbol': symbol,
        'current_price': current_price,
        'd1_trend': d1_trend,
        'd1_rsi': d1_rsi,
        'd1_ma20': d1_ma20,
        'd1_ma50': d1_ma50,
        'h4_trend': h4_trend,
        'h4_rsi': h4_rsi,
        'h4_ma20': h4_ma20,
        'h4_ma50': h4_ma50,
        'h4_structure': h4_structure,
        'h4_last_high': h4_last_high,
        'h4_last_low': h4_last_low,
        'h1_trend': h1_trend,
        'h1_rsi': h1_rsi,
        'm15_trend': m15_trend,
        'm15_rsi': m15_rsi,
        'zone': zone,
        'range_pct': range_pct,
        'd1_high': d1_high,
        'd1_low': d1_low,
        'h4_atr': h4_atr,
        'green_candles_10': green_candles,
        # NEW indicators
        'bullish_fvgs': bullish_fvgs,
        'bearish_fvgs': bearish_fvgs,
        'bullish_obs': bullish_obs,
        'bearish_obs': bearish_obs,
        'volume': volume_analysis,
        'liquidity': liquidity,
        'h4_closes': h4_closes,
    }

def evaluate_setup(analysis):
    """Evaluate if there's a valid trading setup and return confidence level."""
    if analysis is None:
        return None, 0, []

    reasons = []
    confidence = 0
    direction = None

    # HTF Alignment check
    htf_bull = analysis['d1_trend'] == 'BULL' and analysis['h4_trend'] == 'BULL'
    htf_bear = analysis['d1_trend'] == 'BEAR' and analysis['h4_trend'] == 'BEAR'

    # Standard ICT+Quant criteria
    # LONG: HTF bull + discount + RSI < 40
    # SHORT: HTF bear + premium + RSI > 60

    valid_long = htf_bull and analysis['zone'] == 'DISCOUNT' and analysis['h4_rsi'] < 40
    valid_short = htf_bear and analysis['zone'] == 'PREMIUM' and analysis['h4_rsi'] > 60

    if valid_long:
        direction = 'LONG'
        confidence = 70
        reasons.append('HTF BULL aligned')
        reasons.append('Price in DISCOUNT zone (%.1f%%)' % analysis['range_pct'])
        reasons.append('H4 RSI oversold (%.1f)' % analysis['h4_rsi'])

        # Bonus confidence factors
        if analysis['h1_trend'] == 'BULL':
            confidence += 10
            reasons.append('H1 trend confirming')
        if analysis['m15_trend'] == 'BULL':
            confidence += 5
            reasons.append('M15 momentum aligned')
        if analysis['green_candles_10'] >= 6:
            confidence += 5
            reasons.append('Strong recent momentum (%d/10 green)' % analysis['green_candles_10'])
        if analysis['d1_rsi'] < 35:
            confidence += 10
            reasons.append('D1 RSI deeply oversold')

    elif valid_short:
        direction = 'SHORT'
        confidence = 70
        reasons.append('HTF BEAR aligned')
        reasons.append('Price in PREMIUM zone (%.1f%%)' % analysis['range_pct'])
        reasons.append('H4 RSI overbought (%.1f)' % analysis['h4_rsi'])

        # Bonus confidence factors
        if analysis['h1_trend'] == 'BEAR':
            confidence += 10
            reasons.append('H1 trend confirming')
        if analysis['m15_trend'] == 'BEAR':
            confidence += 5
            reasons.append('M15 momentum aligned')
        if analysis['green_candles_10'] <= 4:
            confidence += 5
            reasons.append('Weak recent momentum (%d/10 green)' % analysis['green_candles_10'])
        if analysis['d1_rsi'] > 65:
            confidence += 10
            reasons.append('D1 RSI overbought')

    # Alternative: Strong momentum setup (less strict)
    # When LTF strongly aligned and showing momentum, even without perfect HTF
    if direction is None:
        # Check for momentum breakout setup
        if (analysis['h4_trend'] == 'BULL' and analysis['h1_trend'] == 'BULL' and
            analysis['m15_trend'] == 'BULL' and analysis['zone'] == 'DISCOUNT' and
            analysis['green_candles_10'] >= 7):
            direction = 'LONG'
            confidence = 55
            reasons.append('LTF momentum breakout')
            reasons.append('H4+H1+M15 all BULL')
            reasons.append('Strong momentum (%d/10 green)' % analysis['green_candles_10'])
            reasons.append('Price in discount (%.1f%%)' % analysis['range_pct'])

        elif (analysis['h4_trend'] == 'BEAR' and analysis['h1_trend'] == 'BEAR' and
              analysis['m15_trend'] == 'BEAR' and analysis['zone'] == 'PREMIUM' and
              analysis['green_candles_10'] <= 3):
            direction = 'SHORT'
            confidence = 55
            reasons.append('LTF momentum breakdown')
            reasons.append('H4+H1+M15 all BEAR')
            reasons.append('Weak momentum (%d/10 green)' % analysis['green_candles_10'])
            reasons.append('Price in premium (%.1f%%)' % analysis['range_pct'])

    # Transition momentum setup - H4 transitioning, LTF aligned
    if direction is None:
        # Long: H4 neutral but transitioning up, H1+M15 bull, deep discount, strong momentum
        if (analysis['h4_trend'] == 'NEUTRAL' and analysis['h1_trend'] == 'BULL' and
            analysis['m15_trend'] == 'BULL' and analysis['zone'] == 'DISCOUNT' and
            analysis['range_pct'] < 25 and  # Deep discount
            analysis['green_candles_10'] >= 7 and
            analysis['current_price'] > analysis['h4_ma20']):  # Price above H4 MA20
            direction = 'LONG'
            confidence = 60
            reasons.append('Transition momentum setup')
            reasons.append('H4 transitioning (price > MA20)')
            reasons.append('H1+M15 BULL aligned')
            reasons.append('Deep discount (%.1f%%)' % analysis['range_pct'])
            reasons.append('Strong momentum (%d/10 green)' % analysis['green_candles_10'])

        # Short: H4 neutral but transitioning down, H1+M15 bear, premium, weak momentum
        elif (analysis['h4_trend'] == 'NEUTRAL' and analysis['h1_trend'] == 'BEAR' and
              analysis['m15_trend'] == 'BEAR' and analysis['zone'] == 'PREMIUM' and
              analysis['range_pct'] > 75 and
              analysis['green_candles_10'] <= 3 and
              analysis['current_price'] < analysis['h4_ma20']):
            direction = 'SHORT'
            confidence = 60
            reasons.append('Transition momentum setup')
            reasons.append('H4 transitioning (price < MA20)')
            reasons.append('H1+M15 BEAR aligned')
            reasons.append('High premium (%.1f%%)' % analysis['range_pct'])
            reasons.append('Weak momentum (%d/10 green)' % analysis['green_candles_10'])

    # Bearish continuation SHORT - confirmed breakdown (works in any zone)
    # Requires: Market structure BEARISH (LL+LH) + All TF BEAR + secondary confirmations
    if direction is None:
        # PRIMARY: H4 Market Structure must be BEARISH (LL + LH)
        structure_bearish = analysis.get('h4_structure') == 'BEARISH'

        # PRIMARY: All timeframes BEAR aligned
        all_tf_bear = (analysis['d1_trend'] == 'BEAR' and
                       analysis['h4_trend'] == 'BEAR' and
                       analysis['h1_trend'] == 'BEAR' and
                       analysis['m15_trend'] == 'BEAR')

        # SECONDARY confirmations (need at least 2 of 3)
        below_mas = (analysis['current_price'] < analysis['h4_ma20'] and
                     analysis['current_price'] < analysis['h4_ma50'])
        weak_momentum = analysis['green_candles_10'] <= 4
        rsi_weak = analysis['h4_rsi'] < 45

        secondary_count = sum([below_mas, weak_momentum, rsi_weak])

        # Require BOTH primary + at least 2 secondary confirmations
        if structure_bearish and all_tf_bear and secondary_count >= 2:
            direction = 'SHORT'
            confidence = 70
            reasons.append('CONFIRMED Bearish continuation')
            reasons.append('H4 Structure: BEARISH (LL+LH)')
            reasons.append('D1+H4+H1+M15 all BEAR')

            if below_mas:
                reasons.append('Price below H4 MA20 & MA50')
                confidence += 5
            if weak_momentum:
                reasons.append('Weak momentum (%d/10 green)' % analysis['green_candles_10'])
                confidence += 5
            if rsi_weak:
                reasons.append('H4 RSI weak (%.1f)' % analysis['h4_rsi'])
                confidence += 5

            # Reduce confidence if deeply oversold
            if analysis['range_pct'] < 15:
                confidence -= 10
                reasons.append('Caution: Very deep discount (%.1f%%)' % analysis['range_pct'])
            elif analysis['range_pct'] < 25:
                confidence -= 5
                reasons.append('Note: Deep discount (%.1f%%)' % analysis['range_pct'])

    return direction, confidence, reasons

def get_position_for_symbol(api, symbol):
    """Check if we have an open position for this symbol."""
    positions = api.get_all_positions()
    if positions is None or len(positions) == 0:
        return None

    instruments = api.get_all_instruments()
    name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))
    inst_id = name_to_id.get(symbol)

    for _, p in positions.iterrows():
        if p['tradableInstrumentId'] == inst_id:
            return {
                'id': p['id'],
                'side': p['side'],
                'qty': p['qty'],
                'avgPrice': p['avgPrice'],
                'unrealizedPnL': p.get('unrealizedPnL', 0),
            }
    return None

def execute_trade(api, inst_id, symbol, direction, lot_size, analysis):
    """Execute a trade with proper SL/TP."""
    current_price = analysis['current_price']
    atr = analysis['h4_atr']

    # Use 2x ATR for SL, 3x ATR for TP (1.5 R:R)
    sl_distance = atr * 2
    tp_distance = atr * 3

    if direction == 'LONG':
        side = 'buy'
        sl_price = current_price - sl_distance
        tp_price = current_price + tp_distance
    else:
        side = 'sell'
        sl_price = current_price + sl_distance
        tp_price = current_price - tp_distance

    try:
        order = api.create_order(
            inst_id,
            quantity=lot_size,
            side=side,
            type_='market',
            stop_loss=round(sl_price, 2),
            stop_loss_type='absolute',
            take_profit=round(tp_price, 2),
            take_profit_type='absolute',
        )
        print(f'  >>> SL: ${sl_price:.2f} | TP: ${tp_price:.2f}')
        return order
    except Exception as e:
        print(f'  ERROR executing trade: {e}')
        return None

def add_to_position(api, inst_id, symbol, direction, lot_size, analysis):
    """Add to a winning position (stacking)."""
    # Same as execute but for adding
    return execute_trade(api, inst_id, symbol, direction, lot_size, analysis)

def main():
    print('=' * 70)
    print('CRYPTO MONITOR - BTCUSD & ETHUSD')
    print('Started:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('=' * 70)
    print()

    api = connect()

    instruments = api.get_all_instruments()
    name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

    # Get account info
    state = api.get_account_state()
    print('Account Balance: $%.2f' % state['balance'])
    print('Monitoring: %s' % ', '.join(SYMBOLS))
    print('Check Interval: %d seconds' % CHECK_INTERVAL)
    print()
    print('-' * 70)

    scan_count = 0

    # Store analysis data for correlation
    analysis_cache = {}

    while True:
        scan_count += 1
        now = datetime.now().strftime('%H:%M:%S')

        # Get current session info
        session_info = get_trading_session()

        print(f'\n[{now}] Scan #{scan_count} | Session: {session_info["session"]} ({session_info["volatility"]} volatility)')
        print('-' * 40)

        for symbol in SYMBOLS:
            inst_id = name_to_id.get(symbol)
            if not inst_id:
                print(f'{symbol}: Not found')
                continue

            # Analyze
            analysis = analyze_symbol(api, inst_id, symbol)
            if analysis is None:
                print(f'{symbol}: Analysis failed')
                continue

            # Cache analysis for correlation
            analysis_cache[symbol] = analysis

            # Check for existing position
            position = get_position_for_symbol(api, symbol)

            # Evaluate setup
            direction, confidence, reasons = evaluate_setup(analysis)

            # Print status
            print(f'\n{symbol}: ${analysis["current_price"]:.2f}')
            print(f'  Zone: {analysis["zone"]} ({analysis["range_pct"]:.1f}%)')
            print(f'  Trends: D1={analysis["d1_trend"]} H4={analysis["h4_trend"]} H1={analysis["h1_trend"]} M15={analysis["m15_trend"]}')
            print(f'  H4 Structure: {analysis["h4_structure"]}')
            print(f'  RSI: D1={analysis["d1_rsi"]:.1f} H4={analysis["h4_rsi"]:.1f}')
            print(f'  Momentum: {analysis["green_candles_10"]}/10 green H4 candles')

            # NEW: Display Fair Value Gaps
            bullish_fvgs = analysis.get('bullish_fvgs', [])
            bearish_fvgs = analysis.get('bearish_fvgs', [])
            if bullish_fvgs or bearish_fvgs:
                fvg_str = f'  FVGs: {len(bullish_fvgs)} bullish, {len(bearish_fvgs)} bearish'
                if bullish_fvgs:
                    nearest_bull = bullish_fvgs[0]
                    fvg_str += f' | Nearest bull: ${nearest_bull["bottom"]:.0f}-${nearest_bull["top"]:.0f}'
                if bearish_fvgs:
                    nearest_bear = bearish_fvgs[0]
                    fvg_str += f' | Nearest bear: ${nearest_bear["bottom"]:.0f}-${nearest_bear["top"]:.0f}'
                print(fvg_str)

            # NEW: Display Order Blocks with status
            bullish_obs = analysis.get('bullish_obs', [])
            bearish_obs = analysis.get('bearish_obs', [])

            # Check for critical IN_ZONE status first
            in_zone_obs = [ob for ob in bullish_obs + bearish_obs if ob.get('status') == 'IN_ZONE']
            if in_zone_obs:
                for ob in in_zone_obs:
                    print(f'  >>> PRICE IN {ob["type"]} OB ZONE: ${ob["low"]:.0f}-${ob["high"]:.0f} (strength: {ob["strength"]:.1f}x)')

            # Show other OBs
            if bullish_obs or bearish_obs:
                for ob in bullish_obs:
                    if ob.get('status') != 'IN_ZONE':
                        print(f'  Bullish OB: ${ob["low"]:.0f}-${ob["high"]:.0f} [{ob["status"]}] (strength: {ob["strength"]:.1f}x)')
                for ob in bearish_obs:
                    if ob.get('status') != 'IN_ZONE':
                        print(f'  Bearish OB: ${ob["low"]:.0f}-${ob["high"]:.0f} [{ob["status"]}] (strength: {ob["strength"]:.1f}x)')

            # NEW: Display Volume if available
            vol = analysis.get('volume', {})
            if vol.get('trend') != 'N/A':
                vol_str = f'  Volume: {vol["relative"]:.1f}x avg'
                if vol.get('spike'):
                    vol_str += ' [SPIKE!]'
                vol_str += f' | Trend: {vol["trend"]}'
                print(vol_str)

            # NEW: Display Liquidity levels
            liq = analysis.get('liquidity', {})
            sweeps = liq.get('sweeps', [])
            pending_buy = liq.get('pending_buyside', [])
            pending_sell = liq.get('pending_sellside', [])
            if sweeps:
                for sweep in sweeps[-2:]:  # Show last 2 sweeps
                    print(f'  LIQUIDITY SWEEP: {sweep["type"]} @ ${sweep["level"]:.0f} -> {sweep["direction"]}')
            if pending_buy or pending_sell:
                liq_str = '  Liquidity:'
                if pending_buy:
                    liq_str += f' Buy-side @ ${pending_buy[0]:.0f}'
                if pending_sell:
                    liq_str += f' | Sell-side @ ${pending_sell[0]:.0f}'
                print(liq_str)

            if position:
                pnl = position.get('unrealizedPnL', 0)
                print(f'  POSITION: {position["side"]} {position["qty"]} lots @ ${position["avgPrice"]:.2f} | P/L: ${pnl:.2f}')

                # Check if we should add to winning position
                if pnl > 0.5 and confidence >= 60:  # In profit by at least $0.50
                    # Position is winning and setup still valid
                    if (direction == 'LONG' and position['side'] == 'buy') or \
                       (direction == 'SHORT' and position['side'] == 'sell'):
                        print(f'  >>> STACKING OPPORTUNITY (Confidence: {confidence}%)')
                        for r in reasons:
                            print(f'      - {r}')

                        # Stack size based on confidence and current profit
                        if pnl > 5.0 and confidence >= 80:
                            stack_size = CONFIDENT_LOT_SIZE  # 0.20 lots
                            print(f'  >>> AGGRESSIVE STACK - Adding {stack_size} lots (P/L: ${pnl:.2f})')
                        elif pnl > 2.0 and confidence >= 70:
                            stack_size = BASE_LOT_SIZE  # 0.10 lots
                            print(f'  >>> STACK - Adding {stack_size} lots (P/L: ${pnl:.2f})')
                        else:
                            stack_size = BASE_LOT_SIZE * 0.5  # 0.05 lots
                            print(f'  >>> SMALL STACK - Adding {stack_size} lots (P/L: ${pnl:.2f})')

                        order = add_to_position(api, inst_id, symbol, direction, stack_size, analysis)
                        if order:
                            print(f'  >>> STACKED successfully')
            else:
                # No position - check for new entry
                if direction and confidence >= 60:
                    print(f'  >>> SETUP DETECTED: {direction} (Confidence: {confidence}%)')
                    for r in reasons:
                        print(f'      - {r}')

                    # Determine lot size based on confidence
                    if confidence >= 90:
                        lot_size = MAX_LOT_SIZE
                        print(f'  >>> MAXIMUM CONFIDENCE - Using {lot_size} lots')
                    elif confidence >= 85:
                        lot_size = HIGH_CONFIDENCE_LOT_SIZE
                        print(f'  >>> HIGH CONFIDENCE - Using {lot_size} lots')
                    elif confidence >= 75:
                        lot_size = CONFIDENT_LOT_SIZE
                        print(f'  >>> CONFIDENT - Using {lot_size} lots')
                    else:
                        lot_size = BASE_LOT_SIZE
                        print(f'  >>> STANDARD - Using {lot_size} lots')

                    # Execute
                    print(f'  >>> EXECUTING {direction} {lot_size} lots...')
                    order = execute_trade(api, inst_id, symbol, direction, lot_size, analysis)
                    if order:
                        print(f'  >>> ORDER PLACED: {order}')

                elif direction:
                    print(f'  Setup: {direction} but confidence too low ({confidence}%)')
                else:
                    print(f'  No valid setup')

        # Calculate and display BTC/ETH correlation
        if 'BTCUSD' in analysis_cache and 'ETHUSD' in analysis_cache:
            btc_closes = analysis_cache['BTCUSD'].get('h4_closes')
            eth_closes = analysis_cache['ETHUSD'].get('h4_closes')
            if btc_closes is not None and eth_closes is not None:
                corr = calculate_correlation(btc_closes, eth_closes)
                corr_str = f'\n  BTC/ETH Correlation: {corr["correlation"]:.2f} ({corr["status"]})'
                if corr['diverging']:
                    corr_str += f' [DIVERGING! BTC: {corr["btc_change"]:+.1f}% ETH: {corr["eth_change"]:+.1f}%]'
                print(corr_str)

        # Refresh account state
        state = api.get_account_state()
        equity = state['balance'] + state['openGrossPnL']
        print(f'  Account: Balance ${state["balance"]:.2f} | Equity ${equity:.2f} | Open P/L ${state["openGrossPnL"]:.2f}')

        print(f'\nNext scan in {CHECK_INTERVAL} seconds...')
        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nMonitor stopped by user')
    except Exception as e:
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()
