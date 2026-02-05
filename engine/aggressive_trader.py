import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import time
from datetime import datetime

from config import get_api
api = get_api()

# ACCOUNT GROWTH TARGET
START_BALANCE = 331.67
TARGET_BALANCE = 10000
RISK_PER_TRADE = 0.08  # 8% risk per trade - aggressive but controlled

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# Assets to monitor for opportunities
WATCH_LIST = ['BTCUSD', 'ETHUSD', 'XAUUSD', 'US30', 'NAS100', 'EURUSD', 'GBPUSD']

def get_session():
    hour = datetime.now().hour
    if 8 <= hour < 12:
        return 'LONDON', 'HIGH'
    elif 12 <= hour < 17:
        return 'NEW_YORK', 'HIGH'
    elif 17 <= hour < 21:
        return 'NY_CLOSE', 'MEDIUM'
    elif 0 <= hour < 8:
        return 'ASIAN', 'LOW'
    else:
        return 'OFF_HOURS', 'LOW'

def get_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.0001
    return 100 - (100 / (1 + avg_gain/avg_loss))

def detect_order_blocks(opens, highs, lows, closes, lookback=30):
    obs = []
    current = closes[-1]
    for i in range(2, min(lookback, len(closes)-1)):
        idx = len(closes) - i - 1
        if idx < 1:
            break
        # Bearish OB
        if closes[idx] > opens[idx] and closes[idx+1] < opens[idx+1]:
            ob_range = highs[idx] - lows[idx]
            move = opens[idx+1] - closes[idx+1]
            if ob_range > 0 and move > ob_range * 1.2:
                if current < lows[idx]:
                    status = 'ACTIVE'
                elif lows[idx] <= current <= highs[idx]:
                    status = 'IN_ZONE'
                else:
                    status = 'BROKEN'
                if status != 'BROKEN':
                    obs.append({'type': 'BEARISH', 'high': highs[idx], 'low': lows[idx], 'status': status, 'strength': move/ob_range, 'age': i})
        # Bullish OB
        if closes[idx] < opens[idx] and closes[idx+1] > opens[idx+1]:
            ob_range = highs[idx] - lows[idx]
            move = closes[idx+1] - opens[idx+1]
            if ob_range > 0 and move > ob_range * 1.2:
                if current > highs[idx]:
                    status = 'ACTIVE'
                elif lows[idx] <= current <= highs[idx]:
                    status = 'IN_ZONE'
                else:
                    status = 'BROKEN'
                if status != 'BROKEN':
                    obs.append({'type': 'BULLISH', 'high': highs[idx], 'low': lows[idx], 'status': status, 'strength': move/ob_range, 'age': i})
    return obs[:5]

def detect_fvgs(highs, lows, closes):
    bull_fvgs, bear_fvgs = [], []
    current = closes[-1]
    for i in range(2, min(20, len(closes)-1)):
        idx = len(closes) - i - 1
        if idx < 1:
            break
        if lows[idx+1] > highs[idx-1]:
            if current < lows[idx+1]:
                bull_fvgs.append({'low': highs[idx-1], 'high': lows[idx+1], 'age': i})
        if highs[idx+1] < lows[idx-1]:
            if current > highs[idx+1]:
                bear_fvgs.append({'low': highs[idx+1], 'high': lows[idx-1], 'age': i})
    return bull_fvgs[:3], bear_fvgs[:3]

def analyze_asset(symbol):
    inst_id = symbol_to_id.get(symbol)
    if not inst_id:
        return None
    try:
        time.sleep(1)  # Rate limit protection
        h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='2W')
        time.sleep(0.3)
        h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='3D')
        time.sleep(0.3)
        m15 = api.get_price_history(inst_id, resolution='15m', lookback_period='1D')

        if h4 is None or h1 is None or m15 is None:
            return None
        if len(h4) < 20 or len(h1) < 20 or len(m15) < 20:
            return None

        current = m15['c'].values[-1]

        # Multi-timeframe analysis
        h4_trend = 'BULL' if h4['c'].values[-1] > h4['c'].values[-6] else 'BEAR'
        h1_trend = 'BULL' if h1['c'].values[-1] > h1['c'].values[-6] else 'BEAR'
        m15_trend = 'BULL' if m15['c'].values[-1] > m15['c'].values[-4] else 'BEAR'

        h4_rsi = get_rsi(h4['c'].values)
        h1_rsi = get_rsi(h1['c'].values)
        m15_rsi = get_rsi(m15['c'].values)

        # Order blocks on H1
        obs = detect_order_blocks(h1['o'].values, h1['h'].values, h1['l'].values, h1['c'].values)

        # FVGs on H1
        bull_fvgs, bear_fvgs = detect_fvgs(h1['h'].values, h1['l'].values, h1['c'].values)

        # Premium/Discount
        h4_high = max(h4['h'].values[-20:])
        h4_low = min(h4['l'].values[-20:])
        range_pct = (current - h4_low) / (h4_high - h4_low) * 100 if h4_high > h4_low else 50
        zone = 'PREMIUM' if range_pct > 60 else 'DISCOUNT' if range_pct < 40 else 'EQUILIBRIUM'

        # Volatility (ATR proxy)
        h1_ranges = h1['h'].values[-14:] - h1['l'].values[-14:]
        avg_range = np.mean(h1_ranges)
        current_range = h1['h'].values[-1] - h1['l'].values[-1]
        volatility = current_range / avg_range if avg_range > 0 else 1

        # SIGNAL SCORING
        signal = None
        score = 0
        direction = None
        entry = current
        stop = None
        target = None

        # Check for SHORT setups
        for ob in obs:
            if ob['type'] == 'BEARISH' and ob['status'] == 'IN_ZONE':
                score += 30
                direction = 'SHORT'
                stop = ob['high'] * 1.002  # Just above OB
                target = ob['low'] - (ob['high'] - ob['low'])  # 1:1 minimum

                if h4_trend == 'BEAR':
                    score += 20
                if zone == 'PREMIUM':
                    score += 15
                if h1_rsi > 60:
                    score += 10
                if bear_fvgs:
                    score += 10
                if ob['strength'] > 2:
                    score += 10

        # Check for LONG setups
        for ob in obs:
            if ob['type'] == 'BULLISH' and ob['status'] == 'IN_ZONE':
                if score < 30:  # Don't override stronger short signal
                    score = 30
                    direction = 'LONG'
                    stop = ob['low'] * 0.998
                    target = ob['high'] + (ob['high'] - ob['low'])

                    if h4_trend == 'BULL':
                        score += 20
                    if zone == 'DISCOUNT':
                        score += 15
                    if h1_rsi < 40:
                        score += 10
                    if bull_fvgs:
                        score += 10
                    if ob['strength'] > 2:
                        score += 10

        # Calculate R:R
        if stop and target and direction:
            risk = abs(entry - stop)
            reward = abs(target - entry)
            rr = reward / risk if risk > 0 else 0
        else:
            rr = 0

        # Signal quality
        if score >= 70 and rr >= 2:
            signal = 'A+'
        elif score >= 55 and rr >= 1.5:
            signal = 'A'
        elif score >= 40 and rr >= 1:
            signal = 'B'
        else:
            signal = None

        return {
            'symbol': symbol,
            'price': current,
            'h4_trend': h4_trend,
            'h1_trend': h1_trend,
            'm15_trend': m15_trend,
            'h4_rsi': h4_rsi,
            'h1_rsi': h1_rsi,
            'm15_rsi': m15_rsi,
            'zone': zone,
            'range_pct': range_pct,
            'obs': obs,
            'bull_fvgs': bull_fvgs,
            'bear_fvgs': bear_fvgs,
            'volatility': volatility,
            'signal': signal,
            'score': score,
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'rr': rr
        }
    except Exception as e:
        return None

def calculate_position_size(account_balance, entry, stop, symbol):
    """Calculate aggressive but controlled position size"""
    risk_amount = account_balance * RISK_PER_TRADE
    risk_per_unit = abs(entry - stop)

    if risk_per_unit == 0:
        return 0

    # Get lot size info
    if 'BTC' in symbol:
        pip_value = 1  # $1 per point per lot
        min_lot = 0.01
        lot_step = 0.01
    elif 'ETH' in symbol:
        pip_value = 1
        min_lot = 0.1
        lot_step = 0.1
    elif 'XAU' in symbol:
        pip_value = 1
        min_lot = 0.01
        lot_step = 0.01
    elif 'US30' in symbol or 'NAS' in symbol:
        pip_value = 1
        min_lot = 0.1
        lot_step = 0.1
    else:  # Forex
        pip_value = 10  # $10 per pip per lot
        risk_per_unit = risk_per_unit * 10000  # Convert to pips
        min_lot = 0.01
        lot_step = 0.01

    lots = risk_amount / (risk_per_unit * pip_value)
    lots = max(min_lot, round(lots / lot_step) * lot_step)

    return lots

def get_open_positions():
    try:
        time.sleep(0.5)
        positions = api.get_all_positions()
        if positions is None or len(positions) == 0:
            return []
        return positions.to_dict('records') if hasattr(positions, 'to_dict') else []
    except Exception as e:
        if '429' in str(e):
            print('  [Rate limit hit on positions - waiting...]')
            time.sleep(5)
        return []

def get_account_state():
    try:
        time.sleep(0.5)
        state = api.get_account_state()
        return {
            'balance': float(state.get('balance', 0)),
            'equity': float(state.get('equity', 0)),
            'margin': float(state.get('marginUsed', 0)),
            'free_margin': float(state.get('freeMargin', 0))
        }
    except Exception as e:
        if '429' in str(e):
            print('  [Rate limit hit on account - waiting...]')
            time.sleep(5)
        return {'balance': START_BALANCE, 'equity': START_BALANCE, 'margin': 0, 'free_margin': 0}

print('=' * 70)
print('AGGRESSIVE GROWTH TRADER')
print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('=' * 70)
print(f'START BALANCE: ${START_BALANCE:.2f}')
print(f'TARGET: ${TARGET_BALANCE:,.2f} ({TARGET_BALANCE/START_BALANCE:.0f}x growth)')
print(f'RISK PER TRADE: {RISK_PER_TRADE*100:.0f}%')
print(f'WATCHLIST: {", ".join(WATCH_LIST)}')
print('=' * 70)
print()

scan_count = 0
while True:
    scan_count += 1
    session, volatility = get_session()
    now = datetime.now().strftime("%H:%M:%S")

    account = get_account_state()
    positions = get_open_positions()

    # Progress tracking
    current_balance = account['balance']
    progress = ((current_balance - START_BALANCE) / (TARGET_BALANCE - START_BALANCE)) * 100
    growth = ((current_balance / START_BALANCE) - 1) * 100

    print(f'[{now}] Scan #{scan_count} | {session} ({volatility} vol)')
    print(f'Account: ${account["balance"]:.2f} | Equity: ${account["equity"]:.2f} | Progress: {progress:.1f}% to $10K')
    print('-' * 70)

    # Show open positions
    if positions:
        print('OPEN POSITIONS:')
        for pos in positions:
            sym = pos.get('symbol', pos.get('tradableInstrumentId', 'Unknown'))
            side = pos.get('side', 'Unknown')
            qty = pos.get('qty', 0)
            entry_price = pos.get('avgPrice', 0)
            pnl = pos.get('unrealizedPnL', 0)
            print(f'  {sym}: {side} {qty} @ {entry_price} | P/L: ${pnl:.2f}')
        print()

    # Scan for opportunities
    opportunities = []
    print('SCANNING WATCHLIST...')

    for symbol in WATCH_LIST:
        analysis = analyze_asset(symbol)
        if analysis and analysis['signal']:
            opportunities.append(analysis)

            # Calculate position size
            lot_size = calculate_position_size(
                account['balance'],
                analysis['entry'],
                analysis['stop'],
                symbol
            )

            print()
            print('!' * 70)
            print(f'>>> {analysis["signal"]} SIGNAL: {symbol} {analysis["direction"]}')
            print('!' * 70)
            print(f'Price: {analysis["price"]:.2f}')
            print(f'Score: {analysis["score"]}/100')
            print(f'Trends: H4={analysis["h4_trend"]} H1={analysis["h1_trend"]} M15={analysis["m15_trend"]}')
            print(f'RSI: H4={analysis["h4_rsi"]:.1f} H1={analysis["h1_rsi"]:.1f}')
            print(f'Zone: {analysis["zone"]} ({analysis["range_pct"]:.1f}%)')
            print(f'Entry: {analysis["entry"]:.2f}')
            print(f'Stop: {analysis["stop"]:.2f}')
            print(f'Target: {analysis["target"]:.2f}')
            print(f'R:R = 1:{analysis["rr"]:.1f}')
            print(f'RECOMMENDED SIZE: {lot_size} lots (risking ${account["balance"] * RISK_PER_TRADE:.2f})')

            # Potential profit
            if analysis['direction'] == 'LONG':
                potential = (analysis['target'] - analysis['entry']) * lot_size
            else:
                potential = (analysis['entry'] - analysis['target']) * lot_size
            print(f'POTENTIAL PROFIT: ${potential:.2f} ({potential/account["balance"]*100:.1f}% of account)')
            print()

    if not opportunities:
        print('No A/A+ setups found. Waiting for high-probability entries...')

    # Position management alerts
    for pos in positions:
        # Check if any position needs attention
        pnl = pos.get('unrealizedPnL', 0)
        if pnl < -account['balance'] * 0.05:
            print(f'WARNING: Position underwater by ${abs(pnl):.2f} - review stop loss')
        elif pnl > account['balance'] * 0.10:
            print(f'WINNER: Consider taking partial profits - up ${pnl:.2f}')

    print()
    print(f'Next scan in 90 seconds...')
    print('=' * 70)
    print()

    time.sleep(90)  # Longer interval to avoid rate limits
