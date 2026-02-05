#!/usr/bin/env python
"""
AUTO TRADER - ICT+QUANT Framework with Liquidity Analysis
==========================================================
- Monitors all interesting pairs, stocks, indices
- Calculates confidence scores based on multiple factors
- Executes trades automatically when confidence threshold met
- Scales position size based on confidence level
- Tracks account growth and adjusts risk accordingly
"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
import time
from datetime import datetime
import json

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'min_confidence': 75,          # Minimum confidence to trade (%)
    'base_lot_size': 0.02,         # Base position size
    'max_lot_size': 0.10,          # Maximum position size
    'risk_per_trade': 2.0,         # Risk % per trade
    'max_open_positions': 3,       # Maximum concurrent positions
    'scan_interval': 120,          # Seconds between scans
    'confidence_scaling': {        # Lot multiplier by confidence
        75: 1.0,   # 75-79%: base lots
        80: 1.5,   # 80-84%: 1.5x base
        85: 2.0,   # 85-89%: 2x base
        90: 2.5,   # 90-94%: 2.5x base
        95: 3.0,   # 95%+: 3x base
    }
}

# Full watchlist
WATCHLIST = {
    'FOREX_MAJORS': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF'],
    'FOREX_CROSSES': ['GBPJPY', 'EURJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY',
                      'EURGBP', 'EURAUD', 'EURNZD', 'EURCAD', 'AUDNZD', 'AUDCAD', 'NZDCAD'],
    'INDICES': ['US30', 'NAS100', 'SPX500', 'DE40', 'UK100'],
    'STOCKS': ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'JPM', 'ORCL', 'PFE', 'NFLX'],
    'CRYPTO': ['BTCUSD', 'ETHUSD'],
    'COMMODITIES': ['XAUUSD', 'XAGUSD', 'USOIL']
}

# ============================================================
# API CONNECTION
# ============================================================

from config import get_api
api = get_api()

instruments = api.get_all_instruments()
name_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def calc_rsi(closes, period=14):
    """Calculate RSI"""
    if len(closes) < period + 1:
        return 50
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0.001
    return 100 - (100 / (1 + avg_gain / avg_loss))

def find_liquidity_levels(highs, lows, price):
    """Find nearest BSL and SSL levels"""
    bsl_levels = []
    ssl_levels = []

    # Find swing highs (BSL)
    for i in range(2, min(20, len(highs) - 2)):
        if highs[-i] > highs[-i-1] and highs[-i] > highs[-i+1]:
            if highs[-i] > price:
                bsl_levels.append(highs[-i])

    # Find swing lows (SSL)
    for i in range(2, min(20, len(lows) - 2)):
        if lows[-i] < lows[-i-1] and lows[-i] < lows[-i+1]:
            if lows[-i] < price:
                ssl_levels.append(lows[-i])

    nearest_bsl = min(bsl_levels) if bsl_levels else None
    nearest_ssl = max(ssl_levels) if ssl_levels else None

    return nearest_bsl, nearest_ssl

def analyze_setup(symbol):
    """Complete ICT+Quant analysis with confidence scoring"""
    try:
        inst_id = name_to_id.get(symbol)
        if not inst_id:
            return None

        # Get price data
        d1 = api.get_price_history(inst_id, resolution='1D', start_timestamp=0, end_timestamp=0, lookback_period='30D')
        h4 = api.get_price_history(inst_id, resolution='4H', start_timestamp=0, end_timestamp=0, lookback_period='14D')
        h1 = api.get_price_history(inst_id, resolution='1H', start_timestamp=0, end_timestamp=0, lookback_period='7D')

        if d1 is None or h4 is None or h1 is None:
            return None
        if len(d1) < 10 or len(h4) < 10 or len(h1) < 15:
            return None

        price = api.get_latest_asking_price(inst_id)

        # ============================================
        # 1. HTF BIAS (Daily + 4H alignment)
        # ============================================
        d1_bias = 'BULL' if d1['c'].iloc[-1] > d1['o'].iloc[-5] else 'BEAR'
        h4_bias = 'BULL' if h4['c'].iloc[-1] > h4['o'].iloc[-5] else 'BEAR'
        htf_aligned = d1_bias == h4_bias
        htf_bias = d1_bias if htf_aligned else 'MIX'

        # ============================================
        # 2. RSI (Multiple timeframes)
        # ============================================
        rsi_d1 = calc_rsi(d1['c'].values)
        rsi_h4 = calc_rsi(h4['c'].values)
        rsi_h1 = calc_rsi(h1['c'].values)

        # ============================================
        # 3. ZONE ANALYSIS (Premium/Discount)
        # ============================================
        high_20 = max(h4['h'].values[-20:])
        low_20 = min(h4['l'].values[-20:])
        range_size = high_20 - low_20
        range_pct = (price - low_20) / range_size * 100 if range_size > 0 else 50

        if range_pct < 20:
            zone = 'DEEP_DISC'
        elif range_pct < 35:
            zone = 'DISC'
        elif range_pct > 80:
            zone = 'DEEP_PREM'
        elif range_pct > 65:
            zone = 'PREM'
        else:
            zone = 'EQ'

        # ============================================
        # 4. OTE CHECK (62-79% retracement)
        # ============================================
        swing_high = max(h4['h'].values[-10:])
        swing_low = min(h4['l'].values[-10:])
        swing_range = swing_high - swing_low
        if swing_range > 0:
            retrace_pct = (swing_high - price) / swing_range * 100
            in_ote = 62 <= retrace_pct <= 79
        else:
            in_ote = False

        # ============================================
        # 5. LIQUIDITY ANALYSIS
        # ============================================
        nearest_bsl, nearest_ssl = find_liquidity_levels(
            h4['h'].values, h4['l'].values, price
        )

        # Check if near liquidity (potential sweep setup)
        pip_mult = 100 if 'JPY' in symbol else 10000

        near_bsl = False
        near_ssl = False
        bsl_distance = None
        ssl_distance = None

        if nearest_bsl:
            bsl_distance = (nearest_bsl - price) * pip_mult
            near_bsl = bsl_distance < 15  # Within 15 pips

        if nearest_ssl:
            ssl_distance = (price - nearest_ssl) * pip_mult
            near_ssl = ssl_distance < 15  # Within 15 pips

        # ============================================
        # 6. CONFLUENCE COUNT
        # ============================================
        long_confluences = 0
        short_confluences = 0

        # Zone confluences
        if zone in ['DISC', 'DEEP_DISC']:
            long_confluences += 2 if zone == 'DEEP_DISC' else 1
        if zone in ['PREM', 'DEEP_PREM']:
            short_confluences += 2 if zone == 'DEEP_PREM' else 1

        # RSI confluences
        if rsi_h1 < 30:
            long_confluences += 2
        elif rsi_h1 < 40:
            long_confluences += 1
        if rsi_h1 > 70:
            short_confluences += 2
        elif rsi_h1 > 60:
            short_confluences += 1

        # OTE confluence
        if in_ote:
            long_confluences += 1
            short_confluences += 1

        # Liquidity confluence
        if near_ssl:  # Near sell-side liquidity = potential long after sweep
            long_confluences += 1
        if near_bsl:  # Near buy-side liquidity = potential short after sweep
            short_confluences += 1

        # ============================================
        # 7. SIGNAL DETERMINATION
        # ============================================
        signal = None
        direction = None
        confidence = 0

        # LONG setup
        if htf_bias == 'BULL' and zone in ['DISC', 'DEEP_DISC'] and rsi_h1 < 40:
            signal = 'VALID_LONG'
            direction = 'LONG'
            confidence = 60 + (long_confluences * 8)
            if rsi_h1 < 30:
                confidence += 5
            if zone == 'DEEP_DISC':
                confidence += 5
            if in_ote:
                confidence += 5
            if htf_aligned:
                confidence += 5

        # SHORT setup
        elif htf_bias == 'BEAR' and zone in ['PREM', 'DEEP_PREM'] and rsi_h1 > 60:
            signal = 'VALID_SHORT'
            direction = 'SHORT'
            confidence = 60 + (short_confluences * 8)
            if rsi_h1 > 70:
                confidence += 5
            if zone == 'DEEP_PREM':
                confidence += 5
            if in_ote:
                confidence += 5
            if htf_aligned:
                confidence += 5

        # DEVELOPING setups
        elif htf_bias == 'BULL' and zone in ['DISC', 'DEEP_DISC']:
            signal = 'DEVELOPING_LONG'
            direction = 'LONG'
            confidence = 40 + (long_confluences * 5)

        elif htf_bias == 'BEAR' and zone in ['PREM', 'DEEP_PREM']:
            signal = 'DEVELOPING_SHORT'
            direction = 'SHORT'
            confidence = 40 + (short_confluences * 5)

        # Cap confidence at 98
        confidence = min(confidence, 98)

        # ============================================
        # 8. CALCULATE SL/TP
        # ============================================
        atr = sum(abs(h4['h'].iloc[i] - h4['l'].iloc[i]) for i in range(-5, 0)) / 5

        if direction == 'LONG':
            sl_price = price - (atr * 1.5)
            tp_price = price + (atr * 3)
        elif direction == 'SHORT':
            sl_price = price + (atr * 1.5)
            tp_price = price - (atr * 3)
        else:
            sl_price = None
            tp_price = None

        sl_pips = abs(price - sl_price) * pip_mult if sl_price else 0
        tp_pips = abs(tp_price - price) * pip_mult if tp_price else 0

        return {
            'symbol': symbol,
            'price': price,
            'htf_bias': htf_bias,
            'htf_aligned': htf_aligned,
            'rsi_h1': rsi_h1,
            'rsi_h4': rsi_h4,
            'zone': zone,
            'range_pct': range_pct,
            'in_ote': in_ote,
            'near_bsl': near_bsl,
            'near_ssl': near_ssl,
            'bsl_distance': bsl_distance,
            'ssl_distance': ssl_distance,
            'signal': signal,
            'direction': direction,
            'confidence': confidence,
            'confluences': long_confluences if direction == 'LONG' else short_confluences,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
        }

    except Exception as e:
        return None

def calculate_lot_size(confidence, balance, sl_pips):
    """Calculate position size based on confidence and risk"""
    # Get lot multiplier from confidence
    multiplier = 1.0
    for conf_level, mult in sorted(CONFIG['confidence_scaling'].items()):
        if confidence >= conf_level:
            multiplier = mult

    # Calculate risk-based lot size
    risk_amount = balance * (CONFIG['risk_per_trade'] / 100)
    pip_value = 10  # Approximate pip value per standard lot

    if sl_pips > 0:
        risk_lots = risk_amount / (sl_pips * pip_value)
    else:
        risk_lots = CONFIG['base_lot_size']

    # Apply multiplier and cap
    lot_size = min(risk_lots * multiplier, CONFIG['max_lot_size'])
    lot_size = max(lot_size, 0.01)  # Minimum 0.01

    return round(lot_size, 2)

def execute_trade(setup, lot_size):
    """Execute a trade based on setup"""
    try:
        inst_id = name_to_id.get(setup['symbol'])
        if not inst_id:
            return False, "Instrument not found"

        side = 'buy' if setup['direction'] == 'LONG' else 'sell'

        order = api.create_order(
            instrument_id=inst_id,
            quantity=lot_size,
            side=side,
            type_='market',
            stop_loss=setup['sl_price'],
            take_profit=setup['tp_price']
        )

        if order:
            return True, "Order placed successfully"
        else:
            return False, "Order failed"

    except Exception as e:
        return False, str(e)

# ============================================================
# MAIN TRADING LOOP
# ============================================================

def main():
    print('=' * 80)
    print('AUTO TRADER - ICT+QUANT Framework')
    print('=' * 80)
    print()
    print('Configuration:')
    print('  Min Confidence: %d%%' % CONFIG['min_confidence'])
    print('  Base Lot Size:  %.2f' % CONFIG['base_lot_size'])
    print('  Max Lot Size:   %.2f' % CONFIG['max_lot_size'])
    print('  Risk Per Trade: %.1f%%' % CONFIG['risk_per_trade'])
    print('  Scan Interval:  %d seconds' % CONFIG['scan_interval'])
    print()

    trades_taken = []

    for cycle in range(90):  # 90 cycles x 2 min = 3 hours
        now = datetime.now()
        time_str = now.strftime('%I:%M:%S %p')

        # Get account state
        try:
            state = api.get_account_state()
            balance = state['balance']
            equity = balance + state['openGrossPnL']
            positions = api.get_all_positions()
            open_count = len(positions) if not positions.empty else 0
        except:
            balance = 340.65
            equity = balance
            open_count = 0

        print('=' * 80)
        print('[%s] SCAN CYCLE %d/90 | Balance: $%.2f | Equity: $%.2f | Open: %d' % (
            time_str, cycle + 1, balance, equity, open_count))
        print('=' * 80)

        valid_setups = []
        developing_setups = []
        all_results = []

        # Scan all watchlist
        all_symbols = []
        for category, symbols in WATCHLIST.items():
            all_symbols.extend(symbols)

        for symbol in all_symbols:
            result = analyze_setup(symbol)
            if result:
                all_results.append(result)
                if result['signal'] and 'VALID' in result['signal']:
                    valid_setups.append(result)
                elif result['signal'] and 'DEVELOPING' in result['signal']:
                    developing_setups.append(result)
            time.sleep(0.15)  # Rate limiting

        # Sort by confidence
        valid_setups.sort(key=lambda x: x['confidence'], reverse=True)
        developing_setups.sort(key=lambda x: x['confidence'], reverse=True)

        # Display valid setups
        if valid_setups:
            print()
            print('!' * 80)
            print('VALID SETUPS DETECTED')
            print('!' * 80)

            for s in valid_setups:
                print()
                print('  %s %s | Confidence: %d%%' % (s['symbol'], s['direction'], s['confidence']))
                print('  Price: %.5f | HTF: %s | RSI: %.1f | Zone: %s (%.0f%%)' % (
                    s['price'], s['htf_bias'], s['rsi_h1'], s['zone'], s['range_pct']))
                print('  Confluences: %d | OTE: %s | SL: %.1f pips | TP: %.1f pips' % (
                    s['confluences'], 'Yes' if s['in_ote'] else 'No', s['sl_pips'], s['tp_pips']))

                # Check if we should trade
                if s['confidence'] >= CONFIG['min_confidence'] and open_count < CONFIG['max_open_positions']:
                    lot_size = calculate_lot_size(s['confidence'], balance, s['sl_pips'])
                    print()
                    print('  >>> EXECUTING TRADE: %s %s %.2f lots <<<' % (
                        s['symbol'], s['direction'], lot_size))

                    success, msg = execute_trade(s, lot_size)

                    if success:
                        print('  >>> TRADE PLACED SUCCESSFULLY <<<')
                        trades_taken.append({
                            'time': time_str,
                            'symbol': s['symbol'],
                            'direction': s['direction'],
                            'lots': lot_size,
                            'confidence': s['confidence'],
                            'entry': s['price'],
                            'sl': s['sl_price'],
                            'tp': s['tp_price']
                        })
                        open_count += 1
                    else:
                        print('  >>> TRADE FAILED: %s <<<' % msg)
                else:
                    if s['confidence'] < CONFIG['min_confidence']:
                        print('  [SKIP] Confidence %d%% below threshold %d%%' % (
                            s['confidence'], CONFIG['min_confidence']))
                    elif open_count >= CONFIG['max_open_positions']:
                        print('  [SKIP] Max positions reached (%d)' % CONFIG['max_open_positions'])

        # Display developing setups
        if developing_setups:
            print()
            print('-' * 80)
            print('DEVELOPING SETUPS (Waiting for RSI confirmation)')
            print('-' * 80)
            for s in developing_setups[:5]:
                rsi_need = '<40' if s['direction'] == 'LONG' else '>60'
                print('  %s %s | Conf: %d%% | RSI: %.1f (need %s) | %s %.0f%%' % (
                    s['symbol'], s['direction'], s['confidence'],
                    s['rsi_h1'], rsi_need, s['zone'], s['range_pct']))

        # Display watching (extreme RSI)
        extreme_rsi = [r for r in all_results if r['rsi_h1'] < 25 or r['rsi_h1'] > 75]
        if extreme_rsi:
            print()
            print('-' * 80)
            print('EXTREME RSI (Watching for reversal)')
            print('-' * 80)
            for r in sorted(extreme_rsi, key=lambda x: abs(50 - x['rsi_h1']), reverse=True)[:5]:
                status = 'OVERSOLD' if r['rsi_h1'] < 30 else 'OVERBOUGHT'
                print('  %s | RSI: %.1f %s | HTF: %s | %s %.0f%%' % (
                    r['symbol'], r['rsi_h1'], status, r['htf_bias'], r['zone'], r['range_pct']))

        if not valid_setups and not developing_setups:
            print()
            print('No actionable setups this scan.')

        # Show trade summary
        if trades_taken:
            print()
            print('-' * 80)
            print('TRADES THIS SESSION: %d' % len(trades_taken))
            for t in trades_taken[-3:]:
                print('  %s: %s %s %.2f lots @ %.5f (Conf: %d%%)' % (
                    t['time'], t['symbol'], t['direction'], t['lots'], t['entry'], t['confidence']))

        print()
        print('Next scan in %d seconds...' % CONFIG['scan_interval'])
        print()

        if cycle < 89:
            time.sleep(CONFIG['scan_interval'])

    print()
    print('=' * 80)
    print('AUTO TRADER SESSION COMPLETE')
    print('=' * 80)
    print('Total trades: %d' % len(trades_taken))
    for t in trades_taken:
        print('  %s: %s %s %.2f lots' % (t['time'], t['symbol'], t['direction'], t['lots']))

if __name__ == '__main__':
    main()
