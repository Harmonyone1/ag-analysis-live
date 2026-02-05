"""ICT Entry Scanner - Find valid setups"""
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

def get_trend(closes, ma_period):
    if len(closes) < ma_period:
        return 'UNK', 0
    ma = np.mean(closes[-ma_period:])
    price = closes[-1]
    pct = (price - ma) / ma * 100
    return ('BULL' if price > ma else 'BEAR'), pct

def find_obs(opens, closes, highs, lows, lookback=40):
    bearish_ob = None
    bullish_ob = None

    o = opens[-lookback:]
    c = closes[-lookback:]
    h = highs[-lookback:]
    l = lows[-lookback:]

    for i in range(2, len(o)-3):
        if c[i] > o[i]:  # Bullish candle
            if c[i+1] < o[i+1] and c[i+2] < o[i+2]:
                if l[i+2] < l[i]:
                    bearish_ob = {'high': h[i], 'low': l[i]}
        if c[i] < o[i]:  # Bearish candle
            if c[i+1] > o[i+1] and c[i+2] > o[i+2]:
                if h[i+2] > h[i]:
                    bullish_ob = {'high': h[i], 'low': l[i]}

    return bearish_ob, bullish_ob

symbols = [
    'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF',
    'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY',
    'EURGBP', 'EURAUD', 'EURNZD', 'EURCAD',
    'GBPAUD', 'GBPNZD', 'GBPCAD',
    'AUDNZD', 'AUDCAD', 'NZDCAD',
    'XAUUSD', 'XAGUSD', 'BTCUSD', 'ETHUSD'
]

print('='*75)
print('ICT ENTRY SCAN')
print('='*75)
print()

valid_setups = []

for sym in symbols:
    if sym not in symbol_to_id:
        continue

    try:
        inst_id = symbol_to_id[sym]

        mn = api.get_price_history(inst_id, resolution='1M', lookback_period='365D')
        time.sleep(0.15)
        h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='21D')
        time.sleep(0.15)
        h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='3D')

        price = h1['c'].values[-1]
        mn_trend, _ = get_trend(mn['c'].values, 6)
        h4_rsi = get_rsi(h4['c'].values)
        h1_rsi = get_rsi(h1['c'].values)

        # Range position
        h4_high = h4['h'].values[-30:].max()
        h4_low = h4['l'].values[-30:].min()
        range_pct = (price - h4_low) / (h4_high - h4_low) * 100

        # Order blocks
        minus_ob, plus_ob = find_obs(h4['o'].values, h4['c'].values, h4['h'].values, h4['l'].values)

        # Check OB status
        minus_ob_mitigated = False
        plus_ob_mitigated = False

        if minus_ob and price < minus_ob['low']:
            minus_ob_mitigated = True
        if plus_ob and price > plus_ob['high']:
            plus_ob_mitigated = True

        # LONG SETUP CHECK
        if mn_trend == 'BULL':
            score = 0
            reasons = []
            warnings_list = []

            # Good: discount zone
            if range_pct < 35:
                score += 2
                reasons.append('Discount (%.0f%%)' % range_pct)

            # Good: RSI oversold
            if h4_rsi < 30:
                score += 3
                reasons.append('H4 RSI %.1f' % h4_rsi)
            elif h4_rsi < 40:
                score += 1
                reasons.append('H4 RSI %.1f' % h4_rsi)

            # Good: +OB mitigated
            if plus_ob_mitigated:
                score += 2
                reasons.append('+OB mitigated')

            # Bad: -OB mitigated
            if minus_ob_mitigated:
                score -= 3
                warnings_list.append('-OB mitigated')

            # Bad: premium zone
            if range_pct > 70:
                score -= 2
                warnings_list.append('Premium zone')

            if score >= 3 and len(warnings_list) == 0:
                valid_setups.append({
                    'symbol': sym,
                    'direction': 'LONG',
                    'price': price,
                    'score': score,
                    'h4_rsi': h4_rsi,
                    'range_pct': range_pct,
                    'reasons': reasons,
                    'mn_trend': mn_trend
                })

        # SHORT SETUP CHECK
        elif mn_trend == 'BEAR':
            score = 0
            reasons = []
            warnings_list = []

            # Good: premium zone
            if range_pct > 65:
                score += 2
                reasons.append('Premium (%.0f%%)' % range_pct)

            # Good: RSI overbought
            if h4_rsi > 70:
                score += 3
                reasons.append('H4 RSI %.1f' % h4_rsi)
            elif h4_rsi > 60:
                score += 1
                reasons.append('H4 RSI %.1f' % h4_rsi)

            # Good: -OB mitigated
            if minus_ob_mitigated:
                score += 2
                reasons.append('-OB mitigated')

            # Bad: +OB mitigated
            if plus_ob_mitigated:
                score -= 3
                warnings_list.append('+OB mitigated')

            # Bad: discount zone
            if range_pct < 30:
                score -= 2
                warnings_list.append('Discount zone')

            if score >= 3 and len(warnings_list) == 0:
                valid_setups.append({
                    'symbol': sym,
                    'direction': 'SHORT',
                    'price': price,
                    'score': score,
                    'h4_rsi': h4_rsi,
                    'range_pct': range_pct,
                    'reasons': reasons,
                    'mn_trend': mn_trend
                })

        time.sleep(0.15)
    except Exception as e:
        pass

# Sort by score
valid_setups.sort(key=lambda x: x['score'], reverse=True)

if valid_setups:
    print('VALID ICT SETUPS:')
    print('-'*75)
    for setup in valid_setups:
        sym = setup['symbol']
        if setup['price'] > 1000:
            pstr = '%.2f' % setup['price']
        elif setup['price'] > 10:
            pstr = '%.3f' % setup['price']
        else:
            pstr = '%.5f' % setup['price']

        print()
        print('%s %s | Score: %d' % (sym, setup['direction'], setup['score']))
        print('  Price: %s | H4 RSI: %.1f | Range: %.0f%%' % (pstr, setup['h4_rsi'], setup['range_pct']))
        print('  Monthly: %s' % setup['mn_trend'])
        print('  Factors: %s' % ', '.join(setup['reasons']))
else:
    print('No valid setups found matching ICT criteria.')
    print()
    print('Requirements:')
    print('  LONG: Monthly BULL + Discount zone + RSI low + No -OB mitigation')
    print('  SHORT: Monthly BEAR + Premium zone + RSI high + No +OB mitigation')
