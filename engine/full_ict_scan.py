"""Full ICT Scanner - All Instruments"""
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
all_symbols = instruments['name'].tolist()

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
    return ('BULL' if price > ma else 'BEAR'), (price - ma) / ma * 100

def find_obs(opens, closes, highs, lows, lookback=40):
    bearish_ob = None
    bullish_ob = None

    if len(opens) < lookback:
        return None, None

    o = opens[-lookback:]
    c = closes[-lookback:]
    h = highs[-lookback:]
    l = lows[-lookback:]

    for i in range(2, len(o)-3):
        if c[i] > o[i]:
            if c[i+1] < o[i+1] and c[i+2] < o[i+2]:
                if l[i+2] < l[i]:
                    bearish_ob = {'high': h[i], 'low': l[i]}
        if c[i] < o[i]:
            if c[i+1] > o[i+1] and c[i+2] > o[i+2]:
                if h[i+2] > h[i]:
                    bullish_ob = {'high': h[i], 'low': l[i]}

    return bearish_ob, bullish_ob

# Categorize instruments
categories = {
    'MAJORS': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
    'JPY_CROSSES': ['EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY'],
    'EUR_CROSSES': ['EURGBP', 'EURAUD', 'EURNZD', 'EURCAD', 'EURCHF'],
    'GBP_CROSSES': ['GBPAUD', 'GBPNZD', 'GBPCAD', 'GBPCHF'],
    'OTHER_CROSSES': ['AUDNZD', 'AUDCAD', 'AUDCHF', 'NZDCAD', 'NZDCHF', 'CADCHF'],
    'EXOTICS': ['USDMXN', 'USDZAR', 'USDPLN', 'USDHUF', 'EURNOK', 'EURSEK', 'EURPLN', 'EURHUF', 'EURZAR', 'GBPNOK', 'GBPSEK', 'GBPZAR'],
    'INDICES': ['US30', 'SPX500', 'NAS100', 'DE40', 'UK100', 'JP225', 'AUS200', 'F40', 'ES35', 'EUSTX50', 'HK50', 'NL25', 'CH20', 'RUS2000'],
    'CRYPTO': ['BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD', 'ADAUSD', 'DOGEUSD', 'XLMUSD', 'XMRUSD', 'ZECUSD'],
    'COMMODITIES': ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'USOIL', 'UKOIL', 'NGAS'],
    'US_STOCKS': ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'NFLX', 'NVDA', 'META', 'GOOGL', 'JPM', 'BAC', 'GS', 'V', 'MA', 'JNJ', 'PFE', 'KO', 'PG', 'IBM', 'INTC', 'CSCO', 'ORCL', 'BA', 'GE', 'F', 'GM', 'XOM', 'CVX', 'MCD', 'BABA'],
    'EU_STOCKS': ['SIE', 'ALV', 'BMW', 'MBG', 'VOW', 'BAYN', 'ADS', 'LHA', 'DB', 'CBK', 'SAP', 'BNP', 'LVMH', 'TTE', 'SAN']
}

print('='*80)
print('FULL ICT SCAN - ALL INSTRUMENTS')
print('='*80)
print()

valid_setups = []
all_scanned = []

for category, symbols in categories.items():
    for sym in symbols:
        if sym not in symbol_to_id:
            continue

        try:
            inst_id = symbol_to_id[sym]

            mn = api.get_price_history(inst_id, resolution='1M', lookback_period='365D')
            time.sleep(0.1)
            h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='21D')
            time.sleep(0.1)

            if mn is None or h4 is None or len(h4) < 30:
                continue

            price = h4['c'].values[-1]
            mn_trend, _ = get_trend(mn['c'].values, 6)
            h4_rsi = get_rsi(h4['c'].values)

            h4_high = h4['h'].values[-30:].max()
            h4_low = h4['l'].values[-30:].min()
            range_pct = (price - h4_low) / (h4_high - h4_low) * 100

            minus_ob, plus_ob = find_obs(h4['o'].values, h4['c'].values, h4['h'].values, h4['l'].values)

            minus_ob_mitigated = False
            plus_ob_mitigated = False
            minus_ob_testing = False
            plus_ob_testing = False

            if minus_ob:
                if price < minus_ob['low']:
                    minus_ob_mitigated = True
                elif price <= minus_ob['high'] and price >= minus_ob['low']:
                    minus_ob_testing = True

            if plus_ob:
                if price > plus_ob['high']:
                    plus_ob_mitigated = True
                elif price >= plus_ob['low'] and price <= plus_ob['high']:
                    plus_ob_testing = True

            # Score the setup
            score = 0
            direction = None
            reasons = []
            warnings_list = []

            if mn_trend == 'BULL':
                direction = 'LONG'
                if range_pct < 35:
                    score += 2
                    reasons.append('Discount %.0f%%' % range_pct)
                if h4_rsi < 30:
                    score += 3
                    reasons.append('H4 RSI %.1f' % h4_rsi)
                elif h4_rsi < 40:
                    score += 1
                    reasons.append('H4 RSI %.1f' % h4_rsi)
                if plus_ob_mitigated:
                    score += 2
                    reasons.append('+OB mitigated')
                if plus_ob_testing:
                    score += 1
                    reasons.append('In +OB zone')
                if minus_ob_mitigated:
                    score -= 3
                    warnings_list.append('-OB mitigated')
                if range_pct > 70:
                    score -= 2
                    warnings_list.append('Premium')

            elif mn_trend == 'BEAR':
                direction = 'SHORT'
                if range_pct > 65:
                    score += 2
                    reasons.append('Premium %.0f%%' % range_pct)
                if h4_rsi > 70:
                    score += 3
                    reasons.append('H4 RSI %.1f' % h4_rsi)
                elif h4_rsi > 60:
                    score += 1
                    reasons.append('H4 RSI %.1f' % h4_rsi)
                if minus_ob_mitigated:
                    score += 2
                    reasons.append('-OB mitigated')
                if minus_ob_testing:
                    score += 1
                    reasons.append('In -OB zone')
                if plus_ob_mitigated:
                    score -= 3
                    warnings_list.append('+OB mitigated')
                if range_pct < 30:
                    score -= 2
                    warnings_list.append('Discount')

            all_scanned.append({
                'symbol': sym,
                'category': category,
                'direction': direction,
                'price': price,
                'score': score,
                'h4_rsi': h4_rsi,
                'range_pct': range_pct,
                'reasons': reasons,
                'warnings': warnings_list,
                'mn_trend': mn_trend
            })

            if score >= 3 and len(warnings_list) == 0:
                valid_setups.append(all_scanned[-1])

            time.sleep(0.1)
        except Exception as e:
            pass

# Sort by score
valid_setups.sort(key=lambda x: x['score'], reverse=True)
all_scanned.sort(key=lambda x: x['score'], reverse=True)

print('VALID ICT SETUPS (No warnings, Score >= 3):')
print('-'*80)

if valid_setups:
    for setup in valid_setups[:15]:
        sym = setup['symbol']
        if setup['price'] > 1000:
            pstr = '%.2f' % setup['price']
        elif setup['price'] > 10:
            pstr = '%.3f' % setup['price']
        else:
            pstr = '%.5f' % setup['price']

        print()
        print('%s %s | Score: %d | %s' % (sym, setup['direction'], setup['score'], setup['category']))
        print('  Price: %s | H4 RSI: %.1f | Range: %.0f%%' % (pstr, setup['h4_rsi'], setup['range_pct']))
        print('  Factors: %s' % ', '.join(setup['reasons']))
else:
    print('No setups pass all criteria.')

print()
print('='*80)
print('TOP APPROACHING SETUPS (With warnings):')
print('-'*80)

approaching = [s for s in all_scanned if s['score'] >= 2 and len(s['warnings']) > 0][:10]
for setup in approaching:
    sym = setup['symbol']
    if setup['price'] > 1000:
        pstr = '%.2f' % setup['price']
    elif setup['price'] > 10:
        pstr = '%.3f' % setup['price']
    else:
        pstr = '%.5f' % setup['price']

    print()
    print('%s %s | Score: %d | %s' % (sym, setup['direction'], setup['score'], setup['category']))
    print('  Price: %s | H4 RSI: %.1f | Range: %.0f%%' % (pstr, setup['h4_rsi'], setup['range_pct']))
    print('  Good: %s' % ', '.join(setup['reasons']) if setup['reasons'] else 'None')
    print('  Warn: %s' % ', '.join(setup['warnings']))

print()
print('='*80)
print('Scanned %d instruments' % len(all_scanned))
