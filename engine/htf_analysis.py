"""
Higher Timeframe Analysis - MUST RUN BEFORE ANY ENTRY
Top-down approach: Monthly -> Weekly -> Daily -> H4 -> H1
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

def get_trend(closes, ma_period=10):
    if len(closes) < ma_period:
        return 'UNKNOWN', 0
    ma = np.mean(closes[-ma_period:])
    price = closes[-1]
    pct = (price - ma) / ma * 100
    if price > ma:
        return 'BULLISH', pct
    else:
        return 'BEARISH', pct

def count_candles(opens, closes, count=4):
    bulls = 0
    bears = 0
    for i in range(-count, 0):
        if closes[i] > opens[i]:
            bulls += 1
        else:
            bears += 1
    return bulls, bears

def analyze_instrument(symbol):
    api = get_api()
    instruments = api.get_all_instruments()
    symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))
    
    if symbol not in symbol_to_id:
        print(f"Symbol {symbol} not found")
        return
    
    inst_id = symbol_to_id[symbol]
    
    print('=' * 70)
    print(f'{symbol} - TOP-DOWN ANALYSIS')
    print('=' * 70)
    
    # Fetch all timeframes
    try:
        mn = api.get_price_history(inst_id, resolution='1M', lookback_period='730D')
        time.sleep(0.4)
        w1 = api.get_price_history(inst_id, resolution='1W', lookback_period='365D')
        time.sleep(0.4)
        d1 = api.get_price_history(inst_id, resolution='1D', lookback_period='90D')
        time.sleep(0.4)
        h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='30D')
        time.sleep(0.4)
        h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='7D')
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    price = h1['c'].values[-1]
    
    if price > 1000:
        price_str = f"${price:.2f}"
    elif price > 10:
        price_str = f"${price:.2f}"
    else:
        price_str = f"{price:.5f}"
    
    print(f"Current Price: {price_str}")
    print()
    
    # ============ MONTHLY ============
    print('1. MONTHLY (Primary Trend)')
    print('-' * 70)
    mn_rsi = get_rsi(mn['c'].values)
    mn_trend, mn_pct = get_trend(mn['c'].values, 6)
    mn_bulls, mn_bears = count_candles(mn['o'].values, mn['c'].values, 3)
    mn_ma = np.mean(mn['c'].values[-6:])
    
    print(f"   RSI: {mn_rsi:.1f}")
    print(f"   Trend: {mn_trend} ({mn_pct:+.2f}% from MA)")
    print(f"   Last 3 candles: {mn_bulls} BULL, {mn_bears} BEAR")
    
    if mn_trend == 'BULLISH':
        mn_bias = 'LONG ONLY'
    else:
        mn_bias = 'SHORT ONLY'
    print(f"   >>> Monthly Bias: {mn_bias}")
    print()
    
    # ============ WEEKLY ============
    print('2. WEEKLY (Swing Trend)')
    print('-' * 70)
    w1_rsi = get_rsi(w1['c'].values)
    w1_trend, w1_pct = get_trend(w1['c'].values, 10)
    w1_bulls, w1_bears = count_candles(w1['o'].values, w1['c'].values, 4)
    w1_high = w1['h'].values[-4:].max()
    w1_low = w1['l'].values[-4:].min()
    
    print(f"   RSI: {w1_rsi:.1f}")
    print(f"   Trend: {w1_trend} ({w1_pct:+.2f}% from MA)")
    print(f"   Last 4 candles: {w1_bulls} BULL, {w1_bears} BEAR")
    print(f"   4-Week Range: {w1_low:.5f} - {w1_high:.5f}")
    
    if w1_rsi > 55:
        w1_bias = 'LONG'
    elif w1_rsi < 45:
        w1_bias = 'SHORT'
    else:
        w1_bias = 'NEUTRAL'
    print(f"   >>> Weekly Bias: {w1_bias}")
    print()
    
    # ============ DAILY ============
    print('3. DAILY (Position Trend)')
    print('-' * 70)
    d1_rsi = get_rsi(d1['c'].values)
    d1_trend, d1_pct = get_trend(d1['c'].values, 20)
    d1_bulls, d1_bears = count_candles(d1['o'].values, d1['c'].values, 5)
    
    print(f"   RSI: {d1_rsi:.1f}", end='')
    if d1_rsi < 30:
        print(" <<< OVERSOLD")
    elif d1_rsi > 70:
        print(" <<< OVERBOUGHT")
    else:
        print()
    print(f"   Trend: {d1_trend} ({d1_pct:+.2f}% from MA)")
    print(f"   Last 5 candles: {d1_bulls} BULL, {d1_bears} BEAR")
    print()
    
    # ============ H4 ============
    print('4. H4 (Entry Timeframe)')
    print('-' * 70)
    h4_rsi = get_rsi(h4['c'].values)
    h4_trend, h4_pct = get_trend(h4['c'].values, 20)
    
    print(f"   RSI: {h4_rsi:.1f}", end='')
    if h4_rsi < 30:
        print(" <<< OVERSOLD")
    elif h4_rsi > 70:
        print(" <<< OVERBOUGHT")
    else:
        print()
    print(f"   Trend: {h4_trend}")
    print()
    
    # ============ H1 ============
    print('5. H1 (Timing)')
    print('-' * 70)
    h1_rsi = get_rsi(h1['c'].values)
    
    print(f"   RSI: {h1_rsi:.1f}", end='')
    if h1_rsi < 30:
        print(" <<< OVERSOLD")
    elif h1_rsi > 70:
        print(" <<< OVERBOUGHT")
    else:
        print()
    print()
    
    # ============ DECISION ============
    print('=' * 70)
    print('TRADE DECISION')
    print('=' * 70)
    
    # Alignment check
    aligned = False
    direction = None
    
    # For LONG
    if mn_trend == 'BULLISH' and w1_rsi > 45:
        if d1_rsi < 40 or h4_rsi < 40:
            aligned = True
            direction = 'LONG'
            print("✓ WITH TREND LONG: Monthly bullish, Weekly supportive, Lower TF oversold")
    
    # For SHORT
    elif mn_trend == 'BEARISH' and w1_rsi < 55:
        if d1_rsi > 60 or h4_rsi > 60:
            aligned = True
            direction = 'SHORT'
            print("✓ WITH TREND SHORT: Monthly bearish, Weekly supportive, Lower TF overbought")
    
    # Counter-trend
    if not aligned:
        if d1_rsi < 25 and h1_rsi < 25:
            print("⚠ COUNTER-TREND LONG: Extreme oversold but AGAINST monthly/weekly trend")
            print("  -> Use SMALL size (0.05 lots max)")
            print("  -> Tight stop loss")
            direction = 'COUNTER-LONG'
        elif d1_rsi > 75 and h1_rsi > 75:
            print("⚠ COUNTER-TREND SHORT: Extreme overbought but AGAINST monthly/weekly trend")
            print("  -> Use SMALL size (0.05 lots max)")
            print("  -> Tight stop loss")
            direction = 'COUNTER-SHORT'
        else:
            print("✗ NO TRADE: Not aligned with higher timeframes")
            print("  -> Monthly:", mn_trend)
            print("  -> Weekly RSI:", f"{w1_rsi:.1f}")
            print("  -> Daily RSI:", f"{d1_rsi:.1f}")
            direction = 'SKIP'
    
    print()
    print('POSITION SIZE RECOMMENDATION:')
    if direction in ['LONG', 'SHORT']:
        print("  -> WITH TREND: 0.2 - 0.4 lots")
    elif direction in ['COUNTER-LONG', 'COUNTER-SHORT']:
        print("  -> COUNTER-TREND: 0.05 lots MAX")
    else:
        print("  -> NO ENTRY")
    
    return direction

if __name__ == '__main__':
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = 'USDCHF'
    
    analyze_instrument(symbol)
