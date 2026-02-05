# AI Trading Instructions for AG-Analysis System

## Purpose
This document provides complete instructions for any AI system managing the AG-Analysis forex/crypto trading system. Follow these instructions to maintain consistent, profitable trading using the ICT + Quantitative framework.

---

## System Overview

### Account Details
- **Broker:** TradeLocker (HEROFX)
- **Account ID:** 592535
- **Environment:** https://live.tradelocker.com
- **Username:** porterd98@gmail.com
- **Server:** HEROFX

### Key Files
| File | Purpose |
|------|---------|
| `D:/AG-Analysis/engine/quick_status.py` | Quick account/position status check |
| `D:/AG-Analysis/engine/claude_trading.py` | Trade execution (analyze, trade, close) |
| `D:/AG-Analysis/engine/ICT_MASTER_GUIDE.md` | Complete ICT + Quant strategy reference |
| `D:/AG-Analysis/engine/smc_scanner.py` | SMC analysis scanner |
| `D:/AG-Analysis/engine/scan_additional.py` | Additional instruments scanner |

---

## Quick Start Checklist

### 1. Check Current Time & Session
```python
# Determine which Kill Zone you're in (EST/New York Time)
# Asian:       7:00 PM - 10:00 PM EST
# London:      2:00 AM - 5:00 AM EST  (BEST for EUR/GBP/CHF)
# New York:    8:00 AM - 11:00 AM EST (BEST for USD pairs, indices)
# London Close: 11:00 AM - 1:00 PM EST
```

### 2. Check Account Status
```bash
cd D:/AG-Analysis/engine && /c/Users/DavidPorter/miniconda3/python.exe quick_status.py
```

### 3. Read the Master Guide
Always reference `ICT_MASTER_GUIDE.md` for strategy details before making decisions.

---

## Core Trading Rules

### RULE 1: Only Trade WITH the HTF Bias
- **HTF (4H/Daily)** determines overall direction
- **NEVER** place trades against the HTF bias
- If HTF is BULLISH, only look for LONG setups
- If HTF is BEARISH, only look for SHORT setups

### RULE 2: Require Minimum 3 Confluences
Before any trade, confirm at least 3 of:
- [ ] HTF bias alignment
- [ ] Kill zone timing
- [ ] Liquidity sweep completed
- [ ] FVG (Fair Value Gap) present
- [ ] Order block present
- [ ] OTE zone (62%-79% Fib)
- [ ] MSS/CHoCH confirmation
- [ ] RSI divergence

### RULE 3: Position Sizing by Instrument
| Instrument | Max Lot Size | Notes |
|------------|--------------|-------|
| Major Forex | 0.1 | Standard pairs |
| Cross Forex | 0.05-0.1 | Higher spreads |
| XAUUSD (Gold) | 0.01-0.02 | High margin, volatile |
| Indices | 0.01-0.05 | Very high margin |
| Crypto (BTC/ETH) | 0.01-0.05 | Extreme volatility |

### RULE 4: Risk Management
- Never risk more than 2% per trade
- Maximum 3 correlated positions at once
- Maximum 5% total account risk
- Use ATR-based stops (2-3x ATR)

---

## Trading Workflow

### Step 1: Market Analysis
```bash
# Run ICT+Quant analysis
cd D:/AG-Analysis/engine && /c/Users/DavidPorter/miniconda3/python.exe -c "
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from tradelocker import TLAPI
import numpy as np

api = TLAPI(
    environment='https://live.tradelocker.com',
    username='porterd98@gmail.com',
    password='Amilli@1021!',
    server='HEROFX',
    account_id=592535,
)

instruments = api.get_all_instruments()
symbol_to_id = dict(zip(instruments['name'], instruments['tradableInstrumentId']))

def analyze(symbol):
    inst_id = symbol_to_id.get(symbol)
    if not inst_id:
        return None
    try:
        h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='1W')
        h1 = api.get_price_history(inst_id, resolution='1H', lookback_period='3D')

        if h4 is None or h1 is None or len(h4) < 5 or len(h1) < 15:
            return None

        h4c = h4['c'].values
        h1c = h1['c'].values
        h1h = h1['h'].values
        h1l = h1['l'].values

        current = h1c[-1]
        htf_bias = 'BULLISH' if h4c[-1] > h4c[-5] else 'BEARISH'

        # RSI 14
        deltas = np.diff(h1c[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.0001
        rsi = 100 - (100 / (1 + avg_gain/avg_loss))

        # ATR 14
        tr_list = []
        for i in range(-14, 0):
            tr = max(h1h[i] - h1l[i], abs(h1h[i] - h1c[i-1]), abs(h1l[i] - h1c[i-1]))
            tr_list.append(tr)
        atr = np.mean(tr_list)

        return {'current': current, 'htf': htf_bias, 'rsi': round(rsi,1), 'atr': atr}
    except:
        return None

# Analyze key pairs
pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD', 'ETHUSD']
for sym in pairs:
    r = analyze(sym)
    if r:
        print(f'{sym}: {r[\"current\"]:.5f} | HTF: {r[\"htf\"]} | RSI: {r[\"rsi\"]}')
"
```

### Step 2: Identify Setups
Look for setups that match the framework:

**BULLISH Setup Criteria:**
- HTF bias = BULLISH
- RSI < 40 (pullback/oversold)
- Price in discount zone (below 50% of range)
- Bullish FVG or Order Block present
- Recent liquidity sweep of lows

**BEARISH Setup Criteria:**
- HTF bias = BEARISH
- RSI > 60 (elevated/overbought)
- Price in premium zone (above 50% of range)
- Bearish FVG or Order Block present
- Recent liquidity sweep of highs

### Step 3: Execute Trade
```bash
# Example: Place a LONG trade on EURUSD
cd D:/AG-Analysis/engine && /c/Users/DavidPorter/miniconda3/python.exe claude_trading.py \
    --action trade \
    --symbol EURUSD \
    --direction LONG \
    --lots 0.1 \
    --stop-pips 30 \
    --tp-pips 60
```

### Step 4: Monitor Positions
```bash
# Continuous monitoring
cd D:/AG-Analysis/engine && /c/Users/DavidPorter/miniconda3/python.exe -c "
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from tradelocker import TLAPI
import time
from datetime import datetime

api = TLAPI(
    environment='https://live.tradelocker.com',
    username='porterd98@gmail.com',
    password='Amilli@1021!',
    server='HEROFX',
    account_id=592535,
)

instruments = api.get_all_instruments()
id_to_symbol = dict(zip(instruments['tradableInstrumentId'], instruments['name']))

for cycle in range(10):
    now = datetime.now().strftime('%H:%M:%S')
    state = api.get_account_state()
    equity = state['balance'] + state['openGrossPnL']

    positions = api.get_all_positions()
    pos_str = []
    if not positions.empty:
        for _, p in positions.iterrows():
            sym = id_to_symbol.get(p['tradableInstrumentId'], '?')
            pnl = p['unrealizedPl']
            ind = '+' if pnl > 0 else '-'
            pos_str.append(f'{sym}:{ind}\${abs(pnl):.2f}')

    print(f'[{now}] Eq:\${equity:.2f} | {\" | \".join(pos_str)}')
    time.sleep(30)
"
```

---

## Order Management

### Place Limit Order
```python
# Using direct API for limit orders
from tradelocker import TLAPI
from decimal import Decimal

api = TLAPI(
    environment='https://live.tradelocker.com',
    username='porterd98@gmail.com',
    password='Amilli@1021!',
    server='HEROFX',
    account_id=592535,
)

# Get instrument ID
instruments = api.get_all_instruments()
inst_id = instruments[instruments['name'] == 'EURUSD']['tradableInstrumentId'].values[0]

# Place limit order
api.create_order(
    instrument_id=inst_id,
    quantity=0.1,
    side='buy',  # or 'sell'
    type_='limit',
    price=1.0500,  # limit price
    stop_loss=1.0450,
    take_profit=1.0600
)
```

### Cancel Order
```python
# Get order ID first
orders = api.get_all_orders()
order_id = orders[orders['tradableInstrumentId'] == inst_id]['id'].values[0]

# Cancel
api.delete_order(order_id)
```

### Close Position
```bash
cd D:/AG-Analysis/engine && /c/Users/DavidPorter/miniconda3/python.exe claude_trading.py \
    --action close \
    --position-id <POSITION_ID>
```

---

## Decision Framework

### When to ENTER a Trade
1. Confirm HTF bias (4H chart)
2. Wait for Kill Zone timing
3. Look for liquidity sweep
4. Identify entry zone (FVG, OB, or OTE)
5. Wait for LTF confirmation (15m structure shift)
6. Calculate position size based on ATR stop

### When to HOLD a Position
- Position is aligned with HTF bias
- Price hasn't reached target liquidity
- No change of character (CHoCH) on MTF
- Stop loss hasn't been hit

### When to EXIT a Position
- Take profit hit
- Stop loss hit
- HTF bias changes (CHoCH on 4H)
- Better opportunity elsewhere
- Major news event approaching

### When to CANCEL a Pending Order
- HTF bias changes
- Price structure invalidates setup
- Order becomes counter-trend
- Better entry available

---

## Risk Scenarios

### If Position is in Heavy Drawdown
1. Check if HTF bias still aligns
2. If aligned: HOLD - trust the framework
3. If counter-trend: Consider reducing/closing
4. Never add to losing positions without plan

### If Multiple Positions Correlate
Example: Long EURUSD + Long GBPUSD = 2x USD short exposure
- Reduce one position
- Or ensure combined risk < 3%

### If Account Drops 10%+
1. Close all positions
2. Step back and reassess
3. Wait for clear A+ setup
4. Reduce position sizes by 50%

---

## Session-Specific Strategies

### Asian Session (7PM-10PM EST)
- Expect ranging/consolidation
- Focus on AUD, NZD, JPY pairs
- Scalp targets: 15-20 pips
- This session creates the range London will hunt

### London Session (2AM-5AM EST)
- Highest probability setups
- Look for Asian range liquidity sweeps
- EUR, GBP, CHF most active
- Often creates daily high or low

### New York Session (8AM-11AM EST)
- Overlap with London = highest volume
- OTE setups between 8:30-11:00 AM
- USD pairs and indices most active
- Distribution phase of daily move

### London Close (11AM-1PM EST)
- Reversal opportunities
- Position unwinding
- Lower volume
- Good for counter-trend scalps

---

## API Reference

### Get Account State
```python
state = api.get_account_state()
balance = state['balance']
open_pnl = state['openGrossPnL']
equity = balance + open_pnl
```

### Get Positions
```python
positions = api.get_all_positions()
for _, p in positions.iterrows():
    symbol = id_to_symbol.get(p['tradableInstrumentId'])
    pnl = p['unrealizedPl']
    side = p['side']
    qty = p['qty']
    entry = p['avgPrice']
```

### Get Orders
```python
orders = api.get_all_orders()
limit_orders = orders[(orders['type'] == 'limit') & (orders['positionId'] == 0)]
```

### Get Price History
```python
# Resolutions: '1M', '1W', '1D', '4H', '1H', '30m', '15m', '5m', '1m'
# Lookback: '1W', '3D', '1D', etc.
h4 = api.get_price_history(inst_id, resolution='4H', lookback_period='1W')
closes = h4['c'].values
highs = h4['h'].values
lows = h4['l'].values
```

### Get Current Price
```python
current_price = api.get_latest_asking_price(inst_id)
```

---

## Pip/Point Sizes

| Instrument | Pip Size | Notes |
|------------|----------|-------|
| Standard Forex | 0.0001 | EURUSD, GBPUSD, etc. |
| JPY Pairs | 0.01 | USDJPY, EURJPY, etc. |
| XAUUSD (Gold) | 1.0 | Points, not pips |
| BTCUSD | 1.0 | Points |
| ETHUSD | 1.0 | Points |
| Indices | 1.0 | US30, NAS100, SPX500 |

---

## Troubleshooting

### Connection Issues
```python
# If API fails, create new instance
api = TLAPI(
    environment='https://live.tradelocker.com',
    username='porterd98@gmail.com',
    password='Amilli@1021!',
    server='HEROFX',
    account_id=592535,
)
```

### Wrong Account Data
- Ensure account_id=592535 is correct
- Check server='HEROFX'
- Verify credentials

### Order Rejected
- Check margin availability
- Verify lot size is valid for instrument
- Ensure stop/take profit prices are valid
- Check if market is open

---

## Performance Tracking

### After Each Session, Record:
1. Starting balance
2. Ending balance
3. Number of trades
4. Win/Loss count
5. Best trade
6. Worst trade
7. Notes on market conditions

### Weekly Review:
1. Win rate (target: 55%+)
2. Average R:R achieved
3. Biggest drawdown
4. Framework adherence
5. Lessons learned

---

## Important Reminders

1. **Trust the HTF bias** - It's the foundation of the strategy
2. **Be patient** - Wait for Kill Zone timing
3. **Don't overtrade** - 2-3 quality trades > 10 mediocre trades
4. **Cut counter-trend orders** - Cancel immediately if bias changes
5. **Let winners run** - Use trailing stops or target liquidity
6. **Accept losses** - Part of trading, move on
7. **Stay time-aware** - Know which session you're in
8. **Reference the guide** - ICT_MASTER_GUIDE.md has all details

---

## Contact & Resources

- **ICT Strategy Guide:** `D:/AG-Analysis/engine/ICT_MASTER_GUIDE.md`
- **Quick Status:** `D:/AG-Analysis/engine/quick_status.py`
- **Trade Execution:** `D:/AG-Analysis/engine/claude_trading.py`

---

## Version History
- **v1.0** - December 18, 2025: Initial AI instructions created
- Proven results: $288.03 -> $350.70 (+21.8%) in single session
