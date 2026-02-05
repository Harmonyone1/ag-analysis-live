# Trading Logic

## Overview

The AG Analyzer bot uses a multi-stage pipeline to identify and execute trades:

```
Market Data → Analysis → Scoring → AI Gate → Execution
```

## Trading Symbols

Currently configured for **27 forex pairs**:

**Majors:** EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD

**Crosses:** EURGBP, EURJPY, GBPJPY, AUDJPY, EURAUD, EURNZD, GBPAUD, GBPNZD, AUDNZD, AUDCAD, NZDCAD, CADJPY, CHFJPY, EURCAD, EURCHF, GBPCAD, GBPCHF, AUDCHF, NZDCHF, CADCHF

**Note:** Crypto (ETHUSD, BTCUSD) and indices are NOT included - the model was trained only on forex data.

## Pipeline Stages

### 1. Market Data Ingestion

Every scan cycle (~2-3 minutes), the bot:
- Fetches M15 candles (200 bars) for all symbols
- Fetches H1 candles (100 bars) - cached with TTL
- Fetches H4 candles (50 bars) - cached with TTL

### 2. Market Analysis

The `MarketAnalyzer` evaluates each symbol for:

- **Structure:** Trend direction, swing highs/lows, BOS/CHoCH
- **Strength:** Momentum, RSI divergences, volume analysis
- **Liquidity:** Order blocks, fair value gaps, liquidity pools
- **Events:** Key level tests, breakout confirmations

Output: `MarketView` object with all analysis data

### 3. Confluence Scoring

The `ConfluenceScorer` generates trade setups:

- Identifies entry zones based on structure
- Calculates stop loss (invalidation level)
- Sets take profit targets (typically 2-3 levels)
- Computes confluence score (0-100)

**Minimum confluence score:** 65 (configurable)

### 4. AI Decision Gate

The AI gate is the critical filter that determines whether to take a trade.

**Model:** Trained classifier + regressor on historical forex data

**Inputs:**
- M15, H1, H4 candle data (OHLCV)
- Confluence score and reason codes
- Recent trade history (last 50 closed trades)
- Time features (hour sin/cos for session awareness)

**Outputs:**
- `P(win)` - Probability of winning trade
- `E[R]` - Expected R-multiple
- `P(timeout)` - Probability of timeout/scratch

**Thresholds:**
- `P(win) >= 55%` - Required to approve
- `E[R] >= 0.02` - Minimum expected return
- Must pass both to execute

**Decisions:**
- `APPROVED` - Trade meets all criteria
- `REJECTED` - Failed thresholds
- `NEEDS_REVIEW` - Borderline (not auto-executed)

### 5. Trade Execution

For approved candidates:

**Pre-execution filters:**
1. No duplicate positions (same symbol already open)
2. Cooldown after loss (4-hour wait per symbol)
3. Risk limits (max positions, max exposure)
4. Stop distance validation (minimum 20 pips)
5. R:R validation (minimum 1.5:1)

**Execution:**
- Market order with attached SL/TP
- Default position size: 0.10 lots
- Stop loss: From invalidation level
- Take profit: First TP target

**Post-execution:**
- Position tracked in database
- Reconciliation with broker every cycle
- Orphaned positions detected and logged

## Risk Management

### Per-Trade Risk
- Max risk per trade: 1% of account balance
- Position sizing based on stop distance

### Portfolio Risk
- Max concurrent positions: 5
- Max exposure per currency: 2 positions
- Daily loss limit: 3% of account

### Cooldowns
- 4-hour cooldown per symbol after a loss
- Prevents revenge trading on same pair

## Position Management

### Stop Loss
- Set at invalidation level from analysis
- Minimum 20 pips distance enforced

### Take Profit
- Multiple TP targets supported
- Default uses first TP level
- Partial closes possible (configurable)

### Trailing Stop
- Not currently enabled by default
- Can be configured in settings

## Logging

All trading activity is logged with prefixes:

- `[AI_GATE]` - AI decision with probabilities
- `[EXEC]` - Execution events (OPENED, CLOSED, SKIP, FAILED)
- `[EXEC_MODE]` - Current execution mode

Example log output:
```
[AI_GATE] EURUSD LONG: P(win)=0.6234 E[R]=0.0521 P(timeout)=0.1200 → APPROVED
[EXEC] OPENED EURUSD LONG qty=0.10 @ 1.08542 SL=1.08142 TP=1.09342 broker_id=123456
```
