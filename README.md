# AG Analyzer - Live Trading Bot

AI-driven forex trading bot built for TradeLocker.

## What It Does

1. **Scans 27 forex pairs** every ~2.5 minutes
2. **Analyzes market structure** (trend, liquidity, momentum)
3. **Generates trade setups** with confluence scoring
4. **AI gate filters** - Only takes trades with P(win) > 55% and positive E[R]
5. **Executes automatically** on TradeLocker with SL/TP
6. **Manages risk** - Position limits, cooldowns, exposure caps

## Current Status

- **Live on production server** (DigitalOcean)
- **Trading forex only** (28 pairs)
- **Not trading crypto/indices** (model not trained on these)

## Quick Start

```bash
# Clone
git clone https://github.com/Harmonyone1/ag-analysis-live.git
cd ag-analysis-live

# Configure
cp .env.example .env
# Edit .env with your TradeLocker credentials

# Deploy
docker-compose up -d

# Monitor
docker logs -f ag_analyzer_engine
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Trading Engine                          │
├─────────────────────────────────────────────────────────────┤
│  Market Data → Analysis → Scoring → AI Gate → Execution     │
│       ↓            ↓          ↓         ↓          ↓        │
│   TradeLocker   Structure  Confluence  ML Model  Orders     │
│   M15/H1/H4     Liquidity   Score      P(win)    SL/TP      │
│                 Momentum    Reasons    E[R]                  │
└─────────────────────────────────────────────────────────────┘
```

## Documentation

- **[Deployment Guide](docs/DEPLOYMENT.md)** - Server setup and deployment
- **[Trading Logic](docs/TRADING_LOGIC.md)** - How the bot makes decisions
- **[Configuration](docs/CONFIGURATION.md)** - All settings and options
- **[Architecture](docs/ARCHITECTURE.md)** - System design

## Key Files

```
ag-analysis-live/
├── engine/
│   ├── src/
│   │   ├── main.py           # Main trading loop
│   │   ├── ai/gate.py        # AI decision gate
│   │   ├── analysis/         # Market analysis
│   │   ├── scoring/          # Confluence scoring
│   │   ├── execution/        # Trade execution
│   │   └── adapters/         # TradeLocker API
│   └── Dockerfile
├── backend/                   # REST API
├── models/                    # AI model files
├── docker-compose.yml
└── .env                       # Configuration
```

## Monitoring

### View Live Trades
```bash
docker logs ag_analyzer_engine 2>&1 | grep "\[EXEC\]"
```

### Check AI Decisions
```bash
docker logs ag_analyzer_engine 2>&1 | grep "\[AI_GATE\]"
```

### Account Status
```bash
docker exec ag_analyzer_engine python3 -c "
from tradelocker import TLAPI
import os
tl = TLAPI(
    environment=os.environ.get('TL_ENVIRONMENT'),
    username=os.environ.get('TL_EMAIL'),
    password=os.environ.get('TL_PASSWORD'),
    server=os.environ.get('TL_SERVER'),
    acc_num=1
)
state = tl.get_account_state()
print(f'Balance: \${state[\"balance\"]:.2f}')
print(f'Today P&L: \${state[\"todayNet\"]:.2f}')
"
```

## Log Examples

```
[AI_GATE] EURUSD LONG: P(win)=0.6234 E[R]=0.0521 → APPROVED
[EXEC] OPENED EURUSD LONG qty=0.10 @ 1.08542 SL=1.08142 TP=1.09342
[EXEC] SKIP GBPUSD: cooldown after loss (closed 07:26 UTC, pnl=-12.50)
[AI_GATE] AUDJPY SHORT: P(win)=0.4521 E[R]=-0.1200 → REJECTED
```

## Risk Controls

- **Max 1% risk per trade**
- **Max 5 concurrent positions**
- **4-hour cooldown after loss** (per symbol)
- **Minimum 20 pip stop distance**
- **Minimum 1.5:1 risk:reward**

## Adding Crypto/Indices

The model was trained on forex only. To add ETHUSD, BTCUSD, or indices:

1. Collect training data for those instruments
2. Retrain the AI model
3. Add symbols to `SYMBOLS` list in `main.py`
4. Redeploy

## License

Private repository - not for redistribution.
