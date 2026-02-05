# AG ANALYZER - Technical Architecture

## System Overview

AG ANALYZER is a multi-component trading system designed for analyzing and executing forex trades through the TradeLocker platform.

## Component Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              AG ANALYZER                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐    │
│    │                      PRESENTATION LAYER                          │    │
│    │                                                                   │    │
│    │    ┌─────────────┐          ┌─────────────┐                     │    │
│    │    │  Next.js UI │          │ Claude Code │                     │    │
│    │    │  Dashboard  │          │     CLI     │                     │    │
│    │    └──────┬──────┘          └──────┬──────┘                     │    │
│    │           │                        │                             │    │
│    └───────────┼────────────────────────┼─────────────────────────────┘    │
│                │                        │                                   │
│    ┌───────────┼────────────────────────┼─────────────────────────────┐    │
│    │           ▼          API LAYER     ▼                             │    │
│    │    ┌─────────────────────────────────────┐                      │    │
│    │    │           FastAPI Backend           │                      │    │
│    │    │                                      │                      │    │
│    │    │  • REST Endpoints                   │                      │    │
│    │    │  • WebSocket Server                 │                      │    │
│    │    │  • Authentication                   │                      │    │
│    │    │  • Request Validation               │                      │    │
│    │    └──────────────────┬──────────────────┘                      │    │
│    │                       │                                          │    │
│    └───────────────────────┼──────────────────────────────────────────┘    │
│                            │                                                │
│    ┌───────────────────────┼──────────────────────────────────────────┐    │
│    │                       ▼       BUSINESS LAYER                     │    │
│    │    ┌─────────────────────────────────────────────────────┐      │    │
│    │    │              Trading Engine (Python)                 │      │    │
│    │    │                                                       │      │    │
│    │    │  ┌───────────┐  ┌───────────┐  ┌───────────┐       │      │    │
│    │    │  │ Analysis  │  │    AI     │  │   Risk    │       │      │    │
│    │    │  │  Module   │  │   Gate    │  │  Manager  │       │      │    │
│    │    │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘       │      │    │
│    │    │        │              │              │              │      │    │
│    │    │        └──────────────┼──────────────┘              │      │    │
│    │    │                       ▼                             │      │    │
│    │    │              ┌───────────────┐                     │      │    │
│    │    │              │   Execution   │                     │      │    │
│    │    │              │    Engine     │                     │      │    │
│    │    │              └───────┬───────┘                     │      │    │
│    │    │                      │                              │      │    │
│    │    └──────────────────────┼──────────────────────────────┘      │    │
│    │                           │                                      │    │
│    └───────────────────────────┼──────────────────────────────────────┘    │
│                                │                                            │
│    ┌───────────────────────────┼──────────────────────────────────────┐    │
│    │                           ▼       DATA LAYER                     │    │
│    │    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │    │
│    │    │  PostgreSQL │  │ TradeLocker │  │   Claude    │           │    │
│    │    │  Database   │  │     API     │  │    API      │           │    │
│    │    └─────────────┘  └─────────────┘  └─────────────┘           │    │
│    │                                                                  │    │
│    └──────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Trading Engine (`/engine`)

The core Python application responsible for market analysis and trade execution.

#### Directory Structure

```
engine/
├── src/
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── broker.py         # Base broker interface
│   │   └── tradelocker.py    # TradeLocker implementation
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── features.py       # Feature extraction
│   │   ├── gate.py           # AI decision gate
│   │   ├── backtest.py       # Backtesting engine
│   │   ├── label_engine.py   # Trade labeling
│   │   └── training_pipeline.py  # Model training
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── trade_manager.py  # Trade management
│   │   └── advanced_manager.py  # Advanced order types
│   ├── risk/
│   │   ├── __init__.py
│   │   └── manager.py        # Risk management
│   ├── scoring/
│   │   ├── __init__.py
│   │   └── confluence.py     # Trade scoring
│   ├── config.py             # Configuration management
│   └── main.py               # Entry point
├── claude_trading.py         # Claude CLI interface
├── run_live.py               # Live trading runner
├── run_paper_trading.py      # Paper trading mode
├── requirements.txt
└── Dockerfile
```

#### Key Classes

```python
# TradeLocker Adapter
class TradeLockerAdapter:
    def connect() -> bool
    def disconnect() -> None
    def get_account() -> Account
    def get_candles(symbol, timeframe, limit) -> List[Candle]
    def get_instrument(symbol) -> Instrument
    def place_order(request: OrderRequest) -> Order
    def close_position(position_id) -> bool
    def list_positions() -> List[Position]

# Risk Manager
class RiskManager:
    def check_trade(trade: TradeSignal) -> bool
    def calculate_position_size(account, risk_pct, stop_pips) -> Decimal
    def check_correlation(symbol, positions) -> bool
    def check_daily_loss(account, trades) -> bool

# AI Gate
class AIGate:
    def evaluate(features: dict) -> float  # 0-1 probability
    def extract_features(candles) -> dict
```

### 2. Backend API (`/backend`)

FastAPI-based REST API and WebSocket server.

#### Directory Structure

```
backend/
├── src/
│   ├── services/
│   │   ├── analysis_service.py
│   │   ├── trading_service.py
│   │   └── risk_service.py
│   ├── database/
│   │   ├── models.py
│   │   └── connection.py
│   └── main.py
├── requirements.txt
└── Dockerfile
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/account` | Get account info |
| GET | `/api/positions` | List positions |
| POST | `/api/scan` | Scan market |
| POST | `/api/analyze/{symbol}` | Analyze pair |
| POST | `/api/trade` | Execute trade |
| DELETE | `/api/position/{id}` | Close position |
| WS | `/ws` | Real-time updates |

### 3. User Interface (`/ui`)

Next.js dashboard for monitoring and control.

#### Directory Structure

```
ui/
├── src/
│   ├── app/
│   │   ├── page.tsx          # Dashboard
│   │   ├── positions/        # Position management
│   │   ├── analytics/        # Performance analytics
│   │   └── settings/         # Configuration
│   ├── components/
│   │   ├── Chart.tsx
│   │   ├── PositionCard.tsx
│   │   └── TradeForm.tsx
│   └── lib/
│       └── api.ts            # API client
├── package.json
└── Dockerfile
```

### 4. Database Schema

```sql
-- Core tables for trade tracking and analytics

CREATE TABLE accounts (
    id UUID PRIMARY KEY,
    broker VARCHAR(50) NOT NULL,
    account_id VARCHAR(100) NOT NULL,
    account_name VARCHAR(255),
    balance DECIMAL(15,2),
    equity DECIMAL(15,2),
    last_sync TIMESTAMP
);

CREATE TABLE trades (
    id UUID PRIMARY KEY,
    account_id UUID REFERENCES accounts(id),
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    entry_price DECIMAL(15,5),
    exit_price DECIMAL(15,5),
    quantity DECIMAL(10,2),
    stop_loss DECIMAL(15,5),
    take_profit DECIMAL(15,5),
    pnl DECIMAL(15,2),
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    status VARCHAR(20),
    strategy VARCHAR(100),
    confluence_score INTEGER,
    ai_probability DECIMAL(5,4)
);

CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(15,5),
    high DECIMAL(15,5),
    low DECIMAL(15,5),
    close DECIMAL(15,5),
    volume DECIMAL(20,2),
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE TABLE signals (
    id UUID PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    entry_price DECIMAL(15,5),
    stop_loss DECIMAL(15,5),
    take_profit DECIMAL(15,5),
    confluence_score INTEGER,
    ai_probability DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT NOW(),
    executed BOOLEAN DEFAULT FALSE
);
```

## Data Flow

### Trade Execution Flow

```
1. Market Scan Request
   │
   ▼
2. Fetch Candle Data (TradeLocker API)
   │
   ▼
3. Technical Analysis
   ├── Calculate SMA 20/50
   ├── Calculate RSI 14
   ├── Calculate ATR 14
   └── Determine Price Position
   │
   ▼
4. AI Gate Evaluation
   ├── Extract Features
   ├── Model Inference
   └── Return Probability Score
   │
   ▼
5. Risk Check
   ├── Position Size Calculation
   ├── Correlation Check
   ├── Daily Loss Limit Check
   └── Max Positions Check
   │
   ▼
6. Order Execution
   ├── Create OrderRequest
   ├── Submit to TradeLocker
   └── Log Trade
   │
   ▼
7. Position Monitoring
   ├── Track P&L
   ├── Update Stop/TP if needed
   └── Close on Target/Stop
```

### Analysis Pipeline

```python
def analyze_pair(symbol: str) -> dict:
    # 1. Fetch data
    candles = broker.get_candles(symbol, "M15", limit=100)

    # 2. Calculate indicators
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-50:])
    rsi = calculate_rsi(closes, 14)
    atr = calculate_atr(highs, lows, closes, 14)

    # 3. Determine trend and position
    trend = "BULLISH" if sma20 > sma50 else "BEARISH"
    price_position = (closes[-1] - min(lows[-20:])) / (max(highs[-20:]) - min(lows[-20:]))

    # 4. Return analysis
    return {
        "symbol": symbol,
        "current_price": closes[-1],
        "sma20": sma20,
        "sma50": sma50,
        "rsi": rsi,
        "atr": atr,
        "trend": trend,
        "price_position": price_position
    }
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db

# TradeLocker
TL_ENVIRONMENT=https://live.tradelocker.com
TL_EMAIL=your@email.com
TL_PASSWORD=your_password
TL_SERVER=HEROFX
TL_ACC_NUM=123456

# Trading Parameters
BOT_MODE=paper  # paper or live
MAX_RISK_PER_TRADE=0.01
MAX_DAILY_LOSS=0.03
MAX_OPEN_POSITIONS=5

# AI Thresholds
MIN_CONFLUENCE_SCORE=60
MIN_AI_PROBABILITY=0.55

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: ag_analyzer
      POSTGRES_USER: ag_user
      POSTGRES_PASSWORD: ag_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  engine:
    build: ./engine
    depends_on:
      - db
    environment:
      - DATABASE_URL=postgresql://ag_user:ag_password@db:5432/ag_analyzer
    env_file:
      - .env

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - db
      - engine

  ui:
    build: ./ui
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  postgres_data:
```

## Security Considerations

1. **Credential Storage:** All API credentials encrypted at rest
2. **Authentication:** JWT-based authentication for API
3. **Rate Limiting:** Implemented on all endpoints
4. **Input Validation:** Pydantic models for all requests
5. **Audit Logging:** All trades and actions logged

---

*Document Version: 1.0*
*Last Updated: December 16, 2024*
