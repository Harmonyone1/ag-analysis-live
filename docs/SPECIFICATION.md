# AG ANALYZER - Complete Build Specification

## Decisions & Configuration (Filled In)

This section contains all decisions and configurations that were requested in the original specification.

---

### **Stack Decisions**

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Backend API | **FastAPI (Python)** | Aligns with existing AG1 Python codebase, excellent async support, automatic OpenAPI docs |
| UI Framework | **Next.js 14 + Tailwind + shadcn/ui** | Modern, sleek, excellent DX, recommended in original spec |
| Database | **Supabase (Postgres)** | Managed Postgres with Realtime subscriptions, Auth, and RLS built-in |
| Charts | **TradingView Lightweight Charts** | Industry standard for financial charts |
| ML Framework | **XGBoost/LightGBM** | Robust gradient boosting for tabular data, interpretable with SHAP |

---

### **Strategy Constants (Training Configuration)**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Bias Timeframes** | D1, H4 | Standard multi-timeframe approach for directional bias |
| **Entry Timeframe** | M15 | Provides good balance of precision and noise filtering |
| **Stop Logic** | Structure invalidation + 0.5 ATR buffer | Respects market structure while avoiding stop hunts |
| **Take Profit** | Primary: 1.5R, Secondary: 2.5R (partial) | Positive expectancy with realistic hit rates |
| **Timeout** | 24 M15 bars (6 hours) | Avoids holding stale setups |

---

### **Risk Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_risk_per_trade` | **1.0%** | Maximum account risk per single trade |
| `max_daily_loss` | **3.0%** | Hard daily drawdown limit - halts trading |
| `max_open_positions` | **5** | Maximum concurrent positions |
| `max_correlated_exposure` | **2.0%** | Max combined risk on correlated pairs (e.g., all USD shorts) |
| `max_spread_multiplier` | **3.0x** | Skip if spread > 3x typical spread for instrument |
| `slippage_halt_threshold` | **5 ticks** | Halt if average slippage exceeds threshold over last 5 trades |

---

### **AI Model Thresholds**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `min_confluence_score` | **60** | Minimum deterministic score to generate candidate |
| `min_ai_probability` | **0.55** | Minimum P(win) from AI gate |
| `min_expected_value` | **0.15R** | Minimum expected R-multiple |
| `max_timeout_probability` | **0.40** | Maximum acceptable P(timeout) |
| `ai_confidence_threshold` | **0.60** | Confidence level for full position sizing |

---

### **Instruments (Initial Watchlist)**

#### **FX Majors**
- EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, NZD/USD, USD/CAD

#### **FX Crosses**
- EUR/GBP, EUR/JPY, GBP/JPY, AUD/JPY, EUR/AUD, GBP/AUD

#### **Indices**
- US30 (Dow), US500 (S&P 500), US100 (Nasdaq), GER40 (DAX), UK100 (FTSE)

---

### **News Calendar Integration**

| Provider | Implementation |
|----------|---------------|
| **Primary** | ForexFactory RSS/Scraping or Investing.com API |
| **Fallback** | Manual event windows in `settings` table |
| **High-Impact Buffer** | 30 minutes before/after for USD, 15 minutes for others |

---

### **Session Hours (UTC)**

| Session | Start | End |
|---------|-------|-----|
| Sydney | 21:00 | 06:00 |
| Tokyo | 00:00 | 09:00 |
| London | 07:00 | 16:00 |
| New York | 12:00 | 21:00 |

---

### **Supabase Configuration**

#### **Schema Strategy**
- `public` schema: UI-accessible tables with RLS policies
- `private` schema: Engine-only tables (raw execution data, secrets)

#### **RLS Policies**
| Table | Trader Access | Admin Access |
|-------|--------------|--------------|
| `price_history` | SELECT | SELECT |
| `analysis_snapshot` | SELECT | SELECT |
| `trade_candidates` | SELECT | SELECT, UPDATE (approve/reject) |
| `positions` | SELECT | SELECT |
| `orders` | SELECT | SELECT |
| `executions` | SELECT | SELECT |
| `risk_events` | SELECT | SELECT |
| `bot_state` | SELECT | SELECT, UPDATE |
| `settings` | SELECT | SELECT, UPDATE |
| `model_registry` | SELECT | SELECT, INSERT, UPDATE |

#### **Realtime Subscriptions**
| Page | Subscribed Tables |
|------|-------------------|
| Dashboard | `bot_state`, `positions`, `risk_events`, `trade_candidates` |
| Scanner | `trade_candidates`, `analysis_snapshot` |
| Symbol Analysis | `analysis_snapshot`, `trade_candidates` (filtered by symbol) |
| Positions | `positions`, `orders`, `executions` |

---

## Complete Database Schema

### **Existing Tables (from AG1)**

```sql
-- price_history (existing)
CREATE TABLE price_history (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    bar_time TIMESTAMPTZ NOT NULL,
    open DECIMAL(18, 8) NOT NULL,
    high DECIMAL(18, 8) NOT NULL,
    low DECIMAL(18, 8) NOT NULL,
    close DECIMAL(18, 8) NOT NULL,
    volume DECIMAL(18, 2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timeframe, bar_time)
);

-- orders (existing - extended)
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    broker_order_id VARCHAR(100),
    candidate_id UUID REFERENCES trade_candidates(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 8),
    stop_loss DECIMAL(18, 8),
    take_profit DECIMAL(18, 8),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    filled_qty DECIMAL(18, 8) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- trade_log (existing)
CREATE TABLE trade_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size DECIMAL(18, 8) NOT NULL,
    entry_price DECIMAL(18, 8) NOT NULL,
    exit_price DECIMAL(18, 8),
    pnl DECIMAL(18, 2),
    confidence DECIMAL(5, 4),
    atr DECIMAL(18, 8),
    model_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### **New Tables**

```sql
-- instruments
CREATE TABLE instruments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) UNIQUE NOT NULL,
    asset_class VARCHAR(20) NOT NULL, -- FX, INDEX, CRYPTO, COMMODITY
    base_currency VARCHAR(10),
    quote_currency VARCHAR(10),
    pip_size DECIMAL(18, 10) NOT NULL,
    tick_size DECIMAL(18, 10) NOT NULL,
    contract_size DECIMAL(18, 2) DEFAULT 1,
    min_lot DECIMAL(18, 8) NOT NULL,
    max_spread_pips DECIMAL(10, 2),
    session_hours JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- analysis_snapshot
CREATE TABLE analysis_snapshot (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    snapshot_time TIMESTAMPTZ NOT NULL,
    structure_state VARCHAR(20), -- TREND_UP, TREND_DOWN, RANGE, DISTRIBUTION
    trend_direction VARCHAR(10), -- UP, DOWN, NEUTRAL
    strength_scores JSONB, -- {"1D": {"EUR": 0.8, "USD": -0.3}, "5D": {...}}
    liquidity_zones JSONB, -- [{"type": "PDH", "price": 1.0850, "swept": false}]
    momentum_state JSONB, -- {"rsi": 55, "divergence": null, "efficiency": 0.7}
    regime_state JSONB, -- {"volatility": "medium", "risk_sentiment": "on"}
    event_risk JSONB, -- {"next_event": "FOMC", "minutes_until": 120}
    computed_levels JSONB, -- {"invalidation": 1.0800, "entry_zone": [1.0820, 1.0835]}
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- trade_candidates
CREATE TABLE trade_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- LONG, SHORT
    entry_type VARCHAR(20) NOT NULL, -- MARKET, LIMIT
    entry_zone JSONB NOT NULL, -- {"min": 1.0820, "max": 1.0835}
    invalidation_price DECIMAL(18, 8) NOT NULL,
    stop_price DECIMAL(18, 8) NOT NULL,
    tp_targets JSONB NOT NULL, -- [{"r": 1.5, "price": 1.0900}, {"r": 2.5, "price": 1.0950}]
    confluence_score INTEGER NOT NULL, -- 0-100
    reasons TEXT[] NOT NULL,
    reason_codes VARCHAR(50)[] NOT NULL,
    ai_approved BOOLEAN,
    ai_confidence DECIMAL(5, 4),
    ai_expected_r DECIMAL(5, 2),
    status VARCHAR(20) NOT NULL DEFAULT 'new', -- new, queued, placed, expired, rejected, filled
    snapshot_id UUID REFERENCES analysis_snapshot(id),
    feature_vector JSONB,
    feature_set_version VARCHAR(20),
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- executions
CREATE TABLE executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    broker_trade_id VARCHAR(100),
    order_id UUID REFERENCES orders(id),
    fill_price DECIMAL(18, 8) NOT NULL,
    fill_qty DECIMAL(18, 8) NOT NULL,
    fees DECIMAL(18, 4) DEFAULT 0,
    slippage_pips DECIMAL(10, 2),
    fill_time TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- positions
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    broker_position_id VARCHAR(100),
    candidate_id UUID REFERENCES trade_candidates(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    avg_entry_price DECIMAL(18, 8) NOT NULL,
    current_stop_loss DECIMAL(18, 8),
    current_take_profit DECIMAL(18, 8),
    unrealized_pnl DECIMAL(18, 2),
    realized_pnl DECIMAL(18, 2) DEFAULT 0,
    open_time TIMESTAMPTZ NOT NULL,
    close_time TIMESTAMPTZ,
    close_reason VARCHAR(50), -- TP_HIT, SL_HIT, MANUAL, TIMEOUT, RISK_EVENT
    r_multiple DECIMAL(5, 2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- risk_events
CREATE TABLE risk_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL, -- DAILY_LOSS_HIT, SPREAD_FILTER, DATA_STALE, SLIPPAGE_HALT, etc.
    severity VARCHAR(20) NOT NULL, -- INFO, WARNING, CRITICAL
    payload JSONB,
    resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- bot_state
CREATE TABLE bot_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    trading_enabled BOOLEAN DEFAULT false,
    mode VARCHAR(20) DEFAULT 'paper', -- paper, live
    last_heartbeat TIMESTAMPTZ,
    current_model_version VARCHAR(50),
    last_error TEXT,
    daily_pnl DECIMAL(18, 2) DEFAULT 0,
    daily_trades INTEGER DEFAULT 0,
    open_risk_percent DECIMAL(5, 2) DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT single_row CHECK (id = 1)
);

-- model_registry
CREATE TABLE model_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(50) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- gate_classifier, ev_regressor
    feature_set_version VARCHAR(20) NOT NULL,
    training_start DATE NOT NULL,
    training_end DATE NOT NULL,
    metrics JSONB NOT NULL, -- {"auc": 0.72, "precision": 0.65, "ev": 0.18}
    artifact_path TEXT,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- settings
CREATE TABLE settings (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- economic_calendar
CREATE TABLE economic_calendar (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_name VARCHAR(200) NOT NULL,
    currency VARCHAR(10) NOT NULL,
    impact VARCHAR(20) NOT NULL, -- LOW, MEDIUM, HIGH
    event_time TIMESTAMPTZ NOT NULL,
    actual VARCHAR(50),
    forecast VARCHAR(50),
    previous VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### **Indexes**

```sql
CREATE INDEX idx_price_history_lookup ON price_history(symbol, timeframe, bar_time DESC);
CREATE INDEX idx_analysis_snapshot_lookup ON analysis_snapshot(symbol, timeframe, snapshot_time DESC);
CREATE INDEX idx_trade_candidates_status ON trade_candidates(status, created_at DESC);
CREATE INDEX idx_trade_candidates_symbol ON trade_candidates(symbol, created_at DESC);
CREATE INDEX idx_positions_open ON positions(symbol, open_time DESC) WHERE close_time IS NULL;
CREATE INDEX idx_risk_events_recent ON risk_events(created_at DESC);
CREATE INDEX idx_economic_calendar_upcoming ON economic_calendar(event_time) WHERE event_time > NOW();
```

---

## API Contract

### **REST Endpoints**

#### **Market Data**
```
GET  /api/instruments                    # List all tradeable instruments
GET  /api/instruments/{symbol}           # Get instrument details
GET  /api/candles/{symbol}               # Get OHLCV data
     ?timeframe=M15&start=...&end=...
```

#### **Analysis**
```
GET  /api/analysis/{symbol}              # Latest analysis snapshot
GET  /api/strength                       # Currency strength rankings
     ?horizons=1D,5D,20D
GET  /api/scanner                        # Trade candidates
     ?min_score=60&ai_approved=true&direction=LONG
```

#### **Trading**
```
GET  /api/positions                      # Open positions
GET  /api/positions/{id}                 # Position details
GET  /api/orders                         # Order history
GET  /api/executions                     # Execution history
POST /api/candidates/{id}/approve        # Manual approve candidate
POST /api/candidates/{id}/reject         # Manual reject candidate
```

#### **Risk & Control**
```
GET  /api/bot/state                      # Current bot state
POST /api/bot/enable                     # Enable trading
POST /api/bot/disable                    # Disable trading (kill switch)
POST /api/bot/mode                       # Switch paper/live
GET  /api/risk/events                    # Recent risk events
GET  /api/risk/exposure                  # Current exposure breakdown
```

#### **Performance**
```
GET  /api/performance/summary            # Overall stats
GET  /api/performance/by-setup           # Stats by reason_code
GET  /api/performance/by-session         # Stats by session
GET  /api/performance/equity-curve       # Equity data points
```

#### **AI/Model**
```
GET  /api/models                         # Model registry
GET  /api/models/active                  # Current active model
POST /api/models/{version}/activate      # Activate a model version
GET  /api/models/health                  # Model drift indicators
```

### **WebSocket Events**

```
# Connection
ws://api/ws?token=JWT

# Subscribe
{"action": "subscribe", "channels": ["positions", "candidates", "bot_state"]}

# Events
{"event": "candidate.new", "data": {...}}
{"event": "candidate.updated", "data": {...}}
{"event": "position.opened", "data": {...}}
{"event": "position.closed", "data": {...}}
{"event": "bot_state.changed", "data": {...}}
{"event": "risk_event.triggered", "data": {...}}
```

---

## Project Structure

```
ag-analyzer/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
├── engine/                          # Python trading engine
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py                  # Entry point
│   │   ├── config.py                # Configuration
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   └── tradelocker.py       # TradeLocker broker adapter
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── strength.py          # Relative strength engine
│   │   │   ├── structure.py         # Market structure engine
│   │   │   ├── liquidity.py         # Liquidity engine
│   │   │   ├── momentum.py          # Momentum engine
│   │   │   └── events.py            # Event risk filter
│   │   ├── scoring/
│   │   │   ├── __init__.py
│   │   │   ├── confluence.py        # Confluence scorer
│   │   │   └── reasons.py           # Reason codes
│   │   ├── ai/
│   │   │   ├── __init__.py
│   │   │   ├── gate.py              # AI decision gate
│   │   │   ├── features.py          # Feature engineering
│   │   │   └── training.py          # Training pipeline
│   │   ├── risk/
│   │   │   ├── __init__.py
│   │   │   └── manager.py           # Risk management
│   │   ├── execution/
│   │   │   ├── __init__.py
│   │   │   ├── executor.py          # Order execution
│   │   │   └── reconciliation.py    # Position reconciliation
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── connection.py        # DB connection
│   │   │   └── models.py            # SQLAlchemy models
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── logging.py
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── backend/                         # FastAPI backend
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py
│   │   │   ├── trading.py
│   │   │   ├── risk.py
│   │   │   ├── performance.py
│   │   │   └── models.py
│   │   ├── services/
│   │   │   └── __init__.py
│   │   ├── schemas/
│   │   │   └── __init__.py
│   │   └── websocket/
│   │       └── __init__.py
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── ui/                              # Next.js frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx             # Dashboard
│   │   │   ├── scanner/
│   │   │   ├── analysis/[symbol]/
│   │   │   ├── positions/
│   │   │   ├── performance/
│   │   │   ├── risk/
│   │   │   ├── models/
│   │   │   └── settings/
│   │   ├── components/
│   │   │   ├── ui/                  # shadcn components
│   │   │   ├── charts/
│   │   │   ├── dashboard/
│   │   │   ├── scanner/
│   │   │   └── trading/
│   │   ├── hooks/
│   │   ├── lib/
│   │   │   ├── supabase.ts
│   │   │   └── api.ts
│   │   └── types/
│   ├── public/
│   ├── package.json
│   ├── tailwind.config.ts
│   └── Dockerfile
├── supabase/
│   ├── migrations/
│   │   ├── 001_initial_schema.sql
│   │   ├── 002_instruments.sql
│   │   ├── 003_analysis_tables.sql
│   │   ├── 004_rls_policies.sql
│   │   └── 005_indexes.sql
│   ├── seed.sql
│   └── config.toml
├── docker-compose.yml
├── docker-compose.dev.yml
├── .env.example
├── README.md
└── CHANGELOG.md
```

---

## Environment Variables

```env
# Database (Supabase)
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres
SUPABASE_URL=https://[PROJECT].supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...

# Database (Local/Docker - from AG1)
DB_NAME=ag_analyzer
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# TradeLocker (from AG1 config_manager.py)
# Uses the official tradelocker Python package
TL_ENVIRONMENT=https://demo.tradelocker.com  # or https://live.tradelocker.com
TL_EMAIL=your@email.com
TL_PASSWORD=your_password
TL_SERVER=your_server_name
TL_ACC_NUM=your_account_number

# Bot Configuration
BOT_MODE=paper  # paper | live
TRADING_ENABLED=false
LOG_LEVEL=INFO

# AI Model
MODEL_PATH=./models
ACTIVE_MODEL_VERSION=v1.0.0

# API
API_HOST=0.0.0.0
API_PORT=8000
JWT_SECRET=your_jwt_secret
```

---

## Release Milestones

### **MVP (Release 1) - Core Functionality**
- [ ] Database schema + migrations
- [ ] TradeLocker adapter integration
- [ ] Price data ingestion pipeline
- [ ] Analysis engines (strength, structure, liquidity, momentum)
- [ ] Deterministic confluence scoring
- [ ] Scanner UI + Symbol analysis page
- [ ] Paper execution mode
- [ ] Basic risk controls
- [ ] Audit logging

### **Release 2 - AI Integration**
- [ ] Feature engineering pipeline
- [ ] Training data generation
- [ ] AI gate (classifier + regressor)
- [ ] Model registry + versioning
- [ ] Performance dashboards
- [ ] Model health monitoring

### **Release 3 - Advanced Features**
- [ ] Adaptive weights by regime
- [ ] Drift detection + alerts
- [ ] Advanced execution (partials, trailing)
- [ ] Mobile-responsive UI
- [ ] API rate limiting + security hardening

---

## Original Specification

The complete original specification is preserved below for reference.

---

[Original content from AG ANALYZER.md follows...]
