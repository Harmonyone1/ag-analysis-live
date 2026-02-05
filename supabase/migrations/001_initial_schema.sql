-- AG Analyzer Initial Schema Migration
-- Creates all core tables for the trading system

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- EXISTING TABLES (from AG1)
-- ============================================

-- Price history - OHLCV bars
CREATE TABLE IF NOT EXISTS price_history (
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

-- Trade log - executed trade outcomes
CREATE TABLE IF NOT EXISTS trade_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
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

-- ============================================
-- NEW TABLES
-- ============================================

-- Instruments metadata
CREATE TABLE IF NOT EXISTS instruments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) UNIQUE NOT NULL,
    asset_class VARCHAR(20) NOT NULL,
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

-- Analysis snapshots - powers UI and AI
CREATE TABLE IF NOT EXISTS analysis_snapshot (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    snapshot_time TIMESTAMPTZ NOT NULL,
    structure_state VARCHAR(20),
    trend_direction VARCHAR(10),
    strength_scores JSONB,
    liquidity_zones JSONB,
    momentum_state JSONB,
    regime_state JSONB,
    event_risk JSONB,
    computed_levels JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trade candidates - generated setups
CREATE TABLE IF NOT EXISTS trade_candidates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    entry_type VARCHAR(20) NOT NULL,
    entry_zone JSONB NOT NULL,
    invalidation_price DECIMAL(18, 8) NOT NULL,
    stop_price DECIMAL(18, 8) NOT NULL,
    tp_targets JSONB NOT NULL,
    confluence_score INTEGER NOT NULL,
    reasons TEXT[] NOT NULL,
    reason_codes VARCHAR(50)[] NOT NULL,
    ai_approved BOOLEAN,
    ai_confidence DECIMAL(5, 4),
    ai_expected_r DECIMAL(5, 2),
    status VARCHAR(20) NOT NULL DEFAULT 'new',
    snapshot_id UUID REFERENCES analysis_snapshot(id),
    feature_vector JSONB,
    feature_set_version VARCHAR(20),
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Orders - order lifecycle tracking
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
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

-- Executions - fill details
CREATE TABLE IF NOT EXISTS executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    broker_trade_id VARCHAR(100),
    order_id UUID REFERENCES orders(id),
    fill_price DECIMAL(18, 8) NOT NULL,
    fill_qty DECIMAL(18, 8) NOT NULL,
    fees DECIMAL(18, 4) DEFAULT 0,
    slippage_pips DECIMAL(10, 2),
    fill_time TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Positions - open position tracking
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
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
    close_reason VARCHAR(50),
    r_multiple DECIMAL(5, 2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Risk events - safety tracking
CREATE TABLE IF NOT EXISTS risk_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    payload JSONB,
    resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Bot state - system state
CREATE TABLE IF NOT EXISTS bot_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    trading_enabled BOOLEAN DEFAULT false,
    mode VARCHAR(20) DEFAULT 'paper',
    last_heartbeat TIMESTAMPTZ,
    current_model_version VARCHAR(50),
    last_error TEXT,
    daily_pnl DECIMAL(18, 2) DEFAULT 0,
    daily_trades INTEGER DEFAULT 0,
    open_risk_percent DECIMAL(5, 2) DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT single_row CHECK (id = 1)
);

-- Model registry - AI model versions
CREATE TABLE IF NOT EXISTS model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    feature_set_version VARCHAR(20) NOT NULL,
    training_start DATE NOT NULL,
    training_end DATE NOT NULL,
    metrics JSONB NOT NULL,
    artifact_path TEXT,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Settings - configuration
CREATE TABLE IF NOT EXISTS settings (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Economic calendar - news events
CREATE TABLE IF NOT EXISTS economic_calendar (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_name VARCHAR(200) NOT NULL,
    currency VARCHAR(10) NOT NULL,
    impact VARCHAR(20) NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    actual VARCHAR(50),
    forecast VARCHAR(50),
    previous VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_price_history_lookup
    ON price_history(symbol, timeframe, bar_time DESC);

CREATE INDEX IF NOT EXISTS idx_analysis_snapshot_lookup
    ON analysis_snapshot(symbol, timeframe, snapshot_time DESC);

CREATE INDEX IF NOT EXISTS idx_trade_candidates_status
    ON trade_candidates(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_trade_candidates_symbol
    ON trade_candidates(symbol, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_positions_open
    ON positions(symbol, open_time DESC)
    WHERE close_time IS NULL;

CREATE INDEX IF NOT EXISTS idx_risk_events_recent
    ON risk_events(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_economic_calendar_upcoming
    ON economic_calendar(event_time);

CREATE INDEX IF NOT EXISTS idx_orders_candidate
    ON orders(candidate_id);

CREATE INDEX IF NOT EXISTS idx_executions_order
    ON executions(order_id);

-- ============================================
-- INITIAL DATA
-- ============================================

-- Initialize bot state (single row)
INSERT INTO bot_state (id, trading_enabled, mode)
VALUES (1, false, 'paper')
ON CONFLICT (id) DO NOTHING;

-- Default settings
INSERT INTO settings (key, value, description) VALUES
    ('risk_params', '{"max_risk_per_trade": 0.01, "max_daily_loss": 0.03, "max_open_positions": 5}', 'Risk management parameters'),
    ('scoring_weights', '{"strength": 25, "structure": 25, "liquidity": 20, "momentum": 15, "regime": 10, "sentiment": 5}', 'Confluence scoring weights'),
    ('ai_thresholds', '{"min_confluence": 60, "min_probability": 0.55, "min_ev": 0.15}', 'AI gate thresholds')
ON CONFLICT (key) DO NOTHING;
