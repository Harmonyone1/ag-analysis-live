"""SQLAlchemy models for AG Analyzer database."""

from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
import uuid

from sqlalchemy import (
    Column, String, Integer, BigInteger, Boolean, DateTime, Numeric,
    ForeignKey, Text, Index, CheckConstraint, UniqueConstraint, JSON
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Instrument(Base):
    """Tradeable instrument metadata."""
    __tablename__ = "instruments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    asset_class: Mapped[str] = mapped_column(String(20), nullable=False)  # FX, INDEX, COMMODITY, CRYPTO
    base_currency: Mapped[Optional[str]] = mapped_column(String(10))
    quote_currency: Mapped[Optional[str]] = mapped_column(String(10))
    pip_size: Mapped[Decimal] = mapped_column(Numeric(18, 10), nullable=False)
    tick_size: Mapped[Decimal] = mapped_column(Numeric(18, 10), nullable=False)
    contract_size: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=1)
    min_lot: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    max_spread_pips: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    session_hours: Mapped[Optional[Dict]] = mapped_column(JSONB)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    analysis_snapshots: Mapped[List["AnalysisSnapshot"]] = relationship(back_populates="instrument")
    trade_candidates: Mapped[List["TradeCandidate"]] = relationship(back_populates="instrument")


class PriceHistory(Base):
    """OHLCV price history bars."""
    __tablename__ = "price_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    bar_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    volume: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'bar_time', name='uq_price_history_symbol_tf_time'),
        Index('idx_price_history_lookup', 'symbol', 'timeframe', bar_time.desc()),
    )


class AnalysisSnapshot(Base):
    """Analysis engine output snapshots."""
    __tablename__ = "analysis_snapshot"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(20), ForeignKey("instruments.symbol"), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Market structure
    structure_state: Mapped[Optional[str]] = mapped_column(String(20))  # TREND_UP, TREND_DOWN, RANGE, DISTRIBUTION
    trend_direction: Mapped[Optional[str]] = mapped_column(String(10))  # UP, DOWN, NEUTRAL

    # Analysis data (JSONB for flexibility)
    strength_scores: Mapped[Optional[Dict]] = mapped_column(JSONB)  # {"1D": {"EUR": 0.8}, "5D": {...}}
    liquidity_zones: Mapped[Optional[List]] = mapped_column(JSONB)  # [{"type": "PDH", "price": 1.085}]
    momentum_state: Mapped[Optional[Dict]] = mapped_column(JSONB)  # {"rsi": 55, "divergence": null}
    regime_state: Mapped[Optional[Dict]] = mapped_column(JSONB)  # {"volatility": "medium"}
    event_risk: Mapped[Optional[Dict]] = mapped_column(JSONB)  # {"next_event": "FOMC"}
    computed_levels: Mapped[Optional[Dict]] = mapped_column(JSONB)  # {"invalidation": 1.08}

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    instrument: Mapped["Instrument"] = relationship(back_populates="analysis_snapshots")
    trade_candidates: Mapped[List["TradeCandidate"]] = relationship(back_populates="snapshot")

    __table_args__ = (
        Index('idx_analysis_snapshot_lookup', 'symbol', 'timeframe', snapshot_time.desc()),
    )


class TradeCandidate(Base):
    """Generated trade setup candidates."""
    __tablename__ = "trade_candidates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(20), ForeignKey("instruments.symbol"), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # LONG, SHORT
    entry_type: Mapped[str] = mapped_column(String(20), nullable=False)  # MARKET, LIMIT
    entry_zone: Mapped[Dict] = mapped_column(JSONB, nullable=False)  # {"min": 1.082, "max": 1.084}
    invalidation_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    stop_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    tp_targets: Mapped[List] = mapped_column(JSONB, nullable=False)  # [{"r": 1.5, "price": 1.09}]

    # Scoring
    confluence_score: Mapped[int] = mapped_column(Integer, nullable=False)  # 0-100
    reasons: Mapped[List[str]] = mapped_column(ARRAY(Text), nullable=False)
    reason_codes: Mapped[List[str]] = mapped_column(ARRAY(String(50)), nullable=False)

    # AI gate
    ai_approved: Mapped[Optional[bool]] = mapped_column(Boolean)
    ai_confidence: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    ai_expected_r: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))

    # Status
    status: Mapped[str] = mapped_column(String(20), nullable=False, default='new')
    snapshot_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("analysis_snapshot.id"))

    # Features for ML
    feature_vector: Mapped[Optional[Dict]] = mapped_column(JSONB)
    feature_set_version: Mapped[Optional[str]] = mapped_column(String(20))

    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    instrument: Mapped["Instrument"] = relationship(back_populates="trade_candidates")
    snapshot: Mapped[Optional["AnalysisSnapshot"]] = relationship(back_populates="trade_candidates")
    orders: Mapped[List["Order"]] = relationship(back_populates="candidate")
    positions: Mapped[List["Position"]] = relationship(back_populates="candidate")

    __table_args__ = (
        Index('idx_trade_candidates_status', 'status', created_at.desc()),
        Index('idx_trade_candidates_symbol', 'symbol', created_at.desc()),
    )


class Order(Base):
    """Order lifecycle tracking."""
    __tablename__ = "orders"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(100))
    candidate_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("trade_candidates.id"))
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # buy, sell
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)  # market, limit, stop
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    stop_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    take_profit: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default='pending')
    filled_qty: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    candidate: Mapped[Optional["TradeCandidate"]] = relationship(back_populates="orders")
    executions: Mapped[List["Execution"]] = relationship(back_populates="order")

    __table_args__ = (
        Index('idx_orders_candidate', 'candidate_id'),
    )


class Execution(Base):
    """Order fill/execution details."""
    __tablename__ = "executions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    broker_trade_id: Mapped[Optional[str]] = mapped_column(String(100))
    order_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("orders.id"), nullable=False)
    fill_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    fill_qty: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    fees: Mapped[Decimal] = mapped_column(Numeric(18, 4), default=0)
    slippage_pips: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    fill_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    order: Mapped["Order"] = relationship(back_populates="executions")

    __table_args__ = (
        Index('idx_executions_order', 'order_id'),
    )


class Position(Base):
    """Open/closed position tracking."""
    __tablename__ = "positions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    broker_position_id: Mapped[Optional[str]] = mapped_column(String(100))
    candidate_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("trade_candidates.id"))
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    avg_entry_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    current_stop_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    current_take_profit: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    unrealized_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=0)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    close_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    close_reason: Mapped[Optional[str]] = mapped_column(String(50))  # TP_HIT, SL_HIT, MANUAL, TIMEOUT
    r_multiple: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    candidate: Mapped[Optional["TradeCandidate"]] = relationship(back_populates="positions")

    __table_args__ = (
        Index('idx_positions_open', 'symbol', open_time.desc(), postgresql_where=(close_time == None)),
    )


class RiskEvent(Base):
    """Risk management events and alerts."""
    __tablename__ = "risk_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)  # DAILY_LOSS_HIT, SPREAD_FILTER, etc.
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # INFO, WARNING, CRITICAL
    payload: Mapped[Optional[Dict]] = mapped_column(JSONB)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_risk_events_recent', created_at.desc()),
    )


class BotState(Base):
    """Singleton bot state tracking."""
    __tablename__ = "bot_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    trading_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    mode: Mapped[str] = mapped_column(String(20), default='paper')  # paper, live
    last_heartbeat: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    current_model_version: Mapped[Optional[str]] = mapped_column(String(50))
    last_error: Mapped[Optional[str]] = mapped_column(Text)
    daily_pnl: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=0)
    daily_trades: Mapped[int] = mapped_column(Integer, default=0)
    open_risk_percent: Mapped[Decimal] = mapped_column(Numeric(5, 2), default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint('id = 1', name='single_row'),
    )


class ModelRegistry(Base):
    """AI model version registry."""
    __tablename__ = "model_registry"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # gate_classifier, ev_regressor
    feature_set_version: Mapped[str] = mapped_column(String(20), nullable=False)
    training_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    training_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    metrics: Mapped[Dict] = mapped_column(JSONB, nullable=False)  # {"auc": 0.72, "precision": 0.65}
    artifact_path: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Setting(Base):
    """Application settings key-value store."""
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[Dict] = mapped_column(JSONB, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class EconomicCalendar(Base):
    """Economic news calendar events."""
    __tablename__ = "economic_calendar"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_name: Mapped[str] = mapped_column(String(200), nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False)
    impact: Mapped[str] = mapped_column(String(20), nullable=False)  # LOW, MEDIUM, HIGH
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actual: Mapped[Optional[str]] = mapped_column(String(50))
    forecast: Mapped[Optional[str]] = mapped_column(String(50))
    previous: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_economic_calendar_upcoming', 'event_time', postgresql_where=(event_time > func.now())),
    )


class TradeLog(Base):
    """Legacy trade log for compatibility with AG1."""
    __tablename__ = "trade_log"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    size: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    exit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    confidence: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    atr: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8))
    model_version: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
