"""AG Analyzer Trading Engine - Main Entry Point.

This is the main orchestrator for the trading engine that:
1. Connects to TradeLocker
2. Ingests market data
3. Runs analysis engines
4. Generates and scores trade candidates
5. Applies AI decision gate
6. Executes trades (in paper or live mode)
7. Monitors and logs all activity
"""

import asyncio
import logging
import signal
import sys
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional

import structlog

from src.config import load_config, Config
from src.adapters import TradeLockerAdapter, BrokerAdapter
from src.database import init_db, get_db, DatabaseManager
from src.database.models import (
    BotState, TradeCandidate, Position, AnalysisSnapshot,
    PriceHistory, RiskEvent, Instrument
)
from src.analysis import MarketAnalyzer
from src.scoring import ConfluenceScorer, TradeSetup
from src.risk import RiskManager, RiskLimits, OpenPosition
from src.execution import ExecutionEngine, ExecutionMode
from src.ai import AIGate, GateConfig, GateDecisionType

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Trading symbols to scan
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "EURNZD", "GBPAUD",
    "GBPNZD", "AUDNZD", "AUDCAD", "NZDCAD", "CADJPY", "CHFJPY",
    "EURCAD", "EURCHF", "GBPCAD", "GBPCHF", "AUDCHF", "NZDCHF", "CADCHF"
]


class TradingEngine:
    """Main trading engine orchestrator."""

    def __init__(self, config: Config):
        """Initialize trading engine.

        Args:
            config: Application configuration
        """
        self.config = config
        self.broker: Optional[BrokerAdapter] = None
        self.db: Optional[DatabaseManager] = None
        self.running = False
        self._shutdown_event = asyncio.Event()

        # Initialize components
        self.analyzer: Optional[MarketAnalyzer] = None
        self.scorer: Optional[ConfluenceScorer] = None
        self.risk_manager: Optional[RiskManager] = None
        self.executor: Optional[ExecutionEngine] = None
        self.ai_gate: Optional[AIGate] = None

        # State
        self._last_scan_time: Optional[datetime] = None
        self._active_candidates: List[TradeCandidate] = []
        self._market_views: Dict[str, object] = {}  # symbol → MarketView from latest scan
        self._candle_data: Dict[str, Dict[str, Any]] = {}  # symbol → {m15: {}, h1: {}, h4: {}}
        self._htf_cache_time: Dict[str, datetime] = {}  # symbol → last H1/H4 fetch time

    async def start(self) -> None:
        """Start the trading engine."""
        logger.info("Starting AG Analyzer Trading Engine",
                   mode=self.config.bot.mode,
                   trading_enabled=self.config.bot.trading_enabled)

        # Initialize database
        await self._init_database()

        # Initialize broker connection
        await self._connect_broker()

        # Initialize components
        await self._init_components()

        # Start main loop
        self.running = True
        await self._run_loop()

    async def stop(self) -> None:
        """Stop the trading engine gracefully."""
        logger.info("Stopping trading engine...")
        self.running = False
        self._shutdown_event.set()

        if self.broker:
            self.broker.disconnect()

        logger.info("Trading engine stopped")

    async def _init_database(self) -> None:
        """Initialize database connection."""
        logger.info("Initializing database connection")
        init_db()
        self.db = DatabaseManager()

    async def _connect_broker(self) -> None:
        """Establish connection to TradeLocker."""
        logger.info("Connecting to TradeLocker",
                   environment=self.config.tradelocker.environment)

        self.broker = TradeLockerAdapter(
            environment=self.config.tradelocker.environment,
            email=self.config.tradelocker.email,
            password=self.config.tradelocker.password,
            server=self.config.tradelocker.server,
            acc_num=self.config.tradelocker.acc_num,
            log_level=self.config.bot.log_level.lower(),
        )

        if not self.broker.connect():
            raise RuntimeError("Failed to connect to TradeLocker")

        account = self.broker.get_account()
        logger.info("Connected to TradeLocker",
                   account_name=account.account_name,
                   balance=float(account.balance),
                   currency=account.currency)

    async def _init_components(self) -> None:
        """Initialize analysis and trading components."""
        logger.info("Initializing engine components")

        # Get account balance for risk manager
        account = self.broker.get_account()

        # Initialize risk manager
        risk_limits = RiskLimits(
            max_risk_per_trade=self.config.risk.max_risk_per_trade,
            max_daily_loss=self.config.risk.max_daily_loss,
            max_open_positions=self.config.risk.max_open_positions,
            max_correlated_exposure=self.config.risk.max_correlated_exposure,
        )
        self.risk_manager = RiskManager(
            limits=risk_limits,
            account_balance=account.balance,
        )

        # Initialize analyzer (no broker needed - uses OHLC arrays)
        self.analyzer = MarketAnalyzer()

        # Initialize scorer
        self.scorer = ConfluenceScorer()

        # Initialize execution engine
        exec_mode = (
            ExecutionMode.LIVE
            if self.config.bot.mode == "live"
            else ExecutionMode.PAPER
        )
        self.executor = ExecutionEngine(self.broker, mode=exec_mode)

        # Initialize AI gate with trained model
        gate_config = GateConfig(
            model_path=self.config.ai.model_path,
            min_prob_win=self.config.ai.min_ai_probability,
            min_expected_r=self.config.ai.min_expected_value,
            min_confluence_score=self.config.ai.min_confluence_score,
            max_candidates_per_session=3,
            fallback_to_rules=True,
        )
        self.ai_gate = AIGate(gate_config)
        if self.ai_gate.model_loaded:
            logger.info("AI Gate loaded trained model", version=self.ai_gate.feature_version)
        else:
            logger.warning("AI Gate using rule-based fallback (no model loaded)")

        logger.info("Components initialized")

    async def _run_loop(self) -> None:
        """Main engine loop."""
        logger.info("Entering main loop")

        while self.running:
            try:
                cycle_start = datetime.now()

                # 1. Check broker connection
                if not self.broker.is_connected():
                    logger.warning("Broker connection lost, reconnecting...")
                    await self._connect_broker()

                # 2. Update heartbeat
                await self._update_heartbeat()

                # 3. Ingest latest market data
                await self._ingest_market_data()

                # 4. Run analysis engines
                await self._run_analysis()

                # 5. Generate and score candidates
                await self._generate_candidates()

                # 6. Apply AI gate
                await self._apply_ai_gate()

                # 7. Execute approved trades
                if self.config.bot.trading_enabled:
                    await self._execute_trades()

                # 8. Monitor positions
                await self._monitor_positions()

                # 9. Reconcile with broker
                await self._reconcile()

                # Calculate cycle time and sleep
                cycle_time = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, 15 - cycle_time)  # Target 15-second cycles

                logger.debug("Cycle completed",
                           cycle_time=f"{cycle_time:.2f}s",
                           sleep_time=f"{sleep_time:.2f}s")

                # Wait for next cycle or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=sleep_time
                    )
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop

            except Exception as e:
                logger.error("Error in main loop", error=str(e), exc_info=True)
                await asyncio.sleep(5)  # Back off on error

    async def _update_heartbeat(self) -> None:
        """Update bot heartbeat in database."""
        try:
            with self.db.session() as session:
                bot_state = session.query(BotState).filter(
                    BotState.id == 1
                ).first()

                if bot_state:
                    bot_state.last_heartbeat = datetime.now()
                    bot_state.trading_enabled = self.config.bot.trading_enabled
                    bot_state.mode = self.config.bot.mode
                    session.commit()
        except Exception as e:
            logger.error("Failed to update heartbeat", error=str(e))

    async def _ingest_market_data(self) -> None:
        """Ingest latest price data from broker."""
        try:
            for symbol in SYMBOLS:
                # Get latest candles
                candles = self.broker.get_candles(symbol, "M15", limit=100)

                if not candles:
                    continue

                # Store in database
                with self.db.session() as session:
                    for candle in candles[-10:]:  # Store last 10 candles
                        # Check if exists
                        existing = session.query(PriceHistory).filter(
                            PriceHistory.symbol == symbol,
                            PriceHistory.timeframe == "M15",
                            PriceHistory.bar_time == candle.timestamp,
                        ).first()

                        if not existing:
                            price_record = PriceHistory(
                                symbol=symbol,
                                timeframe="M15",
                                bar_time=candle.timestamp,
                                open=candle.open,
                                high=candle.high,
                                low=candle.low,
                                close=candle.close,
                                volume=candle.volume,
                            )
                            session.add(price_record)

                    session.commit()

        except Exception as e:
            logger.error("Failed to ingest market data", error=str(e))

    async def _run_analysis(self) -> None:
        """Run all analysis engines."""
        try:
            for symbol in SYMBOLS:
                # Get candles for analysis
                candles = self.broker.get_candles(symbol, "M15", limit=200)
                if not candles or len(candles) < 50:
                    continue

                # Extract OHLC arrays for analyzer
                import numpy as np
                opens = np.array([float(c.open) for c in candles])
                highs = np.array([float(c.high) for c in candles])
                lows = np.array([float(c.low) for c in candles])
                closes = np.array([float(c.close) for c in candles])
                timestamps = np.array([c.timestamp for c in candles])

                # Run full analysis
                market_view = self.analyzer.analyze(symbol, "M15", opens, highs, lows, closes, timestamps)

                # Store analysis snapshot
                with self.db.session() as session:
                    snapshot = AnalysisSnapshot(
                        id=uuid.uuid4(),
                        symbol=symbol,
                        timeframe="M15",
                        snapshot_time=datetime.now(),
                        structure_state=market_view.structure.state.value if market_view.structure else None,
                        trend_direction=market_view.structure.trend_direction.value if market_view.structure else None,
                        strength_scores={"strengths": market_view.strength.strengths, "timeframe": market_view.strength.timeframe} if market_view.strength else None,
                        momentum_state={"rsi": market_view.momentum.rsi, "regime": market_view.momentum.rsi_regime.value if market_view.momentum.rsi_regime else None, "atr_percentile": market_view.momentum.atr_percentile} if market_view.momentum else None,
                        liquidity_zones=[{"price": z.price, "type": z.zone_type.value, "touches": z.touch_count} for z in market_view.liquidity.zones] if market_view.liquidity and market_view.liquidity.zones else None,
                    )
                    session.add(snapshot)

                    session.commit()

        except Exception as e:
            logger.error("Failed to run analysis", error=str(e))


    def _fetch_htf_candles(self, symbol: str) -> None:
        """Fetch H1/H4 candles for v3 AI gate, with 1-hour / 4-hour TTL cache."""
        import numpy as np
        now = datetime.utcnow() if not hasattr(datetime, 'now') else datetime.now()

        # Check if we have cached data and it's fresh enough
        cache_key = symbol
        last_fetch = self._htf_cache_time.get(cache_key)
        if last_fetch and (now - last_fetch).total_seconds() < 3600:
            return  # H1/H4 data is fresh enough

        try:
            # Fetch H1 candles (100 bars = ~4 days)
            h1_candles = self.broker.get_candles(symbol, "H1", limit=100)
            if h1_candles and len(h1_candles) >= 10:
                self._candle_data.setdefault(symbol, {})["h1"] = {
                    "opens": np.array([float(c.open) for c in h1_candles]),
                    "highs": np.array([float(c.high) for c in h1_candles]),
                    "lows": np.array([float(c.low) for c in h1_candles]),
                    "closes": np.array([float(c.close) for c in h1_candles]),
                    "volumes": np.array([float(c.volume) for c in h1_candles]),
                }

            # Fetch H4 candles (50 bars = ~8 days)
            h4_candles = self.broker.get_candles(symbol, "H4", limit=50)
            if h4_candles and len(h4_candles) >= 10:
                self._candle_data.setdefault(symbol, {})["h4"] = {
                    "opens": np.array([float(c.open) for c in h4_candles]),
                    "highs": np.array([float(c.high) for c in h4_candles]),
                    "lows": np.array([float(c.low) for c in h4_candles]),
                    "closes": np.array([float(c.close) for c in h4_candles]),
                    "volumes": np.array([float(c.volume) for c in h4_candles]),
                }

            self._htf_cache_time[cache_key] = now

        except Exception as e:
            logger.warning("Failed to fetch HTF candles", symbol=symbol, error=str(e))

    async def _generate_candidates(self) -> None:
        """Generate and score trade candidates."""
        try:
            candidates = []
            self._market_views = {}  # Reset each cycle

            for symbol in SYMBOLS:
                # Get candles
                candles = self.broker.get_candles(symbol, "M15", limit=200)
                if not candles or len(candles) < 50:
                    continue

                # Extract OHLC arrays for analyzer
                import numpy as np
                opens = np.array([float(c.open) for c in candles])
                highs = np.array([float(c.high) for c in candles])
                lows = np.array([float(c.low) for c in candles])
                closes = np.array([float(c.close) for c in candles])
                volumes = np.array([float(c.volume) for c in candles])
                timestamps = np.array([c.timestamp for c in candles])

                # Store M15 candle arrays for v3 AI gate (preserve h1/h4)
                self._candle_data.setdefault(symbol, {})["m15"] = {
                    "opens": opens,
                    "highs": highs,
                    "lows": lows,
                    "closes": closes,
                    "volumes": volumes,
                }

                # Fetch H1/H4 candles for v3 AI gate (TTL cached)
                self._fetch_htf_candles(symbol)

                # Run analysis
                market_view = self.analyzer.analyze(symbol, "M15", opens, highs, lows, closes, timestamps)
                self._market_views[symbol] = market_view

                # Generate setups (returns list)
                setups = self.scorer.generate_setups(market_view)

                for setup in setups:
                    if setup.confluence_score >= self.config.ai.min_confluence_score:
                        # Check risk
                        # Calculate risk amount from account balance
                        try:
                            account = self.broker.get_account()
                            risk_pct = Decimal(str(self.config.risk.max_risk_per_trade))
                            risk_amount = account.balance * risk_pct
                        except Exception:
                            risk_amount = Decimal("5")  # Fallback
                        risk_check = self.risk_manager.check_new_trade(
                            symbol=symbol,
                            side="buy" if setup.direction == "LONG" else "sell",
                            risk_amount=risk_amount,
                        )

                        if risk_check.passed:
                            candidates.append(setup)

            # Store candidates
            self._active_candidates = []
            with self.db.session() as session:
                # Expire old pending candidates
                old_candidates = session.query(TradeCandidate).filter(
                    TradeCandidate.status == "new",
                    TradeCandidate.created_at < datetime.now() - timedelta(hours=1),
                ).all()

                for old in old_candidates:
                    old.status = "expired"

                # Add new candidates
                for setup in candidates[:10]:  # Limit to top 10
                    entry_min = setup.entry_zone[0] if setup.entry_zone else 0
                    entry_max = setup.entry_zone[1] if setup.entry_zone and len(setup.entry_zone) > 1 else entry_min
                    candidate = TradeCandidate(
                        id=uuid.uuid4(),
                        symbol=setup.symbol,
                        direction=setup.direction,
                        entry_type=setup.entry_type or "LIMIT",
                        confluence_score=setup.confluence_score,
                        entry_zone={"min": float(entry_min), "max": float(entry_max)},
                        invalidation_price=Decimal(str(setup.stop_price)),
                        stop_price=Decimal(str(setup.stop_price)),
                        tp_targets=setup.tp_targets or [],
                        reasons=[str(r) for r in setup.reasons] if setup.reasons else [],
                        reason_codes=[str(r) for r in setup.reasons] if setup.reasons else [],
                        status="new",
                        expires_at=datetime.now() + timedelta(hours=1),
                    )
                    session.add(candidate)
                    self._active_candidates.append(candidate)

                session.commit()

            logger.info("Generated candidates", count=len(candidates))

        except Exception as e:
            logger.error("Failed to generate candidates", error=str(e))

    async def _apply_ai_gate(self) -> None:
        """Apply AI decision gate to candidates."""
        try:
            with self.db.session() as session:
                pending = session.query(TradeCandidate).filter(
                    TradeCandidate.status == "new",
                    TradeCandidate.ai_approved.is_(None),
                ).all()

                # Build trade history from closed positions for AI features
                trade_history = []
                recent_positions = session.query(Position).filter(
                    Position.close_time.isnot(None),
                ).order_by(Position.close_time.desc()).limit(50).all()
                for p in recent_positions:
                    trade_history.append({
                        "symbol": p.symbol,
                        "direction": "LONG" if p.side == "buy" else "SHORT",
                        "pnl": float(p.realized_pnl) if p.realized_pnl else 0,
                        "confluence_score": 0,
                    })

                for candidate in pending:
                    # Create setup from candidate
                    entry_zone = (candidate.entry_zone.get("min", 0), candidate.entry_zone.get("max", 0)) if candidate.entry_zone else (0, 0)
                    setup = TradeSetup(
                        symbol=candidate.symbol,
                        direction=candidate.direction,
                        entry_type=candidate.entry_type,
                        entry_zone=entry_zone,
                        stop_price=float(candidate.stop_price),
                        invalidation_price=float(candidate.invalidation_price),
                        tp_targets=candidate.tp_targets or [],
                        confluence_score=candidate.confluence_score,
                        sub_scores={},
                        reasons=candidate.reasons or [],
                        reason_codes=candidate.reason_codes or [],
                        risk_reward=2.0,
                        atr_distance=0.0,
                        timestamp=candidate.created_at,
                        timeframe="M15",
                    )

                    # Apply AI gate with market view, trade history, and candle data
                    mv = self._market_views.get(candidate.symbol)
                    symbol_candles = self._candle_data.get(candidate.symbol, {})
                    decision = self.ai_gate.evaluate(
                        setup,
                        market_view=mv,
                        trade_history=trade_history,
                        m15_candles=symbol_candles.get("m15"),
                        h1_candles=symbol_candles.get("h1"),
                        h4_candles=symbol_candles.get("h4"),
                    )

                    # Update candidate
                    candidate.ai_confidence = decision.probability
                    candidate.ai_expected_r = decision.expected_r
                    candidate.ai_approved = decision.decision == GateDecisionType.APPROVED

                    if decision.decision == GateDecisionType.REJECTED:
                        candidate.status = "rejected"
                    elif decision.decision == GateDecisionType.APPROVED:
                        candidate.status = "approved"
                    elif decision.decision == GateDecisionType.NEEDS_REVIEW:
                        candidate.status = "review"

                    import sys
                    print(f"[AI_GATE] {candidate.symbol} {candidate.direction}: "
                          f"P(win)={decision.probability:.4f} E[R]={decision.expected_r:.4f} "
                          f"P(timeout)={decision.prob_timeout:.4f} → {decision.decision.value} "
                          f"| reasons: {'; '.join(decision.reasons[:3])}", file=sys.stderr)

                session.commit()

        except Exception as e:
            logger.error("Failed to apply AI gate", error=str(e))

    async def _execute_trades(self) -> None:
        """Execute approved trade candidates."""
        try:
            import sys
            print(f"[EXEC_MODE] executor.mode={self.executor.mode}, bot.mode={self.config.bot.mode}", file=sys.stderr)

            # Session filter removed: model time features (hour_sin/cos) handle session awareness

            with self.db.session() as session:
                approved = session.query(TradeCandidate).filter(
                    TradeCandidate.status == "approved",
                ).all()
                print(f"[EXEC_MODE] Found {len(approved)} approved candidates", file=sys.stderr)

                # Get broker positions once to check for dupes
                try:
                    broker_symbols = {bp.symbol.upper() for bp in self.broker.get_positions()}
                except Exception:
                    broker_symbols = set()

                for candidate in approved:
                    # --- Filter 1: Duplicate check (broker) ---
                    if candidate.symbol.upper() in broker_symbols:
                        candidate.status = "executed"
                        continue

                    # --- Filter 2: Duplicate check (DB) ---
                    existing_pos = session.query(Position).filter(
                        Position.symbol == candidate.symbol,
                        Position.close_time.is_(None),
                    ).first()
                    if existing_pos:
                        candidate.status = "executed"
                        continue

                    # --- Filter 3: Cooldown after stop-out (4 hours) ---
                    cooldown_cutoff = datetime.now() - timedelta(hours=4)
                    recent_loss = session.query(Position).filter(
                        Position.symbol == candidate.symbol,
                        Position.close_time.isnot(None),
                        Position.close_time > cooldown_cutoff,
                        Position.realized_pnl < 0,
                    ).first()
                    if recent_loss:
                        candidate.status = "rejected"
                        print(f"[EXEC] SKIP {candidate.symbol}: cooldown after loss "
                              f"(closed {recent_loss.close_time.strftime('%H:%M')} UTC, pnl={recent_loss.realized_pnl})",
                              file=sys.stderr)
                        continue

                    # --- Filter 4: Reject counter-trend setups ---
                    reasons_text = " ".join(candidate.reasons) if candidate.reasons else ""
                    if "counter-trend" in reasons_text.lower():
                        candidate.status = "rejected"
                        print(f"[EXEC] SKIP {candidate.symbol}: counter-trend setup", file=sys.stderr)
                        continue

                    # Fixed lot size
                    position_size = Decimal("0.20")

                    # Create setup for execution
                    entry_zone = (candidate.entry_zone.get("min", 0), candidate.entry_zone.get("max", 0)) if candidate.entry_zone else (0, 0)

                    # --- Filter 5: Minimum stop distance (20 pips) ---
                    entry_mid = (entry_zone[0] + entry_zone[1]) / 2
                    is_jpy = "JPY" in candidate.symbol
                    if entry_mid > 0:
                        stop_dist = abs(entry_mid - float(candidate.stop_price))
                        min_stop_dist = 0.200 if is_jpy else 0.00200  # 20 pips minimum
                        if stop_dist < min_stop_dist:
                            candidate.status = "rejected"
                            pips = stop_dist / (0.01 if is_jpy else 0.0001)
                            print(f"[EXEC] SKIP {candidate.symbol}: stop too tight "
                                  f"({pips:.1f} pips < 20)", file=sys.stderr)
                            continue

                    # --- Filter 6: Minimum R:R ratio of 1.5 ---
                    tp_price_check = None
                    if candidate.tp_targets:
                        first_tp = candidate.tp_targets[0]
                        if isinstance(first_tp, dict):
                            tp_price_check = first_tp.get("price")
                        elif first_tp:
                            tp_price_check = float(first_tp)

                    if tp_price_check and entry_mid > 0:
                        sl_dist = abs(entry_mid - float(candidate.stop_price))
                        tp_dist = abs(float(tp_price_check) - entry_mid)
                        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0
                        if rr_ratio < 1.49:
                            candidate.status = "rejected"
                            print(f"[EXEC] SKIP {candidate.symbol}: R:R too low "
                                  f"({rr_ratio:.2f} < 1.50)", file=sys.stderr)
                            continue
                    setup = TradeSetup(
                        symbol=candidate.symbol,
                        direction=candidate.direction,
                        entry_type="MARKET",
                        entry_zone=entry_zone,
                        stop_price=float(candidate.stop_price),
                        invalidation_price=float(candidate.invalidation_price),
                        tp_targets=candidate.tp_targets or [],
                        confluence_score=candidate.confluence_score,
                        sub_scores={},
                        reasons=candidate.reasons or [],
                        reason_codes=candidate.reason_codes or [],
                        risk_reward=float(candidate.ai_expected_r) if candidate.ai_expected_r else 2.0,
                        atr_distance=0.0,
                        timestamp=candidate.created_at,
                        timeframe="M15",
                    )

                    # Execute
                    result = self.executor.execute_setup(
                        setup=setup,
                        position_size=position_size,
                        candidate_id=str(candidate.id),
                    )

                    if result.success:
                        candidate.status = "executed"

                        # Create Position record
                        tp_price = None
                        if candidate.tp_targets:
                            first_tp = candidate.tp_targets[0]
                            if isinstance(first_tp, dict):
                                tp_price = Decimal(str(first_tp.get("price", 0))) if first_tp.get("price") else None
                            elif first_tp:
                                tp_price = Decimal(str(first_tp))

                        position = Position(
                            id=uuid.uuid4(),
                            broker_position_id=result.broker_order_id,
                            candidate_id=candidate.id,
                            symbol=candidate.symbol,
                            side="sell" if candidate.direction == "SHORT" else "buy",
                            quantity=Decimal(str(result.fill_qty)) if result.fill_qty else Decimal(str(position_size)),
                            avg_entry_price=Decimal(str(result.fill_price)) if result.fill_price else Decimal(str(entry_zone[0])),
                            current_stop_loss=candidate.stop_price,
                            current_take_profit=tp_price,
                            unrealized_pnl=Decimal("0"),
                            realized_pnl=Decimal("0"),
                            open_time=datetime.now(),
                        )
                        session.add(position)

                        import sys
                        print(f"[EXEC] OPENED {candidate.symbol} {candidate.direction} "
                              f"qty={position.quantity} @ {position.avg_entry_price} "
                              f"SL={position.current_stop_loss} TP={position.current_take_profit} "
                              f"broker_id={result.broker_order_id}", file=sys.stderr)

                        logger.info("Trade executed",
                                   symbol=candidate.symbol,
                                   order_id=result.order_id)
                    else:
                        candidate.status = "failed"
                        import sys
                        print(f"[EXEC] FAILED {candidate.symbol} {candidate.direction}: {result.message}", file=sys.stderr)
                        logger.warning("Trade execution failed",
                                      symbol=candidate.symbol,
                                      message=result.message)

                session.commit()

        except Exception as e:
            logger.error("Failed to execute trades", error=str(e))

    async def _monitor_positions(self) -> None:
        """Monitor open positions and update P&L."""
        try:
            # Get positions from broker
            broker_positions = self.broker.get_positions()

            # Update risk manager
            open_positions = []
            for bp in broker_positions:
                symbol = bp.symbol.upper()
                currencies = set()
                if len(symbol) == 6:
                    base = symbol[:3]
                    quote = symbol[3:]
                    if bp.side == "buy":
                        currencies = {f"{base}_LONG", f"{quote}_SHORT"}
                    else:
                        currencies = {f"{base}_SHORT", f"{quote}_LONG"}

                open_positions.append(OpenPosition(
                    symbol=symbol,
                    side=bp.side,
                    size=bp.quantity,
                    risk_amount=Decimal("100"),  # Would calculate
                    currencies=currencies,
                ))

            self.risk_manager.update_positions(open_positions)

            # Update database positions
            with self.db.session() as session:
                for bp in broker_positions:
                    pos = session.query(Position).filter(
                        Position.broker_position_id == str(bp.position_id)
                    ).first()

                    if pos:
                        pos.current_price = bp.entry_price  # Would use current price
                        pos.unrealized_pnl = bp.unrealized_pnl

                session.commit()

            # Update account balance
            account = self.broker.get_account()
            self.risk_manager.update_account_balance(account.balance)

        except Exception as e:
            logger.error("Failed to monitor positions", error=str(e))

    async def _reconcile(self) -> None:
        """Reconcile local state with broker."""
        try:
            from src.execution import PositionReconciler

            reconciler = PositionReconciler(self.broker)

            # Get local positions (open = close_time is NULL)
            with self.db.session() as session:
                db_positions = session.query(Position).filter(
                    Position.close_time.is_(None)
                ).all()

                local_positions = []
                db_pos_map = {}  # position_id str -> Position ORM object
                for p in db_positions:
                    from src.execution.reconciliation import LocalPosition
                    local_positions.append(LocalPosition(
                        position_id=str(p.id),
                        broker_position_id=p.broker_position_id,
                        symbol=p.symbol,
                        side=p.side,
                        quantity=p.quantity,
                        entry_price=p.avg_entry_price,
                        stop_loss=p.current_stop_loss,
                        take_profit=p.current_take_profit,
                        opened_at=p.open_time,
                    ))
                    db_pos_map[str(p.id)] = p

                result = reconciler.reconcile(local_positions)

                if result.orphaned_broker:
                    logger.warning("Found orphaned broker positions",
                                  count=len(result.orphaned_broker))

                # Close DB positions that were closed on broker
                if result.missing_local:
                    logger.warning("Closing positions missing from broker",
                                  count=len(result.missing_local))

                    # Look up actual close prices from broker order history
                    close_prices = {}
                    try:
                        order_df = self.broker._api.get_all_orders(history=True)
                        if order_df is not None and len(order_df) > 0:
                            # Closing orders have isOpen == "false" and a positionId
                            closing = order_df[
                                (order_df["isOpen"] == "false") &
                                (order_df["status"] == "Filled")
                            ]
                            for _, row in closing.iterrows():
                                bp_id = str(int(row["positionId"]))
                                close_prices[bp_id] = Decimal(str(row["avgPrice"]))
                    except Exception as e:
                        logger.warning("Could not fetch order history for PnL", error=str(e))

                    for pos_id in result.missing_local:
                        pos = db_pos_map.get(pos_id)
                        if pos and pos.close_time is None:
                            close_price = close_prices.get(pos.broker_position_id)

                            if close_price and pos.avg_entry_price:
                                # Use actual close price from broker
                                if pos.side == "buy":
                                    pnl_per_unit = close_price - pos.avg_entry_price
                                else:
                                    pnl_per_unit = pos.avg_entry_price - close_price
                                if "JPY" in pos.symbol:
                                    pos.realized_pnl = pnl_per_unit * pos.quantity * Decimal("1000")
                                else:
                                    pos.realized_pnl = pnl_per_unit * pos.quantity * Decimal("100000")
                                logger.info("Closed position with actual price",
                                           symbol=pos.symbol, side=pos.side,
                                           close_price=float(close_price),
                                           realized_pnl=float(pos.realized_pnl))
                            elif pos.current_stop_loss and pos.avg_entry_price:
                                # Fallback: estimate at stop loss if no close order found
                                if pos.side == "buy":
                                    pnl_per_unit = pos.current_stop_loss - pos.avg_entry_price
                                else:
                                    pnl_per_unit = pos.avg_entry_price - pos.current_stop_loss
                                if "JPY" in pos.symbol:
                                    pos.realized_pnl = pnl_per_unit * pos.quantity * Decimal("1000")
                                else:
                                    pos.realized_pnl = pnl_per_unit * pos.quantity * Decimal("100000")
                                logger.warning("Closed position estimated at stop",
                                              symbol=pos.symbol, side=pos.side,
                                              realized_pnl=float(pos.realized_pnl))
                            pos.close_time = datetime.now()

                session.commit()

        except Exception as e:
            logger.error("Failed to reconcile", error=str(e))


async def main():
    """Main entry point."""
    # Load configuration
    try:
        config = load_config()
    except EnvironmentError as e:
        logger.error("Configuration error", error=str(e))
        sys.exit(1)

    # Configure logging level
    logging.basicConfig(level=getattr(logging, config.bot.log_level.upper()))

    # Create engine
    engine = TradingEngine(config)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(engine.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: signal_handler())

    # Start engine
    try:
        await engine.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error("Fatal error", error=str(e), exc_info=True)
        sys.exit(1)
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
