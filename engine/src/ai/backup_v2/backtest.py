"""Backtesting Engine - Generates training data from real market data.

Uses TradeLocker historical data to:
1. Replay market data through the analyzer
2. Generate candidates at each historical point
3. Label outcomes using the LabelEngine (same as live execution)
4. Prepare training data for the ML pipeline

This is the ONLY way to get valid training data.
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import structlog

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters import TradeLockerAdapter
from src.analysis import MarketAnalyzer
from src.scoring import ConfluenceScorer, TradeSetup
from src.ai.features import FeatureExtractor
from src.ai.label_engine import LabelEngine, ExecutionConfig, CandidateLabel, TradeOutcome

logger = structlog.get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    symbols: List[str]
    timeframe: str = "M15"
    lookback_bars: int = 500  # History to fetch per symbol
    analysis_lookback: int = 200  # Bars needed for analysis
    min_confluence_score: int = 55  # Minimum score to generate candidate
    skip_bars_after_signal: int = 10  # Skip bars after signal to avoid overlap


@dataclass
class BacktestResult:
    """Result from backtesting."""
    candidates: List[Dict]
    labels: List[CandidateLabel]
    features: np.ndarray
    labels_class: np.ndarray  # 0=LOSS, 1=WIN, 2=TIMEOUT
    labels_r: np.ndarray  # R-multiples
    timestamps: List[datetime]
    feature_names: List[str]
    statistics: Dict


class BacktestEngine:
    """Generates training data by backtesting on real TradeLocker data.

    Example:
        engine = BacktestEngine(broker)
        result = engine.run()
        # Use result.features, result.labels_class, result.labels_r for training
    """

    def __init__(
        self,
        broker: TradeLockerAdapter,
        config: Optional[BacktestConfig] = None,
        execution_config: Optional[ExecutionConfig] = None,
    ):
        """Initialize backtest engine.

        Args:
            broker: Connected TradeLocker adapter
            config: Backtest configuration
            execution_config: Execution config (MUST match live trading)
        """
        self.broker = broker
        # Default to all 28 major and cross forex pairs
        default_symbols = [
            # Majors (7)
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
            # EUR crosses (6)
            "EURGBP", "EURJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF",
            # GBP crosses (5)
            "GBPJPY", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
            # AUD/NZD crosses (4)
            "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF",
            # NZD crosses (2)
            "NZDCAD", "NZDCHF",
            # CAD/CHF/JPY crosses (4)
            "CADJPY", "CHFJPY", "CADCHF",
        ]
        self.config = config or BacktestConfig(symbols=default_symbols)
        self.execution_config = execution_config or ExecutionConfig()

        # Initialize components
        self.analyzer = MarketAnalyzer()
        self.scorer = ConfluenceScorer()
        self.feature_extractor = FeatureExtractor()
        self.label_engine = LabelEngine(self.execution_config)

    def run(self) -> BacktestResult:
        """Run backtest on all configured symbols.

        Returns:
            BacktestResult with features, labels, and statistics
        """
        logger.info(
            "Starting backtest",
            symbols=len(self.config.symbols),
            timeframe=self.config.timeframe,
            lookback=self.config.lookback_bars,
        )

        all_candidates = []
        all_labels = []
        all_features = []
        all_timestamps = []

        for symbol in self.config.symbols:
            try:
                candidates, labels, features, timestamps = self._backtest_symbol(symbol)
                all_candidates.extend(candidates)
                all_labels.extend(labels)
                all_features.extend(features)
                all_timestamps.extend(timestamps)

                wins = sum(1 for l in labels if l.outcome == TradeOutcome.WIN)
                losses = sum(1 for l in labels if l.outcome == TradeOutcome.LOSS)
                timeouts = sum(1 for l in labels if l.outcome == TradeOutcome.TIMEOUT)

                logger.info(
                    f"Backtested {symbol}",
                    candidates=len(labels),
                    wins=wins,
                    losses=losses,
                    timeouts=timeouts,
                )
            except Exception as e:
                logger.error(f"Failed to backtest {symbol}", error=str(e))
                continue

        if not all_labels:
            raise ValueError("No valid candidates generated")

        # Convert to arrays
        features_array = np.vstack(all_features)
        labels_class = np.array([l.outcome_class for l in all_labels])
        labels_r = np.array([l.r_multiple for l in all_labels])

        # Sort by timestamp (required for time-based splits)
        sort_idx = np.argsort([t.timestamp() for t in all_timestamps])
        features_array = features_array[sort_idx]
        labels_class = labels_class[sort_idx]
        labels_r = labels_r[sort_idx]
        all_timestamps = [all_timestamps[i] for i in sort_idx]
        all_labels = [all_labels[i] for i in sort_idx]

        # Calculate statistics
        stats = self._calculate_statistics(all_labels)

        logger.info(
            "Backtest complete",
            total_candidates=len(all_labels),
            win_rate=stats["win_rate"],
            avg_r=stats["avg_r"],
            profit_factor=stats.get("profit_factor", 0),
        )

        return BacktestResult(
            candidates=all_candidates,
            labels=all_labels,
            features=features_array,
            labels_class=labels_class,
            labels_r=labels_r,
            timestamps=all_timestamps,
            feature_names=self.feature_extractor.feature_names,
            statistics=stats,
        )

    def _backtest_symbol(
        self,
        symbol: str,
    ) -> Tuple[List[Dict], List[CandidateLabel], List[np.ndarray], List[datetime]]:
        """Backtest a single symbol.

        Args:
            symbol: Symbol to backtest

        Returns:
            Tuple of (candidates, labels, features, timestamps)
        """
        # Fetch historical data
        candles = self.broker.get_candles(
            symbol,
            self.config.timeframe,
            limit=self.config.lookback_bars,
        )

        if not candles or len(candles) < self.config.analysis_lookback + 100:
            logger.warning(f"Insufficient data for {symbol}: {len(candles) if candles else 0} bars")
            return [], [], [], []

        # Convert candles to arrays
        opens = np.array([float(c.open) for c in candles])
        highs = np.array([float(c.high) for c in candles])
        lows = np.array([float(c.low) for c in candles])
        closes = np.array([float(c.close) for c in candles])
        timestamps = [c.timestamp for c in candles]

        candidates = []
        labels = []
        features = []
        signal_times = []

        # Walk forward through history
        i = self.config.analysis_lookback

        while i < len(candles) - self.execution_config.timeout_bars:
            # Get historical data up to this point (NO FUTURE DATA)
            hist_opens = opens[:i+1]
            hist_highs = highs[:i+1]
            hist_lows = lows[:i+1]
            hist_closes = closes[:i+1]
            hist_timestamps = timestamps[:i+1]

            # Run analysis
            try:
                market_view = self.analyzer.analyze(
                    symbol,
                    self.config.timeframe,
                    hist_opens,
                    hist_highs,
                    hist_lows,
                    hist_closes,
                    hist_timestamps,
                )
            except Exception as e:
                logger.debug(f"Analysis failed at bar {i}: {e}")
                i += 1
                continue

            # Generate setups
            setups = self.scorer.generate_setups(market_view)

            for setup in setups:
                if setup.confluence_score < self.config.min_confluence_score:
                    continue

                # Extract features at signal time
                feat_vec = self.feature_extractor.extract(setup, market_view)

                # Get entry price from setup
                entry_price = (setup.entry_zone[0] + setup.entry_zone[1]) / 2 if setup.entry_zone else closes[i]

                # Prepare future candles for label engine
                future_start = i + 1
                future_end = min(i + 1 + self.execution_config.timeout_bars, len(candles))
                future_candles = [
                    {
                        "open": opens[j],
                        "high": highs[j],
                        "low": lows[j],
                        "close": closes[j],
                        "time": timestamps[j],
                    }
                    for j in range(future_start, future_end)
                ]

                if len(future_candles) < 10:
                    continue

                # Generate label using label engine (same as live execution)
                label = self.label_engine.generate_label(
                    candidate_id=f"{symbol}_{timestamps[i].strftime('%Y%m%d%H%M')}_{setup.direction}",
                    symbol=symbol,
                    direction=setup.direction,
                    signal_time=timestamps[i],
                    entry_price=entry_price,
                    stop_price=setup.stop_price,
                    tp_targets=setup.tp_targets,
                    future_candles=future_candles,
                )

                # Store results
                candidates.append({
                    "id": label.candidate_id,
                    "symbol": symbol,
                    "direction": setup.direction,
                    "signal_time": timestamps[i],
                    "entry_price": entry_price,
                    "stop_price": setup.stop_price,
                    "confluence_score": setup.confluence_score,
                    "reasons": setup.reasons,
                })
                labels.append(label)
                features.append(feat_vec.feature_array)
                signal_times.append(timestamps[i])

                # Skip ahead to avoid overlapping signals
                i += self.config.skip_bars_after_signal
                break  # Only take first valid setup per bar

            i += 1

        return candidates, labels, features, signal_times

    def _calculate_statistics(self, labels: List[CandidateLabel]) -> Dict:
        """Calculate backtest statistics.

        Args:
            labels: List of candidate labels

        Returns:
            Dict with statistics
        """
        if not labels:
            return {}

        wins = [l for l in labels if l.outcome == TradeOutcome.WIN]
        losses = [l for l in labels if l.outcome == TradeOutcome.LOSS]
        timeouts = [l for l in labels if l.outcome == TradeOutcome.TIMEOUT]

        total_r = sum(l.r_multiple for l in labels)
        avg_r = total_r / len(labels)

        win_rate = len(wins) / len(labels) if labels else 0
        timeout_rate = len(timeouts) / len(labels) if labels else 0

        # Profit factor
        gross_wins = sum(l.r_multiple for l in wins) if wins else 0
        gross_losses = abs(sum(l.r_multiple for l in losses)) if losses else 0
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        # Average excursions
        avg_mae = np.mean([l.max_adverse_excursion for l in labels])
        avg_mfe = np.mean([l.max_favorable_excursion for l in labels])

        return {
            "total_candidates": len(labels),
            "wins": len(wins),
            "losses": len(losses),
            "timeouts": len(timeouts),
            "win_rate": win_rate,
            "timeout_rate": timeout_rate,
            "total_r": total_r,
            "avg_r": avg_r,
            "avg_win_r": np.mean([l.r_multiple for l in wins]) if wins else 0,
            "avg_loss_r": np.mean([l.r_multiple for l in losses]) if losses else 0,
            "profit_factor": profit_factor,
            "avg_bars_to_exit": np.mean([l.bars_to_exit for l in labels]),
            "avg_mae": avg_mae,
            "avg_mfe": avg_mfe,
        }


def run_full_training():
    """Complete training pipeline: backtest + train + save.

    This is the main entry point for training.
    """
    from src.config import load_config
    from src.ai.training_pipeline import TrainingPipeline, TrainingConfig

    # Load config
    config = load_config()

    # Connect to TradeLocker
    logger.info("Connecting to TradeLocker...")
    broker = TradeLockerAdapter(
        environment=config.tradelocker.environment,
        email=config.tradelocker.email,
        password=config.tradelocker.password,
        server=config.tradelocker.server,
        acc_num=0,
        account_id=config.tradelocker.acc_num,
    )

    if not broker.connect():
        raise RuntimeError("Failed to connect to TradeLocker")

    account = broker.get_account()
    logger.info(f"Connected: {account.account_name}, Balance: {account.balance}")

    # Configure backtest - Extended for 1000+ samples
    # All 28 major and cross forex pairs
    all_pairs = [
        # Majors (7)
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
        # EUR crosses (6)
        "EURGBP", "EURJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF",
        # GBP crosses (5)
        "GBPJPY", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
        # AUD/NZD crosses (4)
        "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF",
        # NZD crosses (2)
        "NZDCAD", "NZDCHF",
        # CAD/CHF/JPY crosses (4)
        "CADJPY", "CHFJPY", "CADCHF",
    ]

    backtest_config = BacktestConfig(
        symbols=all_pairs,
        timeframe="M15",
        lookback_bars=2000,  # ~20 days of M15 data for more samples
        min_confluence_score=50,  # Lower threshold to capture more candidates
        skip_bars_after_signal=6,  # 1.5 hours on M15 - tighter to get more samples
    )

    # Execution config must match live trading
    execution_config = ExecutionConfig(
        entry_type="market",
        entry_slippage_pips=0.5,
        spread_pips=1.2,
        default_tp_r=2.0,
        timeout_bars=96,  # 24 hours on M15
    )

    # Run backtest
    logger.info("Running backtest...")
    backtest_engine = BacktestEngine(broker, backtest_config, execution_config)
    result = backtest_engine.run()

    logger.info("Backtest Statistics:")
    for key, value in result.statistics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    if len(result.labels) < 100:
        logger.error(f"Insufficient training data: {len(result.labels)} samples (need 100+)")
        logger.info("Try lowering min_confluence_score or adding more symbols")
        return None

    if len(result.labels) < 500:
        logger.warning(f"Low training data: {len(result.labels)} samples (target: 1000+)")
        logger.info("Model may underperform. Consider running extended backtest.")

    # Train model with config optimized for dataset size
    logger.info("Training model...")
    n_samples = len(result.labels)

    # Adjust hyperparameters based on dataset size
    if n_samples >= 1000:
        # Large dataset - can use more complex model
        n_estimators = 200
        max_depth = 8
        learning_rate = 0.05
    elif n_samples >= 500:
        # Medium dataset
        n_estimators = 150
        max_depth = 6
        learning_rate = 0.08
    else:
        # Small dataset - prevent overfitting
        n_estimators = 100
        max_depth = 4
        learning_rate = 0.1

    training_config = TrainingConfig(
        model_type="xgboost",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        train_ratio=0.65,
        val_ratio=0.20,
        test_ratio=0.15,
        calibrate=True,
        feature_version="v2.0.0",  # Updated version with fixed features
    )

    pipeline = TrainingPipeline(training_config)
    pipeline.load_data(
        features=result.features,
        labels_class=result.labels_class,
        labels_r=result.labels_r,
        timestamps=result.timestamps,
        feature_names=result.feature_names,
    )

    # Walk-forward validation
    logger.info("Running walk-forward validation...")
    wf_results = pipeline.walk_forward_validate()

    # Train final model
    logger.info("Training final model...")
    metrics = pipeline.train()

    logger.info("Training Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Save model
    model_dir = os.environ.get("MODEL_PATH", "/app/models")
    os.makedirs(model_dir, exist_ok=True)

    version = f"v1.0.0-{datetime.now().strftime('%Y%m%d')}"
    saved_paths = pipeline.save(model_dir, version)

    logger.info(f"Model saved to {model_dir}")
    logger.info(f"Version: {version}")

    # Disconnect
    broker.disconnect()

    return {
        "version": version,
        "metrics": metrics,
        "statistics": result.statistics,
        "samples": len(result.labels),
        "model_paths": saved_paths,
    }


if __name__ == "__main__":
    run_full_training()
