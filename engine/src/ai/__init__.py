"""AI Decision Gate for AG Analyzer v3.0."""

from .gate import AIGate, GateConfig, GateDecision, GateDecisionType
from .features import FeatureExtractor, MarketView, FEATURE_VERSION, N_TOTAL_FEATURES
from .label_engine import LabelEngine, ExecutionConfig, LabelRecord, TradeOutcome
from .training_pipeline import TrainingPipeline, TrainingConfig, TrainingResult
from .backtest import BacktestEngine, BacktestConfig, BacktestResult, BacktestSample
from .utils import pip_size, get_spread_pips, spread_in_price, SPREAD_TABLE

__all__ = [
    # Gate
    "AIGate", "GateConfig", "GateDecision", "GateDecisionType",
    # Features
    "FeatureExtractor", "MarketView", "FEATURE_VERSION", "N_TOTAL_FEATURES",
    # Label Engine
    "LabelEngine", "ExecutionConfig", "LabelRecord", "TradeOutcome",
    # Training
    "TrainingPipeline", "TrainingConfig", "TrainingResult",
    # Backtest
    "BacktestEngine", "BacktestConfig", "BacktestResult", "BacktestSample",
    # Utils
    "pip_size", "get_spread_pips", "spread_in_price", "SPREAD_TABLE",
]
