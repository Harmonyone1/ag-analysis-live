"""AI Decision Gate for AG Analyzer."""

from .gate import AIGate, GateDecision, GateConfig, GateDecisionType
from .features import FeatureExtractor
from .label_engine import LabelEngine, ExecutionConfig, CandidateLabel, TradeOutcome
from .training_pipeline import TrainingPipeline, TrainingConfig, TrainedModel
from .backtest import BacktestEngine, BacktestConfig, BacktestResult

__all__ = [
    # Gate
    "AIGate", "GateDecision", "GateConfig", "GateDecisionType",
    # Features
    "FeatureExtractor",
    # Label Engine
    "LabelEngine", "ExecutionConfig", "CandidateLabel", "TradeOutcome",
    # Training
    "TrainingPipeline", "TrainingConfig", "TrainedModel",
    # Backtest
    "BacktestEngine", "BacktestConfig", "BacktestResult",
]
