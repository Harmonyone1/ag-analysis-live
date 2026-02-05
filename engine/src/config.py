"""Configuration management for AG Analyzer Engine."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    name: str
    user: str
    password: str
    host: str
    port: int

    @property
    def url(self) -> str:
        """SQLAlchemy-compatible connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class TradeLockerConfig:
    """TradeLocker API configuration."""
    environment: str
    email: str
    password: str
    server: str
    acc_num: int


@dataclass
class BotConfig:
    """Trading bot configuration."""
    mode: str  # 'paper' or 'live'
    trading_enabled: bool
    log_level: str


@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_risk_per_trade: float
    max_daily_loss: float
    max_open_positions: int
    max_correlated_exposure: float


@dataclass
class AIConfig:
    """AI model configuration."""
    model_path: str
    active_model_version: str
    min_confluence_score: int
    min_ai_probability: float
    min_expected_value: float


@dataclass
class Config:
    """Complete application configuration."""
    database: DatabaseConfig
    tradelocker: TradeLockerConfig
    bot: BotConfig
    risk: RiskConfig
    ai: AIConfig


def load_config(env_path: str = ".env") -> Config:
    """Load configuration from environment variables.

    Args:
        env_path: Path to .env file

    Returns:
        Complete Config object

    Raises:
        EnvironmentError: If required variables are missing
    """
    load_dotenv(env_path)

    required_vars = [
        "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT",
        "TL_ENVIRONMENT", "TL_EMAIL", "TL_PASSWORD", "TL_SERVER", "TL_ACC_NUM",
    ]

    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    return Config(
        database=DatabaseConfig(
            name=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
        ),
        tradelocker=TradeLockerConfig(
            environment=os.getenv("TL_ENVIRONMENT"),
            email=os.getenv("TL_EMAIL"),
            password=os.getenv("TL_PASSWORD"),
            server=os.getenv("TL_SERVER"),
            acc_num=int(os.getenv("TL_ACC_NUM", "0")),
        ),
        bot=BotConfig(
            mode=os.getenv("BOT_MODE", "paper"),
            trading_enabled=os.getenv("TRADING_ENABLED", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        ),
        risk=RiskConfig(
            max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", "0.01")),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "0.03")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "5")),
            max_correlated_exposure=float(os.getenv("MAX_CORRELATED_EXPOSURE", "0.02")),
        ),
        ai=AIConfig(
            model_path=os.getenv("MODEL_PATH", "./models"),
            active_model_version=os.getenv("ACTIVE_MODEL_VERSION", "v1.0.0"),
            min_confluence_score=int(os.getenv("MIN_CONFLUENCE_SCORE", "60")),
            min_ai_probability=float(os.getenv("MIN_AI_PROBABILITY", "0.55")),
            min_expected_value=float(os.getenv("MIN_EXPECTED_VALUE", "0.15")),
        ),
    )
