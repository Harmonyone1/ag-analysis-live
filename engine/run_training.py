#!/usr/bin/env python
"""Training entry point for AI Gate v3.0.0.

Usage:
    python run_training.py
    python run_training.py --lookback-bars 20000
    python run_training.py --dry-run  # synthetic data for testing

Fetches 200+ days of M15/H1/H4 data for all configured symbols,
runs a walk-forward backtest to generate labelled samples, then
trains an XGBoost + Platt-calibrated gate model.
"""

import sys
import os
import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Set up paths - engine/ is the working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("run_training")

# ------------------------------------------------------------------
# Symbol universe
# ------------------------------------------------------------------
DEFAULT_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "EURCHF", "EURCAD", "EURNZD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    "AUDCAD", "AUDCHF", "AUDNZD", "AUDJPY",
    "CADJPY", "CHFJPY", "CADCHF",
    "NZDCAD", "NZDJPY", "NZDCHF",
]


def candles_to_dataframe(candles) -> pd.DataFrame:
    """Convert a list of Candle objects (from adapter) to a DataFrame.

    Expected columns: timestamp (int64 ms), open, high, low, close, volume.
    """
    if not candles:
        return pd.DataFrame()

    rows = []
    for c in candles:
        rows.append({
            "timestamp": int(c.timestamp.timestamp() * 1000),
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AI Gate v3.0.0 model")
    parser.add_argument(
        "--lookback-bars",
        type=int,
        default=20000,
        help="Number of M15 bars to fetch per symbol (default: 20000 ~ 208 days)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override symbol list (default: 28 FX pairs)",
    )
    parser.add_argument(
        "--no-h1",
        action="store_true",
        help="Skip H1 data (faster, fewer features)",
    )
    parser.add_argument(
        "--no-h4",
        action="store_true",
        help="Skip H4 data (faster, fewer features)",
    )
    parser.add_argument(
        "--calibration",
        choices=["platt", "isotonic"],
        default="platt",
        help="Calibration method (default: platt)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Output directory for trained model (default: ../models)",
    )
    parser.add_argument(
        "--signal-interval",
        type=int,
        default=4,
        help="Generate a signal every N M15 bars (default: 4 = hourly)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip data fetch, use dummy data for pipeline testing",
    )
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS
    fetch_h1 = not args.no_h1
    fetch_h4 = not args.no_h4
    model_dir = args.model_dir or os.path.join(
        os.path.dirname(__file__), '..', 'models'
    )
    os.makedirs(model_dir, exist_ok=True)

    # Lazy imports after path setup
    from src.ai.features import FEATURE_VERSION, N_TOTAL_FEATURES
    from src.ai.backtest import BacktestConfig, BacktestEngine
    from src.ai.training_pipeline import TrainingConfig, TrainingPipeline
    from src.ai.utils import SPREAD_TABLE

    logger.info("=" * 70)
    logger.info("AI GATE MODEL TRAINING v3.0.0")
    logger.info("=" * 70)
    logger.info("  Feature version:  %s", FEATURE_VERSION)
    logger.info("  Total features:   %d", N_TOTAL_FEATURES)
    logger.info("  Symbols:          %d", len(symbols))
    logger.info("  Lookback bars:    %d M15 (~%d days)",
                args.lookback_bars, args.lookback_bars * 15 // (60 * 24))
    logger.info("  Fetch H1:         %s", fetch_h1)
    logger.info("  Fetch H4:         %s", fetch_h4)
    logger.info("  Calibration:      %s", args.calibration)
    logger.info("  Signal interval:  every %d bars", args.signal_interval)
    logger.info("  Model output:     %s", model_dir)
    logger.info("  Spread table:     %d symbols configured", len(SPREAD_TABLE))
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1: Data fetching
    # ------------------------------------------------------------------
    if args.dry_run:
        logger.info("DRY RUN: generating synthetic data")
        candle_data = _generate_dummy_data(symbols[:3], args.lookback_bars)
    else:
        candle_data = _fetch_all_data(
            symbols,
            lookback_bars=args.lookback_bars,
            fetch_h1=fetch_h1,
            fetch_h4=fetch_h4,
        )

    if not candle_data:
        logger.error("No data fetched - aborting training")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2: Backtest -> labelled samples
    # ------------------------------------------------------------------
    bt_config = BacktestConfig(
        symbols=list(candle_data.keys()),
        lookback_bars=args.lookback_bars,
        fetch_h1=fetch_h1,
        fetch_h4=fetch_h4,
        signal_interval=args.signal_interval,
    )
    engine = BacktestEngine(bt_config)
    result = engine.run(candle_data)

    if len(result.samples) < 100:
        logger.error(
            "Only %d samples generated - need at least 100 for training. Aborting.",
            len(result.samples),
        )
        sys.exit(1)

    # Build X, y matrices
    X = np.array([s.features for s in result.samples], dtype=np.float32)
    y = np.array([s.label for s in result.samples], dtype=np.int32)
    symbols_arr = np.array([s.symbol for s in result.samples])

    logger.info(
        "Training data: %d samples, %d features, win_rate=%.2f%%",
        len(y),
        X.shape[1],
        100 * y.mean(),
    )

    assert X.shape[1] == N_TOTAL_FEATURES, (
        f"Feature count mismatch: got {X.shape[1]}, expected {N_TOTAL_FEATURES}"
    )

    # ------------------------------------------------------------------
    # Phase 3: Train model
    # ------------------------------------------------------------------
    train_config = TrainingConfig(
        calibration_method=args.calibration,
        feature_version=FEATURE_VERSION,
        model_dir=model_dir,
    )
    pipeline = TrainingPipeline(train_config)

    from src.ai.features import FeatureExtractor
    feature_names = FeatureExtractor().feature_names

    training_result = pipeline.train(
        X,
        y,
        symbols=symbols_arr,
        feature_names=feature_names,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("  Model saved to:       %s", training_result.model_path)
    logger.info("  Total samples:        %d", training_result.n_samples)
    logger.info("  Feature version:      %s", training_result.feature_version)
    logger.info("  Calibration method:   %s", training_result.calibration_method)
    logger.info("  Train AUC:            %.3f", training_result.train_auc)
    logger.info("  Val AUC:              %.3f", training_result.val_auc)
    logger.info("  Raw Brier:            %.4f", training_result.raw_brier)
    logger.info("  Calibrated Brier:     %.4f", training_result.calibrated_brier)
    logger.info("=" * 70)

    # Per-fold summary
    for fold in training_result.walk_forward_metrics:
        logger.info(
            "  Fold %d: train_AUC=%.3f val_AUC=%.3f gap=%.3f val_Brier=%.4f",
            fold["fold"],
            fold["train_auc"],
            fold["val_auc"],
            fold["overfit_gap"],
            fold["val_brier"],
        )


# ------------------------------------------------------------------
# Data fetching helper
# ------------------------------------------------------------------
def _fetch_history_with_retry(
    api,
    instrument_id: int,
    resolution: str,
    lookback_days: int,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch price history from TLAPI with retry + exponential backoff."""
    lookback = f"{lookback_days}D"
    for attempt in range(max_retries):
        try:
            df = api.get_price_history(
                instrument_id=instrument_id,
                resolution=resolution,
                lookback_period=lookback,
            )
            if not isinstance(df, pd.DataFrame) or df.empty:
                return pd.DataFrame()
            # Rename columns to standard names
            col_map = {"t": "timestamp", "o": "open", "h": "high",
                       "l": "low", "c": "close", "v": "volume"}
            df = df.rename(columns=col_map)
            return df
        except Exception as exc:
            wait = 2 ** (attempt + 1)
            logger.warning(
                "Fetch attempt %d/%d failed: %s. Retrying in %ds...",
                attempt + 1, max_retries, str(exc)[:200], wait,
            )
            time.sleep(wait)
    return pd.DataFrame()


def _fetch_all_data(
    symbols: list,
    *,
    lookback_bars: int,
    fetch_h1: bool,
    fetch_h4: bool,
) -> dict:
    """Fetch MTF data for all symbols directly via TLAPI.

    Uses the TLAPI library directly (skips the adapter's Candle conversion)
    with retry logic and 2s delays to avoid 429 rate limiting.
    """
    from tradelocker import TLAPI

    logger.info("Connecting to TradeLocker API (direct)...")
    api = TLAPI(
        environment=os.environ.get("TL_ENVIRONMENT", "https://demo.tradelocker.com"),
        username=os.environ.get("TL_EMAIL", ""),
        password=os.environ.get("TL_PASSWORD", ""),
        server=os.environ.get("TL_SERVER", ""),
        acc_num=int(os.environ.get("TL_ACC_NUM", 0)),
        log_level="warning",
    )
    logger.info("Connected to TradeLocker: account=%s", api.account_name)

    # Pre-resolve all instrument IDs (one API call, cached)
    instrument_ids = {}
    for symbol in symbols:
        try:
            instrument_ids[symbol] = api.get_instrument_id_from_symbol_name(symbol)
        except Exception as exc:
            logger.warning("Instrument not found: %s (%s)", symbol, exc)
    logger.info("Resolved %d/%d instrument IDs", len(instrument_ids), len(symbols))

    # Calculate lookback days per timeframe
    tf_specs = {
        "m15": {"resolution": "15m", "minutes": 15, "bars": lookback_bars},
        "h1": {"resolution": "1H", "minutes": 60, "bars": min(5000, lookback_bars // 4)},
        "h4": {"resolution": "4H", "minutes": 240, "bars": min(1250, lookback_bars // 16)},
    }

    candle_data = {}
    n_total = len(instrument_ids)
    t0 = time.time()

    for i, (symbol, iid) in enumerate(instrument_ids.items()):
        logger.info("Fetching %s (%d/%d)...", symbol, i + 1, n_total)

        # M15
        m15_spec = tf_specs["m15"]
        m15_days = max(1, int(m15_spec["bars"] * m15_spec["minutes"] / 1440) + 1)
        m15_days = min(m15_days, 300)
        m15_df = _fetch_history_with_retry(api, iid, m15_spec["resolution"], m15_days)
        time.sleep(2)  # Rate limit

        if m15_df.empty:
            logger.warning("No M15 data for %s - skipping", symbol)
            continue

        # Trim to requested bars
        if len(m15_df) > lookback_bars:
            m15_df = m15_df.tail(lookback_bars).reset_index(drop=True)

        symbol_data = {"m15": m15_df}

        # H1
        if fetch_h1:
            h1_spec = tf_specs["h1"]
            h1_days = max(1, int(h1_spec["bars"] * h1_spec["minutes"] / 1440) + 1)
            h1_days = min(h1_days, 300)
            h1_df = _fetch_history_with_retry(api, iid, h1_spec["resolution"], h1_days)
            if len(h1_df) > h1_spec["bars"]:
                h1_df = h1_df.tail(h1_spec["bars"]).reset_index(drop=True)
            symbol_data["h1"] = h1_df
            time.sleep(2)
        else:
            symbol_data["h1"] = pd.DataFrame()

        # H4
        if fetch_h4:
            h4_spec = tf_specs["h4"]
            h4_days = max(1, int(h4_spec["bars"] * h4_spec["minutes"] / 1440) + 1)
            h4_days = min(h4_days, 300)
            h4_df = _fetch_history_with_retry(api, iid, h4_spec["resolution"], h4_days)
            if len(h4_df) > h4_spec["bars"]:
                h4_df = h4_df.tail(h4_spec["bars"]).reset_index(drop=True)
            symbol_data["h4"] = h4_df
            time.sleep(2)
        else:
            symbol_data["h4"] = pd.DataFrame()

        candle_data[symbol] = symbol_data
        logger.info(
            "  %s: M15=%d H1=%d H4=%d bars",
            symbol,
            len(symbol_data["m15"]),
            len(symbol_data.get("h1", pd.DataFrame())),
            len(symbol_data.get("h4", pd.DataFrame())),
        )

    elapsed = time.time() - t0
    logger.info(
        "Data fetch complete: %d/%d symbols in %.1fs",
        len(candle_data),
        n_total,
        elapsed,
    )
    return candle_data


# ------------------------------------------------------------------
# Dummy data for dry-run / testing
# ------------------------------------------------------------------
def _generate_dummy_data(
    symbols: list,
    n_bars: int,
) -> dict:
    """Generate synthetic OHLCV data for pipeline testing."""

    candle_data = {}
    for symbol in symbols:
        base_price = 1.1000 if "JPY" not in symbol else 150.0
        noise = np.random.randn(n_bars) * 0.001

        closes = base_price + np.cumsum(noise)
        highs = closes + np.abs(np.random.randn(n_bars) * 0.0005)
        lows = closes - np.abs(np.random.randn(n_bars) * 0.0005)
        opens = closes + np.random.randn(n_bars) * 0.0002
        volumes = np.random.randint(100, 10000, size=n_bars).astype(float)

        now = datetime.now(timezone.utc)
        timestamps = pd.date_range(
            end=now,
            periods=n_bars,
            freq="15min",
        )

        m15_df = pd.DataFrame({
            "timestamp": timestamps.astype(np.int64) // 10**6,  # ms
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })

        # H1: every 4th bar
        h1_df = m15_df.iloc[::4].reset_index(drop=True)
        # H4: every 16th bar
        h4_df = m15_df.iloc[::16].reset_index(drop=True)

        candle_data[symbol] = {
            "m15": m15_df,
            "h1": h1_df,
            "h4": h4_df,
        }

    return candle_data


if __name__ == "__main__":
    main()
