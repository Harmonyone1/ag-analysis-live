#!/usr/bin/env python
"""Collect FX training data and save to .npz for reuse.

Fetches M15/H1/H4 data for all 28 FX pairs from TradeLocker,
runs the backtest labeler to generate feature vectors + labels,
and saves everything to D:/ag-analysis-live/engine/data/fx_training.npz.

Usage:
    cd D:/ag-analysis-live/engine
    py scripts/collect_fx_training_data.py
    py scripts/collect_fx_training_data.py --lookback-bars 10000
    py scripts/collect_fx_training_data.py --symbols EURUSD GBPUSD
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Set up paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("collect_fx_data")

DEFAULT_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "EURCHF", "EURCAD", "EURNZD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    "AUDCAD", "AUDCHF", "AUDNZD", "AUDJPY",
    "CADJPY", "CHFJPY", "CADCHF",
    "NZDCAD", "NZDJPY", "NZDCHF",
]


def _fetch_history_with_retry(api, instrument_id, resolution, lookback_days, max_retries=3):
    """Fetch price history with retry + exponential backoff."""
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
            col_map = {"t": "timestamp", "o": "open", "h": "high",
                       "l": "low", "c": "close", "v": "volume"}
            df = df.rename(columns=col_map)
            return df
        except Exception as exc:
            wait = 2 ** (attempt + 1)
            logger.warning("Fetch attempt %d/%d failed: %s. Retrying in %ds...",
                           attempt + 1, max_retries, str(exc)[:200], wait)
            time.sleep(wait)
    return pd.DataFrame()


def fetch_all_data(symbols, lookback_bars, fetch_h1=True, fetch_h4=True):
    """Fetch MTF data for all symbols from TradeLocker."""
    from tradelocker import TLAPI

    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    logger.info("Connecting to TradeLocker API...")
    api = TLAPI(
        environment=os.environ.get("TL_ENVIRONMENT", "https://demo.tradelocker.com"),
        username=os.environ.get("TL_EMAIL", ""),
        password=os.environ.get("TL_PASSWORD", ""),
        server=os.environ.get("TL_SERVER", ""),
        acc_num=int(os.environ.get("TL_ACC_NUM", 0)),
        log_level="warning",
    )
    logger.info("Connected to TradeLocker: account=%s", api.account_name)

    # Resolve instrument IDs
    instrument_ids = {}
    for symbol in symbols:
        try:
            instrument_ids[symbol] = api.get_instrument_id_from_symbol_name(symbol)
        except Exception as exc:
            logger.warning("Instrument not found: %s (%s)", symbol, exc)
    logger.info("Resolved %d/%d instrument IDs", len(instrument_ids), len(symbols))

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
        m15_days = min(max(1, int(m15_spec["bars"] * m15_spec["minutes"] / 1440) + 1), 300)
        m15_df = _fetch_history_with_retry(api, iid, m15_spec["resolution"], m15_days)
        time.sleep(2)

        if m15_df.empty:
            logger.warning("No M15 data for %s - skipping", symbol)
            continue

        if len(m15_df) > lookback_bars:
            m15_df = m15_df.tail(lookback_bars).reset_index(drop=True)

        symbol_data = {"m15": m15_df}

        # H1
        if fetch_h1:
            h1_spec = tf_specs["h1"]
            h1_days = min(max(1, int(h1_spec["bars"] * h1_spec["minutes"] / 1440) + 1), 300)
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
            h4_days = min(max(1, int(h4_spec["bars"] * h4_spec["minutes"] / 1440) + 1), 300)
            h4_df = _fetch_history_with_retry(api, iid, h4_spec["resolution"], h4_days)
            if len(h4_df) > h4_spec["bars"]:
                h4_df = h4_df.tail(h4_spec["bars"]).reset_index(drop=True)
            symbol_data["h4"] = h4_df
            time.sleep(2)
        else:
            symbol_data["h4"] = pd.DataFrame()

        candle_data[symbol] = symbol_data
        logger.info("  %s: M15=%d H1=%d H4=%d bars",
                     symbol, len(symbol_data["m15"]),
                     len(symbol_data.get("h1", pd.DataFrame())),
                     len(symbol_data.get("h4", pd.DataFrame())))

    elapsed = time.time() - t0
    logger.info("Data fetch complete: %d/%d symbols in %.1fs", len(candle_data), n_total, elapsed)
    return candle_data


def main():
    parser = argparse.ArgumentParser(description="Collect FX training data")
    parser.add_argument("--lookback-bars", type=int, default=20000,
                        help="M15 bars per symbol (default: 20000 ~ 208 days)")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override symbol list")
    parser.add_argument("--signal-interval", type=int, default=4,
                        help="Signal every N bars (default: 4 = hourly)")
    parser.add_argument("--output", default=None,
                        help="Output .npz path (default: data/fx_training.npz)")
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS
    output_path = Path(args.output) if args.output else Path("data/fx_training.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from src.ai.features import FEATURE_VERSION, N_TOTAL_FEATURES, FeatureExtractor
    from src.ai.backtest import BacktestConfig, BacktestEngine

    logger.info("=" * 70)
    logger.info("FX TRAINING DATA COLLECTION")
    logger.info("=" * 70)
    logger.info("  Feature version:  %s", FEATURE_VERSION)
    logger.info("  Total features:   %d", N_TOTAL_FEATURES)
    logger.info("  Symbols:          %d", len(symbols))
    logger.info("  Lookback bars:    %d M15 (~%d days)", args.lookback_bars,
                args.lookback_bars * 15 // (60 * 24))
    logger.info("  Signal interval:  every %d bars", args.signal_interval)
    logger.info("  Output:           %s", output_path.resolve())
    logger.info("=" * 70)

    # Phase 1: Fetch data
    candle_data = fetch_all_data(symbols, args.lookback_bars)
    if not candle_data:
        logger.error("No data fetched - aborting")
        sys.exit(1)

    # Phase 2: Backtest -> labelled samples
    bt_config = BacktestConfig(
        symbols=list(candle_data.keys()),
        lookback_bars=args.lookback_bars,
        fetch_h1=True,
        fetch_h4=True,
        signal_interval=args.signal_interval,
    )
    engine = BacktestEngine(bt_config)
    result = engine.run(candle_data)

    if len(result.samples) < 100:
        logger.error("Only %d samples - need at least 100. Aborting.", len(result.samples))
        sys.exit(1)

    # Build arrays
    X = np.array([s.features for s in result.samples], dtype=np.float32)
    y = np.array([s.label for s in result.samples], dtype=np.int32)
    symbols_arr = np.array([s.symbol for s in result.samples])
    directions = np.array([s.direction for s in result.samples])
    outcomes = np.array([s.outcome for s in result.samples])

    # Extract timestamps from features (bar_index -> we need actual timestamps)
    # The backtest doesn't store timestamps directly, so we reconstruct from candle_data
    timestamps = np.zeros(len(result.samples), dtype=np.int64)
    for i, s in enumerate(result.samples):
        sym_data = candle_data.get(s.symbol, {}).get("m15")
        if sym_data is not None and s.bar_index < len(sym_data):
            ts_val = sym_data["timestamp"].iloc[s.bar_index]
            # Normalize to ms
            if ts_val < 1e12:
                ts_val = int(ts_val * 1000)
            timestamps[i] = int(ts_val)

    feature_names = FeatureExtractor().feature_names

    assert X.shape[1] == N_TOTAL_FEATURES, (
        f"Feature count mismatch: got {X.shape[1]}, expected {N_TOTAL_FEATURES}"
    )

    # Save
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        timestamps=timestamps,
        symbols=symbols_arr,
        directions=directions,
        outcomes=outcomes,
        feature_names=np.array(feature_names),
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info("=" * 70)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("=" * 70)
    logger.info("  Saved to:         %s (%.1f MB)", output_path.resolve(), file_size_mb)
    logger.info("  Total samples:    %d", len(y))
    logger.info("  Features:         %d", X.shape[1])
    logger.info("  Win rate:         %.1f%%", y.mean() * 100)
    logger.info("  Symbols:          %d", len(set(symbols_arr)))
    logger.info("  Date range:       %s to %s",
                datetime.fromtimestamp(timestamps[timestamps > 0][0] / 1000, tz=timezone.utc).date(),
                datetime.fromtimestamp(timestamps[timestamps > 0][-1] / 1000, tz=timezone.utc).date())
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
