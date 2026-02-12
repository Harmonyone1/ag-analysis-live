#!/usr/bin/env python
"""Collect additional historical data for commodities, indices, and crypto.

Saves separate .npz files per asset class for easy reuse:
  - data/commodities_training.npz  (XAUUSD, XAGUSD, USOIL, UKOIL, NGAS)
  - data/indices_training.npz      (NAS100, SPX500, US30, DE40, UK100, JP225)
  - data/crypto_training.npz       (BTCUSD, ETHUSD, ADAUSD, DOGEUSD)
  - data/all_extra_training.npz    (everything combined)

Usage:
    cd D:/ag-analysis-live/engine
    C:/Users/DavidPorter/miniconda3/python.exe scripts/collect_extra_data.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("collect_extra")

ASSET_CLASSES = {
    "commodities": ["XAUUSD", "XAGUSD", "USOIL", "UKOIL", "NGAS", "XPTUSD"],
    "indices": ["NAS100", "SPX500", "US30", "DE40", "UK100", "JP225", "AUS200", "F40"],
    "crypto": ["BTCUSD", "ETHUSD", "ADAUSD", "DOGEUSD", "BCHUSD", "XLMUSD"],
}

ALL_EXTRA_SYMBOLS = []
for syms in ASSET_CLASSES.values():
    ALL_EXTRA_SYMBOLS.extend(syms)


def _fetch_history_with_retry(api, instrument_id, resolution, lookback_days, max_retries=3):
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


def fetch_raw_data(symbols, lookback_bars):
    """Fetch raw OHLCV for symbols, return dict of {symbol: {m15: df, h1: df, h4: df}}."""
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
    logger.info("Connected: %s", api.account_name)

    # Resolve instrument IDs
    instrument_ids = {}
    for symbol in symbols:
        try:
            instrument_ids[symbol] = api.get_instrument_id_from_symbol_name(symbol)
        except Exception as exc:
            logger.warning("Instrument not found: %s (%s)", symbol, exc)
    logger.info("Resolved %d/%d instruments", len(instrument_ids), len(symbols))

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

        symbol_data = {}
        for tf_name, spec in tf_specs.items():
            days = min(max(1, int(spec["bars"] * spec["minutes"] / 1440) + 1), 300)
            df = _fetch_history_with_retry(api, iid, spec["resolution"], days)
            if len(df) > spec["bars"]:
                df = df.tail(spec["bars"]).reset_index(drop=True)
            symbol_data[tf_name] = df
            time.sleep(2)

        if symbol_data.get("m15") is not None and not symbol_data["m15"].empty:
            candle_data[symbol] = symbol_data
            logger.info("  %s: M15=%d H1=%d H4=%d",
                         symbol, len(symbol_data["m15"]),
                         len(symbol_data.get("h1", pd.DataFrame())),
                         len(symbol_data.get("h4", pd.DataFrame())))
        else:
            logger.warning("  %s: No M15 data - skipped", symbol)

    elapsed = time.time() - t0
    logger.info("Fetch complete: %d/%d symbols in %.1fs", len(candle_data), n_total, elapsed)
    return candle_data


def run_backtest_and_save(candle_data, output_path, label="data"):
    """Run backtest on candle data and save features+labels to .npz."""
    from src.ai.features import FeatureExtractor, N_TOTAL_FEATURES
    from src.ai.backtest import BacktestConfig, BacktestEngine

    bt_config = BacktestConfig(
        symbols=list(candle_data.keys()),
        lookback_bars=20000,
        fetch_h1=True,
        fetch_h4=True,
        signal_interval=4,
    )
    engine = BacktestEngine(bt_config)
    result = engine.run(candle_data)

    if len(result.samples) < 10:
        logger.warning("Only %d samples for %s - skipping save", len(result.samples), label)
        return

    X = np.array([s.features for s in result.samples], dtype=np.float32)
    y = np.array([s.label for s in result.samples], dtype=np.int32)
    symbols_arr = np.array([s.symbol for s in result.samples])
    directions = np.array([s.direction for s in result.samples])
    outcomes = np.array([s.outcome for s in result.samples])

    timestamps = np.zeros(len(result.samples), dtype=np.int64)
    for i, s in enumerate(result.samples):
        sym_data = candle_data.get(s.symbol, {}).get("m15")
        if sym_data is not None and s.bar_index < len(sym_data):
            ts_val = sym_data["timestamp"].iloc[s.bar_index]
            if ts_val < 1e12:
                ts_val = int(ts_val * 1000)
            timestamps[i] = int(ts_val)

    feature_names = FeatureExtractor().feature_names

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X, y=y, timestamps=timestamps,
        symbols=symbols_arr, directions=directions,
        outcomes=outcomes, feature_names=np.array(feature_names),
    )

    file_mb = output_path.stat().st_size / (1024 * 1024)
    valid_ts = timestamps[timestamps > 0]
    date_range = ""
    if len(valid_ts) > 0:
        d0 = datetime.fromtimestamp(valid_ts[0] / 1000, tz=timezone.utc).date()
        d1 = datetime.fromtimestamp(valid_ts[-1] / 1000, tz=timezone.utc).date()
        date_range = f"{d0} to {d1}"

    logger.info("  Saved %s: %d samples, %d features, WR=%.1f%%, %.1f MB, %s",
                 output_path.name, len(y), X.shape[1], y.mean() * 100, file_mb, date_range)


def save_raw_candles(candle_data, output_path):
    """Save raw OHLCV data as .npz for reuse without re-fetching."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {}
    for symbol, tf_data in candle_data.items():
        for tf_name, df in tf_data.items():
            if df is not None and not df.empty:
                key = f"{symbol}_{tf_name}"
                # Save as structured array
                save_dict[key] = df.to_numpy()
                save_dict[f"{key}_columns"] = np.array(list(df.columns))

    save_dict["symbols"] = np.array(list(candle_data.keys()))
    np.savez_compressed(output_path, **save_dict)
    file_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("  Saved raw candles: %s (%.1f MB, %d symbols)", output_path.name, file_mb, len(candle_data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-bars", type=int, default=20000)
    parser.add_argument("--raw-only", action="store_true",
                        help="Save raw candles only (skip backtest/labeling)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("EXTRA DATA COLLECTION: Commodities, Indices, Crypto")
    logger.info("=" * 70)

    # Fetch all data at once
    candle_data = fetch_raw_data(ALL_EXTRA_SYMBOLS, args.lookback_bars)

    if not candle_data:
        logger.error("No data fetched - aborting")
        sys.exit(1)

    # Save raw candles (always, for reuse)
    save_raw_candles(candle_data, "data/extra_raw_candles.npz")

    if args.raw_only:
        logger.info("Raw-only mode - skipping backtest")
        return

    # Run backtest + save per asset class
    for class_name, class_symbols in ASSET_CLASSES.items():
        class_data = {s: candle_data[s] for s in class_symbols if s in candle_data}
        if class_data:
            logger.info("")
            logger.info("Processing %s (%d symbols)...", class_name, len(class_data))
            run_backtest_and_save(class_data, f"data/{class_name}_training.npz", class_name)

    # Combined file
    logger.info("")
    logger.info("Processing ALL extras combined...")
    run_backtest_and_save(candle_data, "data/all_extra_training.npz", "all_extra")

    logger.info("")
    logger.info("=" * 70)
    logger.info("EXTRA DATA COLLECTION COMPLETE")
    logger.info("=" * 70)
    logger.info("  Files saved to: D:\\ag-analysis-live\\engine\\data\\")


if __name__ == "__main__":
    main()
