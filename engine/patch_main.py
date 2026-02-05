"""Patch script to apply v3 AI Gate changes to main.py on the droplet.

Run this script from the engine directory. It reads main.py, applies
the necessary modifications for v3 MTF data, and writes the result.
"""

import re
import sys

MAIN_PY = "/root/ag-analysis/engine/src/main.py"

with open(MAIN_PY, "r") as f:
    content = f.read()

# --- Patch 1: Add _candle_data and _htf_cache to __init__ ---
# Find the line with _market_views and add after it
old_state = '        self._market_views: Dict[str, object] = {}  # symbol → MarketView from latest scan'
new_state = '''        self._market_views: Dict[str, object] = {}  # symbol → MarketView from latest scan
        self._candle_data: Dict[str, Dict[str, Any]] = {}  # symbol → {m15: {}, h1: {}, h4: {}}
        self._htf_cache_time: Dict[str, datetime] = {}  # symbol → last H1/H4 fetch time'''

if old_state in content:
    content = content.replace(old_state, new_state)
    print("Patch 1: Added _candle_data and _htf_cache_time")
else:
    print("Patch 1: SKIPPED (already applied or marker not found)")

# --- Patch 2: Add _fetch_htf_candles helper method ---
# Insert before _generate_candidates
htf_helper = '''
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

'''

marker = '    async def _generate_candidates(self) -> None:'
if '_fetch_htf_candles' not in content:
    content = content.replace(marker, htf_helper + marker)
    print("Patch 2: Added _fetch_htf_candles method")
else:
    print("Patch 2: SKIPPED (already applied)")

# --- Patch 3: Store M15 candle arrays in _candle_data during _generate_candidates ---
old_m15_block = '''                # Extract OHLC arrays for analyzer
                import numpy as np
                opens = np.array([float(c.open) for c in candles])
                highs = np.array([float(c.high) for c in candles])
                lows = np.array([float(c.low) for c in candles])
                closes = np.array([float(c.close) for c in candles])
                timestamps = np.array([c.timestamp for c in candles])

                # Run analysis
                market_view = self.analyzer.analyze(symbol, "M15", opens, highs, lows, closes, timestamps)
                self._market_views[symbol] = market_view'''

new_m15_block = '''                # Extract OHLC arrays for analyzer
                import numpy as np
                opens = np.array([float(c.open) for c in candles])
                highs = np.array([float(c.high) for c in candles])
                lows = np.array([float(c.low) for c in candles])
                closes = np.array([float(c.close) for c in candles])
                volumes = np.array([float(c.volume) for c in candles])
                timestamps = np.array([c.timestamp for c in candles])

                # Store M15 candle arrays for v3 AI gate
                self._candle_data[symbol] = {
                    "m15": {
                        "opens": opens,
                        "highs": highs,
                        "lows": lows,
                        "closes": closes,
                        "volumes": volumes,
                    }
                }

                # Fetch H1/H4 candles for v3 AI gate (TTL cached)
                self._fetch_htf_candles(symbol)

                # Run analysis
                market_view = self.analyzer.analyze(symbol, "M15", opens, highs, lows, closes, timestamps)
                self._market_views[symbol] = market_view'''

if old_m15_block in content:
    content = content.replace(old_m15_block, new_m15_block)
    print("Patch 3: Added M15 candle storage + HTF fetch in _generate_candidates")
else:
    print("Patch 3: SKIPPED (already applied or marker not found)")

# --- Patch 4: Pass candle data to ai_gate.evaluate() ---
old_evaluate = '''                    # Apply AI gate with market view and trade history
                    mv = self._market_views.get(candidate.symbol)
                    decision = self.ai_gate.evaluate(
                        setup, market_view=mv, trade_history=trade_history
                    )'''

new_evaluate = '''                    # Apply AI gate with market view, trade history, and candle data
                    mv = self._market_views.get(candidate.symbol)
                    symbol_candles = self._candle_data.get(candidate.symbol, {})
                    decision = self.ai_gate.evaluate(
                        setup,
                        market_view=mv,
                        trade_history=trade_history,
                        m15_candles=symbol_candles.get("m15"),
                        h1_candles=symbol_candles.get("h1"),
                        h4_candles=symbol_candles.get("h4"),
                    )'''

if old_evaluate in content:
    content = content.replace(old_evaluate, new_evaluate)
    print("Patch 4: Updated evaluate() call to pass candle data")
else:
    print("Patch 4: SKIPPED (already applied or marker not found)")

# --- Patch 5: Update ACTIVE_MODEL_VERSION env var default ---
old_version = '''active_model_version=os.getenv("ACTIVE_MODEL_VERSION", "v1.0.0"),'''
new_version = '''active_model_version=os.getenv("ACTIVE_MODEL_VERSION", "v3.0.0"),'''

# This is in config.py, not main.py -- skip here

# Write result
with open(MAIN_PY, "w") as f:
    f.write(content)

print("\\nDone! main.py patched successfully.")
