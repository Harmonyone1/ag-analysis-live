"""Symbol x Session decision table: which combos to stop manually closing."""
import sys
import time
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, ".")
from src.config import load_config
from src.adapters import TradeLockerAdapter

ET = ZoneInfo("America/New_York")

BOT_SYMBOLS = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "EURNZD", "GBPAUD",
    "GBPNZD", "AUDNZD", "AUDCAD", "NZDCAD", "CADJPY", "CHFJPY",
    "EURCAD", "EURCHF", "GBPCAD", "GBPCHF", "AUDCHF", "NZDCHF", "CADCHF",
}


def classify_session(epoch_ms):
    utc_hour = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).hour
    if 12 <= utc_hour < 16:
        return "peak"
    elif 8 <= utc_hour < 12:
        return "london"
    elif 16 <= utc_hour < 20:
        return "ny"
    elif 20 <= utc_hour < 24:
        return "late_ny"
    else:
        return "asia"


def compute_pnl(side, open_price, current_price, qty, symbol):
    if side == "buy":
        pnl_per_unit = current_price - open_price
    else:
        pnl_per_unit = open_price - current_price
    if "JPY" in symbol:
        return pnl_per_unit * qty * 1000
    else:
        return pnl_per_unit * qty * 100000


# ── Connect ──
config = load_config()
broker = TradeLockerAdapter(
    environment=config.tradelocker.environment,
    email=config.tradelocker.email,
    password=config.tradelocker.password,
    server=config.tradelocker.server,
    acc_num=config.tradelocker.acc_num,
    log_level="warning",
)
broker.connect()

# ── Reconstruct trades + recover SL/TP ──
print("Fetching all orders...")
df = broker._api.get_all_orders(history=True)

all_orders_by_pos = defaultdict(list)
for _, row in df.iterrows():
    pos_id = row.get("positionId", 0)
    if pos_id:
        all_orders_by_pos[pos_id].append(row)

sym_cache = {}
def get_symbol(inst_id):
    if inst_id not in sym_cache:
        try:
            sym_cache[inst_id] = broker._api.get_symbol_name_from_instrument_id(int(inst_id))
        except Exception:
            sym_cache[inst_id] = str(inst_id)
    return sym_cache[inst_id]

filled = df[(df["status"] == "Filled") & (df["filledQty"].astype(float) > 0)].copy()
positions_map = defaultdict(list)
for _, row in filled.iterrows():
    pos_id = row.get("positionId", 0)
    if pos_id:
        positions_map[pos_id].append(row)

manual_closes = []
for pos_id, orders in positions_map.items():
    opens = [o for o in orders if o["isOpen"] == "true"]
    closes = [o for o in orders if o["isOpen"] == "false"]
    if not opens or not closes:
        continue

    inst_id = int(opens[0].get("tradableInstrumentId", 0))
    sym = get_symbol(inst_id)
    if sym not in BOT_SYMBOLS:
        continue

    close_types = [str(o.get("type", "")).lower() for o in closes]
    if "stop" in close_types or "limit" in close_types:
        continue  # Only manual closes

    open_side = opens[0]["side"]
    open_qty = sum(float(o["filledQty"]) for o in opens)
    open_avg = sum(float(o["avgPrice"]) * float(o["filledQty"]) for o in opens) / open_qty if open_qty else 0
    close_qty = sum(float(o["filledQty"]) for o in closes)
    close_avg = sum(float(o["avgPrice"]) * float(o["filledQty"]) for o in closes) / close_qty if close_qty else 0

    open_time = min(int(o.get("createdDate", 0)) for o in opens)
    close_time = max(int(o.get("createdDate", 0)) for o in closes)
    trade_qty = min(open_qty, close_qty)
    pnl = compute_pnl(open_side, open_avg, close_avg, trade_qty, sym)

    sl_level = None
    tp_level = None
    for o in all_orders_by_pos.get(pos_id, []):
        otype = str(o.get("type", "")).lower()
        is_open = str(o.get("isOpen", ""))
        if otype == "stop":
            # SL price is in 'price' field, not 'stopPrice' (which is always 0 in history)
            sp = float(o.get("price", 0))
            if sp > 0:
                sl_level = sp
        elif otype == "limit" and is_open == "false":
            p = float(o.get("price", 0))
            if p > 0:
                tp_level = p

    manual_closes.append({
        "pos_id": pos_id, "sym": sym, "side": open_side, "qty": trade_qty,
        "open_price": open_avg, "close_price": close_avg,
        "pnl": pnl, "open_time": open_time, "close_time": close_time,
        "sl_level": sl_level, "tp_level": tp_level,
        "session": classify_session(open_time),
    })

print(f"  Manual closes (bot symbols): {len(manual_closes)}")

# ── Fetch candles ──
symbol_trades = defaultdict(list)
for t in manual_closes:
    symbol_trades[t["sym"]].append(t)

print(f"Fetching 5m candles for {len(symbol_trades)} symbols...")
candle_cache = {}
for sym, trades in symbol_trades.items():
    earliest = min(t["open_time"] for t in trades)
    start_dt = datetime.fromtimestamp(earliest / 1000, tz=timezone.utc)
    end_dt = datetime.now(tz=timezone.utc)
    try:
        candles = broker.get_candles(sym, "5m", start=start_dt, end=end_dt, limit=30000)
        candle_cache[sym] = candles
        print(f"  {sym}: {len(candles)} candles")
    except Exception as e:
        print(f"  {sym}: FAILED - {e}")
        candle_cache[sym] = []
    time.sleep(2)

# ── Walk paths and classify ──
print("\nAnalyzing paths...")

analyzed = []
for t in manual_closes:
    sym = t["sym"]
    candles = candle_cache.get(sym, [])
    sl = t["sl_level"]
    tp = t["tp_level"]
    side = t["side"]
    entry = t["open_price"]
    close_local = datetime.fromtimestamp(t["close_time"] / 1000)
    is_loser = t["pnl"] < 0

    post_close = [c for c in candles if c.timestamp > close_local]

    sl_hit_idx = None
    tp_hit_idx = None
    pnl_2h = pnl_6h = pnl_24h = None
    max_adverse = 0.0
    max_favorable = 0.0

    for i, candle in enumerate(post_close):
        elapsed_h = (candle.timestamp - close_local).total_seconds() / 3600
        h = float(candle.high)
        l = float(candle.low)
        c = float(candle.close)

        if side == "buy":
            if sl and l <= sl and sl_hit_idx is None:
                sl_hit_idx = i
            if tp and h >= tp and tp_hit_idx is None:
                tp_hit_idx = i
            favorable = h - entry
            adverse = entry - l
        else:
            if sl and h >= sl and sl_hit_idx is None:
                sl_hit_idx = i
            if tp and l <= tp and tp_hit_idx is None:
                tp_hit_idx = i
            favorable = entry - l
            adverse = h - entry

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

        if pnl_2h is None and elapsed_h >= 2:
            pnl_2h = compute_pnl(side, entry, c, t["qty"], sym)
        if pnl_6h is None and elapsed_h >= 6:
            pnl_6h = compute_pnl(side, entry, c, t["qty"], sym)
        if pnl_24h is None and elapsed_h >= 24:
            pnl_24h = compute_pnl(side, entry, c, t["qty"], sym)

    # Classify
    if not sl and not tp:
        category = "UNKNOWN"
        hypo_pnl = None
    elif sl_hit_idx is not None and tp_hit_idx is not None:
        if sl_hit_idx < tp_hit_idx:
            category = "GOOD EXIT" if is_loser else "SMART EXIT"
            hypo_pnl = compute_pnl(side, entry, sl, t["qty"], sym)
        else:
            category = "BAD EXIT" if is_loser else "EARLY EXIT"
            hypo_pnl = compute_pnl(side, entry, tp, t["qty"], sym)
    elif sl_hit_idx is not None:
        category = "GOOD EXIT" if is_loser else "SMART EXIT"
        hypo_pnl = compute_pnl(side, entry, sl, t["qty"], sym)
    elif tp_hit_idx is not None:
        category = "BAD EXIT" if is_loser else "EARLY EXIT"
        hypo_pnl = compute_pnl(side, entry, tp, t["qty"], sym)
    else:
        category = "MIXED"
        hypo_pnl = None

    analyzed.append({
        **t, "category": category, "hypo_pnl": hypo_pnl,
        "pnl_2h": pnl_2h, "pnl_6h": pnl_6h, "pnl_24h": pnl_24h,
        "mae": max_adverse, "mfe": max_favorable,
    })

# ── Build decision matrix ──
# key = (symbol, session)
matrix = defaultdict(lambda: {
    "total": 0, "losers": 0, "winners": 0,
    "good": 0, "bad": 0, "smart": 0, "early": 0, "mixed": 0, "unknown": 0,
    "win_2h": 0, "win_6h": 0, "win_24h": 0,
    "actual_pnls": [], "hypo_pnls": [],
    "improvements": [],  # hypo - actual for each trade
    "worst_deterioration": 0,
})

for r in analyzed:
    key = (r["sym"], r["session"])
    d = matrix[key]
    d["total"] += 1
    if r["pnl"] < 0:
        d["losers"] += 1
    else:
        d["winners"] += 1

    cat_key = r["category"].lower().replace(" ", "_")
    if cat_key in d:
        d[cat_key] += 1

    d["actual_pnls"].append(r["pnl"])
    if r["hypo_pnl"] is not None:
        d["hypo_pnls"].append(r["hypo_pnl"])
        improvement = r["hypo_pnl"] - r["pnl"]
        d["improvements"].append(improvement)
        if improvement < d["worst_deterioration"]:
            d["worst_deterioration"] = improvement

    # Count trades that would be winners at various hold times
    if r["pnl_2h"] is not None and r["pnl_2h"] > 0:
        d["win_2h"] += 1
    if r["pnl_6h"] is not None and r["pnl_6h"] > 0:
        d["win_6h"] += 1
    if r["pnl_24h"] is not None and r["pnl_24h"] > 0:
        d["win_24h"] += 1

# Also aggregate by symbol only and session only
sym_agg = defaultdict(lambda: {
    "total": 0, "bad": 0, "good": 0, "early": 0, "smart": 0,
    "actual_sum": 0, "hypo_sum": 0, "hypo_count": 0,
})
sess_agg = defaultdict(lambda: {
    "total": 0, "bad": 0, "good": 0, "early": 0, "smart": 0,
    "actual_sum": 0, "hypo_sum": 0, "hypo_count": 0,
})

for r in analyzed:
    for agg, key in [(sym_agg, r["sym"]), (sess_agg, r["session"])]:
        agg[key]["total"] += 1
        agg[key]["actual_sum"] += r["pnl"]
        if r["hypo_pnl"] is not None:
            agg[key]["hypo_sum"] += r["hypo_pnl"]
            agg[key]["hypo_count"] += 1
        if r["category"] == "BAD EXIT":
            agg[key]["bad"] += 1
        elif r["category"] == "GOOD EXIT":
            agg[key]["good"] += 1
        elif r["category"] == "EARLY EXIT":
            agg[key]["early"] += 1
        elif r["category"] == "SMART EXIT":
            agg[key]["smart"] += 1


def verdict(data):
    if data["total"] < 2:
        return "INSUFFICIENT"
    harmful = data.get("bad", 0) + data.get("early", 0)
    helpful = data.get("good", 0) + data.get("smart", 0)
    harmful_pct = harmful / data["total"]
    if harmful_pct >= 0.5:
        return "STOP EXITING"
    elif harmful_pct <= 0.2:
        return "KEEP EXITING"
    else:
        return "REVIEW"


# ── Print Symbol x Session Table ──
print(f"\n{'=' * 150}")
print("  DECISION TABLE: Manual Exit Quality by Symbol x Session")
print(f"{'=' * 150}")

hdr = (f"{'Symbol':8} {'Session':8} {'#Tot':>4} {'#Loss':>5} {'#Win':>4} "
       f"{'Good%':>5} {'Bad%':>4} {'Smrt%':>5} {'Erl%':>4} "
       f"{'W@2h':>4} {'W@6h':>4} {'W@24h':>5} "
       f"{'Med$Imp':>8} {'Worst$':>8} {'Actual$':>9} {'Hypo$':>9} {'VERDICT':>13}")
print(hdr)
print("-" * 150)

sorted_keys = sorted(matrix.keys(), key=lambda k: -(matrix[k]["bad"] + matrix[k]["early"]))

for key in sorted_keys:
    sym, sess = key
    d = matrix[key]
    t = d["total"]

    good_pct = (d["good"] / t * 100) if t else 0
    bad_pct = (d["bad"] / t * 100) if t else 0
    smart_pct = (d["smart"] / t * 100) if t else 0
    early_pct = (d["early"] / t * 100) if t else 0

    w2h = d["win_2h"]
    w6h = d["win_6h"]
    w24h = d["win_24h"]

    med_imp = statistics.median(d["improvements"]) if d["improvements"] else 0
    worst = d["worst_deterioration"]
    actual_sum = sum(d["actual_pnls"])
    hypo_sum = sum(d["hypo_pnls"]) if d["hypo_pnls"] else 0

    v = verdict(d)

    print(f"{sym:8} {sess:8} {t:4d} {d['losers']:5d} {d['winners']:4d} "
          f"{good_pct:4.0f}% {bad_pct:3.0f}% {smart_pct:4.0f}% {early_pct:3.0f}% "
          f"{w2h:4d} {w6h:4d} {w24h:5d} "
          f"${med_imp:+7.2f} ${worst:+7.2f} ${actual_sum:+8.2f} ${hypo_sum:+8.2f} {v:>13}")

# ── Symbol-level Summary ──
print(f"\n{'=' * 100}")
print("  SUMMARY BY SYMBOL")
print(f"{'=' * 100}")

hdr2 = f"{'Symbol':8} {'#Tot':>4} {'Bad':>3} {'Good':>4} {'Early':>5} {'Smart':>5} {'Actual$':>9} {'Hypo$':>9} {'VERDICT':>13}"
print(hdr2)
print("-" * 100)

for sym in sorted(sym_agg.keys(), key=lambda s: -(sym_agg[s]["bad"] + sym_agg[s]["early"])):
    d = sym_agg[sym]
    v = verdict(d)
    print(f"{sym:8} {d['total']:4d} {d['bad']:3d} {d['good']:4d} {d['early']:5d} {d['smart']:5d} "
          f"${d['actual_sum']:+8.2f} ${d['hypo_sum']:+8.2f} {v:>13}")

# ── Session-level Summary ──
print(f"\n{'=' * 100}")
print("  SUMMARY BY SESSION")
print(f"{'=' * 100}")
print(hdr2)
print("-" * 100)

for sess in ["asia", "london", "peak", "ny", "late_ny"]:
    if sess in sess_agg:
        d = sess_agg[sess]
        v = verdict(d)
        print(f"{sess:8} {d['total']:4d} {d['bad']:3d} {d['good']:4d} {d['early']:5d} {d['smart']:5d} "
              f"${d['actual_sum']:+8.2f} ${d['hypo_sum']:+8.2f} {v:>13}")

# ── Overall Recommendations ──
print(f"\n{'=' * 80}")
print("  RECOMMENDATIONS")
print(f"{'=' * 80}")

stop_exiting = []
keep_exiting = []
review_needed = []

for key in sorted_keys:
    d = matrix[key]
    v = verdict(d)
    sym, sess = key
    if v == "STOP EXITING":
        impact = sum(d["actual_pnls"]) - sum(d["hypo_pnls"]) if d["hypo_pnls"] else 0
        stop_exiting.append((sym, sess, d["total"], impact))
    elif v == "KEEP EXITING":
        keep_exiting.append((sym, sess, d["total"]))
    elif v == "REVIEW":
        review_needed.append((sym, sess, d["total"]))

if stop_exiting:
    print("\n  STOP manually closing these (you're killing winners / exiting too early):")
    total_impact = 0
    for sym, sess, n, impact in stop_exiting:
        print(f"    {sym:8} ({sess:8}) - {n} trades, est. impact: ${impact:+.2f}")
        total_impact += impact
    print(f"  Estimated total $ impact if you stop: ${total_impact:+.2f}")

if keep_exiting:
    print("\n  KEEP exiting these (your manual exits are correct):")
    for sym, sess, n in keep_exiting:
        print(f"    {sym:8} ({sess:8}) - {n} trades")

if review_needed:
    print("\n  REVIEW these (mixed results):")
    for sym, sess, n in review_needed:
        print(f"    {sym:8} ({sess:8}) - {n} trades")

print()
