"""Path-dependent exit analysis: what WOULD have happened after each manual close."""
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
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


def epoch_to_est(epoch_ms):
    dt = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
    return dt.astimezone(ET).strftime("%b %d %I:%M %p")


def classify_session(epoch_ms):
    """Classify trading session from UTC hour."""
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

# ── Phase A: Fetch ALL orders and reconstruct trades ──
print("Fetching all orders (including cancelled)...")
df = broker._api.get_all_orders(history=True)
print(f"  Total orders: {len(df)}")

# Index ALL orders by positionId (any status) for SL/TP recovery
all_orders_by_pos = defaultdict(list)
for _, row in df.iterrows():
    pos_id = row.get("positionId", 0)
    if pos_id:
        all_orders_by_pos[pos_id].append(row)

# Symbol resolution cache
sym_cache = {}
def get_symbol(inst_id):
    if inst_id not in sym_cache:
        try:
            sym_cache[inst_id] = broker._api.get_symbol_name_from_instrument_id(int(inst_id))
        except Exception:
            sym_cache[inst_id] = str(inst_id)
    return sym_cache[inst_id]

# Reconstruct closed trades from filled orders
filled = df[(df["status"] == "Filled") & (df["filledQty"].astype(float) > 0)].copy()
positions_map = defaultdict(list)
for _, row in filled.iterrows():
    pos_id = row.get("positionId", 0)
    if pos_id:
        positions_map[pos_id].append(row)

closed_trades = []
for pos_id, orders in positions_map.items():
    opens = [o for o in orders if o["isOpen"] == "true"]
    closes = [o for o in orders if o["isOpen"] == "false"]
    if not opens or not closes:
        continue

    inst_id = int(opens[0].get("tradableInstrumentId", 0))
    sym = get_symbol(inst_id)
    if sym not in BOT_SYMBOLS:
        continue

    open_side = opens[0]["side"]
    open_qty = sum(float(o["filledQty"]) for o in opens)
    open_avg = sum(float(o["avgPrice"]) * float(o["filledQty"]) for o in opens) / open_qty if open_qty else 0
    close_qty = sum(float(o["filledQty"]) for o in closes)
    close_avg = sum(float(o["avgPrice"]) * float(o["filledQty"]) for o in closes) / close_qty if close_qty else 0

    open_time = min(int(o.get("createdDate", 0)) for o in opens)
    close_time = max(int(o.get("createdDate", 0)) for o in closes)
    trade_qty = min(open_qty, close_qty)
    pnl = compute_pnl(open_side, open_avg, close_avg, trade_qty, sym)

    # Determine exit type from close order types
    close_types = [str(o.get("type", "")).lower() for o in closes]
    if "stop" in close_types:
        exit_type = "SL"
    elif "limit" in close_types:
        exit_type = "TP"
    else:
        exit_type = "MANUAL"

    # Recover SL/TP from ALL orders (including cancelled) for this position
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

    closed_trades.append({
        "pos_id": pos_id,
        "sym": sym, "side": open_side, "qty": trade_qty,
        "open_price": open_avg, "close_price": close_avg,
        "pnl": pnl, "open_time": open_time, "close_time": close_time,
        "exit_type": exit_type, "sl_level": sl_level, "tp_level": tp_level,
        "session": classify_session(open_time),
    })

closed_trades.sort(key=lambda x: x["close_time"], reverse=True)

# Separate manual closes (both losers and winners)
manual_closes = [t for t in closed_trades if t["exit_type"] == "MANUAL"]
manual_losers = [t for t in manual_closes if t["pnl"] < 0]
manual_winners = [t for t in manual_closes if t["pnl"] >= 0]
sl_exits = [t for t in closed_trades if t["exit_type"] == "SL"]
tp_exits = [t for t in closed_trades if t["exit_type"] == "TP"]

print(f"\n=== TRADE EXIT BREAKDOWN ===")
print(f"  Total bot closed trades: {len(closed_trades)}")
print(f"  Manual closes: {len(manual_closes)} ({len(manual_losers)} losers, {len(manual_winners)} winners)")
print(f"  Stop-loss exits: {len(sl_exits)}")
print(f"  Take-profit exits: {len(tp_exits)}")
print(f"  Trades with SL level recovered: {sum(1 for t in manual_closes if t['sl_level'])}/{len(manual_closes)}")
print(f"  Trades with TP level recovered: {sum(1 for t in manual_closes if t['tp_level'])}/{len(manual_closes)}")

# ── Phase B: Fetch candles batched by symbol ──
print(f"\nFetching 5m candles for {len(set(t['sym'] for t in manual_closes))} symbols...")

symbol_trades = defaultdict(list)
for t in manual_closes:
    symbol_trades[t["sym"]].append(t)

candle_cache = {}
for sym, trades in symbol_trades.items():
    earliest_open = min(t["open_time"] for t in trades)
    start_dt = datetime.fromtimestamp(earliest_open / 1000, tz=timezone.utc)
    end_dt = datetime.now(tz=timezone.utc)

    try:
        candles = broker.get_candles(sym, "5m", start=start_dt, end=end_dt, limit=30000)
        candle_cache[sym] = candles
        print(f"  {sym}: {len(candles)} candles ({start_dt.strftime('%m/%d')} -> now)")
    except Exception as e:
        print(f"  {sym}: FAILED - {e}")
        candle_cache[sym] = []
    time.sleep(2)

# ── Phase C: Walk price path after each manual close ──
print("\nAnalyzing price paths...")

# Get current quotes for mark-to-market
current_prices = {}
for sym in symbol_trades.keys():
    try:
        q = broker.get_quote(sym)
        current_prices[sym] = {"bid": float(q.bid), "ask": float(q.ask)}
    except Exception:
        pass

results = []
for t in manual_closes:
    sym = t["sym"]
    candles = candle_cache.get(sym, [])
    if not candles:
        results.append({**t, "category": "UNKNOWN", "reason": "no candles",
                        "pnl_2h": None, "pnl_6h": None, "pnl_24h": None,
                        "pnl_now": None, "mae": 0, "mfe": 0,
                        "sl_hit_time": None, "tp_hit_time": None,
                        "hypo_pnl": None})
        continue

    close_epoch_s = t["close_time"] / 1000
    sl = t["sl_level"]
    tp = t["tp_level"]
    side = t["side"]
    entry = t["open_price"]

    # Filter candles after close time
    # Candle timestamps are naive local time from datetime.fromtimestamp()
    # Convert close_time to local naive for comparison
    close_local = datetime.fromtimestamp(close_epoch_s)
    post_close = [c for c in candles if c.timestamp > close_local]

    sl_hit_time = None
    tp_hit_time = None
    sl_hit_idx = None
    tp_hit_idx = None
    max_adverse = 0.0
    max_favorable = 0.0

    pnl_2h = None
    pnl_6h = None
    pnl_24h = None

    for i, candle in enumerate(post_close):
        elapsed_s = (candle.timestamp - close_local).total_seconds()
        elapsed_h = elapsed_s / 3600

        h = float(candle.high)
        l = float(candle.low)
        c = float(candle.close)

        if side == "buy":
            # Buy: SL hit if low <= SL, TP hit if high >= TP
            if sl and l <= sl and sl_hit_time is None:
                sl_hit_time = candle.timestamp
                sl_hit_idx = i
            if tp and h >= tp and tp_hit_time is None:
                tp_hit_time = candle.timestamp
                tp_hit_idx = i
            favorable = h - entry
            adverse = entry - l
        else:
            # Sell: SL hit if high >= SL, TP hit if low <= TP
            if sl and h >= sl and sl_hit_time is None:
                sl_hit_time = candle.timestamp
                sl_hit_idx = i
            if tp and l <= tp and tp_hit_time is None:
                tp_hit_time = candle.timestamp
                tp_hit_idx = i
            favorable = entry - l
            adverse = h - entry

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

        # Time-based PnL snapshots
        if pnl_2h is None and elapsed_h >= 2:
            pnl_2h = compute_pnl(side, entry, c, t["qty"], sym)
        if pnl_6h is None and elapsed_h >= 6:
            pnl_6h = compute_pnl(side, entry, c, t["qty"], sym)
        if pnl_24h is None and elapsed_h >= 24:
            pnl_24h = compute_pnl(side, entry, c, t["qty"], sym)

    # Current mark-to-market (use bid/ask correctly)
    pnl_now = None
    if sym in current_prices:
        if side == "buy":
            now_price = current_prices[sym]["bid"]
        else:
            now_price = current_prices[sym]["ask"]
        pnl_now = compute_pnl(side, entry, now_price, t["qty"], sym)

    # Convert MAE/MFE to dollar terms
    if "JPY" in sym:
        mult = t["qty"] * 1000
    else:
        mult = t["qty"] * 100000
    mae_dollars = max_adverse * mult
    mfe_dollars = max_favorable * mult

    # Classify
    is_loser = t["pnl"] < 0

    if not sl and not tp:
        category = "UNKNOWN"
        reason = "no SL/TP data"
        hypo_pnl = None
    elif sl_hit_time and tp_hit_time:
        if sl_hit_idx < tp_hit_idx:
            # SL hit first
            if is_loser:
                category = "GOOD EXIT"
                reason = "SL would've hit first"
            else:
                category = "SMART EXIT"
                reason = "would've reversed to SL"
            hypo_pnl = compute_pnl(side, entry, sl, t["qty"], sym)
        else:
            # TP hit first
            if is_loser:
                category = "BAD EXIT"
                reason = "TP would've hit first"
            else:
                category = "EARLY EXIT"
                reason = "TP would've hit for more"
            hypo_pnl = compute_pnl(side, entry, tp, t["qty"], sym)
    elif sl_hit_time and not tp_hit_time:
        if is_loser:
            category = "GOOD EXIT"
            reason = "SL hit, TP never"
        else:
            category = "SMART EXIT"
            reason = "would've hit SL after"
        hypo_pnl = compute_pnl(side, entry, sl, t["qty"], sym)
    elif tp_hit_time and not sl_hit_time:
        if is_loser:
            category = "BAD EXIT"
            reason = "TP hit, SL never"
        else:
            category = "EARLY EXIT"
            reason = "TP hit for more profit"
        hypo_pnl = compute_pnl(side, entry, tp, t["qty"], sym)
    else:
        category = "MIXED"
        reason = "neither SL nor TP hit yet"
        hypo_pnl = pnl_now

    results.append({
        **t,
        "category": category, "reason": reason,
        "pnl_2h": pnl_2h, "pnl_6h": pnl_6h, "pnl_24h": pnl_24h,
        "pnl_now": pnl_now, "mae": mae_dollars, "mfe": mfe_dollars,
        "sl_hit_time": sl_hit_time, "tp_hit_time": tp_hit_time,
        "hypo_pnl": hypo_pnl,
    })

# ── Phase D: Output ──
loser_results = [r for r in results if r["pnl"] < 0]
winner_results = [r for r in results if r["pnl"] >= 0]

# Sort by close time desc
loser_results.sort(key=lambda x: x["close_time"], reverse=True)
winner_results.sort(key=lambda x: x["close_time"], reverse=True)


def fmt_pnl(val):
    if val is None:
        return "     N/A"
    return f"${val:+8.2f}"


def print_trade_table(trades, title):
    print(f"\n{'=' * 160}")
    print(f"  {title} ({len(trades)} trades)")
    print(f"{'=' * 160}")

    hdr = (f"{'Opened (EST)':>17} {'Closed (EST)':>17} {'Sym':7} {'Side':4} "
           f"{'Entry':>10} {'Close@':>10} {'SL':>10} {'TP':>10} "
           f"{'Actual':>9} {'If-Sys':>9} {'Category':>12} "
           f"{'@2h':>9} {'@6h':>9} {'@24h':>9} {'@Now':>9} "
           f"{'MAE':>8} {'MFE':>8} {'Sess':>7}")
    print(hdr)
    print("-" * 160)

    for r in trades:
        ot = epoch_to_est(r["open_time"])
        ct = epoch_to_est(r["close_time"])
        sl_str = f"{r['sl_level']:10.5f}" if r["sl_level"] else "       N/A"
        tp_str = f"{r['tp_level']:10.5f}" if r["tp_level"] else "       N/A"

        print(f"{ot:>17} {ct:>17} {r['sym']:7} {r['side']:4} "
              f"{r['open_price']:10.5f} {r['close_price']:10.5f} {sl_str} {tp_str} "
              f"{fmt_pnl(r['pnl'])} {fmt_pnl(r['hypo_pnl'])} {r['category']:>12} "
              f"{fmt_pnl(r['pnl_2h'])} {fmt_pnl(r['pnl_6h'])} {fmt_pnl(r['pnl_24h'])} {fmt_pnl(r['pnl_now'])} "
              f"${r['mae']:7.2f} ${r['mfe']:7.2f} {r['session']:>7}")


print_trade_table(loser_results, "MANUALLY CLOSED LOSERS")
print_trade_table(winner_results, "MANUALLY CLOSED WINNERS")

# ── Summary: Losing Manual Closes ──
print(f"\n{'=' * 80}")
print("  SUMMARY: LOSING MANUAL CLOSES")
print(f"{'=' * 80}")

cat_counts_l = defaultdict(int)
for r in loser_results:
    cat_counts_l[r["category"]] += 1

total_l = len(loser_results)
for cat in ["GOOD EXIT", "BAD EXIT", "MIXED", "UNKNOWN"]:
    n = cat_counts_l.get(cat, 0)
    pct = n / total_l * 100 if total_l else 0
    print(f"  {cat:12}: {n:3d} ({pct:5.1f}%)")

actual_loss_total = sum(r["pnl"] for r in loser_results)
hypo_total_l = sum(r["hypo_pnl"] for r in loser_results if r["hypo_pnl"] is not None)
hypo_count_l = sum(1 for r in loser_results if r["hypo_pnl"] is not None)

print(f"\n  Your actual total losses:       {fmt_pnl(actual_loss_total)}")
print(f"  Hypothetical 'let system run':  {fmt_pnl(hypo_total_l)} (from {hypo_count_l} trades with SL/TP)")
print(f"  Difference (hypo - actual):     {fmt_pnl(hypo_total_l - actual_loss_total)}")
if hypo_total_l > actual_loss_total:
    print(f"  --> Manual exits SAVED you ${actual_loss_total - hypo_total_l:+.2f}")
else:
    print(f"  --> Manual exits COST you ${hypo_total_l - actual_loss_total:+.2f}")

# Symbols where BAD EXITs cluster
print(f"\n  SYMBOLS WITH HIGH BAD EXIT RATE:")
sym_cats_l = defaultdict(lambda: {"total": 0, "bad": 0, "good": 0})
for r in loser_results:
    sym_cats_l[r["sym"]]["total"] += 1
    if r["category"] == "BAD EXIT":
        sym_cats_l[r["sym"]]["bad"] += 1
    elif r["category"] == "GOOD EXIT":
        sym_cats_l[r["sym"]]["good"] += 1

for sym in sorted(sym_cats_l.keys(), key=lambda s: sym_cats_l[s]["bad"], reverse=True):
    d = sym_cats_l[sym]
    if d["bad"] > 0:
        bad_pct = d["bad"] / d["total"] * 100
        print(f"    {sym:8}: {d['bad']}/{d['total']} BAD ({bad_pct:.0f}%)  "
              f"{d['good']}/{d['total']} GOOD")

# Sessions where BAD EXITs cluster
print(f"\n  SESSIONS WITH HIGH BAD EXIT RATE:")
sess_cats_l = defaultdict(lambda: {"total": 0, "bad": 0, "good": 0})
for r in loser_results:
    sess_cats_l[r["session"]]["total"] += 1
    if r["category"] == "BAD EXIT":
        sess_cats_l[r["session"]]["bad"] += 1
    elif r["category"] == "GOOD EXIT":
        sess_cats_l[r["session"]]["good"] += 1

for sess in ["asia", "london", "peak", "ny", "late_ny"]:
    if sess in sess_cats_l:
        d = sess_cats_l[sess]
        bad_pct = d["bad"] / d["total"] * 100 if d["total"] else 0
        print(f"    {sess:8}: {d['bad']}/{d['total']} BAD ({bad_pct:.0f}%)  "
              f"{d['good']}/{d['total']} GOOD")

# ── Summary: Winning Manual Closes ──
print(f"\n{'=' * 80}")
print("  SUMMARY: PROFITABLE MANUAL CLOSES")
print(f"{'=' * 80}")

cat_counts_w = defaultdict(int)
for r in winner_results:
    cat_counts_w[r["category"]] += 1

total_w = len(winner_results)
for cat in ["EARLY EXIT", "SMART EXIT", "MIXED", "UNKNOWN"]:
    n = cat_counts_w.get(cat, 0)
    pct = n / total_w * 100 if total_w else 0
    print(f"  {cat:12}: {n:3d} ({pct:5.1f}%)")

actual_win_total = sum(r["pnl"] for r in winner_results)
# For EARLY EXITs, compute money left on table
early_exits = [r for r in winner_results if r["category"] == "EARLY EXIT"]
left_on_table = sum(r["hypo_pnl"] - r["pnl"] for r in early_exits if r["hypo_pnl"] is not None)

# For SMART EXITs, compute money saved
smart_exits = [r for r in winner_results if r["category"] == "SMART EXIT"]
saved_by_exit = sum(r["pnl"] - r["hypo_pnl"] for r in smart_exits if r["hypo_pnl"] is not None)

print(f"\n  Your actual total profits:      {fmt_pnl(actual_win_total)}")
if early_exits:
    print(f"  Money left on table (EARLY):    {fmt_pnl(left_on_table)} across {len(early_exits)} trades")
if smart_exits:
    print(f"  Profit saved by exiting (SMART):{fmt_pnl(saved_by_exit)} across {len(smart_exits)} trades")

# Symbols where EARLY EXITs cluster
print(f"\n  SYMBOLS WHERE YOU EXIT WINNERS TOO EARLY:")
sym_cats_w = defaultdict(lambda: {"total": 0, "early": 0, "smart": 0})
for r in winner_results:
    sym_cats_w[r["sym"]]["total"] += 1
    if r["category"] == "EARLY EXIT":
        sym_cats_w[r["sym"]]["early"] += 1
    elif r["category"] == "SMART EXIT":
        sym_cats_w[r["sym"]]["smart"] += 1

for sym in sorted(sym_cats_w.keys(), key=lambda s: sym_cats_w[s]["early"], reverse=True):
    d = sym_cats_w[sym]
    if d["early"] > 0:
        early_pct = d["early"] / d["total"] * 100
        print(f"    {sym:8}: {d['early']}/{d['total']} EARLY ({early_pct:.0f}%)  "
              f"{d['smart']}/{d['total']} SMART")

# ── Overall Impact ──
print(f"\n{'=' * 80}")
print("  OVERALL MANUAL EXIT IMPACT")
print(f"{'=' * 80}")
total_actual = actual_loss_total + actual_win_total
total_hypo = sum(r["hypo_pnl"] for r in results if r["hypo_pnl"] is not None)
print(f"  Total actual PnL (manual closes):  {fmt_pnl(total_actual)}")
print(f"  Total if system ran (SL/TP):       {fmt_pnl(total_hypo)}")
print(f"  Net impact of manual intervention:  {fmt_pnl(total_actual - total_hypo)}")
print()
if total_actual > total_hypo:
    print(f"  --> Your manual exits OUTPERFORMED the system by ${total_actual - total_hypo:.2f}")
else:
    print(f"  --> Your manual exits UNDERPERFORMED the system by ${total_hypo - total_actual:.2f}")

print(f"\nNOTE: TradeLocker adapter does not expose swap/commission data.")
print(f"All PnL figures are based on fill prices only.")
print()
