"""Broker data integrity verification — reconcile order history for accuracy."""
import sys
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

def epoch_to_est(epoch_ms):
    dt = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
    return dt.astimezone(ET).strftime("%b %d %I:%M %p %Z")

def compute_pnl(side, open_price, close_price, qty, symbol):
    if side == "buy":
        pnl_per_unit = close_price - open_price
    else:
        pnl_per_unit = open_price - close_price
    if "JPY" in symbol:
        return pnl_per_unit * qty * 1000
    elif symbol == "ETHUSD":
        return pnl_per_unit * qty
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

acct = broker.get_account()
print(f"ACCOUNT: balance=${acct.balance}  equity=${acct.equity}  "
      f"margin_used=${acct.margin_used}  margin_free=${acct.margin_available}  {acct.currency}")
print()

# ── Fetch ALL orders (any status) ──
print("Fetching all orders (including cancelled)...")
df = broker._api.get_all_orders(history=True)
total_orders = len(df)
print(f"Total orders in history: {total_orders}")
print()

print("=== ORDER BREAKDOWN BY STATUS ===")
status_counts = df.groupby("status").size()
for status, count in status_counts.items():
    print(f"  {status:15} {count:5d}")
print()

print("=== ORDER BREAKDOWN BY TYPE ===")
type_counts = df.groupby("type").size()
for otype, count in type_counts.items():
    print(f"  {otype:15} {count:5d}")
print()

# ── Check 1: Position ID Integrity ──
print("=" * 60)
print("CHECK 1: Position ID Integrity")
print("=" * 60)

pos_orders = defaultdict(lambda: {"opens": [], "closes": [], "other": []})
for _, row in df.iterrows():
    pos_id = row.get("positionId", 0)
    if not pos_id:
        continue
    status = str(row.get("status", ""))
    is_open = str(row.get("isOpen", ""))
    if status == "Filled" and is_open == "true":
        pos_orders[pos_id]["opens"].append(row)
    elif status == "Filled" and is_open == "false":
        pos_orders[pos_id]["closes"].append(row)
    else:
        pos_orders[pos_id]["other"].append(row)

# Resolve symbols
sym_cache = {}
def get_symbol(inst_id):
    if inst_id not in sym_cache:
        try:
            sym_cache[inst_id] = broker._api.get_symbol_name_from_instrument_id(int(inst_id))
        except Exception:
            sym_cache[inst_id] = str(inst_id)
    return sym_cache[inst_id]

fully_closed = 0
still_open = 0
orphan_close_only = 0
orphan_open_only = 0
issues = []

for pos_id, data in pos_orders.items():
    n_opens = len(data["opens"])
    n_closes = len(data["closes"])
    n_other = len(data["other"])

    if n_opens > 0 and n_closes > 0:
        fully_closed += 1
    elif n_opens > 0 and n_closes == 0:
        still_open += 1  # Could be currently open
    elif n_opens == 0 and n_closes > 0:
        orphan_close_only += 1
        inst_id = int(data["closes"][0].get("tradableInstrumentId", 0))
        sym = get_symbol(inst_id)
        issues.append(f"  WARN: posId={pos_id} has {n_closes} close(s) but NO opens ({sym})")
    elif n_opens == 0 and n_closes == 0 and n_other > 0:
        pass  # Only cancelled/pending orders, normal

print(f"  Fully closed positions (opens + closes): {fully_closed}")
print(f"  Positions with opens only (likely still open): {still_open}")
print(f"  Positions with closes only (NO opens): {orphan_close_only}")
print(f"  Total unique positionIds: {len(pos_orders)}")

if orphan_close_only == 0:
    print("  RESULT: PASS - every close has matching opens")
else:
    print(f"  RESULT: WARN - {orphan_close_only} positions have closes without opens")
    for issue in issues:
        print(issue)
print()

# ── Check 2: Open Position Cross-Reference ──
print("=" * 60)
print("CHECK 2: Open Position Cross-Reference")
print("=" * 60)

broker_positions = broker.get_positions()
broker_pos_ids = {p.position_id for p in broker_positions}

# PositionIds that have opens but no filled closes = should be open
order_open_ids = set()
for pos_id, data in pos_orders.items():
    if len(data["opens"]) > 0 and len(data["closes"]) == 0:
        order_open_ids.add(pos_id)

# Cross-reference
broker_only = broker_pos_ids - order_open_ids
orders_only = order_open_ids - broker_pos_ids

print(f"  Broker reports {len(broker_positions)} open positions: {sorted(broker_pos_ids)}")
print(f"  Order history suggests {len(order_open_ids)} open: {sorted(order_open_ids)}")

if not broker_only and not orders_only:
    print("  RESULT: PASS - open positions match perfectly")
else:
    if broker_only:
        print(f"  WARN: On broker but NOT in order history (opens missing): {broker_only}")
    if orders_only:
        print(f"  WARN: In order history (open) but NOT on broker: {orders_only}")
    # If orders_only exists, it could mean positions closed very recently
    if broker_only or orders_only:
        print("  RESULT: WARN - mismatches detected (may be timing)")
    else:
        print("  RESULT: PASS")

print()
for p in broker_positions:
    print(f"  {p.symbol:8} {p.side:4} qty={p.quantity} entry={p.avg_price} "
          f"SL={p.stop_loss} TP={p.take_profit} posId={p.position_id}")
print()

# ── Check 3: Time Continuity ──
print("=" * 60)
print("CHECK 3: Time Continuity")
print("=" * 60)

filled = df[df["status"] == "Filled"].copy()
if len(filled) > 0:
    filled_sorted = filled.sort_values("createdDate")
    first_ts = int(filled_sorted.iloc[0]["createdDate"])
    last_ts = int(filled_sorted.iloc[-1]["createdDate"])
    print(f"  First filled order: {epoch_to_est(first_ts)}")
    print(f"  Last filled order:  {epoch_to_est(last_ts)}")
    print(f"  Total filled orders: {len(filled)}")

    # Check for large gaps
    dates = sorted(filled_sorted["createdDate"].astype(int).tolist())
    max_gap_ms = 0
    max_gap_start = 0
    for i in range(1, len(dates)):
        gap = dates[i] - dates[i - 1]
        if gap > max_gap_ms:
            max_gap_ms = gap
            max_gap_start = dates[i - 1]

    max_gap_hours = max_gap_ms / (1000 * 3600)
    print(f"  Largest gap between filled orders: {max_gap_hours:.1f} hours")
    if max_gap_hours > 0:
        print(f"    From: {epoch_to_est(max_gap_start)}")
        print(f"    To:   {epoch_to_est(max_gap_start + max_gap_ms)}")

    if max_gap_hours > 168:  # > 7 days
        print("  RESULT: WARN - gap exceeds 7 days (possible pagination issue)")
    else:
        print("  RESULT: PASS")
else:
    print("  No filled orders found!")
    print("  RESULT: FAIL")
print()

# ── Check 4: PnL Sanity ──
print("=" * 60)
print("CHECK 4: PnL Reconstruction Sanity")
print("=" * 60)

# Reconstruct closed trades (same logic as trade_review.py)
filled_active = filled[(filled["filledQty"].astype(float) > 0)].copy()
positions_map = defaultdict(list)
for _, row in filled_active.iterrows():
    pos_id = row.get("positionId", 0)
    if pos_id:
        positions_map[pos_id].append(row)

total_pnl = 0
bot_pnl = 0
trade_count = 0
bot_trade_count = 0

for pos_id, orders in positions_map.items():
    opens = [o for o in orders if o["isOpen"] == "true"]
    closes = [o for o in orders if o["isOpen"] == "false"]
    if not opens or not closes:
        continue

    inst_id = int(opens[0].get("tradableInstrumentId", 0))
    sym = get_symbol(inst_id)

    open_side = opens[0]["side"]
    open_qty = sum(float(o["filledQty"]) for o in opens)
    open_avg = sum(float(o["avgPrice"]) * float(o["filledQty"]) for o in opens) / open_qty if open_qty else 0
    close_qty = sum(float(o["filledQty"]) for o in closes)
    close_avg = sum(float(o["avgPrice"]) * float(o["filledQty"]) for o in closes) / close_qty if close_qty else 0

    trade_qty = min(open_qty, close_qty)
    pnl = compute_pnl(open_side, open_avg, close_avg, trade_qty, sym)

    total_pnl += pnl
    trade_count += 1
    if sym in BOT_SYMBOLS:
        bot_pnl += pnl
        bot_trade_count += 1

print(f"  Total reconstructed closed trades: {trade_count}")
print(f"  Bot FX trades: {bot_trade_count}")
print(f"  Total reconstructed PnL (all): ${total_pnl:+.2f}")
print(f"  Bot FX PnL: ${bot_pnl:+.2f}")
print(f"  Account balance: ${acct.balance}")
print()

# Open position unrealized PnL
total_unrealized = sum(float(p.unrealized_pnl) for p in broker_positions)
print(f"  Open position unrealized PnL: ${total_unrealized:+.2f}")
print(f"  Balance + unrealized = equity estimate: ${float(acct.balance) + total_unrealized:.2f}")
print(f"  Broker-reported equity: ${acct.equity}")
print()

equity_diff = abs(float(acct.equity) - (float(acct.balance) + total_unrealized))
if equity_diff < 50:
    print("  RESULT: PASS - equity roughly consistent")
else:
    print(f"  RESULT: WARN - equity differs by ${equity_diff:.2f} (swaps/commissions/deposits?)")
print()

# ── Final Summary ──
print("=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print(f"  Check 1 (Position ID integrity):    {'PASS' if orphan_close_only == 0 else 'WARN'}")
check2_pass = not broker_only and not orders_only
print(f"  Check 2 (Open position cross-ref):  {'PASS' if check2_pass else 'WARN'}")
check3_pass = max_gap_hours <= 168 if len(filled) > 0 else False
print(f"  Check 3 (Time continuity):          {'PASS' if check3_pass else 'WARN'}")
print(f"  Check 4 (PnL sanity):               {'PASS' if equity_diff < 50 else 'WARN'}")
print()
