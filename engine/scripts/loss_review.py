"""Compare losing trades: actual close vs if-held-to-now."""
import sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, ".")
from src.config import load_config
from src.adapters import TradeLockerAdapter

# UTC to EST offset
UTC_TO_EST = timedelta(hours=-5)

def to_est(epoch_ms):
    dt = datetime.utcfromtimestamp(epoch_ms / 1000) + UTC_TO_EST
    return dt.strftime("%m-%d %I:%M %p")

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

BOT_SYMBOLS = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "EURNZD", "GBPAUD",
    "GBPNZD", "AUDNZD", "AUDCAD", "NZDCAD", "CADJPY", "CHFJPY",
    "EURCAD", "EURCHF", "GBPCAD", "GBPCHF", "AUDCHF", "NZDCHF", "CADCHF",
}

# Get current quotes for all symbols
print("Fetching current prices...")
current_prices = {}
for sym in BOT_SYMBOLS:
    try:
        q = broker.get_quote(sym)
        current_prices[sym] = {"bid": float(q.bid), "ask": float(q.ask)}
    except Exception:
        pass

# Reconstruct closed trades from order history
df = broker._api.get_all_orders(history=True)
filled = df[(df["status"] == "Filled") & (df["filledQty"] > 0)].copy()

positions = defaultdict(list)
for _, row in filled.iterrows():
    pos_id = row.get("positionId", 0)
    if pos_id:
        positions[pos_id].append(row)

closed_trades = []
for pos_id, orders in positions.items():
    opens = [o for o in orders if o["isOpen"] == "true"]
    closes = [o for o in orders if o["isOpen"] == "false"]
    if not opens or not closes:
        continue

    inst_id = int(opens[0].get("tradableInstrumentId", 0))
    try:
        sym = broker._api.get_symbol_name_from_instrument_id(inst_id)
    except Exception:
        continue

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
    if open_side == "buy":
        pnl_per_unit = close_avg - open_avg
    else:
        pnl_per_unit = open_avg - close_avg

    if "JPY" in sym:
        pnl = pnl_per_unit * trade_qty * 1000
    else:
        pnl = pnl_per_unit * trade_qty * 100000

    close_types = [str(o.get("type", "")).lower() for o in closes]
    exit_type = "SL" if "stop" in close_types else "TP/Man"

    closed_trades.append({
        "sym": sym, "side": open_side, "qty": trade_qty,
        "open_price": open_avg, "close_price": close_avg,
        "pnl": pnl, "open_time": open_time, "close_time": close_time,
        "exit": exit_type,
    })

closed_trades.sort(key=lambda x: x["close_time"], reverse=True)

# Filter to losing trades
losers = [t for t in closed_trades if t["pnl"] < 0]

print()
print(f"=== LOSING TRADES vs IF-HELD ({len(losers)} trades) ===")
print(f"{'Opened (EST)':>15} {'Closed (EST)':>15} {'Symbol':8} {'Side':5} {'Qty':>5} "
      f"{'Entry':>10} {'Closed@':>10} {'Now':>10} {'Actual PnL':>11} {'If-Held PnL':>12} {'Verdict':>10}")
print("-" * 140)

total_actual = 0
total_ifheld = 0
would_win = 0
still_lose = 0
worse_if_held = 0

for t in losers:
    sym = t["sym"]
    if sym not in current_prices:
        continue

    # Current price: use bid for sells (closing a buy), ask for buys (closing a sell)
    if t["side"] == "buy":
        now_price = current_prices[sym]["bid"]
    else:
        now_price = current_prices[sym]["ask"]

    # If-held PnL
    if t["side"] == "buy":
        held_pnl_per_unit = now_price - t["open_price"]
    else:
        held_pnl_per_unit = t["open_price"] - now_price

    if "JPY" in sym:
        held_pnl = held_pnl_per_unit * t["qty"] * 1000
    else:
        held_pnl = held_pnl_per_unit * t["qty"] * 100000

    actual = t["pnl"]
    total_actual += actual
    total_ifheld += held_pnl

    if held_pnl > 0:
        verdict = "NOW WIN"
        would_win += 1
    elif held_pnl > actual:
        verdict = "LESS LOSS"
        still_lose += 1
    else:
        verdict = "WORSE"
        worse_if_held += 1

    ot = to_est(t["open_time"])
    ct = to_est(t["close_time"])

    print(f"{ot:>15} {ct:>15} {sym:8} {t['side']:5} {t['qty']:5.2f} "
          f"{t['open_price']:10.5f} {t['close_price']:10.5f} {now_price:10.5f} "
          f"${actual:+10.2f} ${held_pnl:+11.2f} {verdict:>10}")

print()
print(f"=== SUMMARY ===")
print(f"Total losing trades: {len(losers)}")
print(f"Would now be winners if held: {would_win}")
print(f"Still losing but less: {still_lose}")
print(f"Worse if held: {worse_if_held}")
print()
print(f"Total actual losses:  ${total_actual:+.2f}")
print(f"Total if-held value:  ${total_ifheld:+.2f}")
print(f"Difference (held - actual): ${total_ifheld - total_actual:+.2f}")
print()

# Also show open positions with current data in EST
print("=== CURRENT OPEN POSITIONS ===")
for p in broker.get_positions():
    if p.symbol not in BOT_SYMBOLS:
        continue
    open_est = (p.open_time + UTC_TO_EST).strftime("%m-%d %I:%M %p")
    sl_dist = abs(float(p.avg_price) - float(p.stop_loss)) if p.stop_loss else 0
    tp_dist = abs(float(p.take_profit) - float(p.avg_price)) if p.take_profit else 0
    if "JPY" in p.symbol:
        risk = sl_dist * float(p.quantity) * 1000
        reward = tp_dist * float(p.quantity) * 1000
    else:
        risk = sl_dist * float(p.quantity) * 100000
        reward = tp_dist * float(p.quantity) * 100000
    rr = reward / risk if risk > 0 else 0
    print(f"  {p.symbol:8} {p.side:4} qty={p.quantity} entry={p.avg_price} "
          f"SL={p.stop_loss} TP={p.take_profit} "
          f"uPnL=${float(p.unrealized_pnl):+.2f} risk=${risk:.0f} reward=${reward:.0f} R:R={rr:.1f} "
          f"opened={open_est}")
