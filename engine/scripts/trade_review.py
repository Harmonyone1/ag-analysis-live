"""Review all closed trades from broker history and compute P&L."""
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, ".")
from src.config import load_config
from src.adapters import TradeLockerAdapter

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
print(f"ACCOUNT: balance=${acct.balance} {acct.currency}")
print()

# Open positions
open_pos = broker.get_positions()
print(f"=== OPEN POSITIONS ({len(open_pos)}) ===")
for p in open_pos:
    sl_dist = abs(float(p.avg_price) - float(p.stop_loss)) if p.stop_loss else 0
    if "JPY" in p.symbol:
        risk = sl_dist * float(p.quantity) * 1000
    elif p.symbol == "ETHUSD":
        risk = sl_dist * float(p.quantity)
    else:
        risk = sl_dist * float(p.quantity) * 100000
    print(f"  {p.symbol:8} {p.side:4} qty={p.quantity} entry={p.avg_price} "
          f"SL={p.stop_loss} TP={p.take_profit} "
          f"uPnL=${float(p.unrealized_pnl):+.2f} risk=${risk:.2f}")
print()

# Get all historical orders
df = broker._api.get_all_orders(history=True)
filled = df[(df["status"] == "Filled") & (df["filledQty"] > 0)].copy()

# Group by positionId
positions = defaultdict(list)
for _, row in filled.iterrows():
    pos_id = row.get("positionId", 0)
    if pos_id:
        positions[pos_id].append(row)

# Reconstruct closed trades
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
        sym = str(inst_id)

    open_side = opens[0]["side"]
    open_qty = sum(float(o["filledQty"]) for o in opens)
    open_avg = (
        sum(float(o["avgPrice"]) * float(o["filledQty"]) for o in opens) / open_qty
        if open_qty
        else 0
    )
    close_qty = sum(float(o["filledQty"]) for o in closes)
    close_avg = (
        sum(float(o["avgPrice"]) * float(o["filledQty"]) for o in closes) / close_qty
        if close_qty
        else 0
    )

    open_time = min(int(o.get("createdDate", 0)) for o in opens)
    close_time = max(int(o.get("createdDate", 0)) for o in closes)

    # PnL calculation
    if open_side == "buy":
        pnl_per_unit = close_avg - open_avg
    else:
        pnl_per_unit = open_avg - close_avg

    trade_qty = min(open_qty, close_qty)
    if sym == "ETHUSD":
        pnl = pnl_per_unit * trade_qty
    elif "JPY" in sym:
        pnl = pnl_per_unit * trade_qty * 1000
    else:
        pnl = pnl_per_unit * trade_qty * 100000

    # Exit type
    close_types = [str(o.get("type", "")).lower() for o in closes]
    exit_type = "SL" if "stop" in close_types else "TP/Man"

    ot = datetime.fromtimestamp(open_time / 1000).strftime("%m-%d %H:%M") if open_time else "?"
    ct = datetime.fromtimestamp(close_time / 1000).strftime("%m-%d %H:%M") if close_time else "?"

    # Session (UTC hour of open)
    open_hour = datetime.fromtimestamp(open_time / 1000).hour if open_time else 0
    if 12 <= open_hour < 16:
        session = "peak"
    elif 8 <= open_hour < 12:
        session = "london"
    elif 16 <= open_hour < 20:
        session = "ny"
    elif 20 <= open_hour < 24:
        session = "late_ny"
    else:
        session = "asia"

    closed_trades.append({
        "sym": sym, "side": open_side, "qty": trade_qty,
        "open_price": open_avg, "close_price": close_avg,
        "pnl": pnl, "open_time": open_time, "close_time": close_time,
        "ot": ot, "ct": ct, "exit": exit_type, "session": session,
    })

# Sort by close time desc
closed_trades.sort(key=lambda x: x["close_time"], reverse=True)

# Only the 27 forex symbols the bot trades
BOT_SYMBOLS = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "EURNZD", "GBPAUD",
    "GBPNZD", "AUDNZD", "AUDCAD", "NZDCAD", "CADJPY", "CHFJPY",
    "EURCAD", "EURCHF", "GBPCAD", "GBPCHF", "AUDCHF", "NZDCHF", "CADCHF"
}

# Separate bot FX trades from everything else
fx_trades = [t for t in closed_trades if t["sym"] in BOT_SYMBOLS]
eth_trades = [t for t in closed_trades if t["sym"] == "ETHUSD"]
other_trades = [t for t in closed_trades if t["sym"] not in BOT_SYMBOLS and t["sym"] != "ETHUSD"]

if other_trades:
    other_total = sum(t["pnl"] for t in other_trades)
    print(f"(Excluded {len(other_trades)} non-bot trades from other symbols, PnL=${other_total:+.2f})")
    print()

# Print recent FX trades
print(f"=== RECENT FX CLOSED TRADES (last 40) ===")
header = f"{'Open':>12} {'Close':>12} {'Symbol':8} {'Side':5} {'Qty':>5} {'Entry':>10} {'Exit Price':>10} {'PnL':>10} {'Type':>6} {'Sess':>7}"
print(header)
print("-" * len(header))
for t in fx_trades[:40]:
    print(f"{t['ot']:>12} {t['ct']:>12} {t['sym']:8} {t['side']:5} {t['qty']:5.2f} "
          f"{t['open_price']:10.5f} {t['close_price']:10.5f} "
          f"${t['pnl']:+8.2f} {t['exit']:>6} {t['session']:>7}")

# FX summary
fx_total = sum(t["pnl"] for t in fx_trades)
fx_wins = sum(1 for t in fx_trades if t["pnl"] > 0)
fx_losses = len(fx_trades) - fx_wins
print()
print(f"FX TOTAL: {len(fx_trades)} trades  {fx_wins}W/{fx_losses}L  "
      f"WR={fx_wins/len(fx_trades)*100:.1f}%  PnL=${fx_total:+.2f}")

# ETH summary
if eth_trades:
    eth_total = sum(t["pnl"] for t in eth_trades)
    eth_wins = sum(1 for t in eth_trades if t["pnl"] > 0)
    print(f"ETH TOTAL: {len(eth_trades)} trades  {eth_wins}W/{len(eth_trades)-eth_wins}L  "
          f"WR={eth_wins/len(eth_trades)*100:.1f}%  PnL=${eth_total:+.2f}")

# FX by symbol
print()
print("=== FX PNL BY SYMBOL ===")
sym_pnl = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0, "sl_hits": 0})
for t in fx_trades:
    sym_pnl[t["sym"]]["pnl"] += t["pnl"]
    sym_pnl[t["sym"]]["trades"] += 1
    if t["pnl"] > 0:
        sym_pnl[t["sym"]]["wins"] += 1
    if t["exit"] == "SL":
        sym_pnl[t["sym"]]["sl_hits"] += 1

for sym in sorted(sym_pnl.keys(), key=lambda s: sym_pnl[s]["pnl"], reverse=True):
    d = sym_pnl[sym]
    wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
    print(f"  {sym:8} {d['trades']:3d} trades  WR={wr:5.1f}%  "
          f"PnL=${d['pnl']:+8.2f}  SL_hits={d['sl_hits']}")

# FX by session
print()
print("=== FX PNL BY SESSION ===")
sess_pnl = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
for t in fx_trades:
    sess_pnl[t["session"]]["pnl"] += t["pnl"]
    sess_pnl[t["session"]]["trades"] += 1
    if t["pnl"] > 0:
        sess_pnl[t["session"]]["wins"] += 1

for sess in ["asia", "london", "peak", "ny", "late_ny"]:
    if sess in sess_pnl:
        d = sess_pnl[sess]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        print(f"  {sess:8} {d['trades']:3d} trades  WR={wr:5.1f}%  PnL=${d['pnl']:+8.2f}")

# By week
print()
print("=== FX PNL BY WEEK ===")
week_pnl = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
for t in fx_trades:
    dt = datetime.fromtimestamp(t["close_time"] / 1000)
    week = dt.strftime("%Y-W%U")
    week_pnl[week]["pnl"] += t["pnl"]
    week_pnl[week]["trades"] += 1
    if t["pnl"] > 0:
        week_pnl[week]["wins"] += 1

for week in sorted(week_pnl.keys()):
    d = week_pnl[week]
    wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
    print(f"  {week}: {d['trades']:3d} trades  WR={wr:5.1f}%  PnL=${d['pnl']:+8.2f}")

# Avg win / avg loss
fx_win_pnl = [t["pnl"] for t in fx_trades if t["pnl"] > 0]
fx_loss_pnl = [t["pnl"] for t in fx_trades if t["pnl"] <= 0]
avg_win = sum(fx_win_pnl) / len(fx_win_pnl) if fx_win_pnl else 0
avg_loss = sum(fx_loss_pnl) / len(fx_loss_pnl) if fx_loss_pnl else 0
print()
print(f"Avg Win: ${avg_win:+.2f}  Avg Loss: ${avg_loss:+.2f}  "
      f"R:R ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss else "")
print(f"Largest Win: ${max(fx_win_pnl):+.2f}" if fx_win_pnl else "")
print(f"Largest Loss: ${min(fx_loss_pnl):+.2f}" if fx_loss_pnl else "")

# Consecutive losses
max_consec_loss = 0
curr = 0
for t in sorted(fx_trades, key=lambda x: x["close_time"]):
    if t["pnl"] <= 0:
        curr += 1
        max_consec_loss = max(max_consec_loss, curr)
    else:
        curr = 0
print(f"Max Consecutive Losses: {max_consec_loss}")
