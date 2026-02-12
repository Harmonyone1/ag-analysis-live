"""Analyze why loser pairs perform badly vs winner pairs."""
import sys, os
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(".env")
from src.adapters.tradelocker import TradeLockerAdapter

LOSERS = {"GBPNZD", "GBPJPY", "EURCAD"}
WINNERS = {"EURAUD", "CADJPY", "GBPAUD"}

tl = TradeLockerAdapter(
    environment=os.getenv("TL_ENVIRONMENT"),
    email=os.getenv("TL_EMAIL"),
    password=os.getenv("TL_PASSWORD"),
    server=os.getenv("TL_SERVER"),
    acc_num=int(os.getenv("TL_ACC_NUM", 0)),
)
tl.connect()

df = tl._api.get_all_orders(history=True)
filled = df[df["status"] == "Filled"].copy()

inst_ids = filled["tradableInstrumentId"].unique()
sym_map = {}
for iid in inst_ids:
    try:
        sym_map[iid] = tl._api.get_symbol_name_from_instrument_id(int(iid))
    except:
        sym_map[iid] = str(iid)
filled["symbol"] = filled["tradableInstrumentId"].map(sym_map)

ALL = LOSERS | WINNERS
filled = filled[filled["symbol"].isin(ALL)]

opens = filled[filled["isOpen"] == "true"].copy()
closes = filled[filled["isOpen"] == "false"].copy()

results = []
for _, cr in closes.iterrows():
    pos_id = cr["positionId"]
    om = opens[opens["positionId"] == pos_id]
    if len(om) == 0:
        continue
    orw = om.iloc[0]
    sym = str(cr["symbol"])
    sd = str(orw["side"])
    qty = float(cr["filledQty"])
    op = float(orw["avgPrice"])
    cp = float(cr["avgPrice"])
    is_jpy = "JPY" in sym
    mult = 1000 if is_jpy else 100000
    pip = 0.01 if is_jpy else 0.0001

    if sd == "buy":
        pnl = (cp - op) * qty * mult
        move_pips = (cp - op) / pip
    else:
        pnl = (op - cp) * qty * mult
        move_pips = (op - cp) / pip

    sl = float(orw.get("stopLoss", 0) or 0)
    tp = float(orw.get("takeProfit", 0) or 0)
    sl_dist = abs(op - sl) / pip if sl > 0 else 0
    tp_dist = abs(tp - op) / pip if tp > 0 else 0
    rr = tp_dist / sl_dist if sl_dist > 0 else 0

    open_ts = int(orw.get("createdDate", 0))
    close_ts = int(cr.get("createdDate", 0))
    dur_hours = (close_ts - open_ts) / (1000 * 3600) if open_ts > 0 and close_ts > 0 else 0

    if sl_dist > 0 and move_pips < 0 and abs(move_pips) >= sl_dist * 0.85:
        exit_type = "SL_HIT"
    elif tp_dist > 0 and move_pips > 0 and move_pips >= tp_dist * 0.85:
        exit_type = "TP_HIT"
    else:
        exit_type = "OTHER"

    grp = "LOSER" if sym in LOSERS else "WINNER"
    results.append({"group": grp, "symbol": sym, "side": sd, "qty": qty,
                   "pnl": round(pnl, 2), "move_pips": round(move_pips, 1),
                   "sl_pips": round(sl_dist, 1), "tp_pips": round(tp_dist, 1),
                   "rr": round(rr, 2), "dur_hours": round(dur_hours, 1),
                   "exit_type": exit_type})

import pandas as pd
rdf = pd.DataFrame(results)

for grp_name, symbols in [("LOSER", LOSERS), ("WINNER", WINNERS)]:
    g = rdf[rdf["group"] == grp_name]
    syms = ", ".join(sorted(symbols))
    wr = 100 * (g["pnl"] > 0).mean() if len(g) > 0 else 0
    print("=== %s PAIRS (%s) ===" % (grp_name, syms))
    print("Trades: %d, WR: %.1f%%, PnL: $%.2f" % (len(g), wr, g["pnl"].sum()))
    print("Avg SL distance: %.1f pips" % g["sl_pips"].mean())
    print("Avg TP distance: %.1f pips" % g["tp_pips"].mean())
    print("Avg R:R: %.2f" % g["rr"].mean())
    print("Avg duration: %.1fh" % g["dur_hours"].mean())
    exits = g["exit_type"].value_counts().to_dict()
    print("Exit types: %s" % exits)
    wins = g[g["pnl"] > 0]
    losses = g[g["pnl"] <= 0]
    if len(wins) > 0:
        print("Avg win: $%.2f (%.1f pips)" % (wins["pnl"].mean(), wins["move_pips"].mean()))
    if len(losses) > 0:
        print("Avg loss: $%.2f (%.1f pips)" % (losses["pnl"].mean(), losses["move_pips"].mean()))
    print()

# Per-symbol breakdown
print("=== PER-SYMBOL BREAKDOWN ===")
for sym in sorted(LOSERS | WINNERS):
    g = rdf[rdf["symbol"] == sym]
    if len(g) == 0:
        continue
    wr = 100 * (g["pnl"] > 0).mean()
    sl_exits = len(g[g["exit_type"] == "SL_HIT"])
    tp_exits = len(g[g["exit_type"] == "TP_HIT"])
    other = len(g[g["exit_type"] == "OTHER"])
    buys = len(g[g["side"] == "buy"])
    sells = len(g[g["side"] == "sell"])
    print("%s: %d trades (%dB/%dS), WR=%.0f%%, PnL=$%.2f, SL=%.0fp, TP=%.0fp, RR=%.1f, exits: %dSL/%dTP/%dOther, dur=%.1fh" % (
        sym, len(g), buys, sells, wr, g["pnl"].sum(),
        g["sl_pips"].mean(), g["tp_pips"].mean(), g["rr"].mean(),
        sl_exits, tp_exits, other, g["dur_hours"].mean()))
print()

# Print every trade for loser symbols
print("=== ALL LOSER PAIR TRADES ===")
losers = rdf[rdf["group"] == "LOSER"].sort_values(["symbol", "pnl"])
for _, r in losers.iterrows():
    print("%s %s %.2f  PnL=$%7.2f  move=%6.1fp  SL=%4.0fp TP=%4.0fp  RR=%.1f  dur=%5.1fh  %s" % (
        r["symbol"], r["side"], r["qty"], r["pnl"], r["move_pips"],
        r["sl_pips"], r["tp_pips"], r["rr"], r["dur_hours"], r["exit_type"]))
