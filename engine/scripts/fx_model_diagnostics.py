"""Full model diagnostics for FX gate model v3.0.0.

Loads saved training data from data/fx_training.npz and runs all 8 diagnostics:
1. Baseline comparisons (AUC uplift, Brier skill score)
2. Calibration curve / reliability diagram
3. EV vs probability bins (monotonicity check)
4. Stability by time (walk-forward folds, months)
5. Regime slices (volatility, time-of-day, day-of-week, trend, symbol)
6. Capacity / approval threshold curve
7. Cost sensitivity / break-even slippage sweep
8. Drawdown / tail profile

Usage:
    cd D:/ag-analysis-live/engine
    py scripts/fx_model_diagnostics.py
"""

import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data_and_model():
    """Load saved training data and the deployed model."""
    data_path = Path(__file__).resolve().parent.parent / "data" / "fx_training.npz"
    if not data_path.exists():
        logger.error("Training data not found at %s", data_path)
        logger.error("Run collect_fx_training_data.py first.")
        sys.exit(1)

    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    timestamps = data["timestamps"]
    feature_names = list(data["feature_names"])
    symbols = data["symbols"]

    logger.info("Loaded training data: %d samples, %d features", len(X), X.shape[1])
    logger.info("  Win rate: %.1f%%", y.mean() * 100)
    logger.info("  Symbols: %d unique", len(set(symbols)))
    valid_ts = timestamps[timestamps > 0]
    if len(valid_ts) > 0:
        logger.info("  Date range: %s to %s",
                     datetime.fromtimestamp(valid_ts[0] / 1000, tz=timezone.utc).date(),
                     datetime.fromtimestamp(valid_ts[-1] / 1000, tz=timezone.utc).date())

    # Load latest v3 model
    model_dir = Path(__file__).resolve().parent.parent.parent / "models"
    model_files = sorted(model_dir.glob("gate_model_v3*.pkl"))
    if not model_files:
        logger.error("No v3 model found in %s", model_dir)
        sys.exit(1)

    model_path = model_files[-1]
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    model = payload["model"]
    calibrator = payload["calibrator"]
    metadata = payload["metadata"]
    cal_method = metadata.get("calibration_method", "platt")

    logger.info("Loaded model: %s", model_path.name)
    logger.info("  Calibration: %s", cal_method)

    return X, y, timestamps, feature_names, symbols, model, calibrator, cal_method


def apply_calibrator(calibrator, method, p_raw):
    """Apply fitted calibrator to raw probabilities."""
    if method == "platt":
        p_clipped = np.clip(p_raw, 1e-6, 1 - 1e-6)
        logit = np.log(p_clipped / (1.0 - p_clipped))
        # Manual Platt scaling: extract coef/intercept and apply sigmoid
        # This avoids sklearn version mismatch issues with predict_proba
        try:
            return calibrator.predict_proba(logit.reshape(-1, 1))[:, 1]
        except (AttributeError, TypeError):
            # Fallback: manual logistic regression
            coef = calibrator.coef_[0, 0]
            intercept = calibrator.intercept_[0]
            z = coef * logit + intercept
            return 1.0 / (1.0 + np.exp(-z))
    else:
        return np.array(calibrator.transform(p_raw))


def get_walk_forward_predictions(X, y, config_params):
    """Re-run walk-forward CV to get out-of-sample predictions."""
    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBClassifier

    tscv = TimeSeriesSplit(n_splits=5)
    oof_probs = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)

    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logger.info("  Training fold %d (%d train, %d val)...",
                     fold_i, len(train_idx), len(val_idx))
        model = XGBClassifier(
            max_depth=config_params.get("max_depth", 6),
            n_estimators=config_params.get("n_estimators", 500),
            learning_rate=config_params.get("learning_rate", 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=config_params.get("min_child_weight", 20),
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42 + fold_i,
            early_stopping_rounds=30,
        )
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        oof_probs[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        fold_ids[val_idx] = fold_i

    return oof_probs, fold_ids


# ======================================================================
# DIAGNOSTIC 1: Baseline Comparisons
# ======================================================================
def diagnostic_baseline(cal_probs, y, base_rate):
    from sklearn.metrics import roc_auc_score, brier_score_loss

    logger.info("")
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 1: BASELINE COMPARISONS")
    logger.info("=" * 70)

    model_auc = roc_auc_score(y, cal_probs)
    logger.info("  Model AUC:     %.4f", model_auc)
    logger.info("  Random AUC:    0.5000")
    logger.info("  AUC uplift:    +%.4f", model_auc - 0.5)
    logger.info("")

    model_brier = brier_score_loss(y, cal_probs)
    base_rate_brier = brier_score_loss(y, np.full(len(y), base_rate))
    logger.info("  Model Brier:       %.4f (lower is better)", model_brier)
    logger.info("  Base-rate Brier:   %.4f (always predict %.1f%%)", base_rate_brier, base_rate * 100)
    brier_skill = 1 - model_brier / base_rate_brier
    logger.info("  Brier Skill Score: %.4f", brier_skill)
    if brier_skill > 0:
        logger.info("  PASS: Model outperforms naive base-rate predictor")
    else:
        logger.warning("  FAIL: Model does NOT outperform naive base-rate predictor")


# ======================================================================
# DIAGNOSTIC 2: Calibration Curve
# ======================================================================
def diagnostic_calibration(cal_probs, y):
    logger.info("")
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 2: CALIBRATION CURVE / RELIABILITY DIAGRAM")
    logger.info("=" * 70)

    bin_edges = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]
    logger.info("")
    logger.info("  %-15s  %-8s  %-12s  %-12s  %-10s", "Bin", "N", "Mean P(win)", "Actual WR", "Gap")
    logger.info("  " + "-" * 62)

    max_gap = 0
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (cal_probs >= lo) & (cal_probs < hi)
        n = mask.sum()
        if n == 0:
            continue
        mean_pred = float(cal_probs[mask].mean())
        actual_wr = float(y[mask].mean())
        gap = abs(mean_pred - actual_wr)
        max_gap = max(max_gap, gap)
        marker = ""
        if lo >= 0.55:
            marker = " <-- APPROVED REGION"
        if gap > 0.10:
            marker += " [MISCALIBRATED]"
        logger.info("  [%.2f, %.2f)    %-8d  %.4f        %.4f        %.4f%s",
                     lo, hi, n, mean_pred, actual_wr, gap, marker)

    logger.info("")
    logger.info("  Max calibration gap: %.4f", max_gap)
    if max_gap < 0.05:
        logger.info("  EXCELLENT calibration (max gap < 5%%)")
    elif max_gap < 0.10:
        logger.info("  GOOD calibration (max gap < 10%%)")
    else:
        logger.warning("  POOR calibration (max gap >= 10%%)")

    approved_mask = cal_probs >= 0.55
    n_approved = approved_mask.sum()
    if n_approved > 0:
        logger.info("")
        logger.info("  Approved region (P>=0.55): %d samples, actual WR=%.1f%%, mean P(win)=%.3f",
                     n_approved, y[approved_mask].mean() * 100, cal_probs[approved_mask].mean())


# ======================================================================
# DIAGNOSTIC 3: EV vs Probability Bins (Monotonicity)
# ======================================================================
def diagnostic_ev_bins(cal_probs, y):
    logger.info("")
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 3: EV vs PROBABILITY BINS (MONOTONICITY CHECK)")
    logger.info("=" * 70)

    # FX bot uses 2:1 R:R (TP=30 pips, SL=15 pips)
    logger.info("")
    logger.info("  EV formula: WinRate * 2.0 - (1 - WinRate) * 1.0  [2:1 R:R]")
    logger.info("")
    logger.info("  %-15s  %-6s  %-10s  %-10s  %-15s", "Bin", "N", "Win Rate", "EV", "Visual")
    logger.info("  " + "-" * 62)

    bins = [(0.30, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55),
            (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80)]

    prev_ev = -999
    monotonic = True

    for lo, hi in bins:
        mask = (cal_probs >= lo) & (cal_probs < hi)
        n = mask.sum()
        if n < 10:
            logger.info("  [%.2f, %.2f)    %-6d  (too few samples)", lo, hi, n)
            continue
        wr = float(y[mask].mean())
        ev = wr * 2.0 - (1 - wr) * 1.0
        bar_len = int(max(0, ev + 1) * 15)
        bar = "#" * bar_len
        sign = "+" if ev > 0 else ""
        marker = ""
        if ev < prev_ev and prev_ev != -999:
            monotonic = False
            marker = " <-- NON-MONOTONIC"
        logger.info("  [%.2f, %.2f)    %-6d  %.3f      %s%.3f      %s%s",
                     lo, hi, n, wr, sign, ev, bar, marker)
        prev_ev = ev

    logger.info("")
    if monotonic:
        logger.info("  PASS: EV is monotonically increasing across bins")
    else:
        logger.warning("  WARNING: EV is NOT monotonic — check for calibration issues")

    breakeven_wr = 1.0 / 3.0
    logger.info("  Breakeven win rate at 2:1 R:R: %.1f%%", breakeven_wr * 100)


# ======================================================================
# DIAGNOSTIC 4: Stability by Time
# ======================================================================
def diagnostic_time_stability(cal_probs, y, timestamps, fold_ids):
    from sklearn.metrics import roc_auc_score

    logger.info("")
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 4: STABILITY BY TIME")
    logger.info("=" * 70)

    threshold = 0.55

    # --- By Walk-Forward Fold ---
    logger.info("")
    logger.info("  A) Walk-Forward Folds (out-of-sample)")
    logger.info("  %-8s  %-6s  %-8s  %-10s  %-10s  %-10s  %-15s",
                "Fold", "N", "N_appr", "AUC", "WR(all)", "WR(appr)", "Date Range")
    logger.info("  " + "-" * 80)

    fold_evs = []
    for fold_i in range(5):
        mask = fold_ids == fold_i
        if mask.sum() == 0:
            continue
        p_fold = cal_probs[mask]
        y_fold = y[mask]
        ts_fold = timestamps[mask]
        try:
            auc = roc_auc_score(y_fold, p_fold)
        except ValueError:
            auc = 0.5
        appr_mask = p_fold >= threshold
        n_appr = appr_mask.sum()
        wr_all = float(y_fold.mean())
        wr_appr = float(y_fold[appr_mask].mean()) if n_appr > 0 else float("nan")
        ev_appr = wr_appr * 2.0 - (1 - wr_appr) if n_appr > 0 else float("nan")
        fold_evs.append(ev_appr)

        valid_ts = ts_fold[ts_fold > 0]
        if len(valid_ts) > 0:
            date_start = datetime.fromtimestamp(valid_ts[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            date_end = datetime.fromtimestamp(valid_ts[-1] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            date_start = date_end = "N/A"

        logger.info("  Fold %-2d  %-6d  %-8d  %.3f      %.1f%%       %.1f%%       %s to %s",
                     fold_i, mask.sum(), n_appr, auc, wr_all * 100,
                     wr_appr * 100 if not np.isnan(wr_appr) else 0, date_start, date_end)

    valid_evs = [e for e in fold_evs if not np.isnan(e)]
    if valid_evs:
        logger.info("")
        logger.info("  Fold EV spread: min=%+.3f, max=%+.3f, std=%.3f",
                     min(valid_evs), max(valid_evs), np.std(valid_evs))
        n_positive = sum(1 for e in valid_evs if e > 0)
        logger.info("  Folds with positive EV: %d/%d", n_positive, len(valid_evs))

    # --- By Month ---
    logger.info("")
    logger.info("  B) By Month")
    logger.info("  %-10s  %-6s  %-8s  %-10s  %-10s  %-10s",
                "Month", "N", "N_appr", "AUC", "WR(all)", "WR(appr)")
    logger.info("  " + "-" * 60)

    valid = ~np.isnan(cal_probs)
    ts_valid = timestamps[valid]
    y_valid = y[valid]
    p_valid = cal_probs[valid]

    months = np.array([datetime.fromtimestamp(max(t, 1) / 1000, tz=timezone.utc).strftime("%Y-%m")
                        for t in ts_valid])
    unique_months = sorted(set(months))

    for month in unique_months:
        mask = months == month
        n = mask.sum()
        if n < 20:
            continue
        p_m = p_valid[mask]
        y_m = y_valid[mask]
        try:
            auc = roc_auc_score(y_m, p_m)
        except ValueError:
            auc = 0.5
        appr = p_m >= threshold
        n_appr = appr.sum()
        wr_all = float(y_m.mean())
        wr_appr = float(y_m[appr].mean()) if n_appr > 0 else float("nan")
        logger.info("  %-10s  %-6d  %-8d  %.3f      %.1f%%       %.1f%%",
                     month, n, n_appr, auc, wr_all * 100,
                     wr_appr * 100 if not np.isnan(wr_appr) else 0)


# ======================================================================
# DIAGNOSTIC 5: Regime Slices
# ======================================================================
def diagnostic_regime_slices(X, cal_probs, y, timestamps, feature_names, symbols):
    logger.info("")
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 5: REGIME SLICES")
    logger.info("=" * 70)

    valid = ~np.isnan(cal_probs)
    X_v = X[valid]
    p_v = cal_probs[valid]
    y_v = y[valid]
    ts_v = timestamps[valid]
    sym_v = symbols[valid]

    threshold = 0.55

    def _slice_stats(name, mask):
        n = mask.sum()
        if n < 50:
            return
        p_s = p_v[mask]
        y_s = y_v[mask]
        appr = p_s >= threshold
        n_appr = appr.sum()
        wr_all = float(y_s.mean())
        wr_appr = float(y_s[appr].mean()) if n_appr > 0 else float("nan")
        ev_appr = wr_appr * 2.0 - (1 - wr_appr) if n_appr > 0 else float("nan")

        from sklearn.metrics import roc_auc_score as _auc
        try:
            auc = _auc(y_s, p_s)
        except ValueError:
            auc = 0.5
        logger.info("    %-25s  N=%-6d  AUC=%.3f  WR=%.1f%%  Appr=%d  WR(appr)=%.1f%%  EV=%+.3f",
                     name, n, auc, wr_all * 100, n_appr,
                     wr_appr * 100 if not np.isnan(wr_appr) else 0,
                     ev_appr if not np.isnan(ev_appr) else 0)

    # --- A) Volatility ---
    logger.info("")
    logger.info("  A) By Volatility Regime")
    atr_idx = None
    for i, name in enumerate(feature_names):
        if "atr" in name.lower():
            atr_idx = i
            break
    if atr_idx is not None:
        atr_vals = X_v[:, atr_idx]
        q33, q66 = np.percentile(atr_vals, [33, 66])
        _slice_stats("Low vol (ATR < p33)", atr_vals < q33)
        _slice_stats("Med vol (p33-p66)", (atr_vals >= q33) & (atr_vals < q66))
        _slice_stats("High vol (ATR > p66)", atr_vals >= q66)

    # --- B) Time-of-Day ---
    logger.info("")
    logger.info("  B) By Time of Day (UTC)")
    hours = np.array([datetime.fromtimestamp(max(t, 1) / 1000, tz=timezone.utc).hour for t in ts_v])

    _slice_stats("Asia (00:00-08:00)", (hours >= 0) & (hours < 8))
    _slice_stats("London (08:00-12:00)", (hours >= 8) & (hours < 12))
    _slice_stats("Peak (12:00-16:00)", (hours >= 12) & (hours < 16))
    _slice_stats("New York (16:00-20:00)", (hours >= 16) & (hours < 20))
    _slice_stats("Late NY (20:00-24:00)", (hours >= 20) & (hours < 24))

    # --- C) Day of Week ---
    logger.info("")
    logger.info("  C) By Day of Week")
    days = np.array([datetime.fromtimestamp(max(t, 1) / 1000, tz=timezone.utc).weekday() for t in ts_v])
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for d in range(7):
        _slice_stats(day_names[d], days == d)

    # --- D) By Symbol (top movers) ---
    logger.info("")
    logger.info("  D) By Symbol")
    unique_syms = sorted(set(sym_v))
    for sym in unique_syms:
        _slice_stats(sym, sym_v == sym)

    # --- E) By Currency (major currencies) ---
    logger.info("")
    logger.info("  E) By Major Currency Exposure")
    for ccy in ["EUR", "GBP", "USD", "JPY", "AUD", "NZD", "CAD", "CHF"]:
        mask = np.array([ccy in s for s in sym_v])
        _slice_stats(f"Contains {ccy}", mask)


# ======================================================================
# DIAGNOSTIC 6: Capacity / Threshold Curve
# ======================================================================
def diagnostic_capacity(cal_probs, y):
    logger.info("")
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 6: CAPACITY / APPROVAL THRESHOLD CURVE")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  %-12s  %-8s  %-8s  %-10s  %-10s  %-15s",
                "Threshold", "N_appr", "Appr %", "Win Rate", "EV (2:1)", "Visual")
    logger.info("  " + "-" * 70)

    thresholds = [0.40, 0.45, 0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75]
    best_ev = -999
    best_thresh = 0

    for thresh in thresholds:
        mask = cal_probs >= thresh
        n = mask.sum()
        pct = n / len(cal_probs) * 100
        if n < 5:
            logger.info("  >= %.2f       %-8d  %.1f%%     (too few)", thresh, n, pct)
            continue
        wr = float(y[mask].mean())
        ev = wr * 2.0 - (1 - wr) * 1.0
        if ev > best_ev:
            best_ev = ev
            best_thresh = thresh
        bar_len = int(max(0, ev + 1) * 10)
        bar = "#" * bar_len
        marker = " <-- CURRENT" if thresh == 0.55 else ""
        logger.info("  >= %.2f       %-8d  %.1f%%     %.1f%%       %+.3f      %s%s",
                     thresh, n, pct, wr * 100, ev, bar, marker)

    logger.info("")
    logger.info("  Optimal threshold (max EV): %.2f (EV=%+.3f)", best_thresh, best_ev)


# ======================================================================
# DIAGNOSTIC 7: Cost Sensitivity / Break-Even Slippage
# ======================================================================
def diagnostic_cost_sensitivity(cal_probs, y, timestamps):
    logger.info("")
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 7: BREAK-EVEN SLIPPAGE SWEEP")
    logger.info("=" * 70)

    # For FX with SL=15 pips: 1 pip on EURUSD ~ 10 bps (on 1.08)
    # So 15 pips SL ~ 150 bps. Cost is spread + slippage in bps
    SL_BPS = 150

    hours = np.array([datetime.fromtimestamp(max(t, 1) / 1000, tz=timezone.utc).hour for t in timestamps])
    days = np.array([datetime.fromtimestamp(max(t, 1) / 1000, tz=timezone.utc).weekday() for t in timestamps])

    sessions = {
        "All": np.ones(len(cal_probs), dtype=bool),
        "London (08-12)": (hours >= 8) & (hours < 12),
        "Peak (12-16)": (hours >= 12) & (hours < 16),
        "NY (16-20)": (hours >= 16) & (hours < 20),
        "Asia (00-08)": (hours >= 0) & (hours < 8),
        "Sat": days == 5,
        "No-Sat": days != 5,
    }

    thresholds = [0.55, 0.60, 0.65]
    cost_bps_range = [0, 2, 5, 10, 15, 20, 30, 50]

    for thresh in thresholds:
        logger.info("")
        logger.info("  Threshold >= %.2f", thresh)
        logger.info("  %-18s  " + "  ".join(f"{c:>5d}bps" for c in cost_bps_range), "Session")
        logger.info("  " + "-" * (20 + len(cost_bps_range) * 9))

        for sess_name, sess_mask in sessions.items():
            mask = (cal_probs >= thresh) & sess_mask
            n = mask.sum()
            if n < 20:
                continue
            wr = float(y[mask].mean())
            base_ev = wr * 2.0 - (1 - wr)
            ev_strs = []
            for cost_bps in cost_bps_range:
                cost_r = cost_bps / SL_BPS
                ev = base_ev - cost_r
                ev_strs.append(f" {ev:+.3f}" if ev > 0 else f" {ev:+.3f}*")
            logger.info("  %-18s  %s  (N=%d, WR=%.1f%%)",
                         sess_name, "  ".join(ev_strs), n, wr * 100)

        logger.info("")
        for sess_name, sess_mask in sessions.items():
            mask = (cal_probs >= thresh) & sess_mask
            n = mask.sum()
            if n < 20:
                continue
            wr = float(y[mask].mean())
            base_ev = wr * 2.0 - (1 - wr)
            be_bps = base_ev * SL_BPS
            logger.info("  Break-even cost (%s): %.0f bps round-trip", sess_name, be_bps)


# ======================================================================
# DIAGNOSTIC 8: Drawdown / Tail Profile
# ======================================================================
def diagnostic_drawdown(cal_probs, y, timestamps):
    logger.info("")
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 8: DRAWDOWN / TAIL PROFILE")
    logger.info("=" * 70)

    hours = np.array([datetime.fromtimestamp(max(t, 1) / 1000, tz=timezone.utc).hour for t in timestamps])
    days = np.array([datetime.fromtimestamp(max(t, 1) / 1000, tz=timezone.utc).weekday() for t in timestamps])

    scenarios = {
        ">=0.55 (all)": cal_probs >= 0.55,
        ">=0.60 (all)": cal_probs >= 0.60,
        ">=0.65 (all)": cal_probs >= 0.65,
        ">=0.55 Peak 12-16": (cal_probs >= 0.55) & (hours >= 12) & (hours < 16),
        ">=0.55 London 08-12": (cal_probs >= 0.55) & (hours >= 8) & (hours < 12),
        ">=0.55 Saturday": (cal_probs >= 0.55) & (days == 5),
        ">=0.55 Asia": (cal_probs >= 0.55) & (hours >= 0) & (hours < 8),
    }

    for name, mask in scenarios.items():
        n = mask.sum()
        if n < 10:
            continue

        outcomes = y[mask]
        n_trades = len(outcomes)
        n_wins = int(outcomes.sum())
        n_losses = n_trades - n_wins
        wr = n_wins / n_trades

        pnl = np.where(outcomes == 1, 2.0, -1.0)
        cum_pnl = np.cumsum(pnl)

        peak = np.maximum.accumulate(cum_pnl)
        drawdown = peak - cum_pnl
        max_dd = float(drawdown.max())
        max_dd_idx = int(np.argmax(drawdown))

        max_consec_loss = 0
        current_streak = 0
        all_streaks = []
        for o in outcomes:
            if o == 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    all_streaks.append(current_streak)
                current_streak = 0
            max_consec_loss = max(max_consec_loss, current_streak)
        if current_streak > 0:
            all_streaks.append(current_streak)

        max_consec_win = 0
        current_streak = 0
        for o in outcomes:
            if o == 1:
                current_streak += 1
            else:
                current_streak = 0
            max_consec_win = max(max_consec_win, current_streak)

        worst_10 = worst_20 = float("inf")
        for window in [10, 20]:
            if n_trades >= window:
                rolling_pnl = np.convolve(pnl, np.ones(window), mode='valid')
                worst = float(rolling_pnl.min())
                if window == 10:
                    worst_10 = worst
                else:
                    worst_20 = worst

        gross_wins = float(pnl[pnl > 0].sum())
        gross_losses = float(abs(pnl[pnl < 0].sum()))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        final_pnl = float(cum_pnl[-1])

        logger.info("")
        logger.info("  === %s ===", name)
        logger.info("  Trades: %d  |  Wins: %d  |  Losses: %d  |  WR: %.1f%%",
                     n_trades, n_wins, n_losses, wr * 100)
        logger.info("  Final PnL: %+.1fR  |  Profit Factor: %.2f", final_pnl, profit_factor)
        logger.info("  Max Drawdown: %.1fR (at trade #%d)", max_dd, max_dd_idx)
        logger.info("  Max Consecutive Losses: %d  |  Max Consecutive Wins: %d",
                     max_consec_loss, max_consec_win)
        if all_streaks:
            logger.info("  Avg Loss Streak: %.1f  |  Median Loss Streak: %.0f",
                         np.mean(all_streaks), np.median(all_streaks))
        if n_trades >= 10:
            logger.info("  Worst 10-trade window: %+.1fR", worst_10)
        if n_trades >= 20:
            logger.info("  Worst 20-trade window: %+.1fR", worst_20)
        logger.info("  Expectancy per trade: %+.3fR", final_pnl / n_trades)

        # Monthly breakdown
        ts_masked = timestamps[mask]
        months = np.array([datetime.fromtimestamp(max(t, 1) / 1000, tz=timezone.utc).strftime("%Y-%m")
                           for t in ts_masked])
        unique_months = sorted(set(months))
        if len(unique_months) >= 3:
            monthly_pnl = []
            logger.info("  Monthly PnL:")
            for month in unique_months:
                m_mask = months == month
                m_pnl = pnl[m_mask]
                m_total = float(m_pnl.sum())
                m_trades = len(m_pnl)
                m_wr = float((m_pnl > 0).mean())
                monthly_pnl.append(m_total)
                bar = "#" * int(max(0, m_total))
                logger.info("    %s: %+6.1fR (%d trades, WR=%.0f%%) %s",
                             month, m_total, m_trades, m_wr * 100, bar)
            if len(monthly_pnl) >= 3:
                monthly_arr = np.array(monthly_pnl)
                sharpe = float(monthly_arr.mean() / monthly_arr.std()) if monthly_arr.std() > 0 else float("inf")
                n_pos = sum(1 for m in monthly_pnl if m > 0)
                logger.info("  Monthly Sharpe: %.2f | Positive months: %d/%d", sharpe, n_pos, len(monthly_pnl))


# ======================================================================
# MAIN
# ======================================================================
def main():
    logger.info("=" * 70)
    logger.info("FX MODEL DIAGNOSTICS (v3.0.0) — ALL 8 CHECKS")
    logger.info("=" * 70)

    X, y, timestamps, feature_names, symbols, model, calibrator, cal_method = load_data_and_model()

    base_rate = float(y.mean())
    # Match the training config: max_depth=6, min_child_weight=20
    config_params = {"max_depth": 6, "n_estimators": 500, "learning_rate": 0.05, "min_child_weight": 20}

    logger.info("")
    logger.info("Re-running walk-forward CV for out-of-sample predictions...")
    oof_probs, fold_ids = get_walk_forward_predictions(X, y, config_params)

    valid = ~np.isnan(oof_probs)
    cal_probs = np.full(len(y), np.nan)
    cal_probs[valid] = apply_calibrator(calibrator, cal_method, oof_probs[valid])

    n_valid = valid.sum()
    logger.info("Got %d out-of-sample predictions (%.0f%% of data)", n_valid, n_valid / len(y) * 100)

    p_valid = cal_probs[valid]
    y_valid = y[valid]

    diagnostic_baseline(p_valid, y_valid, base_rate)
    diagnostic_calibration(p_valid, y_valid)
    diagnostic_ev_bins(p_valid, y_valid)
    diagnostic_time_stability(cal_probs, y, timestamps, fold_ids)
    diagnostic_regime_slices(X, cal_probs, y, timestamps, feature_names, symbols)
    diagnostic_capacity(p_valid, y_valid)

    # Use valid subset with timestamps for cost/drawdown
    ts_valid = timestamps[valid]
    diagnostic_cost_sensitivity(p_valid, y_valid, ts_valid)
    diagnostic_drawdown(p_valid, y_valid, ts_valid)

    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL 8 DIAGNOSTICS COMPLETE")
    logger.info("=" * 70)
    logger.info("  Feature names: %s", ", ".join(feature_names[:10]) + "...")
    logger.info("  Total features: %d", len(feature_names))
    logger.info("")


if __name__ == "__main__":
    main()
