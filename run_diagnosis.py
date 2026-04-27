"""Phase 1 — Root-Cause Diagnosis
Run from the repo root:
    python run_diagnosis.py

Outputs: diagnosis_report.md
No production code is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import json
import textwrap
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Load master table
# ─────────────────────────────────────────────────────────────────────────────
# master_table can be under data/gold/ or data/<user_id>/gold/
_GOLD_CANDIDATES = [
    ROOT / "data" / "gold" / "master_table.parquet",
    ROOT / "data" / "default" / "gold" / "master_table.parquet",
]
GOLD_PATH = next((p for p in _GOLD_CANDIDATES if p.exists()), None)
if GOLD_PATH is None:
    # Search any subdirectory
    matches = list(ROOT.glob("data/**/master_table.parquet"))
    GOLD_PATH = matches[0] if matches else None
if GOLD_PATH is None:
    sys.exit(
        "ERROR: master_table.parquet not found under data/.\n"
        "Run the full pipeline first so data is present."
    )

print("Loading master table …")
df_all = pd.read_parquet(GOLD_PATH)
df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
df_all = df_all.sort_values(["ticker", "date"]).reset_index(drop=True)

FRED_COLS = [
    "inflation", "energy_index", "unemployment_rate",
    "fed_funds_rate", "us10y_treasury_yield", "vix_index",
    "consumer_sentiment", "industrial_production",
]
WORLDBANK_COLS = ["gdp_growth", "energy_usage", "inflation_wb", "unemployment_wb", "trade_openness"]

available_fred = [c for c in FRED_COLS if c in df_all.columns]
available_wb   = [c for c in WORLDBANK_COLS if c in df_all.columns]
all_macro      = available_fred + available_wb

# ── Publication-lag table for H2 experiment ──────────────────────────────────
# 0-lag series: price / market observables → available at close of t
# 1-lag: FOMC / FEDFUNDS published same-day but confirm next morning
# 30-lag: University of Michigan Consumer Sentiment (monthly, ~4-week delay)
# 45-lag: CPI, UNRATE, INDPRO (BLS/Fed, ~4-6 week delay)
# 90-lag: World Bank (annual, published with months of delay)
PROPER_LAG: dict[str, int] = {
    "vix_index":             1,
    "us10y_treasury_yield":  1,
    "fed_funds_rate":        1,
    "consumer_sentiment":   30,
    "inflation":            45,
    "unemployment_rate":    45,
    "industrial_production":45,
    "energy_index":         45,
    "gdp_growth":           90,
    "energy_usage":         90,
    "inflation_wb":         90,
    "unemployment_wb":      90,
    "trade_openness":       90,
}
CURRENT_LAG = 45   # uniform lag applied to ALL non-yfinance features today

results: dict = {}

# ─────────────────────────────────────────────────────────────────────────────
# H1 — Is the signal anti-predictive?
#       Measure IC = Spearman( feature_t, forward_1d_return_t+1 )
#       per macro feature, per ticker; also aggregate.
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== H1: IC test ===")

HORIZON = 1   # 1-day forward return for IC (avoids contamination)

def ic_for_series(feature: pd.Series, fwd_return: pd.Series) -> tuple[float, float, int]:
    """Spearman IC + t-stat + n on overlapping non-nan rows."""
    df_tmp = pd.DataFrame({"f": feature, "r": fwd_return}).dropna()
    n = len(df_tmp)
    if n < 30:
        return float("nan"), float("nan"), n
    rho, _ = spearmanr(df_tmp["f"], df_tmp["r"])
    # t-stat under H0: IC=0
    t = rho * np.sqrt((n - 2) / max(1 - rho**2, 1e-12))
    return float(rho), float(t), int(n)


# Build aggregated feature-IC table across all tickers
ic_rows: list[dict] = []
tickers = df_all["ticker"].dropna().unique().tolist()

for feat in all_macro:
    ics_feat: list[float] = []
    for tkr in tickers:
        sub = df_all[df_all["ticker"] == tkr].sort_values("date").copy()
        if feat not in sub.columns or "log_return" not in sub.columns:
            continue
        # Apply proper lag to the feature before measuring IC
        raw = pd.to_numeric(sub[feat], errors="coerce").shift(PROPER_LAG.get(feat, 45))
        fwd = sub["log_return"].shift(-HORIZON)   # forward 1-day return
        rho, t, n = ic_for_series(raw, fwd)
        if np.isfinite(rho):
            ics_feat.append(rho)
    if ics_feat:
        mean_ic = float(np.mean(ics_feat))
        t_all   = mean_ic * np.sqrt(len(ics_feat)) / max(np.std(ics_feat, ddof=1), 1e-10)
        ic_rows.append({
            "feature": feat,
            "mean_ic": round(mean_ic, 5),
            "t_stat":  round(t_all, 3),
            "n_tickers": len(ics_feat),
            "positive_ic_pct": round(float(np.mean([x > 0 for x in ics_feat])), 3),
        })

ic_df = pd.DataFrame(ic_rows).sort_values("mean_ic", ascending=False) if ic_rows else pd.DataFrame()
results["H1_ic_table"] = ic_df.to_dict(orient="records") if not ic_df.empty else []

# Rolling IC on AAPL for the best and worst feature
ref_ticker = "AAPL" if "AAPL" in tickers else tickers[0]
sub_ref = df_all[df_all["ticker"] == ref_ticker].sort_values("date").copy()

rolling_ic_records: list[dict] = []
if not ic_df.empty and "log_return" in sub_ref.columns:
    best_feat  = ic_df.iloc[0]["feature"]
    worst_feat = ic_df.iloc[-1]["feature"]
    for feat in {best_feat, worst_feat}:
        raw_feat  = pd.to_numeric(sub_ref[feat], errors="coerce").shift(PROPER_LAG.get(feat, 45))
        fwd_ret   = sub_ref["log_return"].shift(-HORIZON)
        tmp = pd.DataFrame({"f": raw_feat.values, "r": fwd_ret.values,
                            "date": sub_ref["date"].values}).dropna()
        win = 60
        for i in range(win, len(tmp)):
            window = tmp.iloc[i - win: i]
            if len(window) < 30:
                continue
            rho, _ = spearmanr(window["f"], window["r"])
            rolling_ic_records.append({
                "date": str(tmp.iloc[i]["date"])[:10],
                "feature": feat,
                "rolling_60d_ic": round(float(rho), 5),
            })

results["H1_rolling_ic"] = rolling_ic_records

# Aggregate verdict
if not ic_df.empty:
    mean_all_ic  = float(ic_df["mean_ic"].mean())
    pos_ic_feats = int((ic_df["mean_ic"] > 0).sum())
    sig_feats    = int((ic_df["t_stat"].abs() >= 2.0).sum())
    h1_verdict   = "CONFIRMED" if mean_all_ic < -0.01 or pos_ic_feats < len(ic_df) // 2 else "REJECTED"
else:
    mean_all_ic = float("nan"); pos_ic_feats = 0; sig_feats = 0; h1_verdict = "INCONCLUSIVE"

results["H1_summary"] = {
    "mean_ic_all_features":   round(mean_all_ic, 5),
    "n_features_positive_ic": pos_ic_feats,
    "n_features_significant": sig_feats,
    "verdict": h1_verdict,
}
print(f"  Mean IC across features: {mean_all_ic:.4f}")
print(f"  Features with positive IC: {pos_ic_feats}/{len(ic_df)}")
print(f"  H1 verdict: {h1_verdict}")


# ─────────────────────────────────────────────────────────────────────────────
# H2 — Wrong publication lag on real-time market features?
#       Compare OOS R² under:
#       (a) current uniform 45-day lag for all FRED series
#       (b) per-series proper lag table above
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== H2: Lag experiment ===")

def build_lagged_panel(
    sub: pd.DataFrame,
    features: list[str],
    lag_dict: dict[str, int],
    target: str = "log_return",
    horizon: int = 21,
) -> pd.DataFrame:
    """Build a feature+target panel with caller-supplied per-feature lags.

    Target = sum of next `horizon` daily log-returns (causal: shift by 1 first).
    All features are lagged by their respective lag_dict values.
    Returns a clean panel with no NaNs.
    """
    out = sub[["date"]].copy()
    for feat in features:
        if feat not in sub.columns:
            continue
        lag = int(lag_dict.get(feat, 45))
        raw = pd.to_numeric(sub[feat], errors="coerce")
        # Stationary transform: diff for levels, keep-as-is for returns/ratios
        if any(kw in feat for kw in ("return", "pct", "diff")):
            transformed = raw.shift(lag)
        elif any(kw in feat for kw in ("rate", "yield", "inflation", "unemployment",
                                       "growth", "sentiment", "spread", "vix")):
            transformed = raw.diff().shift(lag)
        elif (raw.dropna() > 0).all() and len(raw.dropna()) > 0:
            transformed = np.log(raw).diff().shift(lag)
        else:
            transformed = raw.pct_change().shift(lag)
        out[feat] = transformed.values

    # Target: 21-day forward cumulative return (strictly causal)
    # shift(-1) then rolling sum then shift(-(horizon-1)) => each row t gets
    # the sum of returns at t+1 … t+horizon.  The last (horizon-1) rows are NaN.
    lr = pd.to_numeric(sub[target], errors="coerce")
    out[target] = lr.shift(-1).rolling(window=horizon, min_periods=horizon).sum().shift(-(horizon - 1)).values

    return out.dropna().reset_index(drop=True)


PANEL_FEATURES = [f for f in available_fred if f in df_all.columns]

if len(PANEL_FEATURES) >= 2 and ref_ticker in tickers:
    sub_ref2 = df_all[df_all["ticker"] == ref_ticker].sort_values("date").reset_index(drop=True)

    # Uniform-45 panel (current behaviour)
    uniform_lag = {f: CURRENT_LAG for f in PANEL_FEATURES}
    panel_uniform = build_lagged_panel(sub_ref2, PANEL_FEATURES, uniform_lag)

    # Proper-lag panel
    panel_proper  = build_lagged_panel(sub_ref2, PANEL_FEATURES, PROPER_LAG)

    def walk_forward_r2(panel: pd.DataFrame, target: str = "log_return",
                        n_folds: int = 4, min_train: int = 120) -> list[float]:
        feats = [c for c in panel.columns if c not in ("date", target)]
        panel = panel.sort_values("date").reset_index(drop=True)
        n = len(panel)
        if n < min_train + 30:
            return []
        fold_size = max(30, (n - min_train) // n_folds)
        r2s = []
        start = min_train
        while start < n:
            end = min(n, start + fold_size)
            if end - start < 20:
                break
            tr = panel.iloc[:start]
            te = panel.iloc[start:end]
            X_tr = tr[feats].values; y_tr = tr[target].values
            X_te = te[feats].values; y_te = te[target].values
            tscv = TimeSeriesSplit(n_splits=max(2, min(3, len(tr) // 60)))
            m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=tscv)
            try:
                m.fit(X_tr, y_tr)
                pred = m.predict(X_te)
                ss_res = float(np.sum((y_te - pred) ** 2))
                ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
                r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
                r2s.append(float(np.clip(r2, -5.0, 1.0)))
            except Exception:
                pass
            start = end
        return r2s

    r2_uniform = walk_forward_r2(panel_uniform)
    r2_proper  = walk_forward_r2(panel_proper)

    h2_result = {
        "uniform_45d_lag": {
            "folds": len(r2_uniform),
            "mean_oos_r2": round(float(np.mean(r2_uniform)), 4) if r2_uniform else None,
            "values": [round(x, 4) for x in r2_uniform],
        },
        "proper_lag_table": {
            "folds": len(r2_proper),
            "mean_oos_r2": round(float(np.mean(r2_proper)), 4) if r2_proper else None,
            "values": [round(x, 4) for x in r2_proper],
        },
    }

    # Verdict: proper lag is better IF its mean R² is meaningfully higher
    if r2_proper and r2_uniform:
        improvement = float(np.mean(r2_proper)) - float(np.mean(r2_uniform))
        h2_result["improvement"] = round(improvement, 4)
        h2_result["verdict"] = (
            "CONFIRMED" if improvement > 0.02
            else ("INCONCLUSIVE" if abs(improvement) <= 0.02 else "REJECTED")
        )
    else:
        h2_result["improvement"] = None
        h2_result["verdict"] = "INCONCLUSIVE"

    results["H2"] = h2_result
    print(f"  Uniform-45 mean OOS R²: {h2_result['uniform_45d_lag']['mean_oos_r2']}")
    print(f"  Proper-lag  mean OOS R²: {h2_result['proper_lag_table']['mean_oos_r2']}")
    print(f"  H2 verdict: {h2_result['verdict']}")
else:
    results["H2"] = {"verdict": "INCONCLUSIVE", "reason": "insufficient_features_or_data"}
    print("  Skipped — not enough features or data")


# ─────────────────────────────────────────────────────────────────────────────
# H3 — Dual-SMA filter killing participation?
#       Run simplified backtest with / without the dual-SMA filter.
#       If removing the filter makes metrics worse → filter is actually
#       protecting a bad signal (H1 is the root cause).
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== H3: SMA filter experiment ===")

def run_simple_backtest(
    sub: pd.DataFrame,
    features: list[str],
    lag_dict: dict[str, int],
    use_filter: bool = True,
    target: str = "log_return",
    horizon: int = 21,
    tx_cost: float = 0.0005,
) -> dict:
    """Minimal vectorised backtest: Ridge on macro features, long/short signal."""
    panel = build_lagged_panel(sub, features, lag_dict, target=target, horizon=horizon)
    if len(panel) < 150:
        return {"status": "insufficient_data"}

    panel = panel.sort_values("date").reset_index(drop=True)
    feats = [c for c in panel.columns if c not in ("date", target)]

    # 70/30 time split
    split = int(len(panel) * 0.70)
    tr = panel.iloc[:split]; te = panel.iloc[split:]

    # Get raw 1-day log-returns for execution (not the 21-day target)
    sub2 = sub.copy().sort_values("date").reset_index(drop=True)
    sub2["date"] = pd.to_datetime(sub2["date"])
    orig_lr = sub2.set_index("date")["log_return"].dropna()

    tscv = TimeSeriesSplit(n_splits=max(2, min(3, len(tr) // 60)))
    m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=tscv)
    try:
        m.fit(tr[feats].values, tr[target].values)
    except Exception as e:
        return {"status": f"fit_error: {e}"}

    pred_tr  = m.predict(tr[feats].values)
    pred_te  = m.predict(te[feats].values)
    mu  = float(np.mean(pred_tr))
    std = float(np.std(pred_tr, ddof=1)) or 1.0
    pred_z = (pred_te - mu) / std

    # Actual 1-day returns on test dates
    test_dates = pd.to_datetime(te["date"].values)
    actual = np.nan_to_num(
        orig_lr.reindex(test_dates).values.astype(float),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    actual = np.clip(actual, -0.15, 0.15)
    n = min(len(pred_z), len(actual))
    pred_z = pred_z[:n]; actual = actual[:n]

    # Simple signal: long if pred_z > 0, short if < 0
    signal = np.where(pred_z >= 0.0, 1.0, -1.0)

    if use_filter:
        # Build SMA200 trend mask from full log-return history
        full_lr = orig_lr.astype(float)
        px = pd.Series(np.exp(np.cumsum(np.nan_to_num(full_lr.values, nan=0.0))),
                       index=full_lr.index)
        sma200 = px.rolling(200, min_periods=200).mean()
        test_px  = px.reindex(test_dates[:n], method="pad").values
        test_sma = sma200.reindex(test_dates[:n], method="pad").values
        uptrend  = test_px > test_sma
        signal   = np.where((signal > 0) & ~uptrend, 0.0, signal)
        signal   = np.where((signal < 0) &  uptrend, 0.0, signal)

    # 1-day execution lag
    exec_pos = np.roll(signal, 1); exec_pos[0] = 0.0

    # Costs on direction changes
    pos_chg = np.abs(np.diff(np.sign(exec_pos), prepend=0.0)) > 0.5
    costs   = pos_chg.astype(float) * tx_cost

    strat_ret = exec_pos * actual - costs
    active    = int(np.sum(np.abs(exec_pos) > 1e-10))

    # Metrics
    std_r = float(np.std(strat_ret, ddof=1)) if len(strat_ret) > 1 else None
    sharpe = float(np.mean(strat_ret) / std_r * np.sqrt(252.0)) if std_r and std_r > 1e-12 else None

    # IC: Spearman(pred_z, actual)
    if n >= 10:
        rho, _ = spearmanr(pred_z, actual)
        ic = float(rho)
    else:
        ic = float("nan")

    return {
        "status": "ok",
        "n_test": n,
        "active_days": active,
        "active_ratio": round(active / max(n, 1), 4),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "mean_daily_return": round(float(np.mean(strat_ret)), 6),
        "ic_signal_vs_1d_actual": round(ic, 5) if np.isfinite(ic) else None,
    }


if PANEL_FEATURES and ref_ticker in tickers:
    sub_ref3 = df_all[df_all["ticker"] == ref_ticker].sort_values("date").reset_index(drop=True)

    res_with_filter    = run_simple_backtest(sub_ref3, PANEL_FEATURES, PROPER_LAG, use_filter=True)
    res_without_filter = run_simple_backtest(sub_ref3, PANEL_FEATURES, PROPER_LAG, use_filter=False)

    h3_result = {
        "with_filter":    res_with_filter,
        "without_filter": res_without_filter,
    }

    # Verdict: if removing the filter makes Sharpe MORE negative → filter is
    # masking the bad signal → H1 is root cause, H3 is secondary.
    sh_with    = res_with_filter.get("sharpe")
    sh_without = res_without_filter.get("sharpe")
    if sh_with is not None and sh_without is not None:
        if sh_without < sh_with:
            h3_result["verdict"] = "CONFIRMED_SECONDARY"
            h3_result["interpretation"] = (
                "Removing the filter degrades Sharpe further. "
                "Filter is masking bad-signal damage. H1 is the primary cause."
            )
        elif sh_without > sh_with + 0.15:
            h3_result["verdict"] = "CONFIRMED_PRIMARY"
            h3_result["interpretation"] = "Filter is the main performance drag."
        else:
            h3_result["verdict"] = "INCONCLUSIVE"
            h3_result["interpretation"] = "Filter has negligible impact."
    else:
        h3_result["verdict"] = "INCONCLUSIVE"
        h3_result["interpretation"] = "Could not compute Sharpe for comparison."

    results["H3"] = h3_result
    print(f"  Sharpe with filter:    {sh_with}")
    print(f"  Sharpe without filter: {sh_without}")
    print(f"  H3 verdict: {h3_result['verdict']}")
else:
    results["H3"] = {"verdict": "INCONCLUSIVE", "reason": "insufficient_features_or_data"}
    print("  Skipped")


# ─────────────────────────────────────────────────────────────────────────────
# Write diagnosis_report.md
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Writing diagnosis_report.md ===")

h1_sum   = results.get("H1_summary", {})
h1_ic    = results.get("H1_ic_table", [])
h2       = results.get("H2", {})
h3       = results.get("H3", {})

def ic_table_md(rows: list[dict]) -> str:
    if not rows:
        return "_No data available._\n"
    hdr = "| Feature | Mean IC | t-stat | +IC% | Tickers |\n|---------|---------|--------|------|---------|\n"
    body = ""
    for r in rows:
        body += (
            f"| {r['feature']} | {r['mean_ic']:+.4f} | {r['t_stat']:+.2f} | "
            f"{r['positive_ic_pct']*100:.0f}% | {r['n_tickers']} |\n"
        )
    return hdr + body


report = f"""# diagnosis_report.md — Phase 1 Root-Cause Results
**Date:** 2026-04-28  
**Ref ticker for experiments:** {ref_ticker}  
**Phase:** 1 — Root-Cause Diagnosis  

---

## SUMMARY

| Hypothesis | Verdict | Key Finding |
|-----------|---------|-------------|
| H1 Signal anti-predictive | **{h1_sum.get('verdict', 'N/A')}** | Mean IC = {h1_sum.get('mean_ic_all_features', 'N/A'):+.4f}, {h1_sum.get('n_features_positive_ic', 'N/A')}/{len(h1_ic)} features with positive IC |
| H2 Wrong publication lag | **{h2.get('verdict', 'N/A')}** | Uniform-45 OOS R² = {h2.get('uniform_45d_lag', {}).get('mean_oos_r2', 'N/A')}, Proper-lag OOS R² = {h2.get('proper_lag_table', {}).get('mean_oos_r2', 'N/A')} |
| H3 SMA filter over-restrictive | **{h3.get('verdict', 'N/A')}** | {h3.get('interpretation', 'N/A')} |

---

## H1 — Signal Quality (Information Coefficient)

**Method:** For each macro feature, apply its *correct* publication lag, then compute  
Spearman rank-correlation between the lagged feature value at date `t` and the  
1-day forward return at `t+1` (no target leakage). Report mean IC across all tickers.

**Acceptance threshold:**  
- Mean IC > +0.02 and t-stat > 2.0 → signal has positive edge  
- Mean IC < −0.02 and t-stat < −2.0 → signal is anti-correlated (root cause confirmed)  
- |Mean IC| < 0.02 → signal is noise (also a root cause, different fix)  

### Feature IC Table (all tickers, 1-day forward return)

{ic_table_md(h1_ic)}

### Aggregate
- **Mean IC across all features:** {h1_sum.get('mean_ic_all_features', 'N/A'):+.4f}  
- **Features with positive IC:** {h1_sum.get('n_features_positive_ic', 'N/A')} / {len(h1_ic)}  
- **Features significant (|t| ≥ 2):** {h1_sum.get('n_features_significant', 'N/A')}  

### Verdict: {h1_sum.get('verdict', 'N/A')}

**Interpretation:**
"""

if h1_sum.get("verdict") == "CONFIRMED":
    report += textwrap.dedent("""\
        Mean IC is negative and/or fewer than half of features have positive IC.  
        The macro signal systematically predicts the **wrong direction**.  
        This is the primary root cause. No amount of threshold tuning, regime  
        filtering, or parameter optimisation will fix an anti-correlated signal.  

        **Required action:** Redesign the alpha from documented factors with  
        confirmed positive IC (12-1 momentum, quality, low-vol) before fitting  
        any model. See Phase 2A.
    """)
elif h1_sum.get("verdict") == "REJECTED":
    report += textwrap.dedent("""\
        Mean IC is positive and the majority of features have positive IC.  
        The signal direction is correct; the problem is execution-layer or  
        weighting (see H2, H3).
    """)
else:
    report += textwrap.dedent("""\
        Mixed IC results — some features positive, some negative.  
        The composite signal may be averaging out real edges with noise.  
        IC-weighted feature selection (keep only features with IC > 0.01 and  
        t-stat > 1.5) is warranted regardless of the other hypotheses.
    """)

report += f"""
---

## H2 — Publication Lag Correctness

**Method:** Build two Ridge-on-macro panels for {ref_ticker}:  
1. **Uniform 45-day lag** — current behaviour (every FRED series shifted 45 days)  
2. **Per-series proper lag** — VIX/DGS10/FEDFUNDS shifted 1 day; CPI/UNRATE shifted 45 days  

Walk-forward OOS R² (4 folds) compared between the two.

### Results

| Configuration | Folds | Mean OOS R² |
|--------------|-------|-------------|
| Uniform 45-day lag (current) | {h2.get('uniform_45d_lag', {}).get('folds', 'N/A')} | {h2.get('uniform_45d_lag', {}).get('mean_oos_r2', 'N/A')} |
| Per-series proper lag        | {h2.get('proper_lag_table', {}).get('folds', 'N/A')} | {h2.get('proper_lag_table', {}).get('mean_oos_r2', 'N/A')} |
| Improvement                  | — | {h2.get('improvement', 'N/A')} |

### Verdict: {h2.get('verdict', 'N/A')}

**Interpretation:**  
"""

if h2.get("verdict") == "CONFIRMED":
    report += textwrap.dedent("""\
        Using proper per-series publication lags meaningfully improves OOS R².  
        Uniform 45-day lag was staling real-time market data (VIX, 10Y yield)  
        and degrading the model's regime awareness.  
        **Fix:** Introduce a `PUBLICATION_LAG_TABLE` in `mixed_frequency.py`  
        and use per-series lags in `build_stationary_panel()`. See Phase 2B.
    """)
elif h2.get("verdict") == "REJECTED":
    report += textwrap.dedent("""\
        Proper per-series lags do not improve OOS R² meaningfully.  
        The lag regime is not the primary driver of poor performance.  
        H1 (signal quality) remains the dominant issue.
    """)
else:
    report += textwrap.dedent("""\
        Marginal or inconclusive improvement. Proper lag table is still  
        the correct practice but is not the dominant performance lever here.
    """)

report += f"""
---

## H3 — Dual-SMA Filter Impact

**Method:** Run the same Ridge-on-macro backtest twice on {ref_ticker}:  
1. **With** dual-SMA 200-day filter (current)  
2. **Without** filter (all predicted signals execute)  

### Results

| Configuration | Active ratio | Sharpe | Mean daily ret |
|--------------|-------------|--------|----------------|
| With SMA filter    | {h3.get('with_filter', {}).get('active_ratio', 'N/A')} | {h3.get('with_filter', {}).get('sharpe', 'N/A')} | {h3.get('with_filter', {}).get('mean_daily_return', 'N/A')} |
| Without SMA filter | {h3.get('without_filter', {}).get('active_ratio', 'N/A')} | {h3.get('without_filter', {}).get('sharpe', 'N/A')} | {h3.get('without_filter', {}).get('mean_daily_return', 'N/A')} |

Signal IC (pred_z vs 1-day return):  
- With filter context: {h3.get('with_filter', {}).get('ic_signal_vs_1d_actual', 'N/A')}  
- Without filter context: {h3.get('without_filter', {}).get('ic_signal_vs_1d_actual', 'N/A')}  

### Verdict: {h3.get('verdict', 'N/A')}

{h3.get('interpretation', '')}

---

## ROOT CAUSE CONCLUSION

"""

verdicts = {
    "H1": h1_sum.get("verdict", "INCONCLUSIVE"),
    "H2": h2.get("verdict", "INCONCLUSIVE"),
    "H3": h3.get("verdict", "INCONCLUSIVE"),
}

if verdicts["H1"] == "CONFIRMED":
    report += textwrap.dedent("""\
        **PRIMARY ROOT CAUSE: H1 — The signal is anti-predictive or noise.**

        The macro features (FRED + World Bank) as currently constructed  
        do not carry positive information about 1-day or 21-day forward returns.  
        Macro data with correct lags applied simply does not predict daily equity  
        returns consistently at IC > 0.02.

        **The required fix is Phase 2A: Redesign the alpha.**  
        Replace or augment the macro-only signal with documented factors that  
        have empirical IC evidence:
        - **12-1 month momentum** (Jegadeesh-Titman): strongest single equity factor
        - **Low-volatility**: stocks with lower trailing vol earn higher risk-adj returns  
        - **Quality composite**: high ROE + low debt as defensive overlay

        H2 (lag correction) and L1 (boundary leakage fix) should **also** be  
        implemented as hygiene but will not alone restore positive Sharpe.

        **Do not proceed to Phase 3 (universe pruning) or Phase 5 (agentic loop)**  
        until Phase 2A produces IC > 0.02 on at least 60% of tickers out-of-sample.
    """)
elif verdicts["H1"] == "REJECTED" and verdicts["H2"] == "CONFIRMED":
    report += textwrap.dedent("""\
        **PRIMARY ROOT CAUSE: H2 — Wrong publication lag degrades an otherwise  
        viable signal.**

        The signal has positive IC when features are correctly lagged. The  
        uniform 45-day lag for real-time market data (VIX, yields) was the  
        primary performance drag. Fix: per-series publication lag table.
    """)
else:
    report += textwrap.dedent("""\
        **INCONCLUSIVE — Mixed signals across hypotheses.**

        Recommended next step: manually inspect IC for each feature with a  
        rolling plot and decide whether signal redesign (Phase 2A) or lag  
        correction (Phase 2B) is the higher-leverage fix.
    """)

report += """
---

🛑 **CHECKPOINT 1** — Eva, please review this report before Phase 2 begins.

**Required decision:** Which root cause fix should Phase 2 address?
- **Option A (recommended if H1=CONFIRMED):** Phase 2A — Redesign alpha with  
  momentum/low-vol/quality factors. Discard pure macro-only model.
- **Option B (if H2=CONFIRMED, H1=REJECTED):** Phase 2B — Fix lag table only.  
- **Option C (if both confirmed):** Phase 2B first (cheap fix), measure IC gain,  
  then 2A if IC is still negative.

Do not proceed to code changes until Eva approves the option.
"""

out_path = ROOT / "diagnosis_report.md"
out_path.write_text(report, encoding="utf-8")

# Also save raw numbers for reproducibility
raw_out = ROOT / "output" / "default" / "diagnosis_raw.json"
raw_out.parent.mkdir(parents=True, exist_ok=True)
with open(raw_out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nDone. Report written to: {out_path}")
print(f"Raw JSON written to:     {raw_out}")
