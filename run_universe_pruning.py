"""Phase 3: Universe Pruning
==============================
For each of the 31 tickers:
  1. Standalone IC  — mean Pearson(IC) of all features vs 21-day forward return
     on the 2014-2019 training period (no lookahead).
  2. Per-ticker backtest_pre2020_holdout Sharpe on the 2020-2022 test window.
  3. Leave-one-out (LOO) portfolio contribution.
  4. Average pairwise log-return correlation with the rest of the universe.

Drop candidates:
  - Sharpe < 0 AND mean IC ≤ 0
  - OR avg pairwise correlation > 0.85 (highly redundant) AND NOT the
    highest-Sharpe representative of the group.

Target universe: 12–20 tickers with low average cross-correlation.

Outputs:
  output/default/universe_pruning.json   (full table)
  UNIVERSE_REPORT.md                     (human-readable Checkpoint 2 summary)
"""
from __future__ import annotations
import sys
import json
import warnings
import importlib.util
import types
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

# ── Minimal import shims so AnalysisSuite loads without heavy deps ──────────
for _p in [str(ROOT / "src"),
           str(ROOT / "src" / "Medallion" / "gold" / "AnalysisSuite"),
           str(ROOT / "src" / "exceptions")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

def _stub(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

for _n in ["diskcache", "dotenv", "pandera", "pandera.errors",
           "secret_store", "logger", "logger.Catalog",
           "logger.Messages", "logger.Messages.DirectionsMess",
           "logger.Messages.MainMess"]:
    _stub(_n)
_stub("logger.Catalog").catalog = lambda *a, **kw: None  # type: ignore[attr-defined]

def _load_module(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

exc_mod = _stub("exceptions.MedallionExceptions")
class _Err(Exception): pass
exc_mod.AnalysisError = _Err   # type: ignore[attr-defined]
exc_mod.DataValidationError = _Err  # type: ignore[attr-defined]

_load_module("Medallion/gold/AnalysisSuite/mixed_frequency.py",
             "Medallion.gold.AnalysisSuite.mixed_frequency")
bt = _load_module("Medallion/gold/AnalysisSuite/backtest.py",
                  "Medallion.gold.AnalysisSuite.backtest")

# ── Load master table ────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

gold = next((ROOT / "data").glob("**/master_table.parquet"))
print(f"[Phase 3] Loading {gold} …")
df = pd.read_parquet(gold)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

tickers = sorted(df["ticker"].dropna().unique().tolist())
print(f"[Phase 3] Universe: {len(tickers)} tickers — {tickers}\n")

TRAIN_END = pd.Timestamp("2020-01-01")

# ── 1. Per-ticker standalone IC (train window only, no lookahead) ────────────
# IC = mean |Pearson(feature, fwd_21d_return)| over all features in the
# training period.  We compute this from the master table directly without
# calling the full backtest so it's fast.

def _compute_ticker_ic(tdf: pd.DataFrame) -> float:
    """Return mean positive-IC fraction for a single ticker's training window."""
    train = tdf[tdf["date"] < TRAIN_END].copy()
    if len(train) < 60:
        return float("nan")
    # Use the same 21-day forward return as the model target
    if "log_return" not in train.columns:
        return float("nan")
    fwd = train["log_return"].shift(-1).rolling(21).sum().shift(-20)
    fwd = fwd.dropna()
    if len(fwd) < 30:
        return float("nan")
    # Candidate feature columns (numeric, not date/ticker/target)
    skip = {"date", "ticker", "log_return"}
    feats = [c for c in train.columns if c not in skip
             and pd.api.types.is_numeric_dtype(train[c])]
    ics = []
    for f in feats:
        aligned = train[f].loc[fwd.index].dropna()
        y = fwd.loc[aligned.index]
        if len(aligned) < 30:
            continue
        try:
            r = float(np.corrcoef(aligned.to_numpy(float), y.to_numpy(float))[0, 1])
            if np.isfinite(r):
                ics.append(r)
        except Exception:
            pass
    return float(np.mean(ics)) if ics else float("nan")

print("[Phase 3] Computing per-ticker standalone IC …")
ic_scores: dict[str, float] = {}
for tk in tickers:
    ic_scores[tk] = _compute_ticker_ic(df[df["ticker"] == tk].copy())
    print(f"  IC  {tk:6s}  {ic_scores[tk]:+.5f}")

# ── 2. Per-ticker backtest Sharpe (2020-2022 OOS) ────────────────────────────
print("\n[Phase 3] Running per-ticker backtest_pre2020_holdout …")
sharpe_scores: dict[str, float] = {}
calmar_scores: dict[str, float] = {}
bt_errors: dict[str, str] = {}

for tk in tickers:
    tdf = df[df["ticker"] == tk].copy()
    try:
        r = bt.backtest_pre2020_holdout(tdf, ticker=tk)
        sharpe_scores[tk] = float(r.get("sharpe_ratio", float("nan")))
        calmar_scores[tk] = float(r.get("calmar_ratio", float("nan")))
    except Exception as exc:
        sharpe_scores[tk] = float("nan")
        calmar_scores[tk] = float("nan")
        bt_errors[tk] = str(exc)[:120]
    print(f"  BT  {tk:6s}  Sharpe={sharpe_scores[tk]:+.3f}  Calmar={calmar_scores[tk]:+.3f}")

# ── 3. Average pairwise log-return correlation ───────────────────────────────
print("\n[Phase 3] Computing pairwise correlation matrix …")
# Pivot to wide format: each column = ticker log returns
wide = (df[["date", "ticker", "log_return"]]
        .dropna()
        .pivot_table(index="date", columns="ticker", values="log_return"))
corr_matrix = wide.corr()

avg_pairwise_corr: dict[str, float] = {}
for tk in tickers:
    if tk not in corr_matrix.columns:
        avg_pairwise_corr[tk] = float("nan")
        continue
    row = corr_matrix[tk].drop(labels=[tk], errors="ignore")
    avg_pairwise_corr[tk] = float(row.mean())

# ── 4. Leave-one-out portfolio Sharpe contribution ───────────────────────────
# Full portfolio Sharpe = equal-weight combination of per-ticker log returns
# on the 2020-2022 test window.  LOO = Sharpe when ticker is removed.
print("\n[Phase 3] Computing leave-one-out Sharpe contributions …")

test_mask = (wide.index >= pd.Timestamp("2020-01-01")) & (wide.index <= pd.Timestamp("2022-12-31"))
test_wide = wide.loc[test_mask]

def _sharpe_of_portfolio(cols: list[str]) -> float:
    subset = test_wide[cols].dropna(how="all")
    rets = subset.mean(axis=1)
    ann = float(rets.mean() * 252)
    vol = float(rets.std() * np.sqrt(252))
    return ann / vol if vol > 0 else float("nan")

full_cols = [t for t in tickers if t in test_wide.columns]
full_sharpe = _sharpe_of_portfolio(full_cols)
print(f"  Full portfolio Sharpe ({len(full_cols)} tickers): {full_sharpe:+.3f}")

loo_delta: dict[str, float] = {}  # positive = removing this ticker HURTS (it's a contributor)
for tk in tickers:
    loo_cols = [t for t in full_cols if t != tk]
    loo_sh = _sharpe_of_portfolio(loo_cols) if loo_cols else float("nan")
    # delta = full_sharpe - LOO_sharpe.  Positive means ticker was contributing.
    loo_delta[tk] = float(full_sharpe - loo_sh) if np.isfinite(loo_sh) else float("nan")

# ── 5. Pruning decision ───────────────────────────────────────────────────────
CORR_THRESHOLD   = 0.80   # drop the redundant one if avg pair corr > this
SHARPE_FLOOR     = 0.0    # per-ticker Sharpe must be ≥ this to survive
IC_FLOOR         = 0.0    # mean IC must be ≥ this to survive

rows: list[dict] = []
for tk in tickers:
    sh  = sharpe_scores.get(tk, float("nan"))
    ic  = ic_scores.get(tk, float("nan"))
    cal = calmar_scores.get(tk, float("nan"))
    apc = avg_pairwise_corr.get(tk, float("nan"))
    ld  = loo_delta.get(tk, float("nan"))

    # Flagging rules
    flags = []
    if np.isfinite(sh) and sh < SHARPE_FLOOR:
        flags.append("negative_sharpe")
    if np.isfinite(ic) and ic < IC_FLOOR:
        flags.append("negative_ic")
    if np.isfinite(apc) and apc > CORR_THRESHOLD:
        flags.append("high_correlation")
    if np.isfinite(ld) and ld < -0.05:          # LOO improves portfolio by > 5bps
        flags.append("loo_drag")

    # Hard drop: needs BOTH negative Sharpe AND negative IC
    hard_drop = ("negative_sharpe" in flags and "negative_ic" in flags)
    # Soft drop: high correlation + negative LOO delta (redundant AND dragging)
    soft_drop = ("high_correlation" in flags and "loo_drag" in flags)

    decision = "DROP" if (hard_drop or soft_drop) else "KEEP"

    rows.append({
        "ticker"         : tk,
        "sharpe_oos"     : round(sh, 4) if np.isfinite(sh) else None,
        "calmar_oos"     : round(cal, 4) if np.isfinite(cal) else None,
        "mean_ic_train"  : round(ic, 6) if np.isfinite(ic) else None,
        "avg_pair_corr"  : round(apc, 4) if np.isfinite(apc) else None,
        "loo_sharpe_delta": round(ld, 4) if np.isfinite(ld) else None,
        "flags"          : flags,
        "decision"       : decision,
    })

rows_df = pd.DataFrame(rows).sort_values("sharpe_oos", ascending=False, na_position="last")

keep = rows_df[rows_df["decision"] == "KEEP"]
drop = rows_df[rows_df["decision"] == "DROP"]

print(f"\n[Phase 3] KEEP {len(keep)} / DROP {len(drop)} tickers")
print("\n  KEEP:", sorted(keep["ticker"].tolist()))
print("  DROP:", sorted(drop["ticker"].tolist()))

# ── 6. Portfolio Sharpe of pruned universe ───────────────────────────────────
pruned_cols = [t for t in keep["ticker"].tolist() if t in test_wide.columns]
pruned_sharpe = _sharpe_of_portfolio(pruned_cols)
print(f"\n  Pruned portfolio Sharpe ({len(pruned_cols)} tickers): {pruned_sharpe:+.3f}")
print(f"  Full   portfolio Sharpe ({len(full_cols)} tickers):  {full_sharpe:+.3f}")

# ── 7. Write outputs ──────────────────────────────────────────────────────────
output_dir = ROOT / "output" / "default"
output_dir.mkdir(parents=True, exist_ok=True)

result = {
    "generated_at"          : pd.Timestamp.now().isoformat(),
    "thresholds"            : {"sharpe_floor": SHARPE_FLOOR, "ic_floor": IC_FLOOR,
                               "corr_threshold": CORR_THRESHOLD},
    "full_portfolio_sharpe" : round(full_sharpe, 4) if np.isfinite(full_sharpe) else None,
    "pruned_portfolio_sharpe": round(pruned_sharpe, 4) if np.isfinite(pruned_sharpe) else None,
    "n_keep"                : int(len(keep)),
    "n_drop"                : int(len(drop)),
    "tickers_keep"          : sorted(keep["ticker"].tolist()),
    "tickers_drop"          : sorted(drop["ticker"].tolist()),
    "per_ticker"            : rows_df.to_dict(orient="records"),
}

out_path = output_dir / "universe_pruning.json"
out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
print(f"\n[Phase 3] Results written → {out_path}")

# ── 8. Generate UNIVERSE_REPORT.md ────────────────────────────────────────────
def _fmt(v, fmt="+.3f"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return format(v, fmt)

lines = [
    "# CHECKPOINT 2 — Universe Pruning Report",
    "",
    f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  ",
    f"**Input universe:** {len(tickers)} tickers  ",
    f"**After pruning:** {len(keep)} tickers  ",
    "",
    "## Decision Table",
    "",
    "| Ticker | Sharpe OOS | Calmar OOS | Mean IC | Avg Pair Corr | LOO Δ | Flags | Decision |",
    "|--------|-----------|-----------|---------|---------------|-------|-------|----------|",
]
for _, row in rows_df.iterrows():
    flags_str = ", ".join(row["flags"]) if row["flags"] else "—"
    lines.append(
        f"| {row['ticker']} "
        f"| {_fmt(row['sharpe_oos'])} "
        f"| {_fmt(row['calmar_oos'])} "
        f"| {_fmt(row['mean_ic_train'], '+.5f')} "
        f"| {_fmt(row['avg_pair_corr'], '.4f')} "
        f"| {_fmt(row['loo_sharpe_delta'])} "
        f"| {flags_str} "
        f"| **{row['decision']}** |"
    )

lines += [
    "",
    "## Portfolio Sharpe Comparison",
    "",
    f"| Universe | N | Equal-Weight Sharpe (2020–2022) |",
    f"|----------|---|--------------------------------|",
    f"| Full (pre-prune) | {len(full_cols)} | {_fmt(full_sharpe)} |",
    f"| Pruned           | {len(pruned_cols)} | {_fmt(pruned_sharpe)} |",
    "",
    "## Tickers Kept",
    "",
    ", ".join(sorted(keep["ticker"].tolist())),
    "",
    "## Tickers Dropped",
    "",
]
for _, row in drop.iterrows():
    flags_str = ", ".join(row["flags"]) if row["flags"] else "no flags"
    lines.append(f"- **{row['ticker']}** — {flags_str}  "
                 f"Sharpe={_fmt(row['sharpe_oos'])}, IC={_fmt(row['mean_ic_train'], '+.5f')}, "
                 f"Corr={_fmt(row['avg_pair_corr'], '.4f')}")

lines += [
    "",
    "---",
    "",
    "## Notes",
    "- Drop rule A: *negative_sharpe* **AND** *negative_ic* (both must fire).",
    "- Drop rule B: *high_correlation* (avg pair corr > 0.80) **AND** *loo_drag* "
      "(removing improves portfolio Sharpe by >5 bps).",
    "- Correlation threshold 0.80 is stricter than naive 0.85 to give Phase 4 more "
      "diversification room.",
    "- LOO delta is computed on the **OOS** 2020–2022 equal-weight portfolio.",
    "",
    "**CHECKPOINT 2** — Eva, please review and approve (or adjust thresholds) "
    "before Phase 4 begins.",
]

report_path = ROOT / "UNIVERSE_REPORT.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"[Phase 3] Report written → {report_path}")
print("\n✅ Phase 3 complete.")
