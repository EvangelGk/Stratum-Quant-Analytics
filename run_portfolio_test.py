"""Test dynamic portfolio_backtest + print per-ticker CV IC scores for diagnosis."""
import sys, warnings, types, importlib.util
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
for _p in [str(ROOT/"src"),
           str(ROOT/"src"/"Medallion"/"gold"/"AnalysisSuite"),
           str(ROOT/"src"/"exceptions")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

def stub(n):
    m = types.ModuleType(n); sys.modules[n] = m; return m

for n in ["diskcache","dotenv","pandera","pandera.errors","secret_store",
          "logger","logger.Catalog","logger.Messages",
          "logger.Messages.DirectionsMess","logger.Messages.MainMess"]:
    stub(n)
stub("logger.Catalog").catalog = lambda *a, **kw: None

def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"src"/rel)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

em = stub("exceptions.MedallionExceptions")
class _E(Exception): pass
em.AnalysisError = _E
em.DataValidationError = _E

load("Medallion/gold/AnalysisSuite/mixed_frequency.py",
     "Medallion.gold.AnalysisSuite.mixed_frequency")
bt = load("Medallion/gold/AnalysisSuite/backtest.py",
          "Medallion.gold.AnalysisSuite.backtest")

import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

gold = next(Path("data").glob("**/master_table.parquet"))
df   = pd.read_parquet(gold)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ──────────────────────────────────────────────────────────────────
# DIAGNOSTIC: per-ticker CV IC scores (same logic as select_active_universe)
# ──────────────────────────────────────────────────────────────────
print("=== Per-Ticker CV IC Scores (diagnostic) ===")
_FWD = 21
train_end = pd.Timestamp("2020-01-01")
_df = df.copy()

_parts = []
for _tk in _df["ticker"].unique():
    _p = _df[_df["ticker"] == _tk].copy().sort_values("date")
    lr = pd.to_numeric(_p["log_return"], errors="coerce")
    _p["_t_ret5"]   = lr.rolling(5,   min_periods=3).sum()
    _p["_t_ret20"]  = lr.rolling(20,  min_periods=10).sum()
    _p["_t_ret60"]  = lr.rolling(60,  min_periods=30).sum()
    _p["_t_mom12"]  = lr.shift(21).rolling(231, min_periods=80).sum()
    v20 = lr.rolling(20, min_periods=10).std()
    v60 = lr.rolling(60, min_periods=30).std().replace(0.0, np.nan)
    _p["_t_volrat"] = (v20 / v60 - 1.0).fillna(0.0)
    _parts.append(_p)
_df = pd.concat(_parts, ignore_index=True)

_TECH = {"_t_ret5", "_t_ret20", "_t_ret60", "_t_mom12", "_t_volrat"}
skip = {"date", "ticker", "log_return", "adj_close", "close", "open",
        "high", "low", "volume"}
macro_cols = [c for c in _df.columns
              if c not in skip and c not in _TECH
              and pd.api.types.is_numeric_dtype(_df[c])
              and not c.startswith("__")]
all_feat = macro_cols + list(_TECH)

tw = _df[_df["date"] < train_end].copy()
tw["_fwd"] = (tw.groupby("ticker")["log_return"]
               .transform(lambda s: s.shift(-1).rolling(_FWD).sum().shift(-(_FWD - 1))))

tscv = TimeSeriesSplit(n_splits=5)
all_tickers = sorted(df["ticker"].dropna().unique().tolist())

scores = {}
for tk in all_tickers:
    tdf = (tw[tw["ticker"] == tk].dropna(subset=["_fwd"])
           .sort_values("date").reset_index(drop=True))
    if len(tdf) < 100:
        scores[tk] = float("nan")
        continue
    feats = [f for f in all_feat if f in tdf.columns]
    Xmat = tdf[feats].fillna(0.0).to_numpy(float)
    yvec = tdf["_fwd"].to_numpy(float)
    fold_ics = []
    for tr_idx, te_idx in tscv.split(Xmat):
        if len(tr_idx) < 40 or len(te_idx) < 10:
            continue
        X_tr, X_te = Xmat[tr_idx], Xmat[te_idx]
        y_tr, y_te = yvec[tr_idx], yvec[te_idx]
        if not (np.isfinite(y_tr).all() and np.isfinite(y_te).all()):
            continue
        pos_idx = [fi for fi, f in enumerate(feats)
                   if np.isfinite(pearsonr(X_tr[:, fi], y_tr)[0])
                   and pearsonr(X_tr[:, fi], y_tr)[0] > 0]
        sel = pos_idx if pos_idx else list(range(len(feats)))
        try:
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(X_tr[:, sel])
            Xte_s = sc.transform(X_te[:, sel])
            mdl = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=min(3, max(2, len(tr_idx)//30)))
            mdl.fit(Xtr_s, y_tr)
            ic = pearsonr(mdl.predict(Xte_s), y_te)[0]
            if np.isfinite(ic):
                fold_ics.append(float(ic))
        except Exception:
            pass
    scores[tk] = float(np.mean(fold_ics)) if fold_ics else float("nan")

for tk, sc in sorted(scores.items(), key=lambda x: (not np.isfinite(x[1]), -x[1] if np.isfinite(x[1]) else 0)):
    phase3_keep = tk in ["AAPL","ABBV","AMGN","APD","CAT","CVX","HON","JNJ",
                          "KO","LLY","MS","MSFT","NVDA","ORCL","PG","TSLA","XOM"]
    tag = "KEEP" if phase3_keep else "drop"
    print(f"  {tk:6s} IC={sc:+.4f} [{tag}]")

print()
print("Running dynamic portfolio_backtest (dynamic_universe=True, min_n=8, max_n=15)...")
r = bt.portfolio_backtest(df, dynamic_universe=True, min_n=8, max_n=15, mode="both")

port = r.get("portfolio", {}) or {}
print()
print("=== Dynamic Universe Selected ===")
print("  Tickers attempted:", r["tickers_attempted"])
print("  Tickers succeeded:", r["tickers_succeeded"])
print()
print("=== Portfolio Metrics (OOS 2020+) ===")
print("  Sharpe       :", port.get("sharpe_ratio"))
print("  Calmar       :", port.get("calmar_ratio"))
print("  Profit Factor:", port.get("profit_factor"))
print("  Info Ratio   :", port.get("information_ratio"))
print("  Max DD       :", port.get("maximum_drawdown"))
print("  Ann. Return  :", port.get("annualized_return"))
print("  Win Prob     :", port.get("win_probability"))
