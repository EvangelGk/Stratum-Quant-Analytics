"""Phase 2A spot-check: run backtest_pre2020_holdout on AAPL using the
patched IC-gate + L1-fix code.  Imports only the AnalysisSuite module tree
(no Fetchers/Factory chain) to avoid optional dependency errors.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Add src sub-paths individually so the AnalysisSuite modules resolve without
# triggering the full Fetchers / diskcache import chain.
for p in [str(ROOT / "src"),
          str(ROOT / "src" / "Medallion" / "gold" / "AnalysisSuite"),
          str(ROOT / "src" / "exceptions")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Stub out the heavy imports so the backtest module loads standalone.
import types, importlib

def _make_stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

# Only stub what we can't install quickly.
for _n in ["diskcache", "dotenv", "pandera", "pandera.errors",
           "secret_store", "logger", "logger.Catalog",
           "logger.Messages", "logger.Messages.DirectionsMess",
           "logger.Messages.MainMess"]:
    if _n not in sys.modules:
        _make_stub(_n)

# Stub catalog (referenced by GoldLayer but not needed for backtest standalone)
sys.modules["logger.Catalog"].catalog = lambda *a, **kw: None

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

# Direct import — bypasses the Medallion __init__ import chain
import importlib.util

def _load_module(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "src" / rel_path
    )
    mod = importlib.util.load_from_spec = spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load exception stubs
exc_mod = _make_stub("exceptions.MedallionExceptions")
class _Err(Exception): pass
exc_mod.AnalysisError = _Err
exc_mod.DataValidationError = _Err

# Load mixed_frequency then backtest
mf = _load_module("Medallion/gold/AnalysisSuite/mixed_frequency.py",
                   "Medallion.gold.AnalysisSuite.mixed_frequency")
bt = _load_module("Medallion/gold/AnalysisSuite/backtest.py",
                   "Medallion.gold.AnalysisSuite.backtest")

# Load master table
gold = next((ROOT / "data").glob("**/master_table.parquet"))
print(f"Loading {gold} …")
df = pd.read_parquet(gold)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

aapl = df[df["ticker"] == "AAPL"].copy() if "AAPL" in df["ticker"].values else df.head(3000).copy()

print(f"AAPL rows: {len(aapl)}, date range: {aapl['date'].min().date()} → {aapl['date'].max().date()}")

result = bt.backtest_pre2020_holdout(aapl, ticker="AAPL")

print("\n── Phase 2A Results ──────────────────────────────────────")
print(f"  Sharpe Ratio       : {result['sharpe_ratio']}")
print(f"  Calmar Ratio       : {result['calmar_ratio']}")
print(f"  Profit Factor      : {result['profit_factor']}")
print(f"  Expectancy/trade   : {result['expectancy_per_trade']}")
print(f"  Information Ratio  : {result['information_ratio']}")
print(f"  Max Drawdown       : {result['maximum_drawdown']}")
print(f"  Ann. Return        : {result['annualized_return']}")
print(f"  Split mode         : {result['window']['split_mode']}")
print(f"  Train rows         : {result['train_rows']}")
print(f"  Test rows          : {result['test_rows']}")
print(f"  Features used      : {result['features']}")
print(f"  Active ratio       : {result['strategy_parameters']['active_ratio']}")
wf = result.get("walk_forward_validation", {})
print(f"\n── Walk-Forward ({wf.get('windows_completed', 0)} folds) ──")
print(f"  Avg Sharpe   : {wf.get('avg_sharpe_ratio')}")
print(f"  Avg Calmar   : {wf.get('avg_calmar_ratio')}")
print(f"  Worst DD     : {wf.get('worst_max_drawdown')}")
print(f"  +Pearson ratio: {wf.get('positive_pearson_ratio')}")
print(f"  p<0.05 ratio : {wf.get('pvalue_lt_0_05_ratio')}")
