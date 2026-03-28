"""Standalone convenience API for AnalysisSuite functions.

**Role:** Thin wrappers that let external callers (notebooks, scripts, unit
tests, UI widgets) invoke the core analysis functions directly without going
through the full GoldLayer orchestration.

**Important limitations vs GoldLayer:**
- Governance gate is NOT evaluated — results are returned unconditionally.
- Audit trail, survivorship checks, and publication-lag compliance are NOT run.
- For production pipeline outputs use ``GoldLayer.run_all_analyses()`` or
  ``GoldLayer.run_all_analyses_parallel()``, which are the authoritative paths.

**Contract note:** This file must be kept in sync with the function signatures
of the underlying AnalysisSuite modules.  When adding parameters to
``stress_test``, ``monte_carlo``, ``correl_mtrx`` or ``backtest``, update the
wrappers here as well so there is no interface divergence over time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.Medallion.gold.AnalysisSuite.backtest import backtest_pre2020_holdout
from src.Medallion.gold.AnalysisSuite.correl_mtrx import correl_mtrx
from src.Medallion.gold.AnalysisSuite.feature_decay import feature_decay_analysis
from src.Medallion.gold.AnalysisSuite.monte_carlo import monte_carlo
from src.Medallion.gold.AnalysisSuite.stress_test import (
    PRESET_SCENARIOS,
    resolve_stress_scenario,
    stress_test,
)


def run_stress_pack(
    df: pd.DataFrame,
    ticker: str,
    target: str = "log_return",
    scenario_name: str = "geopolitical_conflict",
    custom_shocks: Optional[Dict[str, float]] = None,
    allow_non_governed_run: bool = False,
) -> Dict[str, Any]:
    """High-level stress pack entrypoint used by UI/reporting layers."""
    if not allow_non_governed_run:
        raise ValueError("run_stress_pack requires allow_non_governed_run=True because it bypasses GoldLayer governance gate and audit path.")
    scenario = resolve_stress_scenario(scenario_name=scenario_name, shock_map=custom_shocks)
    stress = stress_test(
        df=df,
        shock_map=dict(scenario.get("factor_shocks", {})),
        target=target,
        scenario_name=scenario_name,
        macro_lag_days=45,
    )
    corr = correl_mtrx(
        df,
        stress_mode=True,
        stress_strength=float(scenario.get("correlation_breakdown_strength", 0.30)),
        scenario_name=scenario_name,
    )
    mc = monte_carlo(
        df=df,
        ticker=ticker,
        macro_scenario=scenario_name,
        scenario_bias=dict(scenario.get("mc_bias", {})),
    )
    return {
        "governance_context": "bypassed_non_governed_entrypoint",
        "scenario": scenario,
        "stress_test": stress,
        "correlation_matrix": corr,
        "monte_carlo": mc,
    }


def run_pre2020_backtest(
    df: pd.DataFrame,
    target: str = "log_return",
    features: Optional[list[str]] = None,
    ticker: Optional[str] = None,
    allow_non_governed_run: bool = False,
) -> Dict[str, Any]:
    if not allow_non_governed_run:
        raise ValueError("run_pre2020_backtest requires allow_non_governed_run=True because it bypasses GoldLayer governance gate and audit path.")
    return backtest_pre2020_holdout(
        df=df,
        target=target,
        features=features,
        ticker=ticker,
    )


__all__ = [
    "PRESET_SCENARIOS",
    "resolve_stress_scenario",
    "run_stress_pack",
    "run_pre2020_backtest",
    "feature_decay_analysis",
]
