from typing import Any, Dict, Optional

import pandas as pd
import statsmodels.api as sm

from exceptions.MedallionExceptions import AnalysisError, DataValidationError
from .mixed_frequency import prepare_supervised_frame


PRESET_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "geopolitical_conflict": {
        "description": "Energy shock + CPI jump + equity drawdown (war supply shock)",
        "factor_shocks": {
            "energy_index": 0.25,
            "inflation": 0.02,
            "log_return": -0.15,
        },
        "correlation_breakdown_strength": 0.35,
        "mc_bias": {
            "drift_shift": -0.0010,
            "vol_multiplier": 1.45,
            "jump_shock": -0.12,
        },
    },
    "monetary_tightening": {
        "description": "Rates +150 bps and GDP growth -1% under higher-for-longer policy",
        "factor_shocks": {
            "fed_funds_rate": 0.015,
            "gdp_growth": -0.01,
        },
        "correlation_breakdown_strength": 0.25,
        "mc_bias": {
            "drift_shift": -0.0006,
            "vol_multiplier": 1.20,
            "jump_shock": -0.05,
        },
    },
    "tech_correction": {
        "description": (
            "Technology sector drawdown with macro factors near unconditional mean. "
            "The log_return shock targets the specific ticker under analysis; "
            "cross-sector isolation requires external ticker metadata (sector_hint='Technology')."
        ),
        "sector_hint": "Technology",
        "factor_shocks": {
            "log_return": -0.30,
        },
        "correlation_breakdown_strength": 0.40,
        "mc_bias": {
            "drift_shift": -0.0012,
            "vol_multiplier": 1.35,
            "jump_shock": -0.20,
        },
    },
    "stagflation": {
        "description": "High inflation with negative growth and flat equities",
        "factor_shocks": {
            "inflation": 0.05,
            "gdp_growth": -0.01,
            "log_return": 0.0,
        },
        "correlation_breakdown_strength": 0.30,
        "mc_bias": {
            "drift_shift": -0.0008,
            "vol_multiplier": 1.30,
            "jump_shock": -0.08,
        },
    },
    "liquidity_freeze": {
        "description": "Credit/liquidity event with cross-asset contagion",
        "factor_shocks": {
            "us10y_treasury_yield": 0.01,
            "consumer_sentiment": -0.10,
            "log_return": -0.18,
        },
        "correlation_breakdown_strength": 0.45,
        "mc_bias": {
            "drift_shift": -0.0014,
            "vol_multiplier": 1.55,
            "jump_shock": -0.15,
        },
    },
    "commodity_super_spike": {
        "description": "Commodity shock with inflation and margin compression",
        "factor_shocks": {
            "energy_index": 0.35,
            "inflation": 0.03,
            "log_return": -0.10,
        },
        "correlation_breakdown_strength": 0.30,
        "mc_bias": {
            "drift_shift": -0.0007,
            "vol_multiplier": 1.40,
            "jump_shock": -0.09,
        },
    },
    "supply_chain_dislocation": {
        "description": "Global logistics bottleneck with cost-push inflation and earnings pressure",
        "factor_shocks": {
            "energy_index": 0.18,
            "inflation": 0.015,
            "industrial_production": -0.02,
            "log_return": -0.08,
        },
        "correlation_breakdown_strength": 0.33,
        "mc_bias": {
            "drift_shift": -0.0005,
            "vol_multiplier": 1.25,
            "jump_shock": -0.07,
        },
    },
    "sovereign_debt_stress": {
        "description": "Rates repricing and sentiment shock under fiscal stress regime",
        "factor_shocks": {
            "us10y_treasury_yield": 0.02,
            "consumer_sentiment": -0.12,
            "gdp_growth": -0.015,
            "log_return": -0.14,
        },
        "correlation_breakdown_strength": 0.42,
        "mc_bias": {
            "drift_shift": -0.0011,
            "vol_multiplier": 1.50,
            "jump_shock": -0.13,
        },
    },
}


def resolve_stress_scenario(
    scenario_name: Optional[str],
    shock_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    scenario_key = str(scenario_name or "custom").strip().lower()
    scenario = PRESET_SCENARIOS.get(scenario_key, {})
    base_shocks: Dict[str, float] = dict(scenario.get("factor_shocks", {}))
    if shock_map:
        base_shocks.update({str(k): float(v) for k, v in shock_map.items()})
    return {
        "name": scenario_key,
        "description": scenario.get("description", "custom scenario"),
        "factor_shocks": base_shocks,
        "correlation_breakdown_strength": float(
            scenario.get("correlation_breakdown_strength", 0.25)
        ),
        "mc_bias": scenario.get("mc_bias", {}),
        # sector_hint is None for non-sector-specific scenarios and for custom ones.
        # Downstream callers can use it to filter shocks to tickers in that sector.
        "sector_hint": scenario.get("sector_hint", None),
    }


def stress_test(
    df: pd.DataFrame,
    shock_map: Dict[str, float],
    target: str = "log_return",
    ticker: Optional[str] = None,
    anchor_event: Optional[str] = None,
    anchor_windows: Optional[Dict[str, tuple[str, str]]] = None,
    macro_lag_days: int = 0,
    scenario_name: Optional[str] = None,
    sector_column: str = "sector",
) -> Dict[str, Any]:
    """Estimate portfolio return impact under hypothetical macro shocks.

    For each factor in ``shock_map`` this function fits a simple OLS model:

        log_return = α + β·factor + ε

    then computes the *predicted impact* of an instantaneous shock Δ as:

        predicted_impact = β · Δ

    This is the classic **scenario analysis** technique used in risk
    management: "If inflation jumps 10%, how many basis points do we lose?"
    It deliberately avoids non-linear interaction terms to keep the result
    interpretable by non-quants (CFOs, risk committees).

    Args:
        df: Master table ``DataFrame`` produced by ``GoldLayer``.  Must
            contain ``'log_return'`` and every key in ``shock_map``.
        shock_map: Mapping of factor name → shock magnitude as a decimal
            (e.g. ``{'inflation': 0.10}`` means a 10% absolute increase).

    Returns:
        A dict mapping each factor to a human-readable impact string such
        as ``"Predicted impact on returns: -3.47%"``, or ``None`` on
        recoverable errors (though specific errors are raised).

    Raises:
        DataValidationError: If ``'log_return'`` or any factor column is
            absent from ``df``.
        AnalysisError: On any statsmodels fitting or runtime failure.
    """
    try:
        if target not in df.columns:
            raise DataValidationError("DataFrame must contain 'log_return' column.")

        work_df = df.copy()
        if anchor_event and anchor_windows and anchor_event in anchor_windows and "date" in work_df.columns:
            start_date, end_date = anchor_windows[anchor_event]
            work_df = work_df.copy()
            work_df["date"] = pd.to_datetime(work_df["date"], errors="coerce")
            work_df = work_df[
                work_df["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
            ].copy()

        scenario_payload = resolve_stress_scenario(
            scenario_name=scenario_name,
            shock_map=shock_map,
        )
        resolved_shocks: Dict[str, float] = dict(scenario_payload["factor_shocks"])
        sector_hint = scenario_payload.get("sector_hint")
        direct_target_shock = None
        sector_scope_match = None

        if target in resolved_shocks:
            direct_target_shock = float(resolved_shocks.pop(target))

        # Sector-targeted scenario guardrail:
        # apply direct target shock only when scope has explicit sector match.
        if sector_hint and direct_target_shock is not None:
            scope_matches = False
            if sector_column in work_df.columns:
                scoped_values = (
                    work_df[sector_column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .unique()
                    .tolist()
                )
                scope_matches = any(v == str(sector_hint).strip().lower() for v in scoped_values)
            sector_scope_match = bool(scope_matches)
            if not scope_matches:
                direct_target_shock = 0.0

        results = {}
        for factor, shock in resolved_shocks.items():
            if factor not in work_df.columns:
                raise DataValidationError(f"Factor {factor} not found in DataFrame.")

            panel, _ = prepare_supervised_frame(
                df=work_df,
                target=target,
                features=[factor],
                ticker=ticker,
                macro_lag_days=macro_lag_days,
            )
            if len(panel) < 30:
                raise DataValidationError(
                    f"Insufficient transformed rows for stress factor {factor}."
                )

            model = sm.OLS(panel[target], sm.add_constant(panel[factor])).fit()
            impact = model.params[factor] * shock
            results[factor] = {
                "shock": float(shock),
                "beta": float(model.params[factor]),
                "predicted_impact": float(impact),
            }
        return {
            "scenario": scenario_payload,
            "anchor_event": anchor_event or "full_sample",
            "target": target,
            "sector_scope_applied": sector_hint if sector_hint else None,
            "sector_scope_match": sector_scope_match,
            "direct_target_shock": direct_target_shock,
            "total_estimated_impact": float(
                sum(float(item.get("predicted_impact", 0.0)) for item in results.values())
                + (float(direct_target_shock) if isinstance(direct_target_shock, (int, float)) else 0.0)
            ),
            "results": results,
        }
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in stress_test: {e}") from e
