from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from exceptions.MedallionExceptions import AnalysisError, DataValidationError

from .mixed_frequency import prepare_supervised_frame


def elasticity(
    df: pd.DataFrame,
    asset_return: str,
    macro_factor: str,
    ticker: Optional[str] = None,
    macro_lag_days: int = 0,
    rolling_window: int = 90,
) -> Dict[str, Any]:
    """Compute the macro-factor elasticity of an asset's returns.

    Elasticity measures the sensitivity of an asset's return to a
    *percentage change* in a macro variable.  It extends a simple OLS
    beta by scaling for the relative magnitudes of each series:

        elasticity = β · (μ_macro / μ_asset)

    where β = Cov(asset, macro) / Var(macro).  This answers the practical
    question: *"If inflation rises 1%, by what percentage do my returns
    change?"* — essential for pricing-power and pass-through analysis.

    Args:
        df: Master table ``DataFrame`` produced by ``GoldLayer``.  Must
            contain both ``asset_return`` and ``macro_factor`` columns.
        asset_return: Column name for the asset log-returns,
            e.g. ``'log_return'``.
        macro_factor: Column name for the macro variable,
            e.g. ``'inflation'`` or ``'energy_index'``.

    Returns:
        Elasticity as a ``float``, or ``None`` on recoverable errors
        (though specific errors are raised instead).

    Raises:
        DataValidationError: If either column is missing from ``df``.
        AnalysisError: If the macro factor has zero variance (constant
            series) or if the mean asset return is zero (division guard).
    """
    try:
        if asset_return not in df.columns or macro_factor not in df.columns:
            raise DataValidationError(f"Columns {asset_return} or {macro_factor} not found in DataFrame.")

        panel, metadata = prepare_supervised_frame(
            df=df,
            target=asset_return,
            features=[macro_factor],
            ticker=ticker,
            macro_lag_days=macro_lag_days,
        )
        effective_window = max(
            rolling_window,
            int(metadata[macro_factor].get("native_horizon_days", 1)),
        )
        if len(panel) < max(effective_window, 30):
            raise DataValidationError("Insufficient aligned data for elasticity.")

        cov = panel[[asset_return, macro_factor]].cov().iloc[0, 1]
        var_macro = panel[macro_factor].var()
        if var_macro == 0:
            raise AnalysisError("Variance of macro factor is zero, cannot compute beta.")

        beta = cov / var_macro

        # Classical price elasticity: ε = β · (μ_macro / μ_asset)
        # This answers "if macro rises 1%, returns change by ε%".
        # Guard: when mean asset return is near zero (common for daily log-
        # returns) the scaling is undefined — fall back to reporting beta and
        # flag the result so consumers can choose how to interpret it.
        mean_asset = float(panel[asset_return].mean())
        mean_macro = float(panel[macro_factor].mean())
        if abs(mean_asset) > 1e-10:
            static_elasticity = beta * (mean_macro / mean_asset)
            elasticity_computable = True
        else:
            static_elasticity = beta
            elasticity_computable = False
        rolling_cov = panel[asset_return].rolling(effective_window).cov(panel[macro_factor])
        rolling_var = panel[macro_factor].rolling(effective_window).var()
        rolling_beta = (rolling_cov / rolling_var).replace([np.inf, -np.inf], np.nan)
        rolling_mean_asset = panel[asset_return].rolling(effective_window).mean()
        rolling_mean_macro = panel[macro_factor].rolling(effective_window).mean()
        rolling_scale = (rolling_mean_macro / rolling_mean_asset).where(rolling_mean_asset.abs() > 1e-10)
        rolling_elasticity = (rolling_beta * rolling_scale).replace([np.inf, -np.inf], np.nan)
        rolling_history = (
            pd.DataFrame(
                {
                    "date": panel["date"],
                    "elasticity": rolling_elasticity,
                }
            )
            .dropna()
            .to_dict("records")
        )

        return {
            "ticker": ticker,
            "asset_return": asset_return,
            "macro_factor": macro_factor,
            "static_elasticity": float(static_elasticity),
            "elasticity_computable": bool(elasticity_computable),
            "mean_asset_return": round(mean_asset, 8),
            "mean_macro_value": round(mean_macro, 8),
            "rolling_window_days": int(effective_window),
            "rolling_elasticity": rolling_history,
            "data_points": int(len(panel)),
            "transformations": metadata,
        }
    except DataValidationError:
        raise  # Re-raise specific errors
    except AnalysisError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in elasticity: {e}") from e
