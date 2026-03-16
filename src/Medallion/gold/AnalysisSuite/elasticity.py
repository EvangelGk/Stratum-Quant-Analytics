from typing import Union

import pandas as pd

from exceptions.MedallionExceptions import AnalysisError, DataValidationError


def elasticity(
    df: pd.DataFrame, asset_return: str, macro_factor: str
) -> Union[float, None]:
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
            raise DataValidationError(
                f"Columns {asset_return} or {macro_factor} not found in DataFrame."
            )

        cov = df[[asset_return, macro_factor]].cov().iloc[0, 1]
        var_macro = df[macro_factor].var()
        if var_macro == 0:
            raise AnalysisError(
                "Variance of macro factor is zero, cannot compute beta."
            )

        beta = cov / var_macro
        avg_macro = df[macro_factor].mean()
        avg_asset = df[asset_return].mean()
        if avg_asset == 0:
            raise AnalysisError(
                "Average asset return is zero, cannot compute elasticity."
            )

        return beta * (avg_macro / avg_asset)
    except DataValidationError:
        raise  # Re-raise specific errors
    except AnalysisError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in elasticity: {e}") from e
