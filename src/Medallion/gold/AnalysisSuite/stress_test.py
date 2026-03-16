from typing import Dict, Union

import pandas as pd
import statsmodels.api as sm

from exceptions.MedallionExceptions import AnalysisError, DataValidationError


def stress_test(
    df: pd.DataFrame, shock_map: Dict[str, float]
) -> Union[Dict[str, str], None]:
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
        if "log_return" not in df.columns:
            raise DataValidationError("DataFrame must contain 'log_return' column.")

        results = {}
        for factor, shock in shock_map.items():
            if factor not in df.columns:
                raise DataValidationError(f"Factor {factor} not found in DataFrame.")

            model = sm.OLS(df["log_return"], sm.add_constant(df[factor])).fit()
            impact = model.params[factor] * shock
            results[factor] = f"Predicted impact on returns: {impact:.2%}"
        return results
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in stress_test: {e}") from e
