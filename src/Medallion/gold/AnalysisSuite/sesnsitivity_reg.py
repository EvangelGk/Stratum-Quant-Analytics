from typing import List, Union

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge

from exceptions.MedallionExceptions import AnalysisError, DataValidationError


def sensitivity_reg(
    df: pd.DataFrame,
    target: str = "log_return",
    factors: List[str] = None,
    model: str = "OLS",
) -> Union[str, dict, None]:
    """
    Multivariate Regression to find coefficients of Macro factors.
    Supports OLS (statsmodels) or Ridge (sklearn) for variations.

    Parameters:
    - df: Master table DataFrame from GoldLayer.
    - target: Target column (e.g., 'log_return').
    - factors: List of factor columns (default: ['inflation', 'energy_index']).
    - model: Regression model ('OLS' or 'Ridge').

    Returns:
    - Model summary (OLS) or coefficients dict (Ridge), or None if error.
    """
    try:
        factors = factors or ["inflation", "energy_index"]
        if target not in df.columns:
            raise DataValidationError(f"Target column {target} not found in DataFrame.")

        missing_factors = [f for f in factors if f not in df.columns]
        if missing_factors:
            raise DataValidationError(
                f"Factor columns {missing_factors} not found in DataFrame."
            )

        if model == "OLS":
            Y = df[target]
            X = sm.add_constant(df[factors])
            fitted_model = sm.OLS(Y, X).fit()
            return fitted_model.summary()
        elif model == "Ridge":
            X = df[factors]
            Y = df[target]
            ridge = Ridge(alpha=0.1)
            ridge.fit(X, Y)
            return {
                "coefficients": dict(zip(factors, ridge.coef_)),
                "intercept": ridge.intercept_,
            }
        else:
            raise DataValidationError("Model must be 'OLS' or 'Ridge'.")
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in sensitivity_reg: {e}") from e
