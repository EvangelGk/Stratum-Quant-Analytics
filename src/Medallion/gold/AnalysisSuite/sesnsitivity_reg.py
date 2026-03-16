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
    """Run multivariate macro sensitivity regression on equity log-returns.

    Estimates the linear relationship between equity log-returns and a
    set of macro factors via:

        log_return = α + β₁·inflation + β₂·energy_index + … + ε

    Two solvers are supported:

    - **OLS** (``statsmodels``): Unbiased BLUE estimator under Gauss-Markov
      assumptions.  Returns a full ``Summary`` object with t-stats, p-values,
      R², and confidence intervals — the standard output for a risk
      attribution report.
    - **Ridge** (``scikit-learn``): L2-regularised estimator that shrinks
      coefficients toward zero, reducing variance at the cost of slight bias.
      Preferred when macro factors are collinear (e.g. inflation & energy
      prices are highly correlated).  Returns a plain coefficient dict.

    Args:
        df: Master table ``DataFrame`` produced by ``GoldLayer``.  Must
            contain ``target`` and all columns listed in ``factors``.
        target: Dependent variable column, default ``'log_return'``.
        factors: List of independent macro-factor column names.  Defaults
            to ``['inflation', 'energy_index']``.
        model: Regression solver — ``'OLS'`` or ``'Ridge'``.

    Returns:
        - ``'OLS'``: A ``statsmodels`` ``Summary`` object (printable).
        - ``'Ridge'``: A ``dict`` with keys ``'coefficients'`` and
          ``'intercept'``.
        - ``None`` is never returned; specific exceptions are raised.

    Raises:
        DataValidationError: If ``target``, any factor column, or an
            invalid model name is provided.
        AnalysisError: On any fitting or runtime failure.
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
