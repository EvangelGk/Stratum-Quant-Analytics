from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge

from exceptions.MedallionExceptions import AnalysisError, DataValidationError
from .mixed_frequency import aggregate_source_importance, prepare_supervised_frame


def sensitivity_reg(
    df: pd.DataFrame,
    target: str = "log_return",
    factors: Optional[List[str]] = None,
    model: str = "OLS",
    ticker: Optional[str] = None,
    macro_lag_days: int = 0,
) -> Any:
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

        panel, metadata = prepare_supervised_frame(
            df=df,
            target=target,
            features=factors,
            ticker=ticker,
            macro_lag_days=macro_lag_days,
        )
        if len(panel) < max(30, len(factors) * 8):
            raise DataValidationError("Insufficient rows after stationarity transforms.")

        Y = panel[target]
        X = panel[factors]

        if model == "OLS":
            design = sm.add_constant(X)
            fitted_model = sm.OLS(Y, design).fit()
            coefficients = {
                factor: float(fitted_model.params.get(factor, 0.0))
                for factor in factors
            }
            raw_importance = {
                factor: float(abs(coefficients[factor]) * X[factor].std())
                for factor in factors
            }
            return {
                "model": "OLS",
                "ticker": ticker,
                "target": target,
                # Coefficients express impact over the forward horizon used by
                # prepare_supervised_frame.  Check target_horizon_days before
                # interpreting a coefficient as an immediate point-in-time effect.
                "target_horizon_days": int(
                    metadata.get(target, {}).get("target_horizon_days", 1)
                ),
                "coefficients": coefficients,
                "intercept": float(fitted_model.params.get("const", 0.0)),
                "p_values": {
                    factor: float(fitted_model.pvalues.get(factor, np.nan))
                    for factor in factors
                },
                "r2": float(fitted_model.rsquared),
                "adj_r2": float(fitted_model.rsquared_adj),
                "n_obs": int(len(panel)),
                "summary_text": fitted_model.summary().as_text(),
                "feature_importance": raw_importance,
                "source_importance": aggregate_source_importance(raw_importance),
                "transformations": metadata,
            }
        elif model == "Ridge":
            ridge = Ridge(alpha=0.5)
            ridge.fit(X, Y)
            coefficients = {
                factor: float(coef) for factor, coef in zip(factors, ridge.coef_)
            }
            raw_importance = {
                factor: float(abs(coefficients[factor]) * X[factor].std())
                for factor in factors
            }
            return {
                "model": "Ridge",
                "ticker": ticker,
                "target": target,
                "target_horizon_days": int(
                    metadata.get(target, {}).get("target_horizon_days", 1)
                ),
                "coefficients": coefficients,
                "intercept": float(ridge.intercept_),
                "r2": float(ridge.score(X, Y)),
                "n_obs": int(len(panel)),
                "feature_importance": raw_importance,
                "source_importance": aggregate_source_importance(raw_importance),
                "transformations": metadata,
            }
        else:
            raise DataValidationError("Model must be 'OLS' or 'Ridge'.")
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in sensitivity_reg: {e}") from e
