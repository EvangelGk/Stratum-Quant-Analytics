from typing import Dict, Union

import pandas as pd

from exceptions.MedallionExceptions import AnalysisError, DataValidationError


def lag_analysis(
    df: pd.DataFrame, column: str, lags: int = 3
) -> Union[Dict[str, float], None]:
    """Compute autocorrelations at multiple lag periods for a time series.

    Autocorrelation at lag *k* measures the Pearson correlation between
    a series and its own value *k* periods earlier:

        ρ(k) = Corr(X_t, X_{t-k})

    In a financial context this quantifies **delayed macro impact** — for
    example, a significant ρ(3) on inflation implies that today's returns
    are correlated with inflation three months ago, suggesting a
    transmission lag in the economy.  This is a prerequisite step before
    selecting the AR order ``p`` for ARIMA forecasting.

    Args:
        df: Master table ``DataFrame`` produced by ``GoldLayer``.  Must
            contain ``column``.
        column: Name of the column to analyse, e.g. ``'inflation'`` or
            ``'log_return'``.
        lags: Number of lag periods to compute.  Must be ≥ 1.

    Returns:
        A dict mapping ``'lag_k'`` → Pearson correlation ``float`` for
        each *k* from 1 to ``lags``, or ``None`` on recoverable errors
        (though specific errors are raised).

    Raises:
        DataValidationError: If ``column`` is absent from ``df`` or
            ``lags`` is less than 1.
        AnalysisError: On any unexpected runtime failure.
    """
    try:
        if column not in df.columns:
            raise DataValidationError(f"Column {column} not found in DataFrame.")

        if lags < 1:
            raise DataValidationError("Lags must be at least 1.")

        return {
            f"lag_{i}": df[column].corr(df[column].shift(i)) for i in range(1, lags + 1)
        }
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in lag_analysis: {e}") from e
