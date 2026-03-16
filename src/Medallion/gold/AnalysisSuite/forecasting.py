from typing import Tuple, Union

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from exceptions.MedallionExceptions import AnalysisError, DataValidationError


def forecasting(
    df: pd.DataFrame,
    column: str,
    steps: int = 10,
    order: Tuple[int, int, int] = (5, 1, 0),
) -> Union[pd.Series, None]:
    """Forecast a time series using an ARIMA(p, d, q) model.

    ARIMA (AutoRegressive Integrated Moving Average) captures short-term
    momentum and mean-reversion in financial series:

    - **AR(p)** — regresses on its own past ``p`` values (trend memory)
    - **I(d)**  — differences the series ``d`` times to achieve stationarity
    - **MA(q)** — models the error as a moving average of past ``q`` shocks

    The default order ``(5, 1, 0)`` is a common starting point for
    non-stationary log-return series: one round of differencing removes
    the unit root while 5 AR lags capture a trading-week autocorrelation.

    Args:
        df: Master table ``DataFrame`` produced by ``GoldLayer``.  Must
            contain both ``column`` and a ``'date'`` column.
        column: Name of the column to forecast, e.g. ``'log_return'``.
        steps: Number of out-of-sample periods to project.
        order: ARIMA hyperparameter tuple ``(p, d, q)``.

    Returns:
        A ``pd.Series`` of length ``steps`` with forecasted values, or
        ``None`` on recoverable errors (though specific errors are raised).

    Raises:
        DataValidationError: If ``column`` or ``'date'`` are absent, or
            if the column contains no non-null data.
        AnalysisError: On any statsmodels fitting or runtime failure.
    """
    try:
        if column not in df.columns:
            raise DataValidationError(f"Column {column} not found in DataFrame.")

        # Assume df has a date column for time series
        if "date" not in df.columns:
            raise DataValidationError(
                "DataFrame must have a 'date' column for time series."
            )

        ts_data = df.set_index("date")[column].dropna()
        if ts_data.empty:
            raise DataValidationError(f"No data in column {column}.")

        # Ensure DatetimeIndex for ARIMA
        if not isinstance(ts_data.index, pd.DatetimeIndex):
            ts_data.index = pd.to_datetime(ts_data.index)

        model = ARIMA(ts_data, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in forecasting: {e}") from e
