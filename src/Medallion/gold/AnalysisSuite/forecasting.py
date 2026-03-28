from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from exceptions.MedallionExceptions import AnalysisError, DataValidationError

from .mixed_frequency import filter_to_ticker


def forecasting(
    df: pd.DataFrame,
    column: str,
    steps: int = 10,
    order: Tuple[int, int, int] = (2, 1, 1),
    ticker: Optional[str] = None,
    volatility_window: int = 30,
) -> Dict[str, Any]:
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
            raise DataValidationError("DataFrame must have a 'date' column for time series.")

        work_df = filter_to_ticker(df, ticker=ticker).copy()
        work_df["date"] = pd.to_datetime(work_df["date"], errors="coerce")
        work_df = work_df.dropna(subset=["date"]).sort_values("date")

        raw_series = pd.to_numeric(work_df[column], errors="coerce")

        # Determine the stationary transformation for the requested column,
        # then forecast that series directly — not a derived volatility measure.
        if column == "log_return":
            # log-returns are already stationary
            transformed = raw_series
            target_label = column
        elif column == "close":
            # convert price level to log-returns for stationarity
            transformed = np.log(raw_series / raw_series.shift(1))
            target_label = "log_return_from_close"
        else:
            # first-difference general macro/economic series
            transformed = raw_series.diff()
            target_label = f"differenced_{column}"

        transformed = transformed.replace([np.inf, -np.inf], np.nan)
        valid_idx = transformed.dropna().index
        ts_data = transformed.loc[valid_idx].copy()
        if ts_data.empty:
            raise DataValidationError(f"No data available for column '{column}' after stationarity transformation.")

        # Align a DateTimeIndex so ARIMA can produce interpretable forecast dates
        ts_data.index = pd.to_datetime(work_df.loc[valid_idx, "date"].values)

        # Series above are explicitly transformed to stationarity, so we keep
        # ARIMA integration order at d=0 to avoid over-differencing.
        effective_order = (int(order[0]), 0, int(order[2]))
        model = ARIMA(ts_data, order=effective_order)
        model_fit = model.fit()
        forecast_res = model_fit.get_forecast(steps=steps)
        forecast = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.10)
        return {
            "target": target_label,
            "column": column,
            "ticker": ticker,
            "current_value": float(ts_data.iloc[-1]),
            "forecast": [float(x) for x in forecast.tolist()],
            "lower_90": [float(x) for x in conf_int.iloc[:, 0].tolist()],
            "upper_90": [float(x) for x in conf_int.iloc[:, 1].tolist()],
            "volatility_window": int(volatility_window),
            "order": list(effective_order),
            "method": "arima",
        }
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in forecasting: {e}") from e
