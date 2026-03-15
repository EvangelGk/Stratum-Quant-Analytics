import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Union, Tuple
import numpy as np
from exceptions.MedallionExceptions import DataValidationError, AnalysisError

def forecasting(df: pd.DataFrame, column: str, steps: int = 10, order: Tuple[int, int, int] = (5, 1, 0)) -> Union[pd.Series, None]:
    """
    Time series forecasting using ARIMA model.

    Parameters:
    - df: Master table DataFrame from GoldLayer.
    - column: Column to forecast (e.g., 'log_return').
    - steps: Number of future steps to forecast.
    - order: ARIMA order (p, d, q).

    Returns:
    - Forecasted series, or None if error.
    """
    try:
        if column not in df.columns:
            raise DataValidationError(f"Column {column} not found in DataFrame.")

        # Assume df has a date column for time series
        if 'date' not in df.columns:
            raise DataValidationError("DataFrame must have a 'date' column for time series.")

        ts_data = df.set_index('date')[column].dropna()
        if ts_data.empty:
            raise DataValidationError(f"No data in column {column}.")

        model = ARIMA(ts_data, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in forecasting: {e}") from e