from typing import Union

import numpy as np
import pandas as pd

from exceptions.MedallionExceptions import AnalysisError, DataValidationError


def monte_carlo(
    df: pd.DataFrame, ticker: str, days: int = 252, iterations: int = 10000
) -> Union[np.ndarray, None]:
    """
    Geometric Brownian Motion (GBM) simulation.
    dS = S * mu * dt + S * sigma * dW

    Parameters:
    - df: Master table DataFrame from GoldLayer.
    - ticker: Stock ticker to simulate (e.g., 'AAPL').
    - days: Number of days to simulate.
    - iterations: Number of simulation paths.

    Returns:
    - Array of price paths (days x iterations), or None if error.
    """
    try:
        if "ticker" not in df.columns or "close" not in df.columns:
            raise DataValidationError(
                "DataFrame must contain 'ticker' and 'close' columns."
            )

        data = df[df["ticker"] == ticker]["close"]
        if data.empty:
            raise DataValidationError(f"No data found for ticker {ticker}.")

        returns = np.log(data / data.shift(1)).dropna()
        if returns.empty:
            raise DataValidationError("Insufficient data to compute returns.")

        mu = returns.mean()
        sigma = returns.std()
        last_price = data.iloc[-1]

        # Vectorized Simulation (efficient)
        shocks = np.exp(
            (mu - 0.5 * sigma**2) + sigma * np.random.normal(0, 1, (days, iterations))
        )
        price_paths = last_price * shocks.cumprod(axis=0)

        return price_paths
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in monte_carlo: {e}") from e
