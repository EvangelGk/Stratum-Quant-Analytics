from typing import cast

import numpy as np
import pandas as pd

from exceptions.MedallionExceptions import AnalysisError, DataValidationError


def monte_carlo(
    df: pd.DataFrame, ticker: str, days: int = 252, iterations: int = 10000
) -> np.ndarray:
    """Simulate future price paths using Geometric Brownian Motion (GBM).

    Models stock price evolution via the stochastic differential equation:

        dS = S·μ·dt + S·σ·dW

    where μ is the annualised drift (mean log-return), σ is historical
    volatility, and dW is a Wiener process increment drawn from N(0,1).
    The discrete-time approximation is:

        S_t = S_{t-1} · exp((μ - 0.5·σ²)·dt + σ·ε·√dt)

    Running ``iterations`` independent paths produces a full probability
    distribution of future prices rather than a single point estimate —
    the industry-standard approach used by quantitative trading desks.

    Args:
        df: Master table ``DataFrame`` produced by ``GoldLayer``.  Must
            contain ``'ticker'`` and ``'close'`` columns.
        ticker: Equity ticker symbol to simulate, e.g. ``'AAPL'``.
        days: Number of trading days to project forward.  252 corresponds
            to one calendar year.
        iterations: Number of independent Monte Carlo paths.  At 10,000
            the standard error of the mean is below 1%.

    Returns:
        A ``numpy`` array of shape ``(days, iterations)`` where each
        column is one simulated price path, or ``None`` on recoverable
        errors (though specific errors are raised instead).

    Raises:
        DataValidationError: If required columns are absent or the ticker
            has no data in ``df``.
        AnalysisError: On any unexpected numerical or runtime failure.
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

        return cast(np.ndarray, price_paths)
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in monte_carlo: {e}") from e
