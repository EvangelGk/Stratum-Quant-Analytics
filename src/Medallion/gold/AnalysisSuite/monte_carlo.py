from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from exceptions.MedallionExceptions import AnalysisError, DataValidationError

from .mixed_frequency import filter_to_ticker


def monte_carlo(
    df: pd.DataFrame,
    ticker: str,
    days: int = 252,
    iterations: int = 1000,
    random_state: Optional[int] = None,
    macro_scenario: Optional[str] = None,
    macro_factor: str = "inflation",
    scenario_bias: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
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
            raise DataValidationError("DataFrame must contain 'ticker' and 'close' columns.")

        scoped = filter_to_ticker(df, ticker=ticker)
        if scoped.empty:
            raise DataValidationError(f"No data found for ticker {ticker}.")

        scoped = scoped.copy()
        if "date" in scoped.columns:
            scoped["date"] = pd.to_datetime(scoped["date"], errors="coerce")
            scoped = scoped.dropna(subset=["date"]).sort_values("date")

        data = pd.to_numeric(scoped["close"], errors="coerce")

        returns = np.log(data / data.shift(1)).dropna()
        if returns.empty:
            raise DataValidationError("Insufficient data to compute returns.")

        current_daily_vol = float(returns.ewm(span=30, min_periods=15).std().iloc[-1])
        base_mu = float(returns.mean())
        base_sigma = float(returns.std())
        if not np.isfinite(current_daily_vol) or current_daily_vol <= 0:
            current_daily_vol = base_sigma
        if not np.isfinite(current_daily_vol) or current_daily_vol <= 0:
            raise DataValidationError("Cannot estimate daily volatility for Monte Carlo simulation.")
        scenario_label = str(macro_scenario or "auto_detected").lower()

        scenario_mu = base_mu
        scenario_sigma = base_sigma
        scenario_sample_size = 0
        if macro_factor in scoped.columns:
            macro_series = pd.to_numeric(scoped[macro_factor], errors="coerce").ffill()
            high_threshold = float(macro_series.quantile(0.80))
            low_threshold = float(macro_series.quantile(0.20))
            if "high" in scenario_label and "inflation" in scenario_label:
                scenario_mask = macro_series >= high_threshold
                scenario_name = "high_inflation"
            elif "low" in scenario_label and "inflation" in scenario_label:
                scenario_mask = macro_series <= low_threshold
                scenario_name = "low_inflation"
            else:
                latest_macro = float(macro_series.dropna().iloc[-1])
                scenario_mask = macro_series >= high_threshold if latest_macro >= high_threshold else macro_series <= low_threshold
                scenario_name = "high_inflation" if latest_macro >= high_threshold else "low_inflation"

            scenario_returns = returns.loc[scenario_mask.reindex(returns.index, fill_value=False)]
            scenario_sample_size = int(len(scenario_returns))
            if len(scenario_returns) >= 30:
                scenario_mu = float(scenario_returns.mean())
                scenario_sigma = float(scenario_returns.std())
        else:
            scenario_name = "baseline"

        if not np.isfinite(scenario_sigma) or scenario_sigma <= 0:
            scenario_sigma = current_daily_vol

        adjusted_mu = scenario_mu
        adjusted_sigma = max(current_daily_vol, scenario_sigma)

        bias = scenario_bias or {}
        adjusted_mu += float(bias.get("drift_shift", 0.0))
        adjusted_sigma *= max(0.1, float(bias.get("vol_multiplier", 1.0)))
        last_price = float(data.iloc[-1])
        if not np.isfinite(last_price) or last_price <= 0:
            raise DataValidationError("Invalid latest close price for Monte Carlo.")
        jump_shock = float(bias.get("jump_shock", 0.0))

        rng = np.random.default_rng(random_state)
        shocks = np.exp((adjusted_mu - 0.5 * adjusted_sigma**2) + adjusted_sigma * rng.normal(0, 1, (days, iterations)))
        if jump_shock != 0.0:
            # Apply the jump to the first-day shock factor BEFORE cumprod so
            # the step-down propagates coherently through all subsequent paths.
            # Patching price_paths[0] after cumprod would leave day 2+ unaffected.
            shocks[0, :] *= 1.0 + jump_shock

        price_paths = last_price * shocks.cumprod(axis=0)
        if not np.isfinite(price_paths).all():
            raise AnalysisError("Monte Carlo produced non-finite price paths.")

        terminal_returns = price_paths[-1] / last_price - 1.0
        var_95_cut = float(np.percentile(terminal_returns, 5))
        var_99_cut = float(np.percentile(terminal_returns, 1))
        cvar_95 = float(terminal_returns[terminal_returns <= var_95_cut].mean())
        es_99 = float(terminal_returns[terminal_returns <= var_99_cut].mean())

        hist_var_95_cut = float(np.percentile(returns, 5))
        hist_es_95 = float(returns[returns <= hist_var_95_cut].mean())

        z95 = float(norm.ppf(0.05))
        pdf95 = float(norm.pdf(z95))
        param_var_95 = float(adjusted_mu + adjusted_sigma * z95)
        param_es_95 = float(adjusted_mu - adjusted_sigma * (pdf95 / 0.05))

        return {
            "ticker": ticker,
            "scenario": scenario_name,
            "macro_factor": macro_factor,
            "scenario_sample_size": scenario_sample_size,
            "drift_daily": round(adjusted_mu, 8),
            "daily_volatility": round(current_daily_vol, 8),
            "scenario_volatility": round(adjusted_sigma, 8),
            "value_at_risk_95": round(abs(var_95_cut), 6),
            "value_at_risk_99": round(abs(var_99_cut), 6),
            "conditional_var_95": round(abs(cvar_95), 6),
            "expected_shortfall_99": round(abs(es_99), 6),
            "historical_var_95": round(abs(hist_var_95_cut), 6),
            "historical_es_95": round(abs(hist_es_95), 6),
            "parametric_var_95": round(abs(param_var_95), 6),
            "parametric_es_95": round(abs(param_es_95), 6),
            "scenario_bias": {
                "drift_shift": float(bias.get("drift_shift", 0.0)),
                "vol_multiplier": float(bias.get("vol_multiplier", 1.0)),
                "jump_shock": float(bias.get("jump_shock", 0.0)),
            },
            "price_paths": price_paths,
        }
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in monte_carlo: {e}") from e
