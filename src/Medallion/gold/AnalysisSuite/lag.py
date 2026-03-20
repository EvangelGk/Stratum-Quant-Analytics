from typing import Any, Dict, Optional

import pandas as pd

from exceptions.MedallionExceptions import AnalysisError, DataValidationError
from .mixed_frequency import prepare_supervised_frame


def lag_analysis(
    df: pd.DataFrame,
    column: str,
    lags: int = 90,
    target: str = "log_return",
    ticker: Optional[str] = None,
    reference_lag_days: int = 30,
) -> Dict[str, Any]:
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
        if column not in df.columns or target not in df.columns:
            raise DataValidationError(
                f"Columns {column} or {target} not found in DataFrame."
            )

        if lags < 1:
            raise DataValidationError("Lags must be at least 1.")

        panel, metadata = prepare_supervised_frame(
            df=df,
            target=target,
            features=[column],
            ticker=ticker,
            macro_lag_days=0,
        )
        if panel.empty or len(panel) <= max(10, reference_lag_days + 5):
            raise DataValidationError("Insufficient aligned data for lag analysis.")

        lag_table = []
        best_row = {"lag_days": 0, "correlation": 0.0}
        target_series = panel[target]
        factor_series = panel[column]
        for lag_days in range(0, lags + 1):
            aligned = pd.concat(
                [target_series, factor_series.shift(lag_days)], axis=1
            ).dropna()
            if len(aligned) < 20:
                correlation = None
            else:
                correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            row = {"lag_days": lag_days, "correlation": correlation}
            lag_table.append(row)
            if isinstance(correlation, float) and abs(correlation) > abs(
                float(best_row.get("correlation") or 0.0)
            ):
                best_row = row

        reference_row = next(
            (
                row
                for row in lag_table
                if int(row.get("lag_days", -1)) == int(reference_lag_days)
            ),
            {"lag_days": reference_lag_days, "correlation": None},
        )

        return {
            "target": target,
            "macro_feature": column,
            "ticker": ticker,
            "best_lag_days": int(best_row["lag_days"]),
            "best_lag_correlation": best_row["correlation"],
            "reference_lag_days": int(reference_lag_days),
            "reference_lag_correlation": reference_row["correlation"],
            "lag_scan": lag_table,
            "data_points": int(len(panel)),
            "transformations": metadata,
        }
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in lag_analysis: {e}") from e
