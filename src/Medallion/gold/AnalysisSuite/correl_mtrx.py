from typing import Optional, Union

import numpy as np
import pandas as pd

from exceptions.MedallionExceptions import AnalysisError, DataValidationError

from .mixed_frequency import build_stationary_panel


def correl_mtrx(
    df: pd.DataFrame,
    stress_mode: bool = False,
    stress_strength: float = 0.30,
    stress_target_correlation: float = 0.85,
    scenario_name: Optional[str] = None,
) -> Union[pd.DataFrame, None]:
    """Compute the Pearson correlation matrix across all numeric features.

    The correlation matrix is the fundamental tool for **portfolio
    diversification analysis**.  A value close to +1 between two assets
    signals they move together (concentrated risk); close to 0 means
    near-independence (diversification benefit); close to -1 signals
    natural hedging.

    For a macro-equity dataset the matrix simultaneously reveals:
    - Cross-asset return correlations (equity co-movement)
    - Equity-macro factor sensitivities (e.g. AAPL vs. energy_index)
    - Inter-macro collinearity (e.g. inflation vs. energy_index)

    The computation uses Pandas ``DataFrame.corr()`` which defaults to
    Pearson's *r* — appropriate for approximately normally distributed
    log-returns.

    Args:
        df: Master table ``DataFrame`` produced by ``GoldLayer``.  All
            numeric columns are included automatically.

    Returns:
        A square ``DataFrame`` of shape ``(n_features, n_features)`` with
        values in ``[-1, 1]``, or ``None`` on recoverable errors (though
        specific errors are raised).

    Raises:
        DataValidationError: If ``df`` contains no numeric columns.
        AnalysisError: On any unexpected runtime failure.
    """
    try:
        exclude_columns = {
            "quality_score",
            "imputed_count",
            "outliers_clipped",
            "initial_rows",
            "initial_nulls",
        }
        numeric_columns = [column for column in df.select_dtypes(include=[np.number]).columns if column not in exclude_columns]
        numeric_df, _ = build_stationary_panel(
            df=df,
            columns=numeric_columns,
            macro_lag_days=0,
        )
        numeric_df = numeric_df.drop(columns=["date"], errors="ignore")
        if numeric_df.empty:
            raise DataValidationError("No numeric columns found in DataFrame.")

        corr = numeric_df.corr(method="pearson")
        if not stress_mode or corr.empty:
            return corr

        alpha = max(0.0, min(float(stress_strength), 1.0))
        target = max(-1.0, min(float(stress_target_correlation), 1.0))
        adjusted = corr.copy()
        for i, row_name in enumerate(adjusted.index):
            for j, col_name in enumerate(adjusted.columns):
                if i == j:
                    adjusted.loc[row_name, col_name] = 1.0
                    continue
                base = float(adjusted.loc[row_name, col_name])
                adjusted.loc[row_name, col_name] = base + alpha * (target - base)
        adjusted.attrs["stress_mode"] = True
        adjusted.attrs["scenario_name"] = scenario_name or "stress"
        adjusted.attrs["stress_strength"] = alpha
        return adjusted
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in correl_mtrx: {e}") from e
