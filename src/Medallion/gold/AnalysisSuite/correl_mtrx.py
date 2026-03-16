from typing import Union

import numpy as np
import pandas as pd

from exceptions.MedallionExceptions import AnalysisError, DataValidationError


def correl_mtrx(df: pd.DataFrame) -> Union[pd.DataFrame, None]:
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
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise DataValidationError("No numeric columns found in DataFrame.")

        return numeric_df.corr()
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in correl_mtrx: {e}") from e
