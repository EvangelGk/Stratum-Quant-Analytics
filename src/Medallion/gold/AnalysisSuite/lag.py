import pandas as pd
from typing import Dict, Union

def lag_analysis(df: pd.DataFrame, column: str, lags: int = 3) -> Union[Dict[str, float], None]:
    """
    Calculates autocorrelation and delayed macro impact.

    Parameters:
    - df: Master table DataFrame from GoldLayer.
    - column: Column name to analyze (e.g., 'inflation').
    - lags: Number of lag periods to compute.

    Returns:
    - Dictionary with lag correlations, or None if error.
    """
    try:
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame.")

        if lags < 1:
            raise ValueError("Lags must be at least 1.")

        return {f"lag_{i}": df[column].corr(df[column].shift(i)) for i in range(1, lags + 1)}
    except Exception as e:
        print(f"Error in lag_analysis: {e}")
        return None
