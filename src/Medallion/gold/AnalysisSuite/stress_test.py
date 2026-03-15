import statsmodels.api as sm
import pandas as pd
from typing import Dict, Union
from exceptions.MedallionExceptions import DataValidationError, AnalysisError

def stress_test(df: pd.DataFrame, shock_map: Dict[str, float]) -> Union[Dict[str, str], None]:
    """
    Simulates 'Black Swan' events.
    Example: {'inflation': 0.10, 'energy_index': 0.20} (10% & 20% increase)

    Parameters:
    - df: Master table DataFrame from GoldLayer.
    - shock_map: Dictionary of factor shocks (e.g., {'inflation': 0.1}).

    Returns:
    - Dictionary with impact strings, or None if error.
    """
    try:
        if 'log_return' not in df.columns:
            raise DataValidationError("DataFrame must contain 'log_return' column.")

        results = {}
        for factor, shock in shock_map.items():
            if factor not in df.columns:
                raise DataValidationError(f"Factor {factor} not found in DataFrame.")

            model = sm.OLS(df['log_return'], sm.add_constant(df[factor])).fit()
            impact = model.params[factor] * shock
            results[factor] = f"Predicted impact on returns: {impact:.2%}"
        return results
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in stress_test: {e}") from e