import numpy as np
import pandas as pd
from typing import Union
from exceptions.MedallionExceptions import DataValidationError, AnalysisError

def elasticity(df: pd.DataFrame, asset_return: str, macro_factor: str) -> Union[float, None]:
    """
    Calculate elasticity: % Change in Asset / % Change in Macro Factor.
    Essential for pricing power analysis.

    Parameters:
    - df: Master table DataFrame from GoldLayer.
    - asset_return: Column name for asset returns (e.g., 'log_return').
    - macro_factor: Column name for macro factor (e.g., 'inflation').

    Returns:
    - Elasticity value as float, or None if error.
    """
    try:
        if asset_return not in df.columns or macro_factor not in df.columns:
            raise DataValidationError(f"Columns {asset_return} or {macro_factor} not found in DataFrame.")

        cov = df[[asset_return, macro_factor]].cov().iloc[0, 1]
        var_macro = df[macro_factor].var()
        if var_macro == 0:
            raise AnalysisError("Variance of macro factor is zero, cannot compute beta.")

        beta = cov / var_macro
        avg_macro = df[macro_factor].mean()
        avg_asset = df[asset_return].mean()
        if avg_asset == 0:
            raise AnalysisError("Average asset return is zero, cannot compute elasticity.")

        return beta * (avg_macro / avg_asset)
    except DataValidationError:
        raise  # Re-raise specific errors
    except AnalysisError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in elasticity: {e}") from e
