import numpy as np
import pandas as pd
from typing import Union
from exceptions.MedallionExceptions import DataValidationError, AnalysisError

def correl_mtrx(df: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """
    Compute correlation matrix for numeric columns.
    Risk Diversification Matrix.

    Parameters:
    - df: Master table DataFrame from GoldLayer.

    Returns:
    - Correlation DataFrame, or None if error.
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
