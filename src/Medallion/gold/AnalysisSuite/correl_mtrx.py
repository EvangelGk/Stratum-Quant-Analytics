import numpy as np
import pandas as pd
from typing import Union

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
            raise ValueError("No numeric columns found in DataFrame.")

        return numeric_df.corr()
    except Exception as e:
        print(f"Error in correl_mtrx: {e}")
        return None
