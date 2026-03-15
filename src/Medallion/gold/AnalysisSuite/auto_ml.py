import pandas as pd
from pycaret.regression import setup, compare_models, predict_model
from typing import Union
from exceptions.MedallionExceptions import DataValidationError, AnalysisError

def auto_ml_regression(df: pd.DataFrame, target: str, features: list) -> Union[dict, None]:
    """
    Auto ML regression using PyCaret for best model selection.

    Parameters:
    - df: Master table DataFrame.
    - target: Target column.
    - features: List of feature columns.

    Returns:
    - Dict with best model and predictions, or None if error.
    """
    try:
        if target not in df.columns or not all(f in df.columns for f in features):
            raise DataValidationError("Columns not found.")

        data = df[features + [target]].dropna()
        setup(data=data, target=target, silent=True, verbose=False)
        best_model = compare_models()
        predictions = predict_model(best_model, data=data)
        return {'best_model': str(best_model), 'predictions': predictions}
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in auto_ml_regression: {e}") from e