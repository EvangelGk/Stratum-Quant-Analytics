from typing import Union

import pandas as pd

try:
    from pycaret.regression import compare_models, predict_model, setup
except ImportError:
    compare_models = None
    predict_model = None
    setup = None

from exceptions.MedallionExceptions import AnalysisError, DataValidationError


def auto_ml_regression(
    df: pd.DataFrame, target: str, features: list
) -> Union[dict, None]:
    """Select and fit the best regression model automatically via PyCaret.

    Uses PyCaret's ``compare_models()`` to benchmark a large suite of
    regression algorithms (Linear, Ridge, Lasso, Random Forest, Gradient
    Boosting, etc.) on the provided feature set, selecting the winner by
    cross-validated R².  This is useful for a quick non-parametric
    baseline to compare against the interpretable OLS results.

    .. note::
        PyCaret is an **optional** dependency and is *not* installed by
        default.  Install it separately::

            pip install pycaret

        If PyCaret is not available the function raises ``AnalysisError``
        with a clear installation message rather than a bare
        ``ImportError``.

    Args:
        df: Master table ``DataFrame`` produced by ``GoldLayer``.  Must
            contain ``target`` and all columns listed in ``features``.
        target: Name of the dependent variable column.
        features: List of independent variable column names.

    Returns:
        A dict with two keys:

        - ``'best_model'`` (``str``): String representation of the
          winning PyCaret model.
        - ``'predictions'`` (``pd.DataFrame``): Input data augmented with
          a ``'prediction_label'`` column.

    Raises:
        AnalysisError: If PyCaret is not installed or on any fitting
            failure.
        DataValidationError: If ``target`` or any feature column is absent
            from ``df``.
    """
    try:
        if any(func is None for func in (setup, compare_models, predict_model)):
            raise AnalysisError(
                "PyCaret is not installed. Install optional dependency 'pycaret' "
                "to use auto_ml_regression."
            )

        if target not in df.columns or not all(f in df.columns for f in features):
            raise DataValidationError("Columns not found.")

        data = df[features + [target]].dropna()
        setup(data=data, target=target, silent=True, verbose=False)
        best_model = compare_models()
        predictions = predict_model(best_model, data=data)
        return {"best_model": str(best_model), "predictions": predictions}
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in auto_ml_regression: {e}") from e
