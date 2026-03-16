import importlib
from typing import Any, Callable, Optional, Union, cast

import pandas as pd

from exceptions.MedallionExceptions import AnalysisError, DataValidationError

compare_models: Optional[Callable[..., Any]] = None
predict_model: Optional[Callable[..., Any]] = None
setup: Optional[Callable[..., Any]] = None

try:
    pycaret_regression = importlib.import_module("pycaret.regression")
    compare_models = getattr(pycaret_regression, "compare_models", None)
    predict_model = getattr(pycaret_regression, "predict_model", None)
    setup = getattr(pycaret_regression, "setup", None)
except Exception:
    compare_models = None
    predict_model = None
    setup = None


def auto_ml_regression(
    df: pd.DataFrame,
    target: str,
    features: list,
    random_state: Optional[int] = None,
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

        setup_fn = cast(Callable[..., Any], setup)
        compare_models_fn = cast(Callable[..., Any], compare_models)
        predict_model_fn = cast(Callable[..., Any], predict_model)

        data = df[features + [target]].dropna()
        setup_fn(
            data=data,
            target=target,
            silent=True,
            verbose=False,
            session_id=random_state,
            fold_shuffle=False,
        )
        best_model = compare_models_fn()
        predictions = predict_model_fn(best_model, data=data)
        return {"best_model": str(best_model), "predictions": predictions}
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in auto_ml_regression: {e}") from e
