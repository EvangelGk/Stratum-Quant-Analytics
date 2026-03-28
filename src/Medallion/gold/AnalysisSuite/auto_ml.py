from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

from exceptions.MedallionExceptions import AnalysisError, DataValidationError

from .mixed_frequency import (
    add_volatility_regime_feature,
    aggregate_source_importance,
    prepare_supervised_frame,
)


def auto_ml_regression(
    df: pd.DataFrame,
    target: str,
    features: list,
    random_state: Optional[int] = None,
    ticker: Optional[str] = None,
    macro_lag_days: int = 0,
) -> Dict[str, Any]:
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
        if target not in df.columns or not all(f in df.columns for f in features):
            raise DataValidationError("Columns not found.")

        panel, metadata = prepare_supervised_frame(
            df=df,
            target=target,
            features=features,
            ticker=ticker,
            macro_lag_days=macro_lag_days,
        )
        panel = add_volatility_regime_feature(
            panel,
            date_col="date",
            return_col=target,
            window=30,
            threshold_quantile=0.75,
        )
        model_features = list(features)
        if "volatility_regime_high" in panel.columns:
            model_features.append("volatility_regime_high")
            metadata["volatility_regime_high"] = {
                "source": "derived",
                "lag_days": 0,
                "transformation": "binary_regime_indicator",
                "native_horizon_days": 1,
            }

        if len(panel) < max(60, len(features) * 12):
            raise DataValidationError("Insufficient stationary rows for AutoML.")

        holdout_size = max(20, int(len(panel) * 0.2))
        train_df = panel.iloc[:-holdout_size].copy()
        test_df = panel.iloc[-holdout_size:].copy()
        if train_df.empty or test_df.empty:
            raise DataValidationError("Unable to create train/test split for AutoML.")

        x_train = train_df[model_features]
        y_train = train_df[target]
        x_test = test_df[model_features]
        y_test = test_df[target]

        # ── NaN guard ────────────────────────────────────────────────────────
        # Financial data has fat tails; NaNs after prepare_supervised_frame
        # indicate a failed outer-join / ffill in the macro merge.
        if x_train.isnull().any().any() or x_test.isnull().any().any():
            raise DataValidationError(
                "NaN values in model features after prepare_supervised_frame. "
                "Ensure the Gold-layer outer-join + ffill completed correctly."
            )

        # ── RobustScaler: resistant to fat tails in financial data ───────────
        # Standard scaling fails when outliers dominate σ.  RobustScaler uses
        # median / IQR, which is stable even with extreme daily moves.
        # Fitted ONLY on train split to prevent test-set leakage.
        _final_scaler = RobustScaler()
        x_train_scaled = pd.DataFrame(
            _final_scaler.fit_transform(x_train),
            columns=x_train.columns,
            index=x_train.index,
        )
        x_test_scaled = pd.DataFrame(
            _final_scaler.transform(x_test),
            columns=x_test.columns,
            index=x_test.index,
        )

        seed = 42 if random_state is None else int(random_state)
        candidates = {
            "Ridge": Ridge(alpha=0.5),
            "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.3, random_state=seed),
            "RandomForest": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=4,
                random_state=seed,
                n_jobs=-1,
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=4,
                random_state=seed,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingRegressor(random_state=seed),
        }

        split_count = max(2, min(5, len(train_df) // 60))
        splitter = TimeSeriesSplit(n_splits=split_count)
        candidate_scores = []
        best_name = ""
        best_score = -np.inf

        for name, estimator in candidates.items():
            fold_r2 = []
            fold_mae = []
            for train_idx, valid_idx in splitter.split(x_train_scaled):
                fold_x_train = x_train_scaled.iloc[train_idx]
                fold_y_train = y_train.iloc[train_idx]
                fold_x_valid = x_train_scaled.iloc[valid_idx]
                fold_y_valid = y_train.iloc[valid_idx]
                # Per-fold scaling prevents look-ahead leakage from validation set.
                _fold_scaler = RobustScaler()
                fold_x_train_s = pd.DataFrame(
                    _fold_scaler.fit_transform(fold_x_train),
                    columns=fold_x_train.columns,
                    index=fold_x_train.index,
                )
                fold_x_valid_s = pd.DataFrame(
                    _fold_scaler.transform(fold_x_valid),
                    columns=fold_x_valid.columns,
                    index=fold_x_valid.index,
                )
                estimator.fit(fold_x_train_s, fold_y_train)
                fold_predictions = estimator.predict(fold_x_valid_s)
                fold_r2.append(float(r2_score(fold_y_valid, fold_predictions)))
                fold_mae.append(float(mean_absolute_error(fold_y_valid, fold_predictions)))

            mean_r2 = float(np.mean(fold_r2))
            mean_mae = float(np.mean(fold_mae))
            candidate_scores.append(
                {
                    "model": name,
                    "cv_r2": mean_r2,
                    "cv_mae": mean_mae,
                }
            )
            if mean_r2 > best_score:
                best_score = mean_r2
                best_name = name

        best_model = candidates[best_name]
        best_model.fit(x_train_scaled, y_train)
        holdout_predictions = best_model.predict(x_test_scaled)

        if hasattr(best_model, "feature_importances_"):
            raw_values = np.asarray(getattr(best_model, "feature_importances_"), dtype=float)
        elif hasattr(best_model, "coef_"):
            raw_values = np.abs(np.asarray(getattr(best_model, "coef_"), dtype=float))
        else:
            perm = permutation_importance(
                best_model,
                x_test_scaled,
                y_test,
                n_repeats=10,
                random_state=seed,
                scoring="r2",
            )
            raw_values = np.abs(np.asarray(perm.importances_mean, dtype=float))

        total_importance = float(np.sum(raw_values)) or 1.0
        feature_importance = {feature: float(value / total_importance) for feature, value in zip(model_features, raw_values)}

        prediction_frame = pd.DataFrame(
            {
                "date": test_df["date"],
                "actual": y_test.values,
                "prediction": holdout_predictions,
            }
        )
        return {
            "best_model": best_name,
            "ticker": ticker,
            "cv_r2": round(best_score, 6),
            "holdout_r2": round(float(r2_score(y_test, holdout_predictions)), 6),
            "holdout_mae": round(float(mean_absolute_error(y_test, holdout_predictions)), 6),
            "candidate_scores": sorted(
                candidate_scores,
                key=lambda item: float(item["cv_r2"]),
                reverse=True,
            ),
            "feature_importance": sorted(
                [
                    {
                        "feature": feature,
                        "importance": round(value, 6),
                    }
                    for feature, value in feature_importance.items()
                ],
                key=lambda item: float(item["importance"]),
                reverse=True,
            ),
            "source_importance": aggregate_source_importance(feature_importance),
            "predictions": prediction_frame,
            "validation_scheme": "walk_forward_time_series_split",
            "feature_scaling": "RobustScaler(per_fold+final_holdout)",
            "regime_feature": {
                "enabled": "volatility_regime_high" in panel.columns,
                "window_days": 30,
                "threshold_quantile": 0.75,
            },
            "transformations": metadata,
        }
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in auto_ml_regression: {e}") from e
