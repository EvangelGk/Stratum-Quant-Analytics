from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from exceptions.MedallionExceptions import AnalysisError, DataValidationError
from .mixed_frequency import prepare_supervised_frame


def _tracking_error(actual: pd.Series, predicted: np.ndarray) -> float:
    diff = np.asarray(actual.values, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.std(diff, ddof=1))


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    equity_curve = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / np.maximum(peaks, 1e-12)) - 1.0
    return float(np.min(drawdowns))


def backtest_pre2020_holdout(
    df: pd.DataFrame,
    target: str = "log_return",
    features: Optional[List[str]] = None,
    date_col: str = "date",
    ticker: Optional[str] = None,
) -> Dict[str, Any]:
    """Train before 2020 and evaluate on 2020-2022 holdout window."""
    try:
        features = features or ["inflation", "energy_index"]
        panel, metadata = prepare_supervised_frame(
            df=df,
            target=target,
            features=features,
            date_col=date_col,
            ticker=ticker,
            macro_lag_days=45,
            align_target_to_features=True,
        )
        if panel.empty or date_col not in panel.columns:
            raise DataValidationError("No aligned rows available for backtest.")

        panel[date_col] = pd.to_datetime(panel[date_col], errors="coerce")
        panel = panel.dropna(subset=[date_col]).sort_values(date_col)

        train_mask = panel[date_col] < pd.Timestamp("2020-01-01")
        test_mask = (panel[date_col] >= pd.Timestamp("2020-01-01")) & (
            panel[date_col] <= pd.Timestamp("2022-12-31")
        )
        train_df = panel.loc[train_mask].copy()
        test_df = panel.loc[test_mask].copy()

        if len(train_df) < max(60, len(features) * 12):
            raise DataValidationError("Insufficient training rows before 2020.")
        if len(test_df) < 30:
            raise DataValidationError("Insufficient 2020-2022 holdout rows.")

        model = Ridge(alpha=1.0)
        model.fit(train_df[features], train_df[target])
        predictions = model.predict(test_df[features])
        actual = pd.to_numeric(test_df[target], errors="coerce").fillna(0.0)

        te = _tracking_error(actual, predictions)
        # MDD is a risk metric for the strategy holding the actual position,
        # not for the model's fitted values.  Use actual returns.
        actual_returns = np.asarray(actual, dtype=float)
        mdd = _max_drawdown_from_returns(actual_returns)

        return {
            "window": {
                "train_end_exclusive": "2020-01-01",
                "test_start": "2020-01-01",
                "test_end": "2022-12-31",
            },
            "ticker": ticker,
            "target": target,
            "features": list(features),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "tracking_error": round(float(te), 8),
            "maximum_drawdown": round(float(mdd), 8),
            "predictions": [float(v) for v in predictions.tolist()],
            "actual": [float(v) for v in actual.tolist()],
            "transformations": metadata,
        }
    except DataValidationError:
        raise
    except Exception as exc:
        raise AnalysisError(f"Unexpected error in backtest_pre2020_holdout: {exc}") from exc
