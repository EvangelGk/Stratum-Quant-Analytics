from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller, kpss

from exceptions.MedallionExceptions import AnalysisError, DataValidationError


def _benjamini_hochberg(
    p_values: Dict[str, float], alpha: float = 0.05
) -> Dict[str, bool]:
    if not p_values:
        return {}

    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    threshold_rank = 0
    for i, (_, p_val) in enumerate(ordered, start=1):
        if p_val <= (i / m) * alpha:
            threshold_rank = i

    rejected: Dict[str, bool] = {key: False for key in p_values.keys()}
    for i, (key, _) in enumerate(ordered, start=1):
        if i <= threshold_rank:
            rejected[key] = True
    return rejected


def _score_from_r2(r2_value: float) -> float:
    clipped = max(-0.5, min(0.5, r2_value))
    return (0.5 - clipped) / 1.0


def _walk_forward_backtest(
    df: pd.DataFrame,
    target: str,
    factors: List[str],
    windows: int,
    min_train_rows: int,
) -> Dict[str, Any]:
    n_rows = len(df)
    windows = max(2, windows)
    min_test_size = 5

    if n_rows < (min_train_rows + min_test_size + windows):
        return {
            "status": "insufficient_data",
            "windows_requested": windows,
            "windows_completed": 0,
            "avg_r2": None,
            "worst_r2": None,
            "window_metrics": [],
        }

    step = max(min_test_size, (n_rows - min_train_rows) // windows)
    metrics: List[Dict[str, Any]] = []

    for idx in range(windows):
        train_end = min_train_rows + idx * step
        test_end = min(n_rows, train_end + step)
        if test_end - train_end < min_test_size or train_end < min_train_rows:
            continue

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        x_train = train_df[factors]
        y_train = train_df[target]
        x_test = test_df[factors]
        y_test = test_df[target]

        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        metrics.append(
            {
                "window": idx + 1,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "r2": float(r2_score(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            }
        )

    if not metrics:
        return {
            "status": "insufficient_data",
            "windows_requested": windows,
            "windows_completed": 0,
            "avg_r2": None,
            "worst_r2": None,
            "window_metrics": [],
        }

    r2_values = [m["r2"] for m in metrics]
    return {
        "status": "ok",
        "windows_requested": windows,
        "windows_completed": len(metrics),
        "avg_r2": float(np.mean(r2_values)),
        "worst_r2": float(np.min(r2_values)),
        "window_metrics": metrics,
    }


def _compute_model_risk_score(
    leakage_flags_count: int,
    factors_count: int,
    stationarity_ratio: float,
    normalized_shift: float,
    oos_r2: float,
    walk_forward_avg_r2: float | None,
) -> float:
    leakage_component = min(1.0, leakage_flags_count / max(1, factors_count))
    stationarity_component = max(0.0, 1.0 - stationarity_ratio)
    stability_component = min(1.0, max(0.0, normalized_shift / 5.0))
    perf_r2 = walk_forward_avg_r2 if walk_forward_avg_r2 is not None else oos_r2
    performance_component = min(1.0, max(0.0, _score_from_r2(perf_r2)))

    score = (
        0.35 * leakage_component
        + 0.25 * stationarity_component
        + 0.20 * stability_component
        + 0.20 * performance_component
    )
    return float(min(max(score, 0.0), 1.0))


def governance_report(
    df: pd.DataFrame,
    target: str,
    factors: List[str],
    date_col: str = "date",
    test_ratio: float = 0.2,
    min_train_rows: int = 24,
    random_seed: int | None = None,
    reproducibility_enforced: bool = True,
    walk_forward_windows: int = 4,
) -> Dict[str, Any]:
    """Build statistical-governance diagnostics for regression analyses.

    Provides three blocks:
    - temporal out-of-sample validation
    - leakage risk signals
    - stability/drift diagnostics between train and test windows
    """
    try:
        if target not in df.columns:
            raise DataValidationError(f"Target column {target} not found.")

        if date_col not in df.columns:
            raise DataValidationError(f"Date column {date_col} not found.")

        valid_factors = [f for f in factors if f in df.columns and f != target]
        if not valid_factors:
            raise DataValidationError("No valid factors found for governance checks.")

        work_df = df[[date_col, target] + valid_factors].copy()
        work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
        work_df = work_df.dropna(subset=[date_col, target]).sort_values(date_col)
        work_df = work_df.dropna(subset=valid_factors)

        if len(work_df) < max(min_train_rows + 5, 30):
            return {
                "status": "insufficient_data",
                "rows": int(len(work_df)),
                "required_rows": int(max(min_train_rows + 5, 30)),
            }

        stationarity: Dict[str, Dict[str, Any]] = {}
        for col in [target] + valid_factors:
            series = work_df[col].dropna()
            if len(series) < 20:
                stationarity[col] = {
                    "status": "insufficient_data",
                    "adf_p_value": None,
                    "kpss_p_value": None,
                    "is_stationary": None,
                }
                continue

            adf_stat, adf_p_value, *_ = adfuller(series, autolag="AIC")
            try:
                kpss_stat, kpss_p_value, *_ = kpss(series, regression="c", nlags="auto")
            except Exception:
                kpss_stat, kpss_p_value = np.nan, np.nan

            is_stationary = bool(
                pd.notna(adf_p_value)
                and adf_p_value < 0.05
                and pd.notna(kpss_p_value)
                and kpss_p_value > 0.05
            )
            stationarity[col] = {
                "status": "ok",
                "adf_stat": float(adf_stat),
                "adf_p_value": float(adf_p_value),
                "kpss_stat": float(kpss_stat) if pd.notna(kpss_stat) else None,
                "kpss_p_value": float(kpss_p_value) if pd.notna(kpss_p_value) else None,
                "is_stationary": is_stationary,
            }

        split_idx = int(len(work_df) * (1 - test_ratio))
        split_idx = max(min_train_rows, min(split_idx, len(work_df) - 5))

        train_df = work_df.iloc[:split_idx].copy()
        test_df = work_df.iloc[split_idx:].copy()

        x_train = train_df[valid_factors]
        y_train = train_df[target]
        x_test = test_df[valid_factors]
        y_test = test_df[target]

        model = LinearRegression()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        walk_forward = _walk_forward_backtest(
            work_df,
            target=target,
            factors=valid_factors,
            windows=walk_forward_windows,
            min_train_rows=min_train_rows,
        )

        leakage_flags: List[str] = []
        correlation_p_values: Dict[str, float] = {}
        leakage_details: Dict[str, Any] = {}
        if work_df[date_col].duplicated().any():
            leakage_flags.append("duplicate_timestamps_detected")

        if work_df[date_col].is_monotonic_increasing is False:
            leakage_flags.append("non_monotonic_timestamps")

        for factor in valid_factors:
            corr = work_df[[target, factor]].corr().iloc[0, 1]
            if pd.notna(corr):
                _, p_val = pearsonr(work_df[target], work_df[factor])
                correlation_p_values[factor] = float(p_val)
                leakage_details[f"corr:{factor}"] = {
                    "pearson": float(corr),
                    "p_value": float(p_val),
                }

            if pd.notna(corr) and abs(float(corr)) > 0.995:
                leakage_flags.append(f"near_perfect_target_correlation:{factor}")

            lead_corr = work_df[target].corr(work_df[factor].shift(-1))
            lag_corr = work_df[target].corr(work_df[factor].shift(1))
            if pd.notna(lead_corr) and pd.notna(lag_corr):
                leakage_details[f"lead_lag:{factor}"] = {
                    "lead_corr": float(lead_corr),
                    "lag_corr": float(lag_corr),
                }
                if abs(float(lead_corr)) > abs(float(lag_corr)) + 0.15:
                    leakage_flags.append(f"possible_future_leakage:{factor}")

        bh_significant = _benjamini_hochberg(correlation_p_values, alpha=0.05)
        for factor, is_significant in bh_significant.items():
            corr_value = float(work_df[[target, factor]].corr().iloc[0, 1])
            if is_significant and abs(corr_value) > 0.95:
                leakage_flags.append(f"high_correlation_after_fdr:{factor}")

        rolling_window = max(8, min(24, len(work_df) // 5))
        rolling_mean = work_df[target].rolling(rolling_window).mean().dropna()
        rolling_std = work_df[target].rolling(rolling_window).std().dropna()
        regime_shift = {
            "window": int(rolling_window),
            "max_rolling_mean": (
                float(rolling_mean.max()) if not rolling_mean.empty else None
            ),
            "min_rolling_mean": (
                float(rolling_mean.min()) if not rolling_mean.empty else None
            ),
            "max_rolling_std": (
                float(rolling_std.max()) if not rolling_std.empty else None
            ),
            "min_rolling_std": (
                float(rolling_std.min()) if not rolling_std.empty else None
            ),
        }

        target_train_mean = float(y_train.mean())
        target_test_mean = float(y_test.mean())
        target_train_std = float(y_train.std()) if pd.notna(y_train.std()) else 0.0
        target_test_std = float(y_test.std()) if pd.notna(y_test.std()) else 0.0

        mean_shift = target_test_mean - target_train_mean
        std_shift = target_test_std - target_train_std
        denom = max(abs(target_train_std), 1e-9)
        normalized_shift = abs(mean_shift) / denom
        oos_r2 = float(r2_score(y_test, predictions))

        stationarity_known = [
            values.get("is_stationary")
            for values in stationarity.values()
            if isinstance(values, dict) and values.get("is_stationary") is not None
        ]
        stationary_ratio = (
            float(
                sum(1 for value in stationarity_known if value)
                / len(stationarity_known)
            )
            if stationarity_known
            else 0.0
        )
        model_risk_score = _compute_model_risk_score(
            leakage_flags_count=len(leakage_flags),
            factors_count=len(valid_factors),
            stationarity_ratio=stationary_ratio,
            normalized_shift=float(normalized_shift),
            oos_r2=oos_r2,
            walk_forward_avg_r2=(
                walk_forward.get("avg_r2")
                if walk_forward.get("status") == "ok"
                else None
            ),
        )

        return {
            "status": "ok",
            "split": {
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_start": str(train_df[date_col].min().date()),
                "train_end": str(train_df[date_col].max().date()),
                "test_start": str(test_df[date_col].min().date()),
                "test_end": str(test_df[date_col].max().date()),
            },
            "out_of_sample": {
                "r2": oos_r2,
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            },
            "stability": {
                "target_mean_shift": float(mean_shift),
                "target_std_shift": float(std_shift),
                "normalized_mean_shift": float(normalized_shift),
                "rolling_regime": regime_shift,
            },
            "stationarity": stationarity,
            "leakage_flags": leakage_flags,
            "leakage_details": leakage_details,
            "multiple_testing": {
                "method": "benjamini_hochberg",
                "alpha": 0.05,
                "significant_factors": [
                    factor for factor, is_sig in bh_significant.items() if is_sig
                ],
            },
            "walk_forward": walk_forward,
            "model_risk_score": model_risk_score,
            "reproducibility": {
                "random_seed": random_seed,
                "enforced": reproducibility_enforced,
                "policy": "stochastic analyses must declare/propagate random_state",
            },
            "coefficients": {
                factor: float(coef)
                for factor, coef in zip(valid_factors, model.coef_)
            },
            "intercept": float(model.intercept_),
        }
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in governance_report: {e}") from e
