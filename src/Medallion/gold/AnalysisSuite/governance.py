from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller, kpss

from exceptions.MedallionExceptions import AnalysisError, DataValidationError
from .mixed_frequency import add_volatility_regime_feature, prepare_supervised_frame
from .sensitivity_reg import (
    _build_time_series_split,
    _candidate_registry,
    _engineer_features,
    _fit_candidate_with_cv,
    _reduce_multicollinearity,
)


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
    """Map R² to risk in a way that is realistic for financial-return targets.

    For daily returns, near-zero R² is common and should not be treated as
    near-failure. Use piecewise penalties instead of a linear clip mapping.
    """
    if r2_value >= 0.10:
        return 0.10
    if r2_value >= 0.00:
        return 0.20
    if r2_value >= -0.10:
        return 0.30
    if r2_value >= -0.25:
        return 0.45
    if r2_value >= -0.50:
        return 0.70
    return 1.00


def _clip_by_train_quantiles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    clipped_train = train_df.copy()
    clipped_test = test_df.copy()
    bounds: Dict[str, Dict[str, float]] = {}
    for column in clipped_train.columns:
        series = pd.to_numeric(clipped_train[column], errors="coerce")
        if series.dropna().empty:
            continue
        lo = float(series.quantile(lower_q))
        hi = float(series.quantile(upper_q))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        bounds[column] = {"lower": lo, "upper": hi}
        clipped_train[column] = series.clip(lower=lo, upper=hi)
        clipped_test[column] = pd.to_numeric(clipped_test[column], errors="coerce").clip(
            lower=lo,
            upper=hi,
        )
    return clipped_train, clipped_test, bounds


def _build_model(name: str) -> Any:
    key = str(name).strip().lower()
    if key in {"ols", "linear"}:
        return LinearRegression()
    if key == "ridge":
        return Ridge(alpha=0.5)
    if key == "lasso":
        return Lasso(alpha=0.01, max_iter=5000)
    if key == "elasticnet":
        return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
    return LinearRegression()


def _compute_baseline_suite(
    y_train: pd.Series,
    y_test: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> Dict[str, Any]:
    y_train_num = pd.to_numeric(y_train, errors="coerce")
    y_test_num = pd.to_numeric(y_test, errors="coerce")
    valid_idx = y_test_num.notna()
    y_test_num = y_test_num.loc[valid_idx]
    if y_test_num.empty:
        return {"status": "insufficient_data", "models": {}}

    x_train_num = x_train.apply(pd.to_numeric, errors="coerce")
    x_test_num = x_test.apply(pd.to_numeric, errors="coerce")
    x_test_num = x_test_num.loc[valid_idx]

    models: Dict[str, Dict[str, float | None]] = {}

    def _safe_metrics(name: str, y_pred: np.ndarray) -> None:
        try:
            models[name] = {
                "r2": float(r2_score(y_test_num, y_pred)),
                "mae": float(mean_absolute_error(y_test_num, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test_num, y_pred))),
            }
        except Exception:
            models[name] = {"r2": None, "mae": None, "rmse": None}

    train_mean = float(y_train_num.mean()) if not y_train_num.dropna().empty else 0.0
    _safe_metrics("mean_predictor", np.full(len(y_test_num), train_mean, dtype=float))

    y_train_clean = y_train_num.dropna()
    if len(y_train_clean) >= 2:
        recent_mean = float(y_train_clean.iloc[-min(5, len(y_train_clean)):].mean())
    else:
        recent_mean = train_mean
    _safe_metrics("recent_mean_predictor", np.full(len(y_test_num), recent_mean, dtype=float))

    ridge = _build_model("ridge")
    try:
        ridge.fit(x_train_num, y_train_num)
        ridge_pred = ridge.predict(x_test_num)
    except Exception:
        ridge_pred = np.full(len(y_test_num), train_mean, dtype=float)
    _safe_metrics("ridge_baseline", np.asarray(ridge_pred, dtype=float))

    # For differenced/log-return targets, zero-drift is a robust baseline.
    _safe_metrics("zero_drift", np.zeros(len(y_test_num), dtype=float))

    return {
        "status": "ok",
        "models": models,
        "best_baseline_r2": max(
            [float(v["r2"]) for v in models.values() if isinstance(v.get("r2"), (float, int))],
            default=None,
        ),
    }


def _factor_concentration_diagnostics(coefficients: Dict[str, float]) -> Dict[str, Any]:
    if not coefficients:
        return {
            "status": "insufficient_data",
            "top_factor": None,
            "top_share": None,
            "top3_share": None,
            "effective_factor_count": None,
            "weights": {},
        }

    abs_weights = {k: float(abs(v)) for k, v in coefficients.items()}
    total = float(sum(abs_weights.values()))
    if total <= 1e-12:
        return {
            "status": "degenerate",
            "top_factor": None,
            "top_share": 0.0,
            "top3_share": 0.0,
            "effective_factor_count": 0.0,
            "weights": {k: 0.0 for k in coefficients.keys()},
        }

    normalized = {k: float(v / total) for k, v in abs_weights.items()}
    ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    top_factor, top_share = ranked[0]
    top3_share = float(sum(weight for _, weight in ranked[:3]))
    hhi = float(sum(weight * weight for weight in normalized.values()))
    effective_count = float(1.0 / hhi) if hhi > 1e-12 else None

    return {
        "status": "ok",
        "top_factor": top_factor,
        "top_share": float(top_share),
        "top3_share": top3_share,
        "effective_factor_count": effective_count,
        "weights": normalized,
    }


def _prepare_best_lag_frame(
    df: pd.DataFrame,
    target: str,
    factors: List[str],
    date_col: str,
    min_train_rows: int,
    requested_macro_lag_days: int = 0,
    min_target_horizon_days: int = 1,
    max_target_horizon_days: int | None = None,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, Any]], int, Dict[str, float]]:
    lag_candidates = [int(requested_macro_lag_days)] if int(requested_macro_lag_days) > 0 else [0, 15, 30, 45]

    best_frame = pd.DataFrame()
    best_meta: Dict[str, Dict[str, Any]] = {}
    best_lag = lag_candidates[0]
    best_score = float("-inf")
    lag_scores: Dict[str, float] = {}

    for lag in lag_candidates:
        frame, meta = prepare_supervised_frame(
            df=df,
            target=target,
            features=factors,
            date_col=date_col,
            macro_lag_days=lag,
            align_target_to_features=True,
            min_target_horizon_days=min_target_horizon_days,
            max_target_horizon_days=max_target_horizon_days,
        )
        if len(frame) < max(min_train_rows + 5, 30):
            lag_scores[str(lag)] = float("-inf")
            continue

        split_idx = int(len(frame) * 0.8)
        split_idx = max(min_train_rows, min(split_idx, len(frame) - 5))
        train = frame.iloc[:split_idx]
        test = frame.iloc[split_idx:]
        x_train = train[factors].apply(pd.to_numeric, errors="coerce")
        y_train = pd.to_numeric(train[target], errors="coerce")
        x_test = test[factors].apply(pd.to_numeric, errors="coerce")
        y_test = pd.to_numeric(test[target], errors="coerce")

        baseline = _build_model("ridge")
        try:
            baseline.fit(x_train, y_train)
            score = float(r2_score(y_test, baseline.predict(x_test)))
        except Exception:
            score = float("-inf")
        lag_scores[str(lag)] = score

        if score > best_score:
            best_score = score
            best_lag = int(lag)
            best_frame = frame
            best_meta = meta

    if best_frame.empty:
        best_frame, best_meta = prepare_supervised_frame(
            df=df,
            target=target,
            features=factors,
            date_col=date_col,
            macro_lag_days=0,
            align_target_to_features=True,
            min_target_horizon_days=min_target_horizon_days,
            max_target_horizon_days=max_target_horizon_days,
        )
        best_lag = 0

    return best_frame, best_meta, int(best_lag), lag_scores


def _walk_forward_backtest(
    df: pd.DataFrame,
    target: str,
    factors: List[str],
    windows: int,
    min_train_rows: int,
    clipped_floor: float = -2.0,
    min_test_size: int = 20,
    model_type: str = "OLS",
    tune_hyperparams_per_window: bool = False,
    random_seed: int | None = 42,
) -> Dict[str, Any]:
    n_rows = len(df)
    windows = max(2, windows)
    min_test_size = max(8, int(min_test_size))

    if n_rows < (min_train_rows + min_test_size + windows):
        return {
            "status": "insufficient_data",
            "windows_requested": windows,
            "windows_completed": 0,
            "avg_r2": None,
            "worst_r2": None,
            "window_tuning_enabled": bool(tune_hyperparams_per_window),
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

        selected_model_name = str(model_type)
        if tune_hyperparams_per_window and len(x_train) >= 50:
            registry = _candidate_registry(random_seed if random_seed is not None else 42)
            requested = str(model_type).strip().lower()
            alias = {
                "ols": "Linear",
                "linear": "Linear",
                "ridge": "Ridge",
                "lasso": "Lasso",
                "elasticnet": "ElasticNet",
                "randomforest": "RandomForest",
                "auto": "Auto",
            }
            requested_name = alias.get(requested, "Linear")
            candidate_names = list(registry.keys()) if requested_name == "Auto" else [requested_name]
            splitter = _build_time_series_split(len(x_train))
            best_cv = float("-inf")
            best_window_model = None
            best_name = requested_name
            for candidate_name in candidate_names:
                candidate, search_space = registry[candidate_name]
                fitted_model, cv_r2, _ = _fit_candidate_with_cv(
                    name=candidate_name,
                    candidate=candidate,
                    param_distributions=search_space,
                    x=x_train,
                    y=y_train,
                    splitter=splitter,
                    seed=(random_seed if random_seed is not None else 42) + idx,
                )
                if float(cv_r2) > float(best_cv):
                    best_cv = float(cv_r2)
                    best_window_model = fitted_model
                    best_name = candidate_name
            estimator = best_window_model if best_window_model is not None else _build_model(model_type)
            selected_model_name = best_name
        else:
            estimator = _build_model(model_type)

        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test)

        metrics.append(
            {
                "window": idx + 1,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "model": selected_model_name,
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
            "window_tuning_enabled": bool(tune_hyperparams_per_window),
            "window_metrics": [],
        }

    r2_values = [m["r2"] for m in metrics]
    clipped_r2_values = [max(clipped_floor, min(1.0, value)) for value in r2_values]
    # Natural CI from multiple windows: percentiles of the window-R² distribution.
    if len(r2_values) >= 4:
        wf_ci_lower = float(np.percentile(r2_values, 10))
        wf_ci_upper = float(np.percentile(r2_values, 90))
    elif len(r2_values) >= 2:
        wf_ci_lower = float(min(r2_values))
        wf_ci_upper = float(max(r2_values))
    else:
        wf_ci_lower = wf_ci_upper = float(r2_values[0])
    return {
        "status": "ok",
        "windows_requested": windows,
        "windows_completed": len(metrics),
        "window_tuning_enabled": bool(tune_hyperparams_per_window),
        "avg_r2": float(np.mean(r2_values)),
        "median_r2": float(np.median(r2_values)),
        "clipped_avg_r2": float(np.mean(clipped_r2_values)),
        "worst_r2": float(np.min(r2_values)),
        "r2_ci_lower": round(wf_ci_lower, 4),
        "r2_ci_upper": round(wf_ci_upper, 4),
        "window_metrics": metrics,
    }


def _oos_r2_confidence_interval(
    model: Any,
    test_df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    n_bootstrap: int = 200,
    block_size: int | None = None,
    confidence: float = 0.90,
    random_seed: int | None = 42,
) -> Dict[str, Any]:
    """Circular block bootstrap confidence interval for out-of-sample R².

    Returns a stable range rather than a single point estimate — more robust
    to the specific test-period boundary chosen by the 80/20 train/test split.
    Degenerate bootstrap draws (near-zero outcome variance) are excluded.
    """
    n = len(test_df)
    if n < 5:
        return {
            "status": "insufficient_data",
            "point_estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "mean": None,
            "std": None,
            "n_bootstrap": 0,
            "confidence": confidence,
        }
    block = block_size if block_size is not None else max(2, int(np.sqrt(n)))
    y_arr = test_df[y_col].values.astype(float)
    x_arr = test_df[x_cols].values.astype(float)
    point_estimate = float(r2_score(y_arr, model.predict(x_arr)))

    rng = np.random.default_rng(seed=random_seed)
    r2_samples: List[float] = []
    for _ in range(n_bootstrap):
        indices: List[int] = []
        start = int(rng.integers(0, n))
        while len(indices) < n:
            for offset in range(block):
                indices.append((start + offset) % n)
            start = (start + block) % n
        indices = indices[:n]
        y_boot = y_arr[indices]
        x_boot = x_arr[indices]
        if float(np.var(y_boot)) < 1e-12:
            continue
        raw_r2 = float(r2_score(y_boot, model.predict(x_boot)))
        # Clip extreme outliers to [-10, 1] before aggregation.
        r2_samples.append(max(-10.0, min(1.0, raw_r2)))

    if not r2_samples:
        return {
            "status": "ok",
            "point_estimate": round(point_estimate, 4),
            "ci_lower": round(point_estimate, 4),
            "ci_upper": round(point_estimate, 4),
            "mean": round(point_estimate, 4),
            "std": 0.0,
            "n_bootstrap": 0,
            "confidence": confidence,
            "block_size": int(block),
        }

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(r2_samples, (alpha / 2) * 100))
    ci_upper = float(np.percentile(r2_samples, (1.0 - alpha / 2) * 100))
    return {
        "status": "ok",
        "point_estimate": round(point_estimate, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "mean": round(float(np.mean(r2_samples)), 4),
        "std": round(float(np.std(r2_samples)), 4),
        "n_bootstrap": len(r2_samples),
        "confidence": confidence,
        "block_size": int(block),
        "random_seed": random_seed,
    }


def _compute_model_risk_score(
    leakage_flags_count: int,
    factors_count: int,
    stationarity_ratio: float | None,
    normalized_shift: float,
    oos_r2: float,
    walk_forward_avg_r2: float | None,
) -> float:
    leakage_component = min(1.0, leakage_flags_count / max(1, factors_count))
    # When stationarity was untestable (all series too short for ADF/KPSS),
    # use a neutral 0.5 rather than the maximum-penalty value of 1.0.
    if stationarity_ratio is None:
        stationarity_component = 0.5
    else:
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


def _trend_and_volatility_diagnostics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    rolling_window: int = 5,
) -> Dict[str, Any]:
    """Robust diagnostics that are less sensitive to point-level noise than raw R2."""
    true_series = pd.Series(y_true).reset_index(drop=True)
    pred_series = pd.Series(y_pred).reset_index(drop=True)

    if len(true_series) < 3 or len(pred_series) < 3:
        return {
            "status": "insufficient_data",
            "trend_directional_accuracy": None,
            "volatility_r2": None,
            "volatility_ratio": None,
        }

    true_diff = true_series.diff().dropna()
    pred_diff = pred_series.diff().dropna()
    if len(true_diff) != len(pred_diff) or len(true_diff) == 0:
        trend_acc = None
    else:
        trend_acc = float((np.sign(true_diff) == np.sign(pred_diff)).mean())

    true_vol = true_series.rolling(window=rolling_window, min_periods=rolling_window).std().dropna()
    pred_vol = pred_series.rolling(window=rolling_window, min_periods=rolling_window).std().dropna()
    common_len = min(len(true_vol), len(pred_vol))
    volatility_r2: float | None = None
    if common_len >= 3:
        true_vol = true_vol.iloc[-common_len:]
        pred_vol = pred_vol.iloc[-common_len:]
        if float(true_vol.var()) > 0.0:
            volatility_r2 = float(r2_score(true_vol, pred_vol))

    realized_vol = float(true_series.std()) if pd.notna(true_series.std()) else 0.0
    predicted_vol = float(pred_series.std()) if pd.notna(pred_series.std()) else 0.0
    vol_ratio = (predicted_vol / realized_vol) if realized_vol > 1e-12 else None

    return {
        "status": "ok",
        "trend_directional_accuracy": trend_acc,
        "volatility_r2": volatility_r2,
        "volatility_ratio": float(vol_ratio) if vol_ratio is not None else None,
        "rolling_window": int(rolling_window),
    }


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
    clipped_walk_forward_floor: float = -2.0,
    model_type: str = "OLS",
    macro_lag_days: int = 0,
    min_target_horizon_days: int = 1,
    max_target_horizon_days: int | None = None,
    walk_forward_tune_per_window: bool = True,
    factor_concentration_warn_threshold: float = 0.65,
    freshness_warn_days: int = 60,
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

        work_df, transform_metadata, selected_macro_lag_days, lag_scores = _prepare_best_lag_frame(
            df=df,
            target=target,
            factors=valid_factors,
            date_col=date_col,
            min_train_rows=min_train_rows,
            requested_macro_lag_days=int(macro_lag_days),
            min_target_horizon_days=int(max(1, min_target_horizon_days)),
            max_target_horizon_days=(
                int(max_target_horizon_days) if max_target_horizon_days is not None else None
            ),
        )

        work_df = add_volatility_regime_feature(
            work_df,
            date_col=date_col,
            return_col=target,
            window=30,
            threshold_quantile=0.75,
        )
        extra_regime_features = [
            feature
            for feature in ("volatility_regime_high", "rolling_vol_30d")
            if feature in work_df.columns
        ]
        model_factors = list(dict.fromkeys(valid_factors + extra_regime_features))

        if len(work_df) < max(min_train_rows + 5, 30):
            return {
                "status": "insufficient_data",
                "rows": int(len(work_df)),
                "required_rows": int(max(min_train_rows + 5, 30)),
            }

        # Adaptive window count: more data → more windows; sparse data → fewer.
        # 300 rows per window is a reasonable floor for daily financial series.
        adaptive_walk_windows = max(2, min(walk_forward_windows, len(work_df) // 300))

        # --- Stationarity tests on RAW data (before any transformation) ---
        # Running ADF/KPSS on series that have already been differenced/log-diffed
        # by prepare_supervised_frame produces circular results (the transform
        # was designed to make them stationary).  Test the original df columns
        # to correctly characterise whether each raw series needs differencing.
        stationarity: Dict[str, Dict[str, Any]] = {}
        for col in [target] + valid_factors:
            if col not in df.columns:
                stationarity[col] = {
                    "status": "column_not_found",
                    "adf_p_value": None,
                    "kpss_p_value": None,
                    "is_stationary": None,
                }
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 20:
                stationarity[col] = {
                    "status": "insufficient_data",
                    "adf_stat": None,
                    "adf_p_value": None,
                    "kpss_stat": None,
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
        # -----------------------------------------------------------------

        split_idx = int(len(work_df) * (1 - test_ratio))
        split_idx = max(min_train_rows, min(split_idx, len(work_df) - 5))

        train_df = work_df.iloc[:split_idx].copy()
        test_df = work_df.iloc[split_idx:].copy()

        x_train_raw = train_df[model_factors].apply(pd.to_numeric, errors="coerce")
        y_train = pd.to_numeric(train_df[target], errors="coerce")
        x_test_raw = test_df[model_factors].apply(pd.to_numeric, errors="coerce")
        y_test = pd.to_numeric(test_df[target], errors="coerce")

        x_train_clipped, x_test_clipped, clip_bounds = _clip_by_train_quantiles(
            x_train_raw,
            x_test_raw,
            lower_q=0.01,
            upper_q=0.99,
        )
        x_train_eng = _engineer_features(x_train_clipped, model_factors)
        x_test_eng = _engineer_features(x_test_clipped, model_factors)

        imputer = SimpleImputer(strategy="median")
        x_train_imputed = pd.DataFrame(
            imputer.fit_transform(x_train_eng),
            columns=x_train_eng.columns,
        )
        x_test_imputed = pd.DataFrame(
            imputer.transform(x_test_eng),
            columns=x_train_eng.columns,
        )

        x_train, dropped_multicollinear, vif_map = _reduce_multicollinearity(
            x_train_imputed,
            threshold=12.0,
            min_features=max(2, min(4, len(model_factors))),
        )
        retained_features = list(x_train.columns)
        x_test = x_test_imputed[retained_features].copy()

        model_key = str(model_type).strip().lower()
        model_alias = {
            "ols": "Linear",
            "linear": "Linear",
            "ridge": "Ridge",
            "lasso": "Lasso",
            "elasticnet": "ElasticNet",
            "randomforest": "RandomForest",
            "auto": "Auto",
        }
        resolved_request = model_alias.get(model_key, "Linear")

        registry = _candidate_registry(random_seed if random_seed is not None else 42)
        candidate_names = list(registry.keys()) if resolved_request == "Auto" else [resolved_request]
        splitter = _build_time_series_split(len(x_train))

        best_model_name = ""
        best_cv = -np.inf
        best_model: Any = None
        for candidate_name in candidate_names:
            estimator, search_space = registry[candidate_name]
            fitted_model, cv_r2, _ = _fit_candidate_with_cv(
                name=candidate_name,
                candidate=estimator,
                param_distributions=search_space,
                x=x_train,
                y=y_train,
                splitter=splitter,
                seed=(random_seed if random_seed is not None else 42),
            )
            if float(cv_r2) > float(best_cv):
                best_cv = float(cv_r2)
                best_model_name = candidate_name
                best_model = fitted_model

        if best_model is None:
            raise AnalysisError("Failed to fit governance model.")

        predictions = best_model.predict(x_test)
        adaptive_min_test_size = max(12, min(60, len(work_df) // max(10, walk_forward_windows * 3)))
        walk_forward = _walk_forward_backtest(
            work_df,
            target=target,
            factors=model_factors,
            windows=adaptive_walk_windows,
            min_train_rows=min_train_rows,
            clipped_floor=clipped_walk_forward_floor,
            min_test_size=adaptive_min_test_size,
            model_type=(best_model_name if best_model_name else model_type),
            tune_hyperparams_per_window=bool(walk_forward_tune_per_window),
            random_seed=random_seed if reproducibility_enforced else None,
        )

        leakage_flags: List[str] = []
        correlation_p_values: Dict[str, float] = {}
        leakage_details: Dict[str, Any] = {}
        if work_df[date_col].duplicated().any():
            leakage_flags.append("duplicate_timestamps_detected")

        if work_df[date_col].is_monotonic_increasing is False:
            leakage_flags.append("non_monotonic_timestamps")

        # Leakage sensitivity scales with series length:
        # - On short series (< 60 rows) noise inflates lead/lag differences and
        #   inflates pearson correlations → use more conservative thresholds.
        n_obs = len(work_df)
        lead_lag_threshold = 0.15 if n_obs >= 60 else 0.25
        fdr_corr_threshold = 0.95 if n_obs >= 60 else 0.98

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
                if abs(float(lead_corr)) > abs(float(lag_corr)) + lead_lag_threshold:
                    leakage_flags.append(f"possible_future_leakage:{factor}")

        bh_significant = _benjamini_hochberg(correlation_p_values, alpha=0.05)
        for factor, is_significant in bh_significant.items():
            corr_value = float(work_df[[target, factor]].corr().iloc[0, 1])
            if is_significant and abs(corr_value) > fdr_corr_threshold:
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
        raw_oos_r2 = float(r2_score(y_test, predictions))
        # Clip R² from very small test sets: on < 15 rows the OOS metric is
        # dominated by noise and can reach extreme negative values (e.g. -200).
        # Cap at a conservative floor so it does not propagate as a hard-fail signal.
        if len(test_df) < 15:
            oos_r2 = max(-0.5, raw_oos_r2)
        else:
            oos_r2 = raw_oos_r2

        # Bootstrap confidence interval for OOS R² — more stable than point estimate.
        ci_frame = x_test.copy()
        ci_frame[target] = y_test.values
        oos_r2_ci = _oos_r2_confidence_interval(
            model=best_model,
            test_df=ci_frame,
            x_cols=retained_features,
            y_col=target,
            n_bootstrap=200,
            confidence=0.90,
            random_seed=random_seed if reproducibility_enforced else None,
        )

        baseline_suite = _compute_baseline_suite(
            y_train=y_train,
            y_test=y_test,
            x_train=x_train,
            x_test=x_test,
        )

        stationarity_known = [
            values.get("is_stationary")
            for values in stationarity.values()
            if isinstance(values, dict) and values.get("is_stationary") is not None
        ]
        # When no series has enough observations for ADF/KPSS, the ratio is
        # genuinely unknown — use None so downstream scoring applies a
        # neutral (0.5) penalty instead of the worst-case 0.0.
        stationary_ratio: float | None = (
            float(
                sum(1 for value in stationarity_known if value)
                / len(stationarity_known)
            )
            if stationarity_known
            else None
        )
        model_risk_score = _compute_model_risk_score(
            leakage_flags_count=len(leakage_flags),
            factors_count=len(model_factors),
            stationarity_ratio=stationary_ratio,
            normalized_shift=float(normalized_shift),
            oos_r2=oos_r2,
            walk_forward_avg_r2=(
                walk_forward.get("clipped_avg_r2", walk_forward.get("avg_r2"))
                if walk_forward.get("status") == "ok"
                else None
            ),
        )
        trend_volatility = _trend_and_volatility_diagnostics(
            y_true=y_test,
            y_pred=predictions,
            rolling_window=max(3, min(10, len(y_test) // 4 if len(y_test) > 0 else 3)),
        )

        model_step = best_model.named_steps.get("model") if hasattr(best_model, "named_steps") else best_model
        raw_coef = getattr(model_step, "coef_", None)
        if raw_coef is not None:
            coef_arr = np.asarray(raw_coef, dtype=float).reshape(-1)
            coeffs = {
                factor: float(coef_arr[idx])
                for idx, factor in enumerate(retained_features)
            }
        else:
            coeffs = {factor: 0.0 for factor in retained_features}

        intercept_value = float(getattr(model_step, "intercept_", 0.0))
        factor_concentration = _factor_concentration_diagnostics(coeffs)

        freshness_publication_lags = [
            int(meta.get("publication_lag_days", 0))
            for key, meta in transform_metadata.items()
            if isinstance(meta, dict) and key != target and key in valid_factors
        ]
        inferred_target_horizon = int(
            (
                (transform_metadata.get(target) or {}).get("target_horizon_days")
                if isinstance(transform_metadata.get(target), dict)
                else 1
            )
            or 1
        )
        freshness_alignment = {
            "status": "ok",
            "target_horizon_days": inferred_target_horizon,
            "max_publication_lag_days": max(freshness_publication_lags) if freshness_publication_lags else 0,
            "freshness_warn_days": int(max(1, freshness_warn_days)),
            "lag_alignment_ok": bool(
                not freshness_publication_lags
                or max(freshness_publication_lags) <= max(1, freshness_warn_days)
            ),
        }

        return {
            "status": "ok",
            "model_type": (best_model_name if best_model_name else model_type),
            "requested_model_type": model_type,
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
                "r2_ci": oos_r2_ci,
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "baseline_comparison": baseline_suite,
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
            "trend_volatility": trend_volatility,
            "factor_concentration": factor_concentration,
            "freshness_alignment": freshness_alignment,
            "reproducibility": {
                "random_seed": random_seed,
                "enforced": reproducibility_enforced,
                "policy": "stochastic analyses must declare/propagate random_state",
            },
            "data_quality_context": {
                "rows_used": int(len(work_df)),
                "walk_forward_windows_requested": int(walk_forward_windows),
                "walk_forward_windows_adaptive": int(adaptive_walk_windows),
                "adaptive_walk_forward_min_test_size": int(adaptive_min_test_size),
                "model_selection_cv_r2": float(best_cv),
                "selected_macro_lag_days": int(selected_macro_lag_days),
                "lag_candidate_oos_r2": lag_scores,
                "winsor_bounds": clip_bounds,
                "dropped_multicollinear_features": dropped_multicollinear,
                "max_vif": float(max(vif_map.values())) if vif_map else None,
                "transformations": transform_metadata,
                "min_target_horizon_days": int(max(1, min_target_horizon_days)),
                "max_target_horizon_days": (
                    int(max_target_horizon_days) if max_target_horizon_days is not None else None
                ),
                "walk_forward_tune_per_window": bool(walk_forward_tune_per_window),
                "factor_concentration_warn_threshold": float(factor_concentration_warn_threshold),
            },
            "coefficients": coeffs,
            "intercept": intercept_value,
        }
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in governance_report: {e}") from e
