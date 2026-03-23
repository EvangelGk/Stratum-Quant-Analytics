from typing import Any, Dict, List, Optional
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from exceptions.MedallionExceptions import AnalysisError, DataValidationError
from .mixed_frequency import aggregate_source_importance, prepare_supervised_frame


def _winsorize_dataframe(
    df: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[pd.DataFrame, int]:
    clipped = df.copy()
    clipped_points = 0
    for column in clipped.columns:
        series = pd.to_numeric(clipped[column], errors="coerce")
        if series.dropna().empty:
            continue
        lo = float(series.quantile(lower_q))
        hi = float(series.quantile(upper_q))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        outlier_mask = (series < lo) | (series > hi)
        clipped_points += int(outlier_mask.sum())
        clipped[column] = series.clip(lower=lo, upper=hi)
    return clipped, clipped_points


def _engineer_features(x: pd.DataFrame, base_factors: List[str]) -> pd.DataFrame:
    engineered = x.copy()
    usable_factors = [f for f in base_factors if f in engineered.columns]

    for factor in usable_factors:
        series = pd.to_numeric(engineered[factor], errors="coerce")
        if series.nunique(dropna=True) > 2:
            engineered[f"{factor}__sq"] = np.square(series)

    interaction_candidates = usable_factors[: min(len(usable_factors), 4)]
    for i in range(len(interaction_candidates)):
        for j in range(i + 1, len(interaction_candidates)):
            left = interaction_candidates[i]
            right = interaction_candidates[j]
            engineered[f"{left}__x__{right}"] = (
                pd.to_numeric(engineered[left], errors="coerce")
                * pd.to_numeric(engineered[right], errors="coerce")
            )
    return engineered


def _vif_values(x: pd.DataFrame) -> Dict[str, float]:
    if x.shape[1] < 2:
        return {column: 1.0 for column in x.columns}
    clean = x.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or clean.shape[1] < 2:
        return {column: 1.0 for column in x.columns}
    values = clean.to_numpy(dtype=float)
    vif_map: Dict[str, float] = {}
    for idx, column in enumerate(clean.columns):
        try:
            vif_map[column] = float(variance_inflation_factor(values, idx))
        except Exception:
            vif_map[column] = float("inf")
    return vif_map


def _reduce_multicollinearity(
    x: pd.DataFrame,
    threshold: float = 12.0,
    min_features: int = 2,
) -> tuple[pd.DataFrame, List[str], Dict[str, float]]:
    reduced = x.copy()
    dropped: List[str] = []

    while reduced.shape[1] > min_features:
        current_vif = _vif_values(reduced)
        if not current_vif:
            break
        feature, max_vif = max(current_vif.items(), key=lambda item: float(item[1]))
        if np.isnan(max_vif) or float(max_vif) <= threshold:
            break
        reduced = reduced.drop(columns=[feature])
        dropped.append(feature)

    return reduced, dropped, _vif_values(reduced)


def _build_time_series_split(n_rows: int) -> TimeSeriesSplit:
    split_count = max(2, min(5, n_rows // 60))
    return TimeSeriesSplit(n_splits=split_count)


def _candidate_registry(seed: int) -> Dict[str, tuple[Pipeline, Dict[str, Any] | None]]:
    linear_pipe = lambda estimator: Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    tree_pipe = lambda estimator: Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )

    return {
        "Linear": (linear_pipe(LinearRegression()), None),
        "Ridge": (
            linear_pipe(Ridge(random_state=seed)),
            {"model__alpha": np.logspace(-3, 2, 40)},
        ),
        "Lasso": (
            linear_pipe(Lasso(random_state=seed, max_iter=5000)),
            {"model__alpha": np.logspace(-4, 0, 40)},
        ),
        "ElasticNet": (
            linear_pipe(ElasticNet(random_state=seed, max_iter=5000)),
            {
                "model__alpha": np.logspace(-4, 0, 30),
                "model__l1_ratio": np.linspace(0.1, 0.9, 9),
            },
        ),
        "RandomForest": (
            tree_pipe(
                RandomForestRegressor(
                    random_state=seed,
                    n_jobs=-1,
                )
            ),
            {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", 0.6],
            },
        ),
    }


def _fit_candidate_with_cv(
    name: str,
    candidate: Pipeline,
    param_distributions: Dict[str, Any] | None,
    x: pd.DataFrame,
    y: pd.Series,
    splitter: TimeSeriesSplit,
    seed: int,
) -> tuple[Pipeline, float, Dict[str, Any]]:
    if param_distributions:
        tuner = RandomizedSearchCV(
            estimator=candidate,
            param_distributions=param_distributions,
            n_iter=12,
            scoring="r2",
            cv=splitter,
            n_jobs=-1,
            random_state=seed,
            refit=True,
        )
        tuner.fit(x, y)
        return (
            tuner.best_estimator_,
            float(tuner.best_score_),
            {
                "method": "randomized_search",
                "best_params": tuner.best_params_,
            },
        )

    scores = cross_val_score(candidate, x, y, cv=splitter, scoring="r2", n_jobs=-1)
    candidate.fit(x, y)
    return (
        candidate,
        float(np.mean(scores)),
        {
            "method": "fixed_params",
            "best_params": {},
        },
    )


def _quick_cv_r2(panel: pd.DataFrame, target: str, features: List[str]) -> float:
    if panel.empty or len(panel) < max(40, len(features) * 10):
        return float("-inf")
    y = pd.to_numeric(panel[target], errors="coerce")
    x = panel[features].apply(pd.to_numeric, errors="coerce")
    valid = y.notna()
    y = y.loc[valid]
    x = x.loc[valid]
    if len(x) < 30:
        return float("-inf")
    splitter = _build_time_series_split(len(x))
    probe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=0.5, random_state=42)),
        ]
    )
    try:
        scores = cross_val_score(probe, x, y, cv=splitter, scoring="r2", n_jobs=-1)
        return float(np.mean(scores))
    except Exception:
        return float("-inf")


def _select_best_macro_lag(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    ticker: Optional[str],
    requested_lag_days: int,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, Any]], int, Dict[str, float]]:
    if int(requested_lag_days) > 0:
        panel, metadata = prepare_supervised_frame(
            df=df,
            target=target,
            features=features,
            ticker=ticker,
            macro_lag_days=int(requested_lag_days),
        )
        return panel, metadata, int(requested_lag_days), {str(int(requested_lag_days)): float("nan")}

    lag_candidates = [0, 15, 30, 45]
    scored: Dict[str, float] = {}
    best_lag = lag_candidates[0]
    best_score = float("-inf")
    best_panel = pd.DataFrame()
    best_meta: Dict[str, Dict[str, Any]] = {}

    for lag in lag_candidates:
        panel, metadata = prepare_supervised_frame(
            df=df,
            target=target,
            features=features,
            ticker=ticker,
            macro_lag_days=lag,
        )
        score = _quick_cv_r2(panel, target=target, features=features)
        scored[str(lag)] = float(score)
        if score > best_score:
            best_score = float(score)
            best_lag = int(lag)
            best_panel = panel
            best_meta = metadata

    if best_panel.empty:
        best_panel, best_meta = prepare_supervised_frame(
            df=df,
            target=target,
            features=features,
            ticker=ticker,
            macro_lag_days=0,
        )
        best_lag = 0
    return best_panel, best_meta, best_lag, scored


def _feature_search_space(
    features: List[str],
    max_features_considered: int,
    max_evaluations: int,
    seed: int,
) -> List[List[str]]:
    ordered = list(dict.fromkeys(features))[: max_features_considered]
    n = len(ordered)
    if n <= 1:
        return [ordered]

    candidates: List[List[str]] = []
    # Always evaluate full and near-full sets.
    candidates.append(list(ordered))
    if n >= 2:
        candidates.append(list(ordered[: max(2, n - 1)]))

    # Exhaustive search for small sets, stochastic capped search for larger sets.
    if n <= 8:
        for k in range(1, n + 1):
            for comb in combinations(ordered, k):
                candidates.append(list(comb))
    else:
        rng = np.random.default_rng(seed)
        candidates.append([ordered[0]])
        candidates.append(ordered[:2])
        candidates.append(ordered[:3])
        while len(candidates) < max_evaluations:
            k = int(rng.integers(1, min(8, n) + 1))
            idx = sorted(rng.choice(n, size=k, replace=False).tolist())
            candidates.append([ordered[i] for i in idx])

    # Deduplicate while preserving order.
    seen: set[tuple[str, ...]] = set()
    unique: List[List[str]] = []
    for item in candidates:
        key = tuple(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= max_evaluations:
            break
    return unique


def _select_feature_subset(
    panel: pd.DataFrame,
    target: str,
    features: List[str],
    seed: int = 42,
    max_features_considered: int = 12,
    max_evaluations: int = 120,
) -> tuple[List[str], Dict[str, Any]]:
    valid_features = [f for f in features if f in panel.columns and f != target]
    if not valid_features:
        return [], {
            "status": "no_valid_features",
            "evaluated_subsets": 0,
            "selected_subset": [],
            "best_cv_r2": None,
        }

    # Prioritize features with signal to keep search tractable on large universes.
    scored = []
    y = pd.to_numeric(panel[target], errors="coerce")
    for feature in valid_features:
        x = pd.to_numeric(panel[feature], errors="coerce")
        corr = y.corr(x)
        scored.append((feature, abs(float(corr)) if pd.notna(corr) else 0.0))
    ranked = [name for name, _ in sorted(scored, key=lambda item: item[1], reverse=True)]

    subset_pool = _feature_search_space(
        features=ranked,
        max_features_considered=max_features_considered,
        max_evaluations=max_evaluations,
        seed=seed,
    )

    best_subset = subset_pool[0]
    best_score = float("-inf")
    subset_scores: List[Dict[str, Any]] = []
    for subset in subset_pool:
        cv_r2 = _quick_cv_r2(panel, target=target, features=subset)
        subset_scores.append(
            {
                "subset": subset,
                "cv_r2": float(cv_r2),
                "n_features": int(len(subset)),
            }
        )
        # Prefer higher score; on ties prefer the smaller subset (lower overfit risk).
        if (cv_r2 > best_score) or (
            np.isclose(cv_r2, best_score, equal_nan=False) and len(subset) < len(best_subset)
        ):
            best_score = float(cv_r2)
            best_subset = list(subset)

    subset_scores_sorted = sorted(
        subset_scores,
        key=lambda item: float(item["cv_r2"]),
        reverse=True,
    )
    return best_subset, {
        "status": "ok",
        "evaluated_subsets": int(len(subset_pool)),
        "selected_subset": list(best_subset),
        "best_cv_r2": float(best_score),
        "top_subsets": subset_scores_sorted[:15],
    }


def sensitivity_reg(
    df: pd.DataFrame,
    target: str = "log_return",
    factors: Optional[List[str]] = None,
    model: str = "OLS",
    ticker: Optional[str] = None,
    macro_lag_days: int = 0,
) -> Any:
    """Run multivariate macro sensitivity regression on equity log-returns.

    Estimates the linear relationship between equity log-returns and a
    set of macro factors via:

        log_return = α + β₁·inflation + β₂·energy_index + … + ε

    Solvers supported:

    - **OLS** (``statsmodels``): Unbiased BLUE estimator under Gauss-Markov
      assumptions.  Returns a full ``Summary`` object with t-stats, p-values,
      R², and confidence intervals — the standard output for a risk
      attribution report.
        - **Ridge** (``scikit-learn``): L2-regularised estimator that shrinks
      coefficients toward zero, reducing variance at the cost of slight bias.
      Preferred when macro factors are collinear (e.g. inflation & energy
      prices are highly correlated).  Returns a plain coefficient dict.
        - **Auto**: Runs a time-series CV model tournament over linear and
            regularised/tree candidates and keeps the best out-of-sample R² model.

    Args:
        df: Master table ``DataFrame`` produced by ``GoldLayer``.  Must
            contain ``target`` and all columns listed in ``factors``.
        target: Dependent variable column, default ``'log_return'``.
        factors: List of independent macro-factor column names.  Defaults
            to ``['inflation', 'energy_index']``.
        model: Regression solver — ``'OLS'``, ``'Ridge'``, ``'Lasso'``,
            ``'ElasticNet'``, ``'RandomForest'``, or ``'Auto'``.

    Returns:
        - ``'OLS'``: A ``statsmodels`` ``Summary`` object (printable).
                - Non-OLS models: ``dict`` with coefficients/intercept where available,
                    plus CV diagnostics and preprocessing audit metadata.
        - ``None`` is never returned; specific exceptions are raised.

    Raises:
        DataValidationError: If ``target``, any factor column, or an
            invalid model name is provided.
        AnalysisError: On any fitting or runtime failure.
    """
    try:
        factors = factors or ["inflation", "energy_index"]
        model_key = str(model).strip()
        model_normalized = model_key.lower()
        model_alias = {
            "ols": "OLS",
            "ridge": "Ridge",
            "lasso": "Lasso",
            "elasticnet": "ElasticNet",
            "randomforest": "RandomForest",
            "auto": "Auto",
        }
        if model_normalized not in model_alias:
            raise DataValidationError(
                "Model must be one of: 'OLS', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'Auto'."
            )
        selected_model = model_alias[model_normalized]

        if target not in df.columns:
            raise DataValidationError(f"Target column {target} not found in DataFrame.")

        missing_factors = [f for f in factors if f not in df.columns]
        if missing_factors:
            raise DataValidationError(
                f"Factor columns {missing_factors} not found in DataFrame."
            )

        panel, metadata, selected_macro_lag_days, lag_scores = _select_best_macro_lag(
            df=df,
            target=target,
            features=factors,
            ticker=ticker,
            requested_lag_days=int(macro_lag_days),
        )
        if len(panel) < max(30, len(factors) * 8):
            raise DataValidationError("Insufficient rows after stationarity transforms.")

        selected_factors, subset_search = _select_feature_subset(
            panel=panel,
            target=target,
            features=factors,
            seed=42,
            max_features_considered=12,
            max_evaluations=120,
        )
        if not selected_factors:
            raise DataValidationError("No valid feature subset found for regression.")

        y = pd.to_numeric(panel[target], errors="coerce")
        x_raw = panel[selected_factors].apply(pd.to_numeric, errors="coerce")

        missing_before = {
            "rows_before_dropna": int(len(panel)),
            "target_missing": int(y.isna().sum()),
            "feature_missing": {
                factor: int(x_raw[factor].isna().sum()) for factor in x_raw.columns
            },
        }

        valid_mask = y.notna()
        y = y.loc[valid_mask].reset_index(drop=True)
        x_raw = x_raw.loc[valid_mask].reset_index(drop=True)

        x_winsor, clipped_points = _winsorize_dataframe(x_raw)
        x_engineered = _engineer_features(x_winsor, selected_factors)
        x_engineered = x_engineered.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median")
        x_imputed = pd.DataFrame(
            imputer.fit_transform(x_engineered),
            columns=x_engineered.columns,
        )

        x_final, dropped_multicollinear, vif_map = _reduce_multicollinearity(
            x_imputed,
            threshold=12.0,
            min_features=max(2, min(4, len(selected_factors))),
        )
        missing_after = {
            "target_missing": int(y.isna().sum()),
            "feature_missing_after_impute": int(x_final.isna().sum().sum()),
        }

        splitter = _build_time_series_split(len(x_final))
        seed = 42

        if selected_model == "OLS":
            design = sm.add_constant(x_final)
            fitted_model = sm.OLS(y, design).fit()
            coefficients = {
                factor: float(fitted_model.params.get(factor, 0.0)) for factor in x_final.columns
            }
            raw_importance = {
                factor: float(abs(coefficients[factor]) * x_final[factor].std())
                for factor in x_final.columns
            }
            cv_model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            )
            cv_scores = cross_val_score(cv_model, x_final, y, cv=splitter, scoring="r2", n_jobs=-1)
            return {
                "model": "OLS",
                "ticker": ticker,
                "target": target,
                # Coefficients express impact over the forward horizon used by
                # prepare_supervised_frame.  Check target_horizon_days before
                # interpreting a coefficient as an immediate point-in-time effect.
                "target_horizon_days": int(
                    metadata.get(target, {}).get("target_horizon_days", 1)
                ),
                "coefficients": coefficients,
                "intercept": float(fitted_model.params.get("const", 0.0)),
                "p_values": {
                    factor: float(fitted_model.pvalues.get(factor, np.nan))
                    for factor in x_final.columns
                },
                "r2": float(fitted_model.rsquared),
                "adj_r2": float(fitted_model.rsquared_adj),
                "cv_r2": float(np.mean(cv_scores)),
                "n_obs": int(len(y)),
                "summary_text": fitted_model.summary().as_text(),
                "feature_importance": raw_importance,
                "source_importance": aggregate_source_importance(raw_importance),
                "transformations": metadata,
                "lag_selection": {
                    "selected_macro_lag_days": int(selected_macro_lag_days),
                    "candidate_cv_r2": lag_scores,
                },
                "feature_subset_search": subset_search,
                "data_inspection": {
                    "missing_values_before": missing_before,
                    "missing_values_after": missing_after,
                    "outliers_winsorized": int(clipped_points),
                },
                "feature_engineering": {
                    "base_factor_count": int(len(selected_factors)),
                    "engineered_feature_count": int(max(0, x_engineered.shape[1] - len(selected_factors))),
                    "retained_feature_count": int(x_final.shape[1]),
                    "retained_features": list(x_final.columns),
                },
                "multicollinearity": {
                    "dropped_features": dropped_multicollinear,
                    "max_vif": (float(max(vif_map.values())) if vif_map else None),
                    "vif": {k: float(v) for k, v in vif_map.items()},
                },
                "validation": {
                    "scheme": "time_series_split",
                    "splits": int(splitter.n_splits),
                },
            }

        registry = _candidate_registry(seed)
        if selected_model == "Auto":
            candidate_names = list(registry.keys())
        elif selected_model in registry:
            candidate_names = [selected_model]
        else:
            raise DataValidationError(
                "Model must be one of: 'OLS', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'Auto'."
            )

        candidate_scores: List[Dict[str, Any]] = []
        best_model_name = ""
        best_cv = -np.inf
        best_estimator: Pipeline | None = None
        best_tuning: Dict[str, Any] = {}

        for candidate_name in candidate_names:
            estimator, search_space = registry[candidate_name]
            fitted_estimator, cv_r2, tuning_info = _fit_candidate_with_cv(
                name=candidate_name,
                candidate=estimator,
                param_distributions=search_space,
                x=x_final,
                y=y,
                splitter=splitter,
                seed=seed,
            )
            candidate_scores.append(
                {
                    "model": candidate_name,
                    "cv_r2": float(cv_r2),
                    "tuning": tuning_info,
                }
            )
            if cv_r2 > best_cv:
                best_cv = float(cv_r2)
                best_model_name = candidate_name
                best_estimator = fitted_estimator
                best_tuning = tuning_info

        if best_estimator is None:
            raise AnalysisError("Model fitting failed: no estimator selected.")

        predictions = best_estimator.predict(x_final)
        model_step = best_estimator.named_steps.get("model")
        raw_coef = getattr(model_step, "coef_", None)
        if raw_coef is not None:
            coef_array = np.asarray(raw_coef, dtype=float).reshape(-1)
            coefficients = {
                factor: float(coef_array[idx]) for idx, factor in enumerate(x_final.columns)
            }
        else:
            coefficients = {factor: 0.0 for factor in x_final.columns}

        intercept_value = float(getattr(model_step, "intercept_", 0.0))

        raw_importance = {
            factor: float(abs(coefficients[factor]) * x_final[factor].std())
            for factor in x_final.columns
        }
        if all(abs(value) < 1e-12 for value in raw_importance.values()) and hasattr(model_step, "feature_importances_"):
            imp = np.asarray(getattr(model_step, "feature_importances_"), dtype=float)
            raw_importance = {
                factor: float(imp[idx]) for idx, factor in enumerate(x_final.columns)
            }

        return {
            "model": best_model_name,
            "ticker": ticker,
            "target": target,
            "target_horizon_days": int(
                metadata.get(target, {}).get("target_horizon_days", 1)
            ),
            "coefficients": coefficients,
            "intercept": intercept_value,
            "r2": float(best_estimator.score(x_final, y)),
            "cv_r2": float(best_cv),
            "n_obs": int(len(y)),
            "feature_importance": raw_importance,
            "source_importance": aggregate_source_importance(raw_importance),
            "transformations": metadata,
            "lag_selection": {
                "selected_macro_lag_days": int(selected_macro_lag_days),
                "candidate_cv_r2": lag_scores,
            },
            "feature_subset_search": subset_search,
            "candidate_scores": sorted(
                candidate_scores,
                key=lambda item: float(item["cv_r2"]),
                reverse=True,
            ),
            "tuning": best_tuning,
            "data_inspection": {
                "missing_values_before": missing_before,
                "missing_values_after": missing_after,
                "outliers_winsorized": int(clipped_points),
            },
            "feature_engineering": {
                "base_factor_count": int(len(selected_factors)),
                "engineered_feature_count": int(max(0, x_engineered.shape[1] - len(selected_factors))),
                "retained_feature_count": int(x_final.shape[1]),
                "retained_features": list(x_final.columns),
            },
            "multicollinearity": {
                "dropped_features": dropped_multicollinear,
                "max_vif": (float(max(vif_map.values())) if vif_map else None),
                "vif": {k: float(v) for k, v in vif_map.items()},
            },
            "validation": {
                "scheme": "time_series_split",
                "splits": int(splitter.n_splits),
                "cv_metric": "r2",
                "mae_in_sample": float(np.mean(np.abs(y.values - predictions))),
            },
        }
    except DataValidationError:
        raise
    except Exception as e:
        raise AnalysisError(f"Unexpected error in sensitivity_reg: {e}") from e
