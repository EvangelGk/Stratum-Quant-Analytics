from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


YFINANCE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "log_return",
    "ticker",
}

WORLDBANK_COLUMNS = {
    "gdp_growth",
    "energy_usage",
    "inflation_wb",
    "unemployment_wb",
    "trade_openness",
}

RATE_LIKE_KEYWORDS = (
    "rate",
    "yield",
    "inflation",
    "unemployment",
    "growth",
    "sentiment",
    "spread",
)

SOURCE_HORIZON_DAYS = {
    "yfinance": 1,
    "fred": 21,
    "worldbank": 252,
}

MANDATORY_PUBLICATION_LAG_DAYS = 45


def infer_source(feature_name: str) -> str:
    feature = str(feature_name).strip().lower()
    if feature in YFINANCE_COLUMNS:
        return "yfinance"
    if feature in WORLDBANK_COLUMNS or feature.endswith("_wb"):
        return "worldbank"
    return "fred"


def is_return_like(feature_name: str) -> bool:
    feature = str(feature_name).strip().lower()
    return any(token in feature for token in ("return", "pct_change", "diff"))


def is_rate_like(feature_name: str) -> bool:
    feature = str(feature_name).strip().lower()
    return any(token in feature for token in RATE_LIKE_KEYWORDS)


def filter_to_ticker(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    work_df = df.copy()
    if ticker and "ticker" in work_df.columns:
        scoped = work_df[work_df["ticker"].astype(str) == str(ticker)].copy()
        if not scoped.empty:
            return scoped
    return work_df


def source_horizon_days(feature_name: str) -> int:
    return int(SOURCE_HORIZON_DAYS.get(infer_source(feature_name), 1))


def stationary_transform(
    series: pd.Series,
    feature_name: str,
    keep_as_is: bool = False,
) -> Tuple[pd.Series, str]:
    clean = pd.to_numeric(series, errors="coerce")
    if keep_as_is or is_return_like(feature_name):
        transformed = clean.replace([np.inf, -np.inf], np.nan)
        return transformed, "as_is"

    if is_rate_like(feature_name):
        transformed = clean.diff()
        return transformed.replace([np.inf, -np.inf], np.nan), "diff"

    non_null = clean.dropna()
    if not non_null.empty and (non_null > 0).all():
        transformed = np.log(clean).diff()
        return transformed.replace([np.inf, -np.inf], np.nan), "log_diff"

    transformed = clean.pct_change()
    return transformed.replace([np.inf, -np.inf], np.nan), "pct_change"


def _interpolate_feature(
    series: pd.Series,
    method: str = "linear",
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric
    try:
        interpolated = numeric.interpolate(method=method, limit_direction="forward")
    except Exception:
        interpolated = numeric
    return interpolated.ffill()


def add_volatility_regime_feature(
    df: pd.DataFrame,
    date_col: str = "date",
    return_col: str = "log_return",
    window: int = 30,
    threshold_quantile: float = 0.75,
) -> pd.DataFrame:
    """Add binary volatility-regime feature for stable vs crisis conditions."""
    work_df = df.copy()
    if return_col not in work_df.columns:
        work_df["volatility_regime_label"] = "unknown"
        work_df["volatility_regime_high"] = 0
        return work_df

    if date_col in work_df.columns:
        work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
        work_df = work_df.sort_values(date_col)

    series = pd.to_numeric(work_df[return_col], errors="coerce")
    rolling_vol = series.rolling(window=max(5, int(window)), min_periods=max(5, int(window) // 2)).std()
    vol_threshold = rolling_vol.quantile(float(threshold_quantile))
    if pd.isna(vol_threshold):
        vol_threshold = rolling_vol.dropna().median()

    high_regime = (rolling_vol >= vol_threshold).fillna(False)
    work_df["volatility_regime_label"] = np.where(high_regime, "high_volatility", "low_volatility")
    work_df["volatility_regime_high"] = high_regime.astype(int)
    work_df["rolling_vol_30d"] = rolling_vol
    return work_df


def _release_event_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    prev = numeric.ffill().shift(1)
    changed = (numeric.notna()) & (prev.isna() | (~np.isclose(numeric, prev, equal_nan=True)))
    return changed.fillna(False)


def _transform_low_frequency_series(series: pd.Series, feature_name: str) -> Tuple[pd.Series, str]:
    numeric = pd.to_numeric(series, errors="coerce")
    release_mask = _release_event_mask(numeric)
    release_values = numeric.loc[release_mask]
    if release_values.empty:
        return numeric * np.nan, "no_release_events"

    transformed_release, method = stationary_transform(release_values, feature_name)
    expanded = transformed_release.reindex(numeric.index).ffill()
    return expanded.replace([np.inf, -np.inf], np.nan), f"release_{method}"


def _future_target_from_series(
    series: pd.Series,
    feature_name: str,
    horizon_days: int,
) -> Tuple[pd.Series, str]:
    numeric = pd.to_numeric(series, errors="coerce")
    horizon = max(1, int(horizon_days))
    if horizon == 1:
        transformed, method = stationary_transform(
            numeric,
            feature_name,
            keep_as_is=is_return_like(feature_name),
        )
        return transformed, method

    if is_return_like(feature_name):
        future = (
            numeric.shift(-1)
            .rolling(window=horizon, min_periods=horizon)
            .sum()
            .shift(-(horizon - 1))
        )
        return future.replace([np.inf, -np.inf], np.nan), f"forward_{horizon}d_cumulative_return"

    positive = numeric.dropna()
    if not positive.empty and (positive > 0).all():
        future = np.log(numeric.shift(-horizon) / numeric)
        return future.replace([np.inf, -np.inf], np.nan), f"forward_{horizon}d_log_return"

    future = numeric.shift(-horizon) - numeric
    return future.replace([np.inf, -np.inf], np.nan), f"forward_{horizon}d_diff"


def resolve_target_horizon(
    features: Sequence[str],
    min_horizon_days: int = 1,
    max_horizon_days: Optional[int] = None,
) -> int:
    if not features:
        base_horizon = 1
    else:
        base_horizon = max(source_horizon_days(feature) for feature in features)

    clamped = max(1, int(min_horizon_days), int(base_horizon))
    if max_horizon_days is not None:
        clamped = min(clamped, max(1, int(max_horizon_days)))
    return int(clamped)


def build_stationary_panel(
    df: pd.DataFrame,
    columns: Sequence[str],
    date_col: str = "date",
    ticker: Optional[str] = None,
    macro_lag_days: int = 0,
    interpolation_method: str = "linear",
    keep_target_as_is: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    keep_target_as_is = keep_target_as_is or []
    required = [date_col] + [col for col in columns if col in df.columns]
    work_df = filter_to_ticker(df, ticker=ticker)
    work_df = work_df[required].copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
    work_df = work_df.dropna(subset=[date_col]).sort_values(date_col)

    transformed: Dict[str, pd.Series] = {date_col: work_df[date_col]}
    metadata: Dict[str, Dict[str, Any]] = {}

    for column in columns:
        if column not in work_df.columns:
            continue

        source = infer_source(column)
        base_series = pd.to_numeric(work_df[column], errors="coerce")
        lag_days = 0
        if source == "yfinance":
            if interpolation_method == "linear":
                base_series = _interpolate_feature(base_series, method=interpolation_method)
            transformed_series, method = stationary_transform(
                base_series,
                column,
                keep_as_is=column in keep_target_as_is,
            )
        else:
            # Point-in-time guardrail: macro values become visible only after
            # publication delay. Forward fill is intentionally applied AFTER lag.
            lag_days = max(MANDATORY_PUBLICATION_LAG_DAYS, int(macro_lag_days))
            lagged_series = base_series.shift(lag_days)
            lagged_series = lagged_series.ffill()
            transformed_series, method = _transform_low_frequency_series(
                lagged_series,
                column,
            )
        transformed[column] = transformed_series
        metadata[column] = {
            "source": source,
            "lag_days": int(lag_days),
            "transformation": method,
            "native_horizon_days": source_horizon_days(column),
            "publication_lag_days": int(lag_days) if source != "yfinance" else 0,
        }

    panel = pd.DataFrame(transformed)
    usable_columns = [column for column in columns if column in panel.columns]
    if usable_columns:
        panel = panel.dropna(subset=usable_columns)
    return panel.reset_index(drop=True), metadata


def prepare_supervised_frame(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    date_col: str = "date",
    ticker: Optional[str] = None,
    macro_lag_days: int = 30,
    interpolation_method: str = "linear",
    align_target_to_features: bool = True,
    min_target_horizon_days: int = 1,
    max_target_horizon_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    work_df = filter_to_ticker(df, ticker=ticker).copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
    work_df = work_df.dropna(subset=[date_col]).sort_values(date_col)

    target_horizon = (
        resolve_target_horizon(
            features,
            min_horizon_days=min_target_horizon_days,
            max_horizon_days=max_target_horizon_days,
        )
        if align_target_to_features
        else 1
    )
    target_series, target_method = _future_target_from_series(
        work_df[target],
        target,
        target_horizon,
    )
    target_df = pd.DataFrame({date_col: work_df[date_col], target: target_series})

    panel, metadata = build_stationary_panel(
        df=df,
        columns=list(features),
        date_col=date_col,
        ticker=ticker,
        macro_lag_days=macro_lag_days,
        interpolation_method=interpolation_method,
        keep_target_as_is=[],
    )
    panel = target_df.merge(panel, on=date_col, how="left")
    metadata[target] = {
        "source": infer_source(target),
        "lag_days": 0,
        "transformation": target_method,
        "native_horizon_days": target_horizon,
        "target_horizon_days": target_horizon,
        "target_horizon_policy": {
            "align_to_features": bool(align_target_to_features),
            "min_target_horizon_days": int(max(1, min_target_horizon_days)),
            "max_target_horizon_days": (
                int(max_target_horizon_days) if max_target_horizon_days is not None else None
            ),
        },
    }
    keep_columns = [date_col, target, *features]
    keep_columns = [column for column in keep_columns if column in panel.columns]
    return panel[keep_columns].dropna().reset_index(drop=True), metadata


def aggregate_source_importance(feature_importance: Dict[str, float]) -> list[Dict[str, float | str]]:
    grouped: Dict[str, float] = {"yfinance": 0.0, "fred": 0.0, "worldbank": 0.0}
    for feature, importance in feature_importance.items():
        grouped[infer_source(feature)] += float(abs(importance))

    total = sum(grouped.values()) or 1.0
    ranked = [
        {
            "source": source,
            "importance": round(value, 6),
            "share": round(value / total, 6),
        }
        for source, value in grouped.items()
    ]
    return sorted(ranked, key=lambda item: float(item["importance"]), reverse=True)