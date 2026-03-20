from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .mixed_frequency import stationary_transform


def feature_decay_analysis(
    df: pd.DataFrame,
    target: str = "log_return",
    features: Optional[List[str]] = None,
    max_lag: int = 180,
) -> Dict[str, Any]:
    """Quantify information decay by lagging each feature against target.

    The function scans lag-days in [0, max_lag] and computes correlation between
    target(t) and feature(t-lag). It then estimates an information half-life as
    the first lag where absolute correlation falls below 50% of lag-0 value.
    """
    features = features or [
        "inflation",
        "energy_index",
        "fed_funds_rate",
        "gdp_growth",
        "consumer_sentiment",
    ]
    if target not in df.columns:
        return {
            "status": "error",
            "error": f"target_not_found:{target}",
            "results": {},
        }

    work_df = df.copy()
    if "date" in work_df.columns:
        work_df["date"] = pd.to_datetime(work_df["date"], errors="coerce")
        work_df = work_df.dropna(subset=["date"]).sort_values("date")

    out: Dict[str, Any] = {}
    # Apply stationary transforms before computing lag correlations.
    # Raw macro level series (inflation index, energy index) trend over time;
    # Pearson correlation on trending levels produces spuriously high values
    # that reflect shared drift, not genuine predictive information.
    y_raw = pd.to_numeric(work_df[target], errors="coerce")
    y, _ = stationary_transform(y_raw, target, keep_as_is=True)
    for feature in features:
        if feature not in work_df.columns:
            continue
        x_raw = pd.to_numeric(work_df[feature], errors="coerce")
        x_stationary, transform_method = stationary_transform(x_raw, feature)
        rows = []
        for lag in range(0, max_lag + 1):
            aligned = pd.concat([y, x_stationary.shift(lag)], axis=1).dropna()
            if len(aligned) < 25:
                corr = None
            else:
                corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            rows.append({"lag_days": int(lag), "correlation": corr})

        baseline = rows[0]["correlation"] if rows else None
        threshold = abs(float(baseline)) * 0.5 if isinstance(baseline, (int, float)) else None
        half_life = None
        if threshold is not None:
            for row in rows[1:]:
                c = row.get("correlation")
                if isinstance(c, (int, float)) and abs(float(c)) <= threshold:
                    half_life = int(row["lag_days"])
                    break

        out[feature] = {
            "baseline_correlation": baseline,
            "half_life_lag_days": half_life,
            "transform_method": transform_method,
            "decay_scan": rows,
        }

    return {
        "status": "ok",
        "target": target,
        "max_lag_days": int(max_lag),
        "results": out,
    }
