from typing import cast

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema


# --- Reusable Checks (Senior Practice: DRY) ---
def z_score_check(series: pd.Series, threshold: float = 0.95) -> bool:
    """Check that no single-day return exceeds ±95%.

    A daily price change beyond ±95% is almost certainly a data error
    (bad API tick, unadjusted split, fat-finger). Using absolute return cap
    instead of Z-score because financial returns are fat-tailed and
    non-normal — Z-score at 3.5σ incorrectly flags legitimate moves
    for volatile or trending stocks.
    The Silver layer's own winsorization handles soft outliers.
    """
    if len(series) < 2:
        return True
    daily_ret = series.pct_change().dropna()
    return cast(bool, (daily_ret.abs() <= threshold).all())


# --- 1. Financials Schema (yFinance) ---
financials_schema = DataFrameSchema(
    columns={
        "date": Column(pa.DateTime, nullable=False),
        "open": Column(float, Check.greater_than(0), nullable=True),
        "high": Column(float, Check.greater_than(0), nullable=True),
        "low": Column(float, Check.greater_than(0), nullable=True),
        "close": Column(
            float,
            [
                Check.greater_than(0),
                Check(
                    z_score_check,
                    error="Statistical anomaly detected in Close price (Z-score > 3.5)",
                ),
            ],
        ),
        "adj_close": Column(float, Check.greater_than(0), nullable=True),
        "volume": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "source_system": Column(str, Check.isin(["yfinance"])),
        "ingested_at": Column(str),
    },
    checks=[
        Check(
            lambda df: df["date"].is_monotonic_increasing,
            error="Dates must be chronologically ordered",
        ),
        Check(
            lambda df: df["date"].nunique() == len(df),
            error="Duplicate dates found in financials",
        ),
    ],
    strict="filter",
    coerce=True,
)

# --- 2. Macro Schema (FRED) ---
macro_schema = DataFrameSchema(
    columns={
        "date": Column(pa.DateTime, nullable=False),
        "value": Column(
            float,
            # No hard numeric range — FRED series span percentages (-10% to 30%),
            # CPI / energy price indices (100–350+), and employment rates (0–100).
            # A fixed range would incorrectly reject valid absolute-index series.
            nullable=True,
        ),
        "source_system": Column(str, Check.isin(["fred"])),
    },
    strict="filter",
    coerce=True,
)

# --- 3. World Bank Schema ---
worldbank_schema = DataFrameSchema(
    columns={
        "date": Column(pa.DateTime, nullable=False),
        "value": Column(float, nullable=True),
        "economy": Column(
            str,
            [
                Check.str_length(3, 3),  # ISO-3 Code (π.χ. GRC, USA)
                Check(
                    lambda s: s.str.isupper(),
                    error="Economy code must be uppercase ISO-3",
                ),
            ],
        ),
        "source_system": Column(str, Check.isin(["worldbank"])),
    },
    checks=[
        Check(
            lambda df: df["value"].isnull().mean() < 0.3,
            error="Too many missing values in World Bank data",
        )
    ],
    strict="filter",
    coerce=True,
)
