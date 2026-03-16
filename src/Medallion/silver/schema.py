import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema


# --- Reusable Checks (Senior Practice: DRY) ---
def z_score_check(series: pd.Series, threshold: float = 3.5):
    """Ελέγχει αν υπάρχουν τιμές που ξεφεύγουν στατιστικά (Outliers)."""
    if series.std() == 0:
        return True
    z_scores = (series - series.mean()).abs() / series.std()
    return (z_scores <= threshold).all()


# --- 1. Financials Schema (yFinance) ---
financials_schema = DataFrameSchema(
    columns={
        "date": Column(pa.DateTime, nullable=False),
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
        "volume": Column(
            float, Check.greater_than_or_equal_to(0), nullable=True
        ),
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
            [
                Check.in_range(
                    -20, 100, error="Value must be between -20 and 100"
                )
            ],
        ),
        "source_system": Column(str, Check.isin(["fred"])),
    },
    strict="filter",
    coerce=True,
)

# --- 3. World Bank Schema ---
worldbank_schema = DataFrameSchema(
    columns={
        "date": Column(pa.Int, Check.in_range(2000, 2030)),
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
