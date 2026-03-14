import pandera as pa
from pandera import Column, Check, DataFrameSchema, Index
import pandas as pd

# 1. Schema for Financial Data (yFinance)
financials_schema = pa.DataFrameSchema({
    "Date": Column(pa.DateTime, Check.in_range(pd.Timestamp("2000-01-01"), pd.Timestamp("2030-12-31"), error="Date must be between 2000 and 2030")),
    "Close": Column(float, Check.greater_than(0, error="Close price must be positive"), nullable=False),
    "Volume": Column(float, Check.greater_than_or_equal(0, error="Volume cannot be negative"), nullable=True),
    "source_system": Column(str, Check.isin(["yfinance"], error="Source must be yfinance")),
    "ingested_at": Column(str, nullable=False)
}, strict="filter", coerce=True, version="1.0")

# 2. Schema for Macroeconomic Data (FRED/Inflation)
macro_schema = pa.DataFrameSchema({
    "Date": Column(pa.DateTime, Check.in_range(pd.Timestamp("2000-01-01"), pd.Timestamp("2030-12-31"), error="Date must be between 2000 and 2030")),
    "Value": Column(float, Check.in_range(-20, 100, error="Value must be between -20 and 100"), nullable=False),
    "source_system": Column(str, Check.isin(["fred"], error="Source must be fred"))
}, strict="filter", coerce=True, version="1.0")

# 3. Schema for World Bank
worldbank_schema = pa.DataFrameSchema({
    "Date": Column(pa.Int, Check.in_range(2000, 2030, error="Year must be between 2000 and 2030")),
    "Value": Column(float, nullable=True),
    "economy": Column(str, Check.str_length(1, 3, error="Economy code must be 1-3 characters")),
    "source_system": Column(str, Check.isin(["worldbank"], error="Source must be worldbank"))
}, strict="filter", coerce=True, version="1.0")

# --- Reusable Checks (Senior Practice: DRY) ---
def z_score_check(series: pd.Series, threshold: float = 3.5):
    """Ελέγχει αν υπάρχουν τιμές που ξεφεύγουν στατιστικά (Outliers)."""
    if series.std() == 0: return True
    z_scores = (series - series.mean()).abs() / series.std()
    return (z_scores <= threshold).all()

# --- 1. Financials Schema ---
financials_schema = DataFrameSchema(
    columns={
        "date": Column(pa.DateTime, nullable=False),
        "close": Column(float, [
            Check.greater_than(0),
            Check(z_score_check, error="Statistical anomaly detected in Close price (Z-score > 3.5)")
        ]),
        "volume": Column(float, Check.greater_than_or_equal(0), nullable=True),
        "source_system": Column(str, Check.isin(["yfinance"])),
        "ingested_at": Column(str)
    },
    # DataFrame-level Check: Η καρδιά του Senior Validation
    checks=[
        Check(lambda df: df['date'].is_monotonic_increasing, error="Dates must be chronologically ordered"),
        Check(lambda df: df['date'].nunique() == len(df), error="Duplicate dates found in financials")
    ],
    strict="filter",
    coerce=True
)

# --- 2. Macro Schema ---
macro_schema = DataFrameSchema(
    columns={
        "date": Column(pa.DateTime, nullable=False),
        "value": Column(float, [
            Check(lambda s: s.diff().abs().max() < 10, error="Suspiciously high volatility in macro value")
        ]),
        "source_system": Column(str, Check.isin(["fred"]))
    },
    index=Index(pa.Int, name="id", nullable=True), 
    strict="filter",
    coerce=True
)

# --- 3. World Bank Schema ---
worldbank_schema = DataFrameSchema(
    columns={
        "date": Column(pa.Int, Check.in_range(2000, 2030)),
        "value": Column(float, nullable=True),
        "economy": Column(str, [
            Check.str_length(3, 3), # ISO-3 Code (π.χ. GRC, USA)
            Check(lambda s: s.str.isupper(), error="Economy code must be uppercase ISO-3")
        ]),
        "source_system": Column(str, Check.isin(["worldbank"]))
    },
    checks=[
        Check(lambda df: df['value'].isnull().mean() < 0.3, error="Too many missing values in World Bank data")
    ],
    strict="filter",
    coerce=True
)