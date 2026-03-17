import pandas as pd
import wbgapi as wb

from .BaseFetcher import BaseFetcher


class WorldBankFetcher(BaseFetcher):
    def __init__(self) -> None:
        super().__init__()

    # Fetching data from the World Bank API with caching.
    def fetch(
        self, indicator: str, country: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        key = f"worldbank_{indicator}_{country}_{start_date}_{end_date}"
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        # The wbgapi accepts a range of years
        start_year = int(pd.to_datetime(start_date).year)
        end_year = int(pd.to_datetime(end_date).year)

        data = wb.data.DataFrame(
            indicator, country, time=range(start_year, end_year + 1)
        )
        df = self._normalize_worldbank_frame(data, indicator)
        self._set_cached(key, df)
        return df

    def _normalize_worldbank_frame(
        self, data: pd.DataFrame, indicator: str
    ) -> pd.DataFrame:
        """Normalize World Bank output into economy/Date/Value schema."""
        if data is None or data.empty:
            return pd.DataFrame(columns=["economy", "Date", "Value"])

        df = data.reset_index()

        # Common long format: economy + time + indicator column.
        if "time" in df.columns and indicator in df.columns:
            df = df.rename(columns={"time": "Date", indicator: "Value"})
        else:
            # Wide format fallback with year columns (e.g., YR2020).
            year_cols = [
                col
                for col in df.columns
                if str(col).startswith("YR") or str(col).isdigit()
            ]
            if year_cols and "economy" in df.columns:
                df = df.melt(
                    id_vars="economy",
                    value_vars=year_cols,
                    var_name="Date",
                    value_name="Value",
                )
            else:
                value_candidates = [
                    col
                    for col in df.columns
                    if col not in {"economy", "time", "Date", "Value"}
                ]
                if "time" in df.columns and value_candidates:
                    df = df.rename(
                        columns={"time": "Date", value_candidates[0]: "Value"}
                    )

        if "economy" not in df.columns:
            df["economy"] = "WLD"

        if "Date" in df.columns:
            df["Date"] = df["Date"].astype(str).str.replace("YR", "", regex=False)
            numeric_dates = pd.to_numeric(df["Date"], errors="coerce")
            if numeric_dates.notna().all():
                df["Date"] = numeric_dates.astype(int)

        if "Value" not in df.columns:
            df["Value"] = pd.NA

        return df[["economy", "Date", "Value"]]
