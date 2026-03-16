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
        df = data.reset_index(drop=True)
        # Conversion from "Wide" to "Long" format to align with other fetchers
        df = df.melt(id_vars="economy", var_name="Date", value_name="Value")
        df["Date"] = df["Date"].str.replace("YR", "").astype(int)
        self._set_cached(key, df)
        return df
