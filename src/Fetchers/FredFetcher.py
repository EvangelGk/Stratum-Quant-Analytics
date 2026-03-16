import pandas as pd
from fredapi import Fred

from .BaseFetcher import BaseFetcher


class FredFetcher(BaseFetcher):
    def __init__(self, api_key: str):
        super().__init__()
        self.fred = Fred(api_key=api_key)

    # Fetching data from FRED with caching.
    def fetch(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        key = f"fred_{series_id}_{start_date}_{end_date}"
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        data = self.fred.get_series(
            series_id, observation_start=start_date, observation_end=end_date
        )
        df = data.to_frame(name="value").reset_index()
        df.columns = ["Date", "Value"]
        self._set_cached(key, df)
        return df
