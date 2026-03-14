import pandas as pd
from fredapi import Fred
from .BaseFetcher import BaseFetcher

class FredFetcher(BaseFetcher):
    def __init__(self, api_key: str):
        self.fred = Fred(api_key=api_key)
   
    # Fetching data from FRED.
    def fetch(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        df = data.to_frame(name="value").reset_index()
        df.columns = ["Date", "Value"]
        return df
