import pandas as pd
from .BaseFetcher import BaseFetcher

class FredFetcher(BaseFetcher):
    from fredapi import Fred
    
    def __init__(self, api_key: str):
        self.fred = self.Fred(api_key=api_key)
   
    #Fetching data from FRED.
    def fetch(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        df = data.to_frame(name="value").reset_index()
        df.columns = ["Date", "Value"]
        return df
