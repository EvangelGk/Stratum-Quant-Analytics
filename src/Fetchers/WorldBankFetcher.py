import wbgapi as wb
import pandas as pd
from typing import List, Union
from .BaseFetcher import BaseFetcher

class WorldBankFetcher(BaseFetcher):
    def __init__(self):
        pass  # No API key needed for World Bank
   
    # Fetching data from the World Bank API.
    def fetch(self, indicator: str, country: str, start_date: str, end_date: str) -> pd.DataFrame:
        # The wbgapi accepts a range of years
        start_year = int(pd.to_datetime(start_date).year)
        end_year = int(pd.to_datetime(end_date).year)
        
        data = wb.data.DataFrame(indicator, country, time=range(start_year, end_year + 1))
        df = data.reset_index()
        # Conversion from "Wide" to "Long" format to align with other fetchers
        df = df.melt(id_vars='economy', var_name='Date', value_name='Value')
        df['Date'] = df['Date'].str.replace('YR', '').astype(int)
        return df
