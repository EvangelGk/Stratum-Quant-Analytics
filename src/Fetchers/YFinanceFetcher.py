import pandas as pd
from .BaseFetcher import BaseFetcher

class YFinanceFetcher(BaseFetcher):
    import yfinance as yf
    #Fetching data from Yahoo Finance.
    def fetch(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        data = self.yf.download(ticker, start=start_date, end=end_date, interval="1mo")
        # Returning DataFrame with the correct format for the Bronze layer
        return data.reset_index()
        