import pandas as pd

from exceptions.FetchersExceptions import APIError, CacheError

from .BaseFetcher import BaseFetcher


class YFinanceFetcher(BaseFetcher):
    import yfinance as yf

    def __init__(self) -> None:
        super().__init__()

    # Fetching data from Yahoo Finance with caching.
    def fetch(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        key = f"yfinance_{ticker}_{start_date}_{end_date}"
        try:
            cached = self._get_cached(key)
            if cached is not None:
                return cached
        except Exception as e:
            raise CacheError(f"Cache read error: {e}") from e

        try:
            data = self.yf.download(
                ticker, start=start_date, end=end_date, interval="1mo"
            )
            df = data.reset_index()
        except Exception as e:
            raise APIError(f"YFinance API error: {e}") from e

        try:
            self._set_cached(key, df)
        except Exception as e:
            raise CacheError(f"Cache write error: {e}") from e

        return df
