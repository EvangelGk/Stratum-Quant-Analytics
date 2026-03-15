import pandas as pd

import Fetchers.YFinanceFetcher as yf_module


def test_yfinance_fetcher_wraps_errors_and_caches(monkeypatch):
    # Patch the yfinance download function to return a simple DataFrame
    class DummyYF:
        @staticmethod
        def download(ticker, start=None, end=None, interval=None):
            df = pd.DataFrame({
                "Open": [1.0],
                "High": [2.0],
                "Low": [0.5],
                "Close": [1.5],
                "Adj Close": [1.4],
                "Volume": [100]
            }, index=pd.to_datetime(["2020-01-01"]))
            return df

    monkeypatch.setattr(yf_module.YFinanceFetcher, "yf", DummyYF)

    # Patch diskcache.Cache to avoid writing to disk
    import Fetchers.BaseFetcher as bf

    class DummyCache:
        def __init__(self, path):
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, expire=None):
            self.store[key] = value

    monkeypatch.setattr(bf.dc, "Cache", DummyCache)

    fetcher = yf_module.YFinanceFetcher()
    df = fetcher.fetch("AAPL", "2020-01-01", "2020-02-01")

    assert "Open" in df.columns
    assert len(df) == 1

    # Calling again should hit cache and not raise
    df2 = fetcher.fetch("AAPL", "2020-01-01", "2020-02-01")
    assert df2.equals(df)
