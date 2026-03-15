import pandas as pd

import Fetchers.FredFetcher as fred_module


def test_fred_fetcher_returns_dataframe(monkeypatch):
    class DummyFred:
        def __init__(self, api_key):
            self.api_key = api_key

        def get_series(self, series_id, observation_start=None, observation_end=None):
            # Return a Series with a DatetimeIndex
            return pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))

    # Patch the Fred client and caching layer to avoid diskcache usage
    monkeypatch.setattr(fred_module, "Fred", DummyFred)

    # Patch diskcache.Cache used in BaseFetcher
    import Fetchers.BaseFetcher as bf

    class DummyCache:
        def __init__(self, path):
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, expire=None):
            self.store[key] = value

    monkeypatch.setattr(bf.dc, "Cache", DummyCache)

    fetcher = fred_module.FredFetcher(api_key="abc")
    df = fetcher.fetch("SERIES", "2020-01-01", "2020-01-02")

    assert list(df.columns) == ["Date", "Value"]
    assert len(df) == 2

    # Second fetch should return cached DataFrame (same object)
    df2 = fetcher.fetch("SERIES", "2020-01-01", "2020-01-02")
    assert df2.equals(df)
