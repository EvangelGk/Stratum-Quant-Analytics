import pandas as pd

import src.Fetchers.WorldBankFetcher as wb_module


def test_worldbank_fetcher_transforms_data(monkeypatch):
    # Patch wbgapi data frame creation
    class DummyWB:
        class data:
            @staticmethod
            def DataFrame(indicator, country, time=None):
                # Create a wide-format DataFrame like wbgapi might
                return pd.DataFrame(
                    {
                        "economy": ["WLD", "WLD"],
                        "YR2020": [1.0, 2.0],
                        "YR2021": [1.5, 2.5],
                    }
                )

    monkeypatch.setattr(wb_module, "wb", DummyWB)

    # Patch diskcache.Cache to avoid disk writes
    import src.Fetchers.BaseFetcher as bf

    class DummyCache:
        def __init__(self, path):
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, expire=None):
            self.store[key] = value

    monkeypatch.setattr(bf.dc, "Cache", DummyCache)

    fetcher = wb_module.WorldBankFetcher()
    df = fetcher.fetch("NY.GDP.MKTP.KD.ZG", "WLD", "2020-01-01", "2021-12-31")

    assert "economy" in df.columns
    assert "Date" in df.columns
    assert "Value" in df.columns
    assert df["Date"].dtype == int
