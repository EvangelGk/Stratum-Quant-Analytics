import pandas as pd
import numpy as np

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
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])


def test_worldbank_fetcher_returns_empty_for_all_nan_values(monkeypatch):
    """Regression test: when the World Bank API returns rows but every Value is
    NaN (e.g. the WLD aggregate for FP.CPI.TOTL.ZG became unavailable), the
    fetcher must return an empty DataFrame so Bronze skips the entity and Silver
    never sees a 100%-null parquet that would trigger a hard-fail guardrail."""

    class DummyWBAllNaN:
        class data:
            @staticmethod
            def DataFrame(indicator, country, time=None):
                # Simulate the wide-format all-NaN response observed on 2026-04-24
                years = list(time) if time is not None else [2020, 2021]
                cols = {f"YR{y}": [np.nan] for y in years}
                cols["economy"] = ["WLD"]
                return pd.DataFrame(cols).set_index("economy")

    monkeypatch.setattr(wb_module, "wb", DummyWBAllNaN)

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
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = fetcher.fetch("FP.CPI.TOTL.ZG", "WLD", "2020-01-01", "2026-12-31")

    assert df.empty, "Expected empty DataFrame when all API values are NaN"
    assert any("all-NaN" in str(w.message) for w in caught), "Expected UserWarning about all-NaN payload"

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
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
