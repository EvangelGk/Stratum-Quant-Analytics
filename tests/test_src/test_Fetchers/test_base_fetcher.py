import pandas as pd


def test_base_fetcher_cache(monkeypatch):
    """BaseFetcher should store and retrieve values from its cache."""

    # Create a dummy cache implementation to avoid diskcache dependency
    store = {}

    class DummyCache:
        def __init__(self, path):
            self.path = path

        def get(self, key):
            return store.get(key)

        def set(self, key, value, expire=None):
            store[key] = value

    # Patch diskcache.Cache used in BaseFetcher
    import Fetchers.BaseFetcher as bf

    monkeypatch.setattr(bf.dc, "Cache", DummyCache)

    class DummyFetcher(bf.BaseFetcher):
        def fetch(
            self, identifier: str, start_date: str, end_date: str
        ) -> pd.DataFrame:
            return pd.DataFrame({"value": [1]})

    f = DummyFetcher()
    assert f._get_cached("missing") is None

    # Store and retrieve a pandas object
    df = pd.DataFrame({"value": [1, 2, 3]})
    f._set_cached("mykey", df)
    retrieved = f._get_cached("mykey")
    assert isinstance(retrieved, pd.DataFrame)
    assert retrieved.equals(df)
