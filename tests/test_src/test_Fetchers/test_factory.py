import pytest

import Fetchers.Factory as factory_module


def test_data_factory_get_fetcher_unknown(monkeypatch):
    # Patch fetcher classes to avoid external calls during instantiation
    monkeypatch.setattr(factory_module, "YFinanceFetcher", lambda: "yf")
    monkeypatch.setattr(factory_module, "FredFetcher", lambda api_key=None: "fred")
    monkeypatch.setattr(factory_module, "WorldBankFetcher", lambda: "wb")

    df = factory_module.DataFactory(fred_api_key="dummy")
    assert df.get_fetcher("yfinance") == "yf"
    assert df.get_fetcher("fred") == "fred"
    assert df.get_fetcher("worldbank") == "wb"

    with pytest.raises(ValueError):
        df.get_fetcher("unsupported")
