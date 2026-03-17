import pytest

import src.Fetchers.Factory as factory_module
from src.exceptions.FetchersExceptions import MissingAPIKeyError


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


def test_data_factory_missing_fred_key_raises_specific_error(monkeypatch):
    monkeypatch.setattr(factory_module, "YFinanceFetcher", lambda: "yf")
    monkeypatch.setattr(factory_module, "WorldBankFetcher", lambda: "wb")
    df = factory_module.DataFactory(fred_api_key=None)

    with pytest.raises(MissingAPIKeyError):
        df.get_fetcher("fred")
