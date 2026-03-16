import os

import pandas as pd

from src.Fetchers.ProjectConfig import ProjectConfig
from src.Medallion.bronze import BronzeLayer


class DummyConfig(ProjectConfig):
    def __init__(self):
        super().__init__(fred_api_key="dummy")
        self.max_workers = 1
        self.max_retries = 1
        self.retry_delay_min = 0
        self.retry_delay_max = 0


class DummyFetcher:
    def fetch(self, *args, **kwargs):
        return pd.DataFrame(
            {
                "Date": ["2020-01-01"],
                "Open": [1.0],
                "High": [2.0],
                "Low": [0.5],
                "Close": [1.5],
                "Adj Close": [1.4],
                "Volume": [100],
            }
        )


def test_get_expected_columns():
    layer = BronzeLayer(DummyConfig(), factory=None)
    assert layer._get_expected_columns("yfinance") == [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    assert layer._get_expected_columns("fred") == ["Date", "Value"]
    assert layer._get_expected_columns("worldbank") == ["economy", "Date", "Value"]


def test_process_and_save_writes_parquet(tmp_path, monkeypatch):
    cfg = DummyConfig()
    layer = BronzeLayer(cfg, factory=None)
    layer.base_path = str(tmp_path)

    df = pd.DataFrame(
        {
            "Date": ["2020-01-01"],
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Adj Close": [1.4],
            "Volume": [100],
        }
    )

    # Ensure that saving creates the file and updates catalog
    layer._process_and_save(df, "test_file", "yfinance")
    parquet_path = os.path.join(layer.base_path, "yfinance", "test_file.parquet")
    assert os.path.exists(parquet_path)
    assert "test_file" in layer.catalog


def test_process_and_save_raises_on_bad_columns(tmp_path):
    cfg = DummyConfig()
    layer = BronzeLayer(cfg, factory=None)
    layer.base_path = str(tmp_path)

    bad_df = pd.DataFrame({"a": [1, 2]})
    try:
        layer._process_and_save(bad_df, "bad", "yfinance")
    except ValueError as e:
        assert "missing required columns" in str(e)
    else:
        raise AssertionError("Expected ValueError for missing columns")
