import os

import pandas as pd

# DummyConfig and BronzeLayer setup is handled by the shared fixtures
# in tests/test_src/test_Medallion/conftest.py (bronze_layer)
# and tests/conftest.py (dummy_config).


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


def test_get_expected_columns(bronze_layer):
    assert bronze_layer._get_expected_columns("yfinance") == [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    assert bronze_layer._get_expected_columns("fred") == ["Date", "Value"]
    assert bronze_layer._get_expected_columns("worldbank") == [
        "economy",
        "Date",
        "Value",
    ]


def test_process_and_save_writes_parquet(bronze_layer):
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
    bronze_layer._process_and_save(df, "test_file", "yfinance")
    parquet_path = os.path.join(bronze_layer.base_path, "yfinance", "test_file.parquet")
    assert os.path.exists(parquet_path)
    assert "test_file" in bronze_layer.catalog


def test_process_and_save_raises_on_bad_columns(bronze_layer):
    bad_df = pd.DataFrame({"a": [1, 2]})
    try:
        bronze_layer._process_and_save(bad_df, "bad", "yfinance")
    except ValueError as e:
        assert "missing required columns" in str(e)
    else:
        raise AssertionError("Expected ValueError for missing columns")


def test_provider_circuit_breaker_opens_after_repeated_failures(bronze_layer):
    source = "fred"
    bronze_layer.provider_state[source] = {
        "last_request_ts": 0.0,
        "consecutive_failures": 0.0,
        "circuit_open_until": 0.0,
    }

    bronze_layer._record_provider_failure(source)
    bronze_layer._record_provider_failure(source)
    bronze_layer._record_provider_failure(source)

    assert bronze_layer.provider_state[source]["consecutive_failures"] >= 3
    assert bronze_layer.provider_state[source]["circuit_open_until"] > 0
