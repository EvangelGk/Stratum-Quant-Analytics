import json

import pandas as pd
import pytest

from src.exceptions.MedallionExceptions import (
    CatalogNotFoundError,
)

# DummyConfig and SilverLayer setup is handled by the shared fixtures
# in tests/test_src/test_Medallion/conftest.py (silver_layer)
# and tests/conftest.py (dummy_config).


def test_standardize_and_impute_and_audit_columns(silver_layer):

    # Test _standardize: date alignment and unit normalization
    df = pd.DataFrame({"Date": ["2020-01-01", "2020-02-01"], "Value": [200.0, 300.0]})
    standardized, unit_norm, temporal_aligned = silver_layer._standardize(df.copy(), "fred")
    assert temporal_aligned is True
    assert unit_norm is True
    assert standardized["value"].max() <= 3.0

    # Test _impute: should fill missing values and compute imputed count
    df2 = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01", "2020-02-01"]), "value": [None, 2.0]}
    )
    imputed_df, imputed_count, outliers = silver_layer._impute(df2.copy(), "fred")
    assert imputed_count >= 1
    assert outliers >= 0
    assert not imputed_df.isnull().any().any()

    # Test audit columns
    audited = silver_layer._add_audit_columns(
        imputed_df, "file", "fred", imputed_count, 2, 1, outliers
    )
    for col in [
        "processed_at",
        "silver_run_id",
        "schema_version",
        "imputed_count",
        "outliers_clipped",
    ]:
        assert col in audited.columns


def test_load_catalog_raises_when_missing(silver_layer):
    with pytest.raises(CatalogNotFoundError):
        silver_layer._load_catalog()


def test_load_catalog_reads_file(silver_layer):
    catalog_file = silver_layer.raw_path / "catalog.json"
    catalog_file.write_text(json.dumps({"a": {"path": "x"}}))

    loaded = silver_layer._load_catalog()
    assert loaded == {"a": {"path": "x"}}


def test_save_to_silver_creates_parquet(silver_layer):
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01"]), "value": [1.0], "category": ["A"]}
    )
    silver_layer._save_to_silver(df, "file", "fred")
    out_path = silver_layer.processed_path / "fred" / "file_silver.parquet"
    assert out_path.exists()
