import json

import pandas as pd
import pytest

from src.exceptions.MedallionExceptions import (
    CatalogNotFoundError,
)
from src.Fetchers.ProjectConfig import ProjectConfig
from src.Medallion.silver.silver import SilverLayer


class DummyConfig(ProjectConfig):
    def __init__(self):
        super().__init__(fred_api_key="dummy")
        self.max_workers = 1


def test_standardize_and_impute_and_audit_columns(tmp_path):
    cfg = DummyConfig()
    layer = SilverLayer(cfg)
    # Override paths to avoid writing in the project directory
    layer.raw_path = tmp_path / "raw"
    layer.processed_path = tmp_path / "processed"
    layer.processed_path.mkdir(parents=True, exist_ok=True)
    layer.raw_path.mkdir(parents=True, exist_ok=True)

    # Test _standardize: date alignment and unit normalization
    df = pd.DataFrame({"Date": ["2020-01-01", "2020-02-01"], "Value": [200.0, 300.0]})
    standardized, unit_norm, temporal_aligned = layer._standardize(df.copy(), "fred")
    assert temporal_aligned is True
    assert unit_norm is True
    assert standardized["value"].max() <= 3.0

    # Test _impute: should fill missing values and compute imputed count
    df2 = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01", "2020-02-01"]), "value": [None, 2.0]}
    )
    imputed_df, imputed_count, outliers = layer._impute(df2.copy(), "fred")
    assert imputed_count >= 1
    assert outliers >= 0
    assert not imputed_df.isnull().any().any()

    # Test audit columns
    audited = layer._add_audit_columns(
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


def test_load_catalog_raises_when_missing(tmp_path):
    cfg = DummyConfig()
    layer = SilverLayer(cfg)
    layer.raw_path = tmp_path / "raw"
    layer.raw_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(CatalogNotFoundError):
        layer._load_catalog()


def test_load_catalog_reads_file(tmp_path):
    cfg = DummyConfig()
    layer = SilverLayer(cfg)
    layer.raw_path = tmp_path / "raw"
    layer.raw_path.mkdir(parents=True, exist_ok=True)

    catalog_file = layer.raw_path / "catalog.json"
    catalog_file.write_text(json.dumps({"a": {"path": "x"}}))

    loaded = layer._load_catalog()
    assert loaded == {"a": {"path": "x"}}


def test_save_to_silver_creates_parquet(tmp_path):
    cfg = DummyConfig()
    layer = SilverLayer(cfg)
    layer.processed_path = tmp_path / "processed"
    layer.processed_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01"]), "value": [1.0], "category": ["A"]}
    )
    layer._save_to_silver(df, "file", "fred")
    out_path = layer.processed_path / "fred" / "file_silver.parquet"
    assert out_path.exists()
