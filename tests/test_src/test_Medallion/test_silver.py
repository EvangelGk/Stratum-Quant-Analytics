import json

import pandas as pd
import pytest

from src.exceptions.MedallionExceptions import (
    CatalogNotFoundError,
    ComplianceViolationError,
)

# DummyConfig and SilverLayer setup is handled by the shared fixtures
# in tests/test_src/test_Medallion/conftest.py (silver_layer)
# and tests/conftest.py (dummy_config).


def test_standardize_and_impute_and_audit_columns(silver_layer):

    # Test _standardize: date alignment and unit normalization
    df = pd.DataFrame({"Date": ["2020-01-01", "2020-02-01"], "Value": [200.0, 300.0]})
    standardized, unit_norm, temporal_aligned = silver_layer._standardize(
        df.copy(), "fred", "inflation"
    )
    assert temporal_aligned is True
    assert unit_norm is True
    assert standardized["value"].max() <= 3.0

    # Non-percentage indicators must not be divided by 100 automatically.
    non_pct_df = pd.DataFrame(
        {"Date": ["2020-01-01", "2020-02-01"], "Value": [200.0, 300.0]}
    )
    standardized_non_pct, non_pct_norm, _ = silver_layer._standardize(
        non_pct_df.copy(), "fred", "energy_index"
    )
    assert non_pct_norm is False
    assert standardized_non_pct["value"].max() >= 200.0

    # Test _impute: should fill missing values and compute imputed count
    df2 = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01", "2020-02-01"]), "value": [None, 2.0]}
    )
    imputed_df, imputed_count, outliers, max_col_null_pct = silver_layer._impute(
        df2.copy(), "fred"
    )
    assert imputed_count >= 1
    assert outliers >= 0
    assert isinstance(max_col_null_pct, float)
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


def test_preflight_contract_checks_raises_for_empty_df(silver_layer):
    with pytest.raises(ComplianceViolationError):
        silver_layer._preflight_contract_checks(
            pd.DataFrame(), "fred", "inflation", {"rows": 100, "path": "x"}
        )


def test_preflight_contract_checks_raises_for_schema_drift(silver_layer):
    drifted = pd.DataFrame({"Date": ["2020-01-01"], "bad_col": [1]})
    with pytest.raises(Exception):
        silver_layer._preflight_contract_checks(
            drifted, "fred", "inflation", {"rows": 10, "path": "x"}
        )


def test_dynamic_threshold_uses_history(silver_layer):
    history_path = silver_layer.processed_path / "quality_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        "\n".join(
            [
                json.dumps({"source": "fred", "max_col_null_pct": 10.0}),
                json.dumps({"source": "fred", "max_col_null_pct": 20.0}),
                json.dumps({"source": "fred", "max_col_null_pct": 25.0}),
            ]
        )
        + "\n"
    )
    threshold = silver_layer._resolve_dynamic_null_threshold("fred")
    assert threshold >= silver_layer.config.silver_base_null_threshold


def test_append_quality_history_writes_rows(silver_layer):
    silver_layer.quality_reports = {
        "inflation": {
            "source": "fred",
            "status": "success",
            "initial_rows": 20,
            "final_rows": 20,
            "initial_nulls": 2,
            "final_nulls": 0,
            "imputed_count": 2,
            "outliers_clipped": 1,
            "max_col_null_pct": 10.0,
        }
    }
    silver_layer._append_quality_history()
    history_path = silver_layer.processed_path / "quality_history.jsonl"
    assert history_path.exists()
    lines = history_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1


def test_append_dead_letter_writes_entry(silver_layer):
    exc = ComplianceViolationError("bad quality")
    silver_layer._append_dead_letter("fileA", {"source": "fred", "path": "x"}, exc)
    assert silver_layer.dead_letter_path.exists()
    content = silver_layer.dead_letter_path.read_text(encoding="utf-8")
    assert "fileA" in content
    assert "ComplianceViolationError" in content
