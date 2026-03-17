import pandas as pd
import pytest

from src.exceptions.MedallionExceptions import (
    ComplianceViolationError,
    SchemaMismatchError,
)
from src.Fetchers.ProjectConfig import ProjectConfig
from src.Medallion.silver.silver import SilverLayer


def _make_config() -> ProjectConfig:
    cfg = ProjectConfig(fred_api_key="dummy")
    cfg.silver_min_rows = 10
    cfg.silver_min_rows_ratio = 0.1
    cfg.silver_base_null_threshold = 30.0
    cfg.silver_dynamic_threshold_window = 20
    return cfg


def test_silver_preflight_raises_compliance_on_empty_df(tmp_path):
    layer = SilverLayer(_make_config())
    layer.raw_path = tmp_path / "raw"
    layer.processed_path = tmp_path / "processed"
    layer.raw_path.mkdir(parents=True, exist_ok=True)
    layer.processed_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ComplianceViolationError):
        layer._preflight_contract_checks(
            pd.DataFrame(), "fred", "inflation", {"rows": 100, "path": "x"}
        )


def test_silver_preflight_raises_schema_mismatch_on_drift(tmp_path):
    layer = SilverLayer(_make_config())
    layer.raw_path = tmp_path / "raw"
    layer.processed_path = tmp_path / "processed"
    layer.raw_path.mkdir(parents=True, exist_ok=True)
    layer.processed_path.mkdir(parents=True, exist_ok=True)

    drifted = pd.DataFrame({"Date": ["2020-01-01"], "unexpected": [1.0]})
    with pytest.raises(SchemaMismatchError):
        layer._preflight_contract_checks(
            drifted, "fred", "inflation", {"rows": 20, "path": "x"}
        )


def test_silver_process_entity_marks_failed_on_schema_mismatch(tmp_path):
    layer = SilverLayer(_make_config())
    layer.raw_path = tmp_path / "raw"
    layer.processed_path = tmp_path / "processed"
    layer.raw_path.mkdir(parents=True, exist_ok=True)
    layer.processed_path.mkdir(parents=True, exist_ok=True)

    bad_file = layer.raw_path / "bad.parquet"
    pd.DataFrame({"Date": ["2020-01-01"], "unexpected": [1.0]}).to_parquet(
        bad_file, index=False
    )

    layer._process_entity("bad", {"path": str(bad_file), "source": "fred", "rows": 20})
    assert layer.quality_reports["bad"]["status"] == "failed"
