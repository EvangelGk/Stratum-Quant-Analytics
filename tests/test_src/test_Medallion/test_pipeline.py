import pytest

import src.Medallion as medallion_module
from src.exceptions.MedallionExceptions import ParallelExecutionError
from src.Medallion import MedallionPipeline

# DummyConfig is provided by the shared dummy_config fixture in tests/conftest.py.


def test_health_check_reports_all_paths(dummy_config, tmp_path):
    cfg = dummy_config
    # Create a minimal factory stub
    factory = object()

    pipeline = MedallionPipeline(cfg, factory)
    # Redirect paths to temporary folder
    pipeline.raw_path = tmp_path / "raw"
    pipeline.processed_path = tmp_path / "processed"
    pipeline.gold_path = tmp_path / "gold"

    # Create required files/directories for health check
    pipeline.raw_path.mkdir(parents=True, exist_ok=True)
    (pipeline.raw_path / "dummy.txt").write_text("x")

    pipeline.processed_path.mkdir(parents=True, exist_ok=True)
    (pipeline.processed_path / "dummy.parquet").write_text("x")

    pipeline.gold_path.mkdir(parents=True, exist_ok=True)
    (pipeline.gold_path / "master_table.parquet").write_text("x")

    checks = pipeline.health_check()
    assert checks["raw_data_exists"] is True
    assert checks["processed_data_exists"] is True
    assert checks["gold_data_exists"] is True
    assert checks["config_valid"] is True


def test_emit_sla_snapshot_uses_real_stage_success_rates(dummy_config, monkeypatch):
    pipeline = MedallionPipeline(dummy_config, object())
    pipeline._stage_durations = {"bronze": 1.0, "silver": 2.0, "gold": 3.0}
    pipeline._stage_success = {"bronze": True, "silver": False, "gold": True}

    captured = {}

    def fake_log_sla_snapshot(
        component,
        p95_latency_seconds,
        error_rate,
        success_rate,
        throughput_ops_per_sec,
    ):
        captured["component"] = component
        captured["error_rate"] = error_rate
        captured["success_rate"] = success_rate
        captured["throughput"] = throughput_ops_per_sec

    monkeypatch.setattr(
        medallion_module.catalog,
        "log_sla_snapshot",
        fake_log_sla_snapshot,
    )

    pipeline._emit_sla_snapshot()

    assert captured["component"] == "medallion"
    assert captured["success_rate"] == 2 / 3
    assert captured["error_rate"] == 1 / 3


def test_pipeline_uses_user_scoped_data_paths(dummy_config):
    dummy_config.data_user_id = "qa_user_01"
    pipeline = MedallionPipeline(dummy_config, object())
    assert "users" in str(pipeline.data_path)
    assert "qa_user_01" in str(pipeline.data_path)


def test_run_silver_raises_when_hard_fail_enabled(dummy_config, monkeypatch):
    pipeline = MedallionPipeline(dummy_config, object())
    pipeline.config.silver_hard_fail = True

    def fake_run():
        return {"success_count": 0, "failed_count": 1, "failed_entities": ["a"]}

    monkeypatch.setattr(pipeline.silver, "run", fake_run)

    with pytest.raises(ParallelExecutionError):
        pipeline.run_silver()


def test_run_silver_does_not_raise_when_hard_fail_disabled(dummy_config, monkeypatch):
    pipeline = MedallionPipeline(dummy_config, object())
    pipeline.config.silver_hard_fail = False

    def fake_run():
        return {"success_count": 0, "failed_count": 1, "failed_entities": ["a"]}

    monkeypatch.setattr(pipeline.silver, "run", fake_run)

    # Should not raise, stage remains marked unsuccessful for observability.
    pipeline.run_silver()
    assert pipeline._stage_success.get("silver") is False
