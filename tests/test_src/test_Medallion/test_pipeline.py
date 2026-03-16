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
