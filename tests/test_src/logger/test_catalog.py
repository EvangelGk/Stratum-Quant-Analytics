import json

from src.logger.Catalog import ApplicationCatalog


def test_catalog_logging_and_summary(tmp_path, monkeypatch):
    # Use a temporary working directory so logs don't pollute repo
    monkeypatch.chdir(tmp_path)

    catalog = ApplicationCatalog(log_file="test_app.log")
    catalog.set_run_context("run-1", "corr-1")
    catalog.log_operation(
        "test_op", "test_component", {"foo": 1}, {"bar": 2}, "Test message"
    )
    catalog.log_data_operation(
        "fetch", "yfinance", records=10, files=1, duration=0.1, success=True
    )
    catalog.log_error("test_component", "TestError", "failure", "test_op")
    catalog.log_sla_snapshot(
        "medallion",
        p95_latency_seconds=0.2,
        error_rate=0.0,
        success_rate=1.0,
        throughput_ops_per_sec=5.0,
    )
    catalog.log_operation(
        "pipeline_stage",
        "medallion",
        {"duration_seconds": 0.5, "success": True},
        {},
        "stage ok",
    )
    catalog.log_slo_window("medallion", 300, "pipeline_stage")

    summary_path = catalog.save_session_summary()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["session_info"]["total_operations"] >= 1
    assert summary["session_info"]["run_id"] == "run-1"
    assert summary["session_info"]["correlation_id"] == "corr-1"
    assert "sla_metrics" in summary
    assert "error_metrics" in summary
    assert any(
        op["operation"] == "slo_window" for op in summary["operations_timeline"]
    )

    metrics = catalog.get_metrics_summary()
    assert metrics["error_count"] >= 1
