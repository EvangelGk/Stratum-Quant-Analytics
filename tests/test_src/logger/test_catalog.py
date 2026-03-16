import json

from logger.Catalog import ApplicationCatalog


def test_catalog_logging_and_summary(tmp_path, monkeypatch):
    # Use a temporary working directory so logs don't pollute repo
    monkeypatch.chdir(tmp_path)

    catalog = ApplicationCatalog(log_file="test_app.log")
    catalog.log_operation(
        "test_op", "test_component", {"foo": 1}, {"bar": 2}, "Test message"
    )
    catalog.log_data_operation(
        "fetch", "yfinance", records=10, files=1, duration=0.1, success=True
    )
    catalog.log_error("test_component", "TestError", "failure", "test_op")

    summary_path = catalog.save_session_summary()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["session_info"]["total_operations"] >= 1
    assert "error_metrics" in summary

    metrics = catalog.get_metrics_summary()
    assert metrics["error_count"] >= 1
