import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.optimizer import AutomatedOptimizationLoop


# ---------------------------------------------------------------------------
# Shared stub factory
# ---------------------------------------------------------------------------

def _loop_stub(tmp_path: Path, monkeypatch) -> AutomatedOptimizationLoop:
    loop = AutomatedOptimizationLoop.__new__(AutomatedOptimizationLoop)
    loop.user_id = "alice"
    # Bind a deterministic command runner for validation tests.
    loop._run_cmd = lambda *args, **kwargs: (True, "ok")  # type: ignore[attr-defined]

    # Patch class properties for this test instance shape.
    monkeypatch.setattr(AutomatedOptimizationLoop, "project_root", property(lambda self: tmp_path))
    monkeypatch.setattr(AutomatedOptimizationLoop, "safe_user", property(lambda self: "alice"))
    return loop


# ---------------------------------------------------------------------------
# _scan_system_health (base)
# ---------------------------------------------------------------------------

def test_scan_system_health_reports_missing_paths_and_tabs(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    report = loop._scan_system_health()

    assert isinstance(report, dict)
    assert report["healthy"] is False
    issues = "\n".join(report.get("issues", []))
    assert "PATH_MISSING" in issues
    assert "UI_TABS_MISSING" in issues


def test_scan_system_health_detects_profile_mismatch(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    os.environ["DATA_USER_ID"] = "different_profile"
    try:
        report = loop._scan_system_health()
        issues = "\n".join(report.get("issues", []))
        assert "PROFILE_MISMATCH" in issues
    finally:
        os.environ.pop("DATA_USER_ID", None)


# ---------------------------------------------------------------------------
# _run_system_validation_suite
# ---------------------------------------------------------------------------

def test_validation_suite_uses_compile_and_optional_tests(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "UI" / "tabs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "UI").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)

    for rel in [
        "src/optimizer.py",
        "src/main.py",
        "UI/app.py",
        "UI/runtime.py",
        "UI/tabs/edge_tab.py",
        "tests/test_gold_output.py",
    ]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("print('ok')\n", encoding="utf-8")

    os.environ["OPTIMIZER_VALIDATION_TESTS"] = "tests/test_gold_output.py"
    try:
        ok, msg = loop._run_system_validation_suite()
        assert ok is True
        assert "Validation passed" in msg
    finally:
        os.environ.pop("OPTIMIZER_VALIDATION_TESTS", None)


# ---------------------------------------------------------------------------
# _run_pytest_and_capture_failures
# ---------------------------------------------------------------------------

def test_run_pytest_capture_failures_no_tests_found(tmp_path, monkeypatch):
    """When there are no test files, the subsystem returns gracefully."""
    loop = _loop_stub(tmp_path, monkeypatch)
    result = loop._run_pytest_and_capture_failures(test_paths=[])
    assert result["passed"] is True
    assert result["failed_count"] == 0
    assert result["failures"] == []


def test_run_pytest_capture_failures_parses_failed_lines(tmp_path, monkeypatch):
    """Verify that FAILED lines are parsed into structured failure dicts."""
    loop = _loop_stub(tmp_path, monkeypatch)
    fake_output = (
        "FAILED tests/test_foo.py::test_bar - AssertionError: 1 != 2\n"
        "  assert 1 == 2\n"
        "FAILED tests/test_baz.py::test_qux\n"
        "  some other error\n"
        "2 failed, 5 passed in 0.30s\n"
    )
    loop._run_cmd = lambda *args, **kwargs: (False, fake_output)  # type: ignore[attr-defined]

    result = loop._run_pytest_and_capture_failures(test_paths=["tests/fake.py"])
    assert result["passed"] is False
    assert result["failed_count"] == 2
    assert len(result["failures"]) == 2
    assert result["failures"][0]["test"] == "tests/test_foo.py::test_bar"


def test_run_pytest_capture_failures_all_pass(tmp_path, monkeypatch):
    """When pytest succeeds, failed_count is 0 and failures list is empty."""
    loop = _loop_stub(tmp_path, monkeypatch)
    loop._run_cmd = lambda *args, **kwargs: (True, "5 passed in 0.20s")  # type: ignore[attr-defined]

    result = loop._run_pytest_and_capture_failures(test_paths=["tests/fake.py"])
    assert result["passed"] is True
    assert result["failed_count"] == 0


# ---------------------------------------------------------------------------
# _scan_pipeline_stages
# ---------------------------------------------------------------------------

def test_scan_pipeline_stages_all_missing(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    result = loop._scan_pipeline_stages()
    issues = "\n".join(result.get("stage_issues", []))
    # At minimum bronze, gold_master, and output_backtest should be missing
    assert "STAGE_MISSING" in issues


def test_scan_pipeline_stages_detects_existing_file(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    bt_path = tmp_path / "output" / "alice" / "backtest_2020.json"
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    bt_path.write_text('{"strategy_returns": []}', encoding="utf-8")

    result = loop._scan_pipeline_stages()
    details = result.get("stage_details", {})
    assert details.get("output_backtest", {}).get("exists") is True


# ---------------------------------------------------------------------------
# _scan_source_modules
# ---------------------------------------------------------------------------

def test_scan_source_modules_missing_files(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    result = loop._scan_source_modules()
    # All modules will be missing in tmp_path
    assert len(result.get("missing_modules", [])) > 0


def test_scan_source_modules_detects_syntax_error(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    broken = tmp_path / "src" / "optimizer.py"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("def bad(:\n    pass\n", encoding="utf-8")

    result = loop._scan_source_modules()
    assert len(result.get("syntax_errors", [])) > 0


def test_scan_source_modules_valid_file_no_error(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    src = tmp_path / "src" / "main.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("x = 1\n", encoding="utf-8")

    result = loop._scan_source_modules()
    # main.py exists and is valid — should not appear in syntax_errors
    assert "src/main.py" not in result.get("syntax_errors", [])


# ---------------------------------------------------------------------------
# _scan_metrics_completeness
# ---------------------------------------------------------------------------

def test_scan_metrics_completeness_missing_keys(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    bt_path = tmp_path / "output" / "alice" / "backtest_2020.json"
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    bt_path.write_text('{"strategy_returns": [0.01, 0.02]}', encoding="utf-8")

    result = loop._scan_metrics_completeness()
    issues = "\n".join(result.get("metrics_issues", []))
    assert "METRICS_MISSING_KEYS" in issues


def test_scan_metrics_completeness_null_values(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    bt_path = tmp_path / "output" / "alice" / "backtest_2020.json"
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_returns": [0.01],
        "maximum_drawdown": None,
        "sharpe_ratio": None,
        "calmar_ratio": None,
        "information_ratio": None,
        "profit_factor": None,
        "expectancy_per_trade": None,
    }
    bt_path.write_text(json.dumps(payload), encoding="utf-8")

    result = loop._scan_metrics_completeness()
    issues = "\n".join(result.get("metrics_issues", []))
    assert "METRICS_NULL_VALUES" in issues


def test_scan_metrics_completeness_healthy(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    bt_path = tmp_path / "output" / "alice" / "backtest_2020.json"
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_returns": [0.01],
        "maximum_drawdown": -0.05,
        "sharpe_ratio": 1.2,
        "calmar_ratio": 0.8,
        "information_ratio": 0.5,
        "profit_factor": 1.3,
        "expectancy_per_trade": 0.002,
    }
    bt_path.write_text(json.dumps(payload), encoding="utf-8")

    result = loop._scan_metrics_completeness()
    assert result.get("metrics_issues") == []


# ---------------------------------------------------------------------------
# _extended_scan_system_health
# ---------------------------------------------------------------------------

def test_extended_scan_merges_all_sub_scans(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    result = loop._extended_scan_system_health()
    assert isinstance(result, dict)
    issues = "\n".join(result.get("issues", []))
    # Expect at minimum path, stage, and module issues in a blank tmp_path
    assert "PATH_MISSING" in issues or "STAGE_MISSING" in issues or "MODULE_MISSING" in issues
    assert result["healthy"] is False
    assert "stage_details" in result.get("checks", {})


# ---------------------------------------------------------------------------
# _run_test_suite_subsystem (disabled path)
# ---------------------------------------------------------------------------

def test_run_test_suite_subsystem_disabled(tmp_path, monkeypatch):
    loop = _loop_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("OPTIMIZER_TEST_SUITE_SCAN_ENABLED", "0")
    result = loop._run_test_suite_subsystem()
    assert result.get("ran") is False
    assert result.get("status") == "disabled"


def test_run_test_suite_subsystem_all_passing(tmp_path, monkeypatch):
    """When suite passes, subsystem reports all_tests_passing without changes."""
    loop = _loop_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("OPTIMIZER_TEST_SUITE_SCAN_ENABLED", "1")

    # Patch _run_pytest_and_capture_failures to simulate a fully passing suite
    loop._run_pytest_and_capture_failures = lambda **kwargs: {  # type: ignore[attr-defined]
        "passed": True,
        "total": 10,
        "failed_count": 0,
        "failures": [],
        "raw_output": "10 passed",
    }
    result = loop._run_test_suite_subsystem()
    assert result.get("ran") is True
    assert result.get("changed") is False
    assert "passing" in result.get("status", "")

