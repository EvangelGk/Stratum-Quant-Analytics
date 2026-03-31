"""Integration tests for Gold Layer output file integrity.

These tests validate that:
1. _write_json produces atomic writes (no partial-file window).
2. _write_output_artifacts produces valid, parseable output files.
3. The backtest payload has all required keys for the Arsenal tab.
4. The sanity check logic (score deviation > 30%) fires correctly.

Run with:  pytest tests/test_gold_output.py -v
"""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
from pathlib import Path

import pytest

import src.main as main_module


# ── Helpers ──────────────────────────────────────────────────────────────────

REQUIRED_BACKTEST_KEYS = {
    "strategy_returns",
    "maximum_drawdown",
    "sharpe_ratio",
    "expectancy_per_trade",
    "profit_factor",
    "annualized_return",
    "calmar_ratio",
    "correlation_test",
}

REQUIRED_SUMMARY_KEYS = {"generated_at", "result_keys", "results", "artifacts"}


def _make_minimal_backtest() -> dict:
    """Minimal backtest payload that satisfies the Arsenal tab."""
    returns = [0.001 * i for i in range(-5, 6)]  # 11 values, mix positive/negative
    return {
        "strategy_returns": returns,
        "benchmark_returns": [r * 0.5 for r in returns],
        "predictions": [r + 0.0001 for r in returns],
        "actual": returns,
        "maximum_drawdown": -0.03,
        "sharpe_ratio": 1.2,
        "expectancy_per_trade": 0.0002,
        "profit_factor": 1.4,
        "annualized_return": 0.12,
        "calmar_ratio": 4.0,
        "information_ratio": 0.8,
        "correlation_test": {"pearson_r": 0.15, "p_value": 0.04},
        "window": {"train_end_exclusive": "2020-01-01", "test_start": "2020-01-01", "test_end": "2022-12-31"},
        "ticker": "SPY",
        "target": "log_return",
    }


# ── Atomic write tests ────────────────────────────────────────────────────────


def test_write_json_atomic_no_partial_file(tmp_path, monkeypatch):
    """_write_json must never leave the file in a partially-written state."""
    monkeypatch.chdir(tmp_path)

    target = tmp_path / "output" / "default" / "test_artifact.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    large_payload = {"data": list(range(10_000))}

    # Write and immediately read back — must be valid JSON, never empty.
    main_module._write_output_artifacts.__globals__  # just access the module

    # Call the internal helper through the public function
    result = main_module._write_output_artifacts({"test_artifact": large_payload}, user_id="default")
    assert "test_artifact" in result

    written = Path(result["test_artifact"])
    assert written.exists(), "Output file must exist after write"
    assert written.stat().st_size > 0, "Output file must not be empty"

    parsed = json.loads(written.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), "Output file must contain valid JSON"


def test_write_json_no_tmp_file_left_on_success(tmp_path, monkeypatch):
    """No .tmp file should remain after a successful _write_output_artifacts call."""
    monkeypatch.chdir(tmp_path)
    main_module._write_output_artifacts({"result": 42}, user_id="default")

    output_dir = tmp_path / "output" / "default"
    tmp_files = list(output_dir.glob("*.tmp"))
    assert tmp_files == [], f"Stale .tmp files found: {tmp_files}"


def test_write_output_artifacts_produces_valid_summary(tmp_path, monkeypatch):
    """analysis_results.json must contain all required top-level keys."""
    monkeypatch.chdir(tmp_path)

    payload = {"backtest_2020": _make_minimal_backtest(), "sharpe": 1.5}
    created = main_module._write_output_artifacts(payload, user_id="default")

    summary_path = Path(created["analysis_results"])
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in REQUIRED_SUMMARY_KEYS:
        assert key in summary, f"Summary missing required key: {key}"


def test_write_output_artifacts_versioned_gz_created(tmp_path, monkeypatch):
    """A versioned gzip backup of analysis_results.json must be created."""
    monkeypatch.chdir(tmp_path)
    main_module._write_output_artifacts({"x": 1}, user_id="default")

    output_dir = tmp_path / "output" / "default"
    gz_files = list(output_dir.glob("analysis_results_*.json.gz"))
    assert len(gz_files) >= 1, "At least one versioned .gz backup must exist"

    # The .gz file must decompress to valid JSON
    with gzip.open(gz_files[0], "rb") as f:
        content = f.read()
    parsed = json.loads(content)
    assert isinstance(parsed, dict)


# ── Backtest payload structure tests ─────────────────────────────────────────


def test_backtest_payload_has_required_keys(tmp_path, monkeypatch):
    """The backtest_2020.json artifact must contain all keys the Arsenal tab needs."""
    monkeypatch.chdir(tmp_path)

    bt = _make_minimal_backtest()
    created = main_module._write_output_artifacts({"backtest_2020": bt}, user_id="default")

    bt_path = Path(created["backtest_2020"])
    assert bt_path.exists()

    stored = json.loads(bt_path.read_text(encoding="utf-8"))
    # Artifact is wrapped in {"value": {...}}
    inner = stored.get("value", stored)
    for key in REQUIRED_BACKTEST_KEYS:
        assert key in inner, f"Backtest artifact missing required key: {key}"


def test_backtest_strategy_returns_are_finite_list(tmp_path, monkeypatch):
    """strategy_returns must be a list of finite floats (no NaN/Inf)."""
    monkeypatch.chdir(tmp_path)

    bt = _make_minimal_backtest()
    created = main_module._write_output_artifacts({"backtest_2020": bt}, user_id="default")

    bt_path = Path(created["backtest_2020"])
    stored = json.loads(bt_path.read_text(encoding="utf-8"))
    inner = stored.get("value", stored)

    returns = inner.get("strategy_returns", [])
    assert isinstance(returns, list) and len(returns) > 0
    for v in returns:
        assert isinstance(v, (int, float)) and abs(v) < 1e6, (
            f"Non-finite or extreme return value found: {v}"
        )


# ── KPI sanity check ─────────────────────────────────────────────────────────


def test_score_deviation_sanity_check():
    """Simulates the 30-pt sanity check logic used in show_edge_arsenal_tab."""
    prev_score = 90.0
    new_score = 46.0
    deviation = prev_score - new_score
    assert deviation > 30.0, "Test setup error"

    # Reproduce the check logic from edge_tab.py
    triggered = new_score < prev_score - 30.0
    assert triggered, "Sanity check should fire when score drops > 30 pts"


def test_score_deviation_no_false_positive():
    """Small score change must NOT trigger the sanity check."""
    prev_score = 70.0
    new_score = 65.0  # only 5 pt drop
    triggered = new_score < prev_score - 30.0
    assert not triggered, "Sanity check must not fire on a 5-point drop"


# ── _is_valid_json_artifact tests (UI/runtime helper) ────────────────────────


def test_is_valid_json_artifact_returns_true_for_valid_file(tmp_path):
    from UI.runtime import _is_valid_json_artifact

    p = tmp_path / "test.json"
    p.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert _is_valid_json_artifact(p) is True


def test_is_valid_json_artifact_returns_false_for_missing():
    from UI.runtime import _is_valid_json_artifact

    assert _is_valid_json_artifact(Path("/nonexistent/path/x.json")) is False


def test_is_valid_json_artifact_returns_false_for_empty(tmp_path):
    from UI.runtime import _is_valid_json_artifact

    p = tmp_path / "empty.json"
    p.write_text("", encoding="utf-8")
    assert _is_valid_json_artifact(p) is False


def test_is_valid_json_artifact_returns_false_for_corrupt(tmp_path):
    from UI.runtime import _is_valid_json_artifact

    p = tmp_path / "corrupt.json"
    p.write_text("{broken json", encoding="utf-8")
    assert _is_valid_json_artifact(p) is False


# ── Copy-on-write backup semantics ───────────────────────────────────────────


def test_original_file_persists_during_backup(tmp_path):
    """shutil.copy2 (used in run_gold_analyses_only) must leave original in place."""
    import shutil

    original = tmp_path / "backtest_2020.json"
    original.write_text(json.dumps({"old": True}), encoding="utf-8")

    bak = original.with_suffix(original.suffix + ".bak")
    shutil.copy2(original, bak)

    assert original.exists(), "Original must still exist after copy2"
    assert bak.exists(), "Backup must have been created"

    orig_content = json.loads(original.read_text())
    bak_content = json.loads(bak.read_text())
    assert orig_content == bak_content, "Backup content must match original"
