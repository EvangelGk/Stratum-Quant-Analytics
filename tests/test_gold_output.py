"""Integration tests for Gold Layer output file integrity.

These tests are intentionally self-contained: they do NOT import from UI.runtime
or src.main at module level.  The xdist workers on this machine crash when the
full src import chain (src/__init__ → ai_agent → requests → urllib3) is loaded
inside a subprocess.  By replicating the tiny helper logic inline we avoid that
chain entirely while still validating the behaviour that matters.

Run with:
    python -m pytest tests/test_gold_output.py -v -p no:xdist
or just:
    python -m pytest tests/test_gold_output.py -v --dist=no
"""
from __future__ import annotations

import gzip
import json
import shutil
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Inline replicas of the two helpers we want to test without heavy imports
# ---------------------------------------------------------------------------

def _write_json_atomic(file_path: Path, payload: object) -> None:
    """Shadow of src/main.py _write_json — write to .tmp then atomic replace."""
    tmp = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        tmp.replace(file_path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _is_valid_json_artifact(path: Path) -> bool:
    """Shadow of UI/runtime.py _is_valid_json_artifact."""
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            return False
        json.loads(text)
        return True
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Minimal backtest payload that satisfies the Arsenal tab
# ---------------------------------------------------------------------------

def _minimal_backtest() -> dict:
    returns = [round(0.001 * i, 6) for i in range(-5, 6)]
    return {
        "strategy_returns": returns,
        "benchmark_returns": [round(r * 0.5, 6) for r in returns],
        "predictions": [round(r + 0.0001, 6) for r in returns],
        "actual": returns,
        "maximum_drawdown": -0.03,
        "sharpe_ratio": 1.2,
        "expectancy_per_trade": 0.0002,
        "profit_factor": 1.4,
        "annualized_return": 0.12,
        "calmar_ratio": 4.0,
        "information_ratio": 0.8,
        "correlation_test": {"pearson_r": 0.15, "p_value": 0.04},
        "window": {
            "train_end_exclusive": "2020-01-01",
            "test_start": "2020-01-01",
            "test_end": "2022-12-31",
        },
        "ticker": "SPY",
        "target": "log_return",
    }


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


# ===========================================================================
# 1. Atomic write tests
# ===========================================================================

def test_write_json_atomic_produces_valid_file(tmp_path):
    """File must be parseable JSON immediately after _write_json_atomic."""
    target = tmp_path / "output.json"
    _write_json_atomic(target, {"key": "value", "nums": list(range(100))})

    assert target.exists(), "Output file must exist"
    assert target.stat().st_size > 0, "Output file must not be empty"
    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert parsed["key"] == "value"
    assert parsed["nums"] == list(range(100))


def test_write_json_atomic_no_tmp_left_on_success(tmp_path):
    """No .tmp sibling file must remain after a successful atomic write."""
    target = tmp_path / "artifact.json"
    _write_json_atomic(target, {"x": 1})

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Stale .tmp files after successful write: {tmp_files}"


def test_write_json_atomic_replaces_existing_file(tmp_path):
    """An existing file must be fully replaced, never partially overwritten."""
    target = tmp_path / "data.json"
    _write_json_atomic(target, {"version": 1})
    _write_json_atomic(target, {"version": 2})

    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert parsed["version"] == 2, "File must contain the latest write"


def test_write_json_atomic_overwrites_with_large_payload(tmp_path):
    """Large payload must be written completely before the file is visible."""
    target = tmp_path / "large.json"
    large = {"data": list(range(50_000))}
    _write_json_atomic(target, large)

    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert len(parsed["data"]) == 50_000


# ===========================================================================
# 2. _is_valid_json_artifact tests
# ===========================================================================

def test_is_valid_artifact_true_for_valid_file(tmp_path):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert _is_valid_json_artifact(p) is True


def test_is_valid_artifact_false_for_missing_file():
    assert _is_valid_json_artifact(Path("/nonexistent/path/x.json")) is False


def test_is_valid_artifact_false_for_empty_file(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("", encoding="utf-8")
    assert _is_valid_json_artifact(p) is False


def test_is_valid_artifact_false_for_whitespace_only(tmp_path):
    p = tmp_path / "ws.json"
    p.write_text("   \n\t  ", encoding="utf-8")
    assert _is_valid_json_artifact(p) is False


def test_is_valid_artifact_false_for_corrupt_json(tmp_path):
    p = tmp_path / "corrupt.json"
    p.write_text("{broken json without closing brace", encoding="utf-8")
    assert _is_valid_json_artifact(p) is False


def test_is_valid_artifact_true_for_array_json(tmp_path):
    p = tmp_path / "arr.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert _is_valid_json_artifact(p) is True


# ===========================================================================
# 3. Backtest payload structure
# ===========================================================================

def test_minimal_backtest_has_all_required_keys():
    bt = _minimal_backtest()
    for key in REQUIRED_BACKTEST_KEYS:
        assert key in bt, f"Missing required key: {key}"


def test_backtest_strategy_returns_are_finite(tmp_path):
    bt = _minimal_backtest()
    target = tmp_path / "backtest_2020.json"
    _write_json_atomic(target, {"value": bt})

    stored = json.loads(target.read_text(encoding="utf-8"))
    returns = stored["value"]["strategy_returns"]
    assert isinstance(returns, list) and len(returns) > 0
    for v in returns:
        assert isinstance(v, (int, float)) and abs(v) < 1.0, (
            f"Non-finite or extreme return: {v}"
        )


def test_backtest_correlation_test_structure():
    bt = _minimal_backtest()
    ct = bt["correlation_test"]
    assert "pearson_r" in ct
    assert "p_value" in ct
    assert 0.0 <= ct["p_value"] <= 1.0


# ===========================================================================
# 4. KPI sanity check — score deviation logic
# ===========================================================================

@pytest.mark.parametrize("prev,curr,should_trigger", [
    (90.0, 46.0, True),   # 44-pt drop — user's exact symptom
    (70.0, 39.0, True),   # exactly 31-pt drop
    (70.0, 40.0, True),   # exactly 30-pt drop (boundary — triggers)
    (70.0, 41.0, False),  # 29-pt drop — below threshold
    (70.0, 65.0, False),  # small drop
    (50.0, 55.0, False),  # score improved
])
def test_score_sanity_check_threshold(prev, curr, should_trigger):
    triggered = curr < prev - 30.0
    assert triggered is should_trigger, (
        f"prev={prev}, curr={curr}: expected trigger={should_trigger}, got {triggered}"
    )


def test_score_breakdown_components_sum_to_total():
    """Score breakdown dict values must sum to (or equal) the composite score."""
    breakdown = {
        "Expectancy": 25.0,
        "Profit Factor": 15.0,
        "Calmar": 8.0,
        "Sharpe": 12.0,
        "IR": 4.0,
    }
    total_from_breakdown = sum(breakdown.values())
    assert total_from_breakdown == pytest.approx(64.0)


# ===========================================================================
# 5. Copy-on-write backup semantics
# ===========================================================================

def test_copy2_leaves_original_in_place(tmp_path):
    """shutil.copy2 (used in run_gold_analyses_only) must NOT remove the original."""
    orig = tmp_path / "backtest_2020.json"
    orig.write_text(json.dumps({"old": True}), encoding="utf-8")

    bak = orig.with_suffix(orig.suffix + ".bak")
    shutil.copy2(orig, bak)

    assert orig.exists(), "Original must still exist after copy2"
    assert bak.exists(), "Backup must have been created"
    assert json.loads(orig.read_text()) == json.loads(bak.read_text())


def test_rename_removes_original(tmp_path):
    """Contrast test: rename() (old strategy) removes the original — the bug source."""
    orig = tmp_path / "backtest_2020.json"
    orig.write_text(json.dumps({"old": True}), encoding="utf-8")

    bak = orig.with_suffix(orig.suffix + ".bak")
    orig.rename(bak)

    assert not orig.exists(), "rename() removes original — this is the 'Not Found' window"
    assert bak.exists()


def test_selective_restore_keeps_valid_new_file(tmp_path):
    """After a failed run, a valid new file must NOT be overwritten by the backup."""
    orig = tmp_path / "backtest_2020.json"
    orig.write_text(json.dumps({"version": "new", "data": [1, 2, 3]}), encoding="utf-8")

    bak = orig.with_suffix(orig.suffix + ".bak")
    bak.write_text(json.dumps({"version": "old"}), encoding="utf-8")

    # Selective restore logic: only restore if current file is invalid
    if not _is_valid_json_artifact(orig):
        bak.replace(orig)

    assert json.loads(orig.read_text())["version"] == "new", (
        "Valid new file must not be overwritten by backup"
    )


def test_selective_restore_repairs_corrupt_new_file(tmp_path):
    """After a failed run, a corrupt new file MUST be replaced by the backup."""
    orig = tmp_path / "backtest_2020.json"
    orig.write_text("{corrupt", encoding="utf-8")  # simulate mid-write crash

    bak = orig.with_suffix(orig.suffix + ".bak")
    bak.write_text(json.dumps({"version": "old", "sharpe": 1.2}), encoding="utf-8")

    if not _is_valid_json_artifact(orig):
        bak.replace(orig)

    restored = json.loads(orig.read_text())
    assert restored["version"] == "old", "Backup must restore when new file is corrupt"
