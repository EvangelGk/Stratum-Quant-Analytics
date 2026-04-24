import json
from pathlib import Path
from types import SimpleNamespace

import UI.app as app
from UI.tabs import edge_tab
from pathing import build_output_dir, output_path_diagnostics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_backtest_payload() -> dict:
    return {
        "value": {
            "strategy_returns": [0.01, -0.005, 0.002],
            "maximum_drawdown": -0.03,
            "sharpe_ratio": 1.1,
        }
    }


def test_initialize_paths_for_main_uses_env_resolution_when_session_missing(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "initialize_active_paths", lambda analyst_id=None: calls.append(analyst_id) or {})
    monkeypatch.setattr(app, "st", SimpleNamespace(session_state={}))

    app._initialize_paths_for_main()

    assert calls == [None]


def test_initialize_paths_for_main_preserves_session_analyst(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "initialize_active_paths", lambda analyst_id=None: calls.append(analyst_id) or {})
    monkeypatch.setattr(app, "st", SimpleNamespace(session_state={"analyst_id": "desk_alpha"}))

    app._initialize_paths_for_main()

    assert calls == ["desk_alpha"]


def test_discover_backtest_payload_does_not_cross_profiles_by_default(monkeypatch, tmp_path):
    output_root = tmp_path / "output"
    active_dir = output_root / "active_profile"
    sibling_dir = output_root / "other_profile"
    active_dir.mkdir(parents=True, exist_ok=True)
    sibling_dir.mkdir(parents=True, exist_ok=True)
    _write_json(sibling_dir / "backtest_2020.json", _valid_backtest_payload())

    monkeypatch.setattr(edge_tab, "_paths", lambda: {"output": active_dir})
    monkeypatch.delenv("EDGE_TAB_ALLOW_CROSS_PROFILE_FALLBACK", raising=False)

    payload, source = edge_tab._discover_backtest_payload()

    assert payload == {}
    assert source is None


def test_discover_backtest_payload_can_cross_profiles_when_explicitly_enabled(monkeypatch, tmp_path):
    output_root = tmp_path / "output"
    active_dir = output_root / "active_profile"
    sibling_dir = output_root / "other_profile"
    active_dir.mkdir(parents=True, exist_ok=True)
    sibling_dir.mkdir(parents=True, exist_ok=True)
    expected_path = sibling_dir / "backtest_2020.json"
    _write_json(expected_path, _valid_backtest_payload())

    monkeypatch.setattr(edge_tab, "_paths", lambda: {"output": active_dir})
    monkeypatch.setenv("EDGE_TAB_ALLOW_CROSS_PROFILE_FALLBACK", "1")

    payload, source = edge_tab._discover_backtest_payload()

    assert isinstance(payload, dict)
    assert payload.get("strategy_returns")
    assert source == expected_path


def test_output_path_diagnostics_match_active_output_dir(tmp_path):
    active_output_dir = build_output_dir(tmp_path, "default")

    search_line, active_output_line = output_path_diagnostics(active_output_dir)

    assert search_line == f"🔍 Searching in: {tmp_path / 'output'}"
    assert active_output_line == f"🔍 Active OUTPUT_DIR: {active_output_dir}"
