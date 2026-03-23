import json

import pandas as pd
import pytest

import src.Medallion.gold.GoldLayer as gold_module


class DummyConfig:
    pass


def test_run_all_analyses_with_stubbed_functions(monkeypatch):
    # Stub the master table load to avoid file I/O
    monkeypatch.setattr(
        gold_module.GoldLayer,
        "_load_or_create_master_table",
        lambda self: pd.DataFrame(
            {
                "ticker": ["A", "A"],
                "close": [100.0, 101.0],
                "log_return": [0.01, 0.02],
                "inflation": [0.01, 0.02],
                "energy_index": [0.1, 0.2],
                "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            }
        ),
    )

    # Patch analysis functions to avoid heavy computation
    monkeypatch.setattr(gold_module, "correl_mtrx", lambda df: "corr")
    monkeypatch.setattr(gold_module, "elasticity", lambda df, a, b: 0.5)
    monkeypatch.setattr(gold_module, "lag_analysis", lambda df, a, b: {"lag_1": 0.1})
    monkeypatch.setattr(gold_module, "monte_carlo", lambda df, t, **kwargs: "mc")
    monkeypatch.setattr(gold_module, "stress_test", lambda df, m: {"shock": "ok"})
    monkeypatch.setattr(
        gold_module, "sensitivity_reg", lambda df, t, f, m: {"coefficients": {}}
    )

    gold = gold_module.GoldLayer(DummyConfig())
    results = gold.run_all_analyses(
        ticker="A",
        macro_factor="inflation",
        lags=1,
        shock_map={"inflation": 0.1},
        target="log_return",
        factors=["inflation"],
    )

    assert results["correlation_matrix"] == "corr"
    assert results["elasticity"] == 0.5
    assert results["lag_analysis"] == {"lag_1": 0.1}
    assert results["monte_carlo"] == "mc"
    assert results["stress_test"]["shock"] == "ok"
    assert "sensitivity_regression" in results
    assert "governance_report" in results


def test_run_all_analyses_parallel_uses_executor(monkeypatch):
    # Provide a minimal master table and patch analysis functions
    monkeypatch.setattr(
        gold_module.GoldLayer,
        "_load_or_create_master_table",
        lambda self: pd.DataFrame(
            {
                "ticker": ["A"],
                "close": [100.0],
                "log_return": [0.01],
                "inflation": [0.01],
                "energy_index": [0.1],
                "date": pd.to_datetime(["2020-01-01"]),
            }
        ),
    )

    def dummy_fn(*args, **kwargs):
        return "ok"

    monkeypatch.setattr(gold_module, "correl_mtrx", lambda df: "corr")
    monkeypatch.setattr(gold_module, "lag_analysis", lambda df, f, lags: {"lag_1": 0.0})
    monkeypatch.setattr(
        gold_module, "sensitivity_reg", lambda df, t, f, m: {"coefficients": {}}
    )
    monkeypatch.setattr(
        gold_module, "forecasting", lambda df, t, steps: pd.Series([0.0] * steps)
    )
    monkeypatch.setattr(
        gold_module, "auto_ml_regression", lambda df, t, f: {"best_model": "X"}
    )

    # Stub the executor to avoid multiprocessing overhead
    class DummyFuture:
        def __init__(self, fn):
            self._fn = fn

        def result(self):
            return self._fn()

    class DummyExecutor:
        def __init__(self, max_workers=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return DummyFuture(lambda: fn(*args, **kwargs))

    monkeypatch.setattr(
        gold_module.concurrent.futures, "ThreadPoolExecutor", DummyExecutor
    )

    gold = gold_module.GoldLayer(DummyConfig())
    results = gold.run_all_analyses_parallel(
        ticker="A",
        macro_factor="inflation",
        lags=1,
        shock_map={"inflation": 0.1},
        target="log_return",
        factors=["inflation"],
        max_workers=2,
    )

    assert results["correlation_matrix"] == "corr"
    assert results["sensitivity_regression"] == {"coefficients": {}}
    assert "governance_report" in results


def test_run_all_analyses_default_uses_dynamic_factor_universe(monkeypatch):
    monkeypatch.setattr(
        gold_module.GoldLayer,
        "_load_or_create_master_table",
        lambda self: pd.DataFrame(
            {
                "ticker": ["A", "A", "A", "A"],
                "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
                "log_return": [0.01, 0.02, -0.01, 0.03],
                "close": [100.0, 101.0, 99.5, 103.0],
                "volume": [1_000_000, 1_100_000, 900_000, 1_200_000],
                "inflation": [0.01, 0.012, 0.011, 0.013],
                "energy_index": [0.2, 0.21, 0.19, 0.22],
                "vix_index": [18.0, 19.5, 21.0, 17.0],
            }
        ),
    )

    monkeypatch.setattr(gold_module, "correl_mtrx", lambda df, *args, **kwargs: "corr")
    monkeypatch.setattr(
        gold_module,
        "governance_report",
        lambda *args, **kwargs: {
            "status": "ok",
            "out_of_sample": {"r2": 0.1},
            "stability": {"normalized_mean_shift": 0.1},
            "leakage_flags": [],
            "stationarity": {"log_return": {"is_stationary": True}},
            "walk_forward": {"avg_r2": 0.05},
            "model_risk_score": 0.2,
        },
    )
    monkeypatch.setattr(gold_module, "elasticity", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(gold_module, "lag_analysis", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(gold_module, "feature_decay_analysis", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(gold_module, "monte_carlo", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(gold_module, "stress_test", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(gold_module, "forecasting", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(gold_module, "backtest_pre2020_holdout", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(gold_module, "auto_ml_regression", lambda *args, **kwargs: {"ok": True})

    captured: dict[str, list[str]] = {"factors": []}

    def _capture_sensitivity(df, t, f, m, ticker=None, macro_lag_days=0):
        captured["factors"] = list(f)
        return {"coefficients": {}}

    monkeypatch.setattr(gold_module, "sensitivity_reg", _capture_sensitivity)

    cfg = DummyConfig()
    cfg.enforce_reproducibility = True
    cfg.random_seed = 42
    cfg.governance_hard_fail = False
    cfg.governance_walk_forward_windows = 2
    cfg.auto_ml_enabled = True

    gold = gold_module.GoldLayer(cfg)
    _ = gold.run_all_analyses(ticker="A", factors=None)

    assert captured["factors"]
    assert "inflation" in captured["factors"]
    assert "energy_index" in captured["factors"]
    assert "vix_index" in captured["factors"]


@pytest.mark.governance
def test_governance_hard_fail_blocks_advanced_analyses(monkeypatch):
    monkeypatch.setattr(
        gold_module.GoldLayer,
        "_load_or_create_master_table",
        lambda self: pd.DataFrame(
            {
                "ticker": ["A", "A", "A", "A"],
                "close": [100.0, 99.0, 98.0, 97.0],
                "log_return": [-0.01, -0.01, -0.01, -0.01],
                "inflation": [0.1, 0.2, 0.3, 0.4],
                "energy_index": [0.2, 0.3, 0.4, 0.5],
                "date": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-02-01",
                        "2020-03-01",
                        "2020-04-01",
                    ]
                ),
            }
        ),
    )

    monkeypatch.setattr(
        gold_module,
        "governance_report",
        lambda *args, **kwargs: {
            "status": "ok",
            "out_of_sample": {"r2": -0.9},
            "stability": {"normalized_mean_shift": 10.0},
            "leakage_flags": ["x", "y", "z"],
            "stationarity": {
                "log_return": {"is_stationary": False},
                "inflation": {"is_stationary": False},
            },
        },
    )
    monkeypatch.setattr(gold_module, "correl_mtrx", lambda df: "corr")

    cfg = DummyConfig()
    cfg.governance_hard_fail = True
    cfg.governance_min_r2 = -0.25
    cfg.governance_max_normalized_shift = 2.5
    cfg.governance_max_leakage_flags = 1
    cfg.governance_min_stationary_ratio = 0.5
    cfg.enforce_reproducibility = True
    cfg.random_seed = 42

    gold = gold_module.GoldLayer(cfg)
    results = gold.run_all_analyses(ticker="A", factors=["inflation", "energy_index"])

    assert results["correlation_matrix"] == "corr"
    assert results["governance_gate"]["passed"] is False
    assert "blocked_by_governance_gate" in results["elasticity"]


@pytest.mark.governance
def test_governance_risk_band_warn(monkeypatch):
    monkeypatch.setattr(
        gold_module.GoldLayer,
        "_load_or_create_master_table",
        lambda self: pd.DataFrame(
            {
                "ticker": ["A", "A", "A", "A"],
                "close": [100.0, 101.0, 102.0, 103.0],
                "log_return": [0.01, 0.01, 0.01, 0.01],
                "inflation": [0.1, 0.2, 0.3, 0.4],
                "energy_index": [0.2, 0.3, 0.4, 0.5],
                "date": pd.to_datetime(
                    ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]
                ),
            }
        ),
    )
    monkeypatch.setattr(
        gold_module,
        "governance_report",
        lambda *args, **kwargs: {
            "status": "ok",
            "out_of_sample": {"r2": 0.1},
            "stability": {"normalized_mean_shift": 0.2},
            "leakage_flags": [],
            "stationarity": {"log_return": {"is_stationary": True}},
            "walk_forward": {"avg_r2": 0.05},
            "model_risk_score": 0.5,
        },
    )
    monkeypatch.setattr(gold_module, "correl_mtrx", lambda df: "corr")
    monkeypatch.setattr(gold_module, "elasticity", lambda df, a, b: 0.5)
    monkeypatch.setattr(gold_module, "lag_analysis", lambda df, a, b: {"lag_1": 0.1})
    monkeypatch.setattr(gold_module, "monte_carlo", lambda df, t, **kwargs: "mc")
    monkeypatch.setattr(gold_module, "stress_test", lambda df, m: {"shock": "ok"})
    monkeypatch.setattr(
        gold_module, "sensitivity_reg", lambda df, t, f, m: {"coefficients": {}}
    )

    cfg = DummyConfig()
    cfg.governance_hard_fail = True
    cfg.governance_regime = "normal"
    cfg.governance_min_r2 = -0.25
    cfg.governance_max_normalized_shift = 2.5
    cfg.governance_max_leakage_flags = 3
    cfg.governance_min_stationary_ratio = 0.0
    cfg.governance_min_walk_forward_r2 = -0.25
    cfg.governance_max_model_risk_score = 0.9
    cfg.governance_model_risk_warn_threshold = 0.4
    cfg.governance_model_risk_fail_threshold = 0.6
    cfg.governance_walk_forward_windows = 2
    cfg.enforce_reproducibility = True
    cfg.random_seed = 42

    gold = gold_module.GoldLayer(cfg)
    results = gold.run_all_analyses(ticker="A", factors=["inflation", "energy_index"])

    assert results["governance_gate"]["severity"] == "warn"
    assert results["governance_gate"]["passed"] is True


@pytest.mark.governance
def test_governance_decision_artifact_export(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gold_module.GoldLayer,
        "_load_or_create_master_table",
        lambda self: pd.DataFrame(
            {
                "ticker": ["A", "A", "A", "A"],
                "close": [100.0, 101.0, 102.0, 103.0],
                "log_return": [0.01, 0.01, 0.01, 0.01],
                "inflation": [0.1, 0.2, 0.3, 0.4],
                "energy_index": [0.2, 0.3, 0.4, 0.5],
                "date": pd.to_datetime(
                    ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]
                ),
            }
        ),
    )
    monkeypatch.setattr(
        gold_module,
        "governance_report",
        lambda *args, **kwargs: {
            "status": "insufficient_data",
            "rows": 4,
            "required_rows": 30,
        },
    )
    monkeypatch.setattr(gold_module, "correl_mtrx", lambda df: "corr")
    monkeypatch.setattr(gold_module, "elasticity", lambda df, a, b: 0.5)
    monkeypatch.setattr(gold_module, "lag_analysis", lambda df, a, b: {"lag_1": 0.1})
    monkeypatch.setattr(gold_module, "monte_carlo", lambda df, t, **kwargs: "mc")
    monkeypatch.setattr(gold_module, "stress_test", lambda df, m: {"shock": "ok"})
    monkeypatch.setattr(
        gold_module, "sensitivity_reg", lambda df, t, f, m: {"coefficients": {}}
    )

    cfg = DummyConfig()
    cfg.governance_hard_fail = True
    cfg.governance_regime = "normal"
    cfg.governance_min_r2 = -0.25
    cfg.governance_max_normalized_shift = 2.5
    cfg.governance_max_leakage_flags = 1
    cfg.governance_min_stationary_ratio = 0.4
    cfg.governance_min_walk_forward_r2 = -0.25
    cfg.governance_max_model_risk_score = 0.6
    cfg.governance_model_risk_warn_threshold = 0.4
    cfg.governance_model_risk_fail_threshold = 0.6
    cfg.governance_walk_forward_windows = 2
    cfg.enforce_reproducibility = True
    cfg.random_seed = 42

    gold = gold_module.GoldLayer(cfg)
    gold.gold_path = tmp_path / "gold"
    gold.gold_path.mkdir(parents=True, exist_ok=True)
    gold.governance_path = gold.gold_path / "governance"
    gold.governance_path.mkdir(parents=True, exist_ok=True)

    _ = gold.run_all_analyses(ticker="A", factors=["inflation", "energy_index"])

    files = list(gold.governance_path.glob("governance_decision_*.json"))
    assert files
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "governance-decision.v1"
    assert "gate" in payload


@pytest.mark.governance
def test_read_governance_history(tmp_path):
    gold = gold_module.GoldLayer.__new__(gold_module.GoldLayer)
    gold.governance_path = tmp_path / "governance"
    gold.governance_path.mkdir(parents=True, exist_ok=True)

    for ts, score in [(1000000, 0.2), (2000000, 0.3)]:
        payload = {
            "schema_version": "governance-decision.v1",
            "generated_at": "2026-01-01T00:00:00",
            "gate": {"passed": True, "severity": "pass"},
            "report": {"model_risk_score": score},
        }
        (gold.governance_path / f"governance_decision_{ts}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    history = gold.read_governance_history()
    assert len(history) == 2
    assert history[0]["report"]["model_risk_score"] == pytest.approx(0.2)
    assert history[1]["report"]["model_risk_score"] == pytest.approx(0.3)


@pytest.mark.governance
def test_governance_trend_summary(tmp_path):
    gold = gold_module.GoldLayer.__new__(gold_module.GoldLayer)
    gold.governance_path = tmp_path / "governance"
    gold.governance_path.mkdir(parents=True, exist_ok=True)

    scores = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    for i, score in enumerate(scores):
        payload = {
            "schema_version": "governance-decision.v1",
            "generated_at": "2026-01-01T00:00:00",
            "gate": {
                "passed": True,
                "severity": "pass" if score < 0.4 else "warn",
            },
            "report": {
                "model_risk_score": score,
                "walk_forward": {"avg_r2": 0.1},
            },
        }
        (gold.governance_path / f"governance_decision_{1000000 + i}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    summary = gold.governance_trend_summary()
    assert summary["status"] == "ok"
    assert summary["count"] == 6
    assert summary["pass_rate"] == pytest.approx(1.0)
    assert "pass" in summary["severity_distribution"]
    assert summary["direction"] == "deteriorating"
    assert summary["avg_model_risk_score"] is not None
    assert summary["worst_walk_forward_r2"] == pytest.approx(0.1)


@pytest.mark.governance
def test_ticker_override_in_profile(monkeypatch):
    monkeypatch.setattr(
        gold_module.GoldLayer,
        "_load_or_create_master_table",
        lambda self: pd.DataFrame(
            {
                "ticker": ["A"],
                "close": [100.0],
                "log_return": [0.01],
                "date": pd.to_datetime(["2020-01-01"]),
            }
        ),
    )

    class TickerOverrideCfg:
        governance_hard_fail = True
        governance_regime = "normal"
        governance_min_r2 = -0.25
        governance_max_normalized_shift = 2.5
        governance_max_leakage_flags = 1
        governance_min_stationary_ratio = 0.4
        governance_min_walk_forward_r2 = -0.25
        governance_max_model_risk_score = 0.6
        governance_model_risk_warn_threshold = 0.4
        governance_model_risk_fail_threshold = 0.6
        governance_walk_forward_windows = 4
        governance_ticker_overrides = {
            "AAPL": {"min_r2": 0.2, "max_model_risk_score": 0.3}
        }
        enforce_reproducibility = True
        random_seed = 42

    gold = gold_module.GoldLayer(TickerOverrideCfg())
    profile = gold._resolve_governance_profile(ticker="AAPL")
    assert profile["min_r2"] == pytest.approx(0.2)
    assert profile["max_model_risk_score"] == pytest.approx(0.3)

    profile_default = gold._resolve_governance_profile(ticker="TSLA")
    assert profile_default["min_r2"] == pytest.approx(-0.25)
    assert profile_default["max_model_risk_score"] == pytest.approx(0.6)
