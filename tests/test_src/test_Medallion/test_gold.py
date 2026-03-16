import pandas as pd

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
                "date": pd.to_datetime([
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-04-01",
                ]),
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
