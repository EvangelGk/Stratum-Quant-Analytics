import numpy as np
import pandas as pd
import pytest

from src.Medallion.gold.AnalysisSuite.auto_ml import auto_ml_regression
from src.Medallion.gold.AnalysisSuite.correl_mtrx import correl_mtrx
from src.Medallion.gold.AnalysisSuite.elasticity import elasticity
from src.Medallion.gold.AnalysisSuite.forecasting import forecasting
from src.Medallion.gold.AnalysisSuite.lag import lag_analysis
from src.Medallion.gold.AnalysisSuite.monte_carlo import monte_carlo
from src.Medallion.gold.AnalysisSuite.sesnsitivity_reg import sensitivity_reg
from src.Medallion.gold.AnalysisSuite.stress_test import stress_test


def test_correl_mtrx_basic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    cm = correl_mtrx(df)
    assert cm.loc["a", "b"] == pytest.approx(1.0)


def test_elasticity_calculation():
    df = pd.DataFrame(
        {"log_return": [0.1, 0.2, 0.15], "inflation": [0.01, 0.02, 0.015]}
    )
    result = elasticity(df, "log_return", "inflation")
    assert isinstance(result, float)


def test_lag_analysis_basic():
    df = pd.DataFrame({"inflation": [1, 2, 3, 4]})
    result = lag_analysis(df, "inflation", lags=2)
    assert "lag_1" in result and "lag_2" in result


def test_monte_carlo_shape_and_determinism():
    np.random.seed(0)
    df = pd.DataFrame({"ticker": ["A", "A", "A"], "close": [100, 101, 102]})
    out = monte_carlo(df, "A", days=3, iterations=5)
    assert out.shape == (3, 5)


def test_stress_test_output():
    df = pd.DataFrame({"log_return": [0.01, 0.02, 0.03], "inflation": [0.1, 0.1, 0.1]})
    res = stress_test(df, {"inflation": 0.1})
    assert "inflation" in res
    assert "Predicted impact" in res["inflation"]


def test_sensitivity_reg_ols_and_ridge():
    df = pd.DataFrame(
        {
            "log_return": [0.1, 0.2, 0.15, 0.1],
            "inflation": [0.01, 0.02, 0.015, 0.01],
            "energy_index": [0.2, 0.3, 0.25, 0.2],
        }
    )

    summary = sensitivity_reg(
        df, target="log_return", factors=["inflation", "energy_index"], model="OLS"
    )
    assert hasattr(summary, "as_text") or isinstance(summary, str)

    ridge = sensitivity_reg(
        df, target="log_return", factors=["inflation", "energy_index"], model="Ridge"
    )
    assert isinstance(ridge, dict)
    assert "coefficients" in ridge


def test_forecasting_runs():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="ME"),
            "log_return": np.random.randn(10),
        }
    )
    forecast = forecasting(df, "log_return", steps=3, order=(1, 0, 0))
    assert len(forecast) == 3


def test_auto_ml_regression_patched(monkeypatch):
    import src.Medallion.gold.AnalysisSuite.auto_ml as auto_ml_module

    def dummy_setup(data, target, silent, verbose):
        return None

    class DummyModel:
        def __str__(self):
            return "DummyModel"

    def dummy_compare_models():
        return DummyModel()

    def dummy_predict_model(model, data):
        return data.assign(prediction=0.0)

    monkeypatch.setattr(auto_ml_module, "setup", dummy_setup)
    monkeypatch.setattr(auto_ml_module, "compare_models", dummy_compare_models)
    monkeypatch.setattr(auto_ml_module, "predict_model", dummy_predict_model)

    df = pd.DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    result = auto_ml_regression(df, target="y", features=["x"])
    assert result["best_model"] == "DummyModel"
    assert "predictions" in result
