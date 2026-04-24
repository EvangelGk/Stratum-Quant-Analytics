"""Test suite for the AnalysisSuite package.

All analysis functions return structured dicts (not raw scalars/arrays).
Tests validate return type, required top-level keys, and key value types.

Data fixtures use ≥60–120 rows to satisfy the minimum sample requirements
of each analysis function after prepare_supervised_frame transforms (≥30
post-transform rows required by most functions; elasticity needs ≥90 for
its default rolling_window=90).
"""

import numpy as np
import pandas as pd
import pytest

from src.Medallion.gold.AnalysisSuite.auto_ml import auto_ml_regression
from src.Medallion.gold.AnalysisSuite.backtest import (
    _make_model,
    _select_top_features,
    _simulate_risk_managed_returns,
)
from src.Medallion.gold.AnalysisSuite.correl_mtrx import correl_mtrx
from src.Medallion.gold.AnalysisSuite.elasticity import elasticity
from src.Medallion.gold.AnalysisSuite.feature_decay import feature_decay_analysis
from src.Medallion.gold.AnalysisSuite.forecasting import forecasting
from src.Medallion.gold.AnalysisSuite.governance import governance_report
from src.Medallion.gold.AnalysisSuite.lag import lag_analysis
from src.Medallion.gold.AnalysisSuite.mixed_frequency import build_stationary_panel
from src.Medallion.gold.AnalysisSuite.monte_carlo import monte_carlo
from src.Medallion.gold.AnalysisSuite.sensitivity_reg import sensitivity_reg
from src.Medallion.gold.AnalysisSuite.stress_test import (
    PRESET_SCENARIOS,
    resolve_stress_scenario,
    stress_test,
)

# ─── Fixture helpers ──────────────────────────────────────────────────────────


def _make_macro_panel(n: int = 120) -> pd.DataFrame:
    """Synthetic business-day macro panel with realistic-length time series."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    inflation = np.linspace(0.01, 0.04, n) + rng.normal(0, 0.002, n)
    energy = np.linspace(0.20, 0.50, n) + rng.normal(0, 0.02, n)
    log_return = 0.05 * inflation + 0.02 * energy + rng.normal(0, 0.01, n)
    return pd.DataFrame(
        {
            "date": dates,
            "log_return": log_return,
            "inflation": inflation,
            "energy_index": energy,
        }
    )


def _make_ticker_panel(ticker: str = "A", n: int = 60) -> pd.DataFrame:
    """Synthetic OHLCV-style close panel for Monte Carlo tests."""
    rng = np.random.default_rng(0)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, n))
    return pd.DataFrame(
        {
            "ticker": [ticker] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "close": close,
        }
    )


# ─── Correlation matrix ───────────────────────────────────────────────────────


def test_correl_mtrx_basic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    cm = correl_mtrx(df)
    assert cm.loc["a", "b"] == pytest.approx(1.0)


def test_correl_mtrx_stress_mode():
    """Stress mode must set DataFrame.attrs and scale correlations."""
    df = _make_macro_panel(60)[["log_return", "inflation", "energy_index"]]
    cm = correl_mtrx(
        df,
        stress_mode=True,
        stress_strength=0.30,
        scenario_name="geopolitical_conflict",
    )
    assert isinstance(cm, pd.DataFrame)
    assert cm.attrs.get("stress_mode") is True
    assert cm.attrs.get("scenario_name") == "geopolitical_conflict"


# ─── Elasticity ───────────────────────────────────────────────────────────────


def test_elasticity_returns_dict():
    """elasticity() returns a structured dict — not a bare float."""
    df = _make_macro_panel(250)
    result = elasticity(df, "log_return", "inflation")
    assert isinstance(result, dict)
    assert "static_elasticity" in result
    assert isinstance(result["static_elasticity"], float)
    assert "rolling_elasticity" in result
    assert isinstance(result["rolling_elasticity"], list)
    assert "data_points" in result


# ─── Lag analysis ─────────────────────────────────────────────────────────────


def test_lag_analysis_returns_dict():
    """lag_analysis() returns a structured dict with lag_scan list."""
    df = _make_macro_panel(120)
    result = lag_analysis(df, "inflation", lags=3)
    assert isinstance(result, dict)
    assert "lag_scan" in result
    assert "best_lag_days" in result
    assert isinstance(result["lag_scan"], list)
    # lag_scan must have entries for each lag day 0..lags
    lag_days_present = {row["lag_days"] for row in result["lag_scan"]}
    assert 1 in lag_days_present
    assert 2 in lag_days_present
    assert 3 in lag_days_present


# ─── Monte Carlo ──────────────────────────────────────────────────────────────


def test_monte_carlo_returns_dict_with_paths():
    """monte_carlo() returns a dict; price_paths ndarray is at key 'price_paths'."""
    df = _make_ticker_panel("A", n=60)
    result = monte_carlo(df, "A", days=3, iterations=5, random_state=0)
    assert isinstance(result, dict)
    assert "price_paths" in result
    assert result["price_paths"].shape == (3, 5)
    # Full VaR/ES suite must be present
    for key in ("value_at_risk_95", "expected_shortfall_99", "parametric_var_95", "parametric_es_95", "historical_var_95", "historical_es_95"):
        assert key in result, f"Missing key: {key}"


# ─── Stress test ──────────────────────────────────────────────────────────────


def test_stress_test_returns_structured_dict():
    """stress_test() returns results nested under 'results' key."""
    df = _make_macro_panel(250)
    result = stress_test(df, shock_map={"inflation": 0.10})
    assert isinstance(result, dict)
    assert "results" in result
    assert "scenario" in result
    assert "anchor_event" in result
    factor_out = result["results"]["inflation"]
    assert "predicted_impact" in factor_out
    assert "beta" in factor_out
    assert "shock" in factor_out
    assert isinstance(factor_out["predicted_impact"], float)


def test_stress_test_preset_scenario():
    """Preset scenario name is resolved; factor results are returned."""
    df = _make_macro_panel(250)
    # Provide explicit shocks for factors that exist in the test panel,
    # overriding the preset — the merged dict drives the OLS loop.
    result = stress_test(
        df,
        shock_map={"inflation": 0.02, "energy_index": 0.25},
        scenario_name="geopolitical_conflict",
    )
    assert result["scenario"]["name"] == "geopolitical_conflict"
    assert "results" in result
    assert "inflation" in result["results"]
    assert "energy_index" in result["results"]


# ─── Sensitivity regression ───────────────────────────────────────────────────


def test_sensitivity_reg_ols_returns_dict():
    """OLS sensitivity_reg() returns dict with summary_text (not a Summary obj)."""
    df = _make_macro_panel(250)
    summary = sensitivity_reg(df, target="log_return", factors=["inflation", "energy_index"], model="OLS")
    assert isinstance(summary, dict)
    assert summary["model"] == "OLS"
    assert "coefficients" in summary
    assert "r2" in summary
    assert "summary_text" in summary
    assert isinstance(summary["summary_text"], str)
    assert "feature_subset_search" in summary
    assert "lag_selection" in summary


def test_sensitivity_reg_ridge_returns_dict():
    df = _make_macro_panel(250)
    ridge = sensitivity_reg(df, target="log_return", factors=["inflation", "energy_index"], model="Ridge")
    assert isinstance(ridge, dict)
    assert ridge["model"] == "Ridge"
    assert "coefficients" in ridge
    assert "intercept" in ridge
    assert "r2" in ridge
    assert "feature_subset_search" in ridge
    assert "lag_selection" in ridge


# ─── Preset scenarios ─────────────────────────────────────────────────────────


def test_resolve_stress_scenario_preset():
    payload = resolve_stress_scenario("geopolitical_conflict")
    assert payload["name"] == "geopolitical_conflict"
    assert "factor_shocks" in payload
    assert "mc_bias" in payload
    assert "sector_hint" in payload
    assert "correlation_breakdown_strength" in payload
    assert isinstance(payload["correlation_breakdown_strength"], float)


def test_resolve_stress_scenario_custom():
    payload = resolve_stress_scenario(None, shock_map={"inflation": 0.05})
    assert payload["name"] == "custom"
    assert payload["factor_shocks"]["inflation"] == pytest.approx(0.05)


def test_preset_scenarios_completeness():
    """Scenario library must include core four + custom house stress packs."""
    required_keys = {
        "description",
        "factor_shocks",
        "correlation_breakdown_strength",
        "mc_bias",
    }
    minimum_expected_names = {
        "geopolitical_conflict",
        "monetary_tightening",
        "tech_correction",
        "stagflation",
    }
    assert minimum_expected_names.issubset(set(PRESET_SCENARIOS.keys()))
    assert len(PRESET_SCENARIOS) >= 6
    for name, scenario in PRESET_SCENARIOS.items():
        missing = required_keys - set(scenario.keys())
        assert not missing, f"Scenario '{name}' missing keys: {missing}"


def test_publication_lag_semantics_enforced_for_macro_features():
    df = _make_macro_panel(120)
    panel, metadata = build_stationary_panel(
        df=df,
        columns=["inflation", "energy_index"],
        macro_lag_days=0,
    )
    assert not panel.empty
    for feature in ("inflation", "energy_index"):
        assert metadata[feature]["publication_lag_days"] >= 45
        assert str(metadata[feature]["transformation"]).startswith("release_")


def test_stress_test_avoids_target_self_reference():
    df = _make_macro_panel(120)
    result = stress_test(
        df,
        shock_map={"log_return": -0.10, "inflation": 0.02},
        scenario_name="custom",
    )
    # target shock is tracked separately, not regressed against itself.
    assert "log_return" not in result["results"]
    assert result.get("direct_target_shock") == pytest.approx(-0.10)


def test_tech_correction_requires_sector_scope_match():
    df = _make_macro_panel(120)
    result = stress_test(df, shock_map={}, scenario_name="tech_correction")
    assert result.get("sector_scope_applied") == "Technology"
    assert result.get("sector_scope_match") is False
    assert result.get("direct_target_shock") == pytest.approx(0.0)


def test_feature_decay_contract():
    df = _make_macro_panel(120)
    out = feature_decay_analysis(
        df,
        target="log_return",
        features=["inflation", "energy_index"],
        max_lag=30,
    )
    assert out["status"] == "ok"
    assert "results" in out
    assert "inflation" in out["results"]
    assert "half_life_lag_days" in out["results"]["inflation"]
    assert "decay_scan" in out["results"]["inflation"]


# ─── Forecasting ──────────────────────────────────────────────────────────────


def test_forecasting_runs():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="ME"),
            "log_return": np.random.randn(10),
        }
    )
    forecast = forecasting(df, "log_return", steps=3, order=(1, 0, 0))
    assert len(forecast["forecast"]) == 3


# ─── AutoML ───────────────────────────────────────────────────────────────────


def test_auto_ml_regression_patched(monkeypatch):
    import src.Medallion.gold.AnalysisSuite.auto_ml as auto_ml_module
    import src.Medallion.gold.AnalysisSuite.mixed_frequency as mf_module

    rng = np.random.default_rng(42)
    n = 300
    fake_panel = pd.DataFrame(
        {
            "date": pd.date_range("2015-01-01", periods=n, freq="B"),
            "y": rng.normal(0, 0.01, n),
            "x": rng.normal(0, 1, n),
        }
    )
    fake_metadata = {
        "x": {"source": "fred", "lag_days": 0, "transformation": "level", "native_horizon_days": 1},
        "y": {"source": "equity", "lag_days": 0, "transformation": "level", "native_horizon_days": 1},
    }

    monkeypatch.setattr(
        auto_ml_module,
        "prepare_supervised_frame",
        lambda *args, **kwargs: (fake_panel.copy(), dict(fake_metadata)),
    )

    df = pd.DataFrame({"x": rng.normal(0, 1, 10), "y": rng.normal(0, 0.01, 10)})
    result = auto_ml_regression(df, target="y", features=["x"])
    assert result["best_model"]
    assert "predictions" in result


# ─── Governance ───────────────────────────────────────────────────────────────


@pytest.mark.governance
def test_governance_report_runs_with_temporal_split():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2014-01-31", periods=200, freq="ME"),
            "log_return": np.random.randn(200) * 0.02,
            "inflation": np.linspace(0.01, 0.03, 200),
            "energy_index": np.linspace(0.2, 0.4, 200) + np.random.randn(200) * 0.01,
        }
    )

    report = governance_report(
        df,
        target="log_return",
        factors=["inflation", "energy_index"],
    )

    assert report["status"] == "ok"
    assert "out_of_sample" in report
    assert "stability" in report
    assert "leakage_flags" in report
    assert "stationarity" in report
    assert "reproducibility" in report
    assert "multiple_testing" in report
    assert "rolling_regime" in report["stability"]
    assert "walk_forward" in report
    assert "model_risk_score" in report
    assert 0.0 <= report["model_risk_score"] <= 1.0


# ─── Backtest: dual-SMA trend filter behaviour ───────────────────────────────


def test_neutral_zone_signals_are_suppressed():
    """In neutral zone (no golden/death cross) the dual-SMA filter kills all signals.
    The macro model has negative edge there — keeping neutral-zone trades hurts metrics.
    """
    rng = np.random.default_rng(42)
    n = 150
    actual = rng.normal(0, 0.01, n).clip(-0.15, 0.15)
    pred_z = np.full(n, 1.5)  # all long signals

    in_uptrend = np.zeros(n, dtype=bool)   # not in uptrend
    in_downtrend = np.zeros(n, dtype=bool)  # not in downtrend → neutral zone

    _strat_ret, positions = _simulate_risk_managed_returns(
        pred_z=pred_z,
        actual_arr=actual,
        in_uptrend=in_uptrend,
        entry_threshold=0.5,
        downtrend_arr=in_downtrend,
    )

    active_days = int(np.sum(np.abs(positions) > 1e-10))
    assert active_days == 0, (
        "Dual-SMA filter must suppress all signals in the neutral zone."
    )


def test_counter_trend_signals_are_suppressed():
    """Longs in confirmed downtrend (death cross) must be fully suppressed."""
    rng = np.random.default_rng(7)
    n = 150
    actual = rng.normal(0, 0.01, n).clip(-0.15, 0.15)
    pred_z = np.full(n, 1.5)
    in_uptrend = np.zeros(n, dtype=bool)
    in_downtrend = np.ones(n, dtype=bool)

    _strat_ret, positions = _simulate_risk_managed_returns(
        pred_z=pred_z,
        actual_arr=actual,
        in_uptrend=in_uptrend,
        entry_threshold=0.5,
        downtrend_arr=in_downtrend,
    )

    active_days = int(np.sum(np.abs(positions) > 1e-10))
    assert active_days == 0, (
        "Long signals in a confirmed downtrend must be fully suppressed."
    )


def test_confirmed_trend_signals_pass_through():
    """Longs in a confirmed uptrend (golden cross) must reach the portfolio."""
    rng = np.random.default_rng(11)
    n = 150
    actual = rng.normal(0, 0.01, n).clip(-0.15, 0.15)
    pred_z = np.full(n, 1.5)
    in_uptrend = np.ones(n, dtype=bool)
    in_downtrend = np.zeros(n, dtype=bool)

    _strat_ret, positions = _simulate_risk_managed_returns(
        pred_z=pred_z,
        actual_arr=actual,
        in_uptrend=in_uptrend,
        entry_threshold=0.5,
        downtrend_arr=in_downtrend,
    )

    active_days = int(np.sum(np.abs(positions) > 1e-10))
    assert active_days > 0, (
        "Long signals in a confirmed uptrend must pass through the dual-SMA filter."
    )


# ─── Feature selection helpers ────────────────────────────────────────────────


def test_select_top_features_ranks_by_pearson():
    """Highest-correlation feature must always be in the selected set."""
    rng = np.random.default_rng(0)
    n = 120
    y = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "signal": y + rng.normal(0, 0.05, n),   # r ≈ 0.999 with target
        "noise1": rng.normal(0, 1, n),
        "noise2": rng.normal(0, 1, n),
        "noise3": rng.normal(0, 1, n),
        "noise4": rng.normal(0, 1, n),
    })
    selected = _select_top_features(df, y, list(df.columns), max_k=2)
    assert len(selected) == 2
    assert "signal" in selected, "Highest-Pearson feature must be selected"


def test_select_top_features_drops_near_constant():
    """Near-zero-variance columns must be excluded regardless of apparent correlation."""
    rng = np.random.default_rng(1)
    n = 100
    y = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "real": rng.normal(0, 1, n),
        "const": np.full(n, 2.718),
    })
    selected = _select_top_features(df, y, ["real", "const"], max_k=5)
    assert "const" not in selected, "Constant column must be dropped"
    assert "real" in selected


def test_select_top_features_returns_all_when_below_max_k():
    """If len(features) <= max_k the function must return them unchanged."""
    rng = np.random.default_rng(2)
    n = 80
    y = rng.normal(0, 1, n)
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    selected = _select_top_features(df, y, ["a", "b"], max_k=10)
    assert set(selected) == {"a", "b"}


def test_make_model_returns_ridgecv():
    """_make_model must return a RidgeCV instance (never ElasticNetCV).

    ElasticNetCV was replaced because it routinely over-regularises on small
    training sets (≤250 rows), collapsing pred_std below 1e-8 and triggering
    the degenerate threshold=0.0 fallback that causes Sharpe≈0.15, Calmar≈0.07.
    """
    from sklearn.linear_model import RidgeCV
    model = _make_model(200)
    assert isinstance(model, RidgeCV)


def test_ridgecv_nonzero_predictions_on_small_dataset():
    """RidgeCV must produce non-zero std predictions even on small training sets,
    preventing the pred_std<1e-8 edge case that breaks z-score normalisation."""
    rng = np.random.default_rng(42)
    n = 120  # deliberately small — mimics 207-row training after macro lag
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.normal(0, 1, n),
    })
    y = 0.3 * X["f1"].values + rng.normal(0, 0.5, n)
    model = _make_model(n)
    model.fit(X, y)
    preds = model.predict(X)
    pred_std = float(np.std(preds, ddof=1))
    assert pred_std > 1e-8, (
        f"RidgeCV predictions must have non-negligible std (got {pred_std:.2e}). "
        "A near-zero pred_std triggers pred_std=1.0 fallback and breaks z-scores."
    )


def test_ridgecv_strongest_coef_on_signal_feature():
    """RidgeCV must assign the largest coefficient to the truly predictive feature."""
    from sklearn.linear_model import RidgeCV
    rng = np.random.default_rng(42)
    n = 300
    X_signal = rng.normal(0, 1, n)
    y = 0.8 * X_signal + rng.normal(0, 0.1, n)
    X = pd.DataFrame({
        "signal": X_signal,
        "noise1": rng.normal(0, 1, n),
        "noise2": rng.normal(0, 1, n),
        "noise3": rng.normal(0, 1, n),
    })
    model = _make_model(n)
    model.fit(X, y)
    coefs = dict(zip(X.columns, model.coef_))
    assert abs(coefs["signal"]) > max(abs(coefs["noise1"]), abs(coefs["noise2"]), abs(coefs["noise3"])), (
        "RidgeCV must assign the largest coefficient to the predictive feature."
    )
