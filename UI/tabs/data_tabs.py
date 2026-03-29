from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from UI.constants import (
    GOLD_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
)
from UI.content import ANALYSIS_HELP, LAYER_HELP
from UI.helpers import load_session_history
from UI.rendering import FETCHERS_MOD, MAIN_MOD, MEDALLION_MOD, render_logger_message
from UI.runtime import rerun_stress_test_only
from UI.traffic_light import (
    badge_html,
    score_governance_gate,
    score_model_risk,
    score_oos_r2,
)


def _human_label(text: str) -> str:
    return str(text).replace("_", " ").strip().title()


def _fmt_scalar(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, bool)):
        return str(value)
    if value is None:
        return "—"
    return str(value)


def _flatten_scalars(payload: object, prefix: str = "", max_items: int = 60) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    def walk(node: object, path: str) -> None:
        if len(rows) >= max_items:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                next_path = f"{path}.{k}" if path else str(k)
                walk(v, next_path)
                if len(rows) >= max_items:
                    return
        elif isinstance(node, list):
            if all(not isinstance(x, (dict, list)) for x in node):
                rows.append((path or "value", ", ".join(_fmt_scalar(x) for x in node[:10])))
            else:
                rows.append((path or "value", f"list[{len(node)}]"))
        else:
            rows.append((path or "value", _fmt_scalar(node)))

    walk(payload, prefix)
    return rows


def _render_key_values(payload: object, header: str = "Summary") -> None:
    rows = _flatten_scalars(payload)
    if not rows:
        st.info("No summary fields available.")
        return
    st.markdown(f"**{header}**")
    st.dataframe(
        pd.DataFrame([{"Field": _human_label(k), "Value": v} for k, v in rows]),
        width="stretch",
        hide_index=True,
    )


@st.cache_data(show_spinner=False)
def _read_json_cached(path_str: str, mtime_ns: int) -> dict:
    _ = mtime_ns  # cache key component (file changes -> cache invalidation)
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _read_csv_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns
    path = Path(path_str)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_parquet_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns
    path = Path(path_str)
    return pd.read_parquet(path)


def _read_json_fast(path: Path) -> dict:
    if not path.exists():
        return {}
    return _read_json_cached(str(path), path.stat().st_mtime_ns)


def _read_csv_fast(path: Path) -> pd.DataFrame:
    return _read_csv_cached(str(path), path.stat().st_mtime_ns)


def _read_parquet_fast(path: Path) -> pd.DataFrame:
    return _read_parquet_cached(str(path), path.stat().st_mtime_ns)


def _unwrap_value_payload(payload: object) -> object:
    current = payload
    while isinstance(current, dict) and "value" in current and len(current) == 1:
        current = current.get("value")
    return current


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _load_analysis_payload(name: str, summary_results: dict | None = None) -> object:
    path = OUTPUT_DIR / f"{name}.json"
    if path.exists():
        return _unwrap_value_payload(_read_json_fast(path))

    if isinstance(summary_results, dict) and name in summary_results:
        return _unwrap_value_payload(summary_results.get(name))
    return {}


def _render_quant_insights(summary: dict) -> None:
    results = summary.get("results", {}) if isinstance(summary, dict) else {}
    if not isinstance(results, dict):
        results = {}

    gov = _load_analysis_payload("governance_report", results)
    mc = _load_analysis_payload("monte_carlo", results)
    lag = _load_analysis_payload("lag_analysis", results)
    elas = _load_analysis_payload("elasticity", results)
    stress = _load_analysis_payload("stress_test", results)
    fc = _load_analysis_payload("forecasting", results)
    decay = _load_analysis_payload("feature_decay", results)
    backtest = _load_analysis_payload("backtest_2020", results)

    if not any(isinstance(item, dict) and item for item in (gov, mc, lag, elas, stress, fc, decay, backtest)):
        return

    # ── Build Insights rows — only include metrics that have actual values ────
    _oos_r2 = (gov.get("out_of_sample") or {}).get("r2")
    _mod_risk = gov.get("model_risk_score")
    _best_lag = lag.get("best_lag_days")
    _elas_b = elas.get("static_elasticity")

    _top_pairs = [
        ("OOS R²", _oos_r2),
        ("Model Risk", _mod_risk),
        ("Best Lag (days)", _best_lag),
        ("Elasticity β", _elas_b),
    ]
    _top_available = [(lbl, val) for lbl, val in _top_pairs if val is not None]

    if _top_available:
        st.markdown("---")
        st.markdown("### 📉 Insights Dashboard")
        _cols = st.columns(len(_top_available))
        for _col, (lbl, val) in zip(_cols, _top_available):
            _col.metric(lbl, _fmt_scalar(val))

    if isinstance(backtest, dict) and backtest:
        _bt_pairs = [
            ("Expectancy/Trade", backtest.get("expectancy_per_trade")),
            ("Profit Factor", backtest.get("profit_factor")),
            ("Calmar", backtest.get("calmar_ratio")),
            ("Information Ratio", backtest.get("information_ratio")),
        ]
        _bt_available = [(lbl, val) for lbl, val in _bt_pairs if val is not None]
        if _bt_available:
            if not _top_available:
                # Show header here if the top row was skipped entirely
                st.markdown("---")
                st.markdown("### 📉 Insights Dashboard")
            _bt_cols = st.columns(len(_bt_available))
            for _col, (lbl, val) in zip(_bt_cols, _bt_available):
                _col.metric(lbl, _fmt_scalar(val))

        positive_signals = []
        pf = backtest.get("profit_factor")
        calmar = backtest.get("calmar_ratio")
        exp = backtest.get("expectancy_per_trade")
        if isinstance(pf, (int, float)) and float(pf) >= 1.5:
            positive_signals.append(f"Profit Factor {float(pf):.2f}")
        if isinstance(calmar, (int, float)) and float(calmar) >= 2.0:
            positive_signals.append(f"Calmar {float(calmar):.2f}")
        if isinstance(exp, (int, float)) and float(exp) > 0.0:
            positive_signals.append(f"Positive Expectancy {float(exp):.4f}")
        if positive_signals:
            st.success(
                "Dominant validated findings: " + " | ".join(positive_signals) + ". These metrics demonstrate realized strategy edge even when R² is modest."
            )

    stress_results = stress.get("results", {}) if isinstance(stress, dict) else {}
    if isinstance(stress_results, dict) and stress_results:
        stress_rows = []
        for factor, metrics in stress_results.items():
            if isinstance(metrics, dict):
                stress_rows.append(
                    {
                        "factor": str(factor),
                        "impact": float(metrics.get("predicted_impact", 0.0)),
                        "shock": float(metrics.get("shock", 0.0)),
                    }
                )
        if stress_rows:
            sdf = pd.DataFrame(stress_rows).sort_values("impact")
            fig = px.bar(
                sdf,
                x="impact",
                y="factor",
                orientation="h",
                title="Stress Impact per Factor",
                color="impact",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")

    lag_scan = lag.get("lag_scan", []) if isinstance(lag, dict) else []
    if isinstance(lag_scan, list) and lag_scan:
        ldf = pd.DataFrame(lag_scan)
        if {"lag_days", "correlation"}.issubset(ldf.columns):
            ldf = ldf.dropna(subset=["correlation"])
            if not ldf.empty:
                fig = px.line(
                    ldf,
                    x="lag_days",
                    y="correlation",
                    markers=True,
                    title="Lag Profile (Correlation vs Lag Days)",
                )
                fig.add_hline(y=0.0, line_dash="dot", line_color="#777")
                fig.update_layout(height=330)
                st.plotly_chart(fig, width="stretch")

    rolling = elas.get("rolling_elasticity", []) if isinstance(elas, dict) else []
    if isinstance(rolling, list) and rolling:
        edf = pd.DataFrame(rolling)
        if {"date", "elasticity"}.issubset(edf.columns):
            edf["date"] = pd.to_datetime(edf["date"], errors="coerce")
            edf = edf.dropna(subset=["date", "elasticity"])
            if not edf.empty:
                fig = px.line(
                    edf,
                    x="date",
                    y="elasticity",
                    title="Rolling Elasticity (Time-varying Macro Sensitivity)",
                )
                fig.add_hline(y=0.0, line_dash="dot", line_color="#777")
                fig.update_layout(height=330)
                st.plotly_chart(fig, width="stretch")

    if isinstance(mc, dict) and mc:
        risk_keys = [
            "value_at_risk_95",
            "value_at_risk_99",
            "conditional_var_95",
            "expected_shortfall_99",
            "historical_var_95",
            "historical_es_95",
            "parametric_var_95",
            "parametric_es_95",
        ]
        risk_rows = []
        for key in risk_keys:
            val = mc.get(key)
            if isinstance(val, (int, float)):
                risk_rows.append({"metric": _human_label(key), "value": float(val)})
        if risk_rows:
            rdf = pd.DataFrame(risk_rows)
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=rdf["metric"],
                        y=rdf["value"],
                        marker_color="#0f766e",
                    )
                ]
            )
            fig.update_layout(
                title="Monte Carlo Risk Stack (VaR / ES)",
                xaxis_title="Risk Metric",
                yaxis_title="Loss (absolute)",
                height=360,
            )
            st.plotly_chart(fig, width="stretch")

    if isinstance(fc, dict) and fc.get("forecast"):
        forecast = fc.get("forecast", [])
        lower = fc.get("lower_90", [])
        upper = fc.get("upper_90", [])
        horizon = list(range(1, len(forecast) + 1))
        if forecast and len(lower) == len(forecast) and len(upper) == len(forecast):
            ribbon_df = pd.DataFrame(
                {
                    "step": horizon,
                    "forecast": forecast,
                    "lower_90": lower,
                    "upper_90": upper,
                }
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ribbon_df["step"],
                    y=ribbon_df["upper_90"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ribbon_df["step"],
                    y=ribbon_df["lower_90"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(15,118,110,0.20)",
                    line=dict(width=0),
                    name="90% CI",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ribbon_df["step"],
                    y=ribbon_df["forecast"],
                    mode="lines+markers",
                    line=dict(color="#0f766e", width=3),
                    name="Forecast",
                )
            )
            fig.update_layout(
                title="Volatility Forecast with 90% Confidence Ribbon",
                xaxis_title="Forecast Horizon (steps)",
                yaxis_title="Annualized Volatility",
                height=360,
            )
            st.plotly_chart(fig, width="stretch")

    if isinstance(decay, dict) and isinstance(decay.get("results"), dict):
        decay_rows = []
        for feature, payload in decay["results"].items():
            if not isinstance(payload, dict):
                continue
            hl = payload.get("half_life_lag_days")
            base = payload.get("baseline_correlation")
            if isinstance(hl, int):
                decay_rows.append(
                    {
                        "feature": str(feature),
                        "half_life_lag_days": hl,
                        "baseline_correlation": float(base) if isinstance(base, (int, float)) else 0.0,
                    }
                )
        if decay_rows:
            ddf = pd.DataFrame(decay_rows).sort_values("half_life_lag_days")
            fig = px.bar(
                ddf,
                x="feature",
                y="half_life_lag_days",
                color="baseline_correlation",
                color_continuous_scale="Tealgrn",
                title="Feature Decay (Information Half-Life)",
            )
            fig.update_layout(height=360, coloraxis_colorbar_title="Baseline corr")
            st.plotly_chart(fig, width="stretch")


def _render_governance_payload(payload: dict) -> None:
    gate = _unwrap_value_payload(payload.get("gate", payload.get("value", payload)))
    report = _unwrap_value_payload(payload.get("report", {}))
    if isinstance(gate, dict):
        gate_passed = _coerce_optional_bool(gate.get("passed"))
        severity = str(gate.get("severity", "unknown"))
        tl_c, tl_l, tl_d = score_governance_gate(bool(gate_passed), severity)
        st.markdown(
            badge_html(tl_l, tl_c, tl_d) + f"&nbsp; <small style='color:#555'>{tl_d}</small>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        if gate_passed is not None:
            c1.metric("Gate Passed", "Yes" if gate_passed else "No")
        if severity and severity.lower() not in {"unknown", ""}:
            c2.metric("Severity", severity.upper())
        _regime = gate.get("regime")
        if _regime and str(_regime).lower() not in {"unknown", ""}:
            c3.metric("Regime", str(_regime).upper())
        if gate_passed is True and severity.lower() == "warn":
            st.info("Gate passed with warnings. Advisory signals are visible, but they do not block analyses.")
        reasons = gate.get("reasons", []) or []
        if reasons:
            st.markdown("**Gate reasons:**")
            for reason in reasons:
                st.markdown(f"- {reason}")

    if isinstance(report, dict):
        oos_r2_val = (report.get("out_of_sample") or {}).get("r2")
        wf = (report.get("walk_forward") or {}).get("avg_r2")
        risk = report.get("model_risk_score")
        m1, m2, m3 = st.columns(3)
        m1.metric("OOS R²", _fmt_scalar(oos_r2_val))
        m2.metric("Walk-forward R²", _fmt_scalar(wf))
        m3.metric("Model Risk", _fmt_scalar(risk))
        r2_c, r2_l, r2_d = score_oos_r2(oos_r2_val)
        mr_c, mr_l, mr_d = score_model_risk(risk)
        b1, b2 = st.columns(2)
        b1.markdown(badge_html(r2_l, r2_c, r2_d), unsafe_allow_html=True)
        b2.markdown(badge_html(mr_l, mr_c, mr_d), unsafe_allow_html=True)


def _render_sensitivity_regression_payload(value: dict) -> None:
    coeffs = value.get("coefficients", {}) if isinstance(value, dict) else {}
    if not isinstance(coeffs, dict) or not coeffs:
        _render_key_values(value, header="Sensitivity Summary")
        return

    st.markdown("### 🧭 What Drives Returns the Most")
    r2 = value.get("r2")
    cv_r2 = value.get("cv_r2")
    n_obs = value.get("n_obs")
    lag_info = value.get("lag_selection", {}) if isinstance(value.get("lag_selection"), dict) else {}
    selected_lag = lag_info.get("selected_macro_lag_days")
    c1, c2, c3, c4 = st.columns(4)
    _model_val = value.get("model")
    c1.metric("Model", str(_model_val) if _model_val else "—")
    c2.metric("R²", _fmt_scalar(r2))
    c3.metric("Observations", _fmt_scalar(n_obs))
    c4.metric("CV R²", _fmt_scalar(cv_r2))

    if selected_lag is not None:
        st.caption(f"Selected macro lag: {selected_lag} days")

    subset = value.get("feature_subset_search", {}) if isinstance(value.get("feature_subset_search"), dict) else {}
    selected_subset = subset.get("selected_subset") if isinstance(subset.get("selected_subset"), list) else []
    if selected_subset:
        st.caption("Selected factors: " + ", ".join(str(x) for x in selected_subset))

    top_subsets = subset.get("top_subsets") if isinstance(subset.get("top_subsets"), list) else []
    if top_subsets:
        with st.expander("Feature subset search (top combinations)", expanded=False):
            sdf = pd.DataFrame(top_subsets)
            if "subset" in sdf.columns:
                sdf["subset"] = sdf["subset"].apply(lambda lst: ", ".join(str(x) for x in lst) if isinstance(lst, list) else str(lst))
            st.dataframe(sdf.head(15), width="stretch")

    cdf = pd.DataFrame([{"factor": str(k), "coefficient": float(v)} for k, v in coeffs.items()])
    cdf["abs_coef"] = cdf["coefficient"].abs()
    cdf = cdf.sort_values("abs_coef", ascending=False)
    fig = px.bar(
        cdf,
        x="factor",
        y="coefficient",
        color="coefficient",
        color_continuous_scale="RdYlGn",
        title="Impact of each macro factor on returns",
    )
    fig.update_layout(height=360, coloraxis_colorbar_title="Coef")
    st.plotly_chart(fig, width="stretch")

    top = cdf.iloc[0]
    direction = "increases" if float(top["coefficient"]) > 0 else "decreases"
    st.success(f"Dominant factor: **{top['factor']}**. When it rises, returns tend to **{direction}**.")


def _render_backtest_payload(value: dict) -> None:
    if not isinstance(value, dict):
        _render_key_values(value, header="Backtest Summary")
        return

    st.markdown("### 🧪 Crisis Backtest And Real Trading Edge")
    te = value.get("tracking_error")
    mdd = value.get("maximum_drawdown")
    train_rows = value.get("train_rows")
    test_rows = value.get("test_rows")
    sharpe = value.get("sharpe_ratio")
    calmar = value.get("calmar_ratio")
    profit_factor = value.get("profit_factor")
    expectancy = value.get("expectancy_per_trade")
    info_ratio = value.get("information_ratio")

    # ── Row 1: Data volume ────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    if train_rows is not None:
        k1.metric("Train rows", str(int(train_rows)))
    if test_rows is not None:
        k2.metric("Test rows", str(int(test_rows)))
    if te is not None:
        k3.metric("Tracking Error", f"{float(te):.4f}", help="Std dev of (predicted − actual) returns. Lower = tighter fit.")
    if mdd is not None:
        _mdd = float(mdd)
        k4.metric(
            "Max Drawdown",
            f"{_mdd * 100.0:+.1f}%",
            delta="Peak-to-trough loss",
            delta_color="off",
            help="Largest cumulative loss from peak. <−20% warrants caution.",
        )

    # ── Row 2: Risk-adjusted ratios — capped, percentage-formatted ────────────
    _ratio_cols = [c for c in [sharpe, calmar, profit_factor, expectancy, info_ratio] if c is not None]
    if _ratio_cols:
        p1, p2, p3, p4, p5 = st.columns(5)
        if sharpe is not None:
            _sh = float(sharpe)
            p1.metric(
                "Sharpe Ratio", f"{_sh:.2f}" if abs(_sh) <= 5.0 else ("≥ 5.0" if _sh > 0 else "≤ −5.0"), help=">0.5 acceptable · >1.0 strong · >2.0 exceptional"
            )
        if calmar is not None:
            _cal = float(calmar)
            p2.metric(
                "Calmar Ratio",
                f"{_cal:.2f}×" if _cal <= 20.0 else "≥ 20×",
                help="Annualised return ÷ Max Drawdown. >1.0 = risk-adjusted profit. Values >20 are capped.",
            )
        if profit_factor is not None and profit_factor != float("inf"):
            _pf = float(profit_factor)
            p3.metric("Profit Factor", f"{_pf:.2f}×" if _pf <= 10.0 else "≥ 10×", help="Gross gains ÷ gross losses. >1.0 = profitable · >1.5 = strong")
        if expectancy is not None:
            _exp = float(expectancy)
            p4.metric(
                "Expectancy / Trade",
                f"{_exp:+.4f}",
                delta="Positive edge" if _exp > 0 else "Negative edge",
                delta_color="normal" if _exp > 0 else "inverse",
                help="Average daily log-return per trade. Positive = systematic edge.",
            )
        if info_ratio is not None:
            _ir = float(info_ratio)
            p5.metric(
                "Info Ratio",
                f"{_ir:.2f}" if abs(_ir) <= 5.0 else ("≥ 5.0" if _ir > 0 else "≤ −5.0"),
                help="Active return vs benchmark per unit of tracking error. >0.5 = adds value.",
            )

    strong_flags: list[str] = []
    if isinstance(profit_factor, (int, float)) and profit_factor != float("inf") and float(profit_factor) >= 1.5:
        strong_flags.append(f"Profit Factor {min(float(profit_factor), 10.0):.2f}×")
    if isinstance(calmar, (int, float)) and 0 < float(calmar) <= 20.0 and float(calmar) >= 2.0:
        strong_flags.append(f"Calmar {float(calmar):.2f}×")
    if isinstance(expectancy, (int, float)) and float(expectancy) > 0.0:
        strong_flags.append(f"Positive expectancy {float(expectancy):+.4f}")
    if isinstance(sharpe, (int, float)) and 0 < float(sharpe) <= 5.0 and float(sharpe) >= 1.0:
        strong_flags.append(f"Sharpe {float(sharpe):.2f}")
    if isinstance(info_ratio, (int, float)) and float(info_ratio) >= 0.5:
        strong_flags.append(f"Information Ratio {float(info_ratio):.2f}")

    if strong_flags:
        st.success("**Validated edge signals:** " + " · ".join(strong_flags))
    else:
        st.info("No exceptional threshold triggered. Review drawdown, expectancy, and factor robustness.")

    corr_test = value.get("correlation_test", {}) if isinstance(value.get("correlation_test"), dict) else {}
    p_val = corr_test.get("p_value")
    r_val = corr_test.get("pearson_r")
    if isinstance(p_val, (int, float)):
        c1, c2 = st.columns(2)
        if r_val is not None:
            c1.metric("Pearson r", f"{float(r_val):.3f}", help="|r| > 0.15 is considered meaningful in macro-equity forecasting.")
        _pv = float(p_val)
        c2.metric("P-value", f"{_pv:.4f}", delta="Significant" if _pv < 0.05 else "Exploratory", delta_color="normal" if _pv < 0.05 else "off")
        if float(p_val) < 0.05:
            st.success("Statistical significance passed: p-value < 0.05")
        else:
            st.caption("Correlation significance is weak in this sample (p-value >= 0.05).")

    preds = value.get("predictions", [])
    actual = value.get("actual", [])
    if isinstance(preds, list) and isinstance(actual, list) and preds and len(preds) == len(actual):
        bdf = pd.DataFrame(
            {
                "step": list(range(1, len(preds) + 1)),
                "predicted_return": [float(x) for x in preds],
                "actual_return": [float(x) for x in actual],
            }
        )
        bdf["predicted_curve"] = (1.0 + bdf["predicted_return"]).cumprod()
        bdf["actual_curve"] = (1.0 + bdf["actual_return"]).cumprod()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bdf["step"],
                y=bdf["actual_curve"],
                mode="lines",
                name="Actual path",
                line=dict(color="#b91c1c", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bdf["step"],
                y=bdf["predicted_curve"],
                mode="lines",
                name="Model prediction",
                line=dict(color="#0f766e", width=3),
            )
        )
        fig.update_layout(
            title="Prediction vs Reality (2020-2022)",
            xaxis_title="Time step",
            yaxis_title="Cumulative equity curve",
            height=380,
        )
        st.plotly_chart(fig, width="stretch")

        strategy_returns = value.get("strategy_returns", [])
        if isinstance(strategy_returns, list) and strategy_returns:
            tdf = pd.DataFrame({"strategy_return": [float(x) for x in strategy_returns]})
            fig_hist = px.histogram(
                tdf,
                x="strategy_return",
                nbins=30,
                title="Trade Distribution Histogram (Gains vs Losses)",
                color_discrete_sequence=["#0f766e"],
            )
            fig_hist.add_vline(x=0.0, line_dash="dot", line_color="#7f1d1d")
            fig_hist.update_layout(height=330)
            st.plotly_chart(fig_hist, width="stretch")

        rolling_sharpe = value.get("rolling_sharpe_30d", [])
        if isinstance(rolling_sharpe, list) and rolling_sharpe:
            rs_df = pd.DataFrame(rolling_sharpe)
            if {"step", "rolling_sharpe"}.issubset(rs_df.columns):
                fig_rs = px.line(
                    rs_df,
                    x="step",
                    y="rolling_sharpe",
                    title="Rolling Sharpe Ratio (30-day)",
                )
                fig_rs.add_hline(y=0.0, line_dash="dot", line_color="#777")
                fig_rs.add_hline(y=1.0, line_dash="dash", line_color="#0f766e")
                fig_rs.update_layout(height=330)
                st.plotly_chart(fig_rs, width="stretch")

        avg_abs_err = float((bdf["predicted_return"] - bdf["actual_return"]).abs().mean())
        st.info(
            "Plain reading: when the 2 curves stay close, the model holds up well under crisis conditions. "
            "The further they diverge, the higher the error. "
            f"Mean absolute error per period: {avg_abs_err:.4f}."
        )


def _render_elasticity_payload(value: dict) -> None:
    if not isinstance(value, dict):
        _render_key_values(value, header="Elasticity Summary")
        return

    static_el = value.get("static_elasticity")
    points = value.get("data_points")
    f1, f2, f3 = st.columns(3)
    f1.metric("Static Elasticity", _fmt_scalar(static_el))
    f2.metric("Data Points", _fmt_scalar(points))
    _mf = value.get("macro_factor")
    f3.metric("Macro Factor", str(_mf) if _mf else "—")

    direction = "rises" if isinstance(static_el, (int, float)) and static_el >= 0 else "falls"
    st.success(f"Simple insight: when the macro factor increases, returns tend to **{direction}**.")

    rolling = value.get("rolling_elasticity", [])
    if isinstance(rolling, list) and rolling:
        rdf = pd.DataFrame(rolling)
        if {"date", "elasticity"}.issubset(rdf.columns):
            rdf["date"] = pd.to_datetime(rdf["date"], errors="coerce")
            rdf = rdf.dropna(subset=["date", "elasticity"])
            if not rdf.empty:
                fig = px.line(
                    rdf,
                    x="date",
                    y="elasticity",
                    title="How sensitivity changes over time",
                )
                fig.add_hline(y=0.0, line_dash="dot", line_color="#777")
                fig.update_layout(height=340)
                st.plotly_chart(fig, width="stretch")


def _render_lag_payload(value: dict) -> None:
    if not isinstance(value, dict):
        _render_key_values(value, header="Lag Summary")
        return

    best_lag = value.get("best_lag_days")
    best_corr = value.get("best_lag_correlation")
    rlag = value.get("reference_lag_days")
    rcorr = value.get("reference_lag_correlation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Lag (days)", _fmt_scalar(best_lag))
    c2.metric("Best Corr", _fmt_scalar(best_corr))
    c3.metric("Reference Lag", _fmt_scalar(rlag))
    c4.metric("Reference Corr", _fmt_scalar(rcorr))

    scan = value.get("lag_scan", [])
    if isinstance(scan, list) and scan:
        ldf = pd.DataFrame(scan)
        if {"lag_days", "correlation"}.issubset(ldf.columns):
            ldf = ldf.dropna(subset=["correlation"])
            if not ldf.empty:
                fig = px.line(
                    ldf,
                    x="lag_days",
                    y="correlation",
                    markers=True,
                    title="Macro factor transmission delay (lag profile)",
                )
                fig.add_hline(y=0, line_dash="dot", line_color="#777")
                fig.update_layout(height=340)
                st.plotly_chart(fig, width="stretch")


def _render_forecasting_payload(value: dict) -> None:
    if not isinstance(value, dict):
        _render_key_values(value, header="Forecast Summary")
        return

    current_vol = value.get("current_volatility")
    vol_window = value.get("volatility_window")
    c1, c2 = st.columns(2)
    c1.metric("Current Annualized Vol", _fmt_scalar(current_vol))
    c2.metric("Volatility Window", _fmt_scalar(vol_window))

    forecast = value.get("forecast", [])
    lower = value.get("lower_90", [])
    upper = value.get("upper_90", [])
    if isinstance(forecast, list) and forecast:
        x = list(range(1, len(forecast) + 1))
        fdf = pd.DataFrame({"step": x, "forecast": forecast})
        fig = go.Figure()
        if isinstance(lower, list) and isinstance(upper, list) and len(lower) == len(forecast) and len(upper) == len(forecast):
            fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=lower,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(2,132,199,0.2)",
                    line=dict(width=0),
                    name="90% CI",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=fdf["step"],
                y=fdf["forecast"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#0284c7", width=3),
            )
        )
        fig.update_layout(height=340, title="Volatility Forecast")
        st.plotly_chart(fig, width="stretch")


def _render_monte_carlo_payload(value: dict) -> None:
    if not isinstance(value, dict):
        _render_key_values(value, header="Monte Carlo Summary")
        return

    st.markdown("### 🎲 Monte Carlo Risk Snapshot")
    c1, c2, c3 = st.columns(3)
    _sc = value.get("scenario")
    c1.metric("Scenario", str(_sc) if _sc else "—")
    c2.metric("Daily Vol", _fmt_scalar(value.get("daily_volatility")))
    c3.metric("Scenario Vol", _fmt_scalar(value.get("scenario_volatility")))

    risk_metrics = [
        "value_at_risk_95",
        "value_at_risk_99",
        "conditional_var_95",
        "expected_shortfall_99",
        "historical_var_95",
        "historical_es_95",
        "parametric_var_95",
        "parametric_es_95",
    ]
    rows = []
    for k in risk_metrics:
        v = value.get(k)
        if isinstance(v, (int, float)):
            rows.append({"metric": _human_label(k), "value": float(v)})
    if rows:
        rdf = pd.DataFrame(rows)
        fig = px.bar(rdf, x="metric", y="value", title="Tail Risk Summary (VaR / ES)")
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")


def _render_auto_ml_payload(value: dict) -> None:
    if not isinstance(value, dict):
        _render_key_values(value, header="Auto ML Summary")
        return

    best_model = value.get("best_model")
    c1 = st.columns(1)[0]
    c1.metric("Best Model", str(best_model) if best_model else "—")
    st.success("AutoML selected the best-performing model for this run's dataset.")

    preds = value.get("predictions")
    if isinstance(preds, list) and preds:
        pdf = pd.DataFrame(preds)
        if {"prediction"}.issubset(pdf.columns):
            fig = px.histogram(pdf, x="prediction", nbins=30, title="Prediction Distribution")
            fig.update_layout(height=320)
            st.plotly_chart(fig, width="stretch")


def _render_feature_decay_payload(value: dict) -> None:
    if not isinstance(value, dict):
        _render_key_values(value, header="Feature Decay Summary")
        return
    results = value.get("results", {}) if isinstance(value.get("results"), dict) else {}
    if not results:
        _render_key_values(value, header="Feature Decay Summary")
        return

    rows = []
    for feature, payload in results.items():
        if isinstance(payload, dict):
            rows.append(
                {
                    "feature": str(feature),
                    "half_life_lag_days": payload.get("half_life_lag_days"),
                    "baseline_correlation": payload.get("baseline_correlation"),
                }
            )
    ddf = pd.DataFrame(rows)
    st.dataframe(ddf, width="stretch", hide_index=True)
    if not ddf.empty and "half_life_lag_days" in ddf.columns:
        ddf = ddf.dropna(subset=["half_life_lag_days"])
        if not ddf.empty:
            fig = px.bar(ddf, x="feature", y="half_life_lag_days", title="Feature Decay: How Quickly Signal Goes Stale")
            fig.update_layout(height=330)
            st.plotly_chart(fig, width="stretch")


def _render_governance_report_payload(value: dict) -> None:
    if not isinstance(value, dict):
        _render_key_values(value, header="Governance Report")
        return
    oos = (value.get("out_of_sample") or {}).get("r2")
    wf = (value.get("walk_forward") or {}).get("avg_r2")
    risk = value.get("model_risk_score")
    c1, c2, c3 = st.columns(3)
    c1.metric("OOS R²", _fmt_scalar(oos))
    c2.metric("Walk-forward R²", _fmt_scalar(wf))
    c3.metric("Model Risk", _fmt_scalar(risk))
    st.caption("The Report is a model quality/risk diagnostic, not a production approval.")


def _render_governance_gate_payload(value: dict) -> None:
    if not isinstance(value, dict):
        _render_key_values(value, header="Governance Gate")
        return
    passed = _coerce_optional_bool(value.get("passed"))
    sev = str(value.get("severity", "unknown")).lower()
    c, label, d = score_governance_gate(bool(passed), sev)
    st.markdown(badge_html(label, c, d), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if passed is not None:
        c1.metric("Gate Decision", "PASS" if passed else "FAIL")
    _sev_raw = value.get("severity")
    if _sev_raw and str(_sev_raw).lower() not in {"unknown", "n/a", ""}:
        c2.metric("Severity", str(_sev_raw).upper())
    if passed is True and sev == "warn":
        st.caption("This is a warn-pass, not a block. Analyses remain available.")
    reasons = value.get("reasons", [])
    if isinstance(reasons, list) and reasons:
        st.markdown("**Why this gate decision was reached**")
        for reason in reasons:
            st.markdown(f"- {reason}")


def _render_analysis_view(analysis_name: str, payload: object) -> None:
    value = payload.get("value") if isinstance(payload, dict) and "value" in payload else payload

    if analysis_name == "stress_test":
        if _render_stress_payload(value):
            return
    if analysis_name == "sensitivity_regression" and isinstance(value, dict):
        _render_sensitivity_regression_payload(value)
        return
    if analysis_name == "backtest_2020" and isinstance(value, dict):
        _render_backtest_payload(value)
        return
    if analysis_name == "elasticity" and isinstance(value, dict):
        _render_elasticity_payload(value)
        return
    if analysis_name == "lag_analysis" and isinstance(value, dict):
        _render_lag_payload(value)
        return
    if analysis_name == "forecasting" and isinstance(value, dict):
        _render_forecasting_payload(value)
        return
    if analysis_name == "monte_carlo" and isinstance(value, dict):
        _render_monte_carlo_payload(value)
        return
    if analysis_name == "auto_ml" and isinstance(value, dict):
        _render_auto_ml_payload(value)
        return
    if analysis_name == "feature_decay" and isinstance(value, dict):
        _render_feature_decay_payload(value)
        return
    if analysis_name == "governance_report" and isinstance(value, dict):
        _render_governance_report_payload(value)
        return
    if analysis_name == "governance_gate" and isinstance(value, dict):
        _render_governance_gate_payload(value)
        return

    _render_analysis_payload(analysis_name, payload)


def _render_stress_rerun_controls() -> None:
    st.markdown("### 🔁 Re-run Stress Test")
    scenario_options = [
        "geopolitical_conflict",
        "monetary_tightening",
        "tech_correction",
        "stagflation",
        "liquidity_freeze",
        "commodity_super_spike",
        "supply_chain_dislocation",
        "sovereign_debt_stress",
        "custom",
    ]
    c1, c2 = st.columns(2)
    scenario_choice = c1.selectbox(
        "Scenario",
        options=scenario_options,
        index=0,
        key="stress_rerun_scenario",
    )
    ticker_choice = (
        c2.text_input(
            "Ticker (optional)",
            value="",
            key="stress_rerun_ticker",
            help="Leave blank to use the full analysis universe.",
        )
        .strip()
        .upper()
    )

    custom_shocks_raw = st.text_area(
        "Custom shocks JSON (optional)",
        value="{}",
        key="stress_rerun_shocks",
        help='Example: {"inflation": 0.03, "energy_index": 0.20}',
    )

    last_key = "stress_rerun_last_result"
    meta_key = "stress_rerun_last_meta"

    if st.button("Run Stress Test Now", type="primary", key="stress_rerun_btn"):
        try:
            parsed = json.loads(custom_shocks_raw) if custom_shocks_raw.strip() else {}
            if not isinstance(parsed, dict):
                raise ValueError("Custom shocks must be a JSON object.")
            shock_map = {str(k): float(v) for k, v in parsed.items()}
        except Exception as exc:
            st.error(f"Invalid custom shocks JSON: {exc}")
            return

        try:
            result = rerun_stress_test_only(
                scenario_name=scenario_choice,
                shock_map=shock_map,
                ticker=ticker_choice or None,
                target="log_return",
            )
            st.session_state[last_key] = result
            st.session_state[meta_key] = {
                "scenario": scenario_choice,
                "ticker": ticker_choice or "ALL",
            }
            st.success("Stress test ολοκληρώθηκε. Το αποτέλεσμα αποθηκεύτηκε και εμφανίζεται ακριβώς από κάτω.")
        except Exception as exc:
            st.error(f"Stress rerun failed: {exc}")

    live_payload = st.session_state.get(last_key)
    live_meta = st.session_state.get(meta_key, {})
    if live_payload is not None:
        st.markdown("---")
        st.markdown("### 🎯 Live Stress Result")
        c1, c2 = st.columns(2)
        c1.info(f"Scenario: {str(live_meta.get('scenario', 'custom')).replace('_', ' ').title()}")
        c2.info(f"Ticker scope: {live_meta.get('ticker', 'ALL')}")
        st.caption("Το παρακάτω είναι το άμεσο αποτέλεσμα του τελευταίου rerun (χωρίς νέο pipeline run).")
        _render_stress_payload(live_payload)


def _render_stress_payload(value: object) -> bool:
    """Return True when stress payload was rendered with a specific UI path."""
    if isinstance(value, str) and "Shock map not provided" in value:
        st.error("No stress scenario was applied in this run.")
        st.caption(
            "stress_test was skipped because no shock map/scenario was provided to the pipeline. Stress results are therefore unavailable for interpretation."
        )
        return True

    if not isinstance(value, dict):
        return False

    scenario = value.get("scenario", {}) if isinstance(value.get("scenario"), dict) else {}
    results = value.get("results", {}) if isinstance(value.get("results"), dict) else {}
    st.markdown("### ⚡ Stress Scenario Snapshot")
    st.metric("Scenario", str(scenario.get("name", "custom")).replace("_", " ").title())
    if scenario.get("description"):
        st.caption(str(scenario.get("description")))

    if results:
        rows = []
        for factor, m in results.items():
            if isinstance(m, dict):
                rows.append(
                    {
                        "Factor": factor,
                        "Shock": float(m.get("shock", 0.0)),
                        "Beta": float(m.get("beta", 0.0)),
                        "Impact": float(m.get("predicted_impact", 0.0)),
                    }
                )
        if rows:
            rdf = pd.DataFrame(rows).sort_values("Impact")
            fig = px.bar(
                rdf,
                x="Impact",
                y="Factor",
                orientation="h",
                title="Estimated impact per stressed factor",
                color="Impact",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(height=340, coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
    return True


def _render_governance_consistency_panel(results: dict) -> None:
    report = _load_analysis_payload("governance_report", results)
    gate = _load_analysis_payload("governance_gate", results)
    if not isinstance(report, dict) and not isinstance(gate, dict):
        return

    st.markdown("### 🛡️ Governance: Report vs Gate")
    st.caption(
        "Governance Report = model quality/risk diagnostic. Governance Gate = final production approval decision. "
        "They are not duplicates — they serve different roles."
    )

    # Extract values — treat missing/unknown as None so we can suppress the slot
    _raw_status = report.get("status") if isinstance(report, dict) else None
    report_status = str(_raw_status).upper() if _raw_status else None
    gate_passed = _coerce_optional_bool(gate.get("passed")) if isinstance(gate, dict) else None
    _raw_sev = gate.get("severity") if isinstance(gate, dict) else None
    gate_severity = str(_raw_sev).upper() if _raw_sev else None

    # Only render metrics that carry actual information
    _gov_metric_pairs: list[tuple[str, str]] = []
    if report_status and report_status not in {"UNKNOWN", "N/A", ""}:
        _gov_metric_pairs.append(("Report status", report_status))
    if gate_passed is not None:
        _gov_metric_pairs.append(("Gate passed", "Yes" if gate_passed else "No"))
    if gate_severity and gate_severity not in {"UNKNOWN", "N/A", ""}:
        _gov_metric_pairs.append(("Gate severity", gate_severity))

    # If nothing meaningful to show, skip the panel entirely
    if not _gov_metric_pairs:
        return

    _gov_cols = st.columns(len(_gov_metric_pairs))
    for _gc, (lbl, val) in zip(_gov_cols, _gov_metric_pairs):
        _gc.metric(lbl, val)

    if isinstance(report, dict) and isinstance(gate, dict):
        if gate_passed is True and gate_severity == "WARN":
            st.info(
                "The governance gate passed with warnings. This means advisory model-risk signals were raised, but policy did not block downstream analyses."
            )
        if report_status in {"FAIL", "ERROR", "CRITICAL"} and gate_passed:
            st.warning(
                "The Report shows a weak quality/risk profile, but the Gate allowed the run under a warning policy. "
                "This is only acceptable when the policy is advisory, not a hard-block."
            )
        elif report_status == "OK" and gate_passed is False:
            st.warning("The Report is OK but the Gate blocked this run due to policy-level rules or thresholds.")


def _render_analysis_payload(analysis_name: str, payload: object) -> None:
    value = payload.get("value") if isinstance(payload, dict) and "value" in payload else payload

    if isinstance(value, str):
        if value.startswith("blocked_by_governance_gate"):
            st.warning("This analysis is blocked by governance gate for this run.")
            st.caption(value)
        else:
            st.info(value)
        return

    if isinstance(value, (int, float, bool)):
        st.metric("Result", _fmt_scalar(value))
        return

    if analysis_name in {"governance_gate", "governance_report"} and isinstance(value, dict):
        wrapper = {"gate": value} if analysis_name == "governance_gate" else {"report": value}
        _render_governance_payload(wrapper)
        return

    if analysis_name == "correlation_matrix" and isinstance(value, dict):
        c1, c2 = st.columns(2)
        shape = value.get("shape", [0, 0])
        c1.metric("Rows", shape[0] if isinstance(shape, list) and len(shape) == 2 else "—")
        c2.metric("Columns", shape[1] if isinstance(shape, list) and len(shape) == 2 else "—")
        cols = value.get("columns", [])
        if isinstance(cols, list) and cols:
            st.caption("Available fields")
            st.write(", ".join(str(x) for x in cols[:20]) + ("..." if len(cols) > 20 else ""))
        return

    if isinstance(value, dict):
        _render_key_values(value)
        return

    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            st.dataframe(pd.DataFrame(value), width="stretch", hide_index=True)
        else:
            for item in value[:20]:
                st.markdown(f"- {_fmt_scalar(item)}")
        return

    st.info("No readable analysis output available for this artifact.")


def show_data_tab() -> None:
    st.subheader("🗂️ Data Lake Explorer")
    with st.expander("Help: How data flows in this app", expanded=False):
        if FETCHERS_MOD is not None:
            render_logger_message("Fetcher guide", getattr(FETCHERS_MOD, "FETCHER_USER_GUIDE", ""))
        if MEDALLION_MOD is not None:
            render_logger_message("Medallion guide", getattr(MEDALLION_MOD, "MEDALLION_USER_GUIDE", ""))

    selected_layer = st.selectbox(
        "Choose data layer to explore:",
        options=["raw", "processed", "gold"],
        format_func=lambda k: f"{LAYER_HELP[k]['icon']}  {LAYER_HELP[k]['title']}",
        index=1,
    )
    info = LAYER_HELP[selected_layer]
    layer_paths = {"raw": RAW_DIR, "processed": PROCESSED_DIR, "gold": GOLD_DIR}

    st.info(f"**{info['what']}**")
    with st.expander("What does this layer contain?", expanded=False):
        for item in info["contains"]:
            st.markdown(f"- {item}")
        st.caption(f"{info['note']}")

    layer_path = layer_paths[selected_layer]
    parquet_files = sorted(layer_path.glob("**/*.parquet"))
    if not parquet_files:
        st.warning(f"No files found in the {selected_layer} layer yet. Run the pipeline first.")
        return

    file_labels = {str(p.relative_to(PROJECT_ROOT)): p for p in parquet_files}
    selected_label = st.selectbox("Choose file to preview:", options=list(file_labels.keys()))
    selected_file = file_labels[selected_label]
    with st.spinner("Loading data..."):
        df = _read_parquet_fast(selected_file)
    st.metric("Data Shape", f"{len(df)} rows × {len(df.columns)} columns")
    st.markdown("**Preview (first 200 rows):**")
    st.dataframe(df.head(200), width="stretch")

    # ── Quantos AI Insights ───────────────────────────────────────────────────
    try:
        from UI.tabs.assistant_tab import render_inline_ai_section
        render_inline_ai_section(
            topic="Data Layer Quality — missing values, schema, data coverage",
            snapshot={"selected_layer": selected_layer, "rows": len(df), "columns": len(df.columns)},
            key_suffix="data_tab",
        )
    except Exception:
        pass


def show_analytics_tab() -> None:
    st.subheader("📈 Analytics & Analysis Results")
    with st.expander("Help: How to interpret analysis output", expanded=False):
        if MAIN_MOD is not None:
            render_logger_message("Analysis summary", getattr(MAIN_MOD, "MAIN_OUTPUT_EXPLANATION", ""))

    corr_path = OUTPUT_DIR / "correlation_matrix.csv"
    summary_path = OUTPUT_DIR / "analysis_results.json"
    st.markdown("### 🔗 Correlation Matrix")
    corr_help = ANALYSIS_HELP.get("correlation_matrix", {})
    if corr_help:
        with st.expander("What is the Correlation Matrix?", expanded=False):
            st.markdown(f"**What it does:** {corr_help['what']}")
            st.markdown(f"**How to read it:** {corr_help['read']}")
            st.markdown(f"**How to use it:** {corr_help['use']}")

    if corr_path.exists():
        corr_df = _read_csv_fast(corr_path)
        if not corr_df.empty:
            corr_df = corr_df.set_index(corr_df.columns[0])
        fig = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Correlation Matrix — Factor Relationships")
        fig.update_layout(height=640)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Correlation matrix not found. Run the pipeline first.")

    summary = _read_json_fast(summary_path)
    results = summary.get("results", {}) if isinstance(summary, dict) else {}
    if isinstance(results, dict):
        _render_governance_consistency_panel(results)
    _render_quant_insights(summary)
    artifacts = summary.get("artifacts", {}) if isinstance(summary, dict) else {}
    if not artifacts:
        return
    st.markdown("---")
    st.markdown("### 🧩 Analysis Explorer")
    st.caption("Select an analysis for a clean insight-first view with charts and plain-language interpretation.")

    analysis_names = sorted(set(artifacts.keys()) | {"stress_test"})
    selected_analysis = st.selectbox(
        "Select analysis",
        options=analysis_names,
        format_func=lambda name: ANALYSIS_HELP.get(name, {}).get("title", _human_label(name)),
    )

    if selected_analysis == "stress_test":
        _render_stress_rerun_controls()
        st.markdown("---")

    selected_file = artifacts.get(selected_analysis)
    if not selected_file and selected_analysis == "stress_test":
        selected_file = str(OUTPUT_DIR / "stress_test.json")
    full_path = OUTPUT_DIR / selected_file if selected_file and not Path(selected_file).is_absolute() else Path(selected_file)
    help_entry = ANALYSIS_HELP.get(selected_analysis, {})
    if help_entry:
        h1, h2, h3 = st.columns(3)
        h1.info(f"What it does: {help_entry['what']}")
        h2.info(f"How to read it: {help_entry['read']}")
        h3.info(f"Decision value: {help_entry['use']}")

    if not full_path.exists():
        st.warning("Analysis file not found. Run the pipeline first.")
        return

    st.caption(f"Source: {selected_file}")
    if full_path.suffix == ".json":
        payload = _read_json_fast(full_path)
        _render_analysis_view(selected_analysis, payload)
        st.download_button(
            "Download JSON archive",
            json.dumps(payload, indent=2, ensure_ascii=False),
            file_name=full_path.name,
            mime="application/json",
            key=f"download_{selected_analysis}",
        )
    elif full_path.suffix == ".csv":
        csv_df = _read_csv_fast(full_path)
        st.dataframe(csv_df.head(300), width="stretch")

    # ── Quantos AI Insights ───────────────────────────────────────────────────
    try:
        from UI.tabs.assistant_tab import render_inline_ai_section
        render_inline_ai_section(
            topic="Analytics & Analysis Results — correlation structure, factor relationships, quant insights",
            snapshot={"selected_analysis": selected_analysis, "has_artifacts": bool(artifacts)},
            key_suffix="analytics_tab",
        )
    except Exception:
        pass


def show_governance_tab() -> None:
    st.subheader("🛡️ Data Governance & Approvals")
    gov_dir = GOLD_DIR / "governance"
    files = sorted(gov_dir.glob("governance_decision_*.json"))
    if not files:
        st.warning("No governance decisions found. Run pipeline first.")
        return
    st.metric("Total Decisions", len(files))
    selected = st.selectbox(
        "View governance decision:",
        options=list(reversed(files)),
        format_func=lambda p: f"{p.name}  —  {pd.Timestamp(p.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}",
    )
    payload = _read_json_fast(selected)
    _render_governance_payload(payload if isinstance(payload, dict) else {"gate": {}})
    _render_key_values(payload, header="Governance details")
    st.download_button(
        "Download governance JSON",
        json.dumps(payload, indent=2, ensure_ascii=False),
        file_name=selected.name,
        mime="application/json",
        key="download_governance_json",
    )

    # ── Quantos AI Insights ───────────────────────────────────────────────────
    try:
        from UI.tabs.assistant_tab import render_inline_ai_section
        render_inline_ai_section(
            topic="Model Governance — R², residuals, stationarity, look-ahead bias",
            snapshot=payload if isinstance(payload, dict) else {},
            key_suffix="governance_tab",
        )
    except Exception:
        pass


def show_logs_tab() -> None:
    st.subheader("📜 Execution Logs & Pipeline Summary")
    history = load_session_history(limit=20)
    if not history:
        st.warning("No session logs found. Run the pipeline first.")
        return
    latest = history[-1]
    st.metric("Total Sessions Recorded", len(history))
    info = latest.get("session_info", {}) if isinstance(latest, dict) else {}
    p1, p2, p3, p4 = st.columns(4)
    _sid = info.get("session_id")
    p1.metric("Session ID", str(_sid)[:12] if _sid else "—")
    p2.metric("Duration (s)", _fmt_scalar(info.get("total_duration_seconds")))
    p3.metric("Operations", _fmt_scalar(info.get("total_operations")))
    _rid = info.get("run_id")
    p4.metric("Run ID", str(_rid)[:12] if _rid else "—")

    timeline = latest.get("operations_timeline", []) if isinstance(latest, dict) else []
    if isinstance(timeline, list) and timeline:
        rows = []
        for event in timeline[-20:]:
            if isinstance(event, dict):
                rows.append(
                    {
                        "Time": str(event.get("timestamp", ""))[11:19],
                        "Operation": event.get("operation", ""),
                        "Component": event.get("component", ""),
                    }
                )
        if rows:
            st.markdown("**Recent timeline events**")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.download_button(
        "Download latest session JSON",
        json.dumps(latest, indent=2, ensure_ascii=False),
        file_name="latest_session_summary.json",
        mime="application/json",
        key="download_latest_session_json",
    )


def show_output_tab() -> None:
    st.subheader("📦 Output Results Folder")
    if not OUTPUT_DIR.exists():
        st.warning("Output directory not found. Run pipeline first.")
        return
    output_files = sorted([f for f in OUTPUT_DIR.glob("*") if f.is_file()])
    if not output_files:
        st.warning("No output files found. Run pipeline first.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Files", len(output_files))
    col2.metric("Folder", "output/")
    col3.metric("Total Size", f"{sum(f.stat().st_size for f in output_files) / 1024:.1f} KB")
    for file_path in output_files:
        with st.expander(file_path.name, expanded=False):
            st.caption(f"Size: {file_path.stat().st_size / 1024:.1f} KB")
            if file_path.suffix == ".json":
                payload = _read_json_fast(file_path)
                _render_analysis_payload(file_path.stem, payload)
                st.download_button(
                    "Download JSON archive",
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    file_name=file_path.name,
                    mime="application/json",
                    key=f"download_output_{file_path.name}",
                )
            elif file_path.suffix == ".csv":
                try:
                    df = _read_csv_fast(file_path)
                    st.dataframe(df.head(100), width="stretch")
                except Exception:
                    st.info("Could not parse CSV")
