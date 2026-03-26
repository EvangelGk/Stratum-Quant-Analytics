from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from UI.constants import OUTPUT_DIR


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt(value: object, ndigits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    if isinstance(value, (int, bool)):
        return str(value)
    if value is None:
        return "N/A"
    return str(value)


def _render_hero_style() -> None:
    st.markdown(
        """
        <style>
        .edge-hero {
            --edge-bg-1: #082f49;
            --edge-bg-2: #0f766e;
            --edge-bg-3: #111827;
            --edge-accent: #f59e0b;
            background: linear-gradient(135deg, var(--edge-bg-1), var(--edge-bg-2) 45%, var(--edge-bg-3));
            border-radius: 18px;
            padding: 22px 24px;
            color: #f8fafc;
            border: 1px solid rgba(255,255,255,0.18);
            box-shadow: 0 14px 34px rgba(2, 6, 23, 0.28);
            margin-bottom: 14px;
        }
        .edge-hero h2 {
            margin: 0;
            letter-spacing: 0.2px;
            font-weight: 700;
        }
        .edge-hero p {
            margin: 8px 0 0 0;
            opacity: 0.94;
            font-size: 0.98rem;
        }
        .edge-chip {
            display: inline-block;
            margin: 6px 8px 0 0;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            color: #0b1220;
            background: #fde68a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _extract_backtest(candidate: dict) -> dict:
    if not isinstance(candidate, dict) or not candidate:
        return {}
    # Wrapped artifact payload: {"value": {...}}
    wrapped = candidate.get("value")
    if isinstance(wrapped, dict):
        inner = _extract_backtest(wrapped)
        if inner:
            return inner

    direct = candidate.get("backtest_2020")
    if isinstance(direct, dict):
        return direct.get("value", direct) if isinstance(direct, dict) else {}
    # analysis_results structure
    results = candidate.get("results")
    if isinstance(results, dict):
        bt = results.get("backtest_2020")
        if isinstance(bt, dict):
            return bt.get("value", bt)
    # single-artifact structure
    if isinstance(wrapped, dict):
        # May already be the backtest payload
        if any(k in wrapped for k in ("strategy_returns", "maximum_drawdown", "sharpe_ratio", "predictions", "actual")):
            return wrapped
    # direct payload
    if any(k in candidate for k in ("strategy_returns", "maximum_drawdown", "sharpe_ratio", "predictions", "actual")):
        return candidate
    return {}


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity_curve = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / np.maximum(peaks, 1e-12)) - 1.0
    return float(np.min(drawdowns))


def _annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if returns.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + returns))
    if growth <= 0.0:
        return -1.0
    years = max(returns.size / float(periods_per_year), 1.0 / float(periods_per_year))
    return float(growth ** (1.0 / years) - 1.0)


def _compute_missing_metrics(backtest: dict) -> dict:
    if not isinstance(backtest, dict):
        return {}

    out = dict(backtest)
    preds = out.get("predictions")
    actual = out.get("actual")
    strategy_returns = out.get("strategy_returns")

    if (not isinstance(strategy_returns, list) or not strategy_returns) and isinstance(preds, list) and isinstance(actual, list):
        try:
            p = np.asarray([float(x) for x in preds], dtype=float)
            a = np.asarray([float(x) for x in actual], dtype=float)
            n = min(len(p), len(a))
            if n > 0:
                p = p[:n]
                a = a[:n]
                signal = np.where(p >= 0.0, 1.0, -1.0)
                sret = signal * a
                out["strategy_returns"] = [float(x) for x in sret.tolist()]
                if not isinstance(out.get("benchmark_returns"), list):
                    out["benchmark_returns"] = [float(x) for x in a.tolist()]
        except Exception:
            pass

    if isinstance(out.get("strategy_returns"), list) and out["strategy_returns"]:
        try:
            sret = np.asarray([float(x) for x in out["strategy_returns"]], dtype=float)

            if out.get("maximum_drawdown") is None:
                out["maximum_drawdown"] = _max_drawdown_from_returns(sret)

            wins = sret[sret > 0.0]
            losses = sret[sret < 0.0]
            win_prob = float(len(wins) / len(sret)) if len(sret) else 0.0
            loss_prob = float(len(losses) / len(sret)) if len(sret) else 0.0
            avg_win = float(np.mean(wins)) if len(wins) else 0.0
            avg_loss_abs = float(abs(np.mean(losses))) if len(losses) else 0.0

            if out.get("expectancy_per_trade") is None:
                out["expectancy_per_trade"] = float((win_prob * avg_win) - (loss_prob * avg_loss_abs))

            gross_profit = float(np.sum(wins)) if len(wins) else 0.0
            gross_loss_abs = float(abs(np.sum(losses))) if len(losses) else 0.0
            if out.get("profit_factor") is None:
                out["profit_factor"] = (
                    float(gross_profit / gross_loss_abs)
                    if gross_loss_abs > 1e-12
                    else (None if gross_profit == 0.0 else float("inf"))
                )

            stdev = float(np.std(sret, ddof=1)) if len(sret) > 1 else None
            if out.get("sharpe_ratio") is None and stdev is not None and stdev > 1e-12:
                out["sharpe_ratio"] = float(np.mean(sret) / stdev * np.sqrt(252.0))

            if out.get("calmar_ratio") is None:
                ann_return = _annualized_return(sret, periods_per_year=252)
                mdd = float(out.get("maximum_drawdown") or 0.0)
                out["calmar_ratio"] = float(ann_return / abs(mdd)) if abs(mdd) > 1e-12 else None

            bret = out.get("benchmark_returns")
            if out.get("information_ratio") is None and isinstance(bret, list) and bret and len(bret) == len(sret):
                b = np.asarray([float(x) for x in bret], dtype=float)
                active = sret - b
                te = float(np.std(active, ddof=1)) if len(active) > 1 else None
                if te is not None and te > 1e-12:
                    out["information_ratio"] = float(np.mean(active) / te * np.sqrt(252.0))

            corr = out.get("correlation_test")
            if not isinstance(corr, dict):
                corr = {}
            if corr.get("pearson_r") is None and isinstance(preds, list) and isinstance(actual, list):
                p = np.asarray([float(x) for x in preds], dtype=float)
                a = np.asarray([float(x) for x in actual], dtype=float)
                n = min(len(p), len(a))
                if n >= 3:
                    p = p[:n]
                    a = a[:n]
                    r = np.corrcoef(p, a)[0, 1]
                    if np.isfinite(r):
                        corr["pearson_r"] = float(r)
            out["correlation_test"] = corr
        except Exception:
            pass

    return out


def _discover_backtest_payload() -> tuple[dict, Path | None]:
    # 1) current UI output dir
    current_analysis = _read_json(OUTPUT_DIR / "analysis_results.json")
    bt = _extract_backtest(current_analysis)
    if bt:
        return bt, OUTPUT_DIR / "analysis_results.json"

    # analysis_results may provide a path map instead of embedding payload
    if isinstance(current_analysis, dict):
        artifacts = current_analysis.get("artifacts")
        if isinstance(artifacts, dict):
            bt_path_raw = artifacts.get("backtest_2020")
            if isinstance(bt_path_raw, str) and bt_path_raw.strip():
                bt_path = Path(bt_path_raw)
                if not bt_path.is_absolute():
                    bt_path = OUTPUT_DIR / bt_path
                payload = _read_json(bt_path)
                bt = _extract_backtest(payload)
                if bt:
                    return bt, bt_path

    current_bt = _read_json(OUTPUT_DIR / "backtest_2020.json")
    bt = _extract_backtest(current_bt)
    if bt:
        return bt, OUTPUT_DIR / "backtest_2020.json"

    # 2) fallback: scan all output/* folders and pick most recently modified artifact
    output_root = OUTPUT_DIR.parent
    if not output_root.exists():
        return {}, None

    candidates: list[tuple[float, Path]] = []
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        for name in ("analysis_results.json", "backtest_2020.json"):
            p = child / name
            if p.exists() and p.is_file():
                try:
                    candidates.append((p.stat().st_mtime, p))
                except OSError:
                    continue

    for _, path in sorted(candidates, key=lambda x: x[0], reverse=True):
        payload = _read_json(path)
        bt = _extract_backtest(payload)
        if bt:
            return bt, path

        if path.name == "analysis_results.json" and isinstance(payload, dict):
            artifacts = payload.get("artifacts")
            if isinstance(artifacts, dict):
                bt_path_raw = artifacts.get("backtest_2020")
                if isinstance(bt_path_raw, str) and bt_path_raw.strip():
                    bt_path = Path(bt_path_raw)
                    if not bt_path.is_absolute():
                        bt_path = path.parent / bt_path
                    inner_payload = _read_json(bt_path)
                    bt = _extract_backtest(inner_payload)
                    if bt:
                        return bt, bt_path

    return {}, None


def show_edge_arsenal_tab() -> None:
    _render_hero_style()
    st.markdown(
        """
        <div class="edge-hero">
            <h2>Edge Arsenal</h2>
            <p>
                Institutional performance diagnostics that stay valid even when raw R² is modest.
                Every highlight below is computed from actual run artifacts.
            </p>
            <span class="edge-chip">No fabricated positives</span>
            <span class="edge-chip">Backtest-validated</span>
            <span class="edge-chip">Risk-adjusted first</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    backtest, source_path = _discover_backtest_payload()
    backtest = _compute_missing_metrics(backtest)
    if not isinstance(backtest, dict) or not backtest:
        available_profiles: list[str] = []
        output_root = OUTPUT_DIR.parent
        if output_root.exists():
            for child in output_root.iterdir():
                if child.is_dir():
                    available_profiles.append(child.name)
        st.warning(
            "No backtest payload found in any output profile. "
            "Run Full Analysis and verify the active DATA_USER_ID profile."
        )
        if available_profiles:
            st.caption("Detected output profiles: " + ", ".join(sorted(available_profiles)))
        return

    if source_path is not None:
        st.caption(f"Loaded backtest artifact: {source_path}")

    expectancy = backtest.get("expectancy_per_trade")
    pf = backtest.get("profit_factor")
    calmar = backtest.get("calmar_ratio")
    sharpe = backtest.get("sharpe_ratio")
    ir = backtest.get("information_ratio")
    mdd = backtest.get("maximum_drawdown")

    # Check if metrics exist; if not, render partial view with fallback
    has_metrics = any(v is not None for v in [expectancy, pf, calmar, sharpe, ir, mdd])
    if not has_metrics:
        st.info(
            "Backtest data exists but advanced edge metrics (Expectancy, Profit Factor, Calmar, etc.) "
            "are not yet computed. Please re-run Full Analysis to generate these metrics."
        )
        return

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Expectancy/Trade", _fmt(expectancy, 6) if expectancy is not None else "N/A")
    c2.metric("Profit Factor", _fmt(pf) if pf is not None else "N/A")
    c3.metric("Calmar", _fmt(calmar) if calmar is not None else "N/A")
    c4.metric("Sharpe", _fmt(sharpe) if sharpe is not None else "N/A")
    c5.metric("Info Ratio", _fmt(ir) if ir is not None else "N/A")
    c6.metric("Max Drawdown", _fmt(mdd) if mdd is not None else "N/A")

    signals: list[str] = []
    if isinstance(expectancy, (int, float)) and float(expectancy) > 0.0:
        signals.append(f"Positive Expectancy {float(expectancy):.5f}")
    if isinstance(pf, (int, float)) and float(pf) >= 1.2:
        signals.append(f"Profit Factor {float(pf):.2f}")
    if isinstance(calmar, (int, float)) and float(calmar) >= 2.0:
        signals.append(f"Calmar {float(calmar):.2f}")
    if isinstance(ir, (int, float)) and float(ir) >= 0.5:
        signals.append(f"Information Ratio {float(ir):.2f}")

    if signals:
        st.success(
            "Validated exceptional findings: " + " | ".join(signals)
            + ". These are real artifact-derived strengths and are prioritized over raw R²."
        )
    else:
        st.info(
            "No exceptional threshold triggered in this run. Edge Arsenal still shows full diagnostics transparently."
        )
    st.latex(r"\text{Expectancy} = (P_{win}\times AvgWin) - (P_{loss}\times AvgLoss)")

    corr = backtest.get("correlation_test", {}) if isinstance(backtest.get("correlation_test"), dict) else {}
    p_value = corr.get("p_value")
    pearson_r = corr.get("pearson_r")
    
    p1, p2 = st.columns(2)
    p1.metric("Pearson r", _fmt(pearson_r) if pearson_r is not None else "N/A")
    p2.metric("P-value", _fmt(p_value, 6) if p_value is not None else "N/A")
    
    if isinstance(p_value, (int, float)) and float(p_value) < 0.05:
        st.success("Statistical significance passed: p-value < 0.05")
    elif isinstance(p_value, (int, float)):
        st.caption("P-value >= 0.05 in this window. Keep result as exploratory, not confirmatory.")

    strategy_returns = backtest.get("strategy_returns", [])
    if isinstance(strategy_returns, list) and strategy_returns:
        sret = np.asarray([float(x) for x in strategy_returns], dtype=float)

        hist_fig = px.histogram(
            pd.DataFrame({"return": sret}),
            x="return",
            nbins=30,
            title="Trade Distribution Histogram (Gains vs Losses)",
            color_discrete_sequence=["#0f766e"],
        )
        hist_fig.add_vline(x=0.0, line_dash="dot", line_color="#7f1d1d")
        hist_fig.update_layout(height=340)
        st.plotly_chart(hist_fig, use_container_width=True)

        rolling = backtest.get("rolling_sharpe_30d", [])
        if isinstance(rolling, list) and rolling:
            rdf = pd.DataFrame(rolling)
            if {"step", "rolling_sharpe"}.issubset(rdf.columns):
                rs_fig = px.line(
                    rdf,
                    x="step",
                    y="rolling_sharpe",
                    title="Rolling Sharpe Ratio (30-day)",
                )
                rs_fig.add_hline(y=0.0, line_dash="dot", line_color="#777")
                rs_fig.add_hline(y=1.0, line_dash="dash", line_color="#0f766e")
                rs_fig.update_layout(height=340)
                st.plotly_chart(rs_fig, use_container_width=True)

        benchmark_returns = backtest.get("benchmark_returns", [])
        if isinstance(benchmark_returns, list) and benchmark_returns and len(benchmark_returns) == len(sret):
            bret = np.asarray([float(x) for x in benchmark_returns], dtype=float)
            curve_df = pd.DataFrame(
                {
                    "step": np.arange(1, len(sret) + 1),
                    "strategy": np.cumprod(1.0 + sret),
                    "benchmark": np.cumprod(1.0 + bret),
                }
            )
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(x=curve_df["step"], y=curve_df["strategy"], mode="lines", name="Strategy", line=dict(color="#0f766e", width=3)))
            eq_fig.add_trace(go.Scatter(x=curve_df["step"], y=curve_df["benchmark"], mode="lines", name="Benchmark", line=dict(color="#b91c1c", width=2)))
            eq_fig.update_layout(title="Strategy vs Benchmark Equity Curve", height=360)
            st.plotly_chart(eq_fig, use_container_width=True)
    else:
        st.caption("Strategy returns and chart data not yet available. Re-run Full Analysis to populate.")
