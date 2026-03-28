from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from UI.constants import OUTPUT_DIR, USER_DATA_DIR


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _artifact_row(path: Path, label: str) -> dict[str, str]:
    if not path.exists():
        return {"Layer": label, "Path": str(path), "Exists": "No", "Modified": "N/A"}
    stat = path.stat()
    return {
        "Layer": label,
        "Path": str(path),
        "Exists": "Yes",
        "Modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def _fmt(value: object, ndigits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    if isinstance(value, (int, bool)):
        return str(value)
    if value is None:
        return "—"
    return str(value)


# ── Executive-grade metric formatters ─────────────────────────────────────────
# Rules: percentages where appropriate, 2 decimal places for ratios,
# hard caps on astronomically large values that would mislead stakeholders.

def _fmt_pct(v: float) -> str:
    """Format a decimal fraction as a percentage string, e.g. -0.082 → '-8.2%'."""
    return f"{v * 100.0:+.1f}%"


def _fmt_ratio(v: float, suffix: str = "×", cap: float | None = None, cap_label: str | None = None) -> str:
    """Format a ratio with optional cap for extreme values."""
    if cap is not None and abs(v) > cap:
        label = cap_label or f"≥ {cap:.0f}{suffix}"
        return label
    return f"{v:.2f}{suffix}"


def _fmt_sharpe(v: float) -> str:
    return f"{v:.2f}" if abs(v) <= 5.0 else ("≥ 5.0" if v > 0 else "≤ -5.0")


def _fmt_expectancy(v: float) -> str:
    """Daily log-return units, 4 decimal places, always signed."""
    return f"{v:+.4f}"


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
    # Use log-sum to avoid np.prod overflow on large series; equivalent to geometric CAGR
    clipped = np.clip(returns, -0.9999, None)
    log_sum = float(np.sum(np.log1p(clipped)))
    if not np.isfinite(log_sum):
        return 0.0
    years = max(returns.size / float(periods_per_year), 1.0 / float(periods_per_year))
    ann = float(np.exp(log_sum / years) - 1.0)
    # Cap at ±9900% to prevent astronomic Calmar values caused by compounding on
    # inflated return data (e.g. returns expressed in non-daily units).
    return float(np.clip(ann, -0.99, 99.0))


def _infer_periods_per_year(backtest: dict) -> int:
    if not isinstance(backtest, dict):
        return 252
    target = str(backtest.get("target", "log_return"))
    transforms = backtest.get("transformations")
    if not isinstance(transforms, dict):
        return 252
    target_meta = transforms.get(target)
    if not isinstance(target_meta, dict):
        return 252
    horizon = int(target_meta.get("target_horizon_days", 1) or 1)
    horizon = max(1, horizon)
    return max(1, int(round(252.0 / float(horizon))))


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

            # Always recompute maximum_drawdown from strategy returns; the value
            # pre-stored in the artifact may have been computed incorrectly (e.g.
            # min of raw return vector instead of peak-to-trough on equity curve).
            out["maximum_drawdown"] = _max_drawdown_from_returns(sret)

            wins = sret[sret > 0.0]
            losses = sret[sret < 0.0]
            win_prob = float(len(wins) / len(sret)) if len(sret) else 0.0
            loss_prob = float(len(losses) / len(sret)) if len(sret) else 0.0
            avg_win = float(np.mean(wins)) if len(wins) else 0.0
            avg_loss_abs = float(abs(np.mean(losses))) if len(losses) else 0.0

            out["expectancy_per_trade"] = float((win_prob * avg_win) - (loss_prob * avg_loss_abs))

            gross_profit = float(np.sum(wins)) if len(wins) else 0.0
            gross_loss_abs = float(abs(np.sum(losses))) if len(losses) else 0.0
            out["profit_factor"] = (
                float(gross_profit / gross_loss_abs)
                if gross_loss_abs > 1e-12
                else (None if gross_profit == 0.0 else float("inf"))
            )

            stdev = float(np.std(sret, ddof=1)) if len(sret) > 1 else None
            if stdev is not None and stdev > 1e-12:
                out["sharpe_ratio"] = float(np.mean(sret) / stdev * np.sqrt(252.0))
            else:
                out["sharpe_ratio"] = None

            ann_return = _annualized_return(
                sret,
                periods_per_year=_infer_periods_per_year(out),
            )
            out["annualized_return"] = float(ann_return)
            mdd = float(out.get("maximum_drawdown") or 0.0)
            # Use a floor of 1% on |MDD| to prevent division-by-near-zero
            denom = max(abs(mdd), 0.01)
            out["calmar_ratio"] = float(ann_return / denom)

            bret = out.get("benchmark_returns")
            if isinstance(bret, list) and bret and len(bret) == len(sret):
                b = np.asarray([float(x) for x in bret], dtype=float)
                active = sret - b
                te = float(np.std(active, ddof=1)) if len(active) > 1 else None
                if te is not None and te > 1e-12:
                    out["information_ratio"] = float(np.mean(active) / te * np.sqrt(252.0))
                else:
                    out["information_ratio"] = None
            else:
                out["information_ratio"] = None

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
    # 1) Prefer the dedicated artifact over embedded summary payloads because
    # summary files are often older and can contain stale copied metrics.
    current_bt = _read_json(OUTPUT_DIR / "backtest_2020.json")
    bt = _extract_backtest(current_bt)
    if bt:
        return bt, OUTPUT_DIR / "backtest_2020.json"

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

    st.markdown("### 🔗 Data Lineage Health")
    lineage_rows = [
        _artifact_row(USER_DATA_DIR / "raw" / "catalog.json", "Bronze catalog"),
        _artifact_row(USER_DATA_DIR / "processed" / "quality" / "quality_report.json", "Silver quality"),
        _artifact_row(USER_DATA_DIR / "gold" / "master_table.parquet", "Gold master"),
        _artifact_row(OUTPUT_DIR / "analysis_results.json", "Output summary"),
        _artifact_row(OUTPUT_DIR / "backtest_2020.json", "Backtest artifact"),
    ]
    st.dataframe(pd.DataFrame(lineage_rows), width="stretch", hide_index=True)

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

    # ── Row 1: Risk-adjusted performance metrics ──────────────────────────────
    # All values formatted for executive/stakeholder readability:
    # percentages for return-scale metrics, capped ratios (no astronomical values),
    # delta indicators showing direction vs institutional thresholds.
    st.markdown("#### Performance Scorecard")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    # Expectancy: signed log-return units, 4 decimal places
    if expectancy is not None:
        _exp_v = float(expectancy)
        c1.metric(
            "Expectancy / Trade",
            _fmt_expectancy(_exp_v),
            delta="Positive edge" if _exp_v > 0 else "Negative edge",
            delta_color="normal" if _exp_v > 0 else "inverse",
            help="Average log-return per trade. Positive = strategy earns more than it loses on average.",
        )

    # Profit Factor: gross_profit / gross_loss, capped at ≥ 10×
    if pf is not None and pf != "inf":
        _pf_v = float(pf)
        _pf_delta = f"vs 1.0 breakeven"
        c2.metric(
            "Profit Factor",
            _fmt_ratio(_pf_v, suffix="×", cap=10.0, cap_label="≥ 10×"),
            delta=_pf_delta,
            delta_color="normal" if _pf_v >= 1.0 else "inverse",
            help="Gross gains ÷ gross losses. >1.0 = profitable, >1.5 = strong, >2.0 = excellent.",
        )

    # Calmar: annualised return / max drawdown, capped at ≥ 20× (>20 is unrealistic for real portfolios)
    if calmar is not None:
        _cal_v = float(calmar)
        c3.metric(
            "Calmar Ratio",
            _fmt_ratio(_cal_v, suffix="×", cap=20.0, cap_label="≥ 20×"),
            delta="Strong" if _cal_v >= 3.0 else ("Acceptable" if _cal_v >= 1.0 else "Weak"),
            delta_color="normal" if _cal_v >= 1.0 else "inverse",
            help="Annualised return ÷ Max Drawdown. >1.0 = risk-adjusted profitability. >3.0 = institutional grade.",
        )

    # Sharpe: capped at ≥ 5.0 (unrealistically high values mislead executives)
    if sharpe is not None:
        _sh_v = float(sharpe)
        c4.metric(
            "Sharpe Ratio",
            _fmt_sharpe(_sh_v),
            delta=f"vs 1.0 target",
            delta_color="normal" if _sh_v >= 0.5 else "inverse",
            help="Return per unit of volatility (risk-free rate = 0). >0.5 = acceptable, >1.0 = strong, >2.0 = exceptional.",
        )

    # Information Ratio: capped at ≥ 5.0
    if ir is not None:
        _ir_v = float(ir)
        c5.metric(
            "Info Ratio",
            _fmt_sharpe(_ir_v),
            delta="vs benchmark",
            delta_color="normal" if _ir_v >= 0.0 else "inverse",
            help="Active return vs benchmark per unit of tracking error. >0.5 = adds value over benchmark.",
        )

    # Max Drawdown: formatted as percentage (−8.2% not −0.082)
    if mdd is not None:
        _mdd_v = float(mdd)
        c6.metric(
            "Max Drawdown",
            _fmt_pct(_mdd_v),
            delta="Peak-to-trough loss",
            delta_color="off",
            help="Largest peak-to-trough loss on the equity curve. <−20% requires close attention.",
        )

    st.markdown("")

    # ── Strategic Edge Quality Score ──────────────────────────────────────────
    score = 0.0
    if isinstance(expectancy, (int, float)) and expectancy > 0:
        score += 25.0
    if isinstance(pf, (int, float)) and pf != float("inf"):
        score += min(max((float(pf) - 1.0) * 30.0, 0.0), 25.0)
    if isinstance(calmar, (int, float)):
        score += min(max(min(float(calmar), 20.0) * 1.0, 0.0), 20.0)
    if isinstance(sharpe, (int, float)):
        score += min(max(min(float(sharpe), 5.0) * 4.0, 0.0), 20.0)
    if isinstance(ir, (int, float)):
        score += min(max(min(float(ir), 5.0) * 2.0, 0.0), 10.0)
    score = min(score, 100.0)

    _score_color = "🟢" if score >= 65 else ("🟡" if score >= 35 else "🔴")
    st.progress(
        score / 100.0,
        text=f"{_score_color} Strategic Edge Quality Score: {score:.0f} / 100"
             " — composite of Expectancy, Profit Factor, Calmar, Sharpe & IR",
    )

    # ── Validated findings banner ─────────────────────────────────────────────
    signals: list[str] = []
    if isinstance(expectancy, (int, float)) and float(expectancy) > 0.0:
        signals.append(f"Positive Expectancy ({_fmt_expectancy(float(expectancy))} / trade)")
    if isinstance(pf, (int, float)) and pf != float("inf") and float(pf) >= 1.2:
        signals.append(f"Profit Factor {float(pf):.2f}×")
    if isinstance(calmar, (int, float)) and 0 < float(calmar) <= 20.0 and float(calmar) >= 2.0:
        signals.append(f"Calmar {float(calmar):.2f}×")
    if isinstance(ir, (int, float)) and float(ir) >= 0.5:
        signals.append(f"Information Ratio {float(ir):.2f}")

    if signals:
        st.success("**Validated edge signals:** " + " · ".join(signals))
    else:
        st.info("No exceptional threshold triggered in this run. Full diagnostics shown transparently below.")

    # ── Statistical significance ──────────────────────────────────────────────
    corr = backtest.get("correlation_test", {}) if isinstance(backtest.get("correlation_test"), dict) else {}
    p_value = corr.get("p_value")
    pearson_r = corr.get("pearson_r")

    if pearson_r is not None or p_value is not None:
        st.markdown("#### Predictive Signal Test")
        p1, p2 = st.columns(2)
        if pearson_r is not None:
            p1.metric(
                "Pearson r",
                f"{float(pearson_r):.3f}",
                help="Correlation between model signal and realised returns. |r| > 0.15 is meaningful in macro forecasting.",
            )
        if p_value is not None:
            _pv = float(p_value)
            p2.metric(
                "P-value",
                f"{_pv:.4f}",
                delta="Significant" if _pv < 0.05 else "Exploratory only",
                delta_color="normal" if _pv < 0.05 else "off",
                help="< 0.05 = statistically significant. In macro settings, treat as supportive context, not sole validation.",
            )
        if isinstance(p_value, (int, float)) and float(p_value) < 0.05:
            st.success("Statistical significance passed: p-value < 0.05")
        else:
            st.caption("P-value ≥ 0.05 — treat as exploratory context. Macro-equity models rarely achieve high significance.")

    # ── Charts ────────────────────────────────────────────────────────────────
    strategy_returns = backtest.get("strategy_returns", [])
    if isinstance(strategy_returns, list) and strategy_returns:
        sret = np.asarray([float(x) for x in strategy_returns], dtype=float)
        sret_pct = sret * 100.0  # convert to basis points / percentage for readability

        st.markdown("#### Return Distribution")
        hist_fig = px.histogram(
            pd.DataFrame({"Daily Return (%)": sret_pct}),
            x="Daily Return (%)",
            nbins=40,
            title="Daily Return Distribution — Gains vs Losses",
            color_discrete_sequence=["#0f766e"],
            labels={"Daily Return (%)": "Daily Return (%)"},
        )
        hist_fig.add_vline(x=0.0, line_dash="dot", line_color="#7f1d1d", annotation_text="Break-even")
        hist_fig.update_layout(
            height=340,
            bargap=0.05,
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            legend_title_text="",
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        rolling = backtest.get("rolling_sharpe_30d", [])
        if isinstance(rolling, list) and rolling:
            rdf = pd.DataFrame(rolling)
            if {"step", "rolling_sharpe"}.issubset(rdf.columns):
                # Clip extreme rolling Sharpe values for chart readability
                rdf["rolling_sharpe"] = rdf["rolling_sharpe"].clip(-4.0, 4.0)
                rs_fig = px.line(
                    rdf,
                    x="step",
                    y="rolling_sharpe",
                    title="Rolling 30-Day Sharpe Ratio  (clipped ±4 for readability)",
                    labels={"rolling_sharpe": "Sharpe Ratio", "step": "Trading Day"},
                    color_discrete_sequence=["#0f766e"],
                )
                rs_fig.add_hline(y=0.0, line_dash="dot", line_color="#9ca3af", annotation_text="0")
                rs_fig.add_hline(y=0.5, line_dash="dash", line_color="#f59e0b", annotation_text="0.5 — acceptable")
                rs_fig.add_hline(y=1.0, line_dash="dash", line_color="#2e7d32", annotation_text="1.0 — strong")
                rs_fig.update_layout(height=340, yaxis_title="Sharpe Ratio")
                st.plotly_chart(rs_fig, use_container_width=True)

        benchmark_returns = backtest.get("benchmark_returns", [])
        if isinstance(benchmark_returns, list) and benchmark_returns and len(benchmark_returns) == len(sret):
            bret = np.asarray([float(x) for x in benchmark_returns], dtype=float)
            _dates = backtest.get("dates")
            if isinstance(_dates, list) and len(_dates) == len(sret):
                _x_axis = pd.to_datetime(_dates, errors="coerce")
            else:
                _x_axis = np.arange(1, len(sret) + 1)

            curve_df = pd.DataFrame({
                "x": _x_axis,
                "Strategy": np.cumprod(1.0 + sret),
                "Buy & Hold": np.cumprod(1.0 + bret),
            })

            st.markdown("#### Strategy vs Buy-and-Hold Equity Curve")
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                x=curve_df["x"], y=curve_df["Strategy"],
                mode="lines", name="Strategy",
                line=dict(color="#0f766e", width=2.5),
            ))
            eq_fig.add_trace(go.Scatter(
                x=curve_df["x"], y=curve_df["Buy & Hold"],
                mode="lines", name="Buy & Hold",
                line=dict(color="#b91c1c", width=1.8, dash="dot"),
            ))
            eq_fig.add_hline(y=1.0, line_dash="dot", line_color="#9ca3af", annotation_text="Starting value")
            eq_fig.update_layout(
                height=400,
                yaxis_title="Portfolio Value  ($1 = initial investment)",
                xaxis_title="Date" if isinstance(_x_axis, pd.DatetimeIndex) else "Trading Day",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            st.plotly_chart(eq_fig, use_container_width=True)
            _final_strat = float(curve_df["Strategy"].iloc[-1])
            _final_bm = float(curve_df["Buy & Hold"].iloc[-1])
            _alpha = _final_strat - _final_bm
            _alpha_pct = _alpha / max(abs(_final_bm), 1e-6) * 100
            _col_a, _col_b, _col_c = st.columns(3)
            _col_a.metric("Strategy final value", f"${_final_strat:.2f}", help="Per $1 invested at start")
            _col_b.metric("Buy & Hold final", f"${_final_bm:.2f}", help="Per $1 invested at start")
            _col_c.metric(
                "Alpha vs Benchmark",
                f"{_alpha_pct:+.1f}%",
                delta_color="normal" if _alpha >= 0 else "inverse",
                help="Strategy outperformance over buy-and-hold over the full period.",
            )
    else:
        st.caption("Strategy returns and chart data not yet available. Re-run Full Analysis to populate.")
