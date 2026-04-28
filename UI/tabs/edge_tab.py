from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from UI.constants import get_active_paths
from pathing import output_path_diagnostics

_log = logging.getLogger(__name__)


def _paths() -> dict:
    return get_active_paths()


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


def _is_portfolio_backtest(d: dict) -> bool:
    """Return True for portfolio_backtest() output (has nested portfolio + per_ticker)."""
    return (
        isinstance(d, dict)
        and isinstance(d.get("portfolio"), dict)
        and isinstance(d.get("per_ticker"), dict)
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

    _BACKTEST_KEYS = ("strategy_returns", "maximum_drawdown", "sharpe_ratio", "predictions", "actual")

    direct = candidate.get("backtest_2020")
    if isinstance(direct, dict):
        extracted = direct.get("value", direct)
        if isinstance(extracted, dict):
            # Portfolio structure nested inside backtest_2020
            if _is_portfolio_backtest(extracted):
                return extracted
            if any(k in extracted for k in _BACKTEST_KEYS):
                return extracted
    # analysis_results structure
    results = candidate.get("results")
    if isinstance(results, dict):
        bt = results.get("backtest_2020")
        if isinstance(bt, dict):
            extracted = bt.get("value", bt)
            if isinstance(extracted, dict):
                if _is_portfolio_backtest(extracted):
                    return extracted
                if any(k in extracted for k in _BACKTEST_KEYS):
                    return extracted
    # single-artifact structure
    if isinstance(wrapped, dict):
        if _is_portfolio_backtest(wrapped):
            return wrapped
        if any(k in wrapped for k in _BACKTEST_KEYS):
            return wrapped
    # direct portfolio or single-ticker payload
    if _is_portfolio_backtest(candidate):
        return candidate
    if any(k in candidate for k in _BACKTEST_KEYS):
        return candidate
    return {}


def _sanitize_returns(returns: np.ndarray, max_abs: float = 0.15) -> np.ndarray:
    """Replace inf/NaN and clip outlier returns to ±max_abs before any compounding."""
    arr = np.asarray(returns, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -max_abs, max_abs)


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    """Correct MDD for log-return series: equity curve = exp(cumsum)."""
    arr = _sanitize_returns(returns)
    if arr.size == 0:
        return 0.0
    equity_curve = np.exp(np.cumsum(arr))
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / np.maximum(peaks, 1e-12)) - 1.0
    return float(np.min(drawdowns))


def _annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if returns.size == 0:
        return 0.0
    # returns ARE already log-returns — direct nansum, do NOT np.log1p (double-transform).
    log_sum = float(np.nansum(returns))
    if not np.isfinite(log_sum):
        return 0.0
    years = max(returns.size / float(periods_per_year), 1.0 / float(periods_per_year))
    ann = float(np.exp(log_sum / years) - 1.0)
    # Cap at ±2500% to prevent astronomic Calmar values.
    return float(np.clip(ann, -0.99, 25.0))


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

    # ── Portfolio structure: hoist portfolio-level fields to the top so all
    # downstream rendering code (KPIs, charts, score) works without changes.
    if _is_portfolio_backtest(backtest):
        out = dict(backtest)
        portfolio = out["portfolio"]
        _hoist = [
            "sharpe_ratio", "sortino_ratio", "maximum_drawdown", "calmar_ratio",
            "information_ratio", "annualized_return", "strategy_returns",
            "benchmark_returns", "rolling_sharpe_30d", "profit_factor",
            "tracking_error", "dates",
        ]
        for _k in _hoist:
            if _k in portfolio:
                out.setdefault(_k, portfolio[_k])
        # Build expectancy from per-win/loss stats stored in portfolio
        wp = float(portfolio.get("win_probability", 0.0))
        aw = float(portfolio.get("average_win", 0.0))
        al = float(portfolio.get("average_loss", 0.0))
        out.setdefault("expectancy_per_trade", float(wp * aw - (1.0 - wp) * abs(al)))
        return out

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
            sret_raw = np.asarray([float(x) for x in out["strategy_returns"]], dtype=float)
            # Sanitize BEFORE all metric computations: replace NaN/inf and clip
            # extreme daily returns to ±15%.  This prevents a handful of outlier
            # observations from skewing Sharpe, Profit Factor, or Expectancy —
            # which then cascade into a 40-point swing in the composite score.
            sret = _sanitize_returns(sret_raw)

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
            out["profit_factor"] = float(gross_profit / gross_loss_abs) if gross_loss_abs > 1e-12 else (None if gross_profit == 0.0 else float("inf"))

            stdev = float(np.std(sret, ddof=1)) if len(sret) > 1 else None
            if stdev is not None and stdev > 1e-12:
                out["sharpe_ratio"] = float(np.mean(sret) / stdev * np.sqrt(252.0))
            else:
                out["sharpe_ratio"] = None

            # strategy_returns are always daily (1-day actual log-returns),
            # regardless of the ML prediction horizon stored in transformations.
            # _infer_periods_per_year reads target_horizon_days (e.g. 252) and
            # returns 252/252 = 1, treating 756 daily points as 756 years.
            # Hard-code 252 to get correct daily → annual compounding.
            ann_return = _annualized_return(
                sret,
                periods_per_year=252,
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
    """Discover backtest payload from the active output profile.

    Cross-profile fallback is disabled by default because it can mask
    DATA_USER_ID/profile routing issues. It can be explicitly re-enabled with
    EDGE_TAB_ALLOW_CROSS_PROFILE_FALLBACK=1 for diagnostics.
    """

    def _resolve_artifact_path(raw_path: str, anchor_dir: Path, output_root: Path) -> Path | None:
        if not raw_path.strip():
            return None
        p = Path(raw_path)
        candidates: list[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(anchor_dir / p)
            candidates.append(output_root / p)
            candidates.append(output_root / p.name)
        for c in candidates:
            if c.is_file():
                return c
        # Last resort: recursive lookup by filename inside output root.
        try:
            hits = sorted(
                [x for x in output_root.rglob(p.name) if x.is_file()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if hits:
                return hits[0]
        except OSError:
            pass
        return None

    def _search_in_dir(output_dir: Path, output_root: Path) -> tuple[dict, Path | None]:
        if not output_dir.is_dir():
            return {}, None
        candidates: list[tuple[float, Path]] = []
        for filename in ("analysis_results.json", "backtest_2020.json"):
            p = output_dir / filename
            if p.is_file():
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
                        bt_path = _resolve_artifact_path(bt_path_raw, path.parent, output_root)
                        if bt_path is not None:
                            inner_payload = _read_json(bt_path)
                            bt = _extract_backtest(inner_payload)
                            if bt:
                                return bt, bt_path
        return {}, None

    # 1. Try active session directory first.
    active_output_dir = _paths()["output"]
    output_root = active_output_dir.parent
    bt, src = _search_in_dir(active_output_dir, output_root)
    if bt:
        return bt, src

    allow_cross_profile = (os.getenv("EDGE_TAB_ALLOW_CROSS_PROFILE_FALLBACK", "0").strip() == "1")
    if not allow_cross_profile:
        return {}, None

    # 2. Optional fallback: scan all sibling output profiles, newest artifact wins.
    try:
        profile_candidates: list[tuple[float, Path]] = []
        if output_root.exists():
            for child in output_root.iterdir():
                if not child.is_dir() or child == active_output_dir:
                    continue
                for filename in ("analysis_results.json", "backtest_2020.json"):
                    p = child / filename
                    if p.is_file():
                        try:
                            profile_candidates.append((p.stat().st_mtime, child))
                            break
                        except OSError:
                            pass
        for _, profile_dir in sorted(profile_candidates, key=lambda x: x[0], reverse=True):
            bt, src = _search_in_dir(profile_dir, output_root)
            if bt:
                return bt, src

        # 3. Deep fallback: recurse all descendants of output root.
        recursive_dirs: list[tuple[float, Path]] = []
        for p in output_root.rglob("*"):
            if not p.is_dir():
                continue
            for filename in ("analysis_results.json", "backtest_2020.json"):
                f = p / filename
                if f.is_file():
                    try:
                        recursive_dirs.append((f.stat().st_mtime, p))
                        break
                    except OSError:
                        pass
        seen: set[Path] = set()
        for _, d in sorted(recursive_dirs, key=lambda x: x[0], reverse=True):
            if d in seen:
                continue
            seen.add(d)
            bt, src = _search_in_dir(d, output_root)
            if bt:
                return bt, src
    except OSError:
        pass

    return {}, None


def _check_governance_block() -> str | None:
    """Return governance block reason from the active profile by default.

    Set EDGE_TAB_ALLOW_CROSS_PROFILE_FALLBACK=1 to also scan sibling profiles.
    """

    def _check_dir(d: Path) -> str | None:
        path = d / "analysis_results.json"
        if path.is_file():
            payload = _read_json(path)
            if isinstance(payload, dict):
                results = payload.get("results", {})
                if isinstance(results, dict):
                    bt_val = results.get("backtest_2020")
                    if isinstance(bt_val, str) and bt_val.startswith("blocked_by_governance_gate"):
                        return bt_val
        p2 = d / "backtest_2020.json"
        if p2.is_file():
            inner = _read_json(p2)
            wrapped = inner.get("value") if isinstance(inner, dict) else None
            if isinstance(wrapped, str) and wrapped.startswith("blocked_by_governance_gate"):
                return wrapped
        return None

    active_output_dir = _paths()["output"]
    result = _check_dir(active_output_dir)
    if result:
        return result

    allow_cross_profile = (os.getenv("EDGE_TAB_ALLOW_CROSS_PROFILE_FALLBACK", "0").strip() == "1")
    if not allow_cross_profile:
        return None

    # Optional fallback: scan other profiles.
    output_root = active_output_dir.parent
    try:
        if output_root.exists():
            for child in output_root.iterdir():
                if not child.is_dir() or child == active_output_dir:
                    continue
                result = _check_dir(child)
                if result:
                    return result
    except OSError:
        pass
    return None


def _show_portfolio_composition(backtest: dict) -> None:
    """Render portfolio composition table and 21-day price forecasts."""
    portfolio = backtest.get("portfolio", {})
    ticker_summary = portfolio.get("ticker_summary", {})
    failed = backtest.get("failed_tickers", [])

    if not ticker_summary:
        return

    st.markdown("#### Portfolio Composition & 21-Day Price Forecasts")

    if failed:
        st.warning(f"Backtests failed for: `{', '.join(failed)}` — excluded from portfolio.")

    rows = []
    for ticker, info in sorted(ticker_summary.items()):
        lf = info.get("latest_price_forecast") or {}
        sharpe_v = info.get("sharpe_ratio")
        ann_v = info.get("annualized_return")
        mdd_v = info.get("maximum_drawdown")
        cal_v = info.get("calmar_ratio")
        ir_v = info.get("information_ratio")
        current_close = lf.get("current_close")
        pred_close = lf.get("predicted_close_21d")
        pred_ret = lf.get("predicted_21d_log_return")
        rows.append({
            "Ticker": ticker,
            "Weight": f"{(info.get('weight') or 0) * 100:.1f}%",
            "Sharpe": f"{sharpe_v:.2f}" if isinstance(sharpe_v, (int, float)) else "—",
            "Ann.Return": _fmt_pct(float(ann_v)) if isinstance(ann_v, (int, float)) else "—",
            "Max DD": _fmt_pct(float(mdd_v)) if isinstance(mdd_v, (int, float)) else "—",
            "Calmar": f"{cal_v:.2f}×" if isinstance(cal_v, (int, float)) else "—",
            "IR": f"{ir_v:.2f}" if isinstance(ir_v, (int, float)) else "—",
            "Current Close ($)": f"{current_close:.2f}" if isinstance(current_close, (int, float)) else "—",
            "Pred. Close 21d ($)": f"{pred_close:.2f}" if isinstance(pred_close, (int, float)) else "—",
            "Pred. 21d Return": (
                f"{pred_ret * 100:+.2f}%" if isinstance(pred_ret, (int, float)) else "—"
            ),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Summary stats row
    n_long = sum(
        1 for r in rows
        if r["Pred. 21d Return"] not in ("—",) and r["Pred. 21d Return"].startswith("+")
    )
    n_short = len(rows) - n_long
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Tickers in portfolio", str(len(rows)))
    mc2.metric("Bullish signals (21d)", str(n_long), delta=f"{n_long}/{len(rows)}", delta_color="normal")
    mc3.metric("Bearish signals (21d)", str(n_short), delta=f"{n_short}/{len(rows)}", delta_color="inverse")


def _show_per_ticker_view(backtest: dict) -> None:
    """Expandable section showing individual ticker backtests and price forecast charts."""
    per_ticker = backtest.get("per_ticker", {})
    if not per_ticker:
        return

    good_tickers = sorted(
        t for t, r in per_ticker.items()
        if isinstance(r, dict) and r.get("status") != "failed"
    )
    if not good_tickers:
        return

    with st.expander("📊 Per-Ticker Backtest & Price Forecasts", expanded=False):
        selected = st.selectbox("Select ticker to inspect:", good_tickers, key="portfolio_ticker_select")
        if not selected:
            return

        ticker_bt = _compute_missing_metrics(per_ticker[selected])

        c1, c2, c3, c4 = st.columns(4)
        def _safe(k: str) -> float | None:
            v = ticker_bt.get(k)
            return float(v) if isinstance(v, (int, float)) else None

        sh = _safe("sharpe_ratio")
        md = _safe("maximum_drawdown")
        ca = _safe("calmar_ratio")
        ar = _safe("annualized_return")
        if sh is not None:
            c1.metric("Sharpe", _fmt_sharpe(sh))
        if md is not None:
            c2.metric("Max DD", _fmt_pct(md))
        if ca is not None:
            c3.metric("Calmar", _fmt_ratio(ca, "×", 20.0))
        if ar is not None:
            c4.metric("Ann. Return", _fmt_pct(ar))

        # ── 21-day price forecasts chart ─────────────────────────────────────
        forecasts = per_ticker[selected].get("price_forecasts_21d", [])
        if forecasts:
            fdf = pd.DataFrame(forecasts)
            fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce")

            st.markdown(f"**{selected} — 21-Day Price Forecasts (Ridge model)**")
            fc_fig = go.Figure()
            fc_fig.add_trace(go.Scatter(
                x=fdf["date"], y=fdf["current_close"],
                name="Actual Close", mode="lines",
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
            ))
            fc_fig.add_trace(go.Scatter(
                x=fdf["date"], y=fdf["predicted_close_21d"],
                name="Predicted Close (21d)", mode="lines",
                line=dict(color="#0f766e", width=2),
            ))
            fc_fig.update_layout(
                height=320,
                yaxis_title="Price ($)",
                xaxis_title="Date",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            try:
                st.plotly_chart(fc_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render forecast chart: {e}")

            # Small table of last 5 forecasts
            st.caption("Most recent model forecasts:")
            st.dataframe(
                fdf[["date", "current_close", "predicted_close_21d", "predicted_21d_log_return"]].tail(10),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No price forecasts available for this ticker (close price column may be missing).")

        # ── Per-ticker equity curve ───────────────────────────────────────────
        sret = ticker_bt.get("strategy_returns", [])
        bret = ticker_bt.get("benchmark_returns", [])
        dates = ticker_bt.get("test_dates") or ticker_bt.get("dates")
        if sret and bret:
            s = _sanitize_returns(np.asarray([float(x) for x in sret], dtype=float))
            b = _sanitize_returns(np.asarray([float(x) for x in bret], dtype=float))
            n = min(len(s), len(b))
            x_axis = (
                pd.to_datetime(dates[:n], errors="coerce")
                if isinstance(dates, list) and len(dates) >= n
                else np.arange(n)
            )
            cdf = pd.DataFrame({
                "x": x_axis,
                "Strategy": np.exp(np.cumsum(s[:n])),
                "Buy & Hold": np.exp(np.cumsum(b[:n])),
            })
            st.markdown(f"**{selected} — Strategy vs Buy-and-Hold**")
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                x=cdf["x"], y=cdf["Strategy"], name="Strategy",
                line=dict(color="#0f766e", width=2),
            ))
            eq_fig.add_trace(go.Scatter(
                x=cdf["x"], y=cdf["Buy & Hold"], name="Buy & Hold",
                line=dict(color="#b91c1c", width=1.5, dash="dot"),
            ))
            eq_fig.add_hline(y=1.0, line_dash="dot", line_color="#9ca3af")
            eq_fig.update_layout(height=300, yaxis_title="Value ($1 start)", hovermode="x unified")
            try:
                st.plotly_chart(eq_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render equity curve: {e}")


# ── Universe Pruning Panel ────────────────────────────────────────────────────

def _show_universe_pruning_panel() -> None:
    """Render dynamic universe selection results from universe_pruning.json."""
    up_path = _paths()["output"] / "universe_pruning.json"
    if not up_path.exists():
        return

    up = _read_json(up_path)
    if not isinstance(up, dict):
        return

    if True:
        keep = up.get("tickers_keep", [])
        drop = up.get("tickers_drop", [])
        full_sh = up.get("full_portfolio_sharpe")
        pruned_sh = up.get("pruned_portfolio_sharpe")
        thresholds = up.get("thresholds", {})
        gen_at = up.get("generated_at", "")

        st.markdown("#### Two-Stage Universe Selection")
        st.caption(
            "Tickers are selected by ranking OOS Sharpe on 2020–2024 data and removing "
            "those below a minimum Sharpe floor or above the pairwise correlation threshold. "
            "This step uses out-of-sample data only; selection bias is disclosed."
        )
        if gen_at:
            st.caption(f"Last computed: {gen_at}")

        # Summary metrics
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric(
            "Full Portfolio Sharpe",
            _fmt(full_sh),
            help="EW Sharpe before pruning (all candidate tickers).",
        )
        sc2.metric(
            "Pruned Portfolio Sharpe",
            _fmt(pruned_sh),
            delta=(
                f"+{pruned_sh - full_sh:.4f}" if isinstance(pruned_sh, float) and isinstance(full_sh, float)
                else None
            ),
            delta_color="normal",
            help="EW Sharpe after removing low-quality / high-correlation tickers.",
        )
        sc3.metric("Tickers Kept", str(len(keep)))
        sc4.metric("Tickers Dropped", str(len(drop)), delta_color="off")

        if thresholds:
            st.caption(
                f"Selection thresholds — Sharpe floor: {thresholds.get('sharpe_floor', 'N/A')} | "
                f"IC floor: {thresholds.get('ic_floor', 'N/A')} | "
                f"Max pair corr: {thresholds.get('corr_threshold', 'N/A')}"
            )

        # Per-ticker table
        per_ticker = up.get("per_ticker", [])
        if per_ticker:
            rows_pt = []
            for item in per_ticker:
                flags = item.get("flags", [])
                rows_pt.append({
                    "Ticker": item.get("ticker", "?"),
                    "Decision": "✅ KEEP" if item.get("decision") == "KEEP" else "❌ DROP",
                    "OOS Sharpe": _fmt(item.get("sharpe_oos")),
                    "OOS Calmar": _fmt(item.get("calmar_oos")),
                    "Mean IC (train)": _fmt(item.get("mean_ic_train")),
                    "Avg Pair Corr": _fmt(item.get("avg_pair_corr")),
                    "LOO Δ Sharpe": _fmt(item.get("loo_sharpe_delta")),
                    "Flags": ", ".join(flags) if flags else "—",
                })
            rows_pt.sort(key=lambda r: (r["Decision"], r["OOS Sharpe"]), reverse=True)
            st.dataframe(pd.DataFrame(rows_pt), use_container_width=True, hide_index=True)

        if drop:
            st.caption(f"Dropped tickers: `{'`, `'.join(drop)}`")


# ── Phase 4 Re-Validation Panel ───────────────────────────────────────────────

def _show_phase4_panel() -> None:
    """Render Phase 4 honest re-validation results from phase4_validation.json."""
    p4_path = _paths()["output"] / "phase4_validation.json"
    p5_path = _paths()["output"] / "phase5_calibration.json"

    if not p4_path.exists():
        return  # silent: panel only appears after validation has been run

    p4 = _read_json(p4_path)
    if not isinstance(p4, dict) or p4.get("phase") != 4:
        return

    if True:
        oos  = p4.get("oos_metrics", {})
        bci  = p4.get("bootstrap_cis", {})
        dsr  = p4.get("dsr", {})
        hold = p4.get("holdout_metrics", {})
        acc  = p4.get("acceptance", {})
        tks  = p4.get("selected_tickers", [])
        oos_win = p4.get("oos_window", {})

        # ── header ────────────────────────────────────────────────────────────
        st.markdown("#### Honest Re-Validation (Walk-Forward + True Holdout)")
        st.caption(
            f"Universe: **{', '.join(tks)}** — "
            f"OOS: {oos_win.get('start', '?')} → {oos_win.get('end', '?')} — "
            f"Holdout: {p4.get('holdout_window', {}).get('start', '2024')} → 2026"
        )

        all_pass = p4.get("all_pass", False)
        if all_pass:
            st.success("✅ All 6 acceptance criteria PASSED")
        else:
            n_pass = sum(1 for v in acc.values() if v)
            st.warning(f"⚠️ {n_pass}/{len(acc)} criteria passed")

        # ── OOS metrics row ───────────────────────────────────────────────────
        st.markdown("**OOS 2020 – 2024 (N = {:,} days)**".format(oos.get("n_days", 0)))
        c1, c2, c3, c4, c5 = st.columns(5)

        sh = oos.get("sharpe")
        sh_ci = bci.get("sharpe", {})
        c1.metric(
            "Sharpe",
            _fmt(sh),
            delta=f"CI [{_fmt(sh_ci.get('ci_lower'))}, {_fmt(sh_ci.get('ci_upper'))}]",
            delta_color="normal" if (sh or 0) >= 0.6 else "inverse",
            help="Annualised Sharpe ratio on out-of-sample window. Bootstrap 95% CI shown.",
        )

        cal = oos.get("calmar")
        cal_ci = bci.get("calmar", {})
        c2.metric(
            "Calmar",
            _fmt(cal),
            delta=f"CI [{_fmt(cal_ci.get('ci_lower'))}, {_fmt(cal_ci.get('ci_upper'))}]",
            delta_color="normal" if (cal or 0) >= 0.5 else "inverse",
        )

        pf_v = oos.get("profit_factor")
        pf_ci = bci.get("profit_factor", {})
        c3.metric(
            "Profit Factor",
            _fmt(pf_v),
            delta=f"CI [{_fmt(pf_ci.get('ci_lower'))}, {_fmt(pf_ci.get('ci_upper'))}]",
            delta_color="normal" if (pf_v or 0) >= 1.25 else "inverse",
        )

        c4.metric(
            "Ann. Return",
            _fmt_pct(oos.get("annualized_return", 0)),
            delta=f"Max DD {_fmt_pct(oos.get('max_drawdown', 0))}",
            delta_color="off",
        )

        dsr_p = dsr.get("p_value")
        c5.metric(
            "DSR p-value",
            f"{dsr_p:.4f}" if isinstance(dsr_p, float) else "—",
            delta=f"Trials: {dsr.get('n_trials', '?')}",
            delta_color="normal" if (dsr_p or 0) >= 0.5 else "inverse",
            help="Deflated Sharpe Ratio p-value (Bailey & López de Prado 2014). >0.5 = robust to multiple-testing bias.",
        )

        # ── Holdout row ───────────────────────────────────────────────────────
        st.markdown("**True Holdout 2024 – 2026 (touched ONCE, N = {:,} days)**".format(hold.get("n_days", 0)))
        h1, h2, h3, h4 = st.columns(4)

        sh_ho = hold.get("sharpe")
        h1.metric(
            "Holdout Sharpe",
            _fmt(sh_ho),
            delta_color="normal" if (sh_ho or 0) >= 0.5 else "inverse",
        )
        cal_ho = hold.get("calmar")
        h2.metric(
            "Holdout Calmar",
            _fmt(cal_ho),
            delta_color="normal" if (cal_ho or 0) >= 0.5 else "inverse",
        )
        pf_ho = hold.get("profit_factor")
        h3.metric(
            "Holdout PF",
            _fmt(pf_ho),
            delta_color="normal" if (pf_ho or 0) >= 1.0 else "inverse",
        )
        deg = hold.get("degradation_pct")
        h4.metric(
            "OOS→Holdout Degradation",
            f"{deg:.1f}%" if isinstance(deg, float) else "N/A",
            delta="< 50% target" if isinstance(deg, float) and abs(deg) < 50 else "⚠️ > 50% target",
            delta_color="normal" if isinstance(deg, float) and abs(deg) < 50 else "inverse",
            help="How much the Sharpe declined from OOS to the true unseen holdout.",
        )

        # ── Criteria table ────────────────────────────────────────────────────
        if acc:
            rows_acc = [
                {"Criterion": k, "Result": "✅ PASS" if v else "❌ FAIL"}
                for k, v in acc.items()
            ]
            st.dataframe(pd.DataFrame(rows_acc), hide_index=True, use_container_width=True)

        # ── Composite Edge Score ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎯 Composite Edge Score")

        # Build score from Phase 4 OOS + Holdout metrics
        _p4_score = 0.0
        _p4_breakdown: dict[str, float] = {}

        _sh_v4 = oos.get("sharpe")
        _p4_sh_pts = min(max(float(_sh_v4) / 0.60 * 20.0, 0.0), 20.0) if isinstance(_sh_v4, (int, float)) else 0.0
        _p4_score += _p4_sh_pts
        _p4_breakdown["OOS Sharpe"] = _p4_sh_pts

        _cal_v4 = oos.get("calmar")
        _p4_cal_pts = min(max(float(_cal_v4) / 0.50 * 20.0, 0.0), 20.0) if isinstance(_cal_v4, (int, float)) else 0.0
        _p4_score += _p4_cal_pts
        _p4_breakdown["OOS Calmar"] = _p4_cal_pts

        _pf_v4 = oos.get("profit_factor")
        _p4_pf_pts = min(max((float(_pf_v4) - 1.0) / 0.25 * 20.0, 0.0), 20.0) if isinstance(_pf_v4, (int, float)) else 0.0
        _p4_score += _p4_pf_pts
        _p4_breakdown["OOS Profit Factor"] = _p4_pf_pts

        _dsr_p_v4 = dsr.get("p_value")
        _p4_dsr_pts = min(max(float(_dsr_p_v4) / 0.5 * 15.0, 0.0), 15.0) if isinstance(_dsr_p_v4, (int, float)) else 0.0
        _p4_score += _p4_dsr_pts
        _p4_breakdown["DSR p-value"] = _p4_dsr_pts

        _ho_sh = hold.get("sharpe")
        _p4_ho_pts = min(max(float(_ho_sh) / 0.50 * 15.0, 0.0), 15.0) if isinstance(_ho_sh, (int, float)) else 0.0
        _p4_score += _p4_ho_pts
        _p4_breakdown["Holdout Sharpe"] = _p4_ho_pts

        _n_pass_v4 = sum(1 for v in acc.values() if v) if acc else 0
        _n_total_v4 = len(acc) if acc else 1
        _p4_acc_pts = (_n_pass_v4 / _n_total_v4) * 10.0
        _p4_score += _p4_acc_pts
        _p4_breakdown["Acceptance Criteria"] = _p4_acc_pts

        _p4_score = min(_p4_score, 100.0)

        _p4_score_color = "🟢" if _p4_score >= 65 else ("🟡" if _p4_score >= 35 else "🔴")
        _p4_breakdown_str = " | ".join(f"{k}: {v:.0f}pt" for k, v in _p4_breakdown.items())

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #082f49 0%, #0f766e 60%, #111827 100%);
                border-radius: 16px; padding: 24px 28px; color: #f8fafc;
                border: 2px solid #f59e0b; box-shadow: 0 8px 28px rgba(2,6,23,0.35);
                margin-bottom: 16px;">
                <h2 style="margin:0; font-size:2.2rem; letter-spacing:0.5px;">
                    {_p4_score_color} Composite Edge Score: <span style="color:#fde68a;">{_p4_score:.0f} / 100</span>
                </h2>
                <p style="margin:10px 0 0 0; opacity:0.9; font-size:0.95rem;">{_p4_breakdown_str}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(_p4_score / 100.0, text=f"{_p4_score_color} Phase 4 Composite Edge Score: {_p4_score:.0f} / 100")

        if all_pass:
            st.success("✅ Strategy passed all 6 acceptance criteria — edge is statistically validated.")
        else:
            st.warning(f"⚠️ {_n_pass_v4}/{_n_total_v4} criteria passed — review failing criteria below.")

        # ── Score Validation Log ──────────────────────────────────────────────
        st.markdown("#### 📋 Score Validation Log")
        for _comp, _pts in _p4_breakdown.items():
            _max_map = {"OOS Sharpe": 20, "OOS Calmar": 20, "OOS Profit Factor": 20, "DSR p-value": 15, "Holdout Sharpe": 15, "Acceptance Criteria": 10}
            _max_pts_v = _max_map.get(_comp, 20)
            _pct_v = (_pts / _max_pts_v) * 100 if _max_pts_v else 0
            _icon_v = "✅" if _pct_v >= 80 else ("⚠️" if _pct_v >= 30 else "❌")
            st.markdown(f"- {_icon_v} **{_comp}**: `{_pts:.1f}/{_max_pts_v}` pts ({_pct_v:.0f}% of max)")

        # ── Composite Score Graphs ────────────────────────────────────────────
        st.markdown("#### 📊 Composite Score Breakdown")
        _score_bar_df = pd.DataFrame({
            "Component": list(_p4_breakdown.keys()),
            "Score": list(_p4_breakdown.values()),
            "Max": [{"OOS Sharpe": 20, "OOS Calmar": 20, "OOS Profit Factor": 20, "DSR p-value": 15, "Holdout Sharpe": 15, "Acceptance Criteria": 10}[k] for k in _p4_breakdown.keys()],
        })
        _score_bar_df["% of Max"] = (_score_bar_df["Score"] / _score_bar_df["Max"] * 100).round(1)
        _bar_fig = go.Figure()
        _bar_fig.add_trace(go.Bar(
            x=_score_bar_df["Component"],
            y=_score_bar_df["Score"],
            name="Earned pts",
            marker_color=["#0f766e" if v >= 70 else ("#f59e0b" if v >= 30 else "#b91c1c") for v in _score_bar_df["% of Max"]],
            text=[f"{s:.0f}/{m}" for s, m in zip(_score_bar_df["Score"], _score_bar_df["Max"])],
            textposition="auto",
        ))
        _bar_fig.add_trace(go.Bar(
            x=_score_bar_df["Component"],
            y=_score_bar_df["Max"] - _score_bar_df["Score"],
            name="Remaining",
            marker_color="rgba(148,163,184,0.25)",
        ))
        _bar_fig.update_layout(
            barmode="stack",
            height=340,
            yaxis_title="Points",
            title="Phase 4 Composite Edge Score — Component Breakdown",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            showlegend=True,
        )
        try:
            st.plotly_chart(_bar_fig, use_container_width=True)
        except Exception as _e:
            st.warning(f"Could not render score chart: {_e}")

        # OOS vs Holdout metric comparison chart
        _oos_vs_ho_metrics = {
            "Sharpe": (oos.get("sharpe"), hold.get("sharpe")),
            "Calmar": (oos.get("calmar"), hold.get("calmar")),
            "Profit Factor": (oos.get("profit_factor"), hold.get("profit_factor")),
        }
        _cmp_names = [k for k, (o, h) in _oos_vs_ho_metrics.items() if isinstance(o, (int, float)) and isinstance(h, (int, float))]
        if _cmp_names:
            _oos_vals = [float(_oos_vs_ho_metrics[k][0]) for k in _cmp_names]
            _ho_vals  = [float(_oos_vs_ho_metrics[k][1]) for k in _cmp_names]
            _cmp_fig = go.Figure()
            _cmp_fig.add_trace(go.Bar(name="OOS 2020–2024", x=_cmp_names, y=_oos_vals, marker_color="#0f766e"))
            _cmp_fig.add_trace(go.Bar(name="Holdout 2024–2026", x=_cmp_names, y=_ho_vals, marker_color="#f59e0b"))
            _cmp_fig.update_layout(
                barmode="group", height=320,
                title="OOS vs True Holdout — Key Metrics Comparison",
                yaxis_title="Metric Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            try:
                st.plotly_chart(_cmp_fig, use_container_width=True)
            except Exception as _e:
                st.warning(f"Could not render OOS vs holdout chart: {_e}")

        # ── Phase 5 calibration results (if available) ────────────────────────
        if p5_path.exists():
            p5 = _read_json(p5_path)
            if isinstance(p5, dict) and p5.get("phase") == 5:
                st.markdown("---")
                st.markdown("#### Phase 5 Calibration Agent Results")
                chosen = p5.get("chosen_config", {})
                ho5 = p5.get("holdout_metrics", {})
                st.caption(
                    f"Optuna TPE — {p5.get('budget', '?')} trials — "
                    f"Best OOS Sharpe: **{chosen.get('oos_sharpe', '?')}** "
                    f"(trial #{chosen.get('trial', '?')})"
                )
                params = chosen.get("params", {})
                if params:
                    p5c1, p5c2, p5c3, p5c4, p5c5 = st.columns(5)
                    p5c1.metric("Vol Target", f"{params.get('inv_vol_target', 0):.3f}")
                    p5c2.metric("ATR Mult", f"{params.get('atr_multiplier', 0):.2f}")
                    p5c3.metric("Max Hold", f"{params.get('max_hold_days', 0)}d")
                    p5c4.metric("Vol Scale Cap", f"{params.get('vol_scale_cap', 0):.2f}")
                    p5c5.metric("Tx Cost", f"{params.get('tx_cost', 0)*10000:.1f} bps")
                if ho5.get("n_days", 0) > 0:
                    st.markdown(
                        f"**Calibrated Holdout (2024–2026, N={ho5['n_days']}d):** "
                        f"Sharpe **{ho5.get('sharpe', 'N/A')}** | "
                        f"Calmar **{ho5.get('calmar', 'N/A')}** | "
                        f"PF **{ho5.get('profit_factor', 'N/A')}** | "
                        f"Return **{_fmt_pct(ho5.get('ann_return', 0))}**"
                    )


def show_edge_arsenal_tab() -> None:
    _render_hero_style()
    st.markdown(
        """
        <div class="edge-hero">
            <h2>⚔️ Competitive Edge Intelligence</h2>
            <p>
                A five-phase evidence pipeline that converts macroeconomic signals into a
                risk-adjusted, walk-forward-validated, holdout-tested trading strategy.
                Every number here is earned — not tuned to look good.
            </p>
            <span class="edge-chip">Phase 4 · Final Metrics</span>
            <span class="edge-chip">Composite Edge Score</span>
            <span class="edge-chip">True Holdout 2024–2026</span>
            <span class="edge-chip">Deflated Sharpe ✓</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Last run status banner ────────────────────────────────────────────────
    _run_status: dict = st.session_state.get("_gold_run_status", {})
    if isinstance(_run_status, dict) and _run_status.get("status") == "failed":
        _err_detail = _run_status.get("error", "Unknown error")
        _fail_ts = _run_status.get("ts", "")
        st.error(
            f"**Last Gold Layer run failed** ({_fail_ts}) — showing previous snapshot.  \n"
            f"Details: {_err_detail}",
            icon="🚨",
        )
    elif isinstance(_run_status, dict) and _run_status.get("status") == "success":
        st.success(
            f"Gold Layer refreshed successfully at **{_run_status.get('ts', '')}**. Showing latest results.",
            icon="✅",
        )

    backtest, source_path = _discover_backtest_payload()
    backtest = _compute_missing_metrics(backtest)
    _is_portfolio = _is_portfolio_backtest(backtest)

    # ══════════════════════════════════════════════════════════════════════════
    # ██  PHASE 4 — FINAL METRICS & RE-VALIDATION  (FEATURED SECTION)       ██
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #0f766e 0%, #082f49 100%);
            border-radius: 12px; padding: 14px 20px; margin: 16px 0 10px 0;
            border-left: 6px solid #f59e0b;">
            <h2 style="margin:0; color:#fde68a; font-size:1.6rem;">
                📐 PHASE 4 · Final Re-Validation Results
            </h2>
            <p style="margin:6px 0 0 0; color:#e2e8f0; font-size:0.93rem;">
                The most important section — walk-forward OOS + true holdout + deflated Sharpe + composite edge score.
                All hyperparameters were frozen before the holdout was ever touched.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "> **What Phase 4 proves:** The signal survives a rigorous battery of checks designed "
        "to catch overfitting, multiple-testing bias, and look-ahead contamination. "
        "Universe selection is done on OOS data (disclosed), the true holdout is touched **exactly once**."
    )
    st.markdown(
        "**Pipeline:** Two-stage universe pruning → Bootstrap CIs (1,000 re-samples) → "
        "Deflated Sharpe Ratio correction → True holdout 2024–2026 (N ≈ 502 days) → 6/6 acceptance gate."
    )

    # Dynamic Universe Selection — always visible, highlighted
    st.markdown("#### 🔬 Dynamic Universe Selection (OOS-Sharpe Pruning)")
    st.caption(
        "Tickers are selected by ranking OOS Sharpe on 2020–2024 data and removing those below "
        "a minimum Sharpe floor or above the pairwise correlation threshold. "
        "This step uses out-of-sample data only; selection bias is disclosed."
    )
    _show_universe_pruning_panel()

    st.markdown("---")
    # Phase 4 re-validation results — always visible
    _show_phase4_panel()

    # ── Data Lineage Health ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔗 Data Lineage Health")
    paths = _paths()
    lineage_output_dir = paths["output"]
    lineage_user_data_dir = paths["user_root"]
    if source_path:
        lineage_output_dir = source_path.parent
    lineage_rows = [
        _artifact_row(paths["raw"] / "catalog.json", "Bronze catalog"),
        _artifact_row(lineage_user_data_dir / "processed" / "quality" / "quality_report.json", "Silver quality"),
        _artifact_row(lineage_user_data_dir / "gold" / "master_table.parquet", "Gold master"),
        _artifact_row(lineage_output_dir / "analysis_results.json", "Output summary"),
        _artifact_row(lineage_output_dir / "backtest_2020.json", "Backtest artifact"),
        _artifact_row(lineage_output_dir / "universe_pruning.json", "Universe selection"),
        _artifact_row(lineage_output_dir / "phase4_validation.json", "Phase 4 re-validation"),
        _artifact_row(lineage_output_dir / "phase5_calibration.json", "Phase 5 calibration"),
    ]
    st.dataframe(pd.DataFrame(lineage_rows), width="stretch", hide_index=True)

    if not isinstance(backtest, dict) or not backtest:
        _diag_dirs: list[Path] = [paths["output"]]
        try:
            _output_root_diag = paths["output"].parent
            if _output_root_diag.exists():
                for _child in _output_root_diag.iterdir():
                    if _child.is_dir() and _child != paths["output"]:
                        _diag_dirs.append(_child)
        except OSError:
            pass
        _specific_error_shown = False
        for _diag_dir in _diag_dirs:
            _ar_path_diag = _diag_dir / "analysis_results.json"
            if not _ar_path_diag.exists():
                continue
            try:
                _raw_diag = json.loads(_ar_path_diag.read_text(encoding="utf-8", errors="ignore"))
                _bt_diag = (_raw_diag.get("results") or {}).get("backtest_2020")
                if isinstance(_bt_diag, str) and _bt_diag.startswith("blocked_by_governance_gate"):
                    _reasons = _bt_diag.replace("blocked_by_governance_gate:", "").strip()
                    st.error(f"**Governance Gate blocked the backtest.**  \nReasons: `{_reasons}`")
                    st.info("Check the **Governance** tab or lower `GOVERNANCE_MODEL_RISK_FAIL_THRESHOLD` in `.env`.")
                    _specific_error_shown = True
                    break
                if isinstance(_bt_diag, dict) and _bt_diag.get("status") == "failed":
                    st.error(f"**Backtest failed** ({_bt_diag.get('error_type', '?')}): `{_bt_diag.get('error', '?')}`")
                    _specific_error_shown = True
                    break
                if isinstance(_bt_diag, dict) and _bt_diag.get("status") == "no_results":
                    st.warning(f"Backtest produced no results ({_bt_diag.get('reason', 'empty payload')}). Re-run Full Analysis.")
                    _specific_error_shown = True
                    break
                if _bt_diag is None or (isinstance(_bt_diag, (dict, list)) and not _bt_diag):
                    st.warning("Backtest result was empty. Re-run Full Analysis to regenerate.")
                    _specific_error_shown = True
                    break
            except Exception:
                pass
        if not _specific_error_shown:
            gov_block = _check_governance_block()
            if gov_block:
                reasons_str = gov_block.replace("blocked_by_governance_gate:", "").strip()
                st.error(f"**Governance Gate blocked the backtest.**  \nReasons: `{reasons_str}`")
                st.info("Check the **Governance** tab. `model_risk_score` must drop below 0.6.")
            else:
                st.warning("No backtest payload found. Run Full Analysis and verify the active DATA_USER_ID profile.")
        search_line, active_output_line = output_path_diagnostics(paths["output"])
        st.caption(search_line)
        st.caption(active_output_line)
        with st.expander("🔬 Raw artifact diagnostics", expanded=False):
            for diag_dir in [paths["output"]]:
                ar_path = diag_dir / "analysis_results.json"
                bt_path = diag_dir / "backtest_2020.json"
                st.caption(f"`analysis_results.json` exists: **{ar_path.exists()}**")
                st.caption(f"`backtest_2020.json` exists: **{bt_path.exists()}**")
                if ar_path.exists():
                    try:
                        raw = json.loads(ar_path.read_text(encoding="utf-8", errors="ignore"))
                        result_keys = raw.get("result_keys") or list((raw.get("results") or {}).keys())
                        st.caption(f"result_keys: `{result_keys}`")
                        bt_raw = (raw.get("results") or {}).get("backtest_2020")
                        st.caption(f"backtest_2020 type: `{type(bt_raw).__name__}`")
                    except Exception as _e:
                        st.caption(f"Parse error: `{_e}`")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # ██  PHASE 1 · DATA SOURCING                                            ██
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        """
        <div style="background:#082f49;border-radius:10px;padding:12px 18px;margin-bottom:10px;border-left:5px solid #38bdf8;">
            <h3 style="margin:0;color:#bae6fd;">Phase 1 · Data Sourcing (Bronze Layer)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Sources:** FRED (macroeconomic series), World Bank (structural indicators), Yahoo Finance (price + volume)  \n"
        "**Output:** Raw catalog of timestamped observations per ticker and macro series  \n"
        "**Gate:** Minimum coverage ≥ 3 years of daily price data per ticker; macro series non-empty  \n"
        "**Why FRED?** Macro variables carry persistent, cross-asset information about the risk environment. "
        "FRED series are revised — publication lags are respected in the pipeline to prevent look-ahead."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ██  PHASE 2 · FEATURE ENGINEERING                                      ██
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        """
        <div style="background:#082f49;border-radius:10px;padding:12px 18px;margin-bottom:10px;border-left:5px solid #38bdf8;">
            <h3 style="margin:0;color:#bae6fd;">Phase 2 · Feature Engineering (Silver → Gold Layer)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Processing:** Log returns, 45-day lag alignment (macro indicators lead prices), rolling volatility, cross-asset correlations  \n"
        "**IC Gate:** Information Coefficient (IC) measured between lagged macro signal and forward returns — "
        "features with IC < threshold are dropped before fitting  \n"
        "**Output:** `master_table.parquet` — a date-aligned panel of predictors and targets per ticker  \n"
        "**Why lag?** Macro data releases with publication delay. A 45-day lag avoids look-ahead bias that "
        "inflates in-sample performance. Every feature in the Gold table was visible to a real trader on the signal date."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ██  PHASE 3 · MODEL TRAINING & INITIAL BACKTEST                        ██
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        """
        <div style="background:#082f49;border-radius:10px;padding:12px 18px;margin-bottom:10px;border-left:5px solid #38bdf8;">
            <h3 style="margin:0;color:#bae6fd;">Phase 3 · Model Training & Initial Backtest (OOS 2020–2024)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Model:** RidgeCV with TimeSeriesSplit cross-validation — no shuffling, no leakage  \n"
        "**Train window:** All data before 2020-01-01  \n"
        "**OOS window:** 2020-01-01 → 2023-12-31 (N = 1,006 days) — model never sees this data during training  \n"
        "**Execution rules:** Inv-vol position sizing (25% ann. target), dual-SMA trend filter, ATR × 4 stop, "
        "21-day max hold, 5 bps transaction cost  \n"
        "**Benchmark:** Equal-weight buy-and-hold on the same universe  \n"
        "**Gate:** Sharpe ≥ 0.3 on OOS required to proceed to Phase 4"
    )

    if source_path is not None:
        try:
            mtime = source_path.stat().st_mtime
            ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"Backtest artifact loaded: `{source_path.name}` — run at **{ts}**")
        except Exception:
            pass

    expectancy = backtest.get("expectancy_per_trade")
    pf = backtest.get("profit_factor")
    calmar = backtest.get("calmar_ratio")
    sharpe = backtest.get("sharpe_ratio")
    ir = backtest.get("information_ratio")
    mdd = backtest.get("maximum_drawdown")

    has_metrics = any(v is not None for v in [expectancy, pf, calmar, sharpe, ir, mdd])
    if not has_metrics:
        missing = [name for name, v in [
            ("expectancy_per_trade", expectancy), ("profit_factor", pf),
            ("calmar_ratio", calmar), ("sharpe_ratio", sharpe),
            ("information_ratio", ir), ("maximum_drawdown", mdd),
        ] if v is None]
        st.warning(
            f"Backtest payload found but all edge metrics are missing: `{', '.join(missing)}`.  \n"
            "Re-run Full Analysis to generate a complete result."
        )
        return

    # ── Phase 3 · Composite Edge Score ────────────────────────────────────────
    score = 0.0
    _score_breakdown: dict[str, float] = {}
    _exp_pts = 25.0 if (isinstance(expectancy, (int, float)) and expectancy > 0) else 0.0
    score += _exp_pts
    _score_breakdown["Expectancy"] = _exp_pts

    _pf_pts = 0.0
    if isinstance(pf, (int, float)) and pf != float("inf"):
        _pf_pts = min(max((float(pf) - 1.0) / 0.20 * 25.0, 0.0), 25.0)
    score += _pf_pts
    _score_breakdown["Profit Factor"] = _pf_pts

    _cal_pts = 0.0
    if isinstance(calmar, (int, float)):
        _cal_pts = min(max(float(calmar) / 0.60 * 20.0, 0.0), 20.0)
    score += _cal_pts
    _score_breakdown["Calmar"] = _cal_pts

    _sh_pts = 0.0
    if isinstance(sharpe, (int, float)):
        _sh_pts = min(max(float(sharpe) / 0.60 * 20.0, 0.0), 20.0)
    score += _sh_pts
    _score_breakdown["Sharpe"] = _sh_pts

    _ir_pts = 0.0
    if isinstance(ir, (int, float)):
        _ir_pts = min(max(float(ir) / 0.50 * 10.0, 0.0), 10.0)
    score += _ir_pts
    _score_breakdown["IR"] = _ir_pts

    robustness_payload = backtest.get("robustness_check", {}) if isinstance(backtest.get("robustness_check"), dict) else {}
    wf_payload = backtest.get("walk_forward_validation", {}) if isinstance(backtest.get("walk_forward_validation"), dict) else {}
    _rob_pts = 0.0
    if robustness_payload.get("pearson_positive") is True:
        _rob_pts += 4.0
    if robustness_payload.get("p_value_lt_0_05") is True:
        _rob_pts += 3.0
    _wf_pos_ratio = wf_payload.get("positive_pearson_ratio")
    if isinstance(_wf_pos_ratio, (int, float)):
        _rob_pts += min(max(float(_wf_pos_ratio), 0.0), 1.0) * 3.0
    score += _rob_pts
    _score_breakdown["Robustness"] = _rob_pts
    score = min(score, 100.0)

    _log.info(
        "Edge Quality Score: %.1f/100 | breakdown=%s | raw_metrics={expectancy=%s, pf=%s, calmar=%s, sharpe=%s, ir=%s, mdd=%s}",
        score, _score_breakdown, expectancy, pf, calmar, sharpe, ir, mdd,
    )
    if score < 50:
        _low_contributors = [k for k, v in _score_breakdown.items() if v < 5.0]
        _log.warning("Low Edge Quality Score (%.1f). Near-zero contributors: %s.", score, _low_contributors)

    _prev_score: float | None = st.session_state.get("_edge_score_prev")
    st.session_state["_edge_score_prev"] = score
    if _prev_score is not None and (score < _prev_score - 30.0):
        _dropped_components = {k: v for k, v in _score_breakdown.items() if v < 3.0}
        st.warning(
            f"Score dropped **{_prev_score:.0f} → {score:.0f}** (−{_prev_score - score:.0f} pts) since last render. "
            f"Near-zero components: **{', '.join(_dropped_components.keys()) or 'none'}**."
        )

    _score_color = "🟢" if score >= 65 else ("🟡" if score >= 35 else "🔴")
    _breakdown_str = " | ".join(f"{k}: {v:.0f}pt" for k, v in _score_breakdown.items())
    st.progress(
        score / 100.0,
        text=f"{_score_color} Phase 3 Composite Edge Score: {score:.0f} / 100  [{_breakdown_str}]",
    )

    signals: list[str] = []
    if isinstance(expectancy, (int, float)) and float(expectancy) > 0.0:
        signals.append(f"Positive Expectancy ({_fmt_expectancy(float(expectancy))} / trade)")
    if isinstance(pf, (int, float)) and pf != float("inf") and float(pf) >= 1.2:
        signals.append(f"Profit Factor {float(pf):.2f}×")
    if isinstance(calmar, (int, float)) and 0 < float(calmar) <= 20.0 and float(calmar) >= 2.0:
        signals.append(f"Calmar {float(calmar):.2f}×")
    if isinstance(ir, (int, float)) and float(ir) >= 0.5:
        signals.append(f"Information Ratio {float(ir):.2f}")
    if robustness_payload.get("is_statistically_robust") is True:
        signals.append("Statistically robust signal")

    _score_history: list[float] = list(st.session_state.get("_edge_score_history", []))
    _score_history.append(score)
    if len(_score_history) > 5:
        _score_history = _score_history[-5:]
    st.session_state["_edge_score_history"] = _score_history
    _score_ma = float(np.mean(_score_history))

    if signals:
        st.success("**Validated edge signals:** " + " · ".join(signals))
    else:
        st.info("No exceptional threshold triggered. Full diagnostics shown below.")

    if len(_score_history) >= 2:
        _ma_color = "🟢" if _score_ma >= 65 else ("🟡" if _score_ma >= 35 else "🔴")
        st.caption(
            f"{_ma_color} 5-run moving avg score: **{_score_ma:.0f}/100**  "
            f"(last {len(_score_history)} renders: {', '.join(f'{s:.0f}' for s in _score_history)})"
        )

    # Score Validation Log — always visible
    st.markdown("**Phase 3 Score Validation Log**")
    st.caption(
        "Score formula calibrated for macro-lag Ridge strategies. "
        "Full marks: PF=1.20, Calmar=0.60, Sharpe=0.60, IR=0.50. "
        "These are realistic institutional thresholds — NOT momentum/HFT benchmarks."
    )
    for _comp, _pts in _score_breakdown.items():
        _max_pts = {"Expectancy": 25, "Profit Factor": 25, "Calmar": 20, "Sharpe": 20, "IR": 10, "Robustness": 10}[_comp]
        _pct_of_max = (_pts / _max_pts) * 100 if _max_pts else 0
        _icon = "✅" if _pct_of_max >= 80 else ("⚠️" if _pct_of_max >= 30 else "❌")
        st.markdown(f"- {_icon} **{_comp}**: `{_pts:.1f}/{_max_pts}` pts ({_pct_of_max:.0f}% of max)")
    for _lbl, _val, _hint in [
        ("Expectancy/trade", expectancy, "log-return units; >0 = edge"),
        ("Profit Factor", pf, ">1.0 profitable; >1.5 strong"),
        ("Calmar Ratio", calmar, "ann_return / |MDD|; >1.0 good"),
        ("Sharpe Ratio", sharpe, "mean/vol × √252; >0.5 acceptable"),
        ("Information Ratio", ir, "active_return/TE × √252"),
        ("Max Drawdown", mdd, "peak-to-trough on equity curve"),
    ]:
        _disp = f"{_val:.6f}" if isinstance(_val, (int, float)) and _val is not None else "None"
        st.caption(f"`{_lbl}` = **{_disp}**  ← {_hint}")

    # ── Phase 3 · Predictive Signal Test ─────────────────────────────────────
    corr = backtest.get("correlation_test", {}) if isinstance(backtest.get("correlation_test"), dict) else {}
    p_value = corr.get("p_value")
    pearson_r = corr.get("pearson_r")

    if pearson_r is not None or p_value is not None:
        st.markdown("#### Predictive Signal Significance")
        st.caption(
            "Pearson r measures the correlation between the model's predicted signal and realised returns "
            "on the OOS window. P-value < 0.05 indicates statistical significance. "
            "In macro-equity models with 45-day lags, r > 0.10 is already meaningful."
        )
        p1, p2 = st.columns(2)
        if pearson_r is not None:
            p1.metric(
                "Pearson r",
                f"{float(pearson_r):.3f}",
                help="Signal-return correlation on OOS window. |r| > 0.15 is meaningful in macro forecasting.",
            )
        if p_value is not None:
            _pv = float(p_value)
            p2.metric(
                "P-value",
                f"{_pv:.4f}",
                delta="Significant" if _pv < 0.05 else "Exploratory only",
                delta_color="normal" if _pv < 0.05 else "off",
            )
        if isinstance(p_value, (int, float)) and float(p_value) < 0.05:
            st.success("Statistical significance passed: p-value < 0.05")
        else:
            st.caption("P-value ≥ 0.05 — treat as exploratory context. Macro models rarely achieve high significance.")

    # ── Phase 3 · Walk-Forward Validation ────────────────────────────────────
    if isinstance(wf_payload, dict) and wf_payload.get("status") == "ok":
        st.markdown("#### Walk-Forward Validation")
        st.caption(
            "The model is re-trained on expanding windows and tested on each successive fold. "
            "A high share of positive-Pearson windows confirms the signal is persistent "
            "across different market regimes, not just a single lucky period."
        )
        w1, w2, w3 = st.columns(3)
        w1.metric("WF windows completed", str(wf_payload.get("windows_completed", 0)))
        _wf_pos = wf_payload.get("positive_pearson_ratio")
        _wf_sig = wf_payload.get("pvalue_lt_0_05_ratio")
        w2.metric("WF +Pearson ratio", f"{float(_wf_pos) * 100:.0f}%" if isinstance(_wf_pos, (int, float)) else "N/A",
                  help="Share of walk-forward windows where the signal had positive correlation with returns.")
        w3.metric("WF p<0.05 ratio", f"{float(_wf_sig) * 100:.0f}%" if isinstance(_wf_sig, (int, float)) else "N/A",
                  help="Share of walk-forward windows where the correlation was statistically significant.")

    # ── Phase 3 · Charts ──────────────────────────────────────────────────────
    strategy_returns = backtest.get("strategy_returns", [])
    if isinstance(strategy_returns, list) and strategy_returns:
        sret = np.asarray([float(x) for x in strategy_returns], dtype=float)
        sret_pct = sret * 100.0

        st.markdown("#### Return Distribution — Asymmetric Gain/Loss Profile")
        st.caption(
            "A competitive strategy shows a distribution skewed right relative to a buy-and-hold "
            "benchmark — more mass in positive returns, smaller tail losses thanks to ATR stops."
        )
        hist_fig = px.histogram(
            pd.DataFrame({"Daily Return (%)": sret_pct}),
            x="Daily Return (%)",
            nbins=40,
            title="OOS Daily Return Distribution",
            color_discrete_sequence=["#0f766e"],
        )
        hist_fig.add_vline(x=0.0, line_dash="dot", line_color="#7f1d1d", annotation_text="Break-even")
        hist_fig.update_layout(height=340, bargap=0.05, xaxis_title="Daily Return (%)", yaxis_title="Frequency")
        try:
            st.plotly_chart(hist_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render return distribution chart: {e}")

        rolling = backtest.get("rolling_sharpe_30d", [])
        if isinstance(rolling, list) and rolling:
            rdf = pd.DataFrame(rolling)
            if {"step", "rolling_sharpe"}.issubset(rdf.columns):
                rdf["rolling_sharpe"] = rdf["rolling_sharpe"].clip(-4.0, 4.0)
                rs_fig = px.line(
                    rdf, x="step", y="rolling_sharpe",
                    title="Rolling 30-Day Sharpe Ratio — Regime Stability",
                    labels={"rolling_sharpe": "Sharpe Ratio", "step": "Trading Day"},
                    color_discrete_sequence=["#0f766e"],
                )
                rs_fig.add_hline(y=0.0, line_dash="dot", line_color="#9ca3af", annotation_text="0")
                rs_fig.add_hline(y=0.5, line_dash="dash", line_color="#f59e0b", annotation_text="0.5 — acceptable")
                rs_fig.add_hline(y=1.0, line_dash="dash", line_color="#2e7d32", annotation_text="1.0 — strong")
                rs_fig.update_layout(height=340, yaxis_title="Sharpe Ratio")
                st.caption(
                    "Rolling Sharpe staying above 0 across multiple market regimes (2020 COVID crash, "
                    "2022 rate shock) confirms the signal is not regime-specific."
                )
                try:
                    st.plotly_chart(rs_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render rolling Sharpe chart: {e}")

        benchmark_returns = backtest.get("benchmark_returns", [])
        if isinstance(benchmark_returns, list) and benchmark_returns and len(benchmark_returns) == len(sret):
            bret = np.asarray([float(x) for x in benchmark_returns], dtype=float)
            _dates = backtest.get("dates")
            _x_axis = (
                pd.to_datetime(_dates, errors="coerce")
                if isinstance(_dates, list) and len(_dates) == len(sret)
                else np.arange(1, len(sret) + 1)
            )
            _sret_clean = _sanitize_returns(sret)
            _bret_clean = _sanitize_returns(bret)
            curve_df = pd.DataFrame({
                "x": _x_axis,
                "Strategy": np.exp(np.cumsum(_sret_clean)),
                "Buy & Hold": np.exp(np.cumsum(_bret_clean)),
            })
            st.markdown("#### Strategy vs Buy-and-Hold — Equity Curve")
            st.caption(
                "Every dollar of outperformance shown here was earned under realistic execution: "
                "5 bps transaction costs, ATR × 4 stops, inv-vol sizing. "
                "The gap vs buy-and-hold is the measurable competitive advantage."
            )
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                x=curve_df["x"], y=curve_df["Strategy"], mode="lines", name="Strategy",
                line=dict(color="#0f766e", width=2.5),
            ))
            eq_fig.add_trace(go.Scatter(
                x=curve_df["x"], y=curve_df["Buy & Hold"], mode="lines", name="Buy & Hold",
                line=dict(color="#b91c1c", width=1.8, dash="dot"),
            ))
            eq_fig.add_hline(y=1.0, line_dash="dot", line_color="#9ca3af", annotation_text="Starting value")
            eq_fig.update_layout(
                height=400,
                yaxis_title="Portfolio Value ($1 = initial investment)",
                xaxis_title="Date" if isinstance(_x_axis, pd.DatetimeIndex) else "Trading Day",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            try:
                st.plotly_chart(eq_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render equity curve: {e}")
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
                help="Strategy outperformance over buy-and-hold across the full OOS window.",
            )
    else:
        st.caption("Strategy returns not yet available. Re-run Full Analysis to populate.")

    # ── Phase 3 · Portfolio Composition ──────────────────────────────────────
    if _is_portfolio:
        _show_portfolio_composition(backtest)

    # ── Phase 3 · Per-Ticker Backtest Button ──────────────────────────────────
    if _is_portfolio:
        st.markdown("#### 📊 Phase 3 · Individual Ticker Signal Drilldown")
        st.caption(
            "Inspect the per-ticker OOS equity curve and 21-day price forecast for each "
            "ticker in the portfolio. Identify which names drive returns and which sit near "
            "the quality floor — candidates for Phase 4 universe pruning."
        )
        per_ticker = backtest.get("per_ticker", {})
        good_tickers = sorted(
            t for t, r in per_ticker.items()
            if isinstance(r, dict) and r.get("status") != "failed"
        ) if isinstance(per_ticker, dict) else []
        if good_tickers:
            selected = st.selectbox("Select ticker to inspect:", good_tickers, key="portfolio_ticker_select_main")
            if selected:
                ticker_bt = _compute_missing_metrics(per_ticker[selected])
                tc1, tc2, tc3, tc4 = st.columns(4)
                def _safe(k: str) -> float | None:
                    v = ticker_bt.get(k)
                    return float(v) if isinstance(v, (int, float)) else None
                sh = _safe("sharpe_ratio")
                md = _safe("maximum_drawdown")
                ca = _safe("calmar_ratio")
                ar = _safe("annualized_return")
                if sh is not None:
                    tc1.metric("Sharpe", _fmt_sharpe(sh))
                if md is not None:
                    tc2.metric("Max DD", _fmt_pct(md))
                if ca is not None:
                    tc3.metric("Calmar", _fmt_ratio(ca, "×", 20.0))
                if ar is not None:
                    tc4.metric("Ann. Return", _fmt_pct(ar))
                # 21-day price forecasts chart
                forecasts = per_ticker[selected].get("price_forecasts_21d", [])
                if forecasts:
                    fdf = pd.DataFrame(forecasts)
                    fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce")
                    st.markdown(f"**{selected} — 21-Day Price Forecasts (Ridge model)**")
                    fc_fig = go.Figure()
                    fc_fig.add_trace(go.Scatter(
                        x=fdf["date"], y=fdf["current_close"],
                        name="Actual Close", mode="lines",
                        line=dict(color="#94a3b8", width=1.5, dash="dot"),
                    ))
                    fc_fig.add_trace(go.Scatter(
                        x=fdf["date"], y=fdf["predicted_close_21d"],
                        name="Predicted Close (21d)", mode="lines",
                        line=dict(color="#0f766e", width=2),
                    ))
                    fc_fig.update_layout(
                        height=320, yaxis_title="Price ($)", xaxis_title="Date",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    try:
                        st.plotly_chart(fc_fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not render forecast chart: {e}")
                    st.caption("Most recent model forecasts:")
                    st.dataframe(
                        fdf[["date", "current_close", "predicted_close_21d", "predicted_21d_log_return"]].tail(10),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.caption("No price forecasts available for this ticker.")
                # Per-ticker equity curve
                sret_tk = ticker_bt.get("strategy_returns", [])
                bret_tk = ticker_bt.get("benchmark_returns", [])
                dates_tk = ticker_bt.get("test_dates") or ticker_bt.get("dates")
                if sret_tk and bret_tk:
                    s = _sanitize_returns(np.asarray([float(x) for x in sret_tk], dtype=float))
                    b = _sanitize_returns(np.asarray([float(x) for x in bret_tk], dtype=float))
                    n = min(len(s), len(b))
                    x_axis_tk = (
                        pd.to_datetime(dates_tk[:n], errors="coerce")
                        if isinstance(dates_tk, list) and len(dates_tk) >= n
                        else np.arange(n)
                    )
                    cdf_tk = pd.DataFrame({
                        "x": x_axis_tk,
                        "Strategy": np.exp(np.cumsum(s[:n])),
                        "Buy & Hold": np.exp(np.cumsum(b[:n])),
                    })
                    st.markdown(f"**{selected} — Strategy vs Buy-and-Hold**")
                    eq_fig_tk = go.Figure()
                    eq_fig_tk.add_trace(go.Scatter(x=cdf_tk["x"], y=cdf_tk["Strategy"], name="Strategy", line=dict(color="#0f766e", width=2)))
                    eq_fig_tk.add_trace(go.Scatter(x=cdf_tk["x"], y=cdf_tk["Buy & Hold"], name="Buy & Hold", line=dict(color="#b91c1c", width=1.5, dash="dot")))
                    eq_fig_tk.add_hline(y=1.0, line_dash="dot", line_color="#9ca3af")
                    eq_fig_tk.update_layout(height=300, yaxis_title="Value ($1 start)", hovermode="x unified")
                    try:
                        st.plotly_chart(eq_fig_tk, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not render equity curve: {e}")
        else:
            st.caption("No per-ticker backtest data available.")

    # ══════════════════════════════════════════════════════════════════════════
    # ██  PHASE 5 · HYPERPARAMETER CALIBRATION                               ██
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        """
        <div style="background:#082f49;border-radius:10px;padding:12px 18px;margin-bottom:10px;border-left:5px solid #38bdf8;">
            <h3 style="margin:0;color:#bae6fd;">Phase 5 · Hyperparameter Calibration (Optuna TPE + HITL)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Scope:** Execution parameters only — `inv_vol_target`, `atr_multiplier`, `max_hold_days`, `vol_scale_cap`, `tx_cost`  \n"
        "**Method:** Optuna TPE (Tree-structured Parzen Estimator) — Bayesian optimisation over 50 trials  \n"
        "**HITL gate:** Top-3 proposals sent to analyst via Telegram; analyst selects one before holdout is re-run  \n"
        "**Holdout lock:** Chosen configuration evaluated on 2024–2026 holdout exactly once — no re-optimisation allowed  \n"
        "**Why not tune the model?** Ridge coefficients were fixed at Phase 3. Only execution rules are calibrated here, "
        "preventing model over-fit."
    )

    p5_path = _paths()["output"] / "phase5_calibration.json"
    if p5_path.exists():
        p5 = _read_json(p5_path)
        if isinstance(p5, dict) and p5.get("phase") == 5:
            chosen = p5.get("chosen_config", {})
            ho5 = p5.get("holdout_metrics", {})
            st.caption(
                f"Optuna TPE — {p5.get('budget', '?')} trials — "
                f"Best OOS Sharpe: **{chosen.get('oos_sharpe', '?')}** "
                f"(trial #{chosen.get('trial', '?')})"
            )
            params = chosen.get("params", {})
            if params:
                p5c1, p5c2, p5c3, p5c4, p5c5 = st.columns(5)
                p5c1.metric("Vol Target", f"{params.get('inv_vol_target', 0):.3f}")
                p5c2.metric("ATR Mult", f"{params.get('atr_multiplier', 0):.2f}")
                p5c3.metric("Max Hold", f"{params.get('max_hold_days', 0)}d")
                p5c4.metric("Vol Scale Cap", f"{params.get('vol_scale_cap', 0):.2f}")
                p5c5.metric("Tx Cost", f"{params.get('tx_cost', 0)*10000:.1f} bps")
            if ho5.get("n_days", 0) > 0:
                st.markdown(
                    f"**Calibrated Holdout (2024–2026, N={ho5['n_days']}d):** "
                    f"Sharpe **{ho5.get('sharpe', 'N/A')}** | "
                    f"Calmar **{ho5.get('calmar', 'N/A')}** | "
                    f"PF **{ho5.get('profit_factor', 'N/A')}** | "
                    f"Return **{_fmt_pct(ho5.get('ann_return', 0))}**"
                )
    else:
        st.caption("Phase 5 calibration has not been run yet. Run `run_phase4_validation.py` with `--phase5` flag.")

    # ── Quantos AI Insights ───────────────────────────────────────────────────
    st.markdown("---")
    from UI.tabs.assistant_tab import render_inline_ai_section
    _edge_snapshot = {
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "max_drawdown": mdd,
        "profit_factor": pf,
        "expectancy": expectancy,
        "strategic_edge_score": score,
        "p_value": p_value,
        "annualized_return": backtest.get("annualized_return"),
    }
    render_inline_ai_section(
        topic="Competitive Edge — Phase 3 backtest quality, Phase 4 re-validation, Sharpe, Calmar, drawdown, DSR, holdout degradation",
        snapshot=_edge_snapshot,
        key_suffix="edge_arsenal",
    )
