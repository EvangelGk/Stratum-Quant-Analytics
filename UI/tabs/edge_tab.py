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

    # ── Last run status banner ────────────────────────────────────────────────
    # Populated by run_gold_analyses_only() after every rerun attempt.
    # Shows a persistent warning when the last run failed so the user knows
    # they are looking at a previous (possibly stale) snapshot.
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

    st.markdown("### 🔗 Data Lineage Health")

    # Determine active profile paths for lineage. By default, use paths from
    # constants, but if a more recent artifact was found, derive paths from its
    # location to ensure the health check is for the correct run.
    paths = _paths()
    lineage_output_dir = paths["output"]
    lineage_user_data_dir = paths["user_root"]

    if source_path:
        active_output_dir = source_path.parent
        lineage_output_dir = active_output_dir

    lineage_rows = [
        _artifact_row(paths["raw"] / "catalog.json", "Bronze catalog"),
        _artifact_row(lineage_user_data_dir / "processed" / "quality" / "quality_report.json", "Silver quality"),
        _artifact_row(lineage_user_data_dir / "gold" / "master_table.parquet", "Gold master"),
        _artifact_row(lineage_output_dir / "analysis_results.json", "Output summary"),
        _artifact_row(lineage_output_dir / "backtest_2020.json", "Backtest artifact"),
    ]
    st.dataframe(pd.DataFrame(lineage_rows), width="stretch", hide_index=True)

    if not isinstance(backtest, dict) or not backtest:
        available_profiles: list[str] = []
        output_root = paths["output"].parent
        if output_root.exists():
            for child in output_root.iterdir():
                if child.is_dir():
                    available_profiles.append(child.name)

        # Check whether the backtest returned a structured error dict (exception was captured).
        # Scan all profiles so we report errors from any previous run.
        _diag_dirs: list[Path] = [paths["output"]]
        try:
            _output_root_diag = paths["output"].parent
            if _output_root_diag.exists():
                for _child in _output_root_diag.iterdir():
                    if _child.is_dir() and _child != paths["output"]:
                        _diag_dirs.append(_child)
        except OSError:
            pass
        # ── Determine the exact reason the backtest is unavailable ─────────
        # Priority: active profile first, then sibling profiles.
        # Each check sets _specific_error_shown=True and breaks out of the
        # detection loop so exactly one message is displayed.
        _specific_error_shown = False
        for _diag_dir in _diag_dirs:
            _ar_path_diag = _diag_dir / "analysis_results.json"
            if not _ar_path_diag.exists():
                continue
            try:
                _raw_diag = json.loads(_ar_path_diag.read_text(encoding="utf-8", errors="ignore"))
                _bt_diag = (_raw_diag.get("results") or {}).get("backtest_2020")

                if isinstance(_bt_diag, str) and _bt_diag.startswith("blocked_by_governance_gate"):
                    # Governance block string stored directly in results
                    _reasons = _bt_diag.replace("blocked_by_governance_gate:", "").strip()
                    st.error(
                        "**Governance Gate blocked the backtest.**  \n"
                        "The pipeline ran but the model risk score exceeded the fail threshold (0.6).  \n"
                        f"**Reasons:** `{_reasons}`"
                    )
                    st.info(
                        "Check the **Governance** tab. Lower `GOVERNANCE_MODEL_RISK_FAIL_THRESHOLD` in `.env` "
                        "(e.g. `0.85`) or re-run after adjusting tickers/macro factors."
                    )
                    _specific_error_shown = True
                    break

                if isinstance(_bt_diag, dict) and _bt_diag.get("status") == "failed":
                    st.error(
                        f"**Backtest failed with exception** ({_bt_diag.get('error_type', '?')}):  \n"
                        f"`{_bt_diag.get('error', 'unknown error')}`"
                    )
                    _specific_error_shown = True
                    break

                if isinstance(_bt_diag, dict) and _bt_diag.get("status") == "no_results":
                    st.warning(
                        f"Backtest ran but produced no results ({_bt_diag.get('reason', 'empty payload')}). "
                        "Re-run Full Analysis to regenerate."
                    )
                    _specific_error_shown = True
                    break

                if _bt_diag is None or (isinstance(_bt_diag, (dict, list)) and not _bt_diag):
                    st.warning(
                        "Backtest result was empty or missing in this run's output. "
                        "Re-run Full Analysis to regenerate the backtest."
                    )
                    _specific_error_shown = True
                    break

            except Exception:
                pass

        if not _specific_error_shown:
            # Cross-profile governance check (reads backtest string from analysis_results)
            gov_block = _check_governance_block()
            if gov_block:
                reasons_str = gov_block.replace("blocked_by_governance_gate:", "").strip()
                st.error(
                    "**Governance Gate blocked the backtest.**  \n"
                    "The pipeline ran successfully but the model risk score exceeded the fail threshold (0.6). "
                    "No backtest metrics are displayed until the model passes governance checks.  \n"
                    f"**Reasons:** `{reasons_str}`"
                )
                st.info(
                    "To resolve: check the **Governance** tab for the full report. "
                    "The model_risk_score must drop below 0.6. "
                    "You can lower thresholds via `.env` (e.g. `GOVERNANCE_MODEL_RISK_FAIL_THRESHOLD=0.85`) "
                    "or re-run Full Analysis after adjusting tickers/macro factors."
                )
            else:
                st.warning("No backtest payload found in the active output profile. Run Full Analysis and verify the active DATA_USER_ID profile.")

        search_line, active_output_line = output_path_diagnostics(paths["output"])
        st.caption(search_line)
        st.caption(active_output_line)
        env_uid = (os.getenv("DATA_USER_ID") or "").strip()
        if env_uid:
            st.caption(f"🔍 DATA_USER_ID (env): `{env_uid}`")
            active_uid = str(paths.get("user_key", "")).strip()
            if active_uid and env_uid != active_uid:
                st.error(
                    "Profile mismatch detected: DATA_USER_ID env does not match active UI profile. "
                    "Set both to the same value and rerun Full Analysis."
                )
        if available_profiles:
            st.caption("Detected output profiles: " + ", ".join(sorted(available_profiles)))

        # ── Deep diagnostics: show raw artifact contents so we can see exactly
        # what the pipeline wrote (governance block, empty, wrong structure, etc.)
        with st.expander("🔬 Raw artifact diagnostics (click to debug)", expanded=False):
            # Show diagnostics for each detected profile, not just the active one.
            diag_profiles: list[Path] = [paths["output"]]
            try:
                output_root_diag = paths["output"].parent
                if output_root_diag.exists():
                    for child in sorted(output_root_diag.iterdir()):
                        if child.is_dir() and child != paths["output"]:
                            diag_profiles.append(child)
            except OSError:
                pass

            for diag_dir in diag_profiles:
                label = f"Profile: `{diag_dir.name}`" + (" *(active)*" if diag_dir == paths["output"] else "")
                st.markdown(f"**{label}**")
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
                        st.caption(f"backtest_2020 value type: `{type(bt_raw).__name__}`")
                        if isinstance(bt_raw, str):
                            st.caption(f"backtest_2020 value: `{bt_raw[:200]}`")
                        elif isinstance(bt_raw, dict):
                            st.caption(f"backtest_2020 keys: `{list(bt_raw.keys())[:10]}`")
                            if bt_raw.get("status") == "failed":
                                st.error(f"**Backtest error** ({bt_raw.get('error_type', '?')}): {bt_raw.get('error', '?')}")
                        generated_at = raw.get("generated_at")
                        st.caption(f"generated_at: `{generated_at}`")
                    except Exception as _e:
                        st.caption(f"Could not parse analysis_results.json: `{_e}`")
                if bt_path.exists():
                    try:
                        raw2 = json.loads(bt_path.read_text(encoding="utf-8", errors="ignore"))
                        inner = raw2.get("value")
                        st.caption(f"backtest_2020.json → value type: `{type(inner).__name__}`")
                        if isinstance(inner, str):
                            st.caption(f"value: `{inner[:200]}`")
                        elif isinstance(inner, dict):
                            st.caption(f"value keys: `{list(inner.keys())[:10]}`")
                    except Exception as _e:
                        st.caption(f"Could not parse backtest_2020.json: `{_e}`")
        return

    if source_path is not None:
        st.caption(f"Loaded backtest artifact: {source_path}")
        try:
            mtime = source_path.stat().st_mtime
            ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"📈 Metrics from analysis run at: **{ts}**")
        except Exception:
            pass

    if _is_portfolio:
        _n_ok = len(backtest.get("tickers_succeeded", []))
        _n_fail = len(backtest.get("failed_tickers", []))
        _wt_label = ", ".join(
            f"{t} {w*100:.1f}%"
            for t, w in sorted(
                (backtest.get("weights") or {}).items(),
                key=lambda x: x[1], reverse=True,
            )[:5]
        )
        _suffix = " …" if _n_ok > 5 else ""
        st.info(
            f"**Portfolio mode** — {_n_ok} tickers backtested"
            + (f", {_n_fail} failed" if _n_fail else "")
            + f".  Top weights: {_wt_label}{_suffix}",
            icon="📊",
        )

    expectancy = backtest.get("expectancy_per_trade")
    pf = backtest.get("profit_factor")
    calmar = backtest.get("calmar_ratio")
    sharpe = backtest.get("sharpe_ratio")
    ir = backtest.get("information_ratio")
    mdd = backtest.get("maximum_drawdown")

    # Check if metrics exist; if not, show specific missing-field alert
    has_metrics = any(v is not None for v in [expectancy, pf, calmar, sharpe, ir, mdd])
    if not has_metrics:
        missing = [name for name, v in [
            ("expectancy_per_trade", expectancy), ("profit_factor", pf),
            ("calmar_ratio", calmar), ("sharpe_ratio", sharpe),
            ("information_ratio", ir), ("maximum_drawdown", mdd),
        ] if v is None]
        st.warning(
            f"Backtest payload found but all edge metrics are missing: `{', '.join(missing)}`.  \n"
            "This usually means the backtest artifact is a failed-status object.  \n"
            "Re-run Full Analysis to generate a complete backtest result."
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
        _pf_delta = "vs 1.0 breakeven"
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
            delta="vs 1.0 target",
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

    # ── Portfolio composition + 21-day price forecasts ────────────────────────
    if _is_portfolio:
        _show_portfolio_composition(backtest)

    # ── Strategic Edge Quality Score ──────────────────────────────────────────
    # Calibrated for macro-factor regression strategies (FRED indicators, 45-day lag).
    # These thresholds reflect realistic institutional-grade performance for this
    # strategy type — NOT momentum/HFT benchmarks.
    #   Expectancy > 0  → 25 pts  (binary: edge sign is positive)
    #   Profit Factor   → 0-25 pts (PF=1.2 → full 25 pts)
    #   Calmar Ratio    → 0-20 pts (Calmar=0.6 → full 20 pts)
    #   Sharpe Ratio    → 0-20 pts (Sharpe=0.6 → full 20 pts)
    #   Info Ratio      → 0-10 pts (IR=0.5 → full 10 pts)
    score = 0.0
    _score_breakdown: dict[str, float] = {}
    _exp_pts = 25.0 if (isinstance(expectancy, (int, float)) and expectancy > 0) else 0.0
    score += _exp_pts
    _score_breakdown["Expectancy"] = _exp_pts

    _pf_pts = 0.0
    if isinstance(pf, (int, float)) and pf != float("inf"):
        # Full 25 pts at PF=1.20 — realistic "excellent" for a macro Ridge strategy
        _pf_pts = min(max((float(pf) - 1.0) / 0.20 * 25.0, 0.0), 25.0)
    score += _pf_pts
    _score_breakdown["Profit Factor"] = _pf_pts

    _cal_pts = 0.0
    if isinstance(calmar, (int, float)):
        # Full 20 pts at Calmar=0.60 — macro strategies rarely exceed 1.0
        _cal_pts = min(max(float(calmar) / 0.60 * 20.0, 0.0), 20.0)
    score += _cal_pts
    _score_breakdown["Calmar"] = _cal_pts

    _sh_pts = 0.0
    if isinstance(sharpe, (int, float)):
        # Full 20 pts at Sharpe=0.60 — macro-lag strategies with Sharpe>0.5 are strong
        _sh_pts = min(max(float(sharpe) / 0.60 * 20.0, 0.0), 20.0)
    score += _sh_pts
    _score_breakdown["Sharpe"] = _sh_pts

    _ir_pts = 0.0
    if isinstance(ir, (int, float)):
        # Full 10 pts at IR=0.50 — active return above tracking error
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

    # ── Validation log: trace what drove the score ────────────────────────────
    # Emit a structured log entry so developers can diagnose unexpected drops
    # without having to re-run the pipeline.
    _log.info(
        "Edge Quality Score: %.1f/100 | breakdown=%s | raw_metrics={expectancy=%s, pf=%s, calmar=%s, sharpe=%s, ir=%s, mdd=%s}",
        score,
        _score_breakdown,
        expectancy, pf, calmar, sharpe, ir, mdd,
    )
    if score < 50:
        _low_contributors = [k for k, v in _score_breakdown.items() if v < 5.0]
        _log.warning(
            "Low Edge Quality Score (%.1f). Zero/near-zero contributors: %s. "
            "Check for regime filter suppression, negative Calmar (ann_return<0), "
            "or low Sharpe caused by high volatility in the 2020-2022 holdout window.",
            score, _low_contributors,
        )

    # ── Sanity check: alert if score dropped >30pts vs last cached run ────────
    _prev_score: float | None = st.session_state.get("_edge_score_prev")
    st.session_state["_edge_score_prev"] = score
    if _prev_score is not None and (score < _prev_score - 30.0):
        _dropped_components = {k: v for k, v in _score_breakdown.items() if v < 3.0}
        st.warning(
            f"Score dropped **{_prev_score:.0f} → {score:.0f}** (−{_prev_score - score:.0f} pts) since last render. "
            f"Near-zero components: **{', '.join(_dropped_components.keys()) or 'none'}**. "
            f"Raw values → Expectancy: `{expectancy}`, PF: `{pf}`, Calmar: `{calmar}`, Sharpe: `{sharpe}`, IR: `{ir}`, MDD: `{mdd}`"
        )

    _score_color = "🟢" if score >= 65 else ("🟡" if score >= 35 else "🔴")
    _breakdown_str = " | ".join(f"{k}: {v:.0f}pt" for k, v in _score_breakdown.items())
    st.progress(
        score / 100.0,
        text=f"{_score_color} Strategic Edge Quality Score: {score:.0f} / 100  [{_breakdown_str}]",
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
    if robustness_payload.get("is_statistically_robust") is True:
        signals.append("Statistically robust signal")

    # ── Moving average of score across last 5 renders ────────────────────────
    # Smooths out single-run shocks so stakeholders see the trend, not noise.
    _score_history: list[float] = list(st.session_state.get("_edge_score_history", []))
    _score_history.append(score)
    if len(_score_history) > 5:
        _score_history = _score_history[-5:]
    st.session_state["_edge_score_history"] = _score_history
    _score_ma = float(np.mean(_score_history))

    if signals:
        st.success("**Validated edge signals:** " + " · ".join(signals))
    else:
        st.info("No exceptional threshold triggered in this run. Full diagnostics shown transparently below.")

    if len(_score_history) >= 2:
        _ma_color = "🟢" if _score_ma >= 65 else ("🟡" if _score_ma >= 35 else "🔴")
        st.caption(
            f"{_ma_color} 5-run moving avg score: **{_score_ma:.0f}/100**  "
            f"(last {len(_score_history)} renders: {', '.join(f'{s:.0f}' for s in _score_history)})"
        )

    # ── Validation Log expander ───────────────────────────────────────────────
    # Shows exactly which component drove the score, enabling root-cause analysis
    # without re-running the full pipeline.
    with st.expander("🔍 Score Validation Log", expanded=(score < 50)):
        st.markdown("**Component breakdown this render:**")
        for _comp, _pts in _score_breakdown.items():
            _max_pts = {"Expectancy": 25, "Profit Factor": 25, "Calmar": 20, "Sharpe": 20, "IR": 10, "Robustness": 10}[_comp]
            _pct_of_max = (_pts / _max_pts) * 100 if _max_pts else 0
            _icon = "✅" if _pct_of_max >= 80 else ("⚠️" if _pct_of_max >= 30 else "❌")
            st.markdown(f"- {_icon} **{_comp}**: `{_pts:.1f}/{_max_pts}` pts ({_pct_of_max:.0f}% of max)")
        st.markdown("**Raw metric values feeding the formula:**")
        _raw_rows = [
            ("Expectancy/trade", expectancy, "log-return units; >0 = edge"),
            ("Profit Factor", pf, ">1.0 profitable; >1.5 strong"),
            ("Calmar Ratio", calmar, "ann_return / |MDD|; >1.0 good"),
            ("Sharpe Ratio", sharpe, "mean/vol × √252; >0.5 acceptable"),
            ("Information Ratio", ir, "active_return/TE × √252"),
            ("Max Drawdown", mdd, "peak-to-trough on equity curve"),
        ]
        for _lbl, _val, _hint in _raw_rows:
            _disp = f"{_val:.6f}" if isinstance(_val, (int, float)) and _val is not None else "None"
            st.caption(f"`{_lbl}` = **{_disp}**  ← {_hint}")
        if score < 70:
            st.warning(
                "Score below 70 — formula is calibrated for macro-lag Ridge strategies: "
                "full score at PF=1.2, Calmar=0.6, Sharpe=0.6. "
                "Likely causes: (1) Calmar low → annualized return below 0.6×|MDD|; "
                "(2) Sharpe low → high daily volatility relative to mean return; "
                "(3) Negative IR → strategy lagging its own benchmark. "
                "Re-run Full Analysis to refresh with latest parameters."
            )

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
    if isinstance(wf_payload, dict) and wf_payload.get("status") == "ok":
        st.markdown("#### Walk-Forward Validation")
        w1, w2, w3 = st.columns(3)
        w1.metric("WF windows", str(wf_payload.get("windows_completed", 0)))
        _wf_pos = wf_payload.get("positive_pearson_ratio")
        _wf_sig = wf_payload.get("pvalue_lt_0_05_ratio")
        w2.metric("WF +Pearson ratio", f"{float(_wf_pos) * 100:.0f}%" if isinstance(_wf_pos, (int, float)) else "N/A")
        w3.metric("WF p<0.05 ratio", f"{float(_wf_sig) * 100:.0f}%" if isinstance(_wf_sig, (int, float)) else "N/A")

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
        try:
            st.plotly_chart(hist_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render return distribution chart. Error: {e}")

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
                try:
                    st.plotly_chart(rs_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render rolling Sharpe ratio chart. Error: {e}")

        benchmark_returns = backtest.get("benchmark_returns", [])
        if isinstance(benchmark_returns, list) and benchmark_returns and len(benchmark_returns) == len(sret):
            bret = np.asarray([float(x) for x in benchmark_returns], dtype=float)
            _dates = backtest.get("dates")
            if isinstance(_dates, list) and len(_dates) == len(sret):
                _x_axis = pd.to_datetime(_dates, errors="coerce")
            else:
                _x_axis = np.arange(1, len(sret) + 1)

            # exp(cumsum) is the correct compounding for log returns.
            # cumprod(1+r) explodes to 10^44 when the series contains outliers.
            _sret_clean = _sanitize_returns(sret)
            _bret_clean = _sanitize_returns(bret)
            curve_df = pd.DataFrame(
                {
                    "x": _x_axis,
                    "Strategy": np.exp(np.cumsum(_sret_clean)),
                    "Buy & Hold": np.exp(np.cumsum(_bret_clean)),
                }
            )

            st.markdown("#### Strategy vs Buy-and-Hold Equity Curve")
            eq_fig = go.Figure()
            eq_fig.add_trace(
                go.Scatter(
                    x=curve_df["x"],
                    y=curve_df["Strategy"],
                    mode="lines",
                    name="Strategy",
                    line=dict(color="#0f766e", width=2.5),
                )
            )
            eq_fig.add_trace(
                go.Scatter(
                    x=curve_df["x"],
                    y=curve_df["Buy & Hold"],
                    mode="lines",
                    name="Buy & Hold",
                    line=dict(color="#b91c1c", width=1.8, dash="dot"),
                )
            )
            eq_fig.add_hline(y=1.0, line_dash="dot", line_color="#9ca3af", annotation_text="Starting value")
            eq_fig.update_layout(
                height=400,
                yaxis_title="Portfolio Value  ($1 = initial investment)",
                xaxis_title="Date" if isinstance(_x_axis, pd.DatetimeIndex) else "Trading Day",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            try:
                st.plotly_chart(eq_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render equity curve chart. Error: {e}")
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

    # ── Per-ticker backtest drilldown (portfolio mode only) ───────────────────
    if _is_portfolio:
        _show_per_ticker_view(backtest)

    # ── Quantos AI Insights ───────────────────────────────────────────────────
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
        topic="Strategic Edge Arsenal — backtest quality, Sharpe, Calmar, drawdown analysis",
        snapshot=_edge_snapshot,
        key_suffix="edge_arsenal",
    )
