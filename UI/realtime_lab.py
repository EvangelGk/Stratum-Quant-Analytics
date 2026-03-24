from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_TABLE = PROJECT_ROOT / "data" / "gold" / "master_table.parquet"


def _max_drawdown(returns: np.ndarray) -> float:
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    dd = (equity / np.maximum(peaks, 1e-12)) - 1.0
    return float(np.min(dd)) if len(dd) else 0.0


def _annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        return 0.0
    growth = float(np.prod(1.0 + returns))
    if growth <= 0.0:
        return -1.0
    years = max(len(returns) / float(periods_per_year), 1.0 / float(periods_per_year))
    return float(growth ** (1.0 / years) - 1.0)


def _rolling_sharpe(returns: np.ndarray, window: int = 30) -> pd.DataFrame:
    s = pd.Series(returns)
    if len(s) < max(window, 5):
        return pd.DataFrame(columns=["step", "rolling_sharpe"])
    mean = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std(ddof=1).replace(0.0, np.nan)
    rs = (mean / std) * np.sqrt(252.0)
    out = pd.DataFrame({"step": np.arange(1, len(s) + 1), "rolling_sharpe": rs.values})
    return out.dropna(subset=["rolling_sharpe"])


def _compute_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float | None]:
    signal = np.where(pred >= 0.0, 1.0, -1.0)
    strategy_returns = signal * actual
    benchmark_returns = actual

    wins = strategy_returns[strategy_returns > 0.0]
    losses = strategy_returns[strategy_returns < 0.0]
    win_prob = float(len(wins) / len(strategy_returns)) if len(strategy_returns) else 0.0
    loss_prob = float(len(losses) / len(strategy_returns)) if len(strategy_returns) else 0.0
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss_abs = float(abs(np.mean(losses))) if len(losses) else 0.0
    expectancy = (win_prob * avg_win) - (loss_prob * avg_loss_abs)

    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss_abs = float(abs(np.sum(losses))) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss_abs) if gross_loss_abs > 1e-12 else None

    mdd = _max_drawdown(strategy_returns)
    ann = _annualized_return(strategy_returns)
    calmar = (ann / abs(mdd)) if abs(mdd) > 1e-12 else None

    vol = float(np.std(strategy_returns, ddof=1)) if len(strategy_returns) > 1 else None
    sharpe = (float(np.mean(strategy_returns)) / vol * np.sqrt(252.0)) if (vol and vol > 1e-12) else None

    active = strategy_returns - benchmark_returns
    te = float(np.std(active, ddof=1)) if len(active) > 1 else None
    ir = (float(np.mean(active)) / te * np.sqrt(252.0)) if (te and te > 1e-12) else None

    p_val = None
    corr_r = None
    if len(pred) >= 3:
        try:
            corr_r, p_val = pearsonr(pred, actual)
            corr_r = float(corr_r)
            p_val = float(p_val)
        except Exception:
            corr_r, p_val = None, None

    return {
        "expectancy_per_trade": float(expectancy),
        "profit_factor": float(profit_factor) if profit_factor is not None else None,
        "max_drawdown": float(mdd),
        "annualized_return": float(ann),
        "calmar_ratio": float(calmar) if calmar is not None else None,
        "sharpe_ratio": float(sharpe) if sharpe is not None else None,
        "information_ratio": float(ir) if ir is not None else None,
        "p_value": float(p_val) if p_val is not None else None,
        "pearson_r": float(corr_r) if corr_r is not None else None,
    }


def main() -> None:
    st.set_page_config(page_title="Quant Real-Time Lab", page_icon="🧪", layout="wide")
    st.title("🧪 Real-Time Quant Lab (No Commit Needed)")
    st.caption("Run fast diagnostics, GridSearchCV tuning, lag-correlation heatmaps, and anti-R² metrics directly from Gold data.")

    if not MASTER_TABLE.exists():
        st.error("Gold master table not found. Run the full pipeline first.")
        return

    df = pd.read_parquet(MASTER_TABLE)
    if "date" not in df.columns or "close" not in df.columns:
        st.error("Master table is missing required columns: date and close.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    tickers = sorted(df["ticker"].dropna().astype(str).unique().tolist()) if "ticker" in df.columns else ["ALL"]
    macro_cols = [
        c
        for c in df.columns
        if c not in {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "log_return"}
        and not str(c).startswith("__")
    ]

    with st.sidebar:
        st.header("Real-Time Controls")
        ticker = st.selectbox("Ticker", options=tickers)
        metric = st.selectbox("FRED metric", options=sorted(macro_cols))
        lag_max = st.slider("Max lag", min_value=5, max_value=40, value=20, step=1)
        test_ratio = st.slider("Test ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        run = st.button("Run Real-Time Test", type="primary", use_container_width=True)

    if not run:
        st.info("Select ticker/metric and press Run Real-Time Test.")
        return

    scoped = df.copy()
    if "ticker" in scoped.columns:
        scoped = scoped[scoped["ticker"].astype(str) == str(ticker)].copy()

    work = scoped[["date", "close", metric]].copy()
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=["close", metric]).sort_values("date")

    if len(work) < 120:
        st.warning("Not enough rows for robust real-time test. Need at least ~120 rows.")
        return

    # Lag-correlation heatmap (1..lag_max)
    lag_values = list(range(1, lag_max + 1))
    lag_corr = []
    for lag in lag_values:
        shifted = work[metric].shift(lag)
        aligned = pd.DataFrame({"close": work["close"], "macro_lag": shifted}).dropna()
        lag_corr.append(float(aligned["close"].corr(aligned["macro_lag"])) if len(aligned) > 20 else np.nan)

    st.subheader("Correlation Heatmap Before Optimization")
    fig_heat = px.imshow(
        [lag_corr],
        labels={"x": "Lag (days)", "y": "Metric", "color": "Correlation"},
        x=lag_values,
        y=[metric],
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=f"{metric} vs {ticker} close across lags 1-{lag_max}",
    )
    fig_heat.update_layout(height=260)
    st.plotly_chart(fig_heat, width="stretch")

    # Build supervised matrix with lagged macro features and target returns
    model_df = work.copy()
    model_df["target_return"] = model_df["close"].pct_change().shift(-1)
    for lag in lag_values:
        model_df[f"{metric}_lag_{lag}"] = model_df[metric].shift(lag)
    model_df = model_df.dropna().reset_index(drop=True)

    feature_cols = [f"{metric}_lag_{lag}" for lag in lag_values]
    split = int(len(model_df) * (1.0 - float(test_ratio)))
    split = max(80, min(split, len(model_df) - 30))
    train = model_df.iloc[:split]
    test = model_df.iloc[split:]

    x_train = train[feature_cols]
    y_train = train["target_return"]
    x_test = test[feature_cols]
    y_test = test["target_return"]

    # Standard GridSearchCV path (no custom random loops)
    candidates = {
        "Ridge": (
            Ridge(),
            {"model__alpha": [1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0]},
        ),
        "Lasso": (
            Lasso(max_iter=5000),
            {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
        ),
        "ElasticNet": (
            ElasticNet(max_iter=5000),
            {
                "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                "model__l1_ratio": [0.2, 0.5, 0.8],
            },
        ),
    }

    tscv = TimeSeriesSplit(n_splits=max(2, min(5, len(x_train) // 60)))
    best_model = None
    best_name = ""
    best_score = float("-inf")
    best_params: dict[str, object] = {}

    for name, (estimator, grid) in candidates.items():
        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("model", estimator)])
        gs = GridSearchCV(pipe, grid, cv=tscv, scoring="r2", n_jobs=-1, refit=True)
        gs.fit(x_train, y_train)
        if float(gs.best_score_) > best_score:
            best_score = float(gs.best_score_)
            best_name = name
            best_model = gs.best_estimator_
            best_params = gs.best_params_

    if best_model is None:
        st.error("GridSearchCV failed to find a model.")
        return

    pred = best_model.predict(x_test)
    oos_r2 = float(r2_score(y_test, pred))
    metrics = _compute_metrics(pred=np.asarray(pred, dtype=float), actual=np.asarray(y_test, dtype=float))

    st.subheader("Model Selection (GridSearchCV)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", best_name)
    c2.metric("CV R²", f"{best_score:.4f}")
    c3.metric("OOS R²", f"{oos_r2:.4f}")
    st.json({"best_params": best_params})

    st.subheader("Anti-R² Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Expectancy", f"{metrics['expectancy_per_trade']:.6f}")
    m2.metric("Profit Factor", "N/A" if metrics["profit_factor"] is None else f"{metrics['profit_factor']:.4f}")
    m3.metric("Calmar", "N/A" if metrics["calmar_ratio"] is None else f"{metrics['calmar_ratio']:.4f}")
    m4.metric("Info Ratio", "N/A" if metrics["information_ratio"] is None else f"{metrics['information_ratio']:.4f}")
    m5.metric("Max Drawdown", f"{metrics['max_drawdown']:.4f}")

    c6, c7, c8 = st.columns(3)
    c6.metric("Sharpe", "N/A" if metrics["sharpe_ratio"] is None else f"{metrics['sharpe_ratio']:.4f}")
    c7.metric("Pearson r", "N/A" if metrics["pearson_r"] is None else f"{metrics['pearson_r']:.4f}")
    c8.metric("P-value", "N/A" if metrics["p_value"] is None else f"{metrics['p_value']:.6f}")

    signal = np.where(np.asarray(pred, dtype=float) >= 0.0, 1.0, -1.0)
    strategy_returns = signal * np.asarray(y_test, dtype=float)

    st.subheader("Trade Distribution Histogram")
    fig_hist = px.histogram(
        pd.DataFrame({"strategy_return": strategy_returns}),
        x="strategy_return",
        nbins=30,
        color_discrete_sequence=["#0f766e"],
        title="Distribution of gains and losses",
    )
    fig_hist.add_vline(x=0.0, line_dash="dot", line_color="#7f1d1d")
    fig_hist.update_layout(height=340)
    st.plotly_chart(fig_hist, width="stretch")

    st.subheader("Rolling Sharpe Ratio (30-day)")
    rs = _rolling_sharpe(strategy_returns, window=30)
    if rs.empty:
        st.caption("Not enough rows for rolling Sharpe.")
    else:
        fig_rs = px.line(rs, x="step", y="rolling_sharpe", title="Rolling Sharpe through time")
        fig_rs.add_hline(y=0.0, line_dash="dot", line_color="#777")
        fig_rs.add_hline(y=1.0, line_dash="dash", line_color="#0f766e")
        fig_rs.update_layout(height=340)
        st.plotly_chart(fig_rs, width="stretch")

    if metrics["expectancy_per_trade"] > 0.0:
        st.success("Positive mathematical edge detected (expectancy > 0).")
    else:
        st.warning("Expectancy is not positive in this sample. Consider feature set or regime split changes.")


if __name__ == "__main__":
    main()
