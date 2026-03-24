"""
lab_sandbox.py — Real-time experiment workbench for Scenario Planner.

Run with:   streamlit run lab_sandbox.py
No commits needed — this is a standalone exploration tool.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── optional heavy deps ───────────────────────────────────────────────────────
try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

try:
    import fredapi
    _HAS_FRED = True
except ImportError:
    _HAS_FRED = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# ── constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output" / "default"
DEFAULT_FRED_METRICS = [
    "UNRATE",
    "CPIAUCSL",
    "FEDFUNDS",
    "GS10",
    "INDPRO",
    "DPCERA3M086SBEA",
]
MODEL_REGISTRY = {
    "LinearRegression": LinearRegression() if _HAS_SKLEARN else None,
    "Ridge": Ridge() if _HAS_SKLEARN else None,
    "Lasso": Lasso(max_iter=5000) if _HAS_SKLEARN else None,
    "ElasticNet": ElasticNet(max_iter=5000) if _HAS_SKLEARN else None,
    "RandomForest": RandomForestRegressor(n_jobs=-1, random_state=42) if _HAS_SKLEARN else None,
}
PARAM_GRIDS = {
    "LinearRegression": {},
    "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "ElasticNet": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "RandomForest": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 5, 10, None],
    },
}


# ── R² commentary ─────────────────────────────────────────────────────────────

def quantos_r2_commentary(r2: float) -> str:
    """
    Standalone rule-based R² interpretation — no LLM required.
    Returns a human-readable HTML-safe string.
    """
    if r2 >= 0.10:
        return (
            "✅ **Strong signal** — R² ≥ 0.10. The model explains meaningful variance in the target. "
            "Validate on out-of-sample windows to ensure this holds under regime changes."
        )
    if r2 >= 0.02:
        return (
            "🟡 **Modest signal** — R² between 0.02 and 0.10. Predictive power is real but modest. "
            "Check directional accuracy alongside R² — even modest R² can translate to directional edge."
        )
    if r2 >= -0.05:
        return (
            "🟠 **Near-zero R²** — entirely **normal** for daily equity return targets with macro variables. "
            "Publication lags (45+ days), regime shifts, and low-frequency FRED data structurally cap R². "
            "Focus on trend/directional accuracy (>55% is commercially viable) and walk-forward stability."
        )
    if r2 >= -0.15:
        return (
            "🔶 **Mildly negative R²** — not a red flag on its own. This range is common when macro indicators "
            "are used against high-frequency targets. The model may still have directional usefulness. "
            "Check: (1) directional accuracy, (2) factor freshness alignment, (3) volatility regime."
        )
    if r2 >= -0.30:
        return (
            "⚠️ **Moderately negative R²** — investigation warranted. Potential causes: stale data (publication "
            "lag mismatch), high factor concentration in one noisy series, or structural regime break. "
            "Review freshness diagnostics and consider restricting to post-2015 training data."
        )
    return (
        "🚨 **Strongly negative R²** — data quality investigation required. "
        "Likely causes: target/feature mismatch, severe look-ahead leakage, or unit root in both series. "
        "Run stationarity checks (ADF) on all inputs and review lag alignment."
    )


# ── data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching price data…")
def load_yfinance_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    if not _HAS_YF:
        st.error("yfinance is not installed. Run: pip install yfinance")
        return pd.DataFrame()
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df[["Close"]].rename(columns={"Close": "price"})
    df.index = pd.to_datetime(df.index)
    df["returns"] = df["price"].pct_change()
    return df


@st.cache_data(ttl=3600, show_spinner="Fetching FRED series…")
def load_fred_data(
    series_ids: list[str],
    fred_api_key: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    if not _HAS_FRED:
        st.error("fredapi is not installed. Run: pip install fredapi")
        return pd.DataFrame()
    try:
        fred = fredapi.Fred(api_key=fred_api_key)
    except Exception as exc:
        st.error(f"FRED init failed: {exc}")
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for sid in series_ids:
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            frames.append(s.rename(sid).to_frame())
        except Exception as exc:
            st.warning(f"Could not fetch {sid}: {exc}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    return df


def build_merged_frame(
    price_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    lag_fill_limit: int = 5,
) -> pd.DataFrame:
    """Merge equity prices with FRED data using merge_asof (forward-fill on FRED)."""
    if price_df.empty or fred_df.empty:
        return pd.DataFrame()
    price_reset = price_df.reset_index().rename(columns={"Date": "date", "index": "date"})
    fred_reset = fred_df.ffill(limit=lag_fill_limit).reset_index().rename(
        columns={"index": "date", "DATE": "date"}
    )
    price_reset["date"] = pd.to_datetime(price_reset["date"])
    fred_reset["date"] = pd.to_datetime(fred_reset["date"])
    merged = pd.merge_asof(
        price_reset.sort_values("date"),
        fred_reset.sort_values("date"),
        on="date",
        direction="backward",
    )
    merged = merged.set_index("date").dropna(how="all")
    return merged


# ── correlation heatmap ───────────────────────────────────────────────────────

def compute_lag_correlations(
    merged: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    max_lag: int = 20,
) -> pd.DataFrame:
    """
    Compute Pearson correlation between each feature (lagged 1..max_lag) and the target.
    Returns DataFrame (features × lags).
    """
    if merged.empty or target_col not in merged.columns:
        return pd.DataFrame()
    records: dict[str, list] = {col: [] for col in feature_cols}
    lags = list(range(1, max_lag + 1))
    target = merged[target_col].dropna()
    for lag in lags:
        for col in feature_cols:
            if col not in merged.columns:
                records[col].append(np.nan)
                continue
            feature_lagged = merged[col].shift(lag)
            aligned = pd.concat([target, feature_lagged], axis=1).dropna()
            if len(aligned) < 10:
                records[col].append(np.nan)
            else:
                r = aligned.corr().iloc[0, 1]
                records[col].append(float(r))
    return pd.DataFrame(records, index=lags).T  # shape: (features, lags)


def render_heatmap_tab(sidebar: dict) -> None:
    st.header("Correlation Heatmap — FRED metrics × Lag")
    st.caption(
        "Pearson correlation between each lagged FRED metric and equity returns. "
        "Strong colour at a particular lag suggests that metric leads equity prices by that many periods."
    )

    fred_api_key = sidebar["fred_api_key"]
    ticker = sidebar["ticker"]
    start = sidebar["start"]
    end = sidebar["end"]
    fred_cols = sidebar["fred_cols"]
    max_lag = sidebar["max_lag"]
    target_col = sidebar["target_col"]

    if st.button("Run Heatmap", key="run_heatmap"):
        price_df = load_yfinance_data(ticker, start, end)
        fred_df = load_fred_data(fred_cols, fred_api_key, start, end)
        merged = build_merged_frame(price_df, fred_df)

        if merged.empty:
            st.warning("No merged data available. Check your ticker, FRED series, and date range.")
            return

        with st.spinner("Computing lag correlations…"):
            corr_df = compute_lag_correlations(merged, target_col, fred_cols, max_lag)

        if corr_df.empty:
            st.warning("No correlation data computed.")
            return

        if _HAS_PLOTLY:
            fig = px.imshow(
                corr_df,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0.0,
                zmin=-1.0,
                zmax=1.0,
                labels={"x": "Lag (periods)", "y": "FRED metric", "color": "Pearson r"},
                title=f"Correlation: FRED metrics × lags 1–{max_lag} vs {ticker} {target_col}",
                aspect="auto",
            )
            fig.update_layout(height=400 + len(fred_cols) * 20)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(corr_df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1))

        # Top signals table
        top_signals = (
            corr_df.stack()
            .reset_index()
            .rename(columns={"level_0": "metric", "level_1": "lag", 0: "pearson_r"})
            .assign(abs_r=lambda x: x["pearson_r"].abs())
            .sort_values("abs_r", ascending=False)
            .head(20)
            .drop(columns="abs_r")
            .reset_index(drop=True)
        )
        st.subheader("Top 20 Signals by |r|")
        st.dataframe(top_signals, use_container_width=True)


# ── GridSearchCV tab ──────────────────────────────────────────────────────────

def build_pipeline(model_name: str) -> Optional["Pipeline"]:
    if not _HAS_SKLEARN:
        return None
    model = MODEL_REGISTRY.get(model_name)
    if model is None:
        return None
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def run_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    param_grid: dict,
    n_splits: int = 5,
    test_size: float = 0.2,
) -> dict:
    if not _HAS_SKLEARN:
        return {"error": "scikit-learn not installed"}

    pipe = build_pipeline(model_name)
    if pipe is None:
        return {"error": f"Model '{model_name}' not available"}

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_train) < n_splits * 2:
        return {"error": f"Not enough training data for {n_splits}-fold CV."}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=tscv,
        scoring="r2",
        return_train_score=True,
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    oos_r2 = float(gs.score(X_test, y_test))
    cv_results_df = pd.DataFrame(gs.cv_results_)

    return {
        "best_params": gs.best_params_,
        "best_cv_r2": float(gs.best_score_),
        "oos_r2": oos_r2,
        "cv_results": cv_results_df,
        "estimator": gs.best_estimator_,
    }


def render_gridsearch_tab(sidebar: dict) -> None:
    st.header("GridSearchCV — Model Selection")

    fred_api_key = sidebar["fred_api_key"]
    ticker = sidebar["ticker"]
    start = sidebar["start"]
    end = sidebar["end"]
    fred_cols = sidebar["fred_cols"]
    target_col = sidebar["target_col"]
    model_name = sidebar["model_name"]
    lag_for_features = sidebar["lag_for_features"]

    st.subheader("Parameter Grid")
    base_grid = PARAM_GRIDS.get(model_name, {})
    # Let user override param grid values
    user_grid: dict = {}
    for param, values in base_grid.items():
        short = param.replace("model__", "")
        raw = st.text_input(
            f"{short}",
            value=", ".join(str(v) for v in values if v is not None),
            key=f"pg_{model_name}_{param}",
        )
        parsed = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.lower() == "none":
                parsed.append(None)
            else:
                try:
                    parsed.append(int(tok))
                except ValueError:
                    try:
                        parsed.append(float(tok))
                    except ValueError:
                        parsed.append(tok)
        if parsed:
            user_grid[param] = parsed
    if not user_grid and base_grid:
        user_grid = base_grid

    n_splits = st.slider("CV folds (TimeSeriesSplit)", 2, 10, 5, key="cv_folds")

    if st.button("Run GridSearchCV", key="run_gs"):
        if not _HAS_SKLEARN:
            st.error("scikit-learn not installed.")
            return

        price_df = load_yfinance_data(ticker, start, end)
        fred_df = load_fred_data(fred_cols, fred_api_key, start, end)
        merged = build_merged_frame(price_df, fred_df)

        if merged.empty or target_col not in merged.columns:
            st.warning("No data available. Check settings.")
            return

        # Build feature matrix: lagged FRED cols
        feature_frames = []
        for col in fred_cols:
            if col in merged.columns:
                feature_frames.append(merged[col].shift(lag_for_features).rename(f"{col}_lag{lag_for_features}"))
        if not feature_frames:
            st.warning("No valid FRED features after lagging.")
            return

        X = pd.concat(feature_frames, axis=1)
        y = merged[target_col]
        aligned = pd.concat([X, y], axis=1).dropna()
        X = aligned.drop(columns=[target_col])
        y = aligned[target_col]

        if len(X) < 50:
            st.warning(f"Only {len(X)} rows after alignment — need at least 50.")
            return

        with st.spinner(f"Running GridSearchCV for {model_name}…"):
            result = run_grid_search(X, y, model_name, user_grid, n_splits=n_splits)

        if "error" in result:
            st.error(result["error"])
            return

        oos_r2 = result["oos_r2"]
        st.success(f"Done. OOS R² = {oos_r2:.4f} | Best CV R² = {result['best_cv_r2']:.4f}")
        st.json(result["best_params"])

        st.subheader("Quantos R² Interpretation")
        st.info(quantos_r2_commentary(oos_r2))

        st.subheader("CV Results")
        cv_df = result["cv_results"]
        display_cols = [c for c in cv_df.columns if "mean_test" in c or "std_test" in c or "param_" in c or "rank_test" in c]
        if display_cols:
            st.dataframe(cv_df[display_cols].sort_values("rank_test_score"), use_container_width=True)

        if _HAS_PLOTLY and "mean_test_score" in cv_df.columns:
            fig = px.histogram(
                cv_df,
                x="mean_test_score",
                nbins=20,
                title="Distribution of mean CV R² across parameter combinations",
                labels={"mean_test_score": "Mean CV R²"},
            )
            fig.add_vline(x=result["best_cv_r2"], line_dash="dash", annotation_text="Best CV R²")
            fig.add_vline(x=0, line_color="red", line_dash="dot", annotation_text="R²=0")
            st.plotly_chart(fig, use_container_width=True)


# ── Diagnostics tab ───────────────────────────────────────────────────────────

def render_diagnostics_tab() -> None:
    st.header("Pipeline Diagnostics")

    result_path = OUTPUT_DIR / "analysis_results.json"
    gov_path = OUTPUT_DIR / "governance_report.json"
    audit_path = OUTPUT_DIR / "audit_report.json"

    def _load(p: Path) -> Optional[dict]:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    analysis = _load(result_path)
    gov = _load(gov_path)
    audit = _load(audit_path)

    col1, col2 = st.columns(2)

    # ── OOS R² ──
    oos_r2: Optional[float] = None
    if gov:
        oos_r2 = gov.get("out_of_sample", {}).get("r2_score")
    elif analysis:
        oos_r2 = analysis.get("governance_report", {}).get("out_of_sample", {}).get("r2_score")

    with col1:
        if oos_r2 is not None:
            st.metric("OOS R²", f"{oos_r2:.4f}")
            st.info(quantos_r2_commentary(float(oos_r2)))
        else:
            st.info("No governance report found. Run the pipeline first.")

    # ── Model risk ──
    with col2:
        risk = None
        if gov:
            risk = gov.get("model_risk_score")
        elif analysis:
            risk = analysis.get("governance_report", {}).get("model_risk_score")
        if risk is not None:
            st.metric("Model Risk Score", f"{risk:.2f}")

    # ── Baseline comparison ──
    baseline = None
    if gov:
        baseline = gov.get("out_of_sample", {}).get("baseline_comparison")
    if baseline:
        st.subheader("Baseline Comparison")
        st.json(baseline)

    # ── Factor concentration ──
    fc = None
    if gov:
        fc = gov.get("factor_concentration")
    if fc:
        st.subheader("Factor Concentration")
        weights = fc.get("normalized_weights", {})
        if weights and _HAS_PLOTLY:
            fig = px.bar(
                x=list(weights.keys()),
                y=list(weights.values()),
                labels={"x": "Factor", "y": "Normalized weight"},
                title="Factor Concentration (normalized |coefficient| weights)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.json(fc)

    # ── Walk-forward chart ──
    wf = None
    if gov:
        wf = gov.get("walk_forward_backtest", {}).get("window_r2_scores")
    if wf:
        st.subheader("Walk-Forward R² per Window")
        wf_df = pd.DataFrame({"window": range(1, len(wf) + 1), "r2": wf})
        if _HAS_PLOTLY:
            fig = px.line(wf_df, x="window", y="r2", markers=True, title="Walk-Forward OOS R²")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(wf_df)

    # ── Freshness alignment ──
    freshness = None
    if gov:
        freshness = gov.get("freshness_alignment")
    if freshness:
        st.subheader("Freshness Alignment")
        st.json(freshness)

    # ── Stationarity summary ──
    stat = None
    if gov:
        stat = gov.get("stationarity_checks")
    if stat:
        st.subheader("Stationarity Summary")
        stat_df = pd.DataFrame(stat).T if isinstance(stat, dict) else pd.DataFrame(stat)
        st.dataframe(stat_df, use_container_width=True)

    # ── Leakage flags ──
    leakage = None
    if gov:
        leakage = gov.get("leakage_flags")
    if leakage:
        st.subheader("Leakage Flags")
        st.warning(f"{len(leakage)} leakage flag(s) detected")
        for lf in leakage:
            st.write(f"- {lf}")

    if not any([oos_r2, baseline, fc, wf, freshness, stat, leakage]):
        st.info(
            "No diagnostic data found. Run the pipeline (main.py or the Streamlit UI) to generate output files, "
            f"then check the `{OUTPUT_DIR}` directory."
        )


# ── sidebar ───────────────────────────────────────────────────────────────────

def build_sidebar() -> dict:
    st.sidebar.title("Lab Sandbox Settings")

    ticker = st.sidebar.text_input("Ticker", value="AAPL")
    fred_api_key = st.sidebar.text_input("FRED API Key", type="password", value="")
    target_col = st.sidebar.selectbox("Target column", ["returns", "price"])

    st.sidebar.subheader("Date Range")
    start = st.sidebar.text_input("Start date", value="2010-01-01")
    end = st.sidebar.text_input("End date", value="2024-12-31")

    st.sidebar.subheader("FRED Metrics")
    fred_cols = st.sidebar.multiselect(
        "Select series",
        DEFAULT_FRED_METRICS,
        default=DEFAULT_FRED_METRICS[:4],
    )

    st.sidebar.subheader("Lag Settings")
    max_lag = st.sidebar.slider("Max lag for heatmap", 1, 60, 20)
    lag_for_features = st.sidebar.slider("Lag to apply for GridSearchCV features", 0, 60, 3)

    st.sidebar.subheader("Model")
    model_name = st.sidebar.selectbox("Model", list(MODEL_REGISTRY.keys()))

    return {
        "ticker": ticker,
        "fred_api_key": fred_api_key,
        "target_col": target_col,
        "start": start,
        "end": end,
        "fred_cols": fred_cols if fred_cols else DEFAULT_FRED_METRICS[:2],
        "max_lag": max_lag,
        "lag_for_features": lag_for_features,
        "model_name": model_name,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Scenario Planner — Lab Sandbox",
        page_icon="🔬",
        layout="wide",
    )
    st.title("🔬 Lab Sandbox — Real-time Experiment Workbench")
    st.caption(
        "Standalone exploration tool. No commits required. "
        "Use the sidebar to configure data sources and model parameters."
    )

    if not _HAS_SKLEARN:
        st.error("scikit-learn is not installed. Run: pip install scikit-learn")

    sidebar = build_sidebar()

    tab_heatmap, tab_gs, tab_diag = st.tabs([
        "🌡️ Correlation Heatmap",
        "🔍 GridSearchCV",
        "📋 Diagnostics",
    ])

    with tab_heatmap:
        render_heatmap_tab(sidebar)

    with tab_gs:
        render_gridsearch_tab(sidebar)

    with tab_diag:
        render_diagnostics_tab()


if __name__ == "__main__":
    main()
