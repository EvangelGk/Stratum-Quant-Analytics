import numpy as np
import pandas as pd

def run_professional_backtest(
    df,
    price_col='close',
    rolling_window=20,
    zscore_threshold=1.0,
    volatility_window=14,
    volatility_mean_window=20,
    min_volatility=None,
    trend_window=200,
    transaction_cost=0.0015,
    annualization_factor=252
):
    # --- Rolling Z-score Signal ---
    returns = df[price_col].pct_change()
    rolling_mean = returns.rolling(window=rolling_window, min_periods=rolling_window).mean()
    rolling_std = returns.rolling(window=rolling_window, min_periods=rolling_window).std()
    zscore = (returns - rolling_mean) / rolling_std
    signal = np.where(zscore > zscore_threshold, 1, np.where(zscore < -zscore_threshold, -1, 0))
    # --- Trend Filter (200-SMA) ---
    sma = df[price_col].rolling(window=trend_window, min_periods=trend_window).mean()
    trend = df[price_col] > sma
    # --- Volatility Filter (ATR or StdDev > rolling mean) ---
    vol = returns.rolling(window=volatility_window, min_periods=volatility_window).std()
    vol_mean = vol.rolling(window=volatility_mean_window, min_periods=volatility_mean_window).mean()
    vol_filter = vol > vol_mean
    # --- Final Signal: Apply all filters, shift for execution lag ---
    filtered_signal = pd.Series(signal, index=df.index)
    filtered_signal = filtered_signal.where(trend, 0)
    filtered_signal = filtered_signal.where(vol_filter, 0)
    exec_signal = filtered_signal.shift(1)
    # --- Only apply transaction cost on trades (signal changes) ---
    trades = exec_signal.fillna(0).diff().abs()
    trades.iloc[0] = abs(exec_signal.iloc[0]) if not pd.isna(exec_signal.iloc[0]) else 0
    # --- Strategy Returns with Transaction Costs ---
    strat_ret = exec_signal * returns - trades * transaction_cost
    # --- Drop NaNs from shifting/rolling ---
    valid = ~(strat_ret.isna() | returns.isna() | exec_signal.isna())
    strat_ret = strat_ret[valid]
    returns = returns[valid]
    exec_signal = exec_signal[valid]
    # --- Cumulative Wealth and Drawdown ---
    wealth = (1 + strat_ret).cumprod()
    cummax_wealth = wealth.cummax()
    drawdown = (wealth / cummax_wealth) - 1
    max_drawdown = drawdown.min()
    # --- Metrics ---
    ann_return = np.expm1(np.log1p(strat_ret).mean() * annualization_factor)
    ann_vol = strat_ret.std(ddof=1) * np.sqrt(annualization_factor)
    sharpe = (strat_ret.mean() / strat_ret.std(ddof=1)) * np.sqrt(annualization_factor) if strat_ret.std(ddof=1) > 0 else np.nan
    downside = strat_ret[strat_ret < 0]
    sortino = (strat_ret.mean() / downside.std(ddof=1)) * np.sqrt(annualization_factor) if downside.std(ddof=1) > 0 else np.nan
    profit_factor = strat_ret[strat_ret > 0].sum() / abs(strat_ret[strat_ret < 0].sum()) if abs(strat_ret[strat_ret < 0].sum()) > 0 else np.nan
    calmar = ann_return / abs(max_drawdown) if abs(max_drawdown) > 0 else np.nan
    metrics = {
        'Annualized Return %': round(ann_return * 100, 2),
        'Annualized Volatility %': round(ann_vol * 100, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Sortino Ratio': round(sortino, 3),
        'Max Drawdown %': round(max_drawdown * 100, 2),
        'Profit Factor': round(profit_factor, 3),
        'Calmar Ratio': round(calmar, 3)
    }
    return metrics

def grid_search_professional(
    df,
    price_col='close',
    rolling_windows=[10, 20, 50, 100],
    zscore_thresholds=[1.0, 1.5, 2.0, 2.5],
    trend_window=200,
    transaction_cost=0.0015
):
    results = []
    for rw in rolling_windows:
        for zt in zscore_thresholds:
            metrics = run_professional_backtest(
                df,
                price_col=price_col,
                rolling_window=rw,
                zscore_threshold=zt,
                trend_window=trend_window,
                transaction_cost=transaction_cost
            )
            if metrics['Profit Factor'] > 1.0 and metrics['Sharpe Ratio'] > 0.5:
                result = metrics.copy()
                result['Rolling Window'] = rw
                result['Z-Score Threshold'] = zt
                results.append(result)
    results = sorted(results, key=lambda x: x['Sortino Ratio'], reverse=True)
    return pd.DataFrame(results).head(5)
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

from exceptions.MedallionExceptions import AnalysisError, DataValidationError
from .mixed_frequency import prepare_supervised_frame


def _tracking_error(actual: pd.Series, predicted: np.ndarray) -> float:
    diff = np.asarray(actual.values, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.std(diff, ddof=1))


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    equity_curve = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / np.maximum(peaks, 1e-12)) - 1.0
    return float(np.min(drawdowns))


def _annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    clipped = np.clip(arr, -0.9999, None)
    log_sum = float(np.sum(np.log1p(clipped)))
    if not np.isfinite(log_sum):
        return 0.0
    years = max(arr.size / float(periods_per_year), 1.0 / float(periods_per_year))
    ann = float(np.exp(log_sum / years) - 1.0)
    return float(np.clip(ann, -0.99, 99.0))


def _effective_periods_per_year(metadata: Dict[str, Any], target: str) -> int:
    target_meta = metadata.get(target) if isinstance(metadata, dict) else None
    horizon = (
        int((target_meta or {}).get("target_horizon_days", 1))
        if isinstance(target_meta, dict)
        else 1
    )
    horizon = max(1, horizon)
    return max(1, int(round(252.0 / float(horizon))))


def _rolling_sharpe(returns: np.ndarray, window: int = 30) -> list[dict[str, float | int]]:
    series = pd.Series(np.asarray(returns, dtype=float))
    if len(series) < max(5, window):
        return []
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=1)
    sharpe = (mean / std.replace(0.0, np.nan)) * np.sqrt(252.0)
    out: list[dict[str, float | int]] = []
    for idx, value in sharpe.items():
        if pd.notna(value):
            out.append({"step": int(idx) + 1, "rolling_sharpe": float(value)})
    return out


def backtest_pre2020_holdout(
    df: pd.DataFrame,
    target: str = "log_return",
    features: Optional[List[str]] = None,
    date_col: str = "date",
    ticker: Optional[str] = None,
) -> Dict[str, Any]:
    """Train before 2020 and evaluate on 2020-2022 holdout window."""
    try:
        features = features or ["inflation", "energy_index"]
        panel, metadata = prepare_supervised_frame(
            df=df,
            target=target,
            features=features,
            date_col=date_col,
            ticker=ticker,
            macro_lag_days=45,
            align_target_to_features=True,
        )
        if panel.empty or date_col not in panel.columns:
            raise DataValidationError("No aligned rows available for backtest.")

        panel[date_col] = pd.to_datetime(panel[date_col], errors="coerce")
        panel = panel.dropna(subset=[date_col]).sort_values(date_col)

        train_mask = panel[date_col] < pd.Timestamp("2020-01-01")
        test_mask = (panel[date_col] >= pd.Timestamp("2020-01-01")) & (
            panel[date_col] <= pd.Timestamp("2022-12-31")
        )
        train_df = panel.loc[train_mask].copy()
        test_df = panel.loc[test_mask].copy()

        if len(train_df) < max(60, len(features) * 12):
            raise DataValidationError("Insufficient training rows before 2020.")
        if len(test_df) < 30:
            raise DataValidationError("Insufficient 2020-2022 holdout rows.")

        model = Ridge(alpha=1.0)
        model.fit(train_df[features], train_df[target])
        predictions = model.predict(test_df[features])
        actual = pd.to_numeric(test_df[target], errors="coerce").fillna(0.0)

        # Trade returns: go long when signal >= 0, short when signal < 0.
        signal = np.where(np.asarray(predictions, dtype=float) >= 0.0, 1.0, -1.0)
        actual_arr = np.asarray(actual, dtype=float)
        strategy_returns = signal * actual_arr
        benchmark_returns = actual_arr

        te = _tracking_error(actual, predictions)
        # MDD is a risk metric for the strategy holding the actual position,
        # not for the model's fitted values.  Use actual returns.
        mdd = _max_drawdown_from_returns(strategy_returns)

        wins = strategy_returns[strategy_returns > 0.0]
        losses = strategy_returns[strategy_returns < 0.0]
        win_prob = float(len(wins) / len(strategy_returns)) if len(strategy_returns) else 0.0
        loss_prob = float(len(losses) / len(strategy_returns)) if len(strategy_returns) else 0.0
        avg_win = float(np.mean(wins)) if len(wins) else 0.0
        avg_loss_abs = float(abs(np.mean(losses))) if len(losses) else 0.0
        expectancy = float((win_prob * avg_win) - (loss_prob * avg_loss_abs))

        gross_profit = float(np.sum(wins)) if len(wins) else 0.0
        gross_loss_abs = float(abs(np.sum(losses))) if len(losses) else 0.0
        profit_factor = (
            float(gross_profit / gross_loss_abs)
            if gross_loss_abs > 1e-12
            else (None if gross_profit == 0.0 else float("inf"))
        )

        periods_per_year = _effective_periods_per_year(metadata, target)
        ann_return = _annualized_return(strategy_returns, periods_per_year=periods_per_year)
        calmar = float(ann_return / max(abs(mdd), 0.01))

        active_returns = strategy_returns - benchmark_returns
        te_active = float(np.std(active_returns, ddof=1)) if len(active_returns) > 1 else None
        ir = (
            float(np.mean(active_returns) / te_active * np.sqrt(252.0))
            if te_active is not None and te_active > 1e-12
            else None
        )

        corr_r = None
        corr_p = None
        if len(strategy_returns) >= 3:
            try:
                corr_r, corr_p = pearsonr(np.asarray(predictions, dtype=float), actual_arr)
                corr_r = float(corr_r)
                corr_p = float(corr_p)
            except Exception:
                corr_r, corr_p = None, None

        sharpe = None
        sortino = None
        stdev = float(np.std(strategy_returns, ddof=1)) if len(strategy_returns) > 1 else None
        if stdev is not None and stdev > 1e-12:
            sharpe = float(np.mean(strategy_returns) / stdev * np.sqrt(252.0))
        downside = strategy_returns[strategy_returns < 0.0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else None
        if downside_std is not None and downside_std > 1e-12:
            sortino = float(np.mean(strategy_returns) / downside_std * np.sqrt(252.0))

        rolling_sharpe = _rolling_sharpe(strategy_returns, window=30)

        # Compact histogram payload for Streamlit rendering without recomputation.
        if len(strategy_returns) > 2:
            hist_counts, hist_edges = np.histogram(strategy_returns, bins=min(24, max(8, int(np.sqrt(len(strategy_returns))))))
            trade_hist = {
                "edges": [float(x) for x in hist_edges.tolist()],
                "counts": [int(x) for x in hist_counts.tolist()],
            }
        else:
            trade_hist = {"edges": [], "counts": []}

        return {
            "window": {
                "train_end_exclusive": "2020-01-01",
                "test_start": "2020-01-01",
                "test_end": "2022-12-31",
            },
            "ticker": ticker,
            "target": target,
            "features": list(features),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "tracking_error": round(float(te), 8),
            "maximum_drawdown": round(float(mdd), 8),
            "sharpe_ratio": round(float(sharpe), 8) if sharpe is not None else None,
            "sortino_ratio": round(float(sortino), 8) if sortino is not None else None,
            "rolling_sharpe_30d": rolling_sharpe,
            "expectancy_per_trade": round(expectancy, 8),
            "win_probability": round(win_prob, 8),
            "loss_probability": round(loss_prob, 8),
            "average_win": round(avg_win, 8),
            "average_loss_abs": round(avg_loss_abs, 8),
            "profit_factor": (
                round(float(profit_factor), 8)
                if isinstance(profit_factor, (float, int)) and np.isfinite(float(profit_factor))
                else ("inf" if profit_factor == float("inf") else None)
            ),
            "annualized_return": round(float(ann_return), 8),
            "annualization_periods_per_year": int(periods_per_year),
            "calmar_ratio": round(float(calmar), 8) if calmar is not None else None,
            "information_ratio": round(float(ir), 8) if ir is not None else None,
            "active_return_tracking_error": round(float(te_active), 8) if te_active is not None else None,
            "correlation_test": {
                "pearson_r": round(float(corr_r), 8) if corr_r is not None else None,
                "p_value": round(float(corr_p), 10) if corr_p is not None else None,
            },
            "trade_distribution_histogram": trade_hist,
            "strategy_returns": [float(v) for v in strategy_returns.tolist()],
            "benchmark_returns": [float(v) for v in benchmark_returns.tolist()],
            "predictions": [float(v) for v in predictions.tolist()],
            "actual": [float(v) for v in actual.tolist()],
            "transformations": metadata,
        }
    except DataValidationError:
        raise
    except Exception as exc:
        raise AnalysisError(f"Unexpected error in backtest_pre2020_holdout: {exc}") from exc
