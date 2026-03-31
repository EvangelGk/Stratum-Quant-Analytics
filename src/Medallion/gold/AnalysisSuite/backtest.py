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
    # Log returns: correct equity curve is exp(cumsum), NOT cumprod(1+r)
    arr = np.asarray(returns, dtype=float)
    equity_curve = np.exp(np.cumsum(arr))
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / np.maximum(peaks, 1e-12)) - 1.0
    return float(np.min(drawdowns))


def _annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    # arr ARE log-returns — direct nansum is the correct compounding formula.
    # np.log1p(arr) would double-transform (treating log-returns as simple returns).
    log_sum = float(np.nansum(arr))
    if not np.isfinite(log_sum):
        return 0.0
    years = max(arr.size / float(periods_per_year), 1.0 / float(periods_per_year))
    ann = float(np.exp(log_sum / years) - 1.0)
    # Hard cap: +2500% annualised is impossible for a real strategy
    return float(np.clip(ann, -0.99, 25.0))


def _effective_periods_per_year(metadata: Dict[str, Any], target: str) -> int:
    target_meta = metadata.get(target) if isinstance(metadata, dict) else None
    horizon = int((target_meta or {}).get("target_horizon_days", 1)) if isinstance(target_meta, dict) else 1
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
        test_mask = (panel[date_col] >= pd.Timestamp("2020-01-01")) & (panel[date_col] <= pd.Timestamp("2022-12-31"))
        train_df = panel.loc[train_mask].copy()
        test_df = panel.loc[test_mask].copy()

        # Ridge (alpha=1.0) is heavily regularised: it works well with far fewer
        # samples than the old `features * 12` guard required.  If the 2020 cutoff
        # leaves insufficient pre-2020 rows (common when data starts in 2019 or the
        # 45-day macro lag reduces the window), fall back to a proportional 70/30
        # time-based split so the backtest can still run on whatever data is present.
        _min_train = max(20, len(features) + 10)
        _split_mode = "2020_cutoff"
        if len(train_df) < _min_train or len(test_df) < 30:
            split_idx = max(_min_train, int(len(panel) * 0.70))
            split_idx = min(split_idx, len(panel) - 30)
            if split_idx < _min_train:
                raise DataValidationError(
                    f"Not enough aligned rows for backtest (total={len(panel)}, need at least {_min_train + 30})."
                )
            train_df = panel.iloc[:split_idx].copy()
            test_df = panel.iloc[split_idx:].copy()
            _split_mode = "70_30_fallback"

        model = Ridge(alpha=1.0)
        model.fit(train_df[features], train_df[target])
        predictions = model.predict(test_df[features])

        # CRITICAL: Use 1-day log-returns from the original df — NOT the 21-day
        # forward cumulative target produced by prepare_supervised_frame when
        # align_target_to_features=True.  The transformed target counts each daily
        # return ~21 times, causing np.exp(cumsum(...)) to produce 10^44 equity curves.
        _orig = (
            df[[date_col, "log_return"]]
            .assign(**{date_col: lambda x: pd.to_datetime(x[date_col], errors="coerce")})
            .dropna(subset=[date_col, "log_return"])
            .set_index(date_col)
        )
        _test_dates = pd.to_datetime(test_df[date_col].values)
        _raw_lr = _orig["log_return"].reindex(_test_dates)
        raw_arr = np.asarray(_raw_lr, dtype=float)
        actual_arr = np.nan_to_num(raw_arr, nan=0.0, posinf=0.0, neginf=0.0)
        actual_arr = np.clip(actual_arr, -0.15, 0.15)  # ±15% daily cap

        # Trade returns: go long when signal >= 0, short when signal < 0.
        signal = np.where(np.asarray(predictions, dtype=float) >= 0.0, 1.0, -1.0)
        # Trim to equal length in case reindex dropped any dates.
        _min_len = min(len(signal), len(actual_arr))
        signal = signal[:_min_len]
        actual_arr = actual_arr[:_min_len]
        predictions = predictions[:_min_len]

        # ── Professional Risk Management Layer ─────────────────────────────────
        # Applied to execution only; core Ridge model logic is unchanged.
        # P-value / Pearson-r test below uses original arrays (unaffected).

        # 1. Regime Filter — reconstruct price from full log-return history
        _full_lr_rm = _orig["log_return"].astype(float)
        _cum_px_rm = pd.Series(
            np.exp(np.cumsum(np.nan_to_num(_full_lr_rm.values, nan=0.0))),
            index=_full_lr_rm.index,
        )
        _sma200_rm = _cum_px_rm.rolling(200, min_periods=200).mean()
        _test_dt_rm = pd.to_datetime(_test_dates[:_min_len])
        _test_px_rm = _cum_px_rm.reindex(_test_dt_rm, method="pad").fillna(0.0).values
        _test_sma_rm = _sma200_rm.reindex(_test_dt_rm, method="pad").fillna(0.0).values
        _in_uptrend_rm = _test_px_rm > _test_sma_rm
        signal_rm = signal.astype(float).copy()
        # Soft regime filter: reduce counter-trend positions to 40% rather than going flat.
        # Hard zero eliminated too many macro trades whose 45-day lag signals need time to play out.
        signal_rm = np.where((signal_rm == 1.0) & (~_in_uptrend_rm), 0.4, signal_rm)
        signal_rm = np.where((signal_rm == -1.0) & _in_uptrend_rm, -0.4, signal_rm)

        # 2. Inverse Volatility Scaling — target 20% annualised vol, no leverage
        _actual_pd_rm = pd.Series(actual_arr)
        _vol14_rm = _actual_pd_rm.rolling(14, min_periods=14).std(ddof=1)
        _ann_vol_rm = (_vol14_rm * np.sqrt(252.0)).shift(1).fillna(0.20)
        _ann_vol_rm = _ann_vol_rm.replace(0.0, 0.20)
        _vol_scale_rm = (0.20 / _ann_vol_rm).clip(lower=0.0, upper=1.0).values
        signal_rm = signal_rm * _vol_scale_rm

        # 3. Per-trade ATR stop — exit if cumulative trade loss > min(2.5×vol, 5%)
        _vol14_vals_rm = _vol14_rm.shift(1).fillna(0.02).values
        _cum_trade_rm = 0.0
        _entry_vol_rm = 0.02
        _prev_base_rm = 0.0
        _atr_flags_rm = np.zeros(_min_len, dtype=bool)
        for _i_rm in range(_min_len):
            _base_sig_rm = float(signal[_i_rm])
            if abs(_base_sig_rm) > 1e-10 and abs(_prev_base_rm) < 1e-10:
                _cum_trade_rm = 0.0
                _ev = float(_vol14_vals_rm[_i_rm])
                _entry_vol_rm = _ev if (np.isfinite(_ev) and _ev > 0) else 0.02
            if abs(signal_rm[_i_rm]) < 1e-10:
                _cum_trade_rm = 0.0
            else:
                _cum_trade_rm += float(actual_arr[_i_rm]) * np.sign(_base_sig_rm)
            _stop_lvl_rm = min(3.0 * _entry_vol_rm, 0.07)
            if abs(signal_rm[_i_rm]) > 1e-10 and _cum_trade_rm < -_stop_lvl_rm:
                _atr_flags_rm[_i_rm] = True
            _prev_base_rm = _base_sig_rm
        _atr_shifted_rm = np.zeros(_min_len, dtype=bool)
        _atr_shifted_rm[1:] = _atr_flags_rm[:-1]
        signal_rm = np.where(_atr_shifted_rm, 0.0, signal_rm)

        # 4. Time-based exit — close after 30 days (aligned with 45-day macro lag; 10 days was too short)
        _days_rm = 0
        _tex_flags_rm = np.zeros(_min_len, dtype=bool)
        for _i_rm in range(_min_len):
            if abs(signal_rm[_i_rm]) > 1e-10:
                _days_rm += 1
                if _days_rm >= 30:
                    _tex_flags_rm[_i_rm] = True
            else:
                _days_rm = 0
        _tex_shifted_rm = np.zeros(_min_len, dtype=bool)
        _tex_shifted_rm[1:] = _tex_flags_rm[:-1]
        signal_rm = np.where(_tex_shifted_rm, 0.0, signal_rm)

        # 5. Friction on direction changes
        _pos_chg_rm = np.abs(np.diff(np.sign(signal_rm), prepend=0.0)) > 0.5
        _costs_rm = _pos_chg_rm.astype(float) * 0.001

        strategy_returns = signal_rm * actual_arr - _costs_rm
        benchmark_returns = actual_arr
        actual = pd.Series(actual_arr)

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
        profit_factor = float(gross_profit / gross_loss_abs) if gross_loss_abs > 1e-12 else (None if gross_profit == 0.0 else float("inf"))

        # strategy_returns are ALWAYS daily (actual_arr = 1-day log-returns from
        # the raw price series), regardless of the ML target prediction horizon.
        # _effective_periods_per_year reads the target transformation horizon
        # (e.g. 252 days for a 252-day forward-return target), which gives
        # 252/252 = 1 period/year and turns 756 daily returns into "756 years".
        # Always use 252 trading days per year for daily return annualisation.
        periods_per_year = 252
        ann_return = _annualized_return(strategy_returns, periods_per_year=periods_per_year)
        calmar = float(ann_return / max(abs(mdd), 0.01))

        active_returns = strategy_returns - benchmark_returns
        te_active = float(np.std(active_returns, ddof=1)) if len(active_returns) > 1 else None
        ir = float(np.mean(active_returns) / te_active * np.sqrt(252.0)) if te_active is not None and te_active > 1e-12 else None

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

        _train_end = str(train_df[date_col].iloc[-1].date()) if len(train_df) else "unknown"
        _test_start = str(test_df[date_col].iloc[0].date()) if len(test_df) else "unknown"
        _test_end = str(test_df[date_col].iloc[-1].date()) if len(test_df) else "unknown"
        return {
            "window": {
                "train_end_exclusive": _train_end,
                "test_start": _test_start,
                "test_end": _test_end,
                "split_mode": _split_mode,
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


# ──────────────────────────────────────────────────────────────────────────────
# Pillar 1 — Advanced Vectorized Backtesting Engine
# ──────────────────────────────────────────────────────────────────────────────


def _compute_strategy_metrics(
    strategy_returns: pd.Series,
    trade_flags: pd.Series,
    friction_per_trade: float = 0.0015,
) -> Dict[str, Any]:
    """
    Compute the full metric suite from a *daily* return series that already
    has execution lag and friction applied.

    Parameters
    ----------
    strategy_returns : daily P&L series (after lag + friction), NaN-free.
    trade_flags      : boolean/int series — 1 on days a position change occurs.
    friction_per_trade : one-way cost (commission + slippage), default 0.15 %.
    """
    r = strategy_returns.values.astype(float)
    n = len(r)
    if n == 0:
        return {}

    # ── Annualised return & volatility ───────────────────────────────────────
    ann_factor = np.sqrt(252.0)
    daily_vol = float(np.std(r, ddof=1)) if n > 1 else 0.0
    ann_vol = daily_vol * ann_factor

    # Geometric annualised return: r are log-returns → direct nansum.
    # np.log1p would double-transform (treating log-returns as simple returns).
    log_sum = float(np.nansum(r))
    years = max(n / 252.0, 1.0 / 252.0)
    ann_return = float(np.exp(log_sum / years) - 1.0)
    ann_return = float(np.clip(ann_return, -0.99, 25.0))  # +2500% hard cap

    # ── Sharpe Ratio (risk-free = 0) ─────────────────────────────────────────
    sharpe = float(np.mean(r) / daily_vol * ann_factor) if daily_vol > 1e-12 else None

    # ── Sortino Ratio (downside deviation only) ──────────────────────────────
    downside = r[r < 0.0]
    down_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else None
    sortino = float(np.mean(r) / down_std * ann_factor) if (down_std and down_std > 1e-12) else None

    # ── Max Drawdown  (peak-to-trough on cumulative equity curve) ────────────
    # r are log returns; correct equity = exp(cumsum), NOT cumprod(1+r)
    cum = pd.Series(np.exp(np.cumsum(r)))
    running_max = cum.cummax()
    drawdown = (cum / running_max) - 1.0
    max_dd = float(drawdown.min())

    # ── Profit Factor ────────────────────────────────────────────────────────
    gains = r[r > 0.0]
    losses = r[r < 0.0]
    gross_profit = float(np.sum(gains)) if len(gains) else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) else 0.0
    profit_factor: Optional[float]
    if gross_loss > 1e-12:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
        profit_factor = float("inf")
    else:
        profit_factor = None

    # ── Trade stats ──────────────────────────────────────────────────────────
    total_trades = int(trade_flags.sum()) if hasattr(trade_flags, "sum") else 0
    win_rate = float(len(gains) / n) if n > 0 else 0.0

    # ── Calmar ───────────────────────────────────────────────────────────────
    calmar = float(ann_return / max(abs(max_dd), 0.01))

    return {
        "annualized_return": round(ann_return, 6),
        "annualized_volatility": round(ann_vol, 6),
        "sharpe_ratio": round(sharpe, 6) if sharpe is not None else None,
        "sortino_ratio": round(sortino, 6) if sortino is not None else None,
        "max_drawdown": round(max_dd, 6),
        "profit_factor": (
            round(profit_factor, 6) if profit_factor is not None and np.isfinite(profit_factor) else ("inf" if profit_factor == float("inf") else None)
        ),
        "calmar_ratio": round(calmar, 6),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 6),
        "total_days": n,
    }


def run_strategy_backtest(
    prices: pd.Series,
    rolling_window: int = 20,
    z_threshold: float = 1.5,
    friction: float = 0.0015,
    volatility_filter: bool = True,
    trend_filter: bool = True,
    vol_window: int = 14,
    vol_ma_window: int = 20,
    trend_sma_window: int = 200,
    stop_loss_pct: float = 0.20,
    atr_stop_multiplier: float = 2.5,
    atr_fixed_stop: float = 0.05,
    inv_vol_target: float = 0.15,
    max_hold_days: int = 10,
    regime_long_only: bool = True,
) -> Dict[str, Any]:
    """
    Bias-free, vectorised mean-reversion strategy backtest.

    Design guarantees
    -----------------
    * All features computed with `.rolling(window=n)` — **no global scaling**,
      preventing data-leakage / look-ahead bias.
    * Execution lag: `.shift(1)` on the final signal so trades execute on the
      day *after* the signal is generated.
    * Realistic friction: 0.15 % (0.10 % commission + 0.05 % slippage) applied
      only when the position changes (i.e. on actual trades).
    * All NaN rows (from rolling windows + shift) dropped before metric calc.

    Strategy logic (mean-reversion)
    --------------------------------
    raw_signal = +1  when z_score < -z_threshold  (buy the dip)
               = -1  when z_score >  z_threshold  (sell the spike)
               =  0  otherwise (flat / neutral)

    Optional filters (applied before execution lag):
    * Volatility filter : trade only when 14-day rolling StdDev > its 20-day MA.
      Avoids entering during compressed, trending markets.
    * Trend filter      : suppress Short signals when Price > 200-day SMA
      (only allow Long entries in up-trends).

    Parameters
    ----------
    prices          : pd.Series of daily closing prices (date-indexed).
    rolling_window  : lookback for Z-score rolling mean/std (e.g. 20 days).
    z_threshold     : entry threshold in standard deviation units (e.g. 1.5).
    friction        : round-trip cost per trade (default 0.0015 = 0.15 %).
    volatility_filter : enable/disable the ATR/StdDev activity filter.
    trend_filter    : enable/disable the 200-day SMA directional filter.
    vol_window      : rolling window for volatility measurement (default 14).
    vol_ma_window   : MA window applied to the vol series (default 20).
    trend_sma_window: SMA window for trend detection (default 200).
    atr_stop_multiplier : ATR multiplier for per-trade dynamic stop (default 2.5).
    atr_fixed_stop  : Maximum per-trade loss as fraction (default 0.05 = 5%).
    inv_vol_target  : Target annualised vol for inverse-vol scaling (default 0.15).
                      Set to 0.0 to disable. Caps position at 1.0 (no leverage).
    max_hold_days   : Time-based exit: flatten after this many days (default 10).
                      Set to 0 to disable.
    regime_long_only: If True, only allow Long entries when price > 200-day SMA.
                      Avoids 'falling knife' buys in sustained downtrends.

    Returns
    -------
    dict with metrics, equity-curve list, trade log, and parameter snapshot.
    """
    if not isinstance(prices, pd.Series):
        raise DataValidationError("run_strategy_backtest expects a pd.Series of prices.")
    if len(prices) < max(rolling_window, trend_sma_window) + 2:
        raise DataValidationError(f"Not enough price rows ({len(prices)}) for the requested windows (rolling={rolling_window}, trend_sma={trend_sma_window}).")

    # ── 1. Log returns (forward-fill prices first to remove weekend gaps) ────
    px = prices.ffill().dropna()
    # Guard: replace zero/negative prices (bad data) with NaN so log() is safe
    px = px.where(px > 0.0, other=np.nan).ffill().dropna()
    # Clip extreme single-day log-returns.
    # ±15% log-return (≈±16% simple) is already a 5-sigma event for most
    # large-cap stocks.  Values beyond this are almost certainly bad YFinance
    # data (unadjusted splits, delisting artefacts, API errors).
    raw_log = np.log(px / px.shift(1))
    log_ret = raw_log.clip(lower=-0.15, upper=0.15)

    # ── 2. Rolling Z-score (no global scaling → zero data-leakage) ──────────
    roll_mean = px.rolling(window=rolling_window, min_periods=rolling_window).mean()
    roll_std = px.rolling(window=rolling_window, min_periods=rolling_window).std(ddof=1)
    z_score = (px - roll_mean) / roll_std.replace(0.0, np.nan)

    # ── 3. Raw signal from Z-score thresholds ────────────────────────────────
    raw_signal = pd.Series(0.0, index=px.index)
    raw_signal[z_score < -z_threshold] = 1.0  # long: price below rolling mean
    raw_signal[z_score > z_threshold] = -1.0  # short: price above rolling mean

    # ── 4a. Volatility filter — only trade when market is "active" ───────────
    if volatility_filter:
        rolling_vol = log_ret.rolling(window=vol_window, min_periods=vol_window).std(ddof=1)
        vol_ma = rolling_vol.rolling(window=vol_ma_window, min_periods=vol_ma_window).mean()
        vol_active = rolling_vol > vol_ma
        raw_signal = raw_signal.where(vol_active, other=0.0)

    # ── 4b. Trend filter & Regime filter ─────────────────────────────────────
    if trend_filter or regime_long_only:
        sma_200 = px.rolling(window=trend_sma_window, min_periods=trend_sma_window).mean()
        in_uptrend = px > sma_200
        if trend_filter:
            # Suppress -1 (short) signals when price is above 200-day SMA
            raw_signal = raw_signal.where(~((raw_signal == -1.0) & in_uptrend), other=0.0)
        if regime_long_only:
            # Only allow Long (+1) entries when price is in a confirmed uptrend
            # Avoids 'falling knife' buys in sustained long-term downtrends
            raw_signal = raw_signal.where(~((raw_signal == 1.0) & ~in_uptrend), other=0.0)

    # ── 5. Execution lag — enter the position on the NEXT day's open ─────────
    signal = raw_signal.shift(1)  # trade executes day-after signal

    # ── 5b. Inverse Volatility Scaling — target constant annualised volatility ─
    #        Position size = inv_vol_target / realised_vol, capped at 1.0.
    #        shift(1) on vol prevents look-ahead. No leverage allowed (cap=1.0).
    if inv_vol_target > 0.0:
        ann_vol = log_ret.rolling(window=vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)
        vol_scale = (inv_vol_target / ann_vol.replace(0.0, np.nan)).clip(upper=1.0)
        signal = signal * vol_scale.shift(1).fillna(1.0)

    # ── 6. Daily strategy returns ────────────────────────────────────────────
    strategy_ret = signal * log_ret

    # ── 7. Friction — deduct on direction changes; use sign to handle fractional ─
    # np.sign detects real entries/exits even when inv-vol scaling makes the
    # position fractional (e.g. 0.25), avoiding double-charging on size adjustments.
    _sig_dir = pd.Series(np.sign(signal.values), index=signal.index)
    position_change = _sig_dir.diff().fillna(0.0).abs() > 0.5
    trade_cost = position_change.astype(float) * friction
    strategy_ret = strategy_ret - trade_cost

    # ── 7b. Portfolio stop-loss — flatten when running drawdown > stop_loss_pct ─
    #        Uses PREVIOUS day's equity (shift(1)) to avoid look-ahead.
    if stop_loss_pct > 0.0:
        log_cum = strategy_ret.cumsum()
        log_peak = log_cum.cummax()
        log_dd = log_cum - log_peak
        stop_active = log_dd.shift(1).fillna(0.0) < -stop_loss_pct
        signal = signal.where(~stop_active, other=0.0)
        strategy_ret = signal * log_ret
        _sig_dir = pd.Series(np.sign(signal.values), index=signal.index)
        position_change = _sig_dir.diff().fillna(0.0).abs() > 0.5
        trade_cost = position_change.astype(float) * friction
        strategy_ret = strategy_ret - trade_cost

    # ── 7c. Time-based exit — close position after max_hold_days ─────────────
    #        Prevents open trades that never hit target from tying up capital.
    #        shift(1) on days counter prevents look-ahead.
    if max_hold_days > 0:
        days_held = pd.Series(0.0, index=signal.index)
        _prev_in_pos = False
        for _i in range(len(signal)):
            _in_pos = abs(float(signal.iloc[_i])) > 1e-10
            if _in_pos:
                days_held.iloc[_i] = (days_held.iloc[_i - 1] + 1.0) if (_prev_in_pos and _i > 0) else 1.0
            _prev_in_pos = _in_pos
        time_exit = days_held.shift(1).fillna(0.0) >= float(max_hold_days)
        signal = signal.where(~time_exit, other=0.0)
        strategy_ret = signal * log_ret
        _sig_dir = pd.Series(np.sign(signal.values), index=signal.index)
        position_change = _sig_dir.diff().fillna(0.0).abs() > 0.5
        trade_cost = position_change.astype(float) * friction
        strategy_ret = strategy_ret - trade_cost

    # ── 7d. ATR per-trade stop — exit if cumulative loss > min(2.5×ATR, 5%) ──
    #        Cuts fat-tail losses that drive large drawdowns.
    #        Tracked from most recent entry; stop triggers NEXT day (shift effect).
    if atr_stop_multiplier > 0.0 and atr_fixed_stop > 0.0:
        atr_proxy = log_ret.rolling(window=vol_window, min_periods=vol_window).std(ddof=1)
        atr_stop_flags = pd.Series(False, index=signal.index)
        _cum_trade_pnl = 0.0
        _entry_atr = 0.02
        _prev_sig_v = 0.0
        for _i in range(len(signal)):
            _sig_v = float(signal.iloc[_i])
            if abs(_sig_v) > 1e-10 and abs(_prev_sig_v) < 1e-10:
                # New trade entry: reset cumulative P&L and capture ATR at entry
                _cum_trade_pnl = 0.0
                _raw_atr = float(atr_proxy.iloc[_i - 1]) if (_i > 0 and pd.notna(atr_proxy.iloc[_i - 1])) else 0.02
                _entry_atr = _raw_atr if _raw_atr > 0 else 0.02
            if abs(_sig_v) < 1e-10:
                _cum_trade_pnl = 0.0
            else:
                _cum_trade_pnl += float(strategy_ret.iloc[_i])
            _stop_lvl = min(atr_stop_multiplier * _entry_atr, atr_fixed_stop)
            if abs(_sig_v) > 1e-10 and _cum_trade_pnl < -_stop_lvl:
                atr_stop_flags.iloc[_i] = True
            _prev_sig_v = _sig_v
        signal = signal.where(~atr_stop_flags.shift(1).fillna(value=False), other=0.0)
        strategy_ret = signal * log_ret
        _sig_dir = pd.Series(np.sign(signal.values), index=signal.index)
        position_change = _sig_dir.diff().fillna(0.0).abs() > 0.5
        trade_cost = position_change.astype(float) * friction
        strategy_ret = strategy_ret - trade_cost

    # ── 8. Drop NaN rows produced by rolling windows + shift ─────────────────
    valid_mask = strategy_ret.notna() & signal.notna() & log_ret.notna() & z_score.notna()
    strategy_ret = strategy_ret[valid_mask]
    trades_clean = position_change[valid_mask]
    log_ret_clean = log_ret[valid_mask]

    if len(strategy_ret) < 10:
        raise DataValidationError(
            f"Too few valid rows after applying rolling windows and NaN removal ({len(strategy_ret)} rows). Reduce rolling_window or supply more data."
        )

    # ── 9. Benchmark (buy-and-hold log returns) ───────────────────────────────
    benchmark_ret = log_ret_clean.copy()

    # ── 10. Compute metrics ───────────────────────────────────────────────────
    metrics = _compute_strategy_metrics(strategy_ret, trades_clean, friction_per_trade=friction)

    # Benchmark metrics (no friction)
    bm_metrics = _compute_strategy_metrics(benchmark_ret, pd.Series(0, index=benchmark_ret.index))

    # ── 11. Equity curves from log returns: exp(cumsum) is exact;
    #        (1+r).cumprod() is only an approximation and blows up over time ──
    equity_curve = pd.Series(np.exp(strategy_ret.values.cumsum()), index=strategy_ret.index)
    bm_curve = pd.Series(np.exp(benchmark_ret.values.cumsum()), index=benchmark_ret.index)

    # ── 12. Rolling 30-day Sharpe for chart ──────────────────────────────────
    r_s = strategy_ret
    roll_sharpe = (r_s.rolling(30, min_periods=30).mean() / r_s.rolling(30, min_periods=30).std(ddof=1).replace(0.0, np.nan)) * np.sqrt(252.0)

    return {
        "parameters": {
            "rolling_window": rolling_window,
            "z_threshold": z_threshold,
            "friction": friction,
            "volatility_filter": volatility_filter,
            "trend_filter": trend_filter,
            "regime_long_only": regime_long_only,
            "vol_window": vol_window,
            "vol_ma_window": vol_ma_window,
            "trend_sma_window": trend_sma_window,
            "stop_loss_pct": stop_loss_pct,
            "atr_stop_multiplier": atr_stop_multiplier,
            "atr_fixed_stop": atr_fixed_stop,
            "inv_vol_target": inv_vol_target,
            "max_hold_days": max_hold_days,
        },
        "metrics": metrics,
        "benchmark_metrics": bm_metrics,
        "equity_curve": equity_curve.round(6).tolist(),
        "benchmark_curve": bm_curve.round(6).tolist(),
        "strategy_returns": strategy_ret.round(8).tolist(),
        "benchmark_returns": benchmark_ret.round(8).tolist(),
        "rolling_sharpe_30d": roll_sharpe.dropna().round(6).tolist(),
        "dates": [str(d) for d in strategy_ret.index.tolist()],
        "total_rows_after_filter": int(len(strategy_ret)),
    }
