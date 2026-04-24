from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from exceptions.MedallionExceptions import AnalysisError, DataValidationError

from .mixed_frequency import prepare_supervised_frame


def _select_top_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    features: List[str],
    max_k: int = 10,
) -> List[str]:
    """Return up to max_k features ranked by |Pearson(feature, target)|.

    Near-zero-variance columns are always dropped first — they produce spurious
    correlations and inflate VIF, destabilising the downstream model.
    """
    # 1. Always remove near-constant columns regardless of max_k
    stds = X_train[features].std()
    floor = float(stds.max()) * 0.005
    alive = [f for f in features if float(stds.get(f, 0.0)) > floor]
    if not alive:
        alive = list(features)

    # 2. If budget not exceeded, return surviving features
    if len(alive) <= max_k:
        return alive

    # 3. Rank remaining by |Pearson| with target
    y_arr = np.asarray(y_train, dtype=float)
    scores: List[tuple[float, str]] = []
    for f in alive:
        try:
            r = float(np.corrcoef(X_train[f].to_numpy(dtype=float), y_arr)[0, 1])
            scores.append((abs(r) if np.isfinite(r) else 0.0, f))
        except Exception:
            scores.append((0.0, f))
    scores.sort(reverse=True)
    return [f for _, f in scores[:max_k]]


def _prune_by_vif(
    features: List[str],
    X_train: pd.DataFrame,
    max_vif: float = 5.0,
) -> List[str]:
    """Iteratively remove the highest-VIF feature until all VIF ≤ max_vif.

    Uses OLS R² via numpy lstsq — no external dependency.
    Stops early if fewer than (n_features + 5) rows remain, to avoid
    rank-deficient fits on small training windows.
    """
    remaining = list(features)
    while len(remaining) > 1:
        X_sub = X_train[remaining].dropna().values.astype(float)
        if X_sub.shape[0] < X_sub.shape[1] + 5:
            break
        vifs: Dict[str, float] = {}
        for i, f in enumerate(remaining):
            y = X_sub[:, i]
            X_rest = np.delete(X_sub, i, axis=1)
            A = np.column_stack([np.ones(len(y)), X_rest])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            y_hat = A @ coef
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))
            vifs[f] = 1.0 / max(1.0 - r2, 1e-6)
        worst_f = max(vifs, key=lambda k: vifs[k])
        if vifs[worst_f] <= max_vif:
            break
        remaining.remove(worst_f)
    return remaining


def _make_model(n_samples: int) -> RidgeCV:
    """RidgeCV with TimeSeriesSplit: always produces non-zero predictions.

    ElasticNetCV was discarded because with small training sets (≤250 rows)
    and a 21-day forward-return target it routinely over-regularises to
    near-zero coefficients, collapsing pred_std below 1e-8 and breaking the
    z-score pipeline.  RidgeCV never produces a degenerate zero-prediction
    model — it simply shrinks coefficients toward zero without zeroing them —
    and auto-selects the regularisation strength via TimeSeriesSplit CV.
    """
    n_splits = max(2, min(3, n_samples // 60))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return RidgeCV(
        alphas=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        cv=tscv,
    )


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


def _backtest_pre2020_holdout_legacy(
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
        # Regime filter: reduce counter-trend positions to 70% (not flat-zero).
        # A macro model with 45-day lag carries forward-looking economic information
        # that price momentum doesn't — cutting to 40% was discarding too much of the
        # signal.  70% preserves the direction while acknowledging trend context.
        signal_rm = np.where((signal_rm == 1.0) & (~_in_uptrend_rm), 0.70, signal_rm)
        signal_rm = np.where((signal_rm == -1.0) & _in_uptrend_rm, -0.70, signal_rm)

        # 2. Inverse Volatility Scaling — target 25% annualised vol, no leverage
        # Raised from 20% → 25% so the strategy captures more of the macro signal
        # when volatility is moderate (typical macro regime).
        _actual_pd_rm = pd.Series(actual_arr)
        _vol14_rm = _actual_pd_rm.rolling(14, min_periods=14).std(ddof=1)
        _ann_vol_rm = (_vol14_rm * np.sqrt(252.0)).shift(1).fillna(0.25)
        _ann_vol_rm = _ann_vol_rm.replace(0.0, 0.25)
        _vol_scale_rm = (0.25 / _ann_vol_rm).clip(lower=0.0, upper=1.0).values
        signal_rm = signal_rm * _vol_scale_rm

        # 3. Per-trade ATR stop — exit if cumulative trade loss > min(5×vol, 12%)
        # Widened from min(3×vol, 7%) to match the 45-day macro holding period.
        # A 7% hard stop on a macro trade that needs 45 days to play out is too tight.
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
            _stop_lvl_rm = min(5.0 * _entry_vol_rm, 0.12)
            if abs(signal_rm[_i_rm]) > 1e-10 and _cum_trade_rm < -_stop_lvl_rm:
                _atr_flags_rm[_i_rm] = True
            _prev_base_rm = _base_sig_rm
        _atr_shifted_rm = np.zeros(_min_len, dtype=bool)
        _atr_shifted_rm[1:] = _atr_flags_rm[:-1]
        signal_rm = np.where(_atr_shifted_rm, 0.0, signal_rm)

        # 4. Time-based exit — close after 50 days (one full signal cycle for 45-day lag)
        # 30 days was closing positions before the 45-day macro signal had time to resolve.
        _days_rm = 0
        _tex_flags_rm = np.zeros(_min_len, dtype=bool)
        for _i_rm in range(_min_len):
            if abs(signal_rm[_i_rm]) > 1e-10:
                _days_rm += 1
                if _days_rm >= 50:
                    _tex_flags_rm[_i_rm] = True
            else:
                _days_rm = 0
        _tex_shifted_rm = np.zeros(_min_len, dtype=bool)
        _tex_shifted_rm[1:] = _tex_flags_rm[:-1]
        signal_rm = np.where(_tex_shifted_rm, 0.0, signal_rm)

        # 5. Friction on direction changes (reduced from 0.001 to 0.0005)
        _pos_chg_rm = np.abs(np.diff(np.sign(signal_rm), prepend=0.0)) > 0.5
        _costs_rm = _pos_chg_rm.astype(float) * 0.0005

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


def _trend_mask_from_log_returns(log_returns: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    lr = pd.to_numeric(log_returns, errors="coerce").fillna(0.0).clip(-0.15, 0.15)
    px = pd.Series(np.exp(np.cumsum(lr.to_numpy(dtype=float))), index=lr.index)
    sma200 = px.rolling(200, min_periods=200).mean()
    test_px = px.reindex(dates, method="pad").astype(float)
    test_sma = sma200.reindex(dates, method="pad").astype(float)
    return (test_px > test_sma).fillna(False).to_numpy(dtype=bool)


def _dual_trend_masks_from_log_returns(
    log_returns: pd.Series,
    dates: pd.DatetimeIndex,
    fast_sma: int = 20,
    slow_sma: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (uptrend_mask, downtrend_mask) using a dual-SMA Golden/Death Cross filter.

    Uptrend  : price > fast_sma AND fast_sma > slow_sma  (both SMAs aligned up)
    Downtrend: price < fast_sma AND fast_sma < slow_sma  (both SMAs aligned down)
    Neutral  : neither condition — no directional trades taken.

    The fast SMA (default 20-day) detects regime changes weeks before the slow
    SMA (default 200-day) alone would, eliminating the 'stuck-long' zone that
    causes the large drawdowns of a single-SMA filter during trend reversals.
    """
    lr = pd.to_numeric(log_returns, errors="coerce").fillna(0.0).clip(-0.15, 0.15)
    px = pd.Series(np.exp(np.cumsum(lr.to_numpy(dtype=float))), index=lr.index)
    sf = px.rolling(fast_sma, min_periods=fast_sma).mean()
    ss = px.rolling(slow_sma, min_periods=slow_sma).mean()
    up = ((px > sf) & (sf > ss)).astype(bool)
    dn = ((px < sf) & (sf < ss)).astype(bool)
    up_r = up.reindex(dates, method="pad").fillna(False).to_numpy(dtype=bool)
    dn_r = dn.reindex(dates, method="pad").fillna(False).to_numpy(dtype=bool)
    return up_r, dn_r


def _simulate_risk_managed_returns(
    pred_z: np.ndarray,
    actual_arr: np.ndarray,
    in_uptrend: np.ndarray,
    entry_threshold: float,
    inv_vol_target: float = 0.20,
    atr_multiplier: float = 2.0,
    max_hold_days: int = 50,
    tx_cost: float = 0.0005,
    vol_scale_cap: float = 1.50,
    downtrend_arr: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(min(len(pred_z), len(actual_arr), len(in_uptrend)))
    if n <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    pred = np.asarray(pred_z[:n], dtype=float)
    actual = np.asarray(actual_arr[:n], dtype=float)
    trend = np.asarray(in_uptrend[:n], dtype=bool)
    # Dual SMA: explicit downtrend mask from the death-cross zone.
    # When None, fall back to single-SMA behaviour (~trend = downtrend).
    dn = np.asarray(downtrend_arr[:n], dtype=bool) if downtrend_arr is not None else (~trend)

    raw_signal = np.where(pred >= entry_threshold, 1.0, np.where(pred <= -entry_threshold, -1.0, 0.0))
    # Dual-SMA Golden/Death Cross filter:
    #   Longs only in confirmed uptrend (price>SMA20, SMA20>SMA200).
    #   Shorts only in confirmed downtrend (price<SMA20, SMA20<SMA200).
    #   Neutral zone (SMAs diverged) → flat on both sides.
    raw_signal = np.where((raw_signal > 0.0) & (~trend), 0.0, raw_signal)
    raw_signal = np.where((raw_signal < 0.0) & (~dn), 0.0, raw_signal)

    vol20 = pd.Series(actual).rolling(20, min_periods=20).std(ddof=1)
    ann_vol = (vol20 * np.sqrt(252.0)).shift(1).replace(0.0, np.nan).fillna(0.20)
    vol_scale = (inv_vol_target / ann_vol).clip(lower=0.10, upper=vol_scale_cap).to_numpy(dtype=float)
    desired_pos = raw_signal * vol_scale

    exec_pos = np.roll(desired_pos, 1)
    exec_pos[0] = 0.0

    atr_proxy = pd.Series(actual).rolling(14, min_periods=14).std(ddof=1).shift(1).fillna(0.02).to_numpy(dtype=float)
    stop_next = np.zeros(n, dtype=bool)
    days_in_trade = 0
    cum_trade = 0.0
    peak_trade = 0.0
    entry_atr = 0.02
    prev_dir = 0.0

    for i in range(n):
        if stop_next[i]:
            exec_pos[i] = 0.0
        pos = float(exec_pos[i])
        pos_dir = float(np.sign(pos))
        if abs(pos_dir) > 1e-10 and abs(prev_dir) <= 1e-10:
            days_in_trade = 0
            cum_trade = 0.0
            peak_trade = 0.0
            raw_atr = float(atr_proxy[i]) if np.isfinite(float(atr_proxy[i])) else 0.02
            entry_atr = max(raw_atr, 0.005)
        if abs(pos_dir) <= 1e-10:
            days_in_trade = 0
            cum_trade = 0.0
            peak_trade = 0.0
            prev_dir = 0.0
            continue
        trade_ret = pos * float(actual[i])
        cum_trade += trade_ret
        peak_trade = max(peak_trade, cum_trade)
        days_in_trade += 1

        trailing_breach = (peak_trade - cum_trade) > (atr_multiplier * entry_atr)
        hard_loss = cum_trade < -min(atr_multiplier * entry_atr, 0.08)
        timed_exit = max_hold_days > 0 and days_in_trade >= max_hold_days
        if (trailing_breach or hard_loss or timed_exit) and (i + 1) < n:
            stop_next[i + 1] = True
        prev_dir = pos_dir

    position_change = np.abs(np.diff(np.sign(exec_pos), prepend=0.0)) > 0.5
    costs = position_change.astype(float) * tx_cost
    strategy_returns = (exec_pos * actual) - costs
    return strategy_returns.astype(float), exec_pos.astype(float)


def _compute_basic_stats(strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> Dict[str, Any]:
    r = np.asarray(strategy_returns, dtype=float)
    b = np.asarray(benchmark_returns[: len(r)], dtype=float)
    if len(r) == 0:
        return {
            "expectancy": 0.0,
            "profit_factor": None,
            "sharpe": None,
            "mdd": 0.0,
            "ann_return": 0.0,
            "calmar": 0.0,
            "ir": None,
        }
    wins = r[r > 0.0]
    losses = r[r < 0.0]
    win_prob = float(len(wins) / len(r)) if len(r) else 0.0
    loss_prob = float(len(losses) / len(r)) if len(r) else 0.0
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss_abs = float(abs(np.mean(losses))) if len(losses) else 0.0
    expectancy = float((win_prob * avg_win) - (loss_prob * avg_loss_abs))

    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss_abs = float(abs(np.sum(losses))) if len(losses) else 0.0
    profit_factor = float(gross_profit / gross_loss_abs) if gross_loss_abs > 1e-12 else (None if gross_profit == 0.0 else float("inf"))

    stdev = float(np.std(r, ddof=1)) if len(r) > 1 else None
    sharpe = float(np.mean(r) / stdev * np.sqrt(252.0)) if stdev is not None and stdev > 1e-12 else None
    mdd = _max_drawdown_from_returns(r)
    ann_return = _annualized_return(r, periods_per_year=252)
    calmar = float(ann_return / max(abs(mdd), 0.01))
    active = r - b
    te_active = float(np.std(active, ddof=1)) if len(active) > 1 else None
    ir = float(np.mean(active) / te_active * np.sqrt(252.0)) if te_active is not None and te_active > 1e-12 else None
    return {
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "mdd": float(mdd),
        "ann_return": float(ann_return),
        "calmar": float(calmar),
        "ir": ir,
        "te_active": te_active,
        "wins": wins,
        "losses": losses,
        "win_prob": win_prob,
        "loss_prob": loss_prob,
        "avg_win": avg_win,
        "avg_loss_abs": avg_loss_abs,
    }


def _optimize_entry_threshold(
    pred_z_train: np.ndarray,
    actual_train: np.ndarray,
    trend_train: np.ndarray,
    downtrend_train: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    # Finer grid in the 0.35–0.75 zone where macro-lag strategies tend to peak.
    candidates = [0.00, 0.20, 0.35, 0.45, 0.55, 0.65, 0.75, 0.95, 1.15]
    best = {"threshold": 0.55, "score": -1e9, "stats": {}}
    bench = np.asarray(actual_train, dtype=float)

    # Temporal validation fold: last 40% of training data.
    # Raised from 25% (val_split=0.75) to 40% (val_split=0.60) because the
    # dual-SMA filter keeps only ~35% of days active.  At 25% we had ~9 active
    # validation days — far too few for reliable threshold selection.  At 40%
    # we get ~29 active days, which is the practical minimum for stable scoring.
    # max(80, ...) ensures at least 80 samples remain in the training fold.
    n_train = len(pred_z_train)
    val_split = max(80, int(n_train * 0.60))
    pred_z_val = pred_z_train[val_split:]
    actual_val = actual_train[val_split:]
    trend_val = trend_train[val_split:]
    dn_val = downtrend_train[val_split:] if downtrend_train is not None else None

    for th in candidates:
        strat_ret, positions = _simulate_risk_managed_returns(
            pred_z=pred_z_train,
            actual_arr=actual_train,
            in_uptrend=trend_train,
            entry_threshold=float(th),
            inv_vol_target=0.20,
            atr_multiplier=2.0,
            max_hold_days=50,
            vol_scale_cap=1.50,
            downtrend_arr=downtrend_train,
        )
        train_stats = _compute_basic_stats(strat_ret, bench[: len(strat_ret)])

        val_ret, val_pos = _simulate_risk_managed_returns(
            pred_z=pred_z_val,
            actual_arr=actual_val,
            in_uptrend=trend_val,
            entry_threshold=float(th),
            inv_vol_target=0.20,
            atr_multiplier=2.0,
            max_hold_days=50,
            vol_scale_cap=1.50,
            downtrend_arr=dn_val,
        )
        val_stats = _compute_basic_stats(val_ret, actual_val[: len(val_ret)])

        # Directional accuracy: % of active days where prediction sign matches return sign.
        # Directly measures signal quality — more reliable than active_ratio reward,
        # which previously encouraged over-trading on noisy macro signals.
        _val_active = np.abs(np.asarray(val_pos, dtype=float)) > 1e-10
        if _val_active.any():
            _p = np.sign(np.asarray(pred_z_val[: len(_val_active)], dtype=float))
            _a = np.sign(np.asarray(actual_val[: len(_val_active)], dtype=float))
            dir_acc = float(np.mean((_p == _a)[_val_active]))
        else:
            dir_acc = 0.5

        train_sharpe = float(train_stats.get("sharpe") or -2.0) if isinstance(train_stats.get("sharpe"), (int, float)) else -2.0
        val_sharpe = float(val_stats.get("sharpe") or -2.0) if isinstance(val_stats.get("sharpe"), (int, float)) else -2.0
        val_pf_raw = val_stats.get("profit_factor")
        val_pf = float(val_pf_raw) if isinstance(val_pf_raw, (int, float)) and np.isfinite(float(val_pf_raw)) else 0.0
        val_mdd = abs(float(val_stats.get("mdd", 0.0)))
        val_expectancy = float(val_stats.get("expectancy", 0.0))

        # 70% validation fold + 30% full training.
        combined_sharpe = max(val_sharpe, -2.0) * 0.7 + max(train_sharpe, -2.0) * 0.3
        score = (
            (val_expectancy * 2000.0)
            + (max(val_pf - 1.0, 0.0) * 30.0)
            + (max(combined_sharpe, 0.0) * 30.0)
            - (val_mdd * 50.0)
            + (max(dir_acc - 0.5, 0.0) * 150.0)
        )
        if score > float(best["score"]):
            best = {"threshold": float(th), "score": float(score), "stats": train_stats}
    return best


def _walk_forward_validation(
    panel: pd.DataFrame,
    features: List[str],
    target: str,
    date_col: str,
    orig_log_returns: pd.Series,
    windows: int = 4,
) -> Dict[str, Any]:
    panel = panel.sort_values(date_col).reset_index(drop=True)
    total = len(panel)
    min_train = max(120, len(features) * 20)
    if total < (min_train + 80):
        return {
            "status": "insufficient_data",
            "windows_requested": int(windows),
            "windows_completed": 0,
            "folds": [],
        }

    windows = max(2, int(windows))
    test_size = max(30, int((total - min_train) / windows))
    folds: List[Dict[str, Any]] = []
    start = min_train
    for _ in range(windows):
        end = min(total, start + test_size)
        if end - start < 20:
            break
        train_df = panel.iloc[:start].copy()
        test_df = panel.iloc[start:end].copy()
        _wf_sel = _select_top_features(train_df[features], train_df[target].to_numpy(), features)
        try:
            model = _make_model(len(train_df))
            model.fit(train_df[_wf_sel], train_df[target])
        except Exception:
            model = Ridge(alpha=1.0)
            model.fit(train_df[_wf_sel], train_df[target])
        pred_train = model.predict(train_df[_wf_sel])
        pred_test = model.predict(test_df[_wf_sel])
        pred_mu = float(np.mean(pred_train))
        pred_std = float(np.std(pred_train, ddof=1)) if len(pred_train) > 1 else 1.0
        pred_std = pred_std if pred_std > 1e-8 else 1.0
        pred_z_train = (np.asarray(pred_train, dtype=float) - pred_mu) / pred_std
        pred_z_test = (np.asarray(pred_test, dtype=float) - pred_mu) / pred_std

        train_dates = pd.to_datetime(train_df[date_col].values)
        test_dates = pd.to_datetime(test_df[date_col].values)
        actual_train = np.nan_to_num(np.asarray(orig_log_returns.reindex(train_dates), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        actual_test = np.nan_to_num(np.asarray(orig_log_returns.reindex(test_dates), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        actual_train = np.clip(actual_train, -0.15, 0.15)
        actual_test = np.clip(actual_test, -0.15, 0.15)
        trend_train, dn_train = _dual_trend_masks_from_log_returns(orig_log_returns, train_dates)
        trend_test, dn_test = _dual_trend_masks_from_log_returns(orig_log_returns, test_dates)

        tuned = _optimize_entry_threshold(pred_z_train, actual_train, trend_train, downtrend_train=dn_train)
        th = float(tuned.get("threshold", 0.75))
        strategy_test, _ = _simulate_risk_managed_returns(
            pred_z_test, actual_test, trend_test,
            entry_threshold=th,
            inv_vol_target=0.20,
            atr_multiplier=2.0,
            vol_scale_cap=1.50,
            downtrend_arr=dn_test,
        )
        stats = _compute_basic_stats(strategy_test, actual_test[: len(strategy_test)])
        corr_r, corr_p = None, None
        if len(actual_test) >= 3:
            try:
                corr_r, corr_p = pearsonr(np.asarray(pred_test, dtype=float)[: len(actual_test)], actual_test)
                corr_r = float(corr_r)
                corr_p = float(corr_p)
            except Exception:
                corr_r, corr_p = None, None

        folds.append(
            {
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "entry_threshold": round(th, 4),
                "expectancy_per_trade": round(float(stats.get("expectancy", 0.0)), 8),
                "profit_factor": (
                    round(float(stats["profit_factor"]), 8)
                    if isinstance(stats.get("profit_factor"), (float, int)) and np.isfinite(float(stats["profit_factor"]))
                    else None
                ),
                "sharpe_ratio": round(float(stats["sharpe"]), 8) if isinstance(stats.get("sharpe"), (float, int)) else None,
                "calmar_ratio": round(float(stats["calmar"]), 8) if isinstance(stats.get("calmar"), (float, int)) else None,
                "maximum_drawdown": round(float(stats.get("mdd", 0.0)), 8),
                "pearson_r": round(float(corr_r), 8) if corr_r is not None else None,
                "p_value": round(float(corr_p), 10) if corr_p is not None else None,
            }
        )
        start = end
        if end >= total:
            break

    if not folds:
        return {
            "status": "insufficient_data",
            "windows_requested": int(windows),
            "windows_completed": 0,
            "folds": [],
        }

    sharpe_vals = [f["sharpe_ratio"] for f in folds if isinstance(f.get("sharpe_ratio"), (float, int))]
    calmar_vals = [f["calmar_ratio"] for f in folds if isinstance(f.get("calmar_ratio"), (float, int))]
    mdd_vals = [f["maximum_drawdown"] for f in folds if isinstance(f.get("maximum_drawdown"), (float, int))]
    pearson_vals = [f["pearson_r"] for f in folds if isinstance(f.get("pearson_r"), (float, int))]
    p_vals = [f["p_value"] for f in folds if isinstance(f.get("p_value"), (float, int))]

    return {
        "status": "ok",
        "windows_requested": int(windows),
        "windows_completed": int(len(folds)),
        "folds": folds,
        "avg_sharpe_ratio": round(float(np.mean(sharpe_vals)), 8) if sharpe_vals else None,
        "avg_calmar_ratio": round(float(np.mean(calmar_vals)), 8) if calmar_vals else None,
        "worst_max_drawdown": round(float(np.min(mdd_vals)), 8) if mdd_vals else None,
        "positive_pearson_ratio": round(float(np.mean([1.0 if x > 0.0 else 0.0 for x in pearson_vals])), 8) if pearson_vals else None,
        "pvalue_lt_0_05_ratio": round(float(np.mean([1.0 if x < 0.05 else 0.0 for x in p_vals])), 8) if p_vals else None,
    }


def backtest_pre2020_holdout(
    df: pd.DataFrame,
    target: str = "log_return",
    features: Optional[List[str]] = None,
    date_col: str = "date",
    ticker: Optional[str] = None,
) -> Dict[str, Any]:
    """Train before 2020 and evaluate on 2020-2022 holdout window."""
    try:
        features = list(features) if features else ["inflation", "energy_index"]

        # ── Inject 60-day momentum feature (Problem 4 fix) ───────────────────
        # return_60d = trailing 60-day cumulative log-return.
        # Named "return_60d" so is_return_like() returns True and
        # stationary_transform keeps the value as-is (no further differencing).
        # This gives Ridge a stock-specific momentum signal that macro features
        # alone cannot capture.
        _df_aug = df.copy()
        if "log_return" in _df_aug.columns:
            _df_aug[date_col] = pd.to_datetime(_df_aug[date_col], errors="coerce")
            if "ticker" in _df_aug.columns:
                _df_aug["return_60d"] = (
                    _df_aug.sort_values([date_col])
                    .groupby("ticker", group_keys=False)["log_return"]
                    .transform(lambda s: s.rolling(60, min_periods=30).sum())
                )
            else:
                _df_aug = _df_aug.sort_values(date_col)
                _df_aug["return_60d"] = _df_aug["log_return"].rolling(60, min_periods=30).sum()
            if "return_60d" not in features:
                features = features + ["return_60d"]
        else:
            _df_aug = df

        panel, metadata = prepare_supervised_frame(
            df=_df_aug,
            target=target,
            features=features,
            date_col=date_col,
            ticker=ticker,
            macro_lag_days=45,
            align_target_to_features=True,
            # Pin the prediction horizon to exactly 21 trading days (~1 month).
            # Without this cap, WorldBank features (252-day horizon) silently push
            # the target to 1-year forward returns, which is not what we want.
            min_target_horizon_days=21,
            max_target_horizon_days=21,
        )
        if panel.empty or date_col not in panel.columns:
            raise DataValidationError("No aligned rows available for backtest.")

        panel[date_col] = pd.to_datetime(panel[date_col], errors="coerce")
        panel = panel.dropna(subset=[date_col]).sort_values(date_col)

        train_mask = panel[date_col] < pd.Timestamp("2020-01-01")
        test_mask = (panel[date_col] >= pd.Timestamp("2020-01-01")) & (panel[date_col] <= pd.Timestamp("2022-12-31"))
        train_df = panel.loc[train_mask].copy()
        test_df = panel.loc[test_mask].copy()

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

        # cap at 7 (down from 10) to limit overfitting on the small pre-2020 window.
        _sel = _select_top_features(train_df[features], train_df[target].to_numpy(), features, max_k=7)
        # VIF pruning: iteratively remove the highest-VIF feature until all ≤ 5.0.
        # Addresses multicollinearity (Problem 3) and factor concentration (Problem 2).
        _sel = _prune_by_vif(_sel, train_df, max_vif=5.0)
        try:
            model = _make_model(len(train_df))
            model.fit(train_df[_sel], train_df[target])
        except Exception:
            model = Ridge(alpha=1.0)
            model.fit(train_df[_sel], train_df[target])
        predictions = model.predict(test_df[_sel])
        train_predictions = model.predict(train_df[_sel])

        _orig = (
            _df_aug[[date_col, "log_return"]]
            .assign(**{date_col: lambda x: pd.to_datetime(x[date_col], errors="coerce")})
            .dropna(subset=[date_col, "log_return"])
            .set_index(date_col)
        )
        _train_dates = pd.to_datetime(train_df[date_col].values)
        _test_dates = pd.to_datetime(test_df[date_col].values)
        _raw_train_lr = _orig["log_return"].reindex(_train_dates)
        _raw_test_lr = _orig["log_return"].reindex(_test_dates)
        train_actual_arr = np.nan_to_num(np.asarray(_raw_train_lr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        actual_arr = np.nan_to_num(np.asarray(_raw_test_lr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        train_actual_arr = np.clip(train_actual_arr, -0.15, 0.15)
        actual_arr = np.clip(actual_arr, -0.15, 0.15)

        _min_train_len = min(len(train_predictions), len(train_actual_arr))
        _min_test_len = min(len(predictions), len(actual_arr))
        train_predictions = np.asarray(train_predictions[:_min_train_len], dtype=float)
        train_actual_arr = np.asarray(train_actual_arr[:_min_train_len], dtype=float)
        _train_dates = _train_dates[:_min_train_len]
        predictions = np.asarray(predictions[:_min_test_len], dtype=float)
        actual_arr = np.asarray(actual_arr[:_min_test_len], dtype=float)
        _test_dates = _test_dates[:_min_test_len]
        if _min_train_len < 30 or _min_test_len < 20:
            raise DataValidationError("Insufficient aligned train/test rows for robust backtest execution.")

        pred_mu = float(np.mean(train_predictions))
        pred_std = float(np.std(train_predictions, ddof=1)) if len(train_predictions) > 1 else 1.0
        pred_std = pred_std if pred_std > 1e-8 else 1.0
        pred_z_train = (train_predictions - pred_mu) / pred_std
        pred_z_test = (predictions - pred_mu) / pred_std

        trend_train, dn_train = _dual_trend_masks_from_log_returns(_orig["log_return"], _train_dates)
        trend_test, dn_test = _dual_trend_masks_from_log_returns(_orig["log_return"], _test_dates)
        threshold_pick = _optimize_entry_threshold(
            pred_z_train, train_actual_arr, trend_train, downtrend_train=dn_train
        )
        selected_threshold = float(threshold_pick.get("threshold", 0.75))

        inv_vol_used = 0.20
        strategy_returns, _positions = _simulate_risk_managed_returns(
            pred_z=pred_z_test,
            actual_arr=actual_arr,
            in_uptrend=trend_test,
            entry_threshold=selected_threshold,
            inv_vol_target=inv_vol_used,
            atr_multiplier=2.0,
            max_hold_days=50,
            tx_cost=0.0005,
            vol_scale_cap=1.50,
            downtrend_arr=dn_test,
        )
        active_days = int(np.sum(np.abs(_positions) > 1e-10))
        min_active_days = max(20, int(0.03 * len(_positions)))
        if active_days < min_active_days:
            # Graduated fallback: try 0.20 before 0.0.
            # Going straight to 0.0 forces entry on every trend day with
            # near-random sign(pred) signals — the root cause of Sharpe≈0.15.
            # 0.20 preserves mild z-score filtering; 0.0 is the last resort.
            for _fb_th in [0.20, 0.0]:
                selected_threshold = _fb_th
                strategy_returns, _positions = _simulate_risk_managed_returns(
                    pred_z=pred_z_test,
                    actual_arr=actual_arr,
                    in_uptrend=trend_test,
                    entry_threshold=selected_threshold,
                    inv_vol_target=0.20,
                    atr_multiplier=2.0,
                    max_hold_days=50,
                    tx_cost=0.0005,
                    vol_scale_cap=1.50,
                    downtrend_arr=dn_test,
                )
                if int(np.sum(np.abs(_positions) > 1e-10)) >= min_active_days:
                    break
            inv_vol_used = 0.20
        benchmark_returns = actual_arr[: len(strategy_returns)]
        actual = pd.Series(benchmark_returns)
        predictions = predictions[: len(strategy_returns)]

        # ── 21-day price forecasts ───────────────────────────────────────────
        # predictions[i] = model's raw output ≈ 21-day forward cumulative log-return
        # predicted_close_21d = close_i × exp(predicted_log_return)
        _price_forecasts: List[Dict[str, Any]] = []
        _n_pred = len(predictions)
        _forecast_dates = _test_dates[:_n_pred]
        if "close" in _df_aug.columns:
            _close_orig = (
                _df_aug[[date_col, "close"]]
                .assign(**{date_col: lambda x: pd.to_datetime(x[date_col], errors="coerce")})
                .dropna(subset=[date_col, "close"])
                .set_index(date_col)
            )
            _test_close_series = _close_orig["close"].reindex(pd.DatetimeIndex(_forecast_dates))
            for _i in range(_n_pred):
                try:
                    _close_val = float(_test_close_series.iloc[_i])
                    _raw_pred = float(predictions[_i])
                    if np.isfinite(_close_val) and _close_val > 0 and np.isfinite(_raw_pred):
                        _price_forecasts.append({
                            "date": str(_forecast_dates[_i])[:10],
                            "current_close": round(_close_val, 4),
                            "predicted_21d_log_return": round(_raw_pred, 6),
                            "predicted_close_21d": round(_close_val * float(np.exp(_raw_pred)), 4),
                        })
                except Exception:
                    continue

        walk_forward = _walk_forward_validation(
            panel=panel,
            features=list(features),
            target=target,
            date_col=date_col,
            orig_log_returns=_orig["log_return"],
            windows=4,
        )

        te = _tracking_error(actual, predictions)
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
                corr_r, corr_p = pearsonr(np.asarray(predictions, dtype=float), benchmark_returns)
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

        wf_pos_ratio = walk_forward.get("positive_pearson_ratio") if isinstance(walk_forward, dict) else None
        wf_sig_ratio = walk_forward.get("pvalue_lt_0_05_ratio") if isinstance(walk_forward, dict) else None
        robust_signal = bool(
            (isinstance(corr_r, (int, float)) and float(corr_r) > 0.0)
            and (
                (isinstance(corr_p, (int, float)) and float(corr_p) < 0.05)
                or (isinstance(wf_pos_ratio, (int, float)) and float(wf_pos_ratio) >= 0.50)
            )
        )

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
            "features": list(_sel),
            "features_input": list(features),
            "strategy_parameters": {
                "entry_threshold_zscore": round(float(selected_threshold), 4),
                "atr_trailing_stop_multiplier": 2.0,
                "inverse_vol_target": inv_vol_used,
                "trend_filter": "dual_sma_20_200",
                "vol_scale_cap": 1.50,
                "execution_lag_days": 1,
                "active_days": int(np.sum(np.abs(_positions) > 1e-10)),
                "active_ratio": round(float(np.mean(np.abs(_positions) > 1e-10)) if len(_positions) else 0.0, 6),
            },
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
            "walk_forward_validation": walk_forward,
            "robustness_check": {
                "pearson_positive": bool(isinstance(corr_r, (int, float)) and float(corr_r) > 0.0),
                "p_value_lt_0_05": bool(isinstance(corr_p, (int, float)) and float(corr_p) < 0.05),
                "walk_forward_positive_pearson_ratio": (
                    round(float(wf_pos_ratio), 8) if isinstance(wf_pos_ratio, (int, float)) else None
                ),
                "walk_forward_significant_ratio": (
                    round(float(wf_sig_ratio), 8) if isinstance(wf_sig_ratio, (int, float)) else None
                ),
                "is_statistically_robust": robust_signal,
            },
            "trade_distribution_histogram": trade_hist,
            "strategy_returns": [float(v) for v in strategy_returns.tolist()],
            "benchmark_returns": [float(v) for v in benchmark_returns.tolist()],
            "test_dates": [str(d)[:10] for d in pd.to_datetime(_test_dates).tolist()],
            "price_forecasts_21d": _price_forecasts,
            "predictions": [float(v) for v in predictions.tolist()],
            "actual": [float(v) for v in actual.tolist()],
            "transformations": metadata,
        }
    except DataValidationError:
        raise
    except Exception as exc:
        raise AnalysisError(f"Unexpected error in backtest_pre2020_holdout: {exc}") from exc


def portfolio_backtest(
    df: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    target: str = "log_return",
    features: Optional[List[str]] = None,
    date_col: str = "date",
    mode: str = "both",
) -> Dict[str, Any]:
    """Run backtest across a portfolio of tickers.

    Parameters
    ----------
    df       : Master table with all tickers (must have a 'ticker' column).
    tickers  : Tickers to include. Defaults to all unique tickers in df.
    weights  : {ticker: weight}. Normalised internally. Defaults to equal weights.
    target   : Return column to predict (default 'log_return').
    features : Feature columns. None = auto-resolved inside each per-ticker call.
    date_col : Date column name (default 'date').
    mode     : 'portfolio' = combined metrics only;
               'individual' = per-ticker results only;
               'both' = combined + per-ticker (default).

    Returns
    -------
    Dict with:
      'mode', 'tickers_attempted', 'tickers_succeeded', 'failed_tickers',
      'weights', 'portfolio' (if mode != 'individual'),
      'per_ticker' (if mode != 'portfolio').
    """
    if "ticker" not in df.columns:
        raise AnalysisError("portfolio_backtest requires a 'ticker' column in df.")

    all_tickers: List[str] = tickers or sorted(df["ticker"].dropna().unique().tolist())
    if not all_tickers:
        raise AnalysisError("No tickers found in df for portfolio_backtest.")

    # ── Run per-ticker backtests ─────────────────────────────────────────────
    per_ticker_results: Dict[str, Any] = {}
    failed_tickers: List[str] = []
    for t in all_tickers:
        ticker_df = df[df["ticker"].astype(str) == str(t)].copy()
        if ticker_df.empty:
            failed_tickers.append(t)
            per_ticker_results[t] = {"status": "failed", "error": "no_data", "ticker": t}
            continue
        try:
            per_ticker_results[t] = backtest_pre2020_holdout(
                ticker_df,
                target=target,
                features=features,
                date_col=date_col,
                ticker=t,
            )
        except (AnalysisError, DataValidationError, Exception) as exc:
            failed_tickers.append(t)
            per_ticker_results[t] = {"status": "failed", "error": str(exc), "ticker": t}

    successful_tickers = [
        t for t in all_tickers
        if t in per_ticker_results and per_ticker_results[t].get("status") != "failed"
    ]

    result: Dict[str, Any] = {
        "mode": mode,
        "tickers_attempted": all_tickers,
        "tickers_succeeded": successful_tickers,
        "failed_tickers": failed_tickers,
    }

    if not successful_tickers:
        result.update({"weights": {}, "portfolio": None, "per_ticker": per_ticker_results, "status": "all_failed"})
        return result

    # ── Normalise weights ────────────────────────────────────────────────────
    raw_w = {t: float(weights.get(t, 0.0)) for t in successful_tickers} if weights else {t: 1.0 for t in successful_tickers}
    total_w = sum(raw_w.values()) or float(len(successful_tickers))
    norm_weights = {t: round(v / total_w, 8) for t, v in raw_w.items()}
    result["weights"] = norm_weights

    if mode != "portfolio":
        result["per_ticker"] = per_ticker_results

    if mode == "individual":
        return result

    # ── Align return series by date ──────────────────────────────────────────
    strat_frames: List[pd.DataFrame] = []
    bench_frames: List[pd.DataFrame] = []
    for t in successful_tickers:
        r = per_ticker_results[t]
        strat_rets = r.get("strategy_returns", [])
        bench_rets = r.get("benchmark_returns", [])
        dates = r.get("test_dates", [])
        n = len(strat_rets)
        if n == 0:
            continue
        if dates and len(dates) == n:
            idx = pd.to_datetime(dates)
        else:
            idx = pd.RangeIndex(n)
        strat_frames.append(pd.DataFrame({t: strat_rets}, index=idx))
        bench_frames.append(pd.DataFrame({t: bench_rets[:n]}, index=idx))

    if not strat_frames:
        result["portfolio"] = None
        return result

    strat_df = pd.concat(strat_frames, axis=1).sort_index().fillna(0.0)
    bench_df = pd.concat(bench_frames, axis=1).sort_index().fillna(0.0)

    port_ret = np.zeros(len(strat_df))
    port_bench = np.zeros(len(bench_df))
    for t in successful_tickers:
        w = norm_weights.get(t, 0.0)
        if t in strat_df.columns:
            port_ret += w * strat_df[t].values
        if t in bench_df.columns:
            port_bench += w * bench_df[t].values

    # ── Portfolio-level metrics ──────────────────────────────────────────────
    n = len(port_ret)
    ann_factor = np.sqrt(252.0)
    mdd = _max_drawdown_from_returns(port_ret)
    ann_ret = _annualized_return(port_ret)
    vol = float(np.std(port_ret, ddof=1) * ann_factor) if n > 1 else 0.0
    sharpe: Optional[float] = float(np.mean(port_ret) / np.std(port_ret, ddof=1) * ann_factor) if (n > 1 and np.std(port_ret, ddof=1) > 1e-12) else None
    calmar: Optional[float] = round(ann_ret / max(abs(mdd), 0.01), 6) if mdd != 0 else None

    downside = port_ret[port_ret < 0.0]
    down_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else None
    sortino: Optional[float] = float(np.mean(port_ret) / down_std * ann_factor) if (down_std and down_std > 1e-12) else None

    bench_ann_ret = _annualized_return(port_bench)
    bench_vol = float(np.std(port_bench, ddof=1) * ann_factor) if len(port_bench) > 1 else 0.0

    active = port_ret - port_bench[:n]
    te_val = float(np.std(active, ddof=1) * ann_factor) if len(active) > 1 else None
    ir: Optional[float] = float(np.mean(active) / np.std(active, ddof=1) * ann_factor) if (te_val and te_val > 1e-12) else None

    wins = port_ret[port_ret > 0.0]
    losses = port_ret[port_ret < 0.0]
    gross_profit = float(np.sum(wins))
    gross_loss = float(abs(np.sum(losses)))
    if gross_loss > 1e-12:
        pf: Any = round(gross_profit / gross_loss, 6)
    elif gross_profit > 0.0:
        pf = "inf"
    else:
        pf = None

    # Ticker-level summary for quick inspection (without full return arrays)
    ticker_summary = {}
    for t in successful_tickers:
        r = per_ticker_results[t]
        forecasts = r.get("price_forecasts_21d", [])
        latest_forecast = forecasts[-1] if forecasts else None
        ticker_summary[t] = {
            "sharpe_ratio": r.get("sharpe_ratio"),
            "annualized_return": r.get("annualized_return"),
            "maximum_drawdown": r.get("maximum_drawdown"),
            "calmar_ratio": r.get("calmar_ratio"),
            "information_ratio": r.get("information_ratio"),
            "weight": norm_weights.get(t, 0.0),
            "latest_price_forecast": latest_forecast,
        }

    result["portfolio"] = {
        "n_tickers": len(successful_tickers),
        "n_failed": len(failed_tickers),
        "annualized_return": round(ann_ret, 6),
        "benchmark_annualized_return": round(bench_ann_ret, 6),
        "annualized_volatility": round(vol, 6),
        "benchmark_volatility": round(bench_vol, 6),
        "sharpe_ratio": round(sharpe, 6) if sharpe is not None else None,
        "sortino_ratio": round(sortino, 6) if sortino is not None else None,
        "maximum_drawdown": round(mdd, 6),
        "calmar_ratio": calmar,
        "information_ratio": round(ir, 6) if ir is not None else None,
        "tracking_error": round(te_val, 6) if te_val is not None else None,
        "win_probability": round(float(len(wins) / n), 6) if n else 0.0,
        "average_win": round(float(np.mean(wins)), 6) if len(wins) else 0.0,
        "average_loss": round(float(np.mean(losses)), 6) if len(losses) else 0.0,
        "profit_factor": pf,
        "rolling_sharpe_30d": _rolling_sharpe(port_ret, window=30),
        "strategy_returns": port_ret.tolist(),
        "benchmark_returns": port_bench.tolist(),
        "dates": [str(d)[:10] for d in strat_df.index.tolist()],
        "ticker_summary": ticker_summary,
    }

    return result


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
