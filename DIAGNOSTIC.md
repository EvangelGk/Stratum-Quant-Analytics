# DIAGNOSTIC.md — Portfolio Strategy Calibration Root-Cause Report
**Date:** 2026-04-27  
**Phase:** 0 — Read-Only Recon  
**Author:** Senior Quant / AI Diagnostic Agent  
**Status:** 🛑 CHECKPOINT 0 — Awaiting Eva's approval before any code change

---

## 1. Pipeline Diagram

```
yfinance / FRED / World Bank
          │
    ┌─────▼──────┐
    │  Bronze    │  Raw fetch, persisted to data/raw/<source>/*.parquet
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │  Silver    │  Standardise, impute, Pandera validate
    │            │  src/Medallion/silver/silver.py
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │  Gold      │  Denormalise → master_table.parquet
    │            │  src/Medallion/gold/GoldLayer.py::create_master_table()
    └─────┬──────┘
          │
    ┌─────▼───────────────────────────────────────────────────────────┐
    │  AnalysisSuite  (all run from GoldLayer.run_analyses())        │
    │  ├── lag.py            → lag_analysis()                        │
    │  ├── backtest.py       → backtest_pre2020_holdout()            │
    │  │                        portfolio_backtest()                  │
    │  ├── mixed_frequency.py→ prepare_supervised_frame()  ◄ KEY     │
    │  ├── governance.py     → governance_report()                   │
    │  ├── auto_ml.py        → auto_ml_regression()                  │
    │  └── [forecasting, elasticity, monte_carlo, stress_test, …]   │
    └─────────────────────────────────────────────────────────────────┘
          │
    output/<user_id>/analysis_results.json  +  UI/app.py (Streamlit)
```

**Entry points:**
- `src/main.py` — CLI / scheduled run
- `UI/streamlit_app.py` — Streamlit dashboard
- `run_optimizer_daily.bat` / `scripts/scheduler_batch.py` — daily batch

**Config hub:** `src/Fetchers/ProjectConfig.py` (dataclass, ~80 fields)  
**Signal generation:** `AnalysisSuite/mixed_frequency.py::prepare_supervised_frame()`  
**Weight computation:** `portfolio_backtest()` — equal-weight by default (`1/N`)  
**Backtest engine:** `backtest.py::_simulate_risk_managed_returns()` (vectorised)

---

## 2. Suspected Leakage Points

### L1 ⚠️ CRITICAL — Forward-return target bleeds across the train/test boundary
**File:** `src/Medallion/gold/AnalysisSuite/mixed_frequency.py`, function `_future_target_from_series()`, ~line 175–195

```python
# horizon = 21 (trading days)
future = numeric.shift(-1).rolling(window=horizon, min_periods=horizon).sum().shift(-(horizon - 1))
```

For `horizon=21`, `panel[target]` at row `t` equals the **sum of log-returns from t+1 through t+21**.  
When the train/test split is cut at `2020-01-01`, the **last 20 training rows** (≈ Dec 2019) have targets that extend INTO January 2020 — the test window. Ridge is fitted on these rows; the predictions during test evaluation therefore carry implicit knowledge of early-2020 returns.

**Severity:** High. Affects exactly `horizon − 1 = 20` boundary rows per split. With only 207 training rows, this is ~10% of the training set.

---

### L2 ⚠️ HIGH — Market-observable features forced through 45-day "publication lag"
**File:** `src/Medallion/gold/AnalysisSuite/mixed_frequency.py`, `build_stationary_panel()` ~line 238

```python
lag_days = max(MANDATORY_PUBLICATION_LAG_DAYS, int(macro_lag_days))  # = 45
lagged_series = base_series.shift(lag_days)
```

`MANDATORY_PUBLICATION_LAG_DAYS = 45` is hard-coded for ALL non-yfinance features.  
This correctly lags monthly CPI/unemployment (which have a ~4-6 week delay).  
**It incorrectly lags daily-observable market data:**  

| Feature          | Real pub. delay | Applied lag |
|------------------|-----------------|-------------|
| `vix_index`      | 0 days (real-time)  | 45 days |
| `us10y_treasury_yield` | 0 days   | 45 days |
| `fed_funds_rate` | 0 days (FOMC day, ~8x/yr) | 45 days |
| `consumer_sentiment` | ~30 days (University of Michigan, monthly) | 45 days |

VIX and the 10-year yield are known the instant the market closes. A 45-day stale VIX is not just uninformative — it is actively misleading during fast-moving regimes (e.g., Feb–Mar 2020).

**Effect:** Model "sees" VIX from 45 days ago when predicting today. During the COVID crash, the signal receives pre-crash VIX values long after the crash has begun, causing severe regime mismatch.

---

### L3 ⚠️ HIGH — OHLCV passed as macro features → 45-day-stale price features in the model
**File:** `src/Medallion/gold/AnalysisSuite/backtest.py`, `backtest_pre2020_holdout()`, ~line 802

The feature list stored in `output/default/backtest_2020.json` includes:
```
"open", "high", "low", "close", "adj_close", "volume"
```
None of these appear in `_TECH_NAMES`, so they are routed to `_macro_features` and then to `prepare_supervised_frame(..., macro_lag_days=45)`. The result is that `close` is transformed to its 45-day-lagged log-difference — a 45-day-old momentum signal. This duplicates information already carried by the engineered `return_60d` / `mom_12_1` features, adding redundant collinear inputs that inflate VIF and destabilise the Ridge solution.

---

### L4 ⚠️ MEDIUM — Entry-threshold optimised in-sample
**File:** `src/Medallion/gold/AnalysisSuite/backtest.py`, `_optimize_entry_threshold()`, ~line 630

The threshold grid (candidates `[0.00, 0.15, …, 1.00]`) is evaluated against a **subset of the training data (last 35%)** and then the best threshold is applied to the **test set**. Because the threshold search is calibrated on data the model was already fitted on, it is an in-sample optimisation that contributes to overfitting.

---

### L5 ⚠️ LOW — `rolling().mean()` without `closed='left'`
**File:** `src/Medallion/gold/AnalysisSuite/mixed_frequency.py`, `add_volatility_regime_feature()`, ~line 127

```python
rolling_vol = series.rolling(window=max(5, int(window)), min_periods=…).std()
```

Pandas `rolling()` is right-closed by default: the current observation is included in the window. This is correct for features that use PAST data, but if this series is later used to construct a label or signal without a `.shift(1)`, it includes same-day information. Not exploited as a stand-alone leakage source but is a latent risk.

---

## 3. Benchmark Assessment

**Per-ticker benchmark:** `benchmark_returns = actual_arr` (single ticker log-returns).  
**Portfolio benchmark:** Equal-weight average of the 30 tickers' actual returns.

**Problems:**
1. **Per-ticker IR is vs. long-only holding** — any day the strategy is flat or short, it "loses" against the benchmark. For a macro-driven strategy that correctly goes to cash, this definition of IR is punitive and meaningless. IR = −1.36 is partly an artefact of benchmarking a long/flat/short strategy against a long-only position.

2. **Portfolio benchmark is NOT SPY** — an equal-weight portfolio of 30 selected growth stocks (AAPL, NVDA, TSLA, …) has likely **outperformed SPY** over 2014-2026. Beating this benchmark is harder. The IR reported may be negative even if the strategy beats SPY, because the benchmark is inflated by survivorship/selection.

3. **Survivorship bias in universe** — the 30-ticker list appears to contain well-known current mega-caps. If chosen with hindsight (today's top names for a 2014-start backtest), the universe has embedded survivorship bias. **Confirmed as a known bias; requires explicit flagging.**

---

## 4. Position Sizing & Rebalancing

- **Per-ticker:** Inverse-volatility targeting (target σ = 25% p.a.) applied via `_vol_scale` in `_simulate_risk_managed_returns()`. Position = signal_z × vol_scale, capped at `vol_scale_cap = 2.0`.
- **Portfolio:** Equal weights (`1/N`). No portfolio-level risk management.
- **Rebalance:** Daily (signal recomputed each day on new Ridge prediction).
- **Costs:** `tx_cost = 0.0005` (5 bps round-trip) applied on direction changes only.
- **Execution lag:** `exec_pos = np.roll(desired_pos, 1)` — 1-day lag, correct.
- **Dual-SMA filter** (20/200-day): blocks longs in downtrend, shorts in uptrend, **flat in neutral zone**. `active_ratio ≈ 35.6%` → strategy is passive 64% of the time.

**Double vol-targeting problem:** the ticker-level vol-targeting (`vol_scale`) is applied BEFORE equal-weight portfolio aggregation. Tickers with recently-low vol will have larger individual positions, creating unintended concentration at the portfolio level.

---

## 5. Governance Model OOS R²

From `output/default/analysis_results.json`:
```
Governance OOS R²  = −1.271
Best baseline R²   = −0.392 (recent_mean_predictor)
```
The model is **worse than predicting the recent mean** out-of-sample. R² < −1 means prediction errors are larger than the variance of the target. The signal is not just noise — it is **anti-correlated** with the outcome.

---

## TOP 3 HYPOTHESES FOR NEGATIVE METRICS

### H1 — **The signal is anti-predictive** (PRIMARY)
**Confidence: HIGH**

Governance OOS R² = −1.27 and raw R² = −1.27 (no floor). The Ridge/Lasso model fitted on macro features + 45-day-stale VIX/yield data has learned a relationship in-sample (2015–2023) that inverts out-of-sample. The model predicts positive returns when the market falls, and vice versa. Every metric will be negative if you run this signal unmodified, because the strategy systematically takes the wrong side.

**Experiment to confirm:** Compute Information Coefficient = `rank_corr(signal_t, forward_return_t+1)`. Plot 60-day rolling IC. If mean IC < 0 and t-stat < −2, the signal is anti-correlated and confirmed as the root cause.

---

### H2 — **Wrong publication lag on real-time market features** (AMPLIFIER)
**Confidence: HIGH**

VIX, 10-year yield, and fed funds rate are observable daily with zero lag. Applying MANDATORY_PUBLICATION_LAG_DAYS = 45 to them makes them 45-day-old market data, rendering them near-useless and potentially harmful during fast-moving regimes. This inflates the apparent importance of monthly macro series (CPI, UNRATE) which have legitimate lags, while making daily market signals stale.

**Experiment to confirm:** Run the governance model twice — once with the current 45-day lag for all FRED series, once with a per-series lag table (VIX=0, DGS10=0, FEDFUNDS=1 day, CPIAUCSL=45, UNRATE=45). Compare OOS R² and IC.

---

### H3 — **Dual-SMA filter kills too much market participation** (AMPLIFIER)
**Confidence: MEDIUM**

`active_ratio ≈ 35.6%`. The neutral zone (SMA20 and SMA200 not aligned) keeps the strategy flat 64% of the time. If the signal were positive IC, this would hurt Sharpe by reducing the number of bets. If the signal is negative IC (H1), the filter actually prevents losses — which means with the current inverted signal, removing the filter would make metrics WORSE. The filter is masking the severity of H1.

**Experiment to confirm:** Remove the dual-SMA filter (set all signals through). If Sharpe goes further negative, the filter is partially protecting the strategy from its bad signal, and H1 is the dominant cause.

---

## SUMMARY TABLE

| ID | Issue | Severity | Actionable Fix |
|----|-------|----------|----------------|
| L1 | 21-day target bleeds into test boundary | HIGH | Trim last `horizon-1` rows from each training fold |
| L2 | VIX/Yield/FedFunds lagged 45 days wrongly | HIGH | Per-series publication lag table |
| L3 | OHLCV passed as macro features | MEDIUM | Add OHLCV to `_TECH_NAMES` or exclude from feature list |
| L4 | Threshold optimised in-sample | MEDIUM | Separate dedicated validation fold; fix threshold before test |
| L5 | rolling() without closed='left' | LOW | Add `.shift(1)` after rolling for feature construction |
| B1 | IR benchmark is long-only single stock | HIGH | Use SPY or equal-weight universe as benchmark |
| B2 | Universe survivorship bias | KNOWN | Acknowledge and document |
| H1 | Signal is anti-predictive (R²=−1.27) | CRITICAL | Redesign alpha; measure IC before fitting model |
| H2 | Wrong lag on real-time market data | HIGH | Per-series lag table |
| H3 | Dual-SMA filter too restrictive | MEDIUM | Address after fixing H1; may be protective currently |

---

🛑 **CHECKPOINT 0** — Eva, please review this document before any code is changed.

**Key question for Eva:**
1. Can you confirm the 30-ticker list and whether it was chosen with hindsight?
2. The stored `backtest_2020.json` shows Sharpe = +0.906 for AAPL. The "current portfolio metrics" in the brief show Sharpe = −0.979. Are these from the same run? If not, which run produced the problem metrics?
3. Should daily-observable market series (VIX, DGS10, FEDFUNDS) be treated as zero-lag or 1-day-lag features?
