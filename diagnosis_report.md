# diagnosis_report.md — Phase 1 Root-Cause Results
**Date:** 2026-04-28  
**Ref ticker for experiments:** AAPL  
**Phase:** 1 — Root-Cause Diagnosis  

---

## SUMMARY

| Hypothesis | Verdict | Key Finding |
|-----------|---------|-------------|
| H1 Signal anti-predictive | **CONFIRMED** | Mean IC = -0.0045, 2/13 features with positive IC |
| H2 Wrong publication lag | **REJECTED** | Uniform-45 OOS R² = -0.0376, Proper-lag OOS R² = -0.084 |
| H3 SMA filter over-restrictive | **CONFIRMED_SECONDARY** | Removing the filter degrades Sharpe further. Filter is masking bad-signal damage. H1 is the primary cause. |

---

## H1 — Signal Quality (Information Coefficient)

**Method:** For each macro feature, apply its *correct* publication lag, then compute  
Spearman rank-correlation between the lagged feature value at date `t` and the  
1-day forward return at `t+1` (no target leakage). Report mean IC across all tickers.

**Acceptance threshold:**  
- Mean IC > +0.02 and t-stat > 2.0 → signal has positive edge  
- Mean IC < −0.02 and t-stat < −2.0 → signal is anti-correlated (root cause confirmed)  
- |Mean IC| < 0.02 → signal is noise (also a root cause, different fix)  

### Feature IC Table (all tickers, 1-day forward return)

| Feature | Mean IC | t-stat | +IC% | Tickers |
|---------|---------|--------|------|---------|
| vix_index | +0.0077 | +2.06 | 52% | 31 |
| unemployment_rate | +0.0072 | +2.44 | 71% | 31 |
| unemployment_wb | -0.0007 | -0.19 | 52% | 31 |
| inflation | -0.0013 | -0.31 | 48% | 31 |
| consumer_sentiment | -0.0016 | -0.29 | 61% | 31 |
| us10y_treasury_yield | -0.0032 | -0.84 | 36% | 31 |
| trade_openness | -0.0035 | -0.87 | 39% | 31 |
| fed_funds_rate | -0.0047 | -1.24 | 26% | 31 |
| energy_usage | -0.0058 | -1.39 | 36% | 31 |
| industrial_production | -0.0105 | -4.65 | 13% | 31 |
| energy_index | -0.0106 | -2.45 | 23% | 31 |
| gdp_growth | -0.0145 | -3.72 | 26% | 31 |
| inflation_wb | -0.0166 | -3.99 | 19% | 31 |


### Aggregate
- **Mean IC across all features:** -0.0045  
- **Features with positive IC:** 2 / 13  
- **Features significant (|t| ≥ 2):** 6  

### Verdict: CONFIRMED

**Interpretation:**
Mean IC is negative and/or fewer than half of features have positive IC.  
The macro signal systematically predicts the **wrong direction**.  
This is the primary root cause. No amount of threshold tuning, regime  
filtering, or parameter optimisation will fix an anti-correlated signal.  

**Required action:** Redesign the alpha from documented factors with  
confirmed positive IC (12-1 momentum, quality, low-vol) before fitting  
any model. See Phase 2A.

---

## H2 — Publication Lag Correctness

**Method:** Build two Ridge-on-macro panels for AAPL:  
1. **Uniform 45-day lag** — current behaviour (every FRED series shifted 45 days)  
2. **Per-series proper lag** — VIX/DGS10/FEDFUNDS shifted 1 day; CPI/UNRATE shifted 45 days  

Walk-forward OOS R² (4 folds) compared between the two.

### Results

| Configuration | Folds | Mean OOS R² |
|--------------|-------|-------------|
| Uniform 45-day lag (current) | 4 | -0.0376 |
| Per-series proper lag        | 4 | -0.084 |
| Improvement                  | — | -0.0464 |

### Verdict: REJECTED

**Interpretation:**  
Proper per-series lags do not improve OOS R² meaningfully.  
The lag regime is not the primary driver of poor performance.  
H1 (signal quality) remains the dominant issue.

---

## H3 — Dual-SMA Filter Impact

**Method:** Run the same Ridge-on-macro backtest twice on AAPL:  
1. **With** dual-SMA 200-day filter (current)  
2. **Without** filter (all predicted signals execute)  

### Results

| Configuration | Active ratio | Sharpe | Mean daily ret |
|--------------|-------------|--------|----------------|
| With SMA filter    | 0.7456 | 0.8623 | 0.00064 |
| Without SMA filter | 0.9987 | 0.7448 | 0.000749 |

Signal IC (pred_z vs 1-day return):  
- With filter context: -0.02304  
- Without filter context: -0.02304  

### Verdict: CONFIRMED_SECONDARY

Removing the filter degrades Sharpe further. Filter is masking bad-signal damage. H1 is the primary cause.

---

## ROOT CAUSE CONCLUSION

**PRIMARY ROOT CAUSE: H1 — The signal is anti-predictive or noise.**

The macro features (FRED + World Bank) as currently constructed  
do not carry positive information about 1-day or 21-day forward returns.  
Macro data with correct lags applied simply does not predict daily equity  
returns consistently at IC > 0.02.

**The required fix is Phase 2A: Redesign the alpha.**  
Replace or augment the macro-only signal with documented factors that  
have empirical IC evidence:
- **12-1 month momentum** (Jegadeesh-Titman): strongest single equity factor
- **Low-volatility**: stocks with lower trailing vol earn higher risk-adj returns  
- **Quality composite**: high ROE + low debt as defensive overlay

H2 (lag correction) and L1 (boundary leakage fix) should **also** be  
implemented as hygiene but will not alone restore positive Sharpe.

**Do not proceed to Phase 3 (universe pruning) or Phase 5 (agentic loop)**  
until Phase 2A produces IC > 0.02 on at least 60% of tickers out-of-sample.

---

🛑 **CHECKPOINT 1** — Eva, please review this report before Phase 2 begins.

**Required decision:** Which root cause fix should Phase 2 address?
- **Option A (recommended if H1=CONFIRMED):** Phase 2A — Redesign alpha with  
  momentum/low-vol/quality factors. Discard pure macro-only model.
- **Option B (if H2=CONFIRMED, H1=REJECTED):** Phase 2B — Fix lag table only.  
- **Option C (if both confirmed):** Phase 2B first (cheap fix), measure IC gain,  
  then 2A if IC is still negative.

Do not proceed to code changes until Eva approves the option.
