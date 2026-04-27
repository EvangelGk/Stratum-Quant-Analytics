# CHECKPOINT 2 — Universe Pruning Report

**Generated:** 2026-04-28 00:31  
**Input universe:** 31 tickers  
**After pruning:** 17 tickers  

## Decision Table

| Ticker | Sharpe OOS | Calmar OOS | Mean IC | Avg Pair Corr | LOO Δ | Flags | Decision |
|--------|-----------|-----------|---------|---------------|-------|-------|----------|
| AAPL | +0.991 | +1.330 | -0.02345 | 0.3950 | +0.005 | negative_ic | **KEEP** |
| ABBV | +0.939 | +0.860 | -0.03440 | 0.2933 | +0.020 | negative_ic | **KEEP** |
| NVDA | +0.875 | +0.695 | -0.11318 | 0.3197 | +0.012 | negative_ic | **KEEP** |
| TSLA | +0.793 | +1.016 | -0.01621 | 0.2393 | +0.041 | negative_ic | **KEEP** |
| JNJ | +0.666 | +0.906 | -0.05743 | 0.3163 | +0.001 | negative_ic | **KEEP** |
| ORCL | +0.359 | +0.308 | -0.05902 | 0.3217 | +0.004 | negative_ic | **KEEP** |
| APD | +0.352 | +0.272 | -0.02105 | 0.3912 | -0.004 | negative_ic | **KEEP** |
| MS | +0.342 | +0.216 | -0.07104 | 0.4569 | +0.000 | negative_ic | **KEEP** |
| CVX | +0.292 | +0.261 | -0.09172 | 0.3717 | +0.004 | negative_ic | **KEEP** |
| CAT | +0.110 | +0.101 | -0.06489 | 0.3979 | +0.005 | negative_ic | **KEEP** |
| MSFT | +0.107 | +0.070 | -0.00143 | 0.4086 | -0.001 | negative_ic | **KEEP** |
| XOM | +0.074 | +0.064 | -0.10498 | 0.3360 | +0.010 | negative_ic | **KEEP** |
| KO | +0.037 | +0.019 | +0.01426 | 0.3456 | -0.004 | — | **KEEP** |
| HON | +0.031 | +0.020 | -0.05319 | 0.4461 | -0.009 | negative_ic | **KEEP** |
| GS | -0.037 | -0.024 | -0.07715 | 0.4418 | -0.003 | negative_sharpe, negative_ic | **DROP** |
| V | -0.057 | -0.041 | -0.01975 | 0.4523 | -0.016 | negative_sharpe, negative_ic | **DROP** |
| NEM | -0.097 | -0.064 | -0.06637 | 0.1189 | -0.000 | negative_sharpe, negative_ic | **DROP** |
| PG | -0.160 | -0.060 | +0.01354 | 0.3008 | +0.000 | negative_sharpe | **KEEP** |
| LIN | -0.322 | -0.214 | -0.02919 | 0.4339 | +0.002 | negative_sharpe, negative_ic | **DROP** |
| ADBE | -0.357 | -0.208 | -0.03267 | 0.3445 | -0.021 | negative_sharpe, negative_ic | **DROP** |
| LLY | -0.367 | -0.159 | +0.03945 | 0.2550 | +0.034 | negative_sharpe | **KEEP** |
| SLB | -0.430 | -0.230 | -0.02659 | 0.3093 | -0.002 | negative_sharpe, negative_ic | **DROP** |
| BAC | -0.477 | -0.201 | -0.04548 | 0.4368 | -0.026 | negative_sharpe, negative_ic | **DROP** |
| JPM | -0.489 | -0.192 | -0.03726 | 0.4494 | -0.021 | negative_sharpe, negative_ic | **DROP** |
| MA | -0.517 | -0.186 | -0.02641 | 0.4515 | -0.016 | negative_sharpe, negative_ic | **DROP** |
| AVGO | -0.529 | -0.299 | -0.04494 | 0.3419 | +0.006 | negative_sharpe, negative_ic | **DROP** |
| PFE | -0.571 | -0.280 | -0.02997 | 0.3010 | +0.007 | negative_sharpe, negative_ic | **DROP** |
| GE | -0.639 | -0.310 | -0.02080 | 0.3407 | -0.026 | negative_sharpe, negative_ic | **DROP** |
| AMGN | -0.640 | -0.221 | +0.01483 | 0.3233 | -0.005 | negative_sharpe | **KEEP** |
| UNH | -0.822 | -0.267 | -0.04120 | 0.2922 | +0.010 | negative_sharpe, negative_ic | **DROP** |
| WMT | -1.220 | -0.372 | -0.01915 | 0.2522 | +0.001 | negative_sharpe, negative_ic | **DROP** |

## Portfolio Sharpe Comparison

| Universe | N | Equal-Weight Sharpe (2020–2022) |
|----------|---|--------------------------------|
| Full (pre-prune) | 31 | +0.580 |
| Pruned           | 17 | +0.765 |

## Tickers Kept

AAPL, ABBV, AMGN, APD, CAT, CVX, HON, JNJ, KO, LLY, MS, MSFT, NVDA, ORCL, PG, TSLA, XOM

## Tickers Dropped

- **GS** — negative_sharpe, negative_ic  Sharpe=-0.037, IC=-0.07715, Corr=0.4418
- **V** — negative_sharpe, negative_ic  Sharpe=-0.057, IC=-0.01975, Corr=0.4523
- **NEM** — negative_sharpe, negative_ic  Sharpe=-0.097, IC=-0.06637, Corr=0.1189
- **LIN** — negative_sharpe, negative_ic  Sharpe=-0.322, IC=-0.02919, Corr=0.4339
- **ADBE** — negative_sharpe, negative_ic  Sharpe=-0.357, IC=-0.03267, Corr=0.3445
- **SLB** — negative_sharpe, negative_ic  Sharpe=-0.430, IC=-0.02659, Corr=0.3093
- **BAC** — negative_sharpe, negative_ic  Sharpe=-0.477, IC=-0.04548, Corr=0.4368
- **JPM** — negative_sharpe, negative_ic  Sharpe=-0.489, IC=-0.03726, Corr=0.4494
- **MA** — negative_sharpe, negative_ic  Sharpe=-0.517, IC=-0.02641, Corr=0.4515
- **AVGO** — negative_sharpe, negative_ic  Sharpe=-0.529, IC=-0.04494, Corr=0.3419
- **PFE** — negative_sharpe, negative_ic  Sharpe=-0.571, IC=-0.02997, Corr=0.3010
- **GE** — negative_sharpe, negative_ic  Sharpe=-0.639, IC=-0.02080, Corr=0.3407
- **UNH** — negative_sharpe, negative_ic  Sharpe=-0.822, IC=-0.04120, Corr=0.2922
- **WMT** — negative_sharpe, negative_ic  Sharpe=-1.220, IC=-0.01915, Corr=0.2522

---

## Notes
- Drop rule A: *negative_sharpe* **AND** *negative_ic* (both must fire).
- Drop rule B: *high_correlation* (avg pair corr > 0.80) **AND** *loo_drag* (removing improves portfolio Sharpe by >5 bps).
- Correlation threshold 0.80 is stricter than naive 0.85 to give Phase 4 more diversification room.
- LOO delta is computed on the **OOS** 2020–2022 equal-weight portfolio.

**CHECKPOINT 2** — Eva, please review and approve (or adjust thresholds) before Phase 4 begins.