# Scenario Planner

> **Macro-Driven Equity Risk Platform** — A production-grade financial analytics pipeline that ingests multi-source market data, processes it through a Medallion Architecture, and delivers quantitative risk insights via Monte Carlo simulation, sensitivity regression, and stress testing.

---

## Why This Project Exists

Traditional investment analysis treats equities and macro factors in silos. This platform answers a single question:

> *"If inflation spikes 10% or energy prices surge 20%, what happens to my portfolio returns?"*

Two key methodological choices underpin every calculation:

### Log Returns — The Senior Standard
Raw price differences are non-stationary and scale-dependent. **Log returns** `ln(P_t / P_{t-1})` are:
- **Additive over time** — portfolio aggregation is algebraically clean
- **Approximately normally distributed** — satisfying the assumptions of OLS regression
- **Scale-invariant** — comparable across AAPL at $180 and XOM at $110

### Monte Carlo via Geometric Brownian Motion
The simulation follows the GBM stochastic differential equation:

```
dS = S·μ·dt + S·σ·dW
```

Where `μ` is the drift (mean log-return), `σ` is historical volatility, and `dW` is a Wiener process increment. With 10,000 paths, the output is a full probability distribution of future prices — not a single point estimate. This is the industry standard used by quant desks at major institutions.

---

## Architecture: Medallion Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│   Yahoo Finance (Equities)  │  FRED (Macro)  │  World Bank      │
└──────────────┬──────────────┴────────┬────────┴──────┬──────────┘
               │                       │               │
               ▼                       ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│  🥉 BRONZE LAYER  (data/raw/)                                   │
│  • Parallel fetch via ThreadPoolExecutor                        │
│  • Retry logic (exponential back-off, max 4 retries)            │
│  • Raw Parquet storage with full metadata catalog               │
│  • Zero transformation — immutable audit trail                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  🥈 SILVER LAYER  (data/processed/)                             │
│  • Pandera schema validation (type, range, null enforcement)    │
│  • Winsorization at 1st/99th percentile (outlier dampening)     │
│  • Z-score standardization (μ=0, σ=1)                          │
│  • Forward-fill imputation for missing macro data               │
│  • ZSTD-compressed Parquet output                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  🥇 GOLD LAYER  (data/gold/)                                    │
│  • Master Feature Table: time-aligned join across all sources   │
│  • Log-return engineering per ticker                            │
│  • Parallel AnalysisSuite execution                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    ANALYSIS SUITE                        │   │
│  │  Monte Carlo GBM    │  ARIMA Forecasting                 │   │
│  │  Sensitivity OLS    │  Elasticity Coefficient            │   │
│  │  Correlation Matrix │  Lag Analysis                      │   │
│  │  Stress Testing     │  Auto-ML (PyCaret, optional)       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
scenario-planner/
├── src/
│   ├── main.py                  # Entry point
│   ├── Fetchers/
│   │   ├── ProjectConfig.py     # Environment-based configuration
│   │   ├── Factory.py           # Fetcher factory (Strategy pattern)
│   │   ├── FredFetcher.py       # FRED API client
│   │   ├── YFinanceFetcher.py   # Yahoo Finance client
│   │   └── WorldBankFetcher.py  # World Bank API client
│   ├── Medallion/
│   │   ├── MedallionPipeline.py # Pipeline orchestrator
│   │   ├── bronze.py            # Bronze ingestion layer
│   │   ├── silver/
│   │   │   ├── silver.py        # Silver transformation layer
│   │   │   └── schema.py        # Pandera schemas
│   │   └── gold/
│   │       ├── GoldLayer.py     # Master table builder + analysis runner
│   │       └── AnalysisSuite/
│   │           ├── monte_carlo.py
│   │           ├── forecasting.py
│   │           ├── sesnsitivity_reg.py
│   │           ├── elasticity.py
│   │           ├── correl_mtrx.py
│   │           ├── lag.py
│   │           ├── stress_test.py
│   │           └── auto_ml.py   # Optional: requires pycaret
│   ├── logger/                  # Structured logging & catalog
│   └── exceptions/              # Domain-specific exceptions
├── tests/                       # 32-test suite (pytest)
├── notebooks/
│   └── demo_analysis.ipynb      # Interactive results demo
├── data/                        # Ignored by Git — generated at runtime
├── .env.example                 # Template for required secrets
├── pyproject.toml               # Poetry dependency management
└── .github/workflows/ci.yml     # Automated CI (ruff + pytest)
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- A free [FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html)

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/scenario-planner.git
cd scenario-planner
poetry install
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and fill in your FRED_API_KEY
```

### 3. Run

```bash
# Sample mode (2 tickers, fast)
poetry run python src/main.py

# Full mode (set ENVIRONMENT=actual in .env)
```

### 4. Explore Results

Open `notebooks/demo_analysis.ipynb` in Jupyter to visualise Monte Carlo paths, the correlation matrix, and stress test impacts:

```bash
poetry run jupyter notebook notebooks/demo_analysis.ipynb
```

---

## Development

```bash
# Run tests
poetry run pytest

# Lint & auto-fix
poetry run ruff check . --fix

# Type check
poetry run mypy .

# Format
poetry run black .
```

---

## Configuration Reference

All configuration is driven by environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `FRED_API_KEY` | *(required)* | FRED API key |
| `ENVIRONMENT` | `sample` | `sample` (2 tickers) or `actual` (5 tickers) |
| `START_DATE` | `2016-01-01` | Historical data start |
| `END_DATE` | `2026-12-31` | Historical data end |
| `MAX_WORKERS` | `10` | Thread pool size for parallel fetching |
| `MAX_RETRIES` | `4` | Retry attempts per failed request |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Ingestion | `fredapi`, `yfinance`, `wbgapi` | Multi-source market & macro data |
| Validation | `pandera` | Schema enforcement at Silver layer |
| Numerics | `numpy`, `pandas`, `scipy` | Array operations, time-series joins |
| Statistics | `statsmodels` | ARIMA, OLS regression |
| ML | `scikit-learn` | Ridge regression |
| Caching | `diskcache` | Rate-limit-aware API response cache |
| Serialisation | `pyarrow` (ZSTD Parquet) | Columnar, compressed data storage |

---

## License

MIT © 2026 EvangelGK — see [LICENSE](LICENSE).
