# Scenario Planner

> **Macro-Driven Equity Risk Platform** — A production-grade financial analytics pipeline that ingests multi-source market data, processes it through a Medallion Architecture, and delivers quantitative risk insights via Monte Carlo simulation, sensitivity regression, and stress testing.

---

## Why This Project Exists

Traditional investment analysis treats equities and macro factors in silos. This platform answers a single question:

> *"If inflation spikes 10% or energy prices surge 20%, what happens to my portfolio returns?"*

Methodological discipline is the core design principle: explicit assumptions, reproducible transforms, and auditable outputs at every layer.

## Methodology

The platform follows a structured quantitative methodology aligned with professional risk workflows:

1. Data provenance first: all raw ingestion is preserved in Bronze with immutable artifacts.
2. Contract-driven validation: Silver enforces schema, null, and outlier guardrails before analysis.
3. Econometric integrity checks: governance gates evaluate leakage risk, shift, walk-forward stability, and model-risk scoring.
4. Risk in distributions, not points: Monte Carlo and stress outputs are interpreted as uncertainty bands.
5. Human-in-the-loop remediation: Quantos can propose code changes, but application is approval-gated.

Two key quantitative choices underpin every calculation:

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
│  🥉 BRONZE LAYER  (data/users/<user_id>/raw/)                   │
│  • Parallel fetch via ThreadPoolExecutor                        │
│  • Retry logic (exponential back-off, max 4 retries)            │
│  • Raw Parquet storage with full metadata catalog               │
│  • Zero transformation — immutable audit trail                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  🥈 SILVER LAYER  (data/users/<user_id>/processed/)             │
│  • Pandera schema validation (type, range, null enforcement)    │
│  • Winsorization at 1st/99th percentile (outlier dampening)     │
│  • Z-score standardization (μ=0, σ=1)                          │
│  • Forward-fill imputation for missing macro data               │
│  • ZSTD-compressed Parquet output                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  🥇 GOLD LAYER  (data/users/<user_id>/gold/)                    │
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
├── UI/
│   ├── streamlit_app.py        # Main Streamlit entrypoint
│   ├── constants.py            # Shared UI paths, roles, and stage metadata
│   └── helpers.py              # Shared UI utilities
├── Auditor.py                  # Independent system auditor
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
│   │           ├── sensitivity_reg.py
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

For hosted/secure deployments, prefer `.streamlit/secrets.toml` for all API/access keys
(`FRED_API_KEY`, `GEMINI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) and avoid
putting real secrets in source-controlled files.

### 3. Run

```bash
# Default mode is ACTUAL (5 tickers)
poetry run python src/main.py

# Optional: set ENVIRONMENT=sample only if you explicitly want a lightweight run

# Streamlit command center
poetry run streamlit run UI/streamlit_app.py
```

### 4. Explore Results

Open `notebooks/demo_analysis.ipynb` in Jupyter to visualise Monte Carlo paths, the correlation matrix, and stress test impacts:

```bash
poetry run jupyter notebook notebooks/demo_analysis.ipynb
```

The Streamlit UI now includes an `Auditor` tab. After every successful pipeline run,
the system automatically executes `Auditor.py`, stores the result under
`output/<user_id>/audit_report.json`, and surfaces the report in the UI.

---

## Public Release & Streamlit Deploy Checklist

Before switching the repository to public and deploying on Streamlit Cloud:

1. Confirm secrets are not tracked:
    - `.env` and `.streamlit/secrets.toml` must stay gitignored.
2. Add production secrets in Streamlit Cloud only:
    - `FRED_API_KEY`
    - `GEMINI_API_KEY`
    - optional: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
3. Ensure dependency manifests are present:
    - `requirements.txt` (Python dependencies)
    - `packages.txt` (system packages for Linux deployment)
4. Verify CI is green on your default branch before publishing.
5. If any key was ever exposed locally, rotate it before making repo public.

### Streamlit Cloud

1. Create app on Streamlit Cloud from this GitHub repo.
2. Set entrypoint to `UI/streamlit_app.py`.
3. Add secrets via App Settings -> Secrets.
4. Deploy and validate:
    - Sidebar API status is green for required keys.
    - Quantos Assistant loads without backend errors.

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
| `ENVIRONMENT` | `actual` | `actual` (5 tickers) by default; set `sample` (2 tickers) only if explicitly needed |
| `START_DATE` | `2016-01-01` | Historical data start |
| `END_DATE` | `2026-12-31` | Historical data end |
| `MAX_WORKERS` | `10` | Thread pool size for parallel fetching |
| `MAX_RETRIES` | `4` | Retry attempts per failed request |
| `RANDOM_SEED` | `42` | Deterministic seed for stochastic analyses |
| `ENFORCE_REPRODUCIBILITY` | `true` | Enforce deterministic policy where supported |
| `DATA_USER_ID` | `default` | Storage namespace for per-user run isolation |
| `SILVER_HARD_FAIL` | `true` | Stop pipeline when Silver guardrails fail |
| `SILVER_MIN_ROWS` | `10` | Minimum rows required per entity in Silver preflight |
| `SILVER_MIN_ROWS_RATIO` | `0.1` | Minimum observed/expected row ratio before hard-stop |
| `SILVER_BASE_NULL_THRESHOLD` | `30.0` | Base null threshold (%) for dynamic data quality gating |
| `SILVER_DYNAMIC_THRESHOLD_WINDOW` | `20` | Rolling history window for adaptive null thresholds |
| `SILVER_WARN_TO_FAIL_BUFFER` | `15.0` | Additional null-threshold buffer between warn and hard-fail |
| `SILVER_OUTLIER_WARNING_RATIO` | `0.1` | Warn when clipped outliers exceed this row ratio |
| `MACRO_SERIES_MAP` | *(built-in)* | Optional JSON override for FRED series registry |
| `WORLDBANK_INDICATOR_MAP` | *(built-in)* | Optional JSON override for World Bank indicator registry |
| `WORLDBANK_ECONOMIES` | `WLD` | Comma-separated ISO-3 economies for World Bank ingestion and controlled aggregation |
| `WORLDBANK_AGGREGATION_STRATEGY` | `mean` | Pre-merge economy aggregation rule: `mean`/`median`/`sum`/`last` |
| `AUTO_ML_ENABLED` | `false` | Enable Auto-ML analysis output |
| `GOVERNANCE_HARD_FAIL` | `true` | Block advanced analyses if governance gate fails |
| `GOVERNANCE_MIN_R2` | `0.0` | Minimum out-of-sample R2 threshold |
| `GOVERNANCE_MAX_NORMALIZED_SHIFT` | `2.5` | Maximum train/test normalized drift |
| `GOVERNANCE_MAX_LEAKAGE_FLAGS` | `1` | Maximum tolerated leakage flags |
| `GOVERNANCE_MIN_STATIONARY_RATIO` | `0.4` | Minimum stationary-series ratio |
| `GOVERNANCE_WALK_FORWARD_WINDOWS` | `4` | Walk-forward backtesting windows |
| `GOVERNANCE_MIN_WALK_FORWARD_R2` | `0.0` | Minimum walk-forward average R2 |
| `GOVERNANCE_UNSTABLE_WALK_FORWARD_FLOOR` | `-5.0` | Instability trigger floor for raw walk-forward R2 |
| `GOVERNANCE_CLIPPED_WALK_FORWARD_FLOOR` | `-2.0` | Clipping floor for robust walk-forward risk scoring |
| `GOVERNANCE_MAX_MODEL_RISK_SCORE` | `0.6` | Maximum composite model risk score |
| `GOVERNANCE_REGIME` | `normal` | Governance policy profile: `normal` / `stress` / `crisis` |
| `GOVERNANCE_MODEL_RISK_WARN_THRESHOLD` | `0.4` | Warning band threshold for model risk score |
| `GOVERNANCE_MODEL_RISK_FAIL_THRESHOLD` | `0.6` | Failure band threshold for model risk score |
| `AUDITOR_ALLOWED_GAP_DAYS` | `7` | Business-day gap threshold used by temporal continuity audit |
| `GIT_COMMIT_SHA` | `unversioned` | Optional code lineage identifier for run-contract logs |

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

## Audit And Quality Controls

- `Auditor.py` acts as an independent-but-integrated judge of the system output.
- It checks source breadth, output density, statistical plausibility, temporal continuity,
    threshold strictness, and governance soundness.
- Temporal continuity in the auditor now uses business-day logic, duplicate removal on
    `['date', 'ticker']`, and a configurable allowed gap threshold.
- The auditor is intentionally separate from the pipeline's core transformations,
    so it can critique the produced artifacts instead of inheriting the pipeline's assumptions.

## World Bank Coverage

- World Bank ingestion supports configurable multi-economy breadth via `WORLDBANK_ECONOMIES`.
- To avoid row explosion in Gold, aggregation is controlled before the merge step rather than
    joining every economy-row directly into the analytical master table.
- Auditor integration checks both indicator breadth and economy coverage, so World Bank is no
    longer judged only as a single global corner of the source.

---

## License

CC BY-NC-ND 4.0 © 2026 EvangelGK. All Rights Reserved. See [LICENSE](LICENSE).
