# Scenario Planner - Complete User Guide

## Application Overview

The Scenario Planner is a comprehensive financial data pipeline that implements the Medallion Architecture for scenario planning and risk analysis. It automatically fetches data from multiple sources, processes it through bronze/silver/gold layers, and performs advanced financial analyses.

## Quick Start

1. **Prerequisites**
   - Python 3.8+
   - FRED API key (free from https://fred.stlouisfed.org/docs/api/api_key.html)

2. **Setup**
   ```bash
   # Clone or download the project
   cd scenario-planner

   # Install dependencies
   poetry install

   # Create .env file with your API key and user namespace
   cp .env.example .env
   # then set FRED_API_KEY and DATA_USER_ID
   ```

3. **Run**
   ```bash
   poetry run python src/main.py

   # Launch the Streamlit command center
   poetry run streamlit run UI/streamlit_app.py
   ```

## What Happens Automatically

The application will:
- Fetch stock data from Yahoo Finance
- Download economic indicators from FRED
- Get global development data from World Bank
- Clean and validate all data
- Create analytical datasets
- Run comprehensive analyses
- Generate reports and save results

After each successful pipeline run, the application also:
- Runs `Auditor.py` automatically
- Writes `output/<user_id>/audit_report.json`
- Exposes the result in the `Auditor` tab inside the Streamlit UI

## Reproducibility and Governance Controls

- **Reproducibility policy**
   - `RANDOM_SEED` controls deterministic stochastic analyses.
   - `ENFORCE_REPRODUCIBILITY=true` enforces seed propagation where supported.

- **Governance hard-fail gate**
   - If `GOVERNANCE_HARD_FAIL=true`, advanced analyses are blocked when governance thresholds fail.
   - Gate checks include out-of-sample R2, walk-forward R2, leakage flags,
      stationarity ratio, normalized drift, and composite model risk score.
   - Regime profiles are configurable with `GOVERNANCE_REGIME` (`normal`, `stress`, `crisis`).
   - Model risk bands are labeled as `pass` / `warn` / `fail` based on score thresholds.

- **Run contract lineage**
   - Every run logs reproducibility and lineage hashes:
      config hash, data catalog hash, pyproject hash, and optional code version (`GIT_COMMIT_SHA`).

## Understanding the Output

### On Screen Messages
- **Progress Bars**: Show current processing status for each stage
- **Success Messages**: Confirm completion with data counts
- **Analysis Results**: Detailed explanations of what each analysis means
- **Error Messages**: Specific guidance when issues occur

### Generated Files
- `data/users/<user_id>/raw/`: Raw downloaded data files
- `data/users/<user_id>/processed/`: Cleaned and transformed data
- `data/users/<user_id>/gold/master_table.parquet`: Combined analytical dataset
- `output/<user_id>/`: Analysis results and reports
- `output/<user_id>/audit_report.json`: Independent system audit result
- `logs/`: Detailed operation logs and metrics

### Streamlit UI
- Launch command: `poetry run streamlit run UI/streamlit_app.py`
- Main tabs include health, analytics, governance, logs, and `Auditor`
- The `Auditor` tab lets you inspect whether outputs are decision-ready and whether
   governance or quality thresholds appear too strict for the current run

## Analysis Explanations

### 1. Correlation Matrix
Shows relationships between financial assets and macro factors.
- Values range from -1 (perfect negative) to +1 (perfect positive)
- Strong correlations (>0.7) indicate important risk factors
- Use to identify which economic variables affect your investments

### 2. Elasticity Analysis
Measures how much asset returns change with economic changes.
- Formula: % Change in Asset ÷ % Change in Economic Factor
- Values >1: Asset is very sensitive to economic changes
- Values <1: Asset is relatively stable
- Helps assess pricing power and market sensitivity

### 3. Lag Analysis
Shows delayed impacts of economic factors on returns.
- Lag 0: Immediate effect
- Higher lags: Delayed reactions (weeks/months)
- Identifies leading indicators for forecasting

### 4. Monte Carlo Simulation
Risk assessment through random scenario generation.
- Generates thousands of possible future price paths
- Based on historical volatility and returns
- Shows probability distributions of outcomes
- Essential for Value at Risk (VaR) calculations

### 5. Stress Testing
Extreme scenario analysis.
- Tests portfolio under severe economic shocks
- Shows potential losses in crisis conditions
- Helps with worst-case scenario planning

### 6. Sensitivity Regression
Multivariate analysis of factor impacts.
- Shows which economic variables drive returns
- Provides quantitative factor sensitivities
- Useful for factor investing and hedging

### 7. Time Series Forecasting
Future trend prediction using ARIMA models.
- Predicts future values based on historical patterns
- Includes confidence intervals
- Helps with trend analysis and expectations

### 8. Auto ML Regression
Automated predictive modeling.
- Tests multiple algorithms automatically
- Finds best model for your data
- Provides feature importance rankings

## Data Sources

### Yahoo Finance (YFinance)
- Stock prices, volumes, and fundamentals
- Historical data for equities and ETFs
- Real-time and historical market data

### Federal Reserve Economic Data (FRED)
- Economic indicators and statistics
- Inflation rates, unemployment, interest rates
- Government economic releases

### World Bank Open Data
- Global development indicators
- GDP, population, trade statistics
- Cross-country economic comparisons

## Troubleshooting

### Common Issues

**API Errors**
- Check your FRED API key in .env file
- Verify internet connection
- Some APIs have rate limits - wait and retry

**Memory Errors**
- Reduce date ranges in configuration
- Use sequential mode instead of parallel
- Close other memory-intensive applications

**Data Quality Issues**
- Check raw data files first
- Review quality reports in processed/
- Some data sources may have gaps

**Slow Performance**
- Use parallel processing (default)
- Reduce analysis iterations
- Check system resources

### Error Messages Guide

- **FetcherError**: API or network issue - check keys and connection
- **DataValidationError**: Data format problem - check source data
- **AnalysisError**: Computation issue - verify data quality
- **ParallelExecutionError**: Multi-threading issue - try sequential mode

## Advanced Usage

### Configuration Options
Edit `src/Fetchers/ProjectConfig.py` for:
- Custom date ranges
- Different tickers or indicators
- Processing parameters
- Performance settings

### Adding New Analyses
1. Create function in `src/Medallion/gold/AnalysisSuite/`
2. Add to `GoldLayer.py` run methods
3. Add user messages in `logger/Messages/MainMess.py`
4. Update catalog logging

### Extending Data Sources
1. Create new fetcher in `src/Fetchers/`
2. Add to `DataFactory.py`
3. Update schema in `src/Medallion/silver/schema.py`

## Performance Monitoring

### Logs and Metrics
- `logs/application_catalog.log`: Detailed operation logs
- `logs/session_summary_*.json`: Session performance summaries
- Real-time metrics during execution

### SLO/SLA Observability
- Stage-level success/error accounting is tracked in pipeline logs.
- Rolling SLO windows are emitted for 5-minute and 1-hour windows.
- SLA snapshots include p95 latency, throughput, success rate, and error rate.

### Governance Decision Artifacts
- Each pipeline run exports a versioned governance decision artifact under
   `data/users/<user_id>/gold/governance/`.
- Artifacts include run context (`run_id`, `correlation_id`), gate decision,
   severity band, and full governance report payload.

### Auditor Temporal Continuity
- Temporal continuity is currently an auditor-side quality check, not a pipeline hard-stop.
- It uses business-day gaps, duplicate removal on `['date', 'ticker']`, and a configurable
   allowed gap threshold.
- This means weekend gaps are not treated as major discontinuities.
- If you later want continuity to affect pipeline governance directly, that would require an
   explicit integration into the pipeline/governance layer rather than the auditor alone.

### Key Metrics
- Records processed per data source
- Analysis completion times
- Error rates and types
- System resource usage

## Support and Development

### Getting Help
- Check console output for specific error messages
- Review log files for detailed diagnostics
- Verify data quality in raw/ and processed/ directories

### Contributing
- Add new analysis functions
- Extend data source support
- Improve error handling
- Enhance user interface

## Architecture Overview

### Medallion Architecture
1. **Bronze Layer**: Raw data ingestion and basic cleaning
2. **Silver Layer**: Data transformation, validation, and quality checks
3. **Gold Layer**: Advanced analytics and feature engineering

### Key Technologies
- Python 3.8+ with async processing
- Pandas for data manipulation
- Statsmodels and scikit-learn for analysis
- PyArrow for efficient data storage
- Custom exception handling for robustness

---

**Version**: 1.0
**Last Updated**: March 2026
**Contact**: Developer documentation for technical support