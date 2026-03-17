# Main Application Messages
# Messages for main script execution and user interaction

APPLICATION_TITLE = "=== Scenario Planner ==="

QUICK_START = (
    "Quick start: configure API keys, run the pipeline, then review output/ and logs/."
)

MAIN_START = "Starting Scenario Planner application in {mode} mode."

MAIN_CONFIG_LOADED = "Configuration loaded successfully. Using {config_details}."

MAIN_PIPELINE_START = (
    "Initializing full Medallion pipeline. This will process data from raw "
    "to analytics."
)

MAIN_PIPELINE_SUCCESS = (
    "Pipeline execution completed successfully. Analysis results available."
)

MAIN_RESULTS_SUMMARY = (
    "Analysis Results Summary: {result_keys}. Check output/ directory for "
    "detailed reports."
)

MAIN_ERROR_HANDLING = (
    "Encountered {error_type}: {error_message}. Application handled gracefully."
)

MAIN_COMPLETION = (
    "Application finished. Total execution time: {execution_time} seconds."
)

# User guidance messages
MAIN_USER_GUIDE = """
Scenario Planner Usage Guide:
1. Ensure you have API keys for FRED and other data sources
2. Run the application - it will automatically fetch, process, and analyze data
3. Check the output/ directory for analysis results
4. Review logs for any issues or progress updates
5. Use the generated reports for scenario planning decisions
"""

# Output explanations
MAIN_OUTPUT_EXPLANATION = """
What you'll see on screen:
- Application startup messages with mode and config
- Progress bars for each pipeline stage
- Success messages with result summaries
- Error messages with specific guidance
- Final completion status and file locations
- Analysis result previews (correlation matrices, forecasts, etc.)
"""

# Analysis Output Messages (detailed for each analysis)
ANALYSIS_CORRELATION_MATRIX = """
Correlation Matrix Analysis Results:
- Shows relationships between financial assets and macro factors
- Values range from -1 (perfect negative) to +1 (perfect positive)
- Look for strong correlations (>0.7 or <-0.7) for risk factors
- Matrix size: {rows}x{columns} variables analyzed
"""

ANALYSIS_ELASTICITY = """
Elasticity Analysis Results:
- Measures how much asset returns change with macro factor changes
- Formula: %ΔAsset / %ΔMacro Factor
- Values >1: Asset is elastic (volatile to macro changes)
- Values <1: Asset is inelastic (stable despite macro changes)
- Current elasticity: {elasticity_value}
"""

ANALYSIS_LAG_ANALYSIS = """
Lag Analysis Results:
- Shows delayed impacts of macro factors on asset returns
- Lag 0: Immediate effect, Lag 1: 1-period delay, etc.
- Higher correlations at longer lags indicate delayed reactions
- Best lag for {factor}: {best_lag} with correlation {correlation}
"""

ANALYSIS_MONTE_CARLO = """
Monte Carlo Simulation Results:
- Generated {iterations} price paths for {ticker}
- Each path simulates {days} days of future prices
- Based on historical volatility and returns
- Use for risk assessment and scenario planning
- Expected final price range: ${min_price} - ${max_price}
"""

ANALYSIS_STRESS_TEST = """
Stress Test Results:
- Simulates extreme market conditions
- Applied shocks: {shock_details}
- Shows potential losses under adverse scenarios
- Risk measure: Maximum drawdown could be {max_drawdown}%
- Use for worst-case scenario planning
"""

ANALYSIS_SENSITIVITY_REGRESSION = """
Sensitivity Regression Results:
- Multivariate analysis of factor impacts on returns
- Model: {model_type} (OLS or Ridge regression)
- Key drivers: {top_factors} with coefficients {coefficients}
- R-squared: {r_squared}% of variance explained
- Use coefficients to understand factor sensitivities
"""

ANALYSIS_FORECASTING = """
Time Series Forecasting Results:
- ARIMA model predictions for {steps} future periods
- Based on historical {column} data
- Model parameters: p={p}, d={d}, q={q}
- Forecast confidence intervals included
- Use for trend analysis and future expectations
"""

ANALYSIS_AUTO_ML = """
Auto ML Regression Results:
- Best model selected: {best_model}
- Performance metrics: {metrics}
- Feature importance: {important_features}
- Automated model comparison completed
- Use for predictive modeling and factor analysis
"""
