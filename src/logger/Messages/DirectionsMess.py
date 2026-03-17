# Directions Messages
# Live step-by-step guidance messages that appear during application execution

APPLICATION_TITLE = """
=====================================
   SCENARIO PLANNER APPLICATION
=====================================
A comprehensive financial data pipeline for scenario planning and risk analysis.
"""

# Live Guidance Messages - Appear in real-time during execution

LIVE_STEP_0_WELCOME = """
>>> WELCOME TO SCENARIO PLANNER!
===============================
This application will automatically:
* Fetch financial data from multiple sources
* Process and clean the data
* Run advanced financial analyses
* Generate comprehensive reports

Sit back and watch the magic happen!
"""

LIVE_STEP_1_PREREQUISITES_CHECK = """
[OK] STEP 1: CHECKING PREREQUISITES
==================================
* Python 3.8+ [OK]
* Required packages [OK]
* FRED API key [OK]
* Internet connection [OK]

All prerequisites verified! Proceeding...
"""

LIVE_STEP_2_CONFIG_LOADING = """
[CONFIG] STEP 2: LOADING CONFIGURATION
=================================
* Reading environment variables
* Validating API keys
* Setting processing parameters
* Initializing data paths

Configuration loaded successfully!
"""

LIVE_STEP_3_DATA_FETCHING_START = """
[FETCH] STEP 3: DATA FETCHING IN PROGRESS
====================================
The application is now fetching data from:

[*] Yahoo Finance: Stock prices and volumes
[*] FRED (Federal Reserve): Economic indicators
[*] World Bank: Global development data

This may take 2-5 minutes depending on data volume...
"""

LIVE_STEP_3_DATA_FETCHING_PROGRESS = """
[PROGRESS] FETCHING PROGRESS:
* Connecting to data sources...
* Downloading historical data...
* Validating data integrity...
* Caching for future use...

{current_source}: {records} records fetched [OK]
"""

LIVE_STEP_3_DATA_FETCHING_COMPLETE = """
[OK] DATA FETCHING COMPLETED!
===========================
Total data retrieved:
* {total_files} data files
* {total_records} total records
* From {sources_count} different sources

All data saved to data/raw/ directory.
"""

LIVE_STEP_4_BRONZE_PROCESSING = """
[BRONZE] STEP 4: BRONZE LAYER PROCESSING
===================================
Raw data is being organized and cleaned:

* Reading raw data files
* Basic data validation
* Initial cleaning and formatting
* Creating data catalog

Bronze layer: Foundation data ready [OK]
"""

LIVE_STEP_5_SILVER_PROCESSING = """
[SILVER] STEP 5: SILVER LAYER TRANSFORMATION
=======================================
Data is being standardized and validated:

* Schema validation
* Missing value imputation
* Outlier detection and handling
* Data quality assessment
* Standardization to consistent formats

Silver layer: Trusted data ready [OK]
"""

LIVE_STEP_6_GOLD_ANALYTICS = """
[GOLD] STEP 6: GOLD LAYER ANALYTICS
================================
Running comprehensive financial analyses:

* Building master analytical table
* Correlation analysis
* Risk modeling (Monte Carlo)
* Forecasting models
* Stress testing scenarios
* Factor sensitivity analysis

Gold layer: Advanced analytics in progress...
"""

LIVE_STEP_6_ANALYSIS_PROGRESS = """
[ANALYSIS] ANALYSIS PROGRESS:
=====================
{analysis_name}: {status}
* {description}
* Expected completion: {eta}
"""

LIVE_STEP_7_RESULTS_GENERATION = """
[RESULTS] STEP 7: GENERATING RESULTS
==============================
Compiling analysis results:

* Creating summary reports
* Exporting data visualizations
* Saving detailed metrics
* Preparing user-friendly outputs

Results are being saved to output/ directory...
"""

LIVE_STEP_8_COMPLETION = """
[SUCCESS] APPLICATION COMPLETED SUCCESSFULLY!
=======================================

EXECUTION SUMMARY:
==================
* Total runtime: {total_time} seconds
* Data processed: {total_records} records
* Analyses completed: {analyses_count}
* Files generated: {files_created}

[OUTPUT LOCATIONS]
===================
* Raw data: data/raw/
* Processed data: data/processed/
* Analytical results: data/gold/
* Reports: output/
* Logs: logs/

[WHAT TO DO NEXT]
===================
1. Check output/ for analysis results
2. Review logs/ for detailed metrics
3. Open USER_GUIDE.md for interpretation help
4. Run again with different parameters if needed

Thank you for using Scenario Planner!
"""

# Error Guidance Messages
LIVE_ERROR_API_KEY = """
[ERROR] API KEY ERROR
================
The FRED API key is missing or invalid.

SOLUTION:
1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html
2. Get a free API key
3. Add to .env file: FRED_API_KEY=your_key_here
4. Run the application again
"""

LIVE_ERROR_NETWORK = """
[ERROR] NETWORK ERROR
================
Unable to connect to data sources.

POSSIBLE SOLUTIONS:
* Check your internet connection
* Try again in a few minutes
* Some APIs may have temporary issues
* Check firewall/antivirus settings
"""

LIVE_ERROR_MEMORY = """
[ERROR] MEMORY ERROR
===============
System ran out of memory during processing.

SOLUTIONS:
* Close other applications
* Reduce date ranges in configuration
* Use sequential processing instead of parallel
* Add more RAM to your system
"""

LIVE_ERROR_DATA_QUALITY = """
[ERROR] DATA QUALITY ERROR
======================
Issues found in downloaded data.

ACTIONS:
* Check data/raw/ for corrupted files
* Verify API keys are correct
* Some data sources may be temporarily unavailable
* Review logs/ for specific error details
"""

# Progress Indicators
LIVE_PROGRESS_BAR = """
[{progress_bar}] {percentage}% Complete
{current_step}
Estimated time remaining: {eta}
"""

LIVE_CURRENT_STATUS = """
[STATUS] CURRENT STATUS: {status}
[TIME] ELAPSED TIME: {elapsed}
[PROGRESS] PROGRESS: {progress}%
[NEXT] NEXT: {next_step}
"""

# Interactive Prompts (for future enhancement)
LIVE_USER_PROMPT_CONTINUE = """
Press Enter to continue or 'q' to quit...
"""

LIVE_USER_PROMPT_CONFIG = """
Would you like to modify configuration? (y/n):

OUTPUT DIRECTORIES:
- data/raw/: Raw downloaded data
- data/processed/: Cleaned and transformed data
- data/gold/: Analytical master table
- output/: Analysis results and reports

STEP 4: INTERPRETING RESULTS
-----------------------------
1. Check output/ directory for analysis files
2. Review correlation matrices for factor relationships
3. Analyze Monte Carlo simulations for risk scenarios
4. Examine forecasting results for future trends
5. Use stress test results for worst-case planning

TROUBLESHOOTING:
- API errors: Check internet and API keys
- Memory errors: Reduce date ranges or iterations
- Slow performance: Use parallel mode (default)
- Data issues: Check raw data quality first

SUPPORT:
- Check logs in the console for detailed error messages
- Review quality reports for data health metrics
- Contact developer for technical issues
"""

QUICK_START = """
QUICK START GUIDE:
1. Set FRED_API_KEY in .env file
2. Run: python src/main.py
3. Wait for completion
4. Check output/ for results
"""

ADVANCED_USAGE = """
ADVANCED USAGE:
- Modify config in ProjectConfig.py for custom settings
- Add new analysis functions in AnalysisSuite/
- Extend data sources in Fetchers/
- Customize exception handling in exceptions/
- Add new metrics in logger/
"""

DATA_SOURCES_EXPLANATION = """
DATA SOURCES INCLUDED:
- Yahoo Finance: Stock prices and volumes
- FRED (Federal Reserve): Economic indicators (inflation, unemployment, interest rates)
- World Bank: Global development data (GDP, population, trade)

All data is automatically cleaned, standardized, and combined for analysis.
"""

ANALYSIS_EXPLANATION = """
AVAILABLE ANALYSES:
1. Correlation Matrix: Relationships between variables
2. Elasticity: Price sensitivity to macro factors
3. Lag Analysis: Delayed factor impacts
4. Monte Carlo: Risk simulation with random scenarios
5. Stress Testing: Extreme condition analysis
6. Sensitivity Regression: Multivariate factor analysis
7. Time Series Forecasting: Future trend prediction
8. Auto ML: Automated predictive modeling

Each analysis provides specific insights for scenario planning.
"""
