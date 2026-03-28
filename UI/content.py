from __future__ import annotations

LAYER_HELP = {
    "raw": {
        "icon": "📥",
        "title": "RAW Layer — Original Data",
        "what": "Unmodified data downloaded directly from external APIs.",
        "contains": [
            "Yahoo Finance: stock prices (AAPL, F, …)",
            "FRED: economic indicators (inflation, energy prices, …)",
            "World Bank: global development data (GDP, population, …)",
        ],
        "note": "Nothing is changed here. This is your single source of truth.",
    },
    "processed": {
        "icon": "🔧",
        "title": "PROCESSED Layer — Cleaned & Validated",
        "what": "Raw data after cleaning, imputation, and quality checks.",
        "contains": [
            "Missing values filled / estimated",
            "Outliers detected and handled (Winsorisation)",
            "Schema validated — consistent column types",
            "Quality report saved to data/users/<user_id>/processed/quality/quality_report.json",
        ],
        "note": "Use this layer when you need clean, trustworthy data.",
    },
    "gold": {
        "icon": "💎",
        "title": "GOLD Layer — Analysis-Ready",
        "what": "Master analytical table that merges all cleaned sources.",
        "contains": [
            "Log-returns plus broad market + macro factor universe",
            "Governance decisions and risk scores applied",
            "Feature-subset search and robust CV diagnostics for model selection",
            "Ready for correlation analysis, forecasting, stress-tests, …",
        ],
        "note": "This is what the Analytics tab visualises.",
    },
}

ANALYSIS_HELP: dict[str, dict[str, str]] = {
    "correlation_matrix": {
        "title": "Correlation Matrix",
        "what": "Measures linear relationships between every pair of variables (assets + macro factors).",
        "read": (
            "Values range from −1 (perfect negative) to +1 (perfect positive). Look for values above ±0.7 — those signal strong dependencies worth watching."
        ),
        "use": "Identify which macro factors move together with a stock, or which assets offset each other in a portfolio.",
    },
    "governance_report": {
        "title": "Governance Report",
        "what": "A record of every automated data-quality decision made during the pipeline run.",
        "read": "Each entry shows whether a dataset was approved or flagged, the risk score, and the reason.",
        "use": "Audit trail — tells you which data entered the analytical layer and why.",
    },
    "elasticity": {
        "title": "Elasticity Analysis",
        "what": "How much do asset returns change for a 1 % change in a macro factor?",
        "read": "Elasticity > 1 means high sensitivity; Elasticity < 1 means weaker sensitivity; negative means inverse relationship.",
        "use": "Spot which stocks are most exposed to inflation, energy prices, or interest rates.",
    },
    "lag_analysis": {
        "title": "Lag Analysis",
        "what": "Shows delayed effects — how a macro change today ripples through asset prices over following periods.",
        "read": "Lag 0 is immediate; larger lags suggest delayed transmission.",
        "use": "Build trading signals or risk alerts that anticipate market moves before they fully materialise.",
    },
    "monte_carlo": {
        "title": "Monte Carlo Simulation",
        "what": "Generates many future price paths using historical volatility and drift.",
        "read": "The wider the band, the higher the uncertainty and tail risk.",
        "use": "Assess tail risk and scenario dispersion.",
    },
    "stress_test": {
        "title": "Stress Test",
        "what": "Applies extreme but plausible macro shocks and measures portfolio impact.",
        "read": "Each scenario shows directional loss or drawdown impact under shock assumptions.",
        "use": "Worst-case planning and hedging assessment.",
    },
    "sensitivity_reg": {
        "title": "Sensitivity Regression",
        "what": "Multivariate model selection over many feature combinations to quantify each factor's contribution.",
        "read": "Panel reports selected factor subset, CV R², lag selection, and signed coefficients.",
        "use": "Pin down dominant drivers and keep only robust features for better unseen-data performance.",
    },
    "forecasting": {
        "title": "Time-Series Forecasting",
        "what": "ARIMA model projecting future values from historical patterns.",
        "read": "Confidence intervals widen as forecast uncertainty increases.",
        "use": "Trend analysis and short-horizon planning.",
    },
    "auto_ml": {
        "title": "Auto ML Regression",
        "what": "Automatically compares multiple machine-learning models and picks the best predictor.",
        "read": "The winning model and performance metrics indicate predictive usefulness.",
        "use": "Predictive modelling with model comparison and feature ranking.",
    },
}

PIPELINE_STAGES = [
    (
        "Prerequisites check",
        "Verifies Python version, installed packages, and API key presence.",
    ),
    (
        "Configuration load",
        "Reads environment variables, API keys, and sets processing parameters.",
    ),
    (
        "Data fetching",
        "Downloads stock prices, economic indicators, and global data from APIs.",
    ),
    (
        "Bronze layer",
        "Organises raw files, applies initial cleaning, updates the data catalog.",
    ),
    (
        "Silver layer",
        "Validates schemas, imputes missing values, detects and clips outliers.",
    ),
    (
        "Gold layer",
        "Builds master table, runs analyses, applies governance decisions.",
    ),
    (
        "Export results",
        "Saves analysis files and governance decisions to user-scoped folders.",
    ),
]
