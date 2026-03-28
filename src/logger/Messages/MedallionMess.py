# Medallion Messages
# Messages for Bronze, Silver, Gold layer operations

# Bronze Layer Messages
BRONZE_START = "Starting Bronze Layer processing. Fetching and organizing raw data."

BRONZE_PROCESSING = "Processing {filename} from {source}. Applying initial cleaning and validation."

BRONZE_SUCCESS = "Bronze Layer completed. Processed {total_files} files. Data ready for Silver processing."

BRONZE_CATALOG_UPDATE = "Updated data catalog with {new_entries} new entries. Catalog saved to raw/catalog.json."

# Silver Layer Messages
SILVER_START = "Starting Silver Layer transformation. Cleaning and standardizing data."

SILVER_VALIDATION = "Validating {filename} schema. Checking data types and required columns."

SILVER_IMPUTATION = "Imputing missing values in {filename}. Filled {filled_values} missing entries."

SILVER_STANDARDIZATION = "Standardizing {filename}. Converted to consistent formats and units."

SILVER_OUTLIER_DETECTION = "Detecting outliers in {filename}. Clipped {outliers} extreme values."

SILVER_SUCCESS = "Silver Layer completed. All data transformed and saved to processed/ directory."

SILVER_QUALITY_REPORT = "Quality report generated. Check processed/quality_report.json for details."

# Gold Layer Messages
GOLD_START = "Starting Gold Layer analytics. Building master analytical table."

GOLD_MASTER_TABLE = "Created master table with {total_rows} rows and {total_columns} columns. Includes log returns and macro data."

GOLD_ANALYSIS_START = "Running {analysis_name} analysis. This may take a moment for large datasets."

GOLD_ANALYSIS_SUCCESS = "{analysis_name} completed successfully. Results available in analysis outputs."

GOLD_PARALLEL_MODE = "Running analyses in parallel mode with {workers} workers for better performance."

GOLD_SEQUENTIAL_MODE = "Running analyses sequentially due to parallel execution error."

GOLD_SUCCESS = "Gold Layer completed. All analyses finished. Check output/ directory for results."

# User guidance messages
MEDALLION_USER_GUIDE = """
Medallion Architecture Guide:
- Bronze: Raw data ingestion and basic cleaning
- Silver: Data transformation, validation, and quality checks
- Gold: Advanced analytics and feature engineering
- Progress bars show current processing status
- Quality reports provide data health metrics
- Master table combines all data for analysis
"""

# Output explanations
MEDALLION_OUTPUT_EXPLANATION = """
What you'll see on screen:
- Layer-by-layer progress updates
- File processing status with counts
- Quality metrics (null values, outliers, etc.)
- Analysis results summaries
- Final completion messages with file locations
- Error messages with specific failure reasons
"""
