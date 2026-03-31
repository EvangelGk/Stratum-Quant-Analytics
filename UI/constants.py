from pathlib import Path
import os

# Project root is one level up from `scenario-planner/UI/`
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT  # Alias for any code that still uses the old name

# Function to get the user ID, respecting DATA_USER_ID environment variable
def _get_data_user_id() -> str:
    # Use os.getenv directly here as get_secret might have Streamlit dependencies
    # or be part of a different module loading order in UI context.
    return os.getenv("DATA_USER_ID", "default").strip() or "default"


USER_ID = _get_data_user_id()

# Core data directories
RAW_DIR = ROOT / "data" / "raw"
USER_DATA_DIR = ROOT / "data" / "users" / USER_ID
PROCESSED_DIR = USER_DATA_DIR / "processed"
GOLD_DIR = USER_DATA_DIR / "gold"
OUTPUT_DIR = ROOT / "output" / USER_ID
LOGS_DIR = ROOT / "logs"

# Specific file paths derived from directories
AUDIT_REPORT_PATH = OUTPUT_DIR / "audit_report.json"
UI_SNAPSHOT_PATH = LOGS_DIR / "ui_snapshot.json"
UI_SCHEDULE_PATH = LOGS_DIR / "ui_schedule.json"

ROLE_PERMISSIONS = {
    "Viewer": {"can_run": False, "can_download": True, "can_schedule": False},
    "Analyst": {"can_run": True, "can_download": True, "can_schedule": False},
    "Admin": {"can_run": True, "can_download": True, "can_schedule": True},
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
    ("Bronze layer", "Organises raw files, applies initial cleaning, updates the data catalog."),
    ("Silver layer", "Validates schemas, imputes missing values, detects and clips outliers."),
    ("Gold layer", "Builds master table, runs analyses, applies governance decisions."),
    (
        "Export results",
        "Saves analysis files and governance decisions to user-scoped folders.",
    ),
]