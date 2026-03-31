from pathlib import Path
import os

# Assuming UI is one level below project root (e.g., scenario-planner/UI)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Function to get the user ID, respecting DATA_USER_ID environment variable
def _get_data_user_id() -> str:
    # Use os.getenv directly here as get_secret might have Streamlit dependencies
    # or be part of a different module loading order in UI context.
    return os.getenv("DATA_USER_ID", "default").strip() or "default"

USER_ID = _get_data_user_id()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "users" / USER_ID / "processed"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
OUTPUT_DIR = PROJECT_ROOT / "output" / USER_ID