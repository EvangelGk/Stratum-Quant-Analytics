from __future__ import annotations

import importlib
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT

_root_path = str(ROOT)
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)

_src_path = str(ROOT / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

_secret_store = importlib.import_module("secret_store")
bootstrap_env_from_secrets = getattr(_secret_store, "bootstrap_env_from_secrets")
get_secret = getattr(_secret_store, "get_secret")

load_dotenv(ROOT / ".env")
bootstrap_env_from_secrets(override=False, only_keys=["DATA_USER_ID"])

_UI_USER_ID = (get_secret("DATA_USER_ID", "default") or "default").strip() or "default"
_SAFE_UI_USER = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in _UI_USER_ID) or "default"

OUTPUT_DIR = ROOT / "output" / _SAFE_UI_USER
USER_DATA_DIR = ROOT / "data" / "users" / _SAFE_UI_USER
RAW_DIR = USER_DATA_DIR / "raw"
PROCESSED_DIR = USER_DATA_DIR / "processed"
GOLD_DIR = USER_DATA_DIR / "gold"
LOGS_DIR = ROOT / "logs"
UI_SNAPSHOT_PATH = LOGS_DIR / "ui_run_snapshots.json"
UI_SCHEDULE_PATH = LOGS_DIR / "ui_schedule.json"
AUDIT_REPORT_PATH = OUTPUT_DIR / "audit_report.json"

ROLE_PERMISSIONS = {
    "Viewer": {"can_run": False, "can_download": False, "can_schedule": False},
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
        "Builds master table, runs all analyses, applies governance decisions.",
    ),
    (
        "Export results",
        "Saves analysis files and governance decisions to user-scoped folders.",
    ),
]

LAYER_HELP = {
    "raw": {
        "icon": "IN",
        "title": "RAW Layer - Original Data",
        "what": "Unmodified data downloaded directly from external APIs.",
        "contains": [
            "Yahoo Finance: stock prices",
            "FRED: economic indicators",
            "World Bank: global development data",
        ],
        "note": "Nothing is changed here. This is your source of truth.",
    },
    "processed": {
        "icon": "PROC",
        "title": "PROCESSED Layer - Cleaned and Validated",
        "what": "Raw data after cleaning, imputation, and quality checks.",
        "contains": [
            "Missing values imputed",
            "Outliers handled",
            "Schema validated",
            "Quality report generated",
        ],
        "note": "Use this layer for reliable analysis.",
    },
    "gold": {
        "icon": "GOLD",
        "title": "GOLD Layer - Analysis-Ready",
        "what": "Master analytical table that merges all cleaned sources.",
        "contains": [
            "Returns, market fields, and macro/worldbank factors combined",
            "Governance decisions applied",
            "Feature-subset search + time-series CV diagnostics",
            "Ready for analytics",
        ],
        "note": "This is the layer used by analytics and reports.",
    },
}
