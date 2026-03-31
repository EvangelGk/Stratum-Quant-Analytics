import os
import re
from pathlib import Path
from typing import Any, Dict

# Project root is one level up from `scenario-planner/UI/`
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT  # Alias for any code that still uses the old name

# Function to get the user ID, respecting DATA_USER_ID environment variable
def _get_data_user_id() -> str:
    # Use os.getenv directly here as get_secret might have Streamlit dependencies
    # or be part of a different module loading order in UI context.
    return os.getenv("DATA_USER_ID", "default").strip() or "default"


USER_ID = _get_data_user_id()

LOGS_DIR = ROOT / "logs"

UI_SNAPSHOT_PATH = LOGS_DIR / "ui_snapshot.json"
UI_SCHEDULE_PATH = LOGS_DIR / "ui_schedule.json"


def _sanitize_analyst_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", (raw or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "default"


def _get_session_hash() -> str:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

        ctx = get_script_run_ctx()
        sid = getattr(ctx, "session_id", None) if ctx is not None else None
        if isinstance(sid, str) and sid.strip():
            return sid.strip()[:6]
    except Exception:
        pass
    return "local0"


def get_user_paths(analyst_id: str) -> Dict[str, Any]:
    safe_analyst_id = _sanitize_analyst_id(analyst_id)
    session_hash = _get_session_hash()
    user_key = f"{safe_analyst_id}_{session_hash}"
    user_root = ROOT / "data" / "users" / user_key
    output_root = ROOT / "output" / user_key
    return {
        "analyst_id": safe_analyst_id,
        "session_hash": session_hash,
        "user_key": user_key,
        "user_root": user_root,
        "bronze": user_root / "raw",
        "raw": user_root / "raw",
        "silver": user_root / "processed",
        "processed": user_root / "processed",
        "gold": user_root / "gold",
        "output": output_root,
        "logs": LOGS_DIR,
    }


def ensure_user_paths(paths: Dict[str, Any]) -> None:
    for key in ("user_root", "raw", "processed", "gold", "output"):
        p = paths.get(key)
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)
    # Common subfolders used by pipeline/UI modules.
    (paths["processed"] / "quality").mkdir(parents=True, exist_ok=True)
    (paths["gold"] / "governance").mkdir(parents=True, exist_ok=True)
    (paths["output"] / ".optimizer").mkdir(parents=True, exist_ok=True)


def initialize_active_paths(analyst_id: str | None = None) -> Dict[str, Any]:
    resolved_analyst = _sanitize_analyst_id(analyst_id or _get_data_user_id())
    paths = get_user_paths(resolved_analyst)
    ensure_user_paths(paths)
    # Keep subprocess compatibility for modules that still read DATA_USER_ID.
    os.environ["DATA_USER_ID"] = paths["user_key"]
    try:
        import streamlit as st

        st.session_state["analyst_id"] = resolved_analyst
        st.session_state["active_paths"] = paths
    except Exception:
        pass
    return paths


def get_active_paths() -> Dict[str, Any]:
    try:
        import streamlit as st

        active = st.session_state.get("active_paths")
        if isinstance(active, dict) and isinstance(active.get("raw"), Path):
            ensure_user_paths(active)
            os.environ["DATA_USER_ID"] = str(active.get("user_key", "default"))
            return active
        analyst = st.session_state.get("analyst_id", _get_data_user_id())
        return initialize_active_paths(str(analyst))
    except Exception:
        return initialize_active_paths(_get_data_user_id())


_DEFAULT_PATHS = get_user_paths(USER_ID)

# Backward-compatible constants (default snapshot). New code should call get_active_paths().
USER_DATA_DIR = _DEFAULT_PATHS["user_root"]
RAW_DIR = _DEFAULT_PATHS["raw"]
PROCESSED_DIR = _DEFAULT_PATHS["processed"]
GOLD_DIR = _DEFAULT_PATHS["gold"]
OUTPUT_DIR = _DEFAULT_PATHS["output"]
AUDIT_REPORT_PATH = OUTPUT_DIR / "audit_report.json"

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
