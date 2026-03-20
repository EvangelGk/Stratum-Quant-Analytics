from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from UI.constants import (
    AUDIT_REPORT_PATH,
    LOGS_DIR,
    OUTPUT_DIR,
    ROOT,
    UI_SCHEDULE_PATH,
)
from UI.constants import UI_SNAPSHOT_PATH, USER_DATA_DIR
from UI.helpers import read_json


class StreamlitError(Exception):
    pass


class PipelineExecutionError(StreamlitError):
    pass


class PipelineSubprocessError(PipelineExecutionError):
    pass


class PipelineProgressTrackingError(PipelineExecutionError):
    pass


def run_pipeline(mode: str = "actual", progress_bar: Any = None) -> tuple[bool, str]:
    env = os.environ.copy()
    env["ENVIRONMENT"] = mode
    cmd = [sys.executable, "src/main.py"]
    estimated_seconds = 420 if mode == "actual" else 150
    stages = [
        (0.00, "Starting pipeline..."),
        (0.08, "Checking prerequisites..."),
        (0.15, "Loading configuration..."),
        (0.30, "Fetching data from APIs..."),
        (0.50, "Bronze layer: organizing raw data..."),
        (0.68, "Silver layer: cleaning and validating..."),
        (0.85, "Gold layer: running analyses..."),
        (0.95, "Exporting results and audit artifacts..."),
    ]

    log_path: Path | None = None
    proc = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            errors="ignore",
            delete=False,
            suffix="_scenario_planner.log",
        ) as tmp:
            log_path = Path(tmp.name)

        try:
            with log_path.open("w", encoding="utf-8", errors="ignore") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
        except FileNotFoundError as exc:
            raise PipelineSubprocessError(f"Python executable not found: {exc}") from exc
        except OSError as exc:
            raise PipelineSubprocessError(f"Failed to start subprocess: {exc}") from exc

        start = time.time()
        stage_i = 0
        while proc.poll() is None:
            elapsed = time.time() - start
            pct = min(elapsed / estimated_seconds, 0.97)
            while stage_i < len(stages) - 1 and pct >= stages[stage_i + 1][0]:
                stage_i += 1
            if progress_bar is not None:
                progress_bar.progress(pct, text=f"{int(pct * 100)}% — {stages[stage_i][1]}")
            time.sleep(0.5)

        output = log_path.read_text(encoding="utf-8", errors="ignore")
        ok = proc.returncode == 0
        if progress_bar is not None:
            progress_bar.progress(1.0, text="100% — Completed!" if ok else "100% — Failed")
        return ok, output.strip()
    except (PipelineSubprocessError, PipelineProgressTrackingError):
        raise
    except Exception as exc:
        raise PipelineExecutionError(f"Unexpected pipeline error: {exc}") from exc
    finally:
        if log_path is not None:
            try:
                log_path.unlink(missing_ok=True)
            except OSError:
                pass


def show_pipeline_failure(raw_output: str) -> None:
    st.error("Pipeline did not complete. Here is what happened:")
    q_report = read_json(USER_DATA_DIR / "processed" / "quality" / "quality_report.json")
    files_info = q_report.get("files", {}) if isinstance(q_report, dict) else {}
    summary_info = q_report.get("summary", {}) if isinstance(q_report, dict) else {}
    failed_entities = [
        (name, meta)
        for name, meta in files_info.items()
        if isinstance(meta, dict) and meta.get("status") == "failed"
    ]
    missing_src = summary_info.get("missing_sources", []) if isinstance(summary_info, dict) else []

    if missing_src:
        st.warning(
            f"No usable data from: {', '.join(missing_src)}. All datasets from these sources were rejected by quality checks."
        )
    if failed_entities:
        st.markdown(f"**{len(failed_entities)} dataset(s) failed quality checks:**")
        for name, meta in failed_entities:
            err = meta.get("error", "unknown error")
            src = meta.get("source", "?")
            if "Schema validation failed" in err:
                plain = "Values outside the expected range for this data type."
            elif "hard stop" in err or "empty" in err.lower():
                plain = "Dataset was empty or too small to process."
            elif "Missing columns" in err or "Schema drift" in err:
                plain = "Dataset is missing required columns."
            elif "null" in err.lower() or "missing" in err.lower():
                plain = "Too many missing values in the dataset."
            else:
                plain = err[:120]
            st.markdown(f"- **{name}** ({src}): {plain}")
    else:
        st.info("Quality report not found or no Silver failures recorded. Check logs for upstream fetch failures.")

    with st.expander("Raw pipeline output", expanded=False):
        st.text(raw_output[-4000:] if len(raw_output) > 4000 else raw_output)


def _load_audit_class() -> Any:
    try:
        spec = importlib.util.spec_from_file_location("Auditor", ROOT / "Auditor.py")
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "ScenarioAuditor", None)
    except Exception:
        return None


def run_and_cache_audit() -> dict[str, Any]:
    AuditorClass = _load_audit_class()
    if AuditorClass is None:
        result: dict[str, Any] = {"status": "ERROR", "error": "Auditor module could not be loaded."}
    else:
        try:
            auditor = AuditorClass(user_id=os.getenv("DATA_USER_ID", "default"))
            result = auditor.run_audit()
        except Exception as exc:
            result = {"status": "ERROR", "error": str(exc)}
    st.session_state["audit_report"] = result
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        AUDIT_REPORT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass
    return result


def get_audit_report() -> dict[str, Any]:
    if "audit_report" in st.session_state:
        return st.session_state["audit_report"]
    report = read_json(AUDIT_REPORT_PATH)
    if report:
        st.session_state["audit_report"] = report
    return report


def clear_all_run_history() -> dict[str, Any]:
    deleted_files = 0
    deleted_dirs = 0
    skipped_locked: list[str] = []
    targets: list[Path] = []
    if LOGS_DIR.exists():
        targets.extend(LOGS_DIR.glob("session_summary_*.json"))
        targets.append(LOGS_DIR / "application_catalog.log")
        targets.append(UI_SNAPSHOT_PATH)
        targets.append(UI_SCHEDULE_PATH)
    if OUTPUT_DIR.exists():
        for child in OUTPUT_DIR.glob("*"):
            targets.append(child)
    for path in targets:
        if path.exists() and path.is_file():
            try:
                path.unlink(missing_ok=True)
                deleted_files += 1
            except PermissionError:
                # Windows can lock active log files; skip and report them.
                skipped_locked.append(str(path))
            except OSError:
                skipped_locked.append(str(path))

    def _on_rmtree_error(_func: Any, p: str, exc_info: Any) -> None:
        if exc_info and len(exc_info) >= 2 and isinstance(exc_info[1], PermissionError):
            skipped_locked.append(p)

    if USER_DATA_DIR.exists():
        shutil.rmtree(USER_DATA_DIR, ignore_errors=False, onerror=_on_rmtree_error)
        deleted_dirs += 1
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR, ignore_errors=False, onerror=_on_rmtree_error)
        deleted_dirs += 1

    for key in [
        "audit_report",
        "selected_preview_file",
        "selected_preview_name",
    ]:
        st.session_state.pop(key, None)

    return {
        "deleted_files": deleted_files,
        "deleted_dirs": deleted_dirs,
        "skipped_locked": sorted(set(skipped_locked)),
        "message": (
            "Run history cleanup completed with some files skipped because they are in use."
            if skipped_locked
            else "Run history, generated data, outputs, and logs were removed."
        ),
    }


__all__ = [
    "run_pipeline",
    "show_pipeline_failure",
    "run_and_cache_audit",
    "get_audit_report",
    "clear_all_run_history",
]
