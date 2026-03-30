from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from UI.constants import (
    AUDIT_REPORT_PATH,
    LOGS_DIR,
    OUTPUT_DIR,
    ROOT,
    UI_SCHEDULE_PATH,
    UI_SNAPSHOT_PATH,
    USER_DATA_DIR,
)
from UI.helpers import read_json

# Ensure project root and src/ are on sys.path before importing secret_store.
_ROOT_PATH = str(ROOT)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

# Ensure src/ is on sys.path before importing secret_store in all contexts.
_SRC_PATH = str(ROOT / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

_secret_store = importlib.import_module("secret_store")
bootstrap_env_from_secrets = getattr(_secret_store, "bootstrap_env_from_secrets")


class StreamlitError(Exception):
    pass


class PipelineExecutionError(StreamlitError):
    pass


class PipelineSubprocessError(PipelineExecutionError):
    pass


class PipelineProgressTrackingError(PipelineExecutionError):
    pass


def run_pipeline(
    mode: str = "actual",
    progress_bar: Any = None,
    resume_from_checkpoint: bool = False,
) -> tuple[bool, str]:
    # Ensure Streamlit secrets are reflected in os.environ before subprocess spawn
    # so DATA_USER_ID and governance knobs are visible to src/main.py.
    bootstrap_env_from_secrets(override=True)
    env = os.environ.copy()
    env["ENVIRONMENT"] = mode
    env["PIPELINE_RESUME_FROM_CHECKPOINT"] = "1" if resume_from_checkpoint else "0"
    cmd = [sys.executable, "src/main.py"]
    estimated_seconds = 720 if mode == "actual" else 240
    stages = [
        (0.00, "Starting pipeline..."),
        (0.08, "Checking prerequisites..."),
        (0.15, "Loading configuration..."),
        (0.30, "Fetching data from APIs..."),
        (0.50, "Bronze layer: organizing raw data..."),
        (0.68, "Silver layer: cleaning and validating..."),
        (0.85, "Gold layer: running analyses..."),
        (0.95, "Gold layer: Monte Carlo & final exports (may take a few minutes)..."),
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
                if elapsed > estimated_seconds:
                    mins = int(elapsed / 60)
                    secs = int(elapsed % 60)
                    msg = f"97% — Gold layer still running ({mins}m {secs:02d}s elapsed) — please wait..."
                else:
                    msg = f"{int(pct * 100)}% — {stages[stage_i][1]}"
                progress_bar.progress(pct, text=msg)
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
    failed_entities = [(name, meta) for name, meta in files_info.items() if isinstance(meta, dict) and meta.get("status") == "failed"]
    missing_src = summary_info.get("missing_sources", []) if isinstance(summary_info, dict) else []

    if missing_src:
        st.warning(f"No usable data from: {', '.join(missing_src)}. All datasets from these sources were rejected by quality checks.")
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
    elif files_info:
        st.success("Όλα τα datasets πέρασαν τα quality checks (Silver layer OK). Δεν υπάρχουν αποτυχίες.")
    else:
        st.info("Quality report not found ή δεν υπάρχουν Silver failures. Check logs for upstream fetch failures.")

    with st.expander("Raw pipeline output", expanded=False):
        st.text(raw_output[-4000:] if len(raw_output) > 4000 else raw_output)


def _load_audit_class() -> Any:
    try:
        spec = importlib.util.spec_from_file_location("Auditor", ROOT / "Auditor.py")
        if spec is None or spec.loader is None:
            try:
                from Auditor import ScenarioAuditor  # type: ignore

                return ScenarioAuditor
            except Exception:
                return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "ScenarioAuditor", None)
    except Exception:
        try:
            from Auditor import ScenarioAuditor  # type: ignore

            return ScenarioAuditor
        except Exception:
            return None


def run_and_cache_audit() -> dict[str, Any]:
    bootstrap_env_from_secrets(override=True)
    AuditorClass = _load_audit_class()
    if AuditorClass is None:
        result: dict[str, Any] = {"status": "ERROR", "error": "Auditor module could not be loaded."}
    else:
        try:
            auditor = AuditorClass(user_id="default")
            result = auditor.run_audit()
        except Exception as exc:
            result = {"status": "ERROR", "error": str(exc)}
    st.session_state["audit_report"] = result

    result_score = _audit_report_completeness(_normalize_audit_report_payload(result))
    should_persist = result_score >= 20 and str(result.get("status", "")).upper() != "ERROR"
    try:
        if should_persist:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            AUDIT_REPORT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass
    return result


def _audit_report_completeness(report: Any) -> int:
    if not isinstance(report, dict) or not report:
        return -1

    score = 0
    checks = report.get("checks")
    if isinstance(checks, dict):
        score += len(checks) * 10
        non_empty_checks = sum(1 for payload in checks.values() if isinstance(payload, dict) and payload)
        score += non_empty_checks * 5

    row_count = report.get("row_count")
    column_count = report.get("column_count")
    if isinstance(row_count, int):
        score += 2
        if row_count > 0:
            score += 10
    if isinstance(column_count, int):
        score += 2
        if column_count > 0:
            score += 10

    if isinstance(report.get("failed_checks"), list):
        score += 2
    if isinstance(report.get("warning_checks"), list):
        score += 2
    if isinstance(report.get("auditor_judgement"), dict):
        score += 2
    if isinstance(report.get("decision_ready"), bool):
        score += 2

    status = report.get("status")
    if isinstance(status, str) and status.strip():
        score += 2

    return score


def _normalize_audit_report_payload(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    # Support wrapped artifacts: {"value": {...}}
    wrapped = report.get("value")
    if isinstance(wrapped, dict):
        return wrapped
    return report


def get_audit_report() -> dict[str, Any]:
    session_report = _normalize_audit_report_payload(st.session_state.get("audit_report"))
    disk_report = _normalize_audit_report_payload(read_json(AUDIT_REPORT_PATH))

    session_score = _audit_report_completeness(session_report)
    disk_score = _audit_report_completeness(disk_report)

    if disk_score > session_score:
        st.session_state["audit_report"] = disk_report
        return disk_report
    if session_score >= 0:
        return session_report
    if disk_score >= 0:
        st.session_state["audit_report"] = disk_report
        return disk_report

    # Fallback: scan other output profiles and pick the most complete report.
    try:
        output_root = OUTPUT_DIR.parent
        best_report: dict[str, Any] = {}
        best_score = -1
        if output_root.exists():
            for child in output_root.iterdir():
                if not child.is_dir():
                    continue
                candidate = _normalize_audit_report_payload(read_json(child / "audit_report.json"))
                score = _audit_report_completeness(candidate)
                if score > best_score:
                    best_score = score
                    best_report = candidate
        if best_score >= 0:
            st.session_state["audit_report"] = best_report
            return best_report
    except OSError:
        pass
    return {}


def _to_serializable(value: Any) -> Any:
    try:
        import numpy as np
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return {
                "type": "dataframe",
                "shape": [int(value.shape[0]), int(value.shape[1])],
                "columns": [str(c) for c in value.columns],
                "data": value.to_dict(orient="records"),
            }
        if isinstance(value, pd.Series):
            return {
                "type": "series",
                "name": str(value.name) if value.name is not None else None,
                "data": value.tolist(),
            }
        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "shape": [int(x) for x in value.shape],
                "data": value.tolist(),
            }
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    return value


def rerun_stress_test_only(
    scenario_name: str = "geopolitical_conflict",
    shock_map: Optional[Dict[str, float]] = None,
    ticker: Optional[str] = None,
    target: str = "log_return",
) -> Dict[str, Any]:
    """Run stress test directly from existing Gold data without rerunning pipeline."""
    from src.Fetchers.ProjectConfig import ProjectConfig
    from src.Medallion.gold.AnalysisSuite.mixed_frequency import filter_to_ticker
    from src.Medallion.gold.AnalysisSuite.stress_test import stress_test
    from src.Medallion.gold.GoldLayer import GoldLayer

    config = ProjectConfig.load_from_env()
    gold = GoldLayer(config)

    # Mirror MedallionPipeline user-scoped paths so UI uses the same dataset.
    gold.processed_path = USER_DATA_DIR / "processed"
    gold.gold_path = USER_DATA_DIR / "gold"
    gold.governance_path = gold.gold_path / "governance"
    gold.governance_path.mkdir(parents=True, exist_ok=True)
    gold.initialize_data()

    analysis_df = filter_to_ticker(gold.df, ticker=ticker)
    result = stress_test(
        analysis_df,
        shock_map or {},
        target=target,
        ticker=ticker,
        macro_lag_days=0,
        scenario_name=scenario_name or "geopolitical_conflict",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stress_file = OUTPUT_DIR / "stress_test.json"
    serializable_result = _to_serializable(result)
    stress_file.write_text(
        json.dumps({"value": serializable_result}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_file = OUTPUT_DIR / "analysis_results.json"
    if summary_file.exists():
        try:
            summary_payload: Dict[str, Any] = json.loads(summary_file.read_text(encoding="utf-8"))
        except Exception:
            summary_payload = {}
    else:
        summary_payload = {}

    results_payload = summary_payload.get("results", {})
    if not isinstance(results_payload, dict):
        results_payload = {}
    artifacts_payload = summary_payload.get("artifacts", {})
    if not isinstance(artifacts_payload, dict):
        artifacts_payload = {}

    results_payload["stress_test"] = serializable_result
    artifacts_payload["stress_test"] = str(stress_file)

    summary_payload["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary_payload["results"] = results_payload
    summary_payload["result_keys"] = sorted(results_payload.keys())
    summary_payload["artifacts"] = artifacts_payload

    summary_file.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return result


def run_gold_analyses_only(progress_bar: Any = None) -> tuple[bool, str]:
    """Re-run only the Gold layer analyses using existing Silver data (no Bronze/Silver fetch)."""
    bootstrap_env_from_secrets(override=True)
    env = os.environ.copy()
    env["ENVIRONMENT"] = "actual"
    env["PIPELINE_RESUME_FROM_CHECKPOINT"] = "1"  # skip Bronze/Silver if already checkpointed
    cmd = [sys.executable, "src/main.py"]
    estimated_seconds = 180
    stages = [
        (0.00, "Initialising Gold layer..."),
        (0.15, "Loading Silver data..."),
        (0.35, "Running analyses (lag, elasticity, forecast)..."),
        (0.65, "Running stress test and Monte Carlo..."),
        (0.85, "Running governance and feature checks..."),
        (0.95, "Exporting results..."),
    ]

    log_path: Path | None = None
    proc = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            errors="ignore",
            delete=False,
            suffix="_gold_only.log",
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
        except (FileNotFoundError, OSError) as exc:
            raise PipelineSubprocessError(f"Failed to start Gold-only subprocess: {exc}") from exc

        start = time.time()
        stage_i = 0
        while proc.poll() is None:
            elapsed = time.time() - start
            pct = min(elapsed / estimated_seconds, 0.97)
            while stage_i < len(stages) - 1 and pct >= stages[stage_i + 1][0]:
                stage_i += 1
            if progress_bar is not None:
                if elapsed > estimated_seconds:
                    mins = int(elapsed / 60)
                    secs = int(elapsed % 60)
                    msg = f"97% — Gold analyses still running ({mins}m {secs:02d}s elapsed) — please wait..."
                else:
                    msg = f"{int(pct * 100)}% — {stages[stage_i][1]}"
                progress_bar.progress(pct, text=msg)
            time.sleep(0.5)

        output = log_path.read_text(encoding="utf-8", errors="ignore")
        ok = proc.returncode == 0
        if progress_bar is not None:
            progress_bar.progress(1.0, text="100% — Completed!" if ok else "100% — Failed")
        return ok, output.strip()
    except (PipelineSubprocessError, PipelineProgressTrackingError):
        raise
    except Exception as exc:
        raise PipelineExecutionError(f"Unexpected Gold-only error: {exc}") from exc
    finally:
        if log_path is not None:
            try:
                log_path.unlink(missing_ok=True)
            except OSError:
                pass


def run_optimizer_background(
    target_score: float = 94.0,
    max_iterations: int = 10,
    progress_bar: Any = None,
) -> tuple[bool, str]:
    """Run the automated optimization loop as a subprocess.

    This function is intentionally NOT exported to the regular UI nav and is
    only called from the owner-gated sidebar section in app.py.
    The subprocess runs optimizer.py which internally uses ApprovalGateway for
    every code mutation; approvals must be granted via terminal or the queue
    file at output/.optimizer/approval_queue.json.
    """
    env = os.environ.copy()
    env["OPTIMIZER_TARGET_SCORE"] = str(target_score)
    env["OPTIMIZER_MAX_ITERATIONS"] = str(max_iterations)
    cmd = [
        sys.executable,
        "src/optimizer.py",
        "--target-score",
        str(target_score),
        "--max-iterations",
        str(max_iterations),
    ]
    estimated_seconds = 60 * max_iterations  # rough upper bound
    stages = [
        (0.00, "Initialising optimizer..."),
        (0.10, "Running pipeline iteration 1..."),
        (0.30, "Diagnosing iteration results..."),
        (0.60, "Applying approved adjustments..."),
        (0.85, "Running final pipeline pass..."),
        (0.95, "Writing optimizer report..."),
    ]

    log_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            errors="ignore",
            delete=False,
            suffix="_optimizer.log",
        ) as tmp:
            log_path = Path(tmp.name)

        with log_path.open("w", encoding="utf-8", errors="ignore") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )

        start = time.time()
        stage_i = 0
        while proc.poll() is None:
            elapsed = time.time() - start
            pct = min(elapsed / estimated_seconds, 0.97)
            while stage_i < len(stages) - 1 and pct >= stages[stage_i + 1][0]:
                stage_i += 1
            if progress_bar is not None:
                progress_bar.progress(pct, text=f"{int(pct * 100)}% — {stages[stage_i][1]}")
            time.sleep(1.0)

        output = log_path.read_text(encoding="utf-8", errors="ignore")
        ok = proc.returncode == 0
        if progress_bar is not None:
            progress_bar.progress(1.0, text="100% — Done!" if ok else "100% — Failed")
        return ok, output.strip()
    except Exception as exc:
        return False, f"Optimizer subprocess error: {exc}"
    finally:
        if log_path is not None:
            try:
                log_path.unlink(missing_ok=True)
            except OSError:
                pass


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
    "run_gold_analyses_only",
    "show_pipeline_failure",
    "run_and_cache_audit",
    "get_audit_report",
    "rerun_stress_test_only",
    "run_optimizer_background",
    "clear_all_run_history",
]
