from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import LOGS_DIR, OUTPUT_DIR, PROCESSED_DIR, UI_SNAPSHOT_PATH


def import_first(*module_names: str) -> Any:
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    return None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_session_files() -> list[Path]:
    return sorted(LOGS_DIR.glob("session_summary_*.json"), key=lambda p: p.stat().st_mtime)


def load_session_history(limit: int = 30) -> list[dict[str, Any]]:
    files = list_session_files()
    if limit > 0:
        files = files[-limit:]
    history: list[dict[str, Any]] = []
    for f in files:
        payload = read_json(f)
        if payload:
            payload["__file"] = f.name
            history.append(payload)
    return history


def count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.glob(pattern) if p.is_file())


def record_ui_snapshot() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = read_json(UI_SNAPSHOT_PATH).get("runs", [])
    if not isinstance(snapshots, list):
        snapshots = []

    output_files = sorted([f for f in OUTPUT_DIR.glob("*") if f.is_file()]) if OUTPUT_DIR.exists() else []
    summary = read_json(OUTPUT_DIR / "analysis_results.json")
    quality = read_json(PROCESSED_DIR / "quality" / "quality_report.json")
    latest_session = load_session_history(limit=1)
    latest = latest_session[-1] if latest_session else {}

    snapshots.append(
        {
            "ts": datetime.now().isoformat(),
            "session_file": latest.get("__file"),
            "output_files": [{"name": f.name, "size": f.stat().st_size} for f in output_files],
            "result_keys": summary.get("result_keys", []),
            "governance_risk": (summary.get("results", {}).get("governance_report", {}) or {}).get("model_risk_score"),
            "quality_failed_files": sum(
                1
                for v in (quality.get("files", {}) or {}).values()
                if isinstance(v, dict) and v.get("status") == "failed"
            ),
        }
    )
    snapshots = snapshots[-120:]
    UI_SNAPSHOT_PATH.write_text(json.dumps({"runs": snapshots}, indent=2), encoding="utf-8")


def compute_data_health() -> dict[str, Any]:
    quality = read_json(PROCESSED_DIR / "quality" / "quality_report.json")
    files = quality.get("files", {}) if isinstance(quality, dict) else {}
    summary = quality.get("summary", {}) if isinstance(quality, dict) else {}

    if not files:
        return {
            "score": 0.0,
            "status": "no_data",
            "details": "No quality_report files detected yet.",
            "null_penalty": 0.0,
            "outlier_penalty": 0.0,
            "schema_penalty": 0.0,
            "freshness_penalty": 0.0,
        }

    total_files = len(files)
    failed_files = sum(1 for v in files.values() if isinstance(v, dict) and v.get("status") == "failed")

    total_rows = 0
    total_nulls = 0
    outliers = 0
    latest_processed_at = None
    for v in files.values():
        if not isinstance(v, dict):
            continue
        total_rows += int(v.get("final_rows", 0) or 0)
        total_nulls += int(v.get("final_nulls", 0) or 0)
        outliers += int(v.get("outliers_clipped", 0) or 0)
        ts = v.get("processed_at")
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts)
                if latest_processed_at is None or dt > latest_processed_at:
                    latest_processed_at = dt
            except ValueError:
                pass

    null_ratio = (total_nulls / total_rows) if total_rows > 0 else 0.0
    outlier_ratio = (outliers / total_rows) if total_rows > 0 else 0.0
    failed_ratio = (failed_files / total_files) if total_files > 0 else 1.0
    missing_sources = summary.get("missing_sources", []) if isinstance(summary, dict) else []

    age_hours = 999.0
    if latest_processed_at is not None:
        age_hours = (datetime.now() - latest_processed_at).total_seconds() / 3600.0

    null_penalty = min(null_ratio * 100.0, 25.0)
    outlier_penalty = min(outlier_ratio * 120.0, 20.0)
    schema_penalty = min(failed_ratio * 50.0, 50.0)
    source_penalty = min(float(len(missing_sources)) * 15.0, 45.0)
    freshness_penalty = 0.0 if age_hours <= 24 else min((age_hours - 24) * 0.8, 20.0)

    score = max(0.0, 100.0 - (null_penalty + outlier_penalty + schema_penalty + source_penalty + freshness_penalty))

    status = "good"
    if score < 70:
        status = "warning"
    if score < 50:
        status = "critical"

    return {
        "score": round(score, 2),
        "status": status,
        "details": f"rows={total_rows}, final_nulls={total_nulls}, outliers={outliers}, missing_sources={missing_sources}",
        "null_penalty": round(null_penalty, 2),
        "outlier_penalty": round(outlier_penalty, 2),
        "schema_penalty": round(schema_penalty, 2),
        "freshness_penalty": round(freshness_penalty, 2),
    }


def build_run_comparison() -> dict[str, Any]:
    history = load_session_history(limit=2)
    if len(history) < 2:
        return {"status": "insufficient", "message": "Need at least 2 runs for comparison."}

    prev = history[-2]
    curr = history[-1]

    prev_info = prev.get("session_info", {}) or {}
    curr_info = curr.get("session_info", {}) or {}

    prev_duration = float(prev_info.get("total_duration_seconds", 0.0) or 0.0)
    curr_duration = float(curr_info.get("total_duration_seconds", 0.0) or 0.0)
    prev_ops = int(prev_info.get("total_operations", 0) or 0)
    curr_ops = int(curr_info.get("total_operations", 0) or 0)

    return {
        "status": "ok",
        "duration_prev": prev_duration,
        "duration_curr": curr_duration,
        "duration_delta": curr_duration - prev_duration,
        "ops_prev": prev_ops,
        "ops_curr": curr_ops,
        "ops_delta": curr_ops - prev_ops,
    }


def build_explainability_lines() -> list[str]:
    summary = read_json(OUTPUT_DIR / "analysis_results.json")
    results = summary.get("results", {}) if isinstance(summary, dict) else {}
    lines: list[str] = []

    gov = results.get("governance_report", {})
    if isinstance(gov, dict):
        score = gov.get("model_risk_score")
        oos = (gov.get("out_of_sample") or {}).get("r2")
        if isinstance(score, (int, float)):
            lines.append(f"Governance model risk score: {score:.3f} (lower is safer).")
        if isinstance(oos, (int, float)):
            lines.append(f"Out-of-sample R2: {oos:.3f} (track trend over runs).")

    ela = results.get("elasticity")
    if isinstance(ela, (int, float)):
        direction = "increases" if ela >= 0 else "decreases"
        lines.append(f"Elasticity: {ela:.3f} (returns tend to {direction} as factor rises).")

    if not lines:
        lines.append("No explainability signals available yet.")
    return lines


def build_executive_report_html() -> str:
    health = compute_data_health()
    diff = build_run_comparison()
    explain = build_explainability_lines()
    summary = read_json(OUTPUT_DIR / "analysis_results.json")
    keys = summary.get("result_keys", []) if isinstance(summary, dict) else []

    diff_html = "Not enough run history yet."
    if diff.get("status") == "ok":
        diff_html = (
            f"Runtime: {diff['duration_prev']:.2f}s -> {diff['duration_curr']:.2f}s "
            f"(Delta {diff['duration_delta']:+.2f}s)<br>"
            f"Operations: {diff['ops_prev']} -> {diff['ops_curr']} "
            f"(Delta {diff['ops_delta']:+d})"
        )

    explain_html = "".join([f"<li>{line}</li>" for line in explain])

    return f"""
<html>
  <head><meta charset='utf-8'><title>Scenario Planner Executive Report</title></head>
  <body style='font-family: Arial; margin: 24px;'>
    <h1>Scenario Planner Executive Report</h1>
    <p>Generated at: {datetime.now().isoformat()}</p>
    <h3>Summary</h3>
    <p><b>Data Health Score:</b> {health['score']}/100 ({health['status']})</p>
    <p><b>Analysis keys:</b> {', '.join(keys)}</p>
    <h3>Run Comparison</h3>
    <p>{diff_html}</p>
    <h3>Explainability</h3>
    <ul>{explain_html}</ul>
  </body>
</html>
"""


def build_smart_alerts() -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    health = compute_data_health()
    if health["score"] < 70:
        alerts.append(
            {
                "severity": "critical" if health["score"] < 50 else "warning",
                "title": "Data quality dropped",
                "message": f"Data Health Score is {health['score']} (target >= 70).",
            }
        )

    diff = build_run_comparison()
    if diff.get("status") == "ok":
        prev_d = float(diff["duration_prev"])
        curr_d = float(diff["duration_curr"])
        if prev_d > 0 and curr_d > prev_d * 1.4:
            alerts.append(
                {
                    "severity": "warning",
                    "title": "Pipeline slowed down",
                    "message": f"Runtime increased from {prev_d:.1f}s to {curr_d:.1f}s.",
                }
            )
    return alerts


def correlation_strength_from_output() -> float | None:
    summary = read_json(OUTPUT_DIR / "analysis_results.json")
    corr = summary.get("results", {}).get("correlation_matrix", {})
    if not isinstance(corr, dict) or not isinstance(corr.get("data"), list):
        return None
    try:
        curr_df = pd.DataFrame(corr["data"])
        numeric_cols = curr_df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            return None
        return float(curr_df[numeric_cols].abs().mean().mean())
    except Exception:
        return None
