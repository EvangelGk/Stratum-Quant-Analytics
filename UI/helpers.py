from __future__ import annotations

import importlib
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import LOGS_DIR, UI_SNAPSHOT_PATH, get_active_paths


def _paths() -> dict[str, Any]:
    return get_active_paths()


def import_first(*module_names: str) -> Any:
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    return None


@lru_cache(maxsize=128)
def _read_json_cached(path_str: str, mtime_ns: int) -> dict[str, Any]:
    _ = mtime_ns  # part of cache key for invalidation after file updates
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json_cached(str(path), path.stat().st_mtime_ns)


def clear_file_caches() -> None:
    _read_json_cached.cache_clear()


def _iter_watched_artifact_files() -> list[Path]:
    files: list[Path] = []
    output_dir = _paths()["output"]
    processed_dir = _paths()["processed"]
    gold_dir = _paths()["gold"]

    if output_dir.exists():
        for pattern in ("*.json", "*.csv", "*.parquet"):
            files.extend([p for p in output_dir.glob(pattern) if p.is_file()])

    quality_report = processed_dir / "quality" / "quality_report.json"
    if quality_report.exists():
        files.append(quality_report)

    master_table = gold_dir / "master_table.parquet"
    if master_table.exists():
        files.append(master_table)

    if LOGS_DIR.exists():
        files.extend([p for p in LOGS_DIR.glob("session_summary_*.json") if p.is_file()])

    unique = {str(path.resolve()): path for path in files}
    return sorted(unique.values(), key=lambda path: str(path).lower())


def compute_artifact_signature() -> str:
    rows: list[dict[str, int | str]] = []
    for path in _iter_watched_artifact_files():
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append(
            {
                "path": str(path),
                "mtime_ns": int(stat.st_mtime_ns),
                "size": int(stat.st_size),
            }
        )
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


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

    output_dir = _paths()["output"]
    processed_dir = _paths()["processed"]
    output_files = sorted([f for f in output_dir.glob("*") if f.is_file()]) if output_dir.exists() else []
    summary = read_json(output_dir / "analysis_results.json")
    quality = read_json(processed_dir / "quality" / "quality_report.json")
    latest_session = load_session_history(limit=1)
    latest = latest_session[-1] if latest_session else {}

    snapshots.append(
        {
            "ts": datetime.now().isoformat(),
            "session_file": latest.get("__file"),
            "output_files": [{"name": f.name, "size": f.stat().st_size} for f in output_files],
            "result_keys": summary.get("result_keys", []),
            "governance_risk": (summary.get("results", {}).get("governance_report", {}) or {}).get("model_risk_score"),
            "quality_failed_files": sum(1 for v in (quality.get("files", {}) or {}).values() if isinstance(v, dict) and v.get("status") == "failed"),
        }
    )
    snapshots = snapshots[-120:]
    UI_SNAPSHOT_PATH.write_text(json.dumps({"runs": snapshots}, indent=2), encoding="utf-8")


def compute_data_health() -> dict[str, Any]:
    quality = read_json(_paths()["processed"] / "quality" / "quality_report.json")
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
    output_dir = _paths()["output"]
    summary = read_json(output_dir / "analysis_results.json")
    results = summary.get("results", {}) if isinstance(summary, dict) else {}
    lines: list[str] = []

    gov_path = output_dir / "governance_report.json"
    gov = read_json(gov_path).get("value", {}) if gov_path.exists() else results.get("governance_report", {})
    if isinstance(gov, dict):
        score = gov.get("model_risk_score")
        oos = (gov.get("out_of_sample") or {}).get("r2")
        if isinstance(score, (int, float)):
            lines.append(f"Governance model risk score: {score:.3f} (lower is safer).")
        if isinstance(oos, (int, float)):
            lines.append(f"Out-of-sample R2: {oos:.3f} (track trend over runs).")

    ela = results.get("elasticity")
    if isinstance(ela, dict):
        static_ela = ela.get("static_elasticity")
    else:
        static_ela = ela
    if isinstance(static_ela, (int, float)):
        direction = "increases" if static_ela >= 0 else "decreases"
        lines.append(f"Elasticity: {static_ela:.3f} (returns tend to {direction} as factor rises).")

    if not lines:
        lines.append("No explainability signals available yet.")
    return lines


def build_executive_report_html() -> str:
    health = compute_data_health()
    diff = build_run_comparison()
    explain = build_explainability_lines()
    summary = read_json(_paths()["output"] / "analysis_results.json")
    keys = summary.get("result_keys", []) if isinstance(summary, dict) else []
    results = summary.get("results", {}) if isinstance(summary, dict) else {}

    analysis_rows: list[str] = []
    if isinstance(results, dict):
        for key in keys:
            value = results.get(key)
            status = "available"
            highlight = ""
            if isinstance(value, str) and value.startswith("blocked_by_governance_gate"):
                status = "blocked"
                highlight = "Blocked by governance gate"
            elif isinstance(value, dict):
                report_val = value.get("value", value)
                if isinstance(report_val, dict):
                    if key == "governance_report":
                        oos = (report_val.get("out_of_sample") or {}).get("r2")
                        risk = report_val.get("model_risk_score")
                        if isinstance(oos, (int, float)) and isinstance(risk, (int, float)):
                            highlight = f"OOS R²={oos:.4f}, Risk={risk:.3f}"
                    elif key == "governance_gate":
                        passed = report_val.get("passed")
                        severity = report_val.get("severity")
                        highlight = f"Gate={'PASS' if passed else 'BLOCK'} ({severity})"
                    elif key == "correlation_matrix":
                        shape = report_val.get("shape", [0, 0])
                        if isinstance(shape, list) and len(shape) == 2:
                            highlight = f"Matrix {shape[0]}x{shape[1]}"
            elif isinstance(value, (int, float)):
                highlight = f"Value={value:.4f}" if isinstance(value, float) else f"Value={value}"
            analysis_rows.append(f"<tr><td>{key}</td><td>{status}</td><td>{highlight or '-'}</td></tr>")

    diff_html = "Not enough run history yet."
    if diff.get("status") == "ok":
        diff_html = (
            f"Runtime: {diff['duration_prev']:.2f}s -> {diff['duration_curr']:.2f}s "
            f"(Delta {diff['duration_delta']:+.2f}s)<br>"
            f"Operations: {diff['ops_prev']} -> {diff['ops_curr']} "
            f"(Delta {diff['ops_delta']:+d})"
        )

    explain_html = "".join([f"<li>{line}</li>" for line in explain])
    analysis_table_html = (
        "<table><thead><tr><th>Analysis</th><th>Status</th><th>Key note</th></tr></thead>"
        f"<tbody>{''.join(analysis_rows) if analysis_rows else '<tr><td colspan=3>No analyses found.</td></tr>'}</tbody></table>"
    )

    runtime_delta_text = f"{diff['duration_delta']:+.2f}s" if diff.get("status") == "ok" else "N/A"

    return f"""
<html>
    <head>
        <meta charset='utf-8'>
        <title>STRATUM QUANT ANALYTICS Executive Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; background: #f4f7fb; color: #1f2937; }}
            .page {{ max-width: 980px; margin: 24px auto; padding: 0 12px; }}
            .hero {{ background: linear-gradient(135deg, #0f766e, #164e63); color: #fff; border-radius: 16px; padding: 22px; }}
            .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; }}
            .card {{ background: #fff; border-radius: 12px; padding: 14px; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08); }}
            h1, h2, h3 {{ margin: 0 0 8px 0; }}
            ul {{ margin-top: 6px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 8px; background: #fff; border-radius: 12px; overflow: hidden; }}
            th, td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 10px; font-size: 14px; }}
            th {{ background: #ecfeff; }}
            .section {{ margin-top: 16px; }}
            .mono {{ font-family: ui-monospace, Menlo, Consolas, monospace; }}
        </style>
    </head>
    <body>
        <div class='page'>
            <div class='hero'>
                <h1>STRATUM QUANT ANALYTICS Executive Report</h1>
                <div>Generated at: <span class='mono'>{datetime.now().isoformat()}</span></div>
            </div>

            <div class='cards'>
                <div class='card'><h3>Data Health</h3><div><b>{health["score"]}/100</b> ({health["status"]})</div></div>
                <div class='card'><h3>Analyses</h3><div><b>{len(keys)}</b> generated outputs</div></div>
                <div class='card'><h3>Runtime Delta</h3><div>{runtime_delta_text}</div></div>
            </div>

            <div class='section'>
                <h2>Run Comparison</h2>
                <p>{diff_html}</p>
            </div>

            <div class='section'>
                <h2>Explainability</h2>
                <ul>{explain_html}</ul>
            </div>

            <div class='section'>
                <h2>Analysis Status Board</h2>
                {analysis_table_html}
            </div>
        </div>
  </body>
</html>
"""


def build_executive_report_text() -> str:
    health = compute_data_health()
    diff = build_run_comparison()
    summary = read_json(_paths()["output"] / "analysis_results.json")
    keys = summary.get("result_keys", []) if isinstance(summary, dict) else []
    lines = [
        "STRATUM QUANT ANALYTICS - Human Report Snapshot",
        f"Generated: {datetime.now().isoformat()}",
        "",
        f"Data Health: {health['score']}/100 ({health['status']})",
        f"Analyses generated: {len(keys)}",
    ]
    if diff.get("status") == "ok":
        lines.append(f"Runtime: {diff['duration_prev']:.2f}s -> {diff['duration_curr']:.2f}s (delta {diff['duration_delta']:+.2f}s)")
    lines.append("")
    lines.append("Explainability")
    for line in build_explainability_lines():
        lines.append(f"- {line}")
    lines.append("")
    lines.append("Analyses")
    for key in keys:
        lines.append(f"- {key}")
    return "\n".join(lines)


def persist_human_report_files() -> dict[str, str]:
    reports_dir = _paths()["output"] / "human_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    html = build_executive_report_html()
    text_report = build_executive_report_text()

    html_latest = reports_dir / "executive_report_latest.html"
    txt_latest = reports_dir / "executive_report_latest.txt"
    html_versioned = reports_dir / f"executive_report_{stamp}.html"
    txt_versioned = reports_dir / f"executive_report_{stamp}.txt"

    html_latest.write_text(html, encoding="utf-8")
    txt_latest.write_text(text_report, encoding="utf-8")
    html_versioned.write_text(html, encoding="utf-8")
    txt_versioned.write_text(text_report, encoding="utf-8")

    return {
        "reports_dir": str(reports_dir),
        "html_latest": str(html_latest),
        "txt_latest": str(txt_latest),
        "html_versioned": str(html_versioned),
        "txt_versioned": str(txt_versioned),
    }


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
    summary = read_json(_paths()["output"] / "analysis_results.json")
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
