from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv


def _import_first(*module_names: str) -> Any:
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    return None


_streamlit_exc = _import_first(
    "src.exceptions.StreamlitExceptions", "exceptions.StreamlitExceptions"
)

if _streamlit_exc is not None:
    StreamlitError = _streamlit_exc.StreamlitError
    PipelineExecutionError = _streamlit_exc.PipelineExecutionError
    PipelineSubprocessError = _streamlit_exc.PipelineSubprocessError
    PipelineProgressTrackingError = _streamlit_exc.PipelineProgressTrackingError
    DataFileNotFoundError = _streamlit_exc.DataFileNotFoundError
    DataFileReadError = _streamlit_exc.DataFileReadError
    JSONParseError = _streamlit_exc.JSONParseError
    SessionLoadError = _streamlit_exc.SessionLoadError
    SessionSnapshotError = _streamlit_exc.SessionSnapshotError
    HealthScoreCalculationError = _streamlit_exc.HealthScoreCalculationError
    AlertGenerationError = _streamlit_exc.AlertGenerationError
    RunComparisonError = _streamlit_exc.RunComparisonError
    ReportGenerationError = _streamlit_exc.ReportGenerationError
    LoggerModuleError = _streamlit_exc.LoggerModuleError
else:
    # Fallback if exceptions module is not available.
    class StreamlitError(Exception):
        pass

    class PipelineExecutionError(StreamlitError):
        pass

    class PipelineSubprocessError(PipelineExecutionError):
        pass

    class PipelineProgressTrackingError(PipelineExecutionError):
        pass

    class DataFileNotFoundError(StreamlitError):
        pass

    class DataFileReadError(StreamlitError):
        pass

    class JSONParseError(StreamlitError):
        pass

    class SessionLoadError(StreamlitError):
        pass

    class SessionSnapshotError(StreamlitError):
        pass

    class HealthScoreCalculationError(StreamlitError):
        pass

    class AlertGenerationError(StreamlitError):
        pass

    class RunComparisonError(StreamlitError):
        pass

    class ReportGenerationError(StreamlitError):
        pass

    class LoggerModuleError(StreamlitError):
        pass


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Ensure src/ is on sys.path so logger.Messages.* imports work when
# Streamlit is launched from the project root (sys.path does not include src/).
_src_path = str(ROOT / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
_UI_USER_ID = os.getenv("DATA_USER_ID", "default").strip() or "default"
_SAFE_UI_USER = "".join(
    ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in _UI_USER_ID
) or "default"
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
    "Viewer": {
        "can_run": False,
        "can_download": False,
        "can_schedule": False,
    },
    "Analyst": {
        "can_run": True,
        "can_download": True,
        "can_schedule": False,
    },
    "Admin": {
        "can_run": True,
        "can_download": True,
        "can_schedule": True,
    },
}

# ---------------------------------------------------------------------------
# Help text sourced from logger/Messages — adapted for UI display
# ---------------------------------------------------------------------------
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
            "Log-returns and macro factors combined",
            "Governance decisions and risk scores applied",
            "Ready for correlation analysis, forecasting, stress-tests, …",
        ],
        "note": "This is what the Analytics tab visualises.",
    },
}

ANALYSIS_HELP: dict[str, dict[str, str]] = {
    "correlation_matrix": {
        "title": "Correlation Matrix",
        "what": "Measures linear relationships between every pair of variables (assets + macro factors).",
        "read": "Values range from −1 (perfect negative) to +1 (perfect positive). Look for values above ±0.7 — those signal strong dependencies worth watching.",
        "use": "Identify which macro factors move together with a stock, or which assets offset each other in a portfolio.",
    },
    "governance_report": {
        "title": "Governance Report",
        "what": "A record of every automated data-quality decision made during the pipeline run.",
        "read": "Each entry shows whether a dataset was *approved* or *flagged*, the risk score (0–100), and the reason.",
        "use": "Audit trail — tells you which data entered the analytical layer and why.",
    },
    "elasticity": {
        "title": "Elasticity Analysis",
        "what": "How much do asset returns change for a 1 % change in a macro factor?",
        "read": "Elasticity > 1: asset is volatile to that factor. Elasticity < 1: asset barely reacts. Negative: they move in opposite directions.",
        "use": "Spot which stocks are most exposed to inflation, energy prices, or interest rates.",
    },
    "lag_analysis": {
        "title": "Lag Analysis",
        "what": "Shows *delayed* effects — how a macro change today ripples through asset prices over the following periods.",
        "read": "Lag 0 = immediate. Lag 1 = one period later. The lag with the highest correlation is the most predictive delay.",
        "use": "Build trading signals or risk alerts that anticipate market moves before they fully materialise.",
    },
    "monte_carlo": {
        "title": "Monte Carlo Simulation",
        "what": "Generates thousands of possible future price paths using historical volatility and drift.",
        "read": "The shaded band shows the 10th–90th percentile range of outcomes. The central line is the median expected path.",
        "use": "Assess tail risk — how bad could things get in the worst 10 % of scenarios?",
    },
    "stress_test": {
        "title": "Stress Test",
        "what": "Applies extreme but plausible market shocks and measures the portfolio impact.",
        "read": "Each scenario (e.g. recession, rate spike) shows the estimated maximum drawdown.",
        "use": "Worst-case planning — decide how much risk you can absorb before adjusting positions.",
    },
    "sensitivity_reg": {
        "title": "Sensitivity Regression",
        "what": "Multivariate regression quantifying each macro factor's contribution to asset returns.",
        "read": "Positive coefficient: factor drives returns up. Negative: drives returns down. R² shows how much of the variance is explained.",
        "use": "Pin down the dominant drivers of each asset so you can hedge them directly.",
    },
    "forecasting": {
        "title": "Time-Series Forecasting",
        "what": "ARIMA model projecting future values based on historical patterns.",
        "read": "Solid line = forecast, shaded area = confidence interval. Wider intervals = higher uncertainty further out.",
        "use": "Trend analysis for planning — set realistic expectations for the next N periods.",
    },
    "auto_ml": {
        "title": "Auto ML Regression",
        "what": "Automatically compares multiple machine-learning models and picks the best predictor.",
        "read": "The winning model, its performance metrics (RMSE, R²), and the most important input features are reported.",
        "use": "Predictive modelling — understand which features matter most and get the best out-of-sample forecast.",
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
        "Builds master table, runs all 9 analyses, applies governance decisions.",
    ),
    (
        "Export results",
        "Saves analysis files to output/<user_id>/ and governance decisions to data/users/<user_id>/gold/governance/.",
    ),
]


_directions_mod = _import_first(
    "logger.Messages.DirectionsMess",
    "src.logger.Messages.DirectionsMess",
)
_main_mod = _import_first(
    "logger.Messages.MainMess",
    "src.logger.Messages.MainMess",
)
_medallion_mod = _import_first(
    "logger.Messages.MedallionMess",
    "src.logger.Messages.MedallionMess",
)
_fetchers_mod = _import_first(
    "logger.Messages.FetchersMess",
    "src.logger.Messages.FetchersMess",
)


def _normalize_message(msg: Any) -> str:
    if not isinstance(msg, str):
        return ""
    cleaned = msg.strip()
    if not cleaned:
        return ""
    # Keep formatting readable in Streamlit markdown panels.
    return cleaned.replace("\r\n", "\n")


def _render_logger_message(title: str, msg: Any) -> None:
    text = _normalize_message(msg)
    if not text:
        return
    st.markdown(f"**{title}**")
    st.caption(text)


st.set_page_config(
    page_title="Welcome to Scenario Planner!",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
            html, body, [class*="css"] {
                font-family: 'Space Grotesk', sans-serif;
            }
            .hero {
                background: linear-gradient(
                    135deg,
                    #0f172a 0%,
                    #1e293b 45%,
                    #0b6e4f 100%
                );
                padding: 1.2rem 1.4rem;
                border-radius: 16px;
                color: #f8fafc;
                margin-bottom: 1rem;
                border: 1px solid rgba(255, 255, 255, 0.12);
            }
            .hero h1 {
                margin: 0;
                font-size: 1.9rem;
                letter-spacing: 0.3px;
            }
            .hero p {
                margin: 0.4rem 0 0;
                color: #cbd5e1;
                font-size: 0.95rem;
            }
            .metric-card {
                background: #0b1220;
                border: 1px solid #1f2937;
                border-radius: 14px;
                padding: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    """Safely read JSON file, return empty dict on error."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise JSONParseError(f"Failed to parse JSON from {path.name}: {e}") from e
    except OSError as e:
        raise DataFileReadError(f"Failed to read file {path}: {e}") from e


def _list_session_files() -> list[Path]:
    return sorted(
        LOGS_DIR.glob("session_summary_*.json"), key=lambda p: p.stat().st_mtime
    )


def _load_session_history(limit: int = 30) -> list[dict[str, Any]]:
    """Load session history from logs, handle errors gracefully."""
    try:
        files = _list_session_files()
        if limit > 0:
            files = files[-limit:]
        history: list[dict[str, Any]] = []
        for f in files:
            try:
                payload = read_json(f)
                if payload:
                    payload["__file"] = f.name
                    history.append(payload)
            except (JSONParseError, DataFileReadError):
                # Skip corrupted files, continue with next
                continue
        return history
    except Exception as e:
        raise SessionLoadError(f"Failed to load session history: {e}") from e


def _record_ui_snapshot() -> None:
    """Record UI run snapshot, handle file errors gracefully."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        snapshots = read_json(UI_SNAPSHOT_PATH).get("runs", [])
    except (JSONParseError, DataFileReadError):
        snapshots = []

    if not isinstance(snapshots, list):
        snapshots = []

    try:
        output_files = (
            sorted([f for f in OUTPUT_DIR.glob("*") if f.is_file()])
            if OUTPUT_DIR.exists()
            else []
        )
        summary = read_json(OUTPUT_DIR / "analysis_results.json")
        quality = read_json(PROCESSED_DIR / "quality_report.json")
        latest_session = _load_session_history(limit=1)
        latest = latest_session[-1] if latest_session else {}

        snapshots.append(
            {
                "ts": datetime.now().isoformat(),
                "session_file": latest.get("__file"),
                "output_files": [
                    {"name": f.name, "size": f.stat().st_size} for f in output_files
                ],
                "result_keys": summary.get("result_keys", []),
                "governance_risk": (
                    summary.get("results", {}).get("governance_report", {}) or {}
                ).get("model_risk_score"),
                "quality_failed_files": sum(
                    1
                    for v in (quality.get("files", {}) or {}).values()
                    if isinstance(v, dict) and v.get("status") == "failed"
                ),
            }
        )

        snapshots = snapshots[-120:]
        with UI_SNAPSHOT_PATH.open("w", encoding="utf-8") as f:
            json.dump({"runs": snapshots}, f, indent=2)
    except (DataFileReadError, JSONParseError, SessionLoadError) as e:
        raise SessionSnapshotError(f"Failed to record UI snapshot: {e}") from e


def _compute_data_health() -> dict[str, Any]:
    """Calculate data health score from quality report."""
    try:
        quality = read_json(PROCESSED_DIR / "quality_report.json")
    except (JSONParseError, DataFileReadError):
        raise HealthScoreCalculationError("Failed to read quality_report.json")

    files = quality.get("files", {}) if isinstance(quality, dict) else {}
    summary = quality.get("summary", {}) if isinstance(quality, dict) else {}

    if not files:
        return {
            "score": 0.0,
            "null_penalty": 0.0,
            "outlier_penalty": 0.0,
            "schema_penalty": 0.0,
            "freshness_penalty": 0.0,
            "failed_files": 0,
            "status": "no_data",
            "details": "No quality_report files detected yet.",
        }

    total_files = len(files)
    failed_files = sum(
        1 for v in files.values() if isinstance(v, dict) and v.get("status") == "failed"
    )

    total_rows = 0
    total_nulls = 0
    outliers = 0
    imputed = 0
    latest_processed_at = None

    for v in files.values():
        if not isinstance(v, dict):
            continue
        total_rows += int(v.get("final_rows", 0) or 0)
        total_nulls += int(v.get("final_nulls", 0) or 0)
        outliers += int(v.get("outliers_clipped", 0) or 0)
        imputed += int(v.get("imputed_count", 0) or 0)
        p_at = v.get("processed_at")
        if isinstance(p_at, str):
            try:
                ts = datetime.fromisoformat(p_at)
                if latest_processed_at is None or ts > latest_processed_at:
                    latest_processed_at = ts
            except (ValueError, TypeError):
                # Skip invalid timestamp, continue
                continue

    null_ratio = (total_nulls / total_rows) if total_rows > 0 else 0.0
    outlier_ratio = (outliers / total_rows) if total_rows > 0 else 0.0
    imputed_ratio = (imputed / total_rows) if total_rows > 0 else 1.0
    failed_ratio = (failed_files / total_files) if total_files > 0 else 1.0
    missing_sources = (
        summary.get("missing_sources", []) if isinstance(summary, dict) else []
    )
    source_coverage_penalty = min(float(len(missing_sources)) * 15.0, 45.0)

    age_hours = 999.0
    if latest_processed_at is not None:
        age_hours = (datetime.now() - latest_processed_at).total_seconds() / 3600.0

    null_penalty = min(null_ratio * 100.0, 25.0)
    outlier_penalty = min(outlier_ratio * 120.0, 20.0)
    imputation_penalty = min(imputed_ratio * 80.0, 20.0)
    schema_penalty = min(failed_ratio * 50.0, 50.0)
    zero_rows_penalty = 30.0 if total_rows == 0 else 0.0
    freshness_penalty = 0.0 if age_hours <= 24 else min((age_hours - 24) * 0.8, 20.0)

    score = max(
        0.0,
        100.0
        - (
            null_penalty
            + outlier_penalty
            + imputation_penalty
            + schema_penalty
            + source_coverage_penalty
            + zero_rows_penalty
            + freshness_penalty
        ),
    )

    status = "good"
    if score < 70:
        status = "warning"
    if score < 50:
        status = "critical"

    return {
        "score": round(score, 2),
        "null_penalty": round(null_penalty, 2),
        "outlier_penalty": round(outlier_penalty, 2),
        "imputation_penalty": round(imputation_penalty, 2),
        "schema_penalty": round(schema_penalty, 2),
        "source_coverage_penalty": round(source_coverage_penalty, 2),
        "zero_rows_penalty": round(zero_rows_penalty, 2),
        "freshness_penalty": round(freshness_penalty, 2),
        "missing_sources": missing_sources,
        "failed_files": failed_files,
        "status": status,
        "details": (
            f"rows={total_rows}, final_nulls={total_nulls}, outliers_clipped={outliers}, "
            f"imputed={imputed}, "
            f"missing_sources={missing_sources}, "
            f"age_hours={age_hours:.1f}"
        ),
    }


def _build_smart_alerts() -> list[dict[str, str]]:
    """Generate smart alerts based on health, performance, risk metrics."""
    alerts: list[dict[str, str]] = []

    try:
        health = _compute_data_health()
    except HealthScoreCalculationError:
        return [
            {
                "severity": "error",
                "title": "Health Score Error",
                "message": "Failed to compute health score.",
            }
        ]

    try:
        history = _load_session_history(limit=2)
    except SessionLoadError:
        return [
            {
                "severity": "error",
                "title": "Session Load Error",
                "message": "Failed to load session history.",
            }
        ]

    if health["score"] < 70:
        severity = "critical" if health["score"] < 50 else "warning"
        alerts.append(
            {
                "severity": severity,
                "title": "Data quality dropped",
                "message": f"Data Health Score is {health['score']} (target >= 70).",
            }
        )

    if len(history) >= 2:
        prev = history[-2]
        curr = history[-1]
        prev_dur = float(
            (prev.get("session_info", {}) or {}).get("total_duration_seconds", 0.0)
            or 0.0
        )
        curr_dur = float(
            (curr.get("session_info", {}) or {}).get("total_duration_seconds", 0.0)
            or 0.0
        )
        if prev_dur > 0 and curr_dur > prev_dur * 1.4:
            alerts.append(
                {
                    "severity": "warning",
                    "title": "Pipeline slowed down",
                    "message": f"Runtime increased from {prev_dur:.1f}s to {curr_dur:.1f}s.",
                }
            )

    try:
        summary = read_json(OUTPUT_DIR / "analysis_results.json")
    except (JSONParseError, DataFileReadError):
        summary = {}

    curr_risk = (summary.get("results", {}).get("governance_report", {}) or {}).get(
        "model_risk_score"
    )

    try:
        snapshots = read_json(UI_SNAPSHOT_PATH).get("runs", [])
    except (JSONParseError, DataFileReadError):
        snapshots = []

    if (
        isinstance(snapshots, list)
        and len(snapshots) >= 2
        and isinstance(curr_risk, (float, int))
    ):
        prev_risk = snapshots[-2].get("governance_risk")
        if isinstance(prev_risk, (float, int)):
            delta = float(curr_risk) - float(prev_risk)
            if abs(delta) >= 0.1:
                sev = "critical" if delta > 0.2 else "warning"
                alerts.append(
                    {
                        "severity": sev,
                        "title": "Risk score changed significantly",
                        "message": f"Governance model risk score changed by {delta:+.3f}.",
                    }
                )

    corr = summary.get("results", {}).get("correlation_matrix", {})
    try:
        if isinstance(corr, dict) and isinstance(corr.get("data"), list):
            curr_df = pd.DataFrame(corr["data"])
            numeric_cols = curr_df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                curr_strength = float(curr_df[numeric_cols].abs().mean().mean())
                if isinstance(snapshots, list) and len(snapshots) >= 2:
                    prev_strength = snapshots[-2].get("corr_strength")
                    if (
                        isinstance(prev_strength, (float, int))
                        and abs(curr_strength - float(prev_strength)) > 0.12
                    ):
                        alerts.append(
                            {
                                "severity": "warning",
                                "title": "Correlation regime shift",
                                "message": "Average absolute correlation changed notably vs previous run.",
                            }
                        )
                if isinstance(snapshots, list) and snapshots:
                    snapshots[-1]["corr_strength"] = curr_strength
                    try:
                        with UI_SNAPSHOT_PATH.open("w", encoding="utf-8") as f:
                            json.dump({"runs": snapshots}, f, indent=2)
                    except OSError:
                        pass  # Silently continue if snapshot write fails
    except (ValueError, TypeError, KeyError) as e:
        raise AlertGenerationError(f"Failed to analyze correlations: {e}") from e

    return alerts


def _build_run_comparison() -> dict[str, Any]:
    """Compare current vs previous pipeline run metrics."""
    try:
        history = _load_session_history(limit=2)
    except SessionLoadError as e:
        return {"status": "error", "message": f"Failed to load session history: {e}"}

    if len(history) < 2:
        return {
            "status": "insufficient",
            "message": "Need at least 2 runs for comparison.",
        }

    prev = history[-2]
    curr = history[-1]

    prev_info = prev.get("session_info", {}) or {}
    curr_info = curr.get("session_info", {}) or {}
    prev_duration = float(prev_info.get("total_duration_seconds", 0.0) or 0.0)
    curr_duration = float(curr_info.get("total_duration_seconds", 0.0) or 0.0)

    prev_ops = int(prev_info.get("total_operations", 0) or 0)
    curr_ops = int(curr_info.get("total_operations", 0) or 0)

    try:
        prev_snap = read_json(UI_SNAPSHOT_PATH).get("runs", [])
    except (JSONParseError, DataFileReadError):
        prev_snap = []

    file_diff = {"added": [], "removed": [], "changed": []}
    if isinstance(prev_snap, list) and len(prev_snap) >= 2:
        old_files = {
            f["name"]: f.get("size", 0) for f in prev_snap[-2].get("output_files", [])
        }
        new_files = {
            f["name"]: f.get("size", 0) for f in prev_snap[-1].get("output_files", [])
        }
        file_diff["added"] = sorted(
            [name for name in new_files if name not in old_files]
        )
        file_diff["removed"] = sorted(
            [name for name in old_files if name not in new_files]
        )
        file_diff["changed"] = sorted(
            [
                name
                for name in new_files
                if name in old_files and new_files[name] != old_files[name]
            ]
        )

    try:
        gov_path = GOLD_DIR / "governance"
        if gov_path.exists():
            gov_files = sorted(gov_path.glob("governance_decision_*.json"))
            gov_latest = read_json(gov_files[-1]) if len(gov_files) >= 1 else {}
            gov_prev = read_json(gov_files[-2]) if len(gov_files) >= 2 else {}
        else:
            gov_latest = {}
            gov_prev = {}
    except (JSONParseError, DataFileReadError):
        gov_latest = {}
        gov_prev = {}

    return {
        "status": "ok",
        "duration_prev": prev_duration,
        "duration_curr": curr_duration,
        "duration_delta": curr_duration - prev_duration,
        "ops_prev": prev_ops,
        "ops_curr": curr_ops,
        "ops_delta": curr_ops - prev_ops,
        "governance_prev": gov_prev,
        "governance_curr": gov_latest,
        "output_file_diff": file_diff,
        "run_prev": prev.get("__file", "N/A"),
        "run_curr": curr.get("__file", "N/A"),
    }


def _build_explainability_lines() -> list[str]:
    summary = read_json(OUTPUT_DIR / "analysis_results.json")
    results = summary.get("results", {}) if isinstance(summary, dict) else {}
    lines: list[str] = []

    corr = results.get("correlation_matrix", {})
    if isinstance(corr, dict):
        shape = corr.get("shape", [0, 0])
        lines.append(
            f"Correlation analyzed {shape[0]} variables. Higher absolute correlations indicate stronger co-movement risk."
        )

    gov = results.get("governance_report", {})
    if isinstance(gov, dict):
        score = gov.get("model_risk_score")
        oos = (gov.get("out_of_sample") or {}).get("r2")
        if isinstance(score, (int, float)):
            lines.append(
                f"Governance model risk score is {score:.3f}. Lower is safer; watch trend across runs for drift."
            )
        if isinstance(oos, (int, float)):
            lines.append(
                f"Out-of-sample R² is {oos:.3f}. If this drops, model explanatory power is weakening."
            )

    ela = results.get("elasticity")
    if isinstance(ela, (int, float)):
        direction = "increases" if ela >= 0 else "decreases"
        lines.append(
            f"Elasticity is {ela:.3f}: returns tend to {direction} when the selected macro factor rises."
        )

    if not lines:
        lines.append(
            "No explainability signals available yet. Run pipeline to populate this panel."
        )

    return lines


def _build_executive_report_html() -> str:
    """Generate HTML executive report from pipeline results."""
    try:
        health = _compute_data_health()
    except HealthScoreCalculationError:
        health = {"score": 0.0, "status": "error"}

    try:
        alerts = _build_smart_alerts()
    except AlertGenerationError:
        alerts = [
            {
                "severity": "error",
                "title": "Alert Error",
                "message": "Failed to generate alerts.",
            }
        ]

    try:
        diff = _build_run_comparison()
    except RunComparisonError:
        diff = {"status": "error", "message": "Failed to compare runs."}

    explain = _build_explainability_lines()

    try:
        summary = read_json(OUTPUT_DIR / "analysis_results.json")
    except (JSONParseError, DataFileReadError):
        summary = {"result_keys": []}

    alerts_html = (
        "".join([f"<li><b>{a['title']}</b>: {a['message']}</li>" for a in alerts])
        or "<li>No active alerts.</li>"
    )
    explain_html = "".join([f"<li>{line}</li>" for line in explain])

    diff_html = ""
    if diff.get("status") == "ok":
        diff_html = (
            f"<p><b>Runtime:</b> {diff['duration_prev']:.2f}s → {diff['duration_curr']:.2f}s "
            f"(Δ {diff['duration_delta']:+.2f}s)</p>"
            f"<p><b>Operations:</b> {diff['ops_prev']} → {diff['ops_curr']} "
            f"(Δ {diff['ops_delta']:+d})</p>"
            f"<p><b>Output changes:</b> added={len(diff['output_file_diff']['added'])}, "
            f"removed={len(diff['output_file_diff']['removed'])}, "
            f"changed={len(diff['output_file_diff']['changed'])}</p>"
        )
    else:
        diff_html = "<p>Not enough run history for comparison yet.</p>"

    try:
        return f"""
<html>
  <head>
    <meta charset='utf-8'>
    <title>Scenario Planner Executive Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; color: #0f172a; }}
      h1 {{ color: #0b6e4f; }}
      .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
    </style>
  </head>
  <body>
    <h1>Scenario Planner Executive Report</h1>
    <p>Generated at: {datetime.now().isoformat()}</p>
    <div class='card'>
      <h3>Summary</h3>
      <p><b>Data Health Score:</b> {health["score"]}/100 ({health["status"]})</p>
      <p><b>Available analysis keys:</b> {", ".join(summary.get("result_keys", []))}</p>
    </div>
    <div class='card'>
      <h3>Key Alerts</h3>
      <ul>{alerts_html}</ul>
    </div>
    <div class='card'>
      <h3>Run Comparison</h3>
      {diff_html}
    </div>
    <div class='card'>
      <h3>Explainability (Plain Language)</h3>
      <ul>{explain_html}</ul>
    </div>
    <div class='card'>
      <h3>Recommended Actions</h3>
      <ol>
        <li>Investigate any active alerts and fix upstream data issues.</li>
        <li>Track model risk and out-of-sample R² over time.</li>
        <li>Review changed output files before publishing decisions.</li>
      </ol>
    </div>
  </body>
</html>
"""
    except Exception as e:
        raise ReportGenerationError(f"Failed to generate HTML report: {e}") from e


def _show_pipeline_failure(raw_output: str) -> None:
    """Render a human-readable failure breakdown after a pipeline error."""
    st.error("Pipeline did not complete. Here is what happened:")

    # Try to read quality report for per-entity breakdown
    q_report: dict[str, Any] = {}
    try:
        q_report = read_json(PROCESSED_DIR / "quality" / "quality_report.json")
    except (JSONParseError, DataFileReadError):
        q_report = {}

    files_info = q_report.get("files", {}) if isinstance(q_report, dict) else {}
    summary_info = q_report.get("summary", {}) if isinstance(q_report, dict) else {}
    failed_entities = [
        (name, meta)
        for name, meta in files_info.items()
        if isinstance(meta, dict) and meta.get("status") == "failed"
    ]
    missing_src = (
        summary_info.get("missing_sources", [])
        if isinstance(summary_info, dict)
        else []
    )

    if missing_src:
        st.warning(
            f"**No usable data from:** {', '.join(missing_src)}.  \n"
            "All datasets from these sources were rejected by the quality checks."
        )

    if failed_entities:
        st.markdown(f"**{len(failed_entities)} dataset(s) failed quality checks:**")
        for name, meta in failed_entities:
            err = meta.get("error", "unknown error")
            src = meta.get("source", "?")
            # Translate technical error into plain language
            if "Schema validation failed" in err:
                plain = "Values outside the expected range for this data type."
            elif "numpy string" in err or "string dtypes" in err:
                plain = "Internal data format issue (string type incompatibility)."
            elif "hard stop" in err or "empty" in err.lower():
                plain = "Dataset was empty or too small to process."
            elif "Missing columns" in err or "Schema drift" in err:
                plain = "Dataset is missing required columns (source API changed?)."
            elif "null" in err.lower() or "missing" in err.lower():
                plain = "Too many missing values in the dataset."
            else:
                plain = err[:120]
            st.markdown(f"- **{name}** ({src}): {plain}")
    elif not files_info:
        # No quality report — pipeline likely crashed before Silver ran
        st.info(
            "Quality report not found. The pipeline may have failed during "
            "data fetching (Bronze layer). Check your API keys and internet connection."
        )

    # Dead-letter quick peek
    dl_path = PROCESSED_DIR / "quality" / "dead_letter.jsonl"
    if dl_path.exists():
        try:
            lines = dl_path.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                last = json.loads(lines[-1])
                st.caption(
                    f"Last dead-letter entry: **{last.get('entity')}** — "
                    f"{last.get('error_type')}: {str(last.get('error_message',''))[:100]}"
                )
        except Exception:
            pass

    # Suggestions
    st.markdown("**What to try:**")
    st.markdown(
        "1. Check that `FRED_API_KEY` is set correctly in your `.env` file.\n"
        "2. Confirm you have internet access (Yahoo Finance, World Bank APIs).\n"
        "3. Open the **Data** tab → *processed* layer → `quality/quality_report.json` "
        "for the full per-dataset breakdown.\n"
        "4. Check the **Logs** tab for the session summary."
    )

    with st.expander("Raw pipeline output (technical details)", expanded=False):
        st.text(raw_output[-4000:] if len(raw_output) > 4000 else raw_output)


def run_pipeline(
    mode: str = "sample",
    progress_bar: Any = None,
) -> tuple[bool, str]:
    """Run pipeline with specified mode.

    mode: 'sample' (2-3 min, small dataset) or 'actual' (5-10 min, full dataset)

    Raises:
        PipelineSubprocessError: If subprocess creation fails
        PipelineExecutionError: If pipeline execution fails
    """
    env = os.environ.copy()
    env["ENVIRONMENT"] = mode
    cmd = [sys.executable, "src/main.py"]

    # Estimated total duration for smooth progress animation
    estimated_seconds = 150 if mode == "sample" else 480

    # Named stages mapped to approximate completion fraction
    stages = [
        (0.00, "Starting pipeline..."),
        (0.08, "Checking prerequisites..."),
        (0.15, "Loading configuration..."),
        (0.30, "Fetching data from APIs..."),
        (0.50, "Bronze layer: organizing raw data..."),
        (0.65, "Silver layer: cleaning & validating..."),
        (0.82, "Gold layer: running analyses..."),
        (0.95, "Exporting results to output/..."),
    ]

    log_path = None
    proc = None

    try:
        # Create temp log file
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+",
                encoding="utf-8",
                errors="ignore",
                delete=False,
                suffix="_scenario_planner.log",
            ) as tmp:
                log_path = Path(tmp.name)
        except OSError as e:
            raise PipelineExecutionError(f"Failed to create temp log file: {e}") from e

        try:
            # Start subprocess with file-based logging (avoid PIPE deadlock)
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
            except FileNotFoundError as e:
                raise PipelineSubprocessError(
                    f"Python executable not found: {e}"
                ) from e
            except OSError as e:
                raise PipelineSubprocessError(f"Failed to start subprocess: {e}") from e

            start = time.time()
            stage_i = 0

            # Poll process and update progress bar
            while proc.poll() is None:
                try:
                    elapsed = time.time() - start
                    pct = min(elapsed / estimated_seconds, 0.97)
                    while stage_i < len(stages) - 1 and pct >= stages[stage_i + 1][0]:
                        stage_i += 1
                    label = stages[stage_i][1]
                    if progress_bar is not None:
                        progress_bar.progress(pct, text=f"{int(pct * 100)}% — {label}")
                    time.sleep(0.5)
                except Exception as e:
                    raise PipelineProgressTrackingError(
                        f"Failed to track pipeline progress: {e}"
                    ) from e

            # Read completed output
            try:
                output = log_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as e:
                raise PipelineExecutionError(
                    f"Failed to read pipeline output: {e}"
                ) from e

            ok = proc.returncode == 0
            if progress_bar is not None:
                final_text = "100% — Completed!" if ok else "100% — Failed"
                progress_bar.progress(1.0, text=final_text)
            return ok, output.strip()

        finally:
            # Clean up log file
            if log_path is not None:
                try:
                    log_path.unlink(missing_ok=True)
                except OSError:
                    pass  # Silently continue if cleanup fails

    except (
        PipelineSubprocessError,
        PipelineExecutionError,
        PipelineProgressTrackingError,
    ):
        raise
    except Exception as e:
        raise PipelineExecutionError(f"Unexpected pipeline error: {e}") from e


def count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))


def show_kpis() -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Files", count_files(RAW_DIR, "**/*.parquet"))
    c2.metric("Processed Files", count_files(PROCESSED_DIR, "**/*.parquet"))
    c3.metric("Gold Assets", count_files(GOLD_DIR, "**/*.parquet"))
    c4.metric("Output Artifacts", count_files(OUTPUT_DIR, "*"))


def show_data_tab() -> None:
    st.subheader("Data Lake Explorer")

    with st.expander("Help: How data flows in this app", expanded=False):
        if _fetchers_mod is not None:
            _render_logger_message(
                "Fetcher guide", getattr(_fetchers_mod, "FETCHER_USER_GUIDE", "")
            )
        if _medallion_mod is not None:
            _render_logger_message(
                "Medallion guide",
                getattr(_medallion_mod, "MEDALLION_USER_GUIDE", ""),
            )

    # Layer selector with icons from LAYER_HELP
    selected_layer = st.selectbox(
        "Choose data layer to explore:",
        options=["raw", "processed", "gold"],
        format_func=lambda k: f"{LAYER_HELP[k]['icon']}  {LAYER_HELP[k]['title']}",
        index=1,
    )

    info = LAYER_HELP[selected_layer]
    layer_paths = {"raw": RAW_DIR, "processed": PROCESSED_DIR, "gold": GOLD_DIR}

    # Themed info card
    st.info(f"**{info['what']}**")
    with st.expander("What does this layer contain?", expanded=False):
        for item in info["contains"]:
            st.markdown(f"- {item}")
        st.caption(f"💡 {info['note']}")

    layer_path = layer_paths[selected_layer]
    parquet_files = sorted(layer_path.glob("**/*.parquet"))
    if not parquet_files:
        st.warning(
            f"No files found in the **{selected_layer}** layer yet. Run the pipeline first."
        )
        return

    file_labels = {str(p.relative_to(ROOT)): p for p in parquet_files}
    selected_label = st.selectbox(
        "Choose file to preview:", options=list(file_labels.keys())
    )
    selected_file = file_labels[selected_label]

    with st.spinner("Loading data..."):
        prog = st.progress(0, text="Reading file...")
        prog.progress(30, text="Parsing parquet...")
        df = pd.read_parquet(selected_file)
        prog.progress(80, text="Rendering table...")
        prog.progress(100, text="Done!")
        prog.empty()

    st.metric("Data Shape", f"{len(df)} rows × {len(df.columns)} columns")
    st.markdown("**Preview (first 200 rows):**")
    st.dataframe(df.head(200), width="stretch")


def show_analytics_tab() -> None:
    st.subheader("Analytics & Analysis Results")

    with st.expander("Help: How to interpret analysis output", expanded=False):
        if _main_mod is not None:
            _render_logger_message(
                "Analysis summary", getattr(_main_mod, "MAIN_OUTPUT_EXPLANATION", "")
            )

    corr_path = OUTPUT_DIR / "correlation_matrix.csv"
    summary_path = OUTPUT_DIR / "analysis_results.json"

    # --- Correlation heatmap ---
    corr_help = ANALYSIS_HELP.get("correlation_matrix", {})
    st.markdown("### Correlation Matrix")
    if corr_help:
        with st.expander("What is the Correlation Matrix?", expanded=False):
            st.markdown(f"**What it does:** {corr_help['what']}")
            st.markdown(f"**How to read it:** {corr_help['read']}")
            st.markdown(f"**How to use it:** {corr_help['use']}")

    if corr_path.exists():
        with st.spinner("Building heatmap..."):
            prog = st.progress(0, text="Reading correlation data...")
            corr_df = pd.read_csv(corr_path, index_col=0)
            prog.progress(50, text="Generating chart...")
            fig = px.imshow(
                corr_df,
                text_auto=".2f",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Correlation Matrix — Factor Relationships",
            )
            fig.update_layout(height=640)
            prog.progress(100, text="Done!")
            prog.empty()
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Correlation matrix not found. Run the pipeline first.")

    # --- Per-analysis artifact list ---
    summary = read_json(summary_path)
    artifacts = summary.get("artifacts", {}) if isinstance(summary, dict) else {}

    if artifacts:
        st.markdown("---")
        st.markdown("### Analysis Result Files")
        for analysis_name, file_path in artifacts.items():
            full_path = (
                OUTPUT_DIR / file_path
                if not Path(file_path).is_absolute()
                else Path(file_path)
            )
            help_entry = ANALYSIS_HELP.get(analysis_name, {})
            title = help_entry.get("title", analysis_name)

            with st.expander(f"📊 {title}", expanded=False):
                if help_entry:
                    col_info, col_act = st.columns([3, 1])
                    with col_info:
                        st.markdown(f"**What it does:** {help_entry['what']}")
                        st.markdown(f"**How to read it:** {help_entry['read']}")
                        st.markdown(f"**How to use it:** {help_entry['use']}")
                else:
                    col_info, col_act = st.columns([3, 1])

                if full_path.exists():
                    file_size = full_path.stat().st_size / 1024
                    st.caption(f"📄 {file_path}  •  {file_size:.1f} KB")
                    with col_act:
                        st.markdown("&nbsp;")
                        if st.button("Preview", key=f"prev_{analysis_name}"):
                            st.session_state["selected_preview_file"] = str(full_path)
                            st.session_state["selected_preview_name"] = analysis_name
                        with open(
                            full_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            st.download_button(
                                "⬇️ Download",
                                f.read(),
                                file_name=full_path.name,
                                key=f"dl_{analysis_name}",
                            )
                    if (
                        st.session_state.get("selected_preview_file")
                        == str(full_path)
                    ):
                        st.markdown("#### Preview")
                        st.json(read_json(full_path))
                else:
                    st.warning("File not found — run pipeline first.")


def show_governance_tab() -> None:
    gov_help = ANALYSIS_HELP.get("governance_report", {})
    st.subheader("Data Governance & Approvals")

    with st.expander("Help: Governance message", expanded=False):
        if _medallion_mod is not None:
            _render_logger_message(
                "Gold layer guidance", getattr(_medallion_mod, "GOLD_SUCCESS", "")
            )

    # Explanation from logger messages
    st.info(
        "**What is Governance?** "
        + (
            gov_help.get("what", "")
            or "An automated audit trail that tracks approvals and risk scores "
            "for every dataset entering the analytical layer."
        )
    )
    with st.expander("How to read a governance decision", expanded=False):
        st.markdown(
            f"**How to read it:** {gov_help.get('read', '')}\n\n"
            f"**How to use it:** {gov_help.get('use', '')}\n\n"
            "Each JSON file contains:\n"
            "- `status` — *approved* or *flagged*\n"
            "- `risk_score` — 0 (safe) to 100 (high risk)\n"
            "- `timestamp` — when the decision was recorded\n"
            "- `notes` — reason and metadata"
        )

    gov_dir = GOLD_DIR / "governance"
    files = sorted(gov_dir.glob("governance_decision_*.json"))
    if not files:
        st.warning("No governance decisions found. Run pipeline first.")
        return

    st.metric("Total Decisions", len(files))
    selected = st.selectbox(
        "View governance decision:",
        options=list(reversed(files)),
        format_func=lambda p: (
            f"{p.name}  —  "
            f"{pd.Timestamp(p.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}"
        ),
    )
    st.caption(f"File: {selected.name}")
    st.json(read_json(selected))


def show_logs_tab() -> None:
    st.subheader("Execution Logs & Pipeline Summary")

    with st.expander("Help: Runtime messages", expanded=False):
        if _main_mod is not None:
            _render_logger_message(
                "Main guide", getattr(_main_mod, "MAIN_USER_GUIDE", "")
            )
        if _fetchers_mod is not None:
            _render_logger_message(
                "Fetcher output",
                getattr(_fetchers_mod, "FETCHER_OUTPUT_EXPLANATION", ""),
            )
        if _medallion_mod is not None:
            _render_logger_message(
                "Medallion output",
                getattr(_medallion_mod, "MEDALLION_OUTPUT_EXPLANATION", ""),
            )

    st.info(
        "Each pipeline run saves a session summary here. "
        "Use it to verify the run completed correctly and to audit "
        "execution times, data counts, and any warnings."
    )

    # Pipeline stage reference card
    with st.expander(
        "Pipeline Stage Reference — what happens at each step", expanded=False
    ):
        for stage, desc in PIPELINE_STAGES:
            st.markdown(f"**{stage}** — {desc}")

    summaries = sorted(
        LOGS_DIR.glob("session_summary_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not summaries:
        st.warning("No session logs found. Run the pipeline first.")
        return

    st.metric("Total Sessions Recorded", len(summaries))

    latest = summaries[-1]
    latest_time = pd.Timestamp(latest.stat().st_mtime, unit="s").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    st.success(f"Latest run: **{latest.name}** — {latest_time}")

    with st.expander("What each field means", expanded=False):
        st.markdown(
            "- **execution_time** — total seconds the pipeline ran\n"
            "- **total_records** — rows processed across all data sources\n"
            "- **files_created** — number of output files generated\n"
            "- **stage_results** — Bronze / Silver / Gold pass/fail status\n"
            "- **errors** — any exceptions caught during the run"
        )

    st.markdown("**Session JSON:**")
    st.json(read_json(latest))


def show_output_tab() -> None:
    st.subheader("Output Results Folder")

    st.markdown(
        """
        ## Your Analysis Results
        
        All files generated by the pipeline are saved here.
        You can preview and download each file.
        """
    )

    if not OUTPUT_DIR.exists():
        st.warning("Output directory not found. Run pipeline first.")
        return

    # Get all files in output directory
    with st.spinner("Scanning output folder..."):
        prog = st.progress(0, text="Listing files...")
        output_files = sorted([f for f in OUTPUT_DIR.glob("*") if f.is_file()])
        prog.progress(100, text="Done!")
        prog.empty()

    if not output_files:
        st.warning("No output files found. Run pipeline first.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Files", len(output_files))
    col2.metric("Folder", "output/")
    col3.metric(
        "Total Size", f"{sum(f.stat().st_size for f in output_files) / 1024:.1f} KB"
    )

    st.markdown("---")
    st.markdown("**File Directory:**")

    for file_path in output_files:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.markdown(f"**{file_path.name}**")
            st.caption(
                f"Size: {file_path.stat().st_size / 1024:.1f} KB | Modified: {pd.Timestamp(file_path.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}"
            )
        with col2:
            # File type badge
            if file_path.suffix == ".json":
                st.badge("JSON", color="blue")
            elif file_path.suffix == ".csv":
                st.badge("CSV", color="green")
            else:
                st.badge(file_path.suffix.upper(), color="orange")
        with col3:
            # Preview button
            if file_path.suffix in [".json", ".csv"]:
                if st.button("👁️", key=f"view_{file_path.name}", help="Preview file"):
                    st.markdown(f"**Preview: {file_path.name}**")
                    if file_path.suffix == ".json":
                        try:
                            st.json(read_json(file_path))
                        except:
                            st.info("Could not parse JSON")
                    else:
                        try:
                            df = pd.read_csv(file_path)
                            st.dataframe(df.head(100), width="stretch")
                        except:
                            st.info("Could not parse CSV")
        with col4:
            # Download button
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    st.download_button(
                        "⬇️",
                        f.read(),
                        file_name=file_path.name,
                        key=f"download_{file_path.name}",
                        help="Download file",
                    )
            except:
                pass


def show_health_alerts_tab() -> None:
    st.subheader("Data Health & Smart Alerts")
    health = _compute_data_health()

    score_col, null_col, out_col, sch_col, fresh_col = st.columns(5)
    score_col.metric("Health Score", f"{health['score']}/100")
    null_col.metric("Null Penalty", health["null_penalty"])
    out_col.metric("Outlier Penalty", health["outlier_penalty"])
    sch_col.metric("Schema Penalty", health["schema_penalty"])
    fresh_col.metric("Freshness Penalty", health["freshness_penalty"])

    if health["status"] == "critical":
        st.error(f"Critical data quality risk. {health['details']}")
    elif health["status"] == "warning":
        st.warning(f"Data quality warning. {health['details']}")
    else:
        st.success(f"Data quality is healthy. {health['details']}")

    st.markdown("### Smart Alerts")
    alerts = _build_smart_alerts()
    if not alerts:
        st.success("No active alerts.")
        return
    for alert in alerts:
        sev = alert.get("severity", "warning")
        msg = f"**{alert['title']}** — {alert['message']}"
        if sev == "critical":
            st.error(msg)
        else:
            st.warning(msg)


def show_run_comparison_tab() -> None:
    st.subheader("Run Comparison (Latest vs Previous)")
    diff = _build_run_comparison()
    if diff.get("status") != "ok":
        st.info(diff.get("message", "Not enough runs yet."))
        return

    c1, c2 = st.columns(2)
    c1.metric(
        "Pipeline Duration",
        f"{diff['duration_curr']:.2f}s",
        f"{diff['duration_delta']:+.2f}s",
    )
    c2.metric("Operations", diff["ops_curr"], f"{diff['ops_delta']:+d}")

    st.markdown("### Output File Diff")
    files = diff["output_file_diff"]
    c3, c4, c5 = st.columns(3)
    c3.metric("Added", len(files["added"]))
    c4.metric("Removed", len(files["removed"]))
    c5.metric("Changed", len(files["changed"]))

    with st.expander("Details: Output file changes", expanded=False):
        st.markdown(f"- Added: {files['added']}")
        st.markdown(f"- Removed: {files['removed']}")
        st.markdown(f"- Changed: {files['changed']}")

    st.markdown("### Governance Decision Diff")
    gov_prev = diff.get("governance_prev", {})
    gov_curr = diff.get("governance_curr", {})
    st.markdown("**Previous governance record:**")
    st.json(gov_prev)
    st.markdown("**Current governance record:**")
    st.json(gov_curr)


def show_scenario_builder_tab() -> None:
    st.subheader("Scenario Builder (Custom Shocks)")
    st.caption("Set custom macro shocks and see immediate impact estimates.")

    c1, c2, c3 = st.columns(3)
    inflation_shock = c1.slider("Inflation shock (%)", -10.0, 10.0, 1.0, 0.5)
    energy_shock = c2.slider("Energy shock (%)", -20.0, 20.0, 2.0, 0.5)
    rate_shock = c3.slider("Rates shock (%)", -5.0, 5.0, 0.5, 0.25)

    summary = read_json(OUTPUT_DIR / "analysis_results.json")
    results = summary.get("results", {}) if isinstance(summary, dict) else {}
    elasticity_val = results.get("elasticity")

    baseline_move = 0.0
    if isinstance(elasticity_val, (int, float)):
        baseline_move = float(elasticity_val) * (inflation_shock / 100.0)

    total_shock = (
        inflation_shock * 0.5 + energy_shock * 0.3 + rate_shock * 0.2
    ) / 100.0
    estimated_return_shift = baseline_move + total_shock

    st.metric("Estimated return shift", f"{estimated_return_shift * 100:.2f}%")
    st.info(
        "This is a fast directional estimate from available elasticity and weighted shocks. "
        "Use Stress Test and Forecast outputs for detailed decisions."
    )


def show_explainability_tab() -> None:
    st.subheader("Explainability Panel")
    st.caption("What changed and why — plain language summaries.")
    lines = _build_explainability_lines()
    for line in lines:
        st.markdown(f"- {line}")


# ---------------------------------------------------------------------------
# Auditor helpers
# ---------------------------------------------------------------------------

def _load_audit_class() -> Any:
    """Dynamically load ScenarioAuditor from Auditor.py at project root."""
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("Auditor", ROOT / "Auditor.py")
        if spec is None or spec.loader is None:
            return None
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
        return getattr(_mod, "ScenarioAuditor", None)
    except Exception:
        return None


def _run_and_cache_audit() -> dict[str, Any]:
    """Run ScenarioAuditor, cache in session_state, persist to disk."""
    AuditorClass = _load_audit_class()
    if AuditorClass is None:
        result: dict[str, Any] = {"status": "ERROR", "error": "Auditor module could not be loaded."}
    else:
        try:
            auditor = AuditorClass(user_id=_UI_USER_ID)
            result = auditor.run_audit()
        except Exception as exc:
            result = {"status": "ERROR", "error": str(exc)}
    st.session_state["audit_report"] = result
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        AUDIT_REPORT_PATH.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
    except OSError:
        pass
    return result


def _get_audit_report() -> dict[str, Any]:
    """Return cached audit report: session_state -> disk -> empty dict."""
    if "audit_report" in st.session_state:
        return st.session_state["audit_report"]
    try:
        report = read_json(AUDIT_REPORT_PATH)
        if report:
            st.session_state["audit_report"] = report
            return report
    except (JSONParseError, DataFileReadError):
        pass
    return {}


def show_auditor_tab() -> None:
    st.subheader("System Auditor")
    st.caption(
        "Independent audit judge: checks source coverage, data density, statistics, "
        "temporal continuity, output quality, threshold design and governance validity. "
        "Runs automatically after every pipeline — or press Re-run Audit at any time."
    )

    col_btn_area, col_status_area = st.columns([1, 4])
    with col_btn_area:
        if st.button("Re-run Audit", key="manual_audit_run"):
            with st.spinner("Running audit..."):
                _run_and_cache_audit()
            st.rerun()

    report = _get_audit_report()

    if not report:
        with col_status_area:
            st.info(
                "No audit report found yet. Run the pipeline first — the audit runs "
                "automatically after each run, or press **Re-run Audit** above."
            )
        return

    status = report.get("status", "UNKNOWN")
    with col_status_area:
        if status == "PASS":
            st.success(f"**Audit Status: {status}** — All checks passed.")
        elif status == "WARN":
            st.warning(f"**Audit Status: {status}** — Some items need attention.")
        elif status == "CRITICAL":
            st.error(f"**Audit Status: {status}** — Critical issues found.")
        elif status == "ERROR":
            st.error(f"**Audit Error:** {report.get('error', 'unknown')}")
        else:
            st.info(f"**Audit Status: {status}**")

    row_count = report.get("row_count", 0)
    col_count = report.get("column_count", 0)
    decision_ready = report.get("decision_ready", False)
    num_failed = len(report.get("failed_checks", []))
    num_warn = len(report.get("warning_checks", []))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Gold Rows", row_count)
    k2.metric("Gold Columns", col_count)
    k3.metric("Decision Ready", "Yes" if decision_ready else "No")
    k4.metric("Failed Checks", num_failed)
    k5.metric("Warning Checks", num_warn)

    judgement = report.get("auditor_judgement", {})
    if judgement:
        st.markdown("---")
        with st.expander("Auditor Judgement", expanded=True):
            jc1, jc2 = st.columns(2)
            jc1.metric(
                "Information Reasonable",
                "Yes" if judgement.get("is_information_reasonable") else "No",
            )
            jc2.metric(
                "Can Support Decisions",
                "Yes" if judgement.get("can_support_decisions") else "No",
            )
            for line in judgement.get("summary", []):
                st.markdown(f"- {line}")

    st.markdown("---")
    st.markdown("### Check Results")

    _CHECK_ICONS: dict[str, str] = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
    _CHECK_LABELS: dict[str, str] = {
        "integration": "Source Integration",
        "density": "Data Density",
        "statistics": "Statistical Plausibility",
        "continuity": "Temporal Continuity",
        "outputs": "Output Quality",
        "thresholds": "Threshold Design",
        "governance": "Governance Validity",
    }

    for check_name, check_result in report.get("checks", {}).items():
        label = _CHECK_LABELS.get(check_name, check_name.capitalize())
        status_code = check_result.get("status", "fail")
        icon = _CHECK_ICONS.get(status_code, "ℹ️")
        passed = check_result.get("passed", False)

        with st.expander(f"{icon} {label}", expanded=not passed):
            interp = check_result.get("interpretation")
            if isinstance(interp, str) and interp:
                st.caption(interp)

            issues = check_result.get("issues", [])
            if issues:
                st.markdown("**Issues:**")
                for issue in issues:
                    st.markdown(f"- `{issue}`")

            if check_name == "integration":
                source_metrics = check_result.get("metrics", {})
                breadth = check_result.get("breadth", {})
                if breadth:
                    _b_rows = [
                        {
                            "Source": src,
                            "Expected": v.get("expected", 0),
                            "Observed": v.get("observed", 0),
                            "Ratio": f"{v.get('ratio', 0.0):.0%}",
                            "Row Coverage": f"{source_metrics.get(src, 0.0):.1f}%",
                        }
                        for src, v in breadth.items()
                    ]
                    st.dataframe(pd.DataFrame(_b_rows), width="stretch")

            elif check_name == "density":
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("Null %", f"{check_result.get('null_pct', 0.0):.1f}%")
                dc2.metric("Row non-null %", f"{check_result.get('row_non_null_pct', 0.0):.1f}%")
                dc3.metric("Zero-variance cols", check_result.get("zero_var_count", 0))
                _zv = check_result.get("zero_var_columns", [])
                if _zv:
                    st.caption(f"Zero-variance: {', '.join(_zv)}")

            elif check_name == "statistics":
                _sm = check_result.get("metrics", {})
                if _sm:
                    _sm_cols = st.columns(min(len(_sm), 3))
                    for _si, (_sk, _sv) in enumerate(list(_sm.items())[:6]):
                        _sm_cols[_si % len(_sm_cols)].metric(
                            _sk, f"{_sv:.4f}" if isinstance(_sv, float) else str(_sv)
                        )

            elif check_name == "continuity":
                _failed_groups = check_result.get("failed_groups", [])
                if _failed_groups:
                    st.warning(f"Groups with large temporal gaps: {', '.join(_failed_groups)}")
                _group_m = check_result.get("metrics", {})
                if _group_m:
                    _cont_rows = [
                        {
                            "Ticker/Group": g,
                            "Median gap (d)": v.get("median_gap_days", 0),
                            "Max gap (d)": v.get("max_gap_days", 0),
                            "Allowed gap (d)": v.get("allowed_gap_days", 0),
                        }
                        for g, v in list(_group_m.items())[:20]
                    ]
                    st.dataframe(pd.DataFrame(_cont_rows), width="stretch")

            elif check_name == "outputs":
                oc1, oc2, oc3 = st.columns(3)
                oc1.metric("Usable", len(check_result.get("usable_outputs", [])))
                oc2.metric("Blocked", len(check_result.get("blocked_outputs", [])))
                oc3.metric("Empty", len(check_result.get("nullish_outputs", [])))
                _usable = check_result.get("usable_outputs", [])
                _blocked = check_result.get("blocked_outputs", [])
                if _usable:
                    st.markdown(f"**Usable:** {', '.join(_usable)}")
                if _blocked:
                    st.markdown(f"**Blocked:** {', '.join(_blocked)}")

            elif check_name == "thresholds":
                _dyn = check_result.get("dynamic_thresholds", {})
                if _dyn:
                    for _dk, _dv in _dyn.items():
                        _t_icon = "✅" if _dv else "⬜"
                        st.markdown(f"- {_t_icon} `{_dk}`")
                _strictness = check_result.get("strictness_findings", [])
                if _strictness:
                    st.markdown("**Strictness findings:**")
                    for _sf in _strictness:
                        st.markdown(f"- `{_sf}`")

            elif check_name == "governance":
                _gm = check_result.get("metrics", {})
                if _gm:
                    _gm_cols = st.columns(min(len(_gm), 3))
                    for _gi, (_gk, _gv) in enumerate(list(_gm.items())[:6]):
                        _gdisp = f"{_gv:.4f}" if isinstance(_gv, float) else str(_gv)
                        _gm_cols[_gi % len(_gm_cols)].metric(_gk, _gdisp)
                if check_result.get("likely_over_strict"):
                    st.warning("Governance may be over-strict for the current data regime.")
                _reasons = check_result.get("reasons", [])
                if _reasons:
                    st.markdown(f"**Block reasons:** {', '.join(_reasons)}")
                _interp_notes = check_result.get("interpretation", [])
                if isinstance(_interp_notes, list) and _interp_notes:
                    for _note in _interp_notes:
                        st.markdown(f"- {_note}")

            with st.expander("Raw check data", expanded=False):
                st.json(check_result)


def show_reports_tab(role: str) -> None:
    st.subheader("Executive Reports (One-click Export)")
    html_report = _build_executive_report_html()

    st.download_button(
        "Download HTML Report",
        html_report,
        file_name=f"executive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
    )

    st.info(
        "PDF export: use browser Print -> Save as PDF on the downloaded HTML report "
        "for consistent layout without extra dependencies."
    )

    if ROLE_PERMISSIONS[role]["can_download"]:
        st.markdown("### Report Preview")
        st.components.v1.html(html_report, height=480, scrolling=True)


def show_ops_tab(role: str) -> None:
    st.subheader("Job History & Scheduling")

    history = _load_session_history(limit=50)
    if history:
        rows = []
        for h in history[-20:][::-1]:
            info = h.get("session_info", {}) or {}
            rows.append(
                {
                    "session": h.get("__file", ""),
                    "duration_s": round(
                        float(info.get("total_duration_seconds", 0.0) or 0.0), 2
                    ),
                    "operations": int(info.get("total_operations", 0) or 0),
                    "status": "ok" if not h.get("error_metrics") else "has_errors",
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("No run history yet.")

    st.markdown("### Nightly Scheduling")
    schedule = read_json(UI_SCHEDULE_PATH) if UI_SCHEDULE_PATH.exists() else {}
    enabled_default = bool(schedule.get("enabled", False))
    hour_default = (
        int(schedule.get("hour", 2)) if isinstance(schedule.get("hour"), int) else 2
    )
    minute_default = (
        int(schedule.get("minute", 0)) if isinstance(schedule.get("minute"), int) else 0
    )

    enabled = st.toggle(
        "Enable nightly schedule",
        value=enabled_default,
        disabled=not ROLE_PERMISSIONS[role]["can_schedule"],
    )
    c1, c2 = st.columns(2)
    hour = c1.number_input(
        "Hour (24h)",
        min_value=0,
        max_value=23,
        value=hour_default,
        step=1,
        disabled=not ROLE_PERMISSIONS[role]["can_schedule"],
    )
    minute = c2.number_input(
        "Minute",
        min_value=0,
        max_value=59,
        value=minute_default,
        step=1,
        disabled=not ROLE_PERMISSIONS[role]["can_schedule"],
    )

    if st.button("Save schedule", disabled=not ROLE_PERMISSIONS[role]["can_schedule"]):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "enabled": enabled,
            "hour": int(hour),
            "minute": int(minute),
            "saved_at": datetime.now().isoformat(),
        }
        with UI_SCHEDULE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        st.success("Schedule saved.")

    st.caption(
        "Tip: wire this schedule to Task Scheduler/cron using `poetry run python src/main.py` at saved time."
    )


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Scenario Planner Command Center</h1>
            <p>
                Modern dashboard for pipeline control, artifact inspection,
                analytics and governance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        role = st.selectbox("Role", options=["Viewer", "Analyst", "Admin"], index=1)
        perms = ROLE_PERMISSIONS[role]
        st.caption(
            f"Permissions: run={perms['can_run']}, download={perms['can_download']}, schedule={perms['can_schedule']}"
        )

        st.header("Pipeline Execution")

        with st.expander("Live guidance (from logger messages)", expanded=False):
            if _directions_mod is not None:
                _render_logger_message(
                    "Welcome",
                    getattr(_directions_mod, "LIVE_STEP_0_WELCOME", ""),
                )
                _render_logger_message(
                    "Quick start",
                    getattr(_main_mod, "QUICK_START", "") if _main_mod else "",
                )

        with st.expander("What does the pipeline do?", expanded=False):
            for stage, desc in PIPELINE_STAGES:
                st.markdown(f"**{stage}** — {desc}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Quick Demo\n(2-3 min)",
                width="stretch",
                type="secondary",
                disabled=not perms["can_run"],
            ):
                st.caption("Running pipeline in SAMPLE mode...")
                prog = st.progress(0, text="Starting...")
                ok, output = run_pipeline(mode="sample", progress_bar=prog)
                prog.empty()
                if ok:
                    _record_ui_snapshot()
                    st.success("✓ Demo completed successfully!")
                    with st.spinner("Running system audit..."):
                        _run_and_cache_audit()
                    st.info("📁 Check **Output** and **Auditor** tabs for results.")
                else:
                    _show_pipeline_failure(output)

        with col2:
            if st.button(
                "Full Analysis\n(5-10 min)",
                width="stretch",
                type="primary",
                disabled=not perms["can_run"],
            ):
                st.caption("Running pipeline in FULL mode...")
                prog = st.progress(0, text="Starting...")
                ok, output = run_pipeline(mode="actual", progress_bar=prog)
                prog.empty()
                if ok:
                    _record_ui_snapshot()
                    st.success("✓ Analysis completed successfully!")
                    with st.spinner("Running system audit..."):
                        _run_and_cache_audit()
                    st.info("📁 Check **Output** and **Auditor** tabs for results.")
                else:
                    _show_pipeline_failure(output)

        st.markdown("---")
        st.info(
            "**Quick Demo**: Small dataset, fast results (recommended for testing)\n\n"
            "**Full Analysis**: Complete dataset, full analyses"
        )
        st.caption(
            "Tip: Results appear in the **Output** tab after successful completion."
        )

    show_kpis()

    pages = [
        "Health & Alerts",
        "Auditor",
        "Run Comparison",
        "Scenario Builder",
        "Explainability",
        "Reports",
        "Ops",
        "Data",
        "Analytics",
        "Output",
        "Governance",
        "Logs",
    ]
    selected_page = st.segmented_control(
        "View",
        options=pages,
        default="Health & Alerts",
    )

    if selected_page == "Health & Alerts":
        show_health_alerts_tab()
    elif selected_page == "Auditor":
        show_auditor_tab()
    elif selected_page == "Run Comparison":
        show_run_comparison_tab()
    elif selected_page == "Scenario Builder":
        show_scenario_builder_tab()
    elif selected_page == "Explainability":
        show_explainability_tab()
    elif selected_page == "Reports":
        show_reports_tab(role)
    elif selected_page == "Ops":
        show_ops_tab(role)
    elif selected_page == "Data":
        show_data_tab()
    elif selected_page == "Analytics":
        show_analytics_tab()
    elif selected_page == "Output":
        show_output_tab()
    elif selected_page == "Governance":
        show_governance_tab()
    elif selected_page == "Logs":
        show_logs_tab()


if __name__ == "__main__":
    main()
