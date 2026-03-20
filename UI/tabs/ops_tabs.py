from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from UI.constants import ROLE_PERMISSIONS, UI_SCHEDULE_PATH
from UI.helpers import load_session_history, read_json
from UI.runtime import get_audit_report, run_and_cache_audit


def _fmt_number(value: object, decimals: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}" if isinstance(value, float) else str(value)
    return "N/A"


def _fmt_pct(value: object, decimals: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{decimals}f}%"
    return "N/A"


def show_auditor_tab() -> None:
    st.subheader("🧪 System Auditor")
    st.caption(
        "Independent audit judge: checks source coverage, data density, statistics, temporal continuity, output quality, threshold design and governance validity."
    )
    if st.button("🔁 Re-run Audit", key="manual_audit_run"):
        with st.spinner("Running audit..."):
            run_and_cache_audit()
        st.rerun()

    report = get_audit_report()
    if not report:
        st.info("No audit report found yet. Run the pipeline first.")
        return

    status = report.get("status", "UNKNOWN")
    if status == "PASS":
        st.success(f"✅ Audit Status: {status}")
    elif status == "WARN":
        st.warning(f"⚠️ Audit Status: {status}")
    elif status == "CRITICAL":
        st.error(f"❌ Audit Status: {status}")
    else:
        st.info(f"ℹ️ Audit Status: {status}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", report.get("row_count", 0))
    c2.metric("Columns", report.get("column_count", 0))
    c3.metric("Decision Ready", "Yes" if report.get("decision_ready") else "No")
    c4.metric("Failed Checks", len(report.get("failed_checks", [])))
    c5.metric("Warning Checks", len(report.get("warning_checks", [])))

    label_map = {
        "integration": "Integration",
        "density": "Density",
        "statistics": "Statistics",
        "continuity": "Temporal Continuity",
        "outputs": "Output Quality",
        "thresholds": "Threshold Design",
        "governance": "Governance Validity",
    }
    icon_map = {"pass": "✅", "warn": "⚠️", "fail": "❌"}

    failed_checks = report.get("failed_checks", [])
    warning_checks = report.get("warning_checks", [])
    if failed_checks:
        st.error(f"Failed checks: {', '.join(failed_checks)}")
    if warning_checks:
        st.warning(f"Warning checks: {', '.join(warning_checks)}")

    if failed_checks or warning_checks:
        with st.expander("🧭 Check Summary", expanded=True):
            if failed_checks:
                for check_name in failed_checks:
                    st.markdown(f"- ❌ {label_map.get(check_name, check_name)}")
            if warning_checks:
                for check_name in warning_checks:
                    st.markdown(f"- ⚠️ {label_map.get(check_name, check_name)}")

    judgement = report.get("auditor_judgement", {})
    if isinstance(judgement, dict) and judgement:
        with st.expander("🧠 Auditor Judgement", expanded=True):
            j1, j2 = st.columns(2)
            j1.metric(
                "Information Reasonable",
                "Yes" if judgement.get("is_information_reasonable") else "No",
            )
            j2.metric(
                "Can Support Decisions",
                "Yes" if judgement.get("can_support_decisions") else "No",
            )
            for line in judgement.get("summary", []):
                st.markdown(f"- {line}")

    for name, result in report.get("checks", {}).items():
        status_code = str(result.get("status", "warn"))
        status_icon = icon_map.get(status_code, "ℹ️")
        title = label_map.get(name, name.capitalize())
        with st.expander(f"{status_icon} {title}", expanded=not result.get("passed", False)):
            interpretation = result.get("interpretation")
            if isinstance(interpretation, str) and interpretation:
                st.caption(interpretation)
            elif isinstance(interpretation, list) and interpretation:
                for note in interpretation:
                    st.markdown(f"- {note}")

            issues = result.get("issues", [])
            reasons = result.get("reasons", [])
            if issues:
                st.markdown("**Issues detected:**")
                for issue in issues:
                    st.markdown(f"- {issue}")
            if reasons:
                st.markdown("**Governance reasons:**")
                for reason in reasons:
                    st.markdown(f"- {reason}")

            zero_var_columns = result.get("zero_var_columns", [])
            all_zero_columns = result.get("all_zero_columns", [])
            blocked_outputs = result.get("blocked_outputs", [])
            nullish_outputs = result.get("nullish_outputs", [])
            strictness_findings = result.get("strictness_findings", [])
            warnings = result.get("warnings", [])
            reasoning = result.get("reasoning", [])

            if zero_var_columns:
                st.markdown("**Zero-variance indicators:**")
                for col in zero_var_columns:
                    st.markdown(f"- {col}")

            if all_zero_columns:
                st.markdown("**Always-zero indicators:**")
                for col in all_zero_columns:
                    st.markdown(f"- {col}")

            if blocked_outputs:
                st.markdown("**Blocked outputs:**")
                for out_name in blocked_outputs:
                    st.markdown(f"- {out_name}")

            if nullish_outputs:
                st.markdown("**Empty/Nullish outputs:**")
                for out_name in nullish_outputs:
                    st.markdown(f"- {out_name}")

            if strictness_findings:
                st.markdown("**Threshold strictness findings:**")
                for finding in strictness_findings:
                    st.markdown(f"- {finding}")

            if warnings:
                st.markdown("**Continuity warnings:**")
                for warn in warnings:
                    st.markdown(f"- {warn}")

            if reasoning:
                st.markdown("**Auditor reasoning:**")
                for reason in reasoning:
                    st.markdown(f"- {reason}")

            if name == "density":
                d1, d2, d3 = st.columns(3)
                d1.metric("Null %", _fmt_pct(result.get("null_pct")))
                d2.metric("Row Coverage %", _fmt_pct(result.get("row_non_null_pct")))
                d3.metric("Zero-variance cols", _fmt_number(result.get("zero_var_count"), decimals=0))

            if name == "continuity":
                cmax, cmed, callowed, cbase = st.columns(4)
                cmax.metric("Max Gap (bdays)", _fmt_number(result.get("max_gap"), decimals=0))
                cmed.metric("Median Gap (bdays)", _fmt_number(result.get("median_gap")))
                callowed.metric("Auto Allowed Gap", _fmt_number(result.get("allowed_gap"), decimals=0))
                cbase.metric("Detected Cadence", _fmt_number(result.get("detected_baseline_gap"), decimals=0))
                cfg_gap = result.get("configured_allowed_gap")
                dyn_gap = result.get("allowed_gap")
                if cfg_gap is not None and dyn_gap is not None and dyn_gap != cfg_gap:
                    cadence_name = "monthly" if (result.get("detected_baseline_gap") or 0) >= 15 else "weekly" if (result.get("detected_baseline_gap") or 0) >= 4 else "daily"
                    st.info(
                        f"Auto-detected **{cadence_name}** data cadence — threshold scaled from "
                        f"{cfg_gap} → **{dyn_gap}** business days."
                    )
                if result.get("duplicate_rows_removed", 0):
                    st.warning(
                        f"Duplicate rows removed: {result.get('duplicate_rows_removed')}"
                    )
                fg = result.get("failed_groups", [])
                if fg:
                    st.error(f"Groups failing gap check: {', '.join(fg)}")
                if result.get("error"):
                    st.error(f"Continuity error: {result.get('error')}")

            if name == "governance":
                metrics = result.get("metrics", {})
                if isinstance(metrics, dict) and metrics:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("OOS R2", _fmt_number(metrics.get("out_of_sample_r2"), decimals=4))
                    m2.metric("Walk-forward avg R2", _fmt_number(metrics.get("walk_forward_avg_r2"), decimals=4))
                    m3.metric("Model risk", _fmt_number(metrics.get("model_risk_score"), decimals=4))
                if result.get("likely_over_strict"):
                    st.warning("Governance appears methodologically valid but potentially over-strict.")

            if name == "thresholds":
                dynamic = result.get("dynamic_thresholds", {})
                if isinstance(dynamic, dict) and dynamic:
                    st.markdown("**🧩 Dynamic threshold flags:**")
                    st.json(dynamic)

            if name == "integration":
                metrics = result.get("metrics", {})
                cell_fill = result.get("cell_fill_pct", {})
                overall = result.get("overall", {})
                if isinstance(overall, dict) and overall:
                    o1, o2 = st.columns(2)
                    o1.metric(
                        "Rows with all 3 sources",
                        _fmt_pct(overall.get("rows_with_all_sources_pct")),
                    )
                    max_groups = overall.get("max_source_groups_per_row", 3)
                    avg_groups = overall.get("avg_active_source_groups_per_row")
                    if isinstance(avg_groups, (int, float)):
                        o2.metric(
                            "Avg active source groups / row",
                            f"{float(avg_groups):.2f} / {max_groups}",
                        )
                    else:
                        o2.metric("Avg active source groups / row", "N/A")
                if isinstance(metrics, dict) and metrics:
                    st.markdown("**Row coverage by source:**")
                    coverage_rows = [
                        {
                            "Source": source,
                            "Row coverage %": _fmt_pct(value),
                            "Cell fill %": _fmt_pct(cell_fill.get(source) if isinstance(cell_fill, dict) else None),
                        }
                        for source, value in metrics.items()
                    ]
                    st.dataframe(pd.DataFrame(coverage_rows), width="stretch")

                breadth = result.get("breadth", {})
                if isinstance(breadth, dict) and breadth:
                    st.markdown("**Configured entity breadth:**")
                    rows = []
                    for source, payload in breadth.items():
                        if isinstance(payload, dict):
                            rows.append(
                                {
                                    "Source": source,
                                    "Expected": payload.get("expected", 0),
                                    "Observed": payload.get("observed", 0),
                                    "Entity coverage ratio": payload.get("ratio", 0),
                                }
                            )
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width="stretch")

            metrics_payload = result.get("metrics", {})
            if isinstance(metrics_payload, dict) and metrics_payload and name not in {
                "governance",
                "integration",
                "continuity",
                "density",
            }:
                with st.expander("Metrics", expanded=False):
                    st.json(metrics_payload)

            with st.expander("🧾 Raw check payload", expanded=False):
                st.json(result)


def show_ops_tab(role: str) -> None:
    st.subheader("⚙️ Job History & Scheduling")

    history = load_session_history(limit=50)
    if history:
        rows = []
        for h in history[-20:][::-1]:
            info = h.get("session_info", {}) or {}
            rows.append(
                {
                    "session": h.get("__file", ""),
                    "duration_s": round(float(info.get("total_duration_seconds", 0.0) or 0.0), 2),
                    "operations": int(info.get("total_operations", 0) or 0),
                    "status": "ok" if not h.get("error_metrics") else "has_errors",
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("No run history yet.")

    st.markdown("### 🌙 Nightly Scheduling")
    schedule = read_json(UI_SCHEDULE_PATH) if UI_SCHEDULE_PATH.exists() else {}
    enabled_default = bool(schedule.get("enabled", False))
    hour_default = int(schedule.get("hour", 2)) if isinstance(schedule.get("hour"), int) else 2
    minute_default = int(schedule.get("minute", 0)) if isinstance(schedule.get("minute"), int) else 0

    enabled = st.toggle(
        "🌙 Enable nightly schedule",
        value=enabled_default,
        disabled=not ROLE_PERMISSIONS[role]["can_schedule"],
    )
    c1, c2 = st.columns(2)
    hour = c1.number_input(
        "Hour (24h)", min_value=0, max_value=23, value=hour_default, step=1, disabled=not ROLE_PERMISSIONS[role]["can_schedule"]
    )
    minute = c2.number_input(
        "Minute", min_value=0, max_value=59, value=minute_default, step=1, disabled=not ROLE_PERMISSIONS[role]["can_schedule"]
    )
    if st.button("💾 Save schedule", disabled=not ROLE_PERMISSIONS[role]["can_schedule"]):
        payload = {
            "enabled": enabled,
            "hour": int(hour),
            "minute": int(minute),
            "saved_at": datetime.now().isoformat(),
        }
        UI_SCHEDULE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        st.success("Schedule saved.")

    st.info("History deletion is available from the sidebar under History Control.")
