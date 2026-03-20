from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from UI.constants import ROLE_PERMISSIONS, UI_SCHEDULE_PATH
from UI.helpers import load_session_history, read_json
from UI.runtime import get_audit_report, run_and_cache_audit
from UI.traffic_light import (
    badge_html,
    score_audit_status,
    score_check_result,
    score_decision_ready,
    score_governance_gate,
    score_model_risk,
    score_null_pct,
    score_oos_r2,
    score_source_coverage,
)


def _fmt_number(value: object, decimals: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}" if isinstance(value, float) else str(value)
    return "N/A"


def _fmt_pct(value: object, decimals: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{decimals}f}%"
    return "N/A"


def _render_kv_table(payload: dict, title: str = "Details") -> None:
    if not isinstance(payload, dict) or not payload:
        return
    rows = []
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            rows.append({"Field": str(key), "Value": f"{type(value).__name__}[{len(value)}]"})
        else:
            rows.append({"Field": str(key), "Value": _fmt_number(value, 4) if isinstance(value, float) else str(value)})
    if rows:
        st.markdown(f"**{title}**")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def show_auditor_tab() -> None:
    st.subheader("System Auditor")
    st.caption(
        "Independent audit judge: checks source coverage, data density, statistics, temporal continuity, output quality, threshold design and governance validity."
    )
    if st.button("Re-run Audit", key="manual_audit_run"):
        with st.spinner("Running audit..."):
            run_and_cache_audit()
        st.rerun()

    report = get_audit_report()
    if not report:
        st.info("No audit report found yet. Run the pipeline first.")
        return

    status = report.get("status", "UNKNOWN")
    tl_color, tl_label, tl_desc = score_audit_status(status)
    st.markdown(
        badge_html(tl_label, tl_color, tl_desc) + f"&nbsp; <span style='color:#555'>{tl_desc}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    dr_color, dr_label, dr_desc = score_decision_ready(bool(report.get("decision_ready")))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", report.get("row_count", 0))
    c2.metric("Columns", report.get("column_count", 0))
    c3.markdown(badge_html(dr_label, dr_color, dr_desc), unsafe_allow_html=True)
    c4.metric("Failed Checks", len(report.get("failed_checks", [])))
    c5.metric("Warning Checks", len(report.get("warning_checks", [])))

    label_map = {
        "integration": "Source Integration",
        "density": "Data Density",
        "statistics": "Statistical Plausibility",
        "continuity": "Temporal Continuity",
        "survivorship": "Survivorship Bias",
        "outputs": "Output Quality",
        "thresholds": "Threshold Design",
        "governance": "Governance Validity",
    }
    icon_map = {"pass": "✅", "warn": "⚠️", "fail": "❌"}

    failed_checks = report.get("failed_checks", [])
    warning_checks = report.get("warning_checks", [])
    if failed_checks:
        st.error(f"Failed checks: {', '.join(label_map.get(c, c) for c in failed_checks)}")
    if warning_checks:
        st.warning(f"Warnings: {', '.join(label_map.get(c, c) for c in warning_checks)}")

    if failed_checks or warning_checks:
        with st.expander("Check Summary", expanded=True):
            if failed_checks:
                for check_name in failed_checks:
                    st.markdown(f"- ❌ {badge_html('Fail', 'red')} {label_map.get(check_name, check_name)}", unsafe_allow_html=True)
            if warning_checks:
                for check_name in warning_checks:
                    st.markdown(f"- ⚠️ {badge_html('Warning', 'yellow')} {label_map.get(check_name, check_name)}", unsafe_allow_html=True)

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
        is_passed = bool(result.get("passed", False))
        tl_c, tl_l, tl_d = score_check_result(is_passed, status_code)
        title = label_map.get(name, name.capitalize())
        expander_label = f"{icon_map.get(status_code, 'ℹ️')} {title}"
        with st.expander(expander_label, expanded=not is_passed):
            # Traffic-light badge at the top of each check section
            st.markdown(
                badge_html(tl_l, tl_c, tl_d) + f"&nbsp; <small style='color:#555'>{tl_d}</small>",
                unsafe_allow_html=True,
            )
            st.markdown("")
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
                null_val = result.get("null_pct")
                d1.metric("Null %", _fmt_pct(null_val))
                d2.metric("Row Coverage %", _fmt_pct(result.get("row_non_null_pct")))
                d3.metric("Zero-variance cols", _fmt_number(result.get("zero_var_count"), decimals=0))
                null_tl_c, null_tl_l, null_tl_d = score_null_pct(null_val)
                st.markdown(
                    f"**Data density:** " + badge_html(null_tl_l, null_tl_c, null_tl_d),
                    unsafe_allow_html=True,
                )

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
                    oos_r2_val = metrics.get("out_of_sample_r2")
                    oos_ci_lo = metrics.get("out_of_sample_r2_ci_lower")
                    oos_ci_hi = metrics.get("out_of_sample_r2_ci_upper")
                    oos_conf = metrics.get("out_of_sample_r2_ci_confidence")
                    if oos_ci_lo is not None and oos_ci_hi is not None:
                        conf_label = f"{int(round((oos_conf or 0.90) * 100))}% CI"
                        oos_label = f"OOS R² ({conf_label})"
                        oos_display = f"{_fmt_number(oos_r2_val, 4)}  [{_fmt_number(oos_ci_lo, 4)}, {_fmt_number(oos_ci_hi, 4)}]"
                    else:
                        oos_label = "OOS R²"
                        oos_display = _fmt_number(oos_r2_val, 4)
                    m1.metric(oos_label, oos_display)

                    # Traffic-light badge for OOS R²
                    r2_tl_c, r2_tl_l, r2_tl_d = score_oos_r2(oos_r2_val)
                    st.markdown(
                        f"**OOS R² quality:** " + badge_html(r2_tl_l, r2_tl_c, r2_tl_d),
                        unsafe_allow_html=True,
                    )

                    wf_avg = metrics.get("walk_forward_avg_r2")
                    wf_ci_lo = metrics.get("walk_forward_r2_ci_lower")
                    wf_ci_hi = metrics.get("walk_forward_r2_ci_upper")
                    if wf_ci_lo is not None and wf_ci_hi is not None:
                        wf_display = f"{_fmt_number(wf_avg, 4)}  [{_fmt_number(wf_ci_lo, 4)}, {_fmt_number(wf_ci_hi, 4)}]"
                        wf_label = "Walk-forward avg R² (range)"
                    else:
                        wf_display = _fmt_number(wf_avg, 4)
                        wf_label = "Walk-forward avg R²"
                    m2.metric(wf_label, wf_display)
                    m3.metric("Model risk", _fmt_number(metrics.get("model_risk_score"), decimals=4))
                    # Traffic-light badge for model risk
                    mr_c, mr_l, mr_d = score_model_risk(metrics.get("model_risk_score"))
                    st.markdown(
                        f"**Model risk:** " + badge_html(mr_l, mr_c, mr_d),
                        unsafe_allow_html=True,
                    )
                    lag_ok = metrics.get("publication_lag_compliant")
                    if lag_ok is True:
                        st.success("Look-ahead protection: macro publication lag is compliant (>=45 days).")
                    elif lag_ok is False:
                        st.error("Look-ahead risk detected: at least one macro feature is below 45-day publication lag.")
                lag_findings = result.get("publication_lag_findings", [])
                if lag_findings:
                    st.markdown("**Publication lag findings:**")
                    for finding in lag_findings:
                        st.markdown(f"- {finding}")
                if result.get("likely_over_strict"):
                    st.warning("Governance appears methodologically valid but conservative for this mixed-frequency data regime.")

            if name == "survivorship":
                metrics = result.get("metrics", {})
                if isinstance(metrics, dict) and metrics:
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Stale tickers", _fmt_number(metrics.get("stale_ticker_count"), decimals=0))
                    s2.metric("Zero-volume tickers", _fmt_number(metrics.get("zero_volume_ticker_count"), decimals=0))
                    s3.metric(
                        "Max zero-volume streak",
                        _fmt_number(metrics.get("max_zero_volume_streak_days"), decimals=0),
                    )
                stale = result.get("stale_tickers", [])
                if stale:
                    st.markdown("**Possible delisted/stale tickers:**")
                    for item in stale:
                        st.markdown(f"- {item}")
                streaks = result.get("zero_volume_streaks", {})
                if isinstance(streaks, dict) and streaks:
                    st.markdown("**Zero-volume streaks (>10 days):**")
                    for tkr, days in streaks.items():
                        st.markdown(f"- {tkr}: {days} days")

            if name == "thresholds":
                dynamic = result.get("dynamic_thresholds", {})
                if isinstance(dynamic, dict) and dynamic:
                    st.markdown("**🧩 Dynamic threshold flags:**")
                    rows = [
                        {
                            "Threshold": str(k).replace("_", " ").title(),
                            "Enabled": "Yes" if bool(v) else "No",
                        }
                        for k, v in dynamic.items()
                    ]
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

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
                    coverage_rows = []
                    for source, value in metrics.items():
                        cov_c, cov_l, _ = score_source_coverage(value)
                        coverage_rows.append(
                            {
                                "Source": source,
                                "Row coverage %": _fmt_pct(value),
                                "Cell fill %": _fmt_pct(cell_fill.get(source) if isinstance(cell_fill, dict) else None),
                                "Status": cov_l,
                            }
                        )
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
                    _render_kv_table(metrics_payload, title="Metrics summary")

            with st.expander("Archive", expanded=False):
                st.download_button(
                    "Download check JSON",
                    json.dumps(result, indent=2, ensure_ascii=False),
                    file_name=f"audit_check_{name}.json",
                    mime="application/json",
                    key=f"audit_download_{name}",
                )


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
