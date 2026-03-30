from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from UI.constants import OUTPUT_DIR, ROLE_PERMISSIONS, UI_SCHEDULE_PATH, USER_DATA_DIR
from UI.helpers import load_session_history, read_json
from UI.runtime import get_audit_report, run_and_cache_audit
from UI.tabs.assistant_tab import render_inline_ai_section
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


def _decode_reason(raw: str) -> str:
    """Translate technical governance reason codes to human-readable text."""
    r = str(raw)
    if "r2_metric_alert_oos_below_threshold" in r:
        try:
            val_part = r.split(":", 1)[1] if ":" in r else r
            val_s, thr_s = val_part.split("<", 1)
            return (
                f"\U0001f4c9 OOS R\u00b2 ({float(val_s):.4f}) is below the advisory threshold "
                f"({float(thr_s):.4f}). Negative OOS R\u00b2 means the model predicts "
                "slightly worse than the historical mean \u2014 this is **common and normal** "
                "in macro factor equity forecasting. "
                "**Advisory only \u2014 does not block outputs.**"
            )
        except Exception:
            return f"OOS R\u00b2 below advisory threshold: {r}"
    if "r2_metric_alert_walk_forward_below_threshold" in r:
        try:
            val_part = r.split(":", 1)[1] if ":" in r else r
            val_s, thr_s = val_part.split("<", 1)
            return (
                f"\U0001f4c9 Walk-forward avg R\u00b2 ({float(val_s):.4f}) is below the advisory "
                f"threshold ({float(thr_s):.4f}). Walk-forward performance varies across "
                "regimes \u2014 macro factor predictiveness changes over time. "
                "**Advisory only \u2014 does not block outputs.**"
            )
        except Exception:
            return f"Walk-forward R\u00b2 below advisory threshold: {r}"
    if "leakage" in r.lower():
        return f"\u26a0\ufe0f Potential data leakage flag: {r}"
    if "stationarity" in r.lower():
        return f"\U0001f4ca Stationarity notice: {r}"
    return r


def _render_check_summary_table(report: dict, label_map: dict) -> None:
    """Compact table of all check statuses."""
    icon_status = {"pass": "\u2705 Pass", "warn": "\u26a0\ufe0f Warn", "fail": "\u274c Fail"}
    rows = []
    for name, result in report.get("checks", {}).items():
        status_code = str(result.get("status", "warn"))
        n_issues = len(result.get("issues", [])) + len(result.get("reasons", []))
        rows.append(
            {
                "Check": label_map.get(name, name.capitalize()),
                "Result": icon_status.get(status_code, f"\u2753 {status_code}"),
                "Advisory items": n_issues if n_issues else "\u2014",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _artifact_facts(path: Path) -> dict[str, str]:
    exists = path.exists()
    if not exists:
        return {
            "Artifact": str(path.name),
            "Exists": "No",
            "Modified": "N/A",
            "Size KB": "N/A",
        }
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    return {
        "Artifact": str(path.name),
        "Exists": "Yes",
        "Modified": modified,
        "Size KB": f"{(stat.st_size / 1024.0):.1f}",
    }


def _render_pipeline_lineage() -> None:
    st.markdown("### 🔗 Pipeline Lineage (Bronze/Silver/Gold -> UI)")
    active_user = os.getenv("DATA_USER_ID", "default")
    st.caption(f"Active profile: {active_user}")

    raw_catalog = USER_DATA_DIR / "raw" / "catalog.json"
    silver_quality = USER_DATA_DIR / "processed" / "quality" / "quality_report.json"
    gold_master = USER_DATA_DIR / "gold" / "master_table.parquet"
    out_analysis = OUTPUT_DIR / "analysis_results.json"
    out_backtest = OUTPUT_DIR / "backtest_2020.json"
    out_audit = OUTPUT_DIR / "audit_report.json"

    rows = [
        _artifact_facts(raw_catalog),
        _artifact_facts(silver_quality),
        _artifact_facts(gold_master),
        _artifact_facts(out_analysis),
        _artifact_facts(out_backtest),
        _artifact_facts(out_audit),
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    available_profiles = sorted([p.name for p in OUTPUT_DIR.parent.iterdir() if p.is_dir()]) if OUTPUT_DIR.parent.exists() else []
    if available_profiles:
        st.caption("Detected output profiles: " + ", ".join(available_profiles))


def _audit_report_is_complete(report: dict) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks:
        return False

    # New audit payloads may omit top-level row_count/column_count while still
    # carrying complete per-check diagnostics and an overall status.
    has_status = isinstance(report.get("status"), str) and bool(str(report.get("status")).strip())
    has_decision = isinstance(report.get("decision_ready"), bool)
    core_names = (
        "integration",
        "density",
        "statistics",
        "continuity",
        "survivorship",
        "outputs",
        "thresholds",
        "governance",
    )
    present_core_checks = sum(1 for name in core_names if isinstance(checks.get(name), dict))
    # Accept legacy payloads that may not contain all eight checks.
    has_core_checks = present_core_checks >= 4

    # If row/column counters exist, validate them; if absent, do not mark stale.
    row_count = report.get("row_count")
    column_count = report.get("column_count")
    counters_valid = True
    if row_count is not None:
        counters_valid = counters_valid and isinstance(row_count, int) and row_count > 0
    if column_count is not None:
        counters_valid = counters_valid and isinstance(column_count, int) and column_count > 0

    return bool(has_status and has_decision and has_core_checks and counters_valid)


def show_auditor_tab() -> None:
    """Multi-tab System Auditor with 6 tabs and human-readable explanations."""
    st.subheader("\U0001f9eb System Auditor")
    st.caption(
        "Independent audit judge: source coverage, data density, statistics, "
        "temporal continuity, survivorship bias, output quality, threshold design "
        "and governance validity."
    )

    header_c1, header_c2 = st.columns([1, 1])
    with header_c2:
        if st.button("Reload Saved Audit", key="reload_saved_audit"):
            st.session_state.pop("audit_report", None)
            st.rerun()

    active_user = os.getenv("DATA_USER_ID", "default")
    st.caption(f"Active audit profile: {active_user}")

    report = get_audit_report()
    if not report:
        st.info("No audit report found yet. Run the pipeline first, then re-run the audit.")
        if st.button("\u27f3 Re-run Audit", key="manual_audit_run_empty"):
            with st.spinner("Running audit..."):
                fresh_report = run_and_cache_audit()
            if isinstance(fresh_report, dict) and str(fresh_report.get("status", "")).upper() == "ERROR":
                st.session_state["audit_last_run_message"] = "Audit re-run failed: " + str(fresh_report.get("error", "unknown error"))
            else:
                st.session_state["audit_last_run_message"] = "Audit re-run completed and saved."
            st.rerun()
        last_msg = st.session_state.get("audit_last_run_message")
        if isinstance(last_msg, str) and last_msg:
            if last_msg.lower().startswith("audit re-run failed"):
                st.error(last_msg)
            else:
                st.success(last_msg)
        return

    _LABEL_MAP = {
        "integration": "Source Integration",
        "density": "Data Density",
        "statistics": "Statistical Plausibility",
        "continuity": "Temporal Continuity",
        "survivorship": "Survivorship Bias",
        "outputs": "Output Quality",
        "thresholds": "Threshold Design",
        "governance": "Governance Validity",
    }
    _ICON_MAP = {"pass": "\u2705", "warn": "\u26a0\ufe0f", "fail": "\u274c"}

    (
        tab_overview,
        tab_governance,
        tab_data_quality,
        tab_sources,
        tab_outputs,
        tab_export,
    ) = st.tabs(
        [
            "\U0001f4cb Overview",
            "\U0001f6e1\ufe0f Governance & Model",
            "\U0001f4ca Data Quality",
            "\U0001f517 Sources & Coverage",
            "\u2699\ufe0f Outputs & Thresholds",
            "\U0001f4e5 Export",
        ]
    )

    # ── Tab 1 · Overview ─────────────────────────────────────────────────────
    with tab_overview:
        _render_pipeline_lineage()
        st.markdown("---")
        report_is_complete = _audit_report_is_complete(report)
        effective_status = report.get("status", "UNKNOWN") if report_is_complete else "ERROR"
        # Pass failed_count so ≤2 advisory failures render yellow, not red
        tl_color, tl_label, tl_desc = score_audit_status(
            effective_status,
            failed_count=len(report.get("failed_checks", [])),
        )
        if not report_is_complete:
            tl_desc = "The loaded audit report is incomplete or stale. Reload the saved audit artifact or re-run the audit."
        st.markdown(
            badge_html(tl_label, tl_color, tl_desc) + f"&nbsp; <span style='font-size:1.05rem;color:#555'>{tl_desc}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        failed_checks = report.get("failed_checks", [])
        warning_checks = report.get("warning_checks", [])
        n_failed = len(failed_checks)
        dr_color, dr_label, dr_desc = score_decision_ready(bool(report.get("decision_ready")), failed_count=n_failed)
        row_count = report.get("row_count")
        column_count = report.get("column_count")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", str(row_count) if isinstance(row_count, int) else "—")
        c2.metric("Columns", str(column_count) if isinstance(column_count, int) else "—")
        c3.markdown(badge_html(dr_label, dr_color, dr_desc), unsafe_allow_html=True)
        c4.metric("Failed Checks", len(failed_checks))
        c5.metric("Warning Checks", len(warning_checks))

        if not report_is_complete:
            st.warning("The audit report loaded in the UI is incomplete or stale. Success messaging is suppressed until a complete report is available.")
        if failed_checks:
            check_names = ", ".join(_LABEL_MAP.get(c, c) for c in failed_checks)
            if n_failed <= 2:
                # 1–2 advisory failures: caution, not a hard block
                st.warning(
                    f"**{n_failed} advisory check(s) flagged:** {check_names}  \n"
                    "These checks are informational in mixed-frequency regimes and "
                    "do not prevent use of the outputs."
                )
            else:
                st.error(f"**{n_failed} checks failed:** {check_names}")
        if warning_checks:
            st.warning("**Warnings:** " + ", ".join(_LABEL_MAP.get(c, c) for c in warning_checks))
        if report_is_complete and not failed_checks and not warning_checks:
            st.success("\u2705 All 8 checks passed \u2014 system is production-ready.")

        st.markdown("---")
        st.markdown("### Check Results at a Glance")
        _render_check_summary_table(report, _LABEL_MAP)

        judgement = report.get("auditor_judgement", {})
        if isinstance(judgement, dict) and judgement:
            st.markdown("---")
            st.markdown("### \U0001f9e0 Auditor Judgement")
            j_n_failed = judgement.get("n_failed", len(report.get("failed_checks", [])))

            def _judgement_label(flag: bool, advisory_threshold: int = 2) -> str:
                if flag:
                    return "\u2705 Yes"
                if j_n_failed <= advisory_threshold:
                    return "\u26a0\ufe0f Limited"
                return "\u274c No"

            j1, j2, j3 = st.columns(3)
            j1.metric(
                "Data Quality Acceptable",
                _judgement_label(bool(judgement.get("is_information_reasonable"))),
            )
            j2.metric(
                "Supports Decision-Making",
                _judgement_label(bool(judgement.get("can_support_decisions"))),
            )
            edge_score = judgement.get("strategic_edge_score")
            j3.metric(
                "Strategic Edge Score",
                f"{edge_score:.0f} / 100" if isinstance(edge_score, (int, float)) else "—",
            )
            for line in judgement.get("summary", []):
                st.markdown(f"- {line}")

        st.markdown("---")
        if st.button("\u27f3 Re-run Audit", key="manual_audit_run"):
            with st.spinner("Running audit..."):
                fresh_report = run_and_cache_audit()
            if isinstance(fresh_report, dict) and str(fresh_report.get("status", "")).upper() == "ERROR":
                st.session_state["audit_last_run_message"] = "Audit re-run failed: " + str(fresh_report.get("error", "unknown error"))
            else:
                st.session_state["audit_last_run_message"] = "Audit re-run completed and saved."
            st.rerun()

        last_msg = st.session_state.get("audit_last_run_message")
        if isinstance(last_msg, str) and last_msg:
            if last_msg.lower().startswith("audit re-run failed"):
                st.error(last_msg)
            else:
                st.success(last_msg)

    # ── Tab 2 · Governance & Model ────────────────────────────────────────────
    with tab_governance:
        gov = report.get("checks", {}).get("governance", {})
        if not gov:
            st.info("No governance check data available.")
        else:
            status_code = str(gov.get("status", "warn"))
            is_passed = bool(gov.get("passed", False))
            tl_c, tl_l, tl_d = score_check_result(is_passed, status_code)
            st.markdown(
                badge_html(tl_l, tl_c, tl_d) + f"&nbsp; <small style='color:#555'>{tl_d}</small>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            metrics = gov.get("metrics", {})
            if isinstance(metrics, dict) and metrics:
                oos_r2_val = metrics.get("out_of_sample_r2")
                oos_ci_lo = metrics.get("out_of_sample_r2_ci_lower")
                oos_ci_hi = metrics.get("out_of_sample_r2_ci_upper")
                oos_conf = metrics.get("out_of_sample_r2_ci_confidence")
                wf_avg = metrics.get("walk_forward_avg_r2")
                wf_med = metrics.get("walk_forward_median_r2")
                wf_clipped = metrics.get("walk_forward_clipped_avg_r2")
                wf_ci_lo = metrics.get("walk_forward_r2_ci_lower")
                wf_ci_hi = metrics.get("walk_forward_r2_ci_upper")
                model_risk = metrics.get("model_risk_score")

                conf_label = f"{int(round((oos_conf or 0.90) * 100))}% CI"
                m1, m2, m3 = st.columns(3)
                oos_display = (
                    f"{_fmt_number(oos_r2_val, 4)}  [{_fmt_number(oos_ci_lo, 4)}, {_fmt_number(oos_ci_hi, 4)}]"
                    if oos_ci_lo is not None and oos_ci_hi is not None
                    else _fmt_number(oos_r2_val, 4)
                )
                m1.metric(f"OOS R\u00b2 ({conf_label})", oos_display)
                wf_display = (
                    f"{_fmt_number(wf_avg, 4)}  [{_fmt_number(wf_ci_lo, 4)}, {_fmt_number(wf_ci_hi, 4)}]"
                    if wf_ci_lo is not None and wf_ci_hi is not None
                    else _fmt_number(wf_avg, 4)
                )
                m2.metric("Walk-forward avg R\u00b2 (range)", wf_display)
                m3.metric("Model Risk Score", _fmt_number(model_risk, 4))

                r2_tl_c, r2_tl_l, r2_tl_d = score_oos_r2(oos_r2_val)
                mr_c, mr_l, mr_d = score_model_risk(model_risk)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(
                        "**OOS R\u00b2 quality:** " + badge_html(r2_tl_l, r2_tl_c, r2_tl_d),
                        unsafe_allow_html=True,
                    )
                    if wf_med is not None:
                        st.caption(
                            f"Walk-forward median R\u00b2 = {_fmt_number(wf_med, 4)}"
                            + (f" | clipped avg = {_fmt_number(wf_clipped, 4)}" if wf_clipped is not None else "")
                        )
                with col_b:
                    st.markdown(
                        "**Model risk:** " + badge_html(mr_l, mr_c, mr_d),
                        unsafe_allow_html=True,
                    )

                lag_ok = metrics.get("publication_lag_compliant")
                if lag_ok is True:
                    st.success("\u2705 Look-ahead protection: macro publication lag compliant (\u226545 days).")
                elif lag_ok is False:
                    st.error("\u26a0\ufe0f Look-ahead risk: at least one macro feature is below the 45-day publication lag requirement.")

            gate_passed = gov.get("gate_passed")
            severity = gov.get("severity", "")
            if gate_passed is not None:
                g_c, g_l, g_d = score_governance_gate(bool(gate_passed), str(severity))
                st.markdown("---")
                st.markdown("### \U0001f512 Governance Gate Decision")
                st.markdown(
                    badge_html(g_l, g_c, g_d) + f"&nbsp; <small>{g_d}</small>",
                    unsafe_allow_html=True,
                )

            reasons = gov.get("reasons", [])
            if reasons:
                st.markdown("---")
                st.markdown("### \u26a0\ufe0f Advisory Alerts")
                st.info(
                    "These alerts report metric observations. They are **advisory only** "
                    "and do **not** cause the audit to fail. "
                    "The governance gate uses them as informational signals only "
                    "(`r2_used_as_validation_gate = false`)."
                )
                for raw_reason in reasons:
                    st.markdown(f"- {_decode_reason(raw_reason)}")

            lag_findings = gov.get("publication_lag_findings", [])
            if lag_findings:
                st.markdown("---")
                st.markdown("**Publication lag findings:**")
                for finding in lag_findings:
                    st.markdown(f"- {finding}")

            if gov.get("likely_over_strict"):
                st.warning("\U0001f52c Governance appears methodologically valid but conservative for this mixed-frequency data regime.")

            interpretation = gov.get("interpretation", [])
            if interpretation:
                with st.expander("\U0001f4d6 Interpretation notes", expanded=False):
                    items = [interpretation] if isinstance(interpretation, str) else interpretation
                    for note in items:
                        st.markdown(f"- {note}")

            reasoning = gov.get("reasoning", [])
            if reasoning:
                with st.expander("\U0001f50d Auditor reasoning", expanded=False):
                    for r_item in reasoning:
                        st.markdown(f"- {r_item}")

            render_inline_ai_section(
                topic="Governance Audit",
                snapshot={
                    "governance_passed": gov.get("passed"),
                    "severity": gov.get("severity"),
                    "advisory_reasons": [_decode_reason(r) for r in reasons],
                    "metrics": gov.get("metrics", {}),
                },
                key_suffix="auditor_gov",
            )

    # ── Tab 3 · Data Quality ──────────────────────────────────────────────────
    with tab_data_quality:
        for check_name in ("density", "statistics", "continuity"):
            result = report.get("checks", {}).get(check_name, {})
            if not result:
                continue
            status_code = str(result.get("status", "warn"))
            is_passed = bool(result.get("passed", False))
            tl_c, tl_l, tl_d = score_check_result(is_passed, status_code)
            icon = _ICON_MAP.get(status_code, "\u2139\ufe0f")
            title = _LABEL_MAP.get(check_name, check_name.capitalize())
            with st.expander(f"{icon} {title}", expanded=not is_passed):
                st.markdown(
                    badge_html(tl_l, tl_c, tl_d) + f"&nbsp; <small style='color:#555'>{tl_d}</small>",
                    unsafe_allow_html=True,
                )
                st.markdown("")
                interpretation = result.get("interpretation", "")
                if isinstance(interpretation, str) and interpretation:
                    st.caption(interpretation)
                elif isinstance(interpretation, list):
                    for note in interpretation:
                        st.markdown(f"- {note}")
                for issue in result.get("issues", []):
                    st.markdown(f"- \u274c {issue}")

                if check_name == "density":
                    d1, d2, d3 = st.columns(3)
                    null_val = result.get("null_pct")
                    d1.metric("Null %", _fmt_pct(null_val))
                    d2.metric("Row Coverage %", _fmt_pct(result.get("row_non_null_pct")))
                    d3.metric("Zero-variance cols", _fmt_number(result.get("zero_var_count"), decimals=0))
                    null_tl_c, null_tl_l, null_tl_d = score_null_pct(null_val)
                    st.markdown(
                        "**Data density:** " + badge_html(null_tl_l, null_tl_c, null_tl_d),
                        unsafe_allow_html=True,
                    )
                    for col in result.get("zero_var_columns", []):
                        st.markdown(f"- Zero-variance: `{col}`")
                    for col in result.get("all_zero_columns", []):
                        st.markdown(f"- Always-zero: `{col}`")

                if check_name == "continuity":
                    cmax, cmed, callowed, cbase = st.columns(4)
                    cmax.metric("Max Gap (bdays)", _fmt_number(result.get("max_gap"), decimals=0))
                    cmed.metric("Median Gap (bdays)", _fmt_number(result.get("median_gap")))
                    callowed.metric("Auto Allowed Gap", _fmt_number(result.get("allowed_gap"), decimals=0))
                    cbase.metric("Detected Cadence", _fmt_number(result.get("detected_baseline_gap"), decimals=0))
                    cfg_gap = result.get("configured_allowed_gap")
                    dyn_gap = result.get("allowed_gap")
                    if cfg_gap is not None and dyn_gap is not None and dyn_gap != cfg_gap:
                        cadence_name = (
                            "monthly"
                            if (result.get("detected_baseline_gap") or 0) >= 15
                            else "weekly"
                            if (result.get("detected_baseline_gap") or 0) >= 4
                            else "daily"
                        )
                        st.info(f"Auto-detected **{cadence_name}** cadence \u2014 threshold scaled from {cfg_gap} \u2192 **{dyn_gap}** business days.")
                    if result.get("duplicate_rows_removed", 0):
                        st.warning(f"Duplicate rows removed: {result.get('duplicate_rows_removed')}")
                    fg = result.get("failed_groups", [])
                    if fg:
                        st.error(f"Groups failing gap check: {', '.join(fg)}")
                    for w in result.get("warnings", []):
                        st.markdown(f"- \u26a0\ufe0f {w}")
                    if result.get("error"):
                        st.error(f"Continuity error: {result.get('error')}")

                if check_name == "statistics":
                    metrics_payload = result.get("metrics", {})
                    if isinstance(metrics_payload, dict) and metrics_payload:
                        _render_kv_table(metrics_payload, title="Statistical plausibility metrics")

    # ── Tab 4 · Sources & Coverage ────────────────────────────────────────────
    with tab_sources:
        for check_name in ("integration", "survivorship"):
            result = report.get("checks", {}).get(check_name, {})
            if not result:
                continue
            status_code = str(result.get("status", "warn"))
            is_passed = bool(result.get("passed", False))
            tl_c, tl_l, tl_d = score_check_result(is_passed, status_code)
            icon = _ICON_MAP.get(status_code, "\u2139\ufe0f")
            title = _LABEL_MAP.get(check_name, check_name.capitalize())
            with st.expander(f"{icon} {title}", expanded=not is_passed):
                st.markdown(
                    badge_html(tl_l, tl_c, tl_d) + f"&nbsp; <small style='color:#555'>{tl_d}</small>",
                    unsafe_allow_html=True,
                )
                st.markdown("")
                interpretation = result.get("interpretation", "")
                if isinstance(interpretation, str) and interpretation:
                    st.caption(interpretation)
                elif isinstance(interpretation, list):
                    for note in interpretation:
                        st.markdown(f"- {note}")
                for issue in result.get("issues", []):
                    st.markdown(f"- \u274c {issue}")

                if check_name == "integration":
                    metrics = result.get("metrics", {})
                    cell_fill = result.get("cell_fill_pct", {})
                    overall = result.get("overall", {})
                    if isinstance(overall, dict) and overall:
                        o1, o2 = st.columns(2)
                        o1.metric("Rows with all 3 sources", _fmt_pct(overall.get("rows_with_all_sources_pct")))
                        avg_groups = overall.get("avg_active_source_groups_per_row")
                        max_groups = overall.get("max_source_groups_per_row", 3)
                        o2.metric(
                            "Avg active source groups / row",
                            f"{float(avg_groups):.2f} / {max_groups}" if isinstance(avg_groups, (int, float)) else "N/A",
                        )
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
                        b_rows = []
                        for source, payload_b in breadth.items():
                            if isinstance(payload_b, dict):
                                b_rows.append(
                                    {
                                        "Source": source,
                                        "Expected": payload_b.get("expected", 0),
                                        "Observed": payload_b.get("observed", 0),
                                        "Coverage ratio": payload_b.get("ratio", 0),
                                    }
                                )
                        if b_rows:
                            st.dataframe(pd.DataFrame(b_rows), width="stretch")

                if check_name == "survivorship":
                    metrics = result.get("metrics", {})
                    if isinstance(metrics, dict) and metrics:
                        s1, s2, s3 = st.columns(3)
                        s1.metric("Stale tickers", _fmt_number(metrics.get("stale_ticker_count"), decimals=0))
                        s2.metric("Zero-volume tickers", _fmt_number(metrics.get("zero_volume_ticker_count"), decimals=0))
                        s3.metric("Max zero-vol streak", _fmt_number(metrics.get("max_zero_volume_streak_days"), decimals=0))
                    for tkr in result.get("stale_tickers", []):
                        st.markdown(f"- Possible stale/delisted ticker: **{tkr}**")
                    streaks = result.get("zero_volume_streaks", {})
                    if isinstance(streaks, dict) and streaks:
                        st.markdown("**Zero-volume streaks (>10 days):**")
                        for tkr, days in streaks.items():
                            st.markdown(f"- {tkr}: {days} days")

    # ── Tab 5 · Outputs & Thresholds ──────────────────────────────────────────
    with tab_outputs:
        for check_name in ("outputs", "thresholds"):
            result = report.get("checks", {}).get(check_name, {})
            if not result:
                continue
            status_code = str(result.get("status", "warn"))
            is_passed = bool(result.get("passed", False))
            tl_c, tl_l, tl_d = score_check_result(is_passed, status_code)
            icon = _ICON_MAP.get(status_code, "\u2139\ufe0f")
            title = _LABEL_MAP.get(check_name, check_name.capitalize())
            with st.expander(f"{icon} {title}", expanded=not is_passed):
                st.markdown(
                    badge_html(tl_l, tl_c, tl_d) + f"&nbsp; <small style='color:#555'>{tl_d}</small>",
                    unsafe_allow_html=True,
                )
                st.markdown("")
                interpretation = result.get("interpretation", "")
                if isinstance(interpretation, str) and interpretation:
                    st.caption(interpretation)
                elif isinstance(interpretation, list):
                    for note in interpretation:
                        st.markdown(f"- {note}")

                if check_name == "outputs":
                    st.metric("Output sections", result.get("result_key_count", 0))
                    usable = result.get("usable_outputs", [])
                    if usable:
                        st.markdown("**\u2705 Usable outputs:**")
                        for o in usable:
                            st.markdown(f"- {o}")
                    blocked = result.get("blocked_outputs", [])
                    if blocked:
                        st.markdown("**\u274c Blocked outputs:**")
                        for o in blocked:
                            st.markdown(f"- {o}")
                    nullish = result.get("nullish_outputs", [])
                    if nullish:
                        st.markdown("**\u26a0\ufe0f Empty/nullish outputs:**")
                        for o in nullish:
                            st.markdown(f"- {o}")

                if check_name == "thresholds":
                    dynamic = result.get("dynamic_thresholds", {})
                    if isinstance(dynamic, dict) and dynamic:
                        st.markdown("**\U0001f9e9 Dynamic threshold flags:**")
                        t_rows = [
                            {
                                "Threshold": str(k).replace("_", " ").title(),
                                "Enabled": "\u2705 Yes" if bool(v) else "\u274c No",
                            }
                            for k, v in dynamic.items()
                        ]
                        st.dataframe(pd.DataFrame(t_rows), width="stretch", hide_index=True)
                    for finding in result.get("strictness_findings", []):
                        st.markdown(f"- {finding}")

    # ── Tab 6 · Export ────────────────────────────────────────────────────────
    with tab_export:
        st.markdown("### \U0001f4e5 Audit Report Downloads")
        st.download_button(
            "\u2b07\ufe0f Download Full Audit Report (JSON)",
            json.dumps(report, indent=2, ensure_ascii=False),
            file_name="audit_report_full.json",
            mime="application/json",
            key="audit_download_full",
        )
        st.markdown("---")
        st.markdown("**Per-check downloads:**")
        dl_cols = st.columns(4)
        for idx, (name, check_result) in enumerate(report.get("checks", {}).items()):
            with dl_cols[idx % 4]:
                st.download_button(
                    f"\u2b07\ufe0f {_LABEL_MAP.get(name, name)}",
                    json.dumps(check_result, indent=2, ensure_ascii=False),
                    file_name=f"audit_check_{name}.json",
                    mime="application/json",
                    key=f"audit_dl_{name}",
                )

    # ── Inline Quantos at the bottom (outside tabs) ────────────────────────────
    st.markdown("---")
    render_inline_ai_section(
        topic="Full Audit Analysis",
        snapshot={
            "audit_status": report.get("status"),
            "failed_checks": report.get("failed_checks", []),
            "warning_checks": report.get("warning_checks", []),
            "decision_ready": report.get("decision_ready"),
            "auditor_judgement": report.get("auditor_judgement", {}),
            "governance_advisory_reasons": [_decode_reason(r) for r in report.get("checks", {}).get("governance", {}).get("reasons", [])],
            "governance_metrics": report.get("checks", {}).get("governance", {}).get("metrics", {}),
        },
        key_suffix="auditor",
    )


def show_ops_tab(role: str) -> None:
    st.subheader("Job History & Scheduling")

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

    st.markdown("### Nightly Scheduling")
    schedule = read_json(UI_SCHEDULE_PATH) if UI_SCHEDULE_PATH.exists() else {}
    enabled_default = bool(schedule.get("enabled", False))
    hour_default = int(schedule.get("hour", 2)) if isinstance(schedule.get("hour"), int) else 2
    minute_default = int(schedule.get("minute", 0)) if isinstance(schedule.get("minute"), int) else 0

    enabled = st.toggle(
        "Enable nightly schedule",
        value=enabled_default,
        disabled=not ROLE_PERMISSIONS[role]["can_schedule"],
    )
    c1, c2 = st.columns(2)
    hour = c1.number_input("Hour (24h)", min_value=0, max_value=23, value=hour_default, step=1, disabled=not ROLE_PERMISSIONS[role]["can_schedule"])
    minute = c2.number_input("Minute", min_value=0, max_value=59, value=minute_default, step=1, disabled=not ROLE_PERMISSIONS[role]["can_schedule"])
    if st.button("Save schedule", disabled=not ROLE_PERMISSIONS[role]["can_schedule"]):
        payload = {
            "enabled": enabled,
            "hour": int(hour),
            "minute": int(minute),
            "saved_at": datetime.now().isoformat(),
        }
        UI_SCHEDULE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        st.success("Schedule saved.")

    st.info("History deletion is available from the sidebar under History Control.")
