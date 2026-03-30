from __future__ import annotations

from datetime import datetime

import streamlit as st

from UI.constants import OUTPUT_DIR, ROLE_PERMISSIONS
from UI.runtime import run_and_cache_audit, run_gold_analyses_only
from UI.helpers import (
    build_executive_report_html,
    build_explainability_lines,
    build_run_comparison,
    build_smart_alerts,
    compute_data_health,
    persist_human_report_files,
    read_json,
)
from UI.tabs.assistant_tab import render_inline_ai_section


def show_health_alerts_tab() -> None:
    st.subheader("🩺 Data Health & Smart Alerts")
    health = compute_data_health()

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

    st.markdown("### 🚨 Smart Alerts")
    alerts = build_smart_alerts()
    if not alerts:
        st.success("No active alerts.")
    else:
        for alert in alerts:
            msg = f"**{alert['title']}** — {alert['message']}"
            if alert.get("severity") == "critical":
                st.error(msg)
            else:
                st.warning(msg)

    # Inline AI insight contextualised to current health snapshot
    render_inline_ai_section(
        topic="Data Health & Alerts",
        snapshot={
            "health": health,
            "alerts": [a.get("title", "") + ": " + a.get("message", "") for a in (alerts or [])],
        },
        key_suffix="health",
    )


def show_run_comparison_tab() -> None:
    st.subheader("📊 Run Comparison (Latest vs Previous)")
    diff = build_run_comparison()
    if diff.get("status") != "ok":
        st.info(diff.get("message", "Not enough runs yet."))
        return

    c1, c2 = st.columns(2)
    c1.metric("Pipeline Duration", f"{diff['duration_curr']:.2f}s", f"{diff['duration_delta']:+.2f}s")
    c2.metric("Operations", diff["ops_curr"], f"{diff['ops_delta']:+d}")

    if "output_file_diff" in diff:
        st.markdown("### 📁 Output File Diff")
        files = diff["output_file_diff"]
        c3, c4, c5 = st.columns(3)
        c3.metric("Added", len(files["added"]))
        c4.metric("Removed", len(files["removed"]))
        c5.metric("Changed", len(files["changed"]))

    # ── Quantos AI Insights ───────────────────────────────────────────────────
    render_inline_ai_section(
        topic="Run Comparison — what changed between pipeline runs",
        snapshot={"diff_summary": diff if isinstance(diff, dict) else {}},
        key_suffix="run_comparison",
    )


def show_scenario_builder_tab() -> None:
    st.subheader("🎛️ Scenario Builder (Custom Shocks)")
    st.caption("Set custom macro shocks and see immediate impact estimates.")

    # ── Preset scenario selector (wired to analysis_suite PRESET_SCENARIOS) ──
    try:
        from analysis_suite import PRESET_SCENARIOS as _PRESETS
    except ImportError:
        _PRESETS = {}

    preset_options = ["custom"] + sorted(_PRESETS.keys())
    selected_preset = st.selectbox(
        "Preset scenario",
        options=preset_options,
        format_func=lambda k: k.replace("_", " ").title(),
    )
    if selected_preset != "custom" and selected_preset in _PRESETS:
        preset_meta = _PRESETS[selected_preset]
        st.info(f"**{selected_preset.replace('_', ' ').title()}** — {preset_meta['description']}")
        shocks_preview = ", ".join(f"{k}: {v:+.1%}" for k, v in preset_meta["factor_shocks"].items())
        st.caption(f"Preset factor shocks: {shocks_preview}")

    st.divider()

    c1, c2, c3 = st.columns(3)
    inflation_shock = c1.slider("Inflation shock (%)", -10.0, 10.0, 1.0, 0.5)
    energy_shock = c2.slider("Energy shock (%)", -20.0, 20.0, 2.0, 0.5)
    rate_shock = c3.slider("Rates shock (%)", -5.0, 5.0, 0.5, 0.25)

    summary = read_json(OUTPUT_DIR / "analysis_results.json")
    results = summary.get("results", {}) if isinstance(summary, dict) else {}
    elasticity_val = results.get("elasticity")

    # ── Contract-aware elasticity extraction: elasticity() returns a dict ──
    baseline_move = 0.0
    _static_elast: float | None = None
    if isinstance(elasticity_val, dict):
        _static_elast = elasticity_val.get("static_elasticity")
    elif isinstance(elasticity_val, (int, float)):
        _static_elast = float(elasticity_val)
    if isinstance(_static_elast, (int, float)):
        baseline_move = float(_static_elast) * (inflation_shock / 100.0)

    total_shock = (inflation_shock * 0.5 + energy_shock * 0.3 + rate_shock * 0.2) / 100.0
    estimated_return_shift = baseline_move + total_shock

    st.metric("Estimated return shift", f"{estimated_return_shift * 100:.2f}%")
    st.info("This is a fast directional estimate. Use Stress Test and Forecast outputs for detailed decisions.")

    # ── Stress test results from last pipeline run ────────────────────────────
    stress_res = results.get("stress_test")
    if isinstance(stress_res, dict) and "results" in stress_res:
        st.markdown("### 📊 Last-Run Stress Test Results")
        scenario_meta = stress_res.get("scenario", {})
        if scenario_meta.get("name"):
            st.caption(f"Scenario: **{scenario_meta['name'].replace('_', ' ').title()}** — {scenario_meta.get('description', '')}")
        for factor, detail in stress_res["results"].items():
            ca, cb, cc = st.columns(3)
            ca.metric(factor, f"{detail.get('shock', 0.0):+.2%}", delta_color="off")
            cb.metric("β", f"{detail.get('beta', 0.0):.4f}")
            cc.metric("Impact", f"{detail.get('predicted_impact', 0.0):.2%}", delta_color="inverse")

    # ── Quantos AI Insights ───────────────────────────────────────────────────
    render_inline_ai_section(
        topic="Scenario Builder — stress test shocks, macro impact analysis",
        snapshot={"stress_results": stress_res.get("results", {}) if isinstance(stress_res, dict) else {}},
        key_suffix="scenario_builder",
    )


def show_explainability_tab() -> None:
    st.subheader("🧠 Explainability Panel")
    st.caption("What changed and why — plain language summaries.")
    lines = build_explainability_lines()
    for line in lines:
        st.markdown(f"- {line}")

    # ── Quantos AI Insights ───────────────────────────────────────────────────
    try:
        render_inline_ai_section(
            topic="Strategy Explainability — factor attribution, signal sources, what drove recent P&L",
            snapshot={"explainability_lines": lines[:10] if lines else []},
            key_suffix="explainability_tab",
        )
    except Exception:
        pass


def show_gold_rerun_tab(role: str) -> None:
    st.subheader("⚡ Re-run Gold Analyses")
    st.caption(
        "Re-runs only the Gold layer analyses using the existing Silver data — no Bronze/Silver data fetch. "
        "Use when you want fresh analysis results without waiting for a full pipeline run (~2 min vs ~6 min)."
    )

    perms = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["Viewer"])
    if not perms.get("can_run"):
        st.warning("You do not have permission to run analyses.")
        return

    st.info(
        "This skips Bronze (data fetch) and Silver (cleaning) stages. "
        "Your existing Silver dataset is reused as-is."
    )

    if st.button("⚡ Re-run Gold Analyses Only", type="primary", width="stretch"):
        prog = st.progress(0, text="Starting Gold layer...")
        try:
            ok, output = run_gold_analyses_only(progress_bar=prog)
        except Exception as exc:
            prog.empty()
            st.error(f"Gold-only run failed: {exc}")
            return
        prog.empty()
        if ok:
            st.cache_data.clear()
            for _k in ["audit_report", "backtest_payload", "_backtest_payload_hash", "backtest_payload_loaded_at"]:
                st.session_state.pop(_k, None)
            st.success("Gold analyses completed successfully.")
            with st.spinner("Updating audit report..."):
                run_and_cache_audit()
            st.session_state["selected_page"] = "💎 Edge Arsenal"
            st.rerun()
        else:
            st.error("Gold-only run did not complete successfully.")
            with st.expander("Raw output", expanded=False):
                st.text(output[-3000:] if len(output) > 3000 else output)


def show_reports_tab(role: str) -> None:
    st.subheader("🧾 Human-Friendly Reports")
    st.caption("Readable report for humans in the UI. Raw JSON is kept only for archive/download.")
    html_report = build_executive_report_html()

    if st.button("💾 Save Human Report Files", key="save_human_report"):
        paths = persist_human_report_files()
        st.success("Saved human report snapshots (latest + versioned).")
        st.caption(f"Folder: {paths.get('reports_dir', 'N/A')}")

    st.download_button(
        "Download HTML Report",
        html_report,
        file_name=f"executive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
    )
    st.info("PDF export: use browser Print -> Save as PDF on the downloaded HTML report.")
    if ROLE_PERMISSIONS[role]["can_download"]:
        st.markdown("### Report Preview")
        st.components.v1.html(html_report, height=480, scrolling=True)

    # ── Quantos AI Insights ───────────────────────────────────────────────────
    try:
        render_inline_ai_section(
            topic="Executive Report — strategy performance narrative, risk summary, deployment readiness",
            snapshot={"role": role, "report_generated": bool(html_report)},
            key_suffix="reports_tab",
        )
    except Exception:
        pass
