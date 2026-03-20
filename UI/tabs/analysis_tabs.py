from __future__ import annotations

from datetime import datetime

import streamlit as st

from UI.constants import OUTPUT_DIR, ROLE_PERMISSIONS
from UI.helpers import (
    build_executive_report_html,
    build_explainability_lines,
    build_run_comparison,
    build_smart_alerts,
    compute_data_health,
    read_json,
)


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
        return
    for alert in alerts:
        msg = f"**{alert['title']}** — {alert['message']}"
        if alert.get("severity") == "critical":
            st.error(msg)
        else:
            st.warning(msg)


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


def show_scenario_builder_tab() -> None:
    st.subheader("🎛️ Scenario Builder (Custom Shocks)")
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

    total_shock = (inflation_shock * 0.5 + energy_shock * 0.3 + rate_shock * 0.2) / 100.0
    estimated_return_shift = baseline_move + total_shock

    st.metric("Estimated return shift", f"{estimated_return_shift * 100:.2f}%")
    st.info("This is a fast directional estimate. Use Stress Test and Forecast outputs for detailed decisions.")


def show_explainability_tab() -> None:
    st.subheader("🧠 Explainability Panel")
    st.caption("What changed and why — plain language summaries.")
    for line in build_explainability_lines():
        st.markdown(f"- {line}")


def show_reports_tab(role: str) -> None:
    st.subheader("🧾 Executive Reports (One-click Export)")
    html_report = build_executive_report_html()
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
