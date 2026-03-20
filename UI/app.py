from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UI.constants import PIPELINE_STAGES, ROLE_PERMISSIONS
from UI.rendering import DIRECTIONS_MOD, MAIN_MOD, inject_styles, render_logger_message, show_kpis
from UI.runtime import clear_all_run_history, run_and_cache_audit, run_pipeline, show_pipeline_failure
from UI.tabs import (
    show_analytics_tab,
    show_auditor_tab,
    show_data_tab,
    show_explainability_tab,
    show_governance_tab,
    show_health_alerts_tab,
    show_logs_tab,
    show_ops_tab,
    show_output_tab,
    show_reports_tab,
    show_run_comparison_tab,
    show_scenario_builder_tab,
)


def _render_sidebar() -> str:
    with st.sidebar:
        role = st.selectbox("Role", options=["Viewer", "Analyst", "Admin"], index=1)
        perms = ROLE_PERMISSIONS[role]

        with st.expander("Role meaning", expanded=False):
            st.markdown("- Viewer: read-only access to dashboards and artifacts.")
            st.markdown("- Analyst: can run pipelines and export reports, but cannot schedule jobs.")
            st.markdown("- Admin: full operational control (run, export, schedule, history deletion).")

        st.caption(
            f"Permissions: run={perms['can_run']}, download={perms['can_download']}, schedule={perms['can_schedule']}"
        )

        st.header(" Pipeline Execution")

        with st.expander("💬 Live guidance from logger messages", expanded=False):
            if DIRECTIONS_MOD is not None:
                render_logger_message(
                    "Welcome",
                    getattr(DIRECTIONS_MOD, "LIVE_STEP_0_WELCOME", ""),
                )
                render_logger_message(
                    "Quick start",
                    getattr(MAIN_MOD, "QUICK_START", "") if MAIN_MOD else "",
                )

        with st.expander("What does the pipeline do?", expanded=False):
            for stage, desc in PIPELINE_STAGES:
                st.markdown(f"**{stage}** - {desc}")

        resume_mode = st.checkbox(
            "Fast rerun mode (reuse completed stages)",
            value=True,
            help=(
                "When enabled, pipeline can skip already completed Bronze/Silver/Gold "
                "stages using checkpoints. Disable for a strict full rebuild."
            ),
        )

        if st.button(
            "🚀 Run Full Analysis",
            width="stretch",
            type="primary",
            disabled=not perms["can_run"],
        ):
            st.caption("Running pipeline in FULL mode...")
            prog = st.progress(0, text="Starting...")
            ok, output = run_pipeline(
                mode="actual",
                progress_bar=prog,
                resume_from_checkpoint=bool(resume_mode),
            )
            prog.empty()
            if ok:
                st.success("Full analysis completed successfully.")
                with st.spinner("Running system audit..."):
                    run_and_cache_audit()
                st.info("Check Output and Auditor tabs for results.")
            else:
                show_pipeline_failure(output)

        st.info("Full Analysis: complete dataset and all analyses. Typical runtime: 4-10 minutes (usually around 6).")

        st.markdown("---")
        st.markdown("### 🧹 History Control")
        can_delete = role == "Admin"
        confirm = st.checkbox(
            "I understand this deletes generated runs, outputs, user data, and session logs",
            key="confirm_delete_history_sidebar",
            disabled=not can_delete,
        )
        if st.button(
            "🗑️ Delete All Run History",
            type="secondary",
            width="stretch",
            disabled=not can_delete or not confirm,
        ):
            result = clear_all_run_history()
            if result.get("skipped_locked"):
                st.warning(result["message"])
                with st.expander("Locked files (close related apps/processes and retry)", expanded=False):
                    for p in result["skipped_locked"]:
                        st.markdown(f"- {p}")
            else:
                st.success(result["message"])
            st.caption(
                f"Deleted files: {result['deleted_files']} | Deleted directories: {result['deleted_dirs']}"
            )
            st.rerun()

    return role


def main() -> None:
    st.set_page_config(
        page_title=" Welcome to Scenario Planner!",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Scenario Planner User Platform</h1>
            <p>Pipeline orchestration, audit controls, metrics,
            analytics and many more to discover. All in one place! </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    role = _render_sidebar()
    show_kpis()

    pages = [
        "🩺 Health & Alerts",
        "🧪 Auditor",
        "📊 Run Comparison",
        "🎛️ Scenario Builder",
        "🧠 Explainability",
        "🧾 Reports",
        "⚙️ Ops",
        "🗂️ Data",
        "📈 Analytics",
        "📦 Output",
        "🛡️ Governance",
        "📜 Logs",
    ]
    selected_page = st.segmented_control("View", options=pages, default="🩺 Health & Alerts")

    if selected_page == "🩺 Health & Alerts":
        show_health_alerts_tab()
    elif selected_page == "🧪 Auditor":
        show_auditor_tab()
    elif selected_page == "📊 Run Comparison":
        show_run_comparison_tab()
    elif selected_page == "🎛️ Scenario Builder":
        show_scenario_builder_tab()
    elif selected_page == "🧠 Explainability":
        show_explainability_tab()
    elif selected_page == "🧾 Reports":
        show_reports_tab(role)
    elif selected_page == "⚙️ Ops":
        show_ops_tab(role)
    elif selected_page == "🗂️ Data":
        show_data_tab()
    elif selected_page == "📈 Analytics":
        show_analytics_tab()
    elif selected_page == "📦 Output":
        show_output_tab()
    elif selected_page == "🛡️ Governance":
        show_governance_tab()
    elif selected_page == "📜 Logs":
        show_logs_tab()


if __name__ == "__main__":
    main()
