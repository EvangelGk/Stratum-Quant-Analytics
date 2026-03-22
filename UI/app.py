from __future__ import annotations

# Copyright (c) 2026 EvangelGK. All Rights Reserved.

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UI.constants import OUTPUT_DIR, PIPELINE_STAGES, ROLE_PERMISSIONS
from UI.rendering import DIRECTIONS_MOD, MAIN_MOD, inject_styles, render_logger_message, show_kpis
from UI.runtime import clear_all_run_history, run_and_cache_audit, run_pipeline, show_pipeline_failure
from UI.runtime import run_optimizer_background
try:
    from src.secret_store import get_secret
except ModuleNotFoundError:
    from src.secret_store import get_secret
from UI.tabs import (
    render_sidebar_ai_widget,
    show_ai_assistant_tab,
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


def _render_api_keys_status() -> None:
    """Render required API key connectivity status in the sidebar."""
    @st.cache_data(show_spinner=False, ttl=60)
    def _status_snapshot() -> tuple[list[str], list[str]]:
        fred_key = (get_secret("FRED_API_KEY") or "").strip()
        gemini_key = (get_secret("GEMINI_API_KEY") or "").strip()

        connected_local: list[str] = []
        missing_local: list[str] = []
        if fred_key:
            connected_local.append("FRED_API_KEY")
        else:
            missing_local.append("FRED_API_KEY")

        if gemini_key:
            connected_local.append("GEMINI_API_KEY")
        else:
            missing_local.append("GEMINI_API_KEY")
        return connected_local, missing_local

    connected, missing = _status_snapshot()

    st.markdown("### π” API Keys Status")
    if not missing:
        st.success("πΆ Connected: ΟΞ»Ξ± Ο„Ξ± Ξ±Ξ½Ξ±Ξ³ΞΊΞ±Ξ―Ξ± API keys ΞµΞ―Ξ½Ξ±ΞΉ Ξ΄ΞΉΞ±ΞΈΞ­ΟƒΞΉΞΌΞ±")
    else:
        st.error("π”΄ Missing: Ξ»ΞµΞ―Ο€ΞµΞΉ 1 Ξ® Ο€ΞµΟΞΉΟƒΟƒΟΟ„ΞµΟΞ± Ξ±Ξ½Ξ±Ξ³ΞΊΞ±Ξ―Ξ± API keys")

    st.caption(f"Connected ({len(connected)}/2): {', '.join(connected) if connected else 'none'}")
    if missing:
        st.caption(f"Missing: {', '.join(missing)}")


def _render_sidebar() -> str:
    with st.sidebar:
        role = st.selectbox("Role", options=["Viewer", "Analyst", "Admin"], index=1)
        perms = ROLE_PERMISSIONS[role]

        _render_api_keys_status()
        st.markdown("---")

        with st.expander("Role meaning", expanded=False):
            st.markdown("- Viewer: read-only access to dashboards and artifacts.")
            st.markdown("- Analyst: can run pipelines and export reports, but cannot schedule jobs.")
            st.markdown("- Admin: full operational control (run, export, schedule, history deletion).")

        st.caption(
            f"Permissions: run={perms['can_run']}, download={perms['can_download']}, schedule={perms['can_schedule']}"
        )

        st.header(" Pipeline Execution")

        with st.expander("π’¬ Live guidance from logger messages", expanded=False):
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
            "π€ Run Full Analysis",
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
        st.markdown("### π§Ή History Control")
        can_delete = role == "Admin"
        confirm = st.checkbox(
            "I understand this deletes generated runs, outputs, user data, and session logs",
            key="confirm_delete_history_sidebar",
            disabled=not can_delete,
        )
        if st.button(
            "π—‘οΈ Delete All Run History",
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

        # ----------------------------------------------------------------
        # AI Copilot β€” always-present mini-chat in the sidebar.
        # Status is checked once per session; heavy work only on user action.
        # ----------------------------------------------------------------
        st.markdown("---")
        render_sidebar_ai_widget()

        # ----------------------------------------------------------------
        # Owner-only: Automated Optimizer
        # Only visible when OPTIMIZER_OWNER_MODE=1 is set in the environment.
        # Never surfaced to regular users or shown in public deployments.
        # ----------------------------------------------------------------
        import os as _os
        if _os.getenv("OPTIMIZER_OWNER_MODE", "").strip() == "1":
            st.markdown("---")
            st.markdown("### π”¬ Automated Optimizer")
            st.caption(
                "Owner-only: runs the self-correcting 10-iteration optimization loop. "
                "Each code mutation requires your approval via terminal prompt or "
                "approval queue file."
            )
            opt_target = st.slider(
                "Target score", min_value=80, max_value=99, value=94, step=1,
                key="opt_target_score",
            )
            opt_iters = st.number_input(
                "Max iterations", min_value=1, max_value=20, value=10, step=1,
                key="opt_max_iters",
            )

            # Show last optimizer report if available
            _opt_report = OUTPUT_DIR / "optimizer_report.json"
            if _opt_report.exists():
                with st.expander("Last optimizer report", expanded=False):
                    try:
                        import json as _json
                        _rpt = _json.loads(_opt_report.read_text(encoding="utf-8"))
                        st.json(_rpt.get("final_quant_evaluation", _rpt))
                    except Exception:
                        st.caption("Could not parse optimizer report.")

            if st.button(
                "π“ Run Optimizer",
                key="run_optimizer_btn",
                type="secondary",
                width="stretch",
            ):
                opt_prog = st.progress(0, text="Starting optimizer...")
                ok, opt_out = run_optimizer_background(
                    target_score=float(opt_target),
                    max_iterations=int(opt_iters),
                    progress_bar=opt_prog,
                )
                opt_prog.empty()
                if ok:
                    st.success("Optimizer completed. Check Output tab for optimizer_report.json.")
                else:
                    st.error("Optimizer failed.")
                    with st.expander("Optimizer output", expanded=False):
                        st.text(opt_out[-3000:] if len(opt_out) > 3000 else opt_out)

    return role


def main() -> None:
    st.set_page_config(
        page_title=" Welcome to STRATUM QUANT ANALYTICS!",
        page_icon="π",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>STRATUM QUANT ANALYTICS User Platform</h1>
            <p>Pipeline orchestration, audit controls, metrics,
            analytics and many more to discover. All in one place! </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    role = _render_sidebar()
    show_kpis()

    pages = [
        "π¤– Quantos Assistant",
        "π©Ί Health & Alerts",
        "π§ Auditor",
        "π“ Run Comparison",
        "π›οΈ Scenario Builder",
        "π§  Explainability",
        "π§Ύ Reports",
        "β™οΈ Ops",
        "π—‚οΈ Data",
        "π“ Analytics",
        "π“¦ Output",
        "π›΅οΈ Governance",
        "π“ Logs",
    ]
    selected_page = st.segmented_control("View", options=pages, default="π©Ί Health & Alerts")
    # Track active page in session state so sidebar AI and chips know context
    if selected_page:
        st.session_state["selected_page"] = selected_page

    if selected_page == "π¤– Quantos Assistant":
        show_ai_assistant_tab()
    elif selected_page == "π©Ί Health & Alerts":
        show_health_alerts_tab()
    elif selected_page == "π§ Auditor":
        show_auditor_tab()
    elif selected_page == "π“ Run Comparison":
        show_run_comparison_tab()
    elif selected_page == "π›οΈ Scenario Builder":
        show_scenario_builder_tab()
    elif selected_page == "π§  Explainability":
        show_explainability_tab()
    elif selected_page == "π§Ύ Reports":
        show_reports_tab(role)
    elif selected_page == "β™οΈ Ops":
        show_ops_tab(role)
    elif selected_page == "π—‚οΈ Data":
        show_data_tab()
    elif selected_page == "π“ Analytics":
        show_analytics_tab()
    elif selected_page == "π“¦ Output":
        show_output_tab()
    elif selected_page == "π›΅οΈ Governance":
        show_governance_tab()
    elif selected_page == "π“ Logs":
        show_logs_tab()

    # Render footer with license attribution
    _render_footer()


def _render_footer() -> None:
    """Render fixed bottom-right copyright badge."""
    st.markdown(
        """
        <style>
        .copyright-fixed {
            position: fixed;
            right: 16px;
            bottom: 10px;
            z-index: 9999;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(0, 0, 0, 0.12);
            border-radius: 10px;
            padding: 6px 10px;
            font-size: 12px;
            color: #111111;
            backdrop-filter: blur(2px);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
        }
        </style>
        <div class="copyright-fixed">
            CC BY-NC-ND 4.0 Β© 2026 EvangelGK. All Rights Reserved
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

