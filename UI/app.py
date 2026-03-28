from __future__ import annotations

# Copyright (c) 2026 EvangelGK. All Rights Reserved.

import hmac
import sys
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UI.constants import OUTPUT_DIR, PIPELINE_STAGES, ROLE_PERMISSIONS
from UI.helpers import clear_file_caches, compute_artifact_signature
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
    show_edge_arsenal_tab,
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
    fred_key = (get_secret("FRED_API_KEY") or "").strip()
    gemini_key = (get_secret("GEMINI_API_KEY") or "").strip()
    groq_key = (get_secret("GROQ_API_KEY") or "").strip()

    connected: list[str] = []
    missing: list[str] = []
    if fred_key:
        connected.append("FRED_API_KEY")
    else:
        missing.append("FRED_API_KEY")

    if gemini_key:
        connected.append("GEMINI_API_KEY")
    else:
        missing.append("GEMINI_API_KEY")

    if groq_key:
        connected.append("GROQ_API_KEY")
    else:
        missing.append("GROQ_API_KEY")

    # AI is reachable if at least one LLM key (Gemini or Groq) is configured
    llm_keys = {"GEMINI_API_KEY", "GROQ_API_KEY"}
    llm_ok = bool(llm_keys & set(connected))
    total_required = 3  # FRED + GEMINI + GROQ

    st.markdown("### 🔐 API Keys Status")
    if not missing:
        st.success("🟢 Connected: όλα τα αναγκαία API keys είναι διαθέσιμα")
    elif llm_ok:
        st.warning("🟡 Partial: τουλάχιστον ένα LLM key διαθέσιμο")
    else:
        st.error("🔴 Missing: λείπει 1 ή περισσότερα αναγκαία API keys")

    st.caption(f"Connected ({len(connected)}/{total_required}): {', '.join(connected) if connected else 'none'}")
    if missing:
        st.caption(f"Missing: {', '.join(missing)}")


def _approval_queue_path() -> Path:
    return OUTPUT_DIR / ".optimizer" / "approval_queue.json"


def _read_approval_queue() -> dict | None:
    path = _approval_queue_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_approval_queue(data: dict) -> bool:
    path = _approval_queue_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return True
    except Exception:
        return False


def _render_optimizer_approval_panel() -> None:
    queue = _read_approval_queue()
    if not queue:
        st.caption("No pending optimizer approval request detected.")
        return

    status = str(queue.get("status", "unknown"))
    if status == "pending":
        st.warning("Pending optimizer approval detected.")
    elif status == "YES":
        st.success("Last optimizer proposal was approved.")
    elif status == "NO":
        st.info("Last optimizer proposal was rejected or timed out.")
    else:
        st.caption(f"Approval queue status: {status}")

    with st.expander("Approval queue", expanded=(status == "pending")):
        st.caption(f"Action: {queue.get('description', 'N/A')}")
        st.caption(f"Requested: {queue.get('requested_at', 'N/A')}")
        details = queue.get("details", {})
        if isinstance(details, dict):
            st.json(details)

        if status == "pending":
            col_approve, col_reject = st.columns(2)
            if col_approve.button("Approve", key="approve_optimizer_change", use_container_width=True):
                queue["status"] = "YES"
                queue["approved_at"] = datetime.utcnow().isoformat() + "Z"
                if _write_approval_queue(queue):
                    st.success("Approval recorded. Optimizer will detect it within ~2 seconds.")
                    st.rerun()
                else:
                    st.error("Could not write approval queue file.")
            if col_reject.button("Reject", key="reject_optimizer_change", use_container_width=True):
                queue["status"] = "NO"
                queue["approved_at"] = datetime.utcnow().isoformat() + "Z"
                if _write_approval_queue(queue):
                    st.info("Rejection recorded. Optimizer will detect it within ~2 seconds.")
                    st.rerun()
                else:
                    st.error("Could not write approval queue file.")


def _render_pre_optimizer_lag_heatmap() -> None:
    """Render lag-correlation heatmap (1-20 days) before optimizer execution."""
    master_path = PROJECT_ROOT / "data" / "gold" / "master_table.parquet"
    if not master_path.exists():
        st.caption("No Gold master table found yet. Run Full Analysis first.")
        return

    try:
        import pandas as pd
        import plotly.express as px

        df = pd.read_parquet(master_path)
    except Exception as exc:
        st.caption(f"Could not load master table for lag heatmap: {exc}")
        return

    if "date" not in df.columns or "close" not in df.columns:
        st.caption("Master table missing required date/close columns for lag heatmap.")
        return

    tickers = []
    if "ticker" in df.columns:
        tickers = sorted(df["ticker"].dropna().astype(str).unique().tolist())
    selected_ticker = st.selectbox(
        "Ticker for lag heatmap",
        options=tickers if tickers else ["ALL"],
        key="opt_lag_heatmap_ticker",
    )

    if "ticker" in df.columns and selected_ticker != "ALL":
        df = df[df["ticker"].astype(str) == str(selected_ticker)].copy()

    candidate_metrics = [
        c
        for c in df.columns
        if c
        not in {
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "log_return",
        }
        and not str(c).startswith("__")
    ]
    if not candidate_metrics:
        st.caption("No macro metrics available in Gold table for lag heatmap.")
        return

    metric = st.selectbox(
        "FRED metric",
        options=sorted(candidate_metrics),
        key="opt_lag_heatmap_metric",
    )

    work = df[["date", "close", metric]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=["close", metric])
    if len(work) < 40:
        st.caption("Not enough aligned rows to estimate lag correlations (need >= 40).")
        return

    lags = list(range(1, 21))
    corr_vals: list[float] = []
    for lag in lags:
        shifted = work[metric].shift(lag)
        valid = work[["close"]].copy()
        valid["macro_lag"] = shifted
        valid = valid.dropna(subset=["close", "macro_lag"])
        if len(valid) < 20:
            corr_vals.append(float("nan"))
            continue
        corr_vals.append(float(valid["close"].corr(valid["macro_lag"])))

    hdf = pd.DataFrame(
        {
            "Lag": [f"L{lag}" for lag in lags],
            "Correlation": corr_vals,
        }
    )
    fig = px.imshow(
        [corr_vals],
        labels={"x": "Lag (days)", "y": "Metric", "color": "Correlation"},
        x=lags,
        y=[metric],
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Pre-Optimizer Correlation Heatmap: FRED metric vs stock close by lag (1-20)",
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(hdf, width="stretch", hide_index=True)


def _sync_artifact_freshness() -> None:
    current_signature = compute_artifact_signature()
    previous_signature = st.session_state.get("_artifact_signature")

    if previous_signature is None:
        st.session_state["_artifact_signature"] = current_signature
        return

    if current_signature != previous_signature:
        st.session_state["_artifact_signature"] = current_signature
        st.session_state["_artifacts_updated_message"] = (
            "New artifacts detected. Refreshing tabs with latest outputs."
        )
        clear_file_caches()
        st.cache_data.clear()
        st.rerun()




def _check_admin_pin(entered: str) -> bool:
    """Constant-time comparison of entered PIN against the stored secret.

    Returns True only when ADMIN_PIN secret is set, is exactly 6 digits,
    and the entered value matches.
    """
    stored = (get_secret("ADMIN_PIN") or "").strip()
    if not stored or not stored.isdigit() or len(stored) != 6:
        return False
    return hmac.compare_digest(stored.encode(), entered.strip().encode())


def _render_sidebar() -> str:
    with st.sidebar:
        # ── Role selector ────────────────────────────────────────────────
        role_selection = st.selectbox("Role", options=["Viewer", "Analyst", "Admin"], index=1)

        # ── Admin PIN gate ───────────────────────────────────────────────
        if "admin_authenticated" not in st.session_state:
            st.session_state.admin_authenticated = False

        if role_selection == "Admin":
            if not st.session_state.admin_authenticated:
                st.markdown("#### 🔒 Admin Authentication")
                with st.form(key="admin_pin_form", clear_on_submit=True):
                    pin_input = st.text_input(
                        "Enter 6-digit Admin PIN",
                        type="password",
                        max_chars=6,
                        placeholder="••••••",
                    )
                    submitted = st.form_submit_button("Unlock Admin", use_container_width=True)
                if submitted:
                    if pin_input.isdigit() and len(pin_input) == 6 and _check_admin_pin(pin_input):
                        st.session_state.admin_authenticated = True
                        st.success("Admin access granted.")
                        st.rerun()
                    else:
                        st.error("Incorrect PIN. Admin access denied.")
                # While not yet authenticated, fall back to Analyst perms
                role = "Analyst"
            else:
                role = "Admin"
                if st.button("🔓 Revoke Admin Access", key="admin_signout", use_container_width=True):
                    st.session_state.admin_authenticated = False
                    st.rerun()
        else:
            # If user switches away from Admin, clear the authenticated flag
            if st.session_state.admin_authenticated:
                st.session_state.admin_authenticated = False
            role = role_selection

        perms = ROLE_PERMISSIONS[role]

        _render_api_keys_status()
        st.markdown("---")

        with st.expander("Role meaning", expanded=False):
            st.markdown("- Viewer: read-only access to dashboards and artifacts.")
            st.markdown("- Analyst: can run pipelines and export reports, but cannot schedule jobs.")
            st.markdown("- Admin: full operational control (run, export, schedule, history deletion). Requires PIN.")

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

        # ----------------------------------------------------------------
        # AI Copilot — always-present mini-chat in the sidebar.
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
            st.markdown("### 🔬 Automated Optimizer")
            st.caption(
                "Owner-only: runs the self-correcting 10-iteration optimization loop. "
                "Each code mutation requires your approval via terminal prompt, CLI, "
                "or the approval queue panel below."
            )
            _render_optimizer_approval_panel()
            with st.expander("Pre-Optimizer Lag Correlation Heatmap (1-20)", expanded=True):
                _render_pre_optimizer_lag_heatmap()
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
                "📈 Run Optimizer",
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
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    if hasattr(st, "fragment"):
        @st.fragment(run_every="6s")
        def _artifact_watchdog() -> None:
            _sync_artifact_freshness()
    else:
        def _artifact_watchdog() -> None:
            components.html(
                """
                <script>
                setTimeout(function () {
                    window.parent.location.reload();
                }, 6000);
                </script>
                """,
                height=0,
            )

    _artifact_watchdog()
    if st.session_state.get("_artifacts_updated_message"):
        st.info(st.session_state.pop("_artifacts_updated_message"))

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

    # Load latest backtest artifact into session state once per rerun so that
    # Pipeline → Auditor → Edge Arsenal share the same payload snapshot.
    _load_backtest_payload_to_session()

    pages = [
        "🤖 Quantos Assistant",
        "💎 Edge Arsenal",
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
    # Use session-state as default so st.rerun() (e.g. from Rerun Audit button)
    # preserves the current tab instead of jumping to the hard-coded default.
    _default_page = st.session_state.get("selected_page", "💎 Edge Arsenal")
    if _default_page not in pages:
        _default_page = "💎 Edge Arsenal"
    selected_page = st.segmented_control(
        "View", options=pages, default=_default_page, key="main_page_control"
    )
    # Track active page in session state so sidebar AI and chips know context
    if selected_page:
        st.session_state["selected_page"] = selected_page

    if selected_page == "🤖 Quantos Assistant":
        show_ai_assistant_tab()
    elif selected_page == "💎 Edge Arsenal":
        show_edge_arsenal_tab()
    elif selected_page == "🩺 Health & Alerts":
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

    # Render footer with license attribution
    _render_footer()


# ──────────────────────────────────────────────────────────────────────────────
# Pillar 3 — Session State: Backtest Payload Handshake
# ──────────────────────────────────────────────────────────────────────────────
# How secrets work in this app
# ─────────────────────────────
# All API keys (FRED_API_KEY, GEMINI_API_KEY, GROQ_API_KEY) are resolved by
# `get_secret()` in secret_store.py, which checks (in priority order):
#   1. st.secrets  — Streamlit Cloud dashboard (Settings → Secrets)
#   2. os.environ  — injected by Docker / GitHub Actions
#   3. .streamlit/secrets.toml — local dev file
#   4. .env        — fallback dotenv file
#
# Streamlit Cloud deployment:
#   Go to App → Settings → Secrets and add each key as:
#       FRED_API_KEY = "your-key"
#       GEMINI_API_KEY = "your-key"
#       GROQ_API_KEY  = "your-key"
#   New users sharing the deployment automatically inherit these secrets.
#   Never commit real keys to .streamlit/secrets.toml in a public repo.


def _load_backtest_payload_to_session() -> None:
    """
    Read the latest pipeline output artifact and push the backtest payload into
    ``st.session_state["backtest_payload"]``.

    This creates a **single source of truth** shared by the Pipeline, Auditor,
    and Edge Arsenal modules — they all read from session_state rather than each
    independently hitting disk, which guarantees consistency within a Streamlit
    rerun cycle.

    The load is **skipped** when:
    * The artifact file has not changed since the last load (tracked via a
      content-hash stored in ``st.session_state["_backtest_payload_hash"]``).
    * The artifact file does not exist yet.

    Called once per ``main()`` invocation before any tabs are rendered.
    """
    try:
        artifact_path = OUTPUT_DIR / "default" / "analysis_results.json"
        # Respect DATA_USER_ID when set
        user_id = (
            st.secrets.get("DATA_USER_ID", "")
            or ""
        ).strip()
        if user_id:
            safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in user_id)
            candidate = OUTPUT_DIR / safe / "analysis_results.json"
            if candidate.exists():
                artifact_path = candidate

        if not artifact_path.exists():
            return

        import hashlib as _hashlib
        raw = artifact_path.read_bytes()
        file_hash = _hashlib.md5(raw, usedforsecurity=False).hexdigest()

        # Skip re-parse if the file hasn't changed since last load
        if st.session_state.get("_backtest_payload_hash") == file_hash:
            return

        payload = json.loads(raw.decode("utf-8", errors="replace"))
        st.session_state["backtest_payload"] = payload
        st.session_state["_backtest_payload_hash"] = file_hash
        st.session_state["backtest_payload_loaded_at"] = datetime.utcnow().isoformat() + "Z"
    except Exception:
        # Non-fatal: tabs handle a missing / empty backtest_payload gracefully.
        pass


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
            CC BY-NC-ND 4.0 © 2026 EvangelGK. All Rights Reserved
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()


