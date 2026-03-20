from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from UI.constants import GOLD_DIR, LOGS_DIR, OUTPUT_DIR, PROJECT_ROOT, RAW_DIR, PROCESSED_DIR
from UI.content import ANALYSIS_HELP, LAYER_HELP
from UI.helpers import load_session_history, read_json
from UI.rendering import FETCHERS_MOD, MAIN_MOD, MEDALLION_MOD, render_logger_message


def show_data_tab() -> None:
    st.subheader("🗂️ Data Lake Explorer")
    with st.expander("Help: How data flows in this app", expanded=False):
        if FETCHERS_MOD is not None:
            render_logger_message("Fetcher guide", getattr(FETCHERS_MOD, "FETCHER_USER_GUIDE", ""))
        if MEDALLION_MOD is not None:
            render_logger_message("Medallion guide", getattr(MEDALLION_MOD, "MEDALLION_USER_GUIDE", ""))

    selected_layer = st.selectbox(
        "Choose data layer to explore:",
        options=["raw", "processed", "gold"],
        format_func=lambda k: f"{LAYER_HELP[k]['icon']}  {LAYER_HELP[k]['title']}",
        index=1,
    )
    info = LAYER_HELP[selected_layer]
    layer_paths = {"raw": RAW_DIR, "processed": PROCESSED_DIR, "gold": GOLD_DIR}

    st.info(f"**{info['what']}**")
    with st.expander("What does this layer contain?", expanded=False):
        for item in info["contains"]:
            st.markdown(f"- {item}")
        st.caption(f"{info['note']}")

    layer_path = layer_paths[selected_layer]
    parquet_files = sorted(layer_path.glob("**/*.parquet"))
    if not parquet_files:
        st.warning(f"No files found in the {selected_layer} layer yet. Run the pipeline first.")
        return

    file_labels = {str(p.relative_to(PROJECT_ROOT)): p for p in parquet_files}
    selected_label = st.selectbox("Choose file to preview:", options=list(file_labels.keys()))
    selected_file = file_labels[selected_label]
    with st.spinner("Loading data..."):
        df = pd.read_parquet(selected_file)
    st.metric("Data Shape", f"{len(df)} rows × {len(df.columns)} columns")
    st.markdown("**Preview (first 200 rows):**")
    st.dataframe(df.head(200), width="stretch")


def show_analytics_tab() -> None:
    st.subheader("📈 Analytics & Analysis Results")
    with st.expander("Help: How to interpret analysis output", expanded=False):
        if MAIN_MOD is not None:
            render_logger_message("Analysis summary", getattr(MAIN_MOD, "MAIN_OUTPUT_EXPLANATION", ""))

    corr_path = OUTPUT_DIR / "correlation_matrix.csv"
    summary_path = OUTPUT_DIR / "analysis_results.json"
    st.markdown("### 🔗 Correlation Matrix")
    corr_help = ANALYSIS_HELP.get("correlation_matrix", {})
    if corr_help:
        with st.expander("What is the Correlation Matrix?", expanded=False):
            st.markdown(f"**What it does:** {corr_help['what']}")
            st.markdown(f"**How to read it:** {corr_help['read']}")
            st.markdown(f"**How to use it:** {corr_help['use']}")

    if corr_path.exists():
        corr_df = pd.read_csv(corr_path, index_col=0)
        fig = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Correlation Matrix — Factor Relationships")
        fig.update_layout(height=640)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Correlation matrix not found. Run the pipeline first.")

    summary = read_json(summary_path)
    artifacts = summary.get("artifacts", {}) if isinstance(summary, dict) else {}
    if not artifacts:
        return
    st.markdown("---")
    st.markdown("### 📊 Analysis Result Files")
    for analysis_name, file_path in artifacts.items():
        full_path = OUTPUT_DIR / file_path if not Path(file_path).is_absolute() else Path(file_path)
        help_entry = ANALYSIS_HELP.get(analysis_name, {})
        title = help_entry.get("title", analysis_name)
        with st.expander(f"📊 {title}", expanded=False):
            if help_entry:
                st.markdown(f"**What it does:** {help_entry['what']}")
                st.markdown(f"**How to read it:** {help_entry['read']}")
                st.markdown(f"**How to use it:** {help_entry['use']}")
            if full_path.exists():
                st.caption(f"📄 {file_path}")
                if full_path.suffix == ".json":
                    st.json(read_json(full_path))
            else:
                st.warning("File not found — run pipeline first.")


def show_governance_tab() -> None:
    st.subheader("🛡️ Data Governance & Approvals")
    gov_dir = GOLD_DIR / "governance"
    files = sorted(gov_dir.glob("governance_decision_*.json"))
    if not files:
        st.warning("No governance decisions found. Run pipeline first.")
        return
    st.metric("Total Decisions", len(files))
    selected = st.selectbox(
        "View governance decision:",
        options=list(reversed(files)),
        format_func=lambda p: f"{p.name}  —  {pd.Timestamp(p.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}",
    )
    st.json(read_json(selected))


def show_logs_tab() -> None:
    st.subheader("📜 Execution Logs & Pipeline Summary")
    history = load_session_history(limit=20)
    if not history:
        st.warning("No session logs found. Run the pipeline first.")
        return
    latest = history[-1]
    st.metric("Total Sessions Recorded", len(history))
    st.json(latest)


def show_output_tab() -> None:
    st.subheader("📦 Output Results Folder")
    if not OUTPUT_DIR.exists():
        st.warning("Output directory not found. Run pipeline first.")
        return
    output_files = sorted([f for f in OUTPUT_DIR.glob("*") if f.is_file()])
    if not output_files:
        st.warning("No output files found. Run pipeline first.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Files", len(output_files))
    col2.metric("Folder", "output/")
    col3.metric("Total Size", f"{sum(f.stat().st_size for f in output_files) / 1024:.1f} KB")
    for file_path in output_files:
        with st.expander(file_path.name, expanded=False):
            st.caption(f"Size: {file_path.stat().st_size / 1024:.1f} KB")
            if file_path.suffix == ".json":
                st.json(read_json(file_path))
            elif file_path.suffix == ".csv":
                try:
                    df = pd.read_csv(file_path)
                    st.dataframe(df.head(100), width="stretch")
                except Exception:
                    st.info("Could not parse CSV")
