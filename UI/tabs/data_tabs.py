from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from UI.constants import GOLD_DIR, LOGS_DIR, OUTPUT_DIR, PROJECT_ROOT, RAW_DIR, PROCESSED_DIR
from UI.content import ANALYSIS_HELP, LAYER_HELP
from UI.helpers import load_session_history, read_json
from UI.rendering import FETCHERS_MOD, MAIN_MOD, MEDALLION_MOD, render_logger_message


def _human_label(text: str) -> str:
    return str(text).replace("_", " ").strip().title()


def _fmt_scalar(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, bool)):
        return str(value)
    if value is None:
        return "N/A"
    return str(value)


def _flatten_scalars(payload: object, prefix: str = "", max_items: int = 60) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    def walk(node: object, path: str) -> None:
        if len(rows) >= max_items:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                next_path = f"{path}.{k}" if path else str(k)
                walk(v, next_path)
                if len(rows) >= max_items:
                    return
        elif isinstance(node, list):
            if all(not isinstance(x, (dict, list)) for x in node):
                rows.append((path or "value", ", ".join(_fmt_scalar(x) for x in node[:10])))
            else:
                rows.append((path or "value", f"list[{len(node)}]"))
        else:
            rows.append((path or "value", _fmt_scalar(node)))

    walk(payload, prefix)
    return rows


def _render_key_values(payload: object, header: str = "Summary") -> None:
    rows = _flatten_scalars(payload)
    if not rows:
        st.info("No summary fields available.")
        return
    st.markdown(f"**{header}**")
    st.dataframe(
        pd.DataFrame([{"Field": _human_label(k), "Value": v} for k, v in rows]),
        width="stretch",
        hide_index=True,
    )


def _render_governance_payload(payload: dict) -> None:
    gate = payload.get("gate", payload.get("value", payload))
    report = payload.get("report", {})
    if isinstance(gate, dict):
        c1, c2, c3 = st.columns(3)
        c1.metric("Gate Passed", "Yes" if gate.get("passed") else "No")
        c2.metric("Severity", str(gate.get("severity", "unknown")).upper())
        c3.metric("Regime", str(gate.get("regime", "unknown")).upper())
        reasons = gate.get("reasons", []) or []
        if reasons:
            st.markdown("**Gate reasons:**")
            for reason in reasons:
                st.markdown(f"- {reason}")

    if isinstance(report, dict):
        oos = (report.get("out_of_sample") or {}).get("r2")
        wf = (report.get("walk_forward") or {}).get("avg_r2")
        risk = report.get("model_risk_score")
        m1, m2, m3 = st.columns(3)
        m1.metric("OOS R²", _fmt_scalar(oos))
        m2.metric("Walk-forward R²", _fmt_scalar(wf))
        m3.metric("Model Risk", _fmt_scalar(risk))


def _render_analysis_payload(analysis_name: str, payload: object) -> None:
    value = payload.get("value") if isinstance(payload, dict) and "value" in payload else payload

    if isinstance(value, str):
        if value.startswith("blocked_by_governance_gate"):
            st.warning("This analysis is blocked by governance gate for this run.")
            st.caption(value)
        else:
            st.info(value)
        return

    if isinstance(value, (int, float, bool)):
        st.metric("Result", _fmt_scalar(value))
        return

    if analysis_name in {"governance_gate", "governance_report"} and isinstance(value, dict):
        wrapper = {"gate": value} if analysis_name == "governance_gate" else {"report": value}
        _render_governance_payload(wrapper)
        return

    if analysis_name == "correlation_matrix" and isinstance(value, dict):
        c1, c2 = st.columns(2)
        shape = value.get("shape", [0, 0])
        c1.metric("Rows", shape[0] if isinstance(shape, list) and len(shape) == 2 else "N/A")
        c2.metric("Columns", shape[1] if isinstance(shape, list) and len(shape) == 2 else "N/A")
        cols = value.get("columns", [])
        if isinstance(cols, list) and cols:
            st.caption("Available fields")
            st.write(", ".join(str(x) for x in cols[:20]) + ("..." if len(cols) > 20 else ""))
        return

    if isinstance(value, dict):
        _render_key_values(value)
        return

    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            st.dataframe(pd.DataFrame(value), width="stretch", hide_index=True)
        else:
            for item in value[:20]:
                st.markdown(f"- {_fmt_scalar(item)}")
        return

    st.info("No readable analysis output available for this artifact.")


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
                    payload = read_json(full_path)
                    _render_analysis_payload(analysis_name, payload)
                    st.download_button(
                        "Download JSON archive",
                        json.dumps(payload, indent=2, ensure_ascii=False),
                        file_name=full_path.name,
                        mime="application/json",
                        key=f"download_{analysis_name}",
                    )
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
    payload = read_json(selected)
    _render_governance_payload(payload if isinstance(payload, dict) else {"gate": {}})
    _render_key_values(payload, header="Governance details")
    st.download_button(
        "Download governance JSON",
        json.dumps(payload, indent=2, ensure_ascii=False),
        file_name=selected.name,
        mime="application/json",
        key="download_governance_json",
    )


def show_logs_tab() -> None:
    st.subheader("📜 Execution Logs & Pipeline Summary")
    history = load_session_history(limit=20)
    if not history:
        st.warning("No session logs found. Run the pipeline first.")
        return
    latest = history[-1]
    st.metric("Total Sessions Recorded", len(history))
    info = latest.get("session_info", {}) if isinstance(latest, dict) else {}
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Session ID", str(info.get("session_id", "N/A"))[:12])
    p2.metric("Duration (s)", _fmt_scalar(info.get("total_duration_seconds")))
    p3.metric("Operations", _fmt_scalar(info.get("total_operations")))
    p4.metric("Run ID", str(info.get("run_id", "N/A"))[:12])

    timeline = latest.get("operations_timeline", []) if isinstance(latest, dict) else []
    if isinstance(timeline, list) and timeline:
        rows = []
        for event in timeline[-20:]:
            if isinstance(event, dict):
                rows.append(
                    {
                        "Time": str(event.get("timestamp", ""))[11:19],
                        "Operation": event.get("operation", ""),
                        "Component": event.get("component", ""),
                    }
                )
        if rows:
            st.markdown("**Recent timeline events**")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.download_button(
        "Download latest session JSON",
        json.dumps(latest, indent=2, ensure_ascii=False),
        file_name="latest_session_summary.json",
        mime="application/json",
        key="download_latest_session_json",
    )


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
                payload = read_json(file_path)
                _render_analysis_payload(file_path.stem, payload)
                st.download_button(
                    "Download JSON archive",
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    file_name=file_path.name,
                    mime="application/json",
                    key=f"download_output_{file_path.name}",
                )
            elif file_path.suffix == ".csv":
                try:
                    df = pd.read_csv(file_path)
                    st.dataframe(df.head(100), width="stretch")
                except Exception:
                    st.info("Could not parse CSV")
