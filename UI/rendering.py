from __future__ import annotations

import importlib
from typing import Any

import streamlit as st

from UI.constants import GOLD_DIR, OUTPUT_DIR, PROCESSED_DIR, RAW_DIR
from UI.content import PIPELINE_STAGES
from UI.helpers import count_files


def import_first(*module_names: str) -> Any:
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    return None


DIRECTIONS_MOD = import_first(
    "logger.Messages.DirectionsMess", "src.logger.Messages.DirectionsMess"
)
MAIN_MOD = import_first("logger.Messages.MainMess", "src.logger.Messages.MainMess")
MEDALLION_MOD = import_first(
    "logger.Messages.MedallionMess", "src.logger.Messages.MedallionMess"
)
FETCHERS_MOD = import_first(
    "logger.Messages.FetchersMess", "src.logger.Messages.FetchersMess"
)


def normalize_message(msg: Any) -> str:
    if not isinstance(msg, str):
        return ""
    cleaned = msg.strip()
    return cleaned.replace("\r\n", "\n") if cleaned else ""


def render_logger_message(title: str, msg: Any) -> None:
    text = normalize_message(msg)
    if not text:
        return
    st.markdown(f"**{title}**")
    st.caption(text)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
            html, body, [class*="css"] {
                font-family: 'Space Grotesk', sans-serif;
            }
            .hero {
                background: radial-gradient(circle at top left, rgba(17, 94, 89, 0.55), transparent 40%),
                            linear-gradient(135deg, #0f172a 0%, #111827 52%, #0b6e4f 100%);
                padding: 1.25rem 1.4rem;
                border-radius: 22px;
                color: #f8fafc;
                margin-bottom: 1rem;
                border: 1px solid rgba(255, 255, 255, 0.10);
                box-shadow: 0 18px 40px rgba(2, 6, 23, 0.18);
            }
            .hero h1 {
                margin: 0;
                font-size: 1.95rem;
                letter-spacing: 0.2px;
            }
            .hero p {
                margin: 0.45rem 0 0;
                color: #dbe4f0;
                font-size: 0.96rem;
            }
            [data-testid="stSidebar"] .stButton > button {
                width: 100%;
                min-height: 46px;
                border-radius: 999px;
                border: 1px solid rgba(15, 23, 42, 0.08);
                background: linear-gradient(135deg, #ffffff 0%, #f8fbfb 100%);
                color: #0f172a;
                font-weight: 700;
                letter-spacing: 0.01em;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.07);
                transition: all 0.18s ease;
                white-space: nowrap;
                padding: 0.5rem 1rem;
            }
            [data-testid="stSidebar"] .stButton > button:hover {
                transform: translateY(-1px);
                border-color: rgba(11, 110, 79, 0.40);
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10);
                color: #064e3b;
            }
            [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #0b6e4f 0%, #0f766e 100%);
                color: #f8fafc;
                border: 1px solid rgba(11, 110, 79, 0.65);
            }
            [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
                color: #ffffff;
                border-color: rgba(15, 118, 110, 0.9);
            }
            [data-testid="stSidebar"] [data-testid="stExpander"] {
                border: none;
                box-shadow: none;
                background: transparent;
            }
            [data-testid="stSidebar"] .role-note {
                color: #475569;
                font-size: 0.82rem;
                line-height: 1.35;
                margin-top: 0.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_kpis() -> None:
    @st.cache_data(show_spinner=False, ttl=60)
    def _kpi_counts() -> dict[str, int]:
        return {
            "raw": count_files(RAW_DIR, "**/*.parquet"),
            "processed": count_files(PROCESSED_DIR, "**/*.parquet"),
            "gold_runs": count_files(GOLD_DIR / "governance", "governance_decision_*.json"),
            "artifacts": count_files(OUTPUT_DIR, "**/*"),
        }

    k = _kpi_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Files", k["raw"])
    c2.metric("Processed Files", k["processed"])
    # master_table.parquet is intentionally overwritten each run, so run progress
    # is better represented by governance decision artifacts.
    c3.metric("Gold Runs", k["gold_runs"])
    c4.metric("Output Artifacts", k["artifacts"])


__all__ = [
    "DIRECTIONS_MOD",
    "MAIN_MOD",
    "MEDALLION_MOD",
    "FETCHERS_MOD",
    "PIPELINE_STAGES",
    "inject_styles",
    "render_logger_message",
    "show_kpis",
]
