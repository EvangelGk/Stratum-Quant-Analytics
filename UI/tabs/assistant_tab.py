from __future__ import annotations

# Copyright (c) 2026 EvangelGK. All Rights Reserved.
import os
import time
from typing import Any

import requests
import streamlit as st

from src.ai_agent import PAGE_CONTEXT_QUESTIONS, QuantosAgent
from src.exceptions.AIAgentExceptions import AIOutputError, BackendSelectionError
from src.secret_store import get_secret

# ── Internal helpers ──────────────────────────────────────────────────────────


def _current_agent_signature() -> tuple[str, str, str, str, str]:
    """Return the current backend-relevant settings fingerprint for this session."""
    return (
        "1" if (get_secret("GEMINI_API_KEY") or "").strip() else "0",
        "1" if (get_secret("GROQ_API_KEY") or "").strip() else "0",
        (get_secret("OLLAMA_API_URL") or "http://127.0.0.1:11434/api/generate").strip(),
        (get_secret("OLLAMA_MODEL") or "llama3.2:1b").strip(),
        (get_secret("GROQ_MODEL") or "llama-3.3-70b-versatile").strip(),
    )


def _get_agent() -> QuantosAgent | None:
    """Return a session-cached agent. Rebuilds if missing. Returns None when no backend available."""
    signature = _current_agent_signature()
    if st.session_state.get("_ai_agent_signature") != signature:
        st.session_state.pop("_ai_agent_instance", None)
        st.session_state["_ai_agent_signature"] = signature

    if "_ai_agent_instance" not in st.session_state:
        try:
            st.session_state["_ai_agent_instance"] = QuantosAgent()
        except BackendSelectionError as exc:
            # Mark offline so the Retry button becomes visible immediately.
            st.session_state["ai_ready"] = False
            st.session_state["ai_offline_reason"] = str(exc)
            st.session_state["ai_ready_checked_at"] = time.time()
            return None
    return st.session_state["_ai_agent_instance"]  # type: ignore[return-value]


def _ensure_messages() -> list[dict[str, str]]:
    """Return (and init if needed) the shared chat message list in session state."""
    if "ai_messages" not in st.session_state:
        st.session_state["ai_messages"] = []
    return st.session_state["ai_messages"]  # type: ignore[return-value]


def _check_ready(force: bool = False) -> bool:
    """Resolve Quantos readiness.

    Performance policy:
    - Online keys (Gemini / Groq): optimistic — key configured → Ready immediately,
      no network probe. The actual health is checked live on every API call inside
      _generate(), which already cascades Gemini → Groq automatically.
    - Offline (only-Ollama) mode: TTL-based auto-recheck every 30 s so the UI
      recovers automatically without requiring a manual Retry press.
    """
    _ONLINE_TTL = 120.0  # seconds before re-verifying that online keys are still present
    _RECHECK_TTL = 30.0  # seconds before re-probing a Failed / Offline state
    now = time.time()
    last_checked = st.session_state.get("ai_ready_checked_at", 0.0)

    # Fast path for Ready state — but only within the online TTL window.
    # Once expired, fall through to re-verify keys are still present in secrets.
    if not force and st.session_state.get("ai_ready") is True:
        if (now - last_checked) < _ONLINE_TTL:
            return True
        # TTL expired → re-check keys (no network call, just re-read secrets)

    # Fast path for Failed/Offline state — honour cache only within short TTL
    if not force and st.session_state.get("ai_ready") is False:
        if (now - last_checked) < _RECHECK_TTL:
            return False
        # TTL expired → fall through and re-probe

    # Online key check — optimistic, no network call needed.
    # Gemini and/or Groq both count as valid online backends.
    gkey = (get_secret("GEMINI_API_KEY") or "").strip()
    qkey = (get_secret("GROQ_API_KEY") or "").strip()
    if gkey or qkey:
        if gkey and qkey:
            label = "gemini + groq configured"
        elif gkey:
            label = "gemini configured"
        else:
            label = "groq configured"
        st.session_state["ai_ready"] = True
        st.session_state["ai_offline_reason"] = label
        st.session_state["ai_ready_checked_at"] = now
        return True

    # No online keys — probe Ollama.
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=(0.6, 1.0))
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            found = any("llama3.2" in m for m in models)
            ok = bool(found)
            reason = "ollama connected" if ok else "ollama up but llama3.2:1b not found"
        else:
            ok, reason = False, f"ollama HTTP {r.status_code}"
    except Exception as exc:
        ok, reason = False, str(exc)

    st.session_state["ai_last_ollama_ok"] = ok
    st.session_state["ai_ready"] = ok
    st.session_state["ai_offline_reason"] = "" if ok else reason
    st.session_state["ai_ready_checked_at"] = now
    return ok


def _submit_question(
    question: str,
    user_id: str,
    current_page: str = "",
) -> str:
    """Ask the agent a question, push both turns to shared history, return answer."""
    agent = _get_agent()
    msgs = _ensure_messages()

    if agent is None:
        # Agent couldn't be constructed — backend unavailable.
        offline_msg = st.session_state.get("ai_offline_reason", "AI backend unavailable.")
        st.session_state["ai_last_error"] = offline_msg
        return offline_msg

    result = agent.answer_question(question, user_id=user_id, current_page=current_page)
    answer: str = result.get("answer", "")

    # If the call failed (quota exhausted, outage, auth error), evict the cached
    # agent and clear the ready flag so the next interaction re-probes the backend
    # automatically — no manual Retry required.
    if not result.get("success"):
        st.session_state["ai_last_error"] = answer
        st.session_state.pop("_ai_agent_instance", None)
        st.session_state["ai_ready"] = False
        st.session_state["ai_ready_checked_at"] = 0.0  # re-probe on very next event
        return answer

    st.session_state.pop("ai_last_error", None)
    msgs.append({"role": "user", "content": question})
    msgs.append({"role": "assistant", "content": answer})
    return answer


# ── Sidebar mini-chat (always visible) ───────────────────────────────────────


def render_sidebar_ai_widget() -> None:
    """Compact AI widget called from inside _render_sidebar().

    Shows status badge, last AI answer preview, quick-ask form, and a link
    to the full Quantos Assistant tab. All questions use the shared message history
    so they also appear in the full chat tab.
    """
    ai_ready = _check_ready()
    status_label = "🟢 Ready" if ai_ready else "⚫ Offline"

    with st.expander(f"🤖 Quantos — {status_label}", expanded=True):
        last_error = str(st.session_state.get("ai_last_error", "") or "")
        if last_error:
            st.warning(last_error)

        if not ai_ready:
            reason = st.session_state.get("ai_offline_reason", "")
            st.caption(f"Quantos offline{': ' + reason[:120] if reason else ''}.")
            if st.button("🔄 Retry", key="sidebar_ai_retry", use_container_width=True):
                st.session_state.pop("ai_last_error", None)
                _check_ready(force=True)
                st.rerun()
            return

        # Preview of last AI response
        messages = _ensure_messages()
        last_ai = next(
            (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
            None,
        )
        if last_ai:
            preview = last_ai[:220] + "…" if len(last_ai) > 220 else last_ai
            with st.expander("Last answer", expanded=False):
                st.caption(preview)

        # Quick-ask form (clear_on_submit keeps the field empty after asking)
        with st.form(key="sidebar_ai_form", clear_on_submit=True, border=False):
            sidebar_q = st.text_input(
                "sidebar_q_label",
                placeholder="Ρώτα τον Quantos…",
                label_visibility="collapsed",
                key="sidebar_ai_input",
            )
            submitted = st.form_submit_button("Ask AI →", use_container_width=True, type="primary")

        if submitted and sidebar_q.strip():
            user_id = os.getenv("DATA_USER_ID", "default")
            current_page = st.session_state.get("selected_page", "")
            with st.spinner("Thinking…"):
                _submit_question(sidebar_q.strip(), user_id, current_page)
            st.rerun()

        st.caption("Full chat → **🤖 Quantos Assistant** tab")


# ── Inline contextual AI section (embeddable in any tab) ─────────────────────


def render_inline_ai_section(
    topic: str,
    snapshot: dict[str, Any],
    key_suffix: str = "",
) -> None:
    """Collapsible inline AI insight capsule.

    Place at the bottom of any tab section to give users contextual AI access
    without navigating away. Questions asked here are pushed to the shared chat
    history so they also appear in the full Quantos Assistant tab.
    """
    with st.expander("🤖 AI insight on this section", expanded=False):
        if not st.session_state.get("ai_ready"):
            st.caption("⚫ Quantos offline. Retry when backend is available.")
            return

        user_id = os.getenv("DATA_USER_ID", "default")
        current_page = st.session_state.get("selected_page", "")
        agent = _get_agent()
        if agent is None:
            st.caption("⚫ Quantos offline. Retry when backend is available.")
            return

        col_a, col_b = st.columns([3, 1])

        custom_q = col_a.text_input(
            "inline_q_label",
            placeholder=f"Ask about {topic}…",
            key=f"inline_q_{key_suffix}",
            label_visibility="collapsed",
        )

        if col_b.button("Ask", key=f"inline_ask_{key_suffix}", use_container_width=True) and custom_q.strip():
            with st.spinner("Thinking…"):
                answer = _submit_question(custom_q.strip(), user_id, current_page)
            st.markdown(answer)
            return

        if st.button(
            f"Get AI insight on: {topic}",
            key=f"inline_insight_{key_suffix}",
            use_container_width=True,
        ):
            with st.spinner("AI analyzing…"):
                result = agent.quick_insight(topic=topic, snapshot=snapshot, user_id=user_id)
            if result.get("success"):
                insight: str = result["insight"]
                st.session_state.pop("ai_last_error", None)
                st.markdown(insight)
                msgs = _ensure_messages()
                msgs.append({"role": "user", "content": f"[Insight] {topic}"})
                msgs.append({"role": "assistant", "content": insight})
            else:
                st.session_state["ai_last_error"] = str(result.get("insight", "AI error"))
                st.session_state.pop("_ai_agent_instance", None)
                st.session_state["ai_ready"] = False
                st.session_state["ai_ready_checked_at"] = 0.0
                st.error(result.get("insight", "AI error"))


# ── Full Quantos Assistant tab ───────────────────────────────────────────────


def show_ai_assistant_tab() -> None:
    """Full Quantos chat tab — bubbles, quick chips, pipeline brief."""
    user_id = os.getenv("DATA_USER_ID", "default")

    # ── Status / action bar ───────────────────────────────────────────
    ai_ready = _check_ready()
    if not ai_ready:
        reason = st.session_state.get("ai_offline_reason", "unknown")
        st.error("🔴 **Quantos offline**")
        last_error = str(st.session_state.get("ai_last_error", "") or "")
        if last_error:
            st.warning(last_error)
        with st.expander("🔍 Diagnostic details", expanded=True):
            st.code(reason, language=None)
        if st.button("🔄 Retry connection", key="main_ai_retry", type="primary"):
            st.session_state.pop("ai_last_error", None)
            _check_ready(force=True)
            st.rerun()
        return

    agent = _get_agent()
    # Guard: _get_agent() may return None if backend detection failed right here.
    if agent is None:
        reason = st.session_state.get("ai_offline_reason", "No backend available.")
        st.error(f"🔴 **Quantos offline**: {reason}")
        if st.button("🔄 Retry", key="main_ai_retry_agent", type="primary"):
            st.session_state.pop("ai_ready", None)
            st.rerun()
        return

    c_status, c_brief, c_clear = st.columns([5, 3, 1])
    c_status.success("🟢 Quantos online")
    last_error = str(st.session_state.get("ai_last_error", "") or "")
    if last_error:
        st.warning(last_error)

    if c_brief.button("📋 Pipeline Brief", use_container_width=True, help="Generate a full AI briefing from the latest pipeline outputs"):
        with st.spinner("Generating Quantos pipeline briefing…"):
            try:
                brief = agent.create_pipeline_brief(user_id=user_id)
            except AIOutputError as exc:
                st.error(f"Brief failed (output write error): {exc}")
                brief = {"success": False}
        if brief.get("success"):
            st.session_state.pop("ai_last_error", None)
            st.toast("Brief saved to output/", icon="✅")
            with st.container(border=True):
                st.markdown("#### 📋 Quantos Pipeline Brief")
                st.markdown(brief.get("brief", ""))
        else:
            if brief.get("error"):
                st.session_state["ai_last_error"] = str(brief.get("error", "unknown"))
                st.error(f"Brief failed: {brief.get('error', 'unknown')}")
            # LLM failure → reset agent for auto-recovery on next interaction
            if not brief.get("success"):
                st.session_state.pop("_ai_agent_instance", None)
                st.session_state["ai_ready"] = False
                st.session_state["ai_ready_checked_at"] = 0.0

    if c_clear.button("🗑️", help="Clear chat history", use_container_width=True):
        st.session_state["ai_messages"] = []
        st.session_state.pop("ai_pending_question", None)
        st.rerun()

    st.divider()

    # ── Context-aware quick-action chips ─────────────────────────────
    current_page = st.session_state.get("selected_page", "🤖 Quantos Assistant")
    page_questions = PAGE_CONTEXT_QUESTIONS.get(current_page, PAGE_CONTEXT_QUESTIONS["🤖 Quantos Assistant"])
    st.caption(f"Quick questions for **{current_page}** — click to ask instantly:")
    chip_cols = st.columns(min(len(page_questions), 4))
    for col, q in zip(chip_cols, page_questions[:4]):
        label = q[:38] + "…" if len(q) > 38 else q
        if col.button(label, key=f"chip_{abs(hash(q))}", use_container_width=True, help=q):
            st.session_state["ai_pending_question"] = q

    st.divider()

    # ── Chat message history ──────────────────────────────────────────
    messages = _ensure_messages()

    if not messages:
        st.info("Δεν υπάρχουν μηνύματα ακόμα. Χρησιμοποίησε τα quick chips παραπάνω ή γράψε στο πεδίο κάτω για να ξεκινήσεις συνομιλία.")

    for msg in messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Process pending question from chips ───────────────────────────
    pending: str | None = st.session_state.pop("ai_pending_question", None)
    if pending:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(pending)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                answer = _submit_question(pending, user_id, current_page)
            st.markdown(answer)

    # ── Chat input (pinned at bottom by Streamlit) ────────────────────
    if prompt := st.chat_input("Ρώτα τον Quantos για pipeline, αναλύσεις, governance, optimizer…"):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                answer = _submit_question(prompt, user_id, current_page)
            st.markdown(answer)
