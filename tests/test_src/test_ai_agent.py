"""
Unit tests for src/ai_agent.py — ScenarioAIAgent.

All Ollama HTTP calls are mocked via unittest.mock so no live server is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.ai_agent import PAGE_CONTEXT_QUESTIONS, ScenarioAIAgent
from src.exceptions.AIAgentExceptions import (
    AIAgentError,
    AIOutputError,
    BackendSelectionError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    ModelNotFoundError,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def agent(tmp_path: Path) -> ScenarioAIAgent:
    """ScenarioAIAgent rooted at a temporary directory.

    Uses ``backend='local'`` to bypass auto-detection (no live Ollama needed in CI).
    """
    return ScenarioAIAgent(root=tmp_path, backend="local")


@pytest.fixture
def agent_online(tmp_path: Path) -> ScenarioAIAgent:
    """ScenarioAIAgent forced to the online (Gemini) backend for testing."""
    return ScenarioAIAgent(root=tmp_path, backend="online")


@pytest.fixture
def agent_with_outputs(tmp_path: Path) -> ScenarioAIAgent:
    """Agent with pre-seeded output and log files."""
    output_dir = tmp_path / "output" / "default"
    output_dir.mkdir(parents=True)

    (output_dir / "analysis_results.json").write_text(
        json.dumps({"result_keys": ["elasticity"], "results": {"governance_report": {"model_risk_score": 0.3}}}),
        encoding="utf-8",
    )
    (output_dir / "audit_report.json").write_text(
        json.dumps({"status": "PASS", "failed_checks": []}),
        encoding="utf-8",
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "session_summary_abc_123.json").write_text(
        json.dumps({"session_id": "abc", "score": 91}),
        encoding="utf-8",
    )
    return ScenarioAIAgent(root=tmp_path, backend="local")


# ── Exception hierarchy tests ─────────────────────────────────────────────────


def test_ai_agent_error_is_exception():
    assert issubclass(AIAgentError, Exception)


def test_llm_connection_error_inherits_from_ai_agent_error():
    assert issubclass(LLMConnectionError, AIAgentError)


def test_llm_timeout_error_inherits_from_ai_agent_error():
    assert issubclass(LLMTimeoutError, AIAgentError)


def test_llm_unavailable_error_inherits_from_ai_agent_error():
    assert issubclass(LLMUnavailableError, AIAgentError)


def test_llm_unavailable_error_inherits_from_llm_connection_error():
    assert issubclass(LLMUnavailableError, LLMConnectionError)


def test_llm_response_error_inherits_from_ai_agent_error():
    assert issubclass(LLMResponseError, AIAgentError)


def test_model_not_found_error_inherits_from_ai_agent_error():
    assert issubclass(ModelNotFoundError, AIAgentError)


def test_ai_output_error_inherits_from_ai_agent_error():
    assert issubclass(AIOutputError, AIAgentError)


def test_exception_carries_message():
    exc = LLMTimeoutError("timed out after 300s")
    assert "300s" in str(exc)


# ── ScenarioAIAgent construction ──────────────────────────────────────────────


def test_agent_default_model(agent: ScenarioAIAgent):
    assert agent.MODEL_NAME == "llama3.2:1b"


def test_agent_default_timeout_at_least_180(agent: ScenarioAIAgent):
    assert agent.timeout_seconds >= 180


def test_agent_root_is_path(agent: ScenarioAIAgent, tmp_path: Path):
    assert agent._root == tmp_path


def test_output_dir_under_root(agent: ScenarioAIAgent, tmp_path: Path):
    assert agent._output_dir("default") == tmp_path / "output" / "default"


def test_user_data_dir_under_root(agent: ScenarioAIAgent, tmp_path: Path):
    assert agent._user_data_dir("default") == tmp_path / "data" / "users" / "default"


def test_safe_user_id_strips_traversal(agent: ScenarioAIAgent):
    assert ".." not in agent._safe_user_id("../../etc/passwd")


# ── PAGE_CONTEXT_QUESTIONS ────────────────────────────────────────────────────


def test_page_context_questions_is_dict():
    assert isinstance(PAGE_CONTEXT_QUESTIONS, dict)


def test_page_context_questions_all_values_are_lists():
    for key, val in PAGE_CONTEXT_QUESTIONS.items():
        assert isinstance(val, list), f"Key '{key}' has non-list value"
        assert len(val) > 0, f"Key '{key}' has empty list"


def test_page_context_questions_has_ai_assistant_key():
    assert "🤖 Quantos Assistant" in PAGE_CONTEXT_QUESTIONS


# ── ping() ────────────────────────────────────────────────────────────────────


def test_ping_returns_true_when_model_available(agent: ScenarioAIAgent):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "llama3.2:1b"}]}

    with patch("src.ai_agent.requests.get", return_value=mock_resp):
        ok, reason = agent.ping()

    assert ok is True
    assert "connected" in reason


def test_ping_returns_false_when_model_missing(agent: ScenarioAIAgent):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "mistral:7b"}]}

    with patch("src.ai_agent.requests.get", return_value=mock_resp):
        ok, reason = agent.ping()

    assert ok is False
    assert "not found" in reason.lower() or "ollama pull" in reason


def test_ping_returns_false_on_http_error(agent: ScenarioAIAgent):
    mock_resp = MagicMock()
    mock_resp.status_code = 503

    with patch("src.ai_agent.requests.get", return_value=mock_resp):
        ok, reason = agent.ping()

    assert ok is False
    assert "503" in reason


def test_ping_returns_false_on_connection_error(agent: ScenarioAIAgent):
    with patch("src.ai_agent.requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        ok, reason = agent.ping()

    assert ok is False
    assert reason  # non-empty


def test_ping_returns_false_on_timeout(agent: ScenarioAIAgent):
    with patch("src.ai_agent.requests.get", side_effect=requests.exceptions.Timeout("timed out")):
        ok, reason = agent.ping()

    assert ok is False


# ── _llama_generate() ─────────────────────────────────────────────────────────


def test_llama_generate_returns_response_text(agent: ScenarioAIAgent):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "Η απάντηση είναι 42."}

    with patch("src.ai_agent.requests.post", return_value=mock_resp):
        result = agent._llama_generate("test prompt")

    assert result == "Η απάντηση είναι 42."


def test_llama_generate_raises_llm_timeout(agent: ScenarioAIAgent):
    with patch(
        "src.ai_agent.requests.post",
        side_effect=requests.exceptions.Timeout("timed out"),
    ):
        with pytest.raises(LLMTimeoutError):
            agent._llama_generate("test")


def test_llama_generate_raises_llm_connection_error(agent: ScenarioAIAgent):
    with patch(
        "src.ai_agent.requests.post",
        side_effect=requests.exceptions.ConnectionError("refused"),
    ):
        with pytest.raises(LLMConnectionError):
            agent._llama_generate("test")


def test_llama_generate_raises_llm_unavailable_on_non200(agent: ScenarioAIAgent):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    with patch("src.ai_agent.requests.post", return_value=mock_resp):
        with pytest.raises(LLMUnavailableError):
            agent._llama_generate("test")


def test_llama_generate_raises_llm_response_error_on_bad_json(agent: ScenarioAIAgent):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("not JSON")

    with patch("src.ai_agent.requests.post", return_value=mock_resp):
        with pytest.raises(LLMResponseError):
            agent._llama_generate("test")


# ── build_context_bundle() ────────────────────────────────────────────────────


def test_build_context_bundle_returns_dict(agent: ScenarioAIAgent):
    bundle = agent.build_context_bundle("default")
    assert isinstance(bundle, dict)
    assert "generated_at" in bundle
    assert "quick_signals" in bundle


def test_build_context_bundle_reads_output_files(agent_with_outputs: ScenarioAIAgent):
    bundle = agent_with_outputs.build_context_bundle("default")
    assert bundle["audit_report"].get("status") == "PASS"
    assert bundle["quick_signals"]["audit_status"] == "PASS"


def test_build_context_bundle_handles_missing_files_gracefully(agent: ScenarioAIAgent):
    # Files don't exist in tmp_path — should not raise
    bundle = agent.build_context_bundle("nonexistent_user")
    assert isinstance(bundle, dict)
    assert bundle["analysis_results"] == {}


# ── answer_question() ─────────────────────────────────────────────────────────


def test_answer_question_success(agent: ScenarioAIAgent):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "Το pipeline λειτουργεί κανονικά."}

    with patch("src.ai_agent.requests.post", return_value=mock_resp):
        result = agent.answer_question("Πώς πάει το pipeline;")

    assert result["success"] is True
    assert "pipeline" in result["answer"].lower()


def test_answer_question_failure_on_llm_error(agent: ScenarioAIAgent):
    with patch(
        "src.ai_agent.requests.post",
        side_effect=requests.exceptions.Timeout("timeout"),
    ):
        result = agent.answer_question("Ερώτηση;")

    assert result["success"] is False
    assert "LLMTimeoutError" in result["answer"]


def test_answer_question_includes_page_hint(agent: ScenarioAIAgent):
    """Verify current_page is injected into the prompt."""
    captured_prompts: list[str] = []

    def mock_post(url, json=None, timeout=None):
        captured_prompts.append(json.get("prompt", "") if json else "")
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"response": "OK"}
        return m

    with patch("src.ai_agent.requests.post", side_effect=mock_post):
        agent.answer_question("question", current_page="🩺 Health & Alerts")

    assert any("Health & Alerts" in p for p in captured_prompts)


# ── quick_insight() ───────────────────────────────────────────────────────────


def test_quick_insight_success(agent: ScenarioAIAgent):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "Το score είναι υψηλό."}

    with patch("src.ai_agent.requests.post", return_value=mock_resp):
        result = agent.quick_insight("Health", {"score": 92})

    assert result["success"] is True
    assert result["insight"]


def test_quick_insight_failure_returns_error_dict(agent: ScenarioAIAgent):
    with patch(
        "src.ai_agent.requests.post",
        side_effect=requests.exceptions.ConnectionError("down"),
    ):
        result = agent.quick_insight("Health", {})

    assert result["success"] is False
    assert "LLMConnectionError" in result["insight"]


# ── create_pipeline_brief() ───────────────────────────────────────────────────


def test_create_pipeline_brief_writes_files(agent: ScenarioAIAgent, tmp_path: Path):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "1) Run Health: OK\n2) Key Signals: ..."}

    with patch("src.ai_agent.requests.post", return_value=mock_resp):
        result = agent.create_pipeline_brief("default")

    assert result["success"] is True
    assert Path(result["json_path"]).exists()
    assert Path(result["md_path"]).exists()


def test_create_pipeline_brief_failure_on_llm_error(agent: ScenarioAIAgent):
    with patch(
        "src.ai_agent.requests.post",
        side_effect=requests.exceptions.Timeout("timeout"),
    ):
        result = agent.create_pipeline_brief("default")

    assert result["success"] is False
    assert "LLMTimeoutError" in result["error"]


# ── _detect_backend() ─────────────────────────────────────────────────────────


def test_detect_backend_returns_local_when_ollama_up(tmp_path: Path):
    """_detect_backend returns 'local' when Ollama /api/tags responds 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("src.ai_agent.requests.get", return_value=mock_resp):
        ag = ScenarioAIAgent(root=tmp_path)  # no explicit backend → auto-detect

    assert ag.backend == "local"


def test_detect_backend_returns_online_when_ollama_down_and_key_set(tmp_path: Path, monkeypatch):
    """Falls through to 'online' when Ollama is unreachable but GEMINI_API_KEY is set."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    with patch("src.ai_agent.requests.get", side_effect=requests.exceptions.ConnectionError()):
        ag = ScenarioAIAgent(root=tmp_path)

    assert ag.backend == "online"


def test_detect_backend_raises_when_neither_available(tmp_path: Path, monkeypatch):
    """Raises BackendSelectionError when Ollama is down AND no Gemini key."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with patch("src.ai_agent.requests.get", side_effect=requests.exceptions.ConnectionError()):
        with pytest.raises(BackendSelectionError):
            ScenarioAIAgent(root=tmp_path)


def test_invalid_backend_raises_on_init(tmp_path: Path):
    """Passing an unsupported backend string raises BackendSelectionError."""
    with pytest.raises(BackendSelectionError):
        ScenarioAIAgent(root=tmp_path, backend="invalid")


# ── _get_gemini_api_key() ─────────────────────────────────────────────────────


def test_get_gemini_api_key_reads_env(agent: ScenarioAIAgent, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key-123")
    assert agent._get_gemini_api_key() == "env-key-123"


def test_get_gemini_api_key_returns_none_when_missing(agent: ScenarioAIAgent, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Also ensure no st.secrets leak from test environment
    with patch("src.ai_agent.requests"):
        key = agent._get_gemini_api_key()
    assert key is None


# ── _gemini_generate() ────────────────────────────────────────────────────────


def test_gemini_generate_returns_text(agent_online: ScenarioAIAgent, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "Test answer"}]}}]}

    with patch("src.ai_agent.requests.post", return_value=mock_resp):
        result = agent_online._gemini_generate("hello")

    assert result == "Test answer"


def test_gemini_generate_raises_auth_error_without_key(agent_online: ScenarioAIAgent, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMAuthenticationError):
        agent_online._gemini_generate("hello")


def test_gemini_generate_raises_timeout(agent_online: ScenarioAIAgent, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    with patch("src.ai_agent.requests.post", side_effect=requests.exceptions.Timeout()):
        with pytest.raises(LLMTimeoutError):
            agent_online._gemini_generate("hello")


def test_gemini_generate_raises_connection_error(agent_online: ScenarioAIAgent, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    with patch("src.ai_agent.requests.post", side_effect=requests.exceptions.ConnectionError()):
        with pytest.raises(LLMConnectionError):
            agent_online._gemini_generate("hello")


def test_gemini_generate_raises_unavailable_on_non200(agent_online: ScenarioAIAgent, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Rate limit exceeded"

    with patch("src.ai_agent.requests.post", return_value=mock_resp):
        with pytest.raises(LLMUnavailableError):
            agent_online._gemini_generate("hello")


def test_gemini_generate_raises_response_error_on_bad_json(agent_online: ScenarioAIAgent, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("not JSON")

    with patch("src.ai_agent.requests.post", return_value=mock_resp):
        with pytest.raises(LLMResponseError):
            agent_online._gemini_generate("hello")


# ── _ping_gemini() ────────────────────────────────────────────────────────────


def test_ping_gemini_returns_true_when_api_responds(agent_online: ScenarioAIAgent, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("src.ai_agent.requests.post", return_value=mock_resp):
        ok, msg = agent_online.ping()

    assert ok is True
    assert "gemini" in msg.lower()


def test_ping_gemini_returns_false_without_key(agent_online: ScenarioAIAgent, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ok, msg = agent_online.ping()
    assert ok is False
    assert msg


# ── _generate() dispatch ──────────────────────────────────────────────────────


def test_generate_dispatches_to_llama_for_local(agent: ScenarioAIAgent):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "local answer"}

    with patch("src.ai_agent.requests.post", return_value=mock_resp):
        result = agent._generate("hello")

    assert result == "local answer"


def test_generate_dispatches_to_gemini_for_online(agent_online: ScenarioAIAgent, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "online answer"}]}}]}

    with patch("src.ai_agent.requests.post", return_value=mock_resp):
        result = agent_online._generate("hello")

    assert result == "online answer"
