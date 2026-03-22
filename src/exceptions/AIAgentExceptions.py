"""
Custom Exceptions for the AI Agent (Ollama / LLM integration).
Ordered from most critical to least critical.
"""


class AIAgentError(Exception):
    """Base exception for all AI agent errors."""

    pass


# ── Connectivity & Communication ─────────────────────────────────────────────

class LLMConnectionError(AIAgentError):
    """Raised when the agent cannot reach the Ollama / LLM endpoint."""

    pass


class LLMTimeoutError(AIAgentError):
    """Raised when an LLM request exceeds the configured timeout."""

    pass


class LLMUnavailableError(LLMConnectionError):
    """Raised when Ollama is running but returns a non-200 status."""

    pass


# ── Authentication & Configuration ───────────────────────────────────────────

class AIAgentConfigError(AIAgentError):
    """Raised for missing or invalid AI agent configuration (model name, base URL, etc.)."""

    pass


class LLMAuthenticationError(AIAgentConfigError):
    """Raised when the API key or authentication token is invalid."""

    pass


class ModelNotFoundError(AIAgentConfigError):
    """Raised when the requested model is not available on the Ollama server."""

    pass


class BackendSelectionError(AIAgentConfigError):
    """Raised when no suitable AI backend (local or online) is available."""

    pass


# ── Response & Parsing ────────────────────────────────────────────────────────

class LLMResponseError(AIAgentError):
    """Raised when the LLM returns a malformed or unexpected response."""

    pass


class AIResponseParseError(LLMResponseError):
    """Raised when the agent fails to parse structured output (JSON, sections, etc.) from the LLM."""

    pass


class ContextWindowError(LLMResponseError):
    """Raised when the prompt payload exceeds the model's context window."""

    pass


# ── Context & Data ────────────────────────────────────────────────────────────

class AIContextError(AIAgentError):
    """Raised for errors when building or reading the context bundle."""

    pass


class MissingContextError(AIContextError):
    """Raised when required context files (analysis results, audit report, etc.) are missing."""

    pass


class ContextSerializationError(AIContextError):
    """Raised when the context bundle cannot be serialised to JSON for the prompt."""

    pass


# ── Output & Persistence ──────────────────────────────────────────────────────

class AIOutputError(AIAgentError):
    """Raised when the agent fails to persist a briefing or insight to disk."""

    pass
