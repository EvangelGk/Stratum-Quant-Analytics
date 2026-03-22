from __future__ import annotations

# Copyright (c) 2026 EvangelGK. All Rights Reserved.

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

try:
    from src.secret_store import bootstrap_env_from_secrets, get_secret
except ModuleNotFoundError:
    from secret_store import bootstrap_env_from_secrets, get_secret

try:
    from src.exceptions.AIAgentExceptions import (
        AIAgentConfigError,
        AIOutputError,
        BackendSelectionError,
        ContextSerializationError,
        LLMAuthenticationError,
        LLMConnectionError,
        LLMResponseError,
        LLMTimeoutError,
        LLMUnavailableError,
        ModelNotFoundError,
    )
except ModuleNotFoundError:
    from exceptions.AIAgentExceptions import (
        AIAgentConfigError,
        AIOutputError,
        BackendSelectionError,
        ContextSerializationError,
        LLMAuthenticationError,
        LLMConnectionError,
        LLMResponseError,
        LLMTimeoutError,
        LLMUnavailableError,
        ModelNotFoundError,
    )


# Suggested questions per UI section, shown as one-click chips
PAGE_CONTEXT_QUESTIONS: dict[str, list[str]] = {
    "🤖 Quantos Assistant": [
        "Ποια είναι η συνολική κατάσταση του τελευταίου run;",
        "Ποιες αναλύσεις χρειάζονται άμεση προσοχή;",
        "Εξήγησε την governance απόφαση και το risk score.",
        "Ποιες είναι οι 3 ενέργειες βελτίωσης αξιοπιστίας;",
    ],
    "🩺 Health & Alerts": [
        "Γιατί το data health score είναι στο τρέχον επίπεδο;",
        "Ποιες ελλείπουσες πηγές επηρεάζουν τα αποτελέσματα;",
        "Πόσο κρίσιμα τα alerts και τι πρέπει να γίνει;",
        "Ποιο penalty μειώνει περισσότερο το health score;",
    ],
    "🧪 Auditor": [
        "Τι σημαίνουν στην πράξη τα failed audit checks;",
        "Είναι αποδεκτό το OOS R² για production χρήση;",
        "Εξήγησε το model risk score και τις προεκτάσεις του.",
        "Τι πρέπει να διορθώσω για να περάσει το audit;",
    ],
    "🛡️ Governance": [
        "Πέρασε ή απέτυχε το governance και γιατί;",
        "Ποιος είναι ο κίνδυνος publication lag;",
        "Πρέπει να εμπιστευθώ τα walk-forward αποτελέσματα;",
        "Τι σημαίνει το severity level του governance gate;",
    ],
    "📈 Analytics": [
        "Τι μας λέει το τρέχον elasticity β;",
        "Πώς να ερμηνεύσω τα Monte Carlo risk metrics;",
        "Ποιοι stress factors έχουν τον μεγαλύτερο αντίκτυπο;",
        "Πόσο αξιόπιστο είναι το volatility forecast;",
    ],
    "🎛️ Scenario Builder": [
        "Είναι ρεαλιστικά τα scenario shocks;",
        "Πόσο αξιόπιστη είναι η εκτίμηση impact;",
        "Ποιο scenario είναι πιο επικίνδυνο για το χαρτοφυλάκιο;",
    ],
    "📊 Run Comparison": [
        "Είναι σημαντική η αλλαγή runtime μεταξύ runs;",
        "Τι προκάλεσε την αλλαγή στον αριθμό operations;",
    ],
    "⚙️ Ops": [
        "Υπάρχουν operational κίνδυνοι;",
        "Τι σημαίνουν τα source coverage issues;",
        "Ποια είναι η υγεία του pipeline scheduler;",
    ],
    "🗂️ Data": [
        "Ποιο layer έχει τα περισσότερα προβλήματα ποιότητας;",
        "Γιατί υπάρχουν κενά στα raw δεδομένα;",
        "Ποιος έλεγχος silver αποτυχαίνει συχνότερα;",
    ],
    "🧠 Explainability": [
        "Γιατί άλλαξαν τα αποτελέσματα σε σχέση με το προηγούμενο run;",
        "Ποιος παράγοντας επηρεάζει περισσότερο τις αποδόσεις;",
        "Εξήγησε την αλλαγή στο elasticity σε απλή γλώσσα.",
    ],
    "🧾 Reports": [
        "Ποια είναι τα βασικά ευρήματα για τη διοίκηση;",
        "Σε ποιο σημείο υπάρχει ο μεγαλύτερος κίνδυνος;",
    ],
}


class QuantosAgent:
    """Context-aware Quantos assistant for UI Q&A and pipeline briefings.
    
    Supports both local (Ollama/Llama) and online (Google Gemini) backends.
    Detects availability and uses appropriate backend automatically.
    """

    # Local backend (Ollama)
    OLLAMA_API_URL = get_secret("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate") or "http://127.0.0.1:11434/api/generate"
    MODEL_NAME = get_secret("OLLAMA_MODEL", "llama3.2:1b") or "llama3.2:1b"
    
    # Online backend (Google Gemini)
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    RESTING_MESSAGE = "Quantos is currently resting, please try again later."

    def __init__(self, root: Path | None = None, timeout_seconds: int | None = None, backend: str | None = None) -> None:
        bootstrap_env_from_secrets(override=False)
        self._root = root or Path(__file__).resolve().parent.parent
        # Default 300s — llama3.2:1b can be slow on CPU; override with OLLAMA_TIMEOUT env var
        default_timeout = int(get_secret("OLLAMA_TIMEOUT", "300") or "300")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else default_timeout
        
        # Detect backend if not specified
        if backend is None:
            self.backend = self._detect_backend()
        else:
            self.backend = backend
        
        if self.backend not in ("local", "online"):
            raise BackendSelectionError(
                f"Invalid backend '{backend}'. Must be 'local' or 'online'."
            )

    def _detect_backend(self) -> str:
        """Detect which backend is available: 'local' (Ollama) or 'online' (Gemini)."""
        # Try to check if Ollama is accessible
        try:
            response = requests.get(
                f"{self._base_url()}/api/tags",
                timeout=(2, 5),  # Fast check: 2s connect, 5s read
            )
            if response.status_code == 200:
                return "local"
        except Exception:
            pass  # Ollama not available, fall through to Gemini
        
        # Check if Gemini is configured
        if self._get_gemini_api_key():
            return "online"
        
        # No backend available
        raise BackendSelectionError(
            "No AI backend available. Either:\n"
            "1) Ensure Ollama is running at 127.0.0.1:11434, OR\n"
            "2) Set GEMINI_API_KEY in Streamlit secrets or environment."
        )

    def _base_url(self) -> str:
        """Strip /api/generate suffix to get the Ollama root URL."""
        url = self.OLLAMA_API_URL
        for suffix in ("/api/generate", "/api/chat"):
            if url.endswith(suffix):
                return url[: -len(suffix)]
        return url.rsplit("/api/", 1)[0] if "/api/" in url else url

    def _get_gemini_api_key(self) -> str | None:
        """Get Gemini API key from the unified secret store."""
        return get_secret("GEMINI_API_KEY")

    def _friendly_user_error(self, exc: Exception) -> str:
        txt = str(exc).lower()
        if any(token in txt for token in ("quota", "resource_exhausted", "rate limit", "429")):
            return self.RESTING_MESSAGE
        return f"Quantos is temporarily unavailable: {type(exc).__name__}. Please try again later."

    def _gemini_generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Send prompt to Google Gemini and return the response text.

        Raises:
            LLMAuthenticationError: Gemini API key is missing or invalid.
            LLMConnectionError: Cannot reach the Gemini endpoint.
            LLMTimeoutError: Request exceeded configured timeout.
            LLMUnavailableError: Gemini returned a non-200 HTTP status.
            LLMResponseError: Response payload is malformed.
        """
        api_key = self._get_gemini_api_key()
        if not api_key:
            raise LLMAuthenticationError(
                "GEMINI_API_KEY not found in Streamlit secrets or environment. "
                "Set it via: st.secrets['GEMINI_API_KEY'] = 'xxx'"
            )

        try:
            response = requests.post(
                f"{self.GEMINI_API_URL}?key={api_key}",
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": 1024,
                    }
                },
                timeout=self.timeout_seconds,
            )
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(
                f"Gemini request timed out after {self.timeout_seconds}s: {exc}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Cannot connect to Gemini API: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMConnectionError(f"Gemini request failed: {exc}") from exc

        if response.status_code == 429:
            raise LLMUnavailableError(self.RESTING_MESSAGE)
        if response.status_code != 200:
            body = response.text[:200].lower()
            if "quota" in body or "resource_exhausted" in body:
                raise LLMUnavailableError(self.RESTING_MESSAGE)
            raise LLMUnavailableError(
                f"Gemini returned HTTP {response.status_code}: {response.text[:200]}"
            )
        
        try:
            payload = response.json()
        except Exception as exc:
            raise LLMResponseError(f"Failed to parse Gemini JSON response: {exc}") from exc

        # Extract text from Gemini response structure
        try:
            candidates = payload.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return str(parts[0].get("text", "")).strip()
        except (KeyError, IndexError, TypeError):
            pass

        raise LLMResponseError(f"Unexpected Gemini response format: {json.dumps(payload)[:200]}")

    @staticmethod
    def _safe_user_id(user_id: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(user_id))
        return cleaned or "default"


    def _output_dir(self, user_id: str) -> Path:
        return self._root / "output" / self._safe_user_id(user_id)

    def _user_data_dir(self, user_id: str) -> Path:
        return self._root / "data" / "users" / self._safe_user_id(user_id)

    def _latest_session_summary(self) -> dict[str, Any]:
        logs_dir = self._root / "logs"
        if not logs_dir.exists():
            return {}
        files = sorted(logs_dir.glob("session_summary_*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            return {}
        return self._read_json(files[-1])

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def ping(self) -> tuple[bool, str]:
        """Check AI backend availability (Ollama for local, Gemini for online)."""
        if self.backend == "local":
            return self._ping_ollama()
        elif self.backend == "online":
            return self._ping_gemini()
        else:
            return False, f"Unknown backend: {self.backend}"

    def _ping_ollama(self) -> tuple[bool, str]:
        """Check Ollama server availability using /api/tags (no model loading)."""
        try:
            response = requests.get(
                f"{self._base_url()}/api/tags",
                timeout=(5, 15),  # connect=5s, read=15s
            )
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                model_base = self.MODEL_NAME.split(":")[0]
                available = any(model_base in m for m in models)
                if not available:
                    return False, ModelNotFoundError(
                        f"Model '{self.MODEL_NAME}' not found on server. "
                        f"Run: ollama pull {self.MODEL_NAME}"
                    ).args[0]
                return True, "Ollama connected"
            return False, LLMUnavailableError(f"HTTP {response.status_code}").args[0]
        except requests.exceptions.Timeout as exc:
            return False, LLMTimeoutError(str(exc)).args[0]
        except Exception as exc:
            return False, LLMConnectionError(str(exc)).args[0]

    def _ping_gemini(self) -> tuple[bool, str]:
        """Check Gemini API availability."""
        api_key = self._get_gemini_api_key()
        if not api_key:
            return False, LLMAuthenticationError(
                "GEMINI_API_KEY not configured. Set it in Streamlit secrets."
            ).args[0]
        
        try:
            # Make a very simple request to verify API key and connectivity
            response = requests.post(
                f"{self.GEMINI_API_URL}?key={api_key}",
                json={
                    "contents": [{"parts": [{"text": "hi"}]}],
                    "generationConfig": {"maxOutputTokens": 10},
                },
                timeout=(5, 15),
            )
            if response.status_code == 200:
                return True, "Gemini connected"
            return False, LLMUnavailableError(f"HTTP {response.status_code}").args[0]
        except requests.exceptions.Timeout as exc:
            return False, LLMTimeoutError(str(exc)).args[0]
        except Exception as exc:
            return False, LLMConnectionError(str(exc)).args[0]

    def build_context_bundle(self, user_id: str = "default") -> dict[str, Any]:
        output_dir = self._output_dir(user_id)
        user_data_dir = self._user_data_dir(user_id)
        analysis = self._read_json(output_dir / "analysis_results.json")
        audit = self._read_json(output_dir / "audit_report.json")
        optimizer = self._read_json(output_dir / "optimizer_report.json")
        quality = self._read_json(user_data_dir / "processed" / "quality" / "quality_report.json")
        latest_session = self._latest_session_summary()

        results = analysis.get("results", {}) if isinstance(analysis, dict) else {}
        gov = results.get("governance_report", {}) if isinstance(results, dict) else {}
        oos = (gov.get("out_of_sample") or {}).get("r2") if isinstance(gov, dict) else None
        risk = gov.get("model_risk_score") if isinstance(gov, dict) else None

        return {
            "generated_at": datetime.now().isoformat(),
            "user_id": self._safe_user_id(user_id),
            "result_keys": analysis.get("result_keys", []),
            "analysis_results": analysis,
            "audit_report": audit,
            "optimizer_report": optimizer,
            "quality_report": quality,
            "latest_session_summary": latest_session,
            "quick_signals": {
                "oos_r2": oos,
                "model_risk_score": risk,
                "audit_status": audit.get("status") if isinstance(audit, dict) else None,
                "quality_missing_sources": (
                    (quality.get("summary") or {}).get("missing_sources", [])
                    if isinstance(quality, dict)
                    else []
                ),
            },
        }

    def _llama_generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Send prompt to Ollama and return the response text.

        Raises:
            LLMConnectionError: Cannot reach the Ollama endpoint.
            LLMTimeoutError: Request exceeded configured timeout.
            LLMUnavailableError: Ollama returned a non-200 HTTP status.
            LLMResponseError: Response payload is malformed.
        """
        try:
            response = requests.post(
                self.OLLAMA_API_URL,
                json={
                    "model": self.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                },
                timeout=self.timeout_seconds,
            )
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(
                f"Ollama request timed out after {self.timeout_seconds}s: {exc}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.OLLAMA_API_URL}: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMConnectionError(f"Ollama request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMUnavailableError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            payload = response.json()
        except Exception as exc:
            raise LLMResponseError(f"Failed to parse Ollama JSON response: {exc}") from exc

        return str(payload.get("response", "")).strip()

    def _generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Unified generation method that dispatches to either Llama or Gemini backend.
        
        Args:
            prompt: The input prompt to send to the LLM.
            temperature: Sampling temperature (0.0 to 1.0).
        
        Returns:
            The generated text response.
        
        Raises:
            Same exceptions as _llama_generate() or _gemini_generate() depending on backend.
        """
        if self.backend == "local":
            return self._llama_generate(prompt, temperature)
        elif self.backend == "online":
            return self._gemini_generate(prompt, temperature)
        else:
            raise BackendSelectionError(f"Unknown backend: {self.backend}")

    def answer_question(
        self,
        question: str,
        user_id: str = "default",
        current_page: str = "",
    ) -> dict[str, Any]:
        context_bundle = self.build_context_bundle(user_id=user_id)
        if current_page:
            context_bundle["user_current_page"] = current_page
        page_hint = (
            f"The user is currently viewing the '{current_page}' section of the app.\n\n"
            if current_page
            else ""
        )
        prompt = (
            "You are Quantos, the in-app quantitative copilot for Scenario Planner. "
            "Answer in clear Greek with short sections and practical next steps. "
            "Use ONLY the provided project context. "
            "If evidence is missing from the context, say so explicitly.\n\n"
            f"{page_hint}"
            "Focus areas:\n"
            "1) Pipeline and data quality\n"
            "2) Analyses and governance\n"
            "3) Optimizer and Llama workflow\n"
            "4) Concrete action items\n\n"
            f"CONTEXT_JSON:\n{json.dumps(context_bundle, ensure_ascii=False, indent=2)}\n\n"
            f"USER_QUESTION:\n{question}\n"
        )
        try:
            answer = self._generate(prompt, temperature=0.15)
            return {
                "success": True,
                "answer": answer,
                "context_generated_at": context_bundle.get("generated_at"),
            }
        except (LLMConnectionError, LLMTimeoutError, LLMUnavailableError, LLMResponseError) as exc:
            return {"success": False, "answer": self._friendly_user_error(exc)}

    def quick_insight(
        self,
        topic: str,
        snapshot: dict[str, Any],
        user_id: str = "default",
    ) -> dict[str, Any]:
        """Generate a fast focused 2-3 sentence inline insight in Greek for a given topic."""
        prompt = (
            "Είσαι ο Quantos, ο ποσοτικός AI βοηθός του Scenario Planner. "
            "Δώσε ΣΥΝΟΠΤΙΚΗ ερμηνεία 2-3 προτάσεων στα Ελληνικά. "
            "Να είσαι άμεσος και τεκμηριωμένος, χωρίς εισαγωγές.\n\n"
            f"ΘΕΜΑ: {topic}\n\n"
            f"ΔΕΔΟΜΕΝΑ:\n{json.dumps(snapshot, ensure_ascii=False)[:1800]}"
        )
        try:
            answer = self._generate(prompt, temperature=0.1)
            return {"success": True, "insight": answer}
        except (LLMConnectionError, LLMTimeoutError, LLMUnavailableError, LLMResponseError) as exc:
            return {"success": False, "insight": self._friendly_user_error(exc)}

    def create_pipeline_brief(self, user_id: str = "default") -> dict[str, Any]:
        context_bundle = self.build_context_bundle(user_id=user_id)
        prompt = (
            "You are a senior quant reviewer. Generate a concise execution brief after a pipeline run.\n"
            "Return exactly these sections:\n"
            "1) Run Health\n"
            "2) Key Signals\n"
            "3) Risks & Gaps\n"
            "4) Recommended Next Actions\n"
            "Keep each section short and evidence-based.\n\n"
            f"CONTEXT_JSON:\n{json.dumps(context_bundle, ensure_ascii=False, indent=2)}"
        )
        try:
            brief_text = self._generate(prompt, temperature=0.1)
        except (LLMConnectionError, LLMTimeoutError, LLMUnavailableError, LLMResponseError) as exc:
            return {"success": False, "error": f"{type(exc).__name__}: {exc}", "brief": ""}

        output_dir = self._output_dir(user_id)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            model_name = self.MODEL_NAME if self.backend == "local" else "gemini-1.5-flash"
            brief_payload = {
                "generated_at": datetime.now().isoformat(),
                "backend": self.backend,
                "model": model_name,
                "brief": brief_text,
                "quick_signals": context_bundle.get("quick_signals", {}),
            }
            json_path = output_dir / "ai_pipeline_briefing.json"
            md_path = output_dir / "ai_pipeline_briefing.md"
            json_path.write_text(json.dumps(brief_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            md_path.write_text(
                "# AI Pipeline Briefing\n\n"
                f"Generated at: {brief_payload['generated_at']}\n"
                f"Backend: {self.backend.upper()}\n"
                f"Model: {model_name}\n\n"
                f"{brief_text}\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise AIOutputError(f"Failed to write pipeline briefing to {output_dir}: {exc}") from exc

        return {
            "success": True,
            "brief": brief_text,
            "json_path": str(json_path),
            "md_path": str(md_path),
        }


def generate_ai_pipeline_brief(user_id: str = "default") -> dict[str, Any]:
    """Generate a pipeline brief using the automatically detected AI backend."""
    try:
        agent = QuantosAgent()
    except BackendSelectionError as exc:
        return {"success": False, "error": str(exc)}
    
    ok, reason = agent.ping()
    if not ok:
        return {"success": False, "error": f"AI backend not available: {reason}"}
    return agent.create_pipeline_brief(user_id=user_id)


# Backward-compatible alias for existing imports/tests.
ScenarioAIAgent = QuantosAgent
