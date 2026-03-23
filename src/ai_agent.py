from __future__ import annotations

# Copyright (c) 2026 EvangelGK. All Rights Reserved.

import json
import os
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

# ── Module-level TTL caches (survive across Streamlit reruns in same process) ──
# Avoids re-reading large JSON files and re-globbing logs on every question.
_context_bundle_cache: dict[str, dict[str, Any]] = {}
_CONTEXT_BUNDLE_TTL: float = 30.0  # seconds
_MAX_CONTEXT_BUNDLE_CACHE_ENTRIES: int = 32
_session_summary_cache: dict[str, dict[str, Any]] = {}
_SESSION_SUMMARY_TTL: float = 60.0  # seconds

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

    # Local backend (Ollama) — static fallback defaults
    OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
    MODEL_NAME = "llama3.2:1b"

    # Online backend (Google Gemini)
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    # Online backend fallback (Groq — OpenAI-compatible)
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL_NAME = "llama-3.3-70b-versatile"  # static fallback default

    RESTING_MESSAGE = "Quantos is currently resting, please try again later."

    def __init__(self, root: Path | None = None, timeout_seconds: int | None = None, backend: str | None = None) -> None:
        bootstrap_env_from_secrets(override=False)
        self._root = root or Path(__file__).resolve().parent.parent
        # Re-read dynamic config fresh on each construction so secret changes are
        # picked up whenever the agent is rebuilt (e.g. after a backend failure).
        self.OLLAMA_API_URL = (
            get_secret("OLLAMA_API_URL", self.__class__.OLLAMA_API_URL)
            or self.__class__.OLLAMA_API_URL
        )
        self.MODEL_NAME = (
            get_secret("OLLAMA_MODEL", self.__class__.MODEL_NAME)
            or self.__class__.MODEL_NAME
        )
        self.GROQ_MODEL_NAME = (
            get_secret("GROQ_MODEL", self.__class__.GROQ_MODEL_NAME)
            or self.__class__.GROQ_MODEL_NAME
        )
        # Default 200s; override with OLLAMA_TIMEOUT env var
        default_timeout = int(get_secret("OLLAMA_TIMEOUT", "200") or "200")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else default_timeout
        
        # Detect backend if not specified
        if backend is None:
            self.backend = self._detect_backend()
        else:
            self.backend = backend

        self.last_used_backend = self.backend
        self.last_used_model = ""
        
        if self.backend not in ("local", "online"):
            raise BackendSelectionError(
                f"Invalid backend '{backend}'. Must be 'local' or 'online'."
            )

    def _detect_backend(self) -> str:
        """Detect which backend is available: 'local' (Ollama) or 'online' (Gemini/Groq).

        Performance: checks online API keys first (O(1)) to avoid blocking
        network probes. Ollama probe is only done when no online key is present.
        """
        # Fast path: any online key present → go online immediately, no network probe
        if self._get_gemini_api_key() or self._get_groq_api_key():
            return "online"

        # Ollama probe — very short timeout so UI never blocks > ~1s
        try:
            response = requests.get(
                f"{self._base_url()}/api/tags",
                timeout=(1.0, 1.5),
            )
            if response.status_code == 200:
                return "local"
        except Exception:
            pass

        raise BackendSelectionError(
            "No AI backend available. Either:\n"
            "1) Ensure Ollama is running at 127.0.0.1:11434, OR\n"
            "2) Set GEMINI_API_KEY or GROQ_API_KEY in Streamlit secrets or environment."
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

    def _get_groq_api_key(self) -> str | None:
        """Get Groq API key from the unified secret store."""
        return get_secret("GROQ_API_KEY")

    def _friendly_user_error(self, exc: Exception) -> str:
        txt = str(exc).lower()
        if any(token in txt for token in ("quota", "resource_exhausted", "rate limit", "429")):
            return self.RESTING_MESSAGE
        detail = str(exc).strip()
        if detail:
            safe_detail = " ".join(detail.split())[:260]
            return f"Quantos is temporarily unavailable: {safe_detail}"
        return "Quantos is temporarily unavailable. Please try again later."

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
                        "maxOutputTokens": 4096,
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
                    self.last_used_backend = "online"
                    self.last_used_model = "gemini-1.5-flash"
                    return str(parts[0].get("text", "")).strip()
        except (KeyError, IndexError, TypeError):
            pass

        raise LLMResponseError(f"Unexpected Gemini response format: {json.dumps(payload)[:200]}")

    def _groq_generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Send prompt to Groq (OpenAI-compatible) and return the response text.

        Raises:
            LLMAuthenticationError: Groq API key is missing or invalid.
            LLMConnectionError: Cannot reach the Groq endpoint.
            LLMTimeoutError: Request exceeded configured timeout.
            LLMUnavailableError: Groq returned a non-200 HTTP status.
            LLMResponseError: Response payload is malformed.
        """
        api_key = self._get_groq_api_key()
        if not api_key:
            raise LLMAuthenticationError(
                "GROQ_API_KEY not found in Streamlit secrets or environment. "
                "Set it via: st.secrets['GROQ_API_KEY'] = 'gsk_...'"
            )

        try:
            response = requests.post(
                self.GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.GROQ_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 4096,
                },
                timeout=self.timeout_seconds,
            )
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(
                f"Groq request timed out after {self.timeout_seconds}s: {exc}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Cannot connect to Groq API: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMConnectionError(f"Groq request failed: {exc}") from exc

        if response.status_code == 429:
            raise LLMUnavailableError(self.RESTING_MESSAGE)
        if response.status_code != 200:
            body = response.text[:200].lower()
            if "rate_limit" in body or "quota" in body:
                raise LLMUnavailableError(self.RESTING_MESSAGE)
            raise LLMUnavailableError(
                f"Groq returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            payload = response.json()
        except Exception as exc:
            raise LLMResponseError(f"Failed to parse Groq JSON response: {exc}") from exc

        try:
            self.last_used_backend = "online"
            self.last_used_model = f"groq/{self.GROQ_MODEL_NAME}"
            return str(payload["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError):
            raise LLMResponseError(f"Unexpected Groq response format: {json.dumps(payload)[:200]}")

    @staticmethod
    def _safe_user_id(user_id: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(user_id))
        return cleaned or "default"


    def _output_dir(self, user_id: str) -> Path:
        return self._root / "output" / self._safe_user_id(user_id)

    def _user_data_dir(self, user_id: str) -> Path:
        return self._root / "data" / "users" / self._safe_user_id(user_id)

    @staticmethod
    def _file_signature(path: Path) -> tuple[str, int, int, bool]:
        """Return a stable change signature for cache invalidation."""
        try:
            stat = path.stat()
            return (str(path), stat.st_mtime_ns, stat.st_size, True)
        except OSError:
            return (str(path), 0, 0, False)

    def _evict_old_context_cache_entries(self) -> None:
        """Keep the per-user context cache bounded in long-lived processes."""
        if len(_context_bundle_cache) <= _MAX_CONTEXT_BUNDLE_CACHE_ENTRIES:
            return
        oldest_key = min(
            _context_bundle_cache,
            key=lambda key: float(_context_bundle_cache[key].get("_cached_at", 0.0)),
        )
        _context_bundle_cache.pop(oldest_key, None)

    @staticmethod
    def _json_scalar(value: Any) -> bool:
        return isinstance(value, (str, int, float, bool)) or value is None

    def _output_inventory(self, output_dir: Path, max_files: int = 40) -> list[dict[str, Any]]:
        if not output_dir.exists():
            return []
        files = sorted(
            [p for p in output_dir.rglob("*") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:max_files]
        inventory: list[dict[str, Any]] = []
        for file_path in files:
            try:
                stat = file_path.stat()
                inventory.append(
                    {
                        "path": str(file_path.relative_to(output_dir)).replace("\\", "/"),
                        "size_bytes": int(stat.st_size),
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                )
            except OSError:
                continue
        return inventory

    def _analysis_digest(self, analysis_payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(analysis_payload, dict):
            return {}
        results = analysis_payload.get("results") or {}
        if not isinstance(results, dict):
            return {}

        digest: dict[str, Any] = {}
        metric_priority = (
            "status",
            "model",
            "best_model",
            "r2",
            "cv_r2",
            "holdout_r2",
            "mae",
            "rmse",
            "mape",
            "n_obs",
            "model_risk_score",
            "decision",
            "passed",
            "severity",
        )

        for section, value in results.items():
            if isinstance(value, dict):
                section_digest: dict[str, Any] = {}
                for key in metric_priority:
                    if key in value and self._json_scalar(value.get(key)):
                        section_digest[key] = value.get(key)

                if not section_digest:
                    scalar_items = [
                        (k, v) for k, v in value.items() if self._json_scalar(v)
                    ][:12]
                    section_digest = {k: v for k, v in scalar_items}

                if "out_of_sample" in value and isinstance(value.get("out_of_sample"), dict):
                    oos = value.get("out_of_sample") or {}
                    oos_slim = {
                        k: oos.get(k)
                        for k in ("r2", "mae", "rmse")
                        if self._json_scalar(oos.get(k))
                    }
                    if oos_slim:
                        section_digest["out_of_sample"] = oos_slim

                if "walk_forward" in value and isinstance(value.get("walk_forward"), dict):
                    wf = value.get("walk_forward") or {}
                    wf_slim = {
                        k: wf.get(k)
                        for k in ("status", "avg_r2", "median_r2", "worst_r2")
                        if self._json_scalar(wf.get(k))
                    }
                    if wf_slim:
                        section_digest["walk_forward"] = wf_slim

                digest[section] = section_digest
            elif self._json_scalar(value):
                digest[section] = value
            else:
                digest[section] = {"type": type(value).__name__}

        return digest

    def _latest_session_summary(self) -> dict[str, Any]:
        """Return latest session summary, cached for _SESSION_SUMMARY_TTL seconds."""
        now = _time.monotonic()
        logs_dir = self._root / "logs"
        cache_key = str(logs_dir)
        cached_entry = _session_summary_cache.get(cache_key)
        latest_file_signature: tuple[str, int, int, bool] | None = None

        if logs_dir.exists():
            files = sorted(
                logs_dir.glob("session_summary_*.json"),
                key=lambda p: p.stat().st_mtime,
            )
            if files:
                latest_file_signature = self._file_signature(files[-1])

        if (
            cached_entry is not None
            and (now - float(cached_entry.get("ts", 0.0))) < _SESSION_SUMMARY_TTL
            and cached_entry.get("file_signature") == latest_file_signature
        ):
            return cached_entry.get("data", {})  # type: ignore[return-value]

        result: dict[str, Any] = {}
        if latest_file_signature and latest_file_signature[-1]:
            result = self._read_json(Path(latest_file_signature[0]))

        _session_summary_cache[cache_key] = {
            "data": result,
            "ts": now,
            "file_signature": latest_file_signature,
        }
        return result

    def _read_json(self, path: Path) -> dict[str, Any]:
        data, _ = self._read_json_with_error(path)
        return data

    def _read_json_with_error(
        self,
        path: Path,
        required: bool = True,
    ) -> tuple[dict[str, Any], str | None]:
        if not path.exists():
            if required:
                return {}, f"Missing file: {path.name}"
            return {}, None
        try:
            return json.loads(path.read_text(encoding="utf-8")), None
        except Exception as exc:
            if required:
                return {}, f"Failed to read {path.name}: {type(exc).__name__}: {exc}"
            return {}, None

    @staticmethod
    def _has_minimum_context(bundle: dict[str, Any]) -> bool:
        """Return True when at least one primary context source contains data."""
        primary_keys = ("analysis_results", "audit_report", "quality_report")
        return any(bool(bundle.get(key)) for key in primary_keys)

    def ping(self) -> tuple[bool, str]:
        """Check AI backend availability. For 'online', tries Gemini, Groq, then Ollama."""
        if self.backend == "local":
            return self._ping_ollama()
        elif self.backend == "online":
            if self._get_gemini_api_key():
                ok, msg = self._ping_gemini()
                if ok:
                    return True, msg
            if self._get_groq_api_key():
                ok, msg = self._ping_groq()
                if ok:
                    return True, msg
            ok, msg = self._ping_ollama()
            if ok:
                return True, msg
            return False, "No AI backend available after checking Gemini, Groq, and Ollama."
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

    def _ping_groq(self) -> tuple[bool, str]:
        """Check Groq API availability."""
        api_key = self._get_groq_api_key()
        if not api_key:
            return False, LLMAuthenticationError(
                "GROQ_API_KEY not configured. Set it in Streamlit secrets."
            ).args[0]

        try:
            response = requests.post(
                self.GROQ_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.GROQ_MODEL_NAME,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
                timeout=(5, 15),
            )
            if response.status_code == 200:
                return True, f"Groq connected ({self.GROQ_MODEL_NAME})"
            return False, LLMUnavailableError(f"HTTP {response.status_code}").args[0]
        except requests.exceptions.Timeout as exc:
            return False, LLMTimeoutError(str(exc)).args[0]
        except Exception as exc:
            return False, LLMConnectionError(str(exc)).args[0]

    def build_context_bundle(self, user_id: str = "default") -> dict[str, Any]:
        """Build full context bundle, cached per-user for _CONTEXT_BUNDLE_TTL seconds."""
        safe_uid = self._safe_user_id(user_id)
        output_dir = self._output_dir(user_id)
        user_data_dir = self._user_data_dir(user_id)
        source_paths = [
            output_dir / "analysis_results.json",
            output_dir / "audit_report.json",
            output_dir / "optimizer_report.json",
            user_data_dir / "processed" / "quality" / "quality_report.json",
        ]
        source_signature = tuple(self._file_signature(path) for path in source_paths)
        now = _time.monotonic()
        cached = _context_bundle_cache.get(safe_uid)
        if (
            cached
            and (now - float(cached.get("_cached_at", 0.0))) < _CONTEXT_BUNDLE_TTL
            and cached.get("_source_signature") == source_signature
        ):
            return cached

        analysis, analysis_error = self._read_json_with_error(output_dir / "analysis_results.json")
        audit, audit_error = self._read_json_with_error(output_dir / "audit_report.json")
        optimizer, optimizer_error = self._read_json_with_error(
            output_dir / "optimizer_report.json",
            required=False,
        )
        quality, quality_error = self._read_json_with_error(user_data_dir / "processed" / "quality" / "quality_report.json")
        latest_session = self._latest_session_summary()
        read_errors = [err for err in (analysis_error, audit_error, optimizer_error, quality_error) if err]
        analysis_digest = self._analysis_digest(analysis if isinstance(analysis, dict) else {})
        output_inventory = self._output_inventory(output_dir)

        results = analysis.get("results", {}) if isinstance(analysis, dict) else {}
        gov = results.get("governance_report", {}) if isinstance(results, dict) else {}
        oos = (gov.get("out_of_sample") or {}).get("r2") if isinstance(gov, dict) else None
        risk = gov.get("model_risk_score") if isinstance(gov, dict) else None

        bundle: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "user_id": self._safe_user_id(user_id),
            "result_keys": analysis.get("result_keys", []),
            "analysis_results": analysis,
            "audit_report": audit,
            "optimizer_report": optimizer,
            "quality_report": quality,
            "latest_session_summary": latest_session,
            "analysis_digest": analysis_digest,
            "pipeline_visibility": {
                "output_inventory": output_inventory,
                "available_result_sections": sorted(list(results.keys())) if isinstance(results, dict) else [],
                "output_dir": str(output_dir),
            },
            "agent_policy": {
                "mode": "read_only",
                "allowed_actions": ["inspect", "summarize", "recommend"],
                "forbidden_actions": ["edit_code", "modify_config", "run_pipeline", "execute_optimizer"],
            },
            "read_errors": read_errors,
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
            "_source_signature": source_signature,
            "_cached_at": now,
        }
        _context_bundle_cache[safe_uid] = bundle
        self._evict_old_context_cache_entries()
        return bundle

    def _lean_prompt_context(self, bundle: dict[str, Any]) -> dict[str, Any]:
        """Build a compact context dict for LLM prompts.

        Strips full JSON files and keeps only the fields the LLM needs.
        Reduces LLM input from potentially megabytes to a few KB.
        """
        audit = bundle.get("audit_report", {}) or {}
        latest = bundle.get("latest_session_summary", {}) or {}
        optimizer = bundle.get("optimizer_report", {}) or {}
        quality = bundle.get("quality_report", {}) or {}
        analysis = bundle.get("analysis_results", {}) or {}
        analysis_digest = bundle.get("analysis_digest", {}) or {}
        pipeline_visibility = bundle.get("pipeline_visibility", {}) or {}
        agent_policy = bundle.get("agent_policy", {}) or {}

        # All failed audit checks + summary of passed ones count
        raw_checks = audit.get("checks") or []
        all_checks = [c for c in raw_checks if isinstance(c, dict)]
        failed_checks = [
            {
                "name": c.get("name"),
                "reason": c.get("reason"),
                "severity": c.get("severity"),
            }
            for c in all_checks
            if not c.get("passed", True)
        ]
        passed_count = sum(1 for c in all_checks if c.get("passed", True))

        # Quality: only summary block
        q_summary = (quality.get("summary") or {}) if isinstance(quality, dict) else {}

        # Session: only lightweight status keys
        session_slim = {k: latest[k] for k in ("status", "duration_seconds", "errors", "warnings", "stages_completed") if k in latest}

        # Optimizer: only final score/status
        opt_slim = {k: optimizer[k] for k in ("final_score", "status", "iterations", "best_iteration") if k in optimizer} if isinstance(optimizer, dict) else {}

        # Analysis: only governance top-level and result_keys (omit raw arrays)
        results = analysis.get("results", {}) if isinstance(analysis, dict) else {}
        gov_report = (results.get("governance_report") or {}) if isinstance(results, dict) else {}
        gov_slim = {k: gov_report[k] for k in ("decision", "passed", "severity", "model_risk_score", "out_of_sample") if k in gov_report}

        return {
            "generated_at": bundle.get("generated_at"),
            "result_keys": bundle.get("result_keys", []),
            "quick_signals": bundle.get("quick_signals", {}),
            "read_errors": bundle.get("read_errors", []),
            "agent_policy": agent_policy,
            "audit": {
                "status": audit.get("status"),
                "model_risk_score": audit.get("model_risk_score"),
                "overall_score": audit.get("overall_score"),
                "failed_checks": failed_checks,
                "passed_checks_count": passed_count,
                "total_checks": len(all_checks),
            },
            "governance": gov_slim,
            "quality_summary": q_summary,
            "session": session_slim,
            "optimizer": opt_slim,
            "analysis_digest": analysis_digest,
            "pipeline_visibility": {
                "available_result_sections": pipeline_visibility.get("available_result_sections", []),
                "output_inventory": pipeline_visibility.get("output_inventory", [])[:25],
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

        self.last_used_backend = "local"
        self.last_used_model = self.MODEL_NAME
        return str(payload.get("response", "")).strip()

    def _generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Unified generation method: Gemini → Groq → Ollama → raise.

        For the 'online' backend, tries Gemini first. If Gemini is unavailable
        (quota, outage, auth failure) it falls through to Groq automatically,
        then to Ollama if a local backend is available.
        Only raises to the caller when all candidate backends have been exhausted.

        Args:
            prompt: The input prompt to send to the LLM.
            temperature: Sampling temperature (0.0 to 1.0).

        Returns:
            The generated text response.

        Raises:
            Same exceptions as the underlying backend methods.
        """
        if self.backend == "local":
            return self._llama_generate(prompt, temperature)
        elif self.backend == "online":
            last_exc: Exception | None = None

            # 1️⃣  Try Gemini
            if self._get_gemini_api_key():
                try:
                    return self._gemini_generate(prompt, temperature)
                except (
                    LLMAuthenticationError,
                    LLMUnavailableError,
                    LLMConnectionError,
                    LLMTimeoutError,
                    LLMResponseError,
                ) as exc:
                    last_exc = exc  # Gemini failed — fall through to Groq

            # 2️⃣  Try Groq as fallback
            if self._get_groq_api_key():
                try:
                    return self._groq_generate(prompt, temperature)
                except (
                    LLMAuthenticationError,
                    LLMUnavailableError,
                    LLMConnectionError,
                    LLMTimeoutError,
                    LLMResponseError,
                ) as exc:
                    last_exc = exc

            # 3) Try local Ollama as final fallback if it is currently available
            ok, _ = self._ping_ollama()
            if ok:
                try:
                    return self._llama_generate(prompt, temperature)
                except (
                    LLMUnavailableError,
                    LLMConnectionError,
                    LLMTimeoutError,
                    LLMResponseError,
                ) as exc:
                    last_exc = exc

            # Both exhausted — surface the last known error
            raise last_exc or LLMUnavailableError(
                "No AI backend available. Set GEMINI_API_KEY or GROQ_API_KEY in secrets, or run Ollama locally."
            )
        else:
            raise BackendSelectionError(f"Unknown backend: {self.backend}")

    def answer_question(
        self,
        question: str,
        user_id: str = "default",
        current_page: str = "",
    ) -> dict[str, Any]:
        context_bundle = self.build_context_bundle(user_id=user_id)
        # Continue with partial context so Quantos can still help even when some
        # outputs are missing; missing files are already surfaced via read_errors.
        lean = self._lean_prompt_context(context_bundle)
        if current_page:
            lean["user_current_page"] = current_page
        page_hint = (
            f"The user is viewing the '{current_page}' section.\n"
            if current_page
            else ""
        )
        prompt = (
            "You are Quantos, the in-app quantitative copilot for STRATUM QUANT ANALYTICS. "
            "Answer in detailed, structured English by default. Use sections with headers (##) and bullet points where appropriate. "
            "Give complete, practical explanations and concrete next steps — never truncate or summarise without evidence. "
            "You are in strict read-only mode: do NOT claim to have edited code, changed configuration, executed the pipeline, or executed optimizer actions. "
            "You may only inspect context, explain findings, and propose recommendations. "
            "Use ONLY the provided context. If data for a specific question is missing from the context, say so clearly but still answer what you can.\n"
            f"{page_hint}"
            f"CONTEXT:{json.dumps(lean, ensure_ascii=False)}\n"
            f"QUESTION:{question}"
        )
        try:
            answer = self._generate(prompt, temperature=0.15)
            return {
                "success": True,
                "answer": answer,
                "context_generated_at": context_bundle.get("generated_at"),
            }
        except (LLMAuthenticationError, LLMConnectionError, LLMTimeoutError, LLMUnavailableError, LLMResponseError) as exc:
            return {"success": False, "answer": self._friendly_user_error(exc)}

    def quick_insight(
        self,
        topic: str,
        snapshot: dict[str, Any],
        user_id: str = "default",
    ) -> dict[str, Any]:
        """Generate a contextual inline insight in English for a given topic."""
        prompt = (
            "You are Quantos, the quantitative AI copilot of STRATUM QUANT ANALYTICS. "
            "Provide a complete, practical interpretation in English with bullet points and concrete actions. "
            "Rely only on the provided data.\n"
            f"TOPIC: {topic}\n"
            f"DATA:{json.dumps(snapshot, ensure_ascii=False)[:3000]}"
        )
        try:
            answer = self._generate(prompt, temperature=0.1)
            return {"success": True, "insight": answer}
        except (LLMAuthenticationError, LLMConnectionError, LLMTimeoutError, LLMUnavailableError, LLMResponseError) as exc:
            return {"success": False, "insight": self._friendly_user_error(exc)}

    def create_pipeline_brief(self, user_id: str = "default") -> dict[str, Any]:
        context_bundle = self.build_context_bundle(user_id=user_id)
        lean = self._lean_prompt_context(context_bundle)
        prompt = (
            "You are a senior quant reviewer. Generate a concise execution brief after a pipeline run.\n"
            "Read-only policy: do not claim that you edited code/config or executed optimizer/pipeline actions.\n"
            "Return exactly these sections:\n"
            "1) Run Health\n"
            "2) Key Signals\n"
            "3) Risks & Gaps\n"
            "4) Recommended Next Actions\n"
            "Keep each section short and evidence-based.\n"
            f"CONTEXT:{json.dumps(lean, ensure_ascii=False)}"
        )
        try:
            brief_text = self._generate(prompt, temperature=0.1)
        except (LLMAuthenticationError, LLMConnectionError, LLMTimeoutError, LLMUnavailableError, LLMResponseError) as exc:
            return {"success": False, "error": f"{type(exc).__name__}: {exc}", "brief": ""}

        output_dir = self._output_dir(user_id)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            model_name = self.last_used_model or (
                self.MODEL_NAME if self.backend == "local" else "gemini-1.5-flash"
            )
            brief_payload = {
                "generated_at": datetime.now().isoformat(),
                "backend": self.last_used_backend or self.backend,
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


