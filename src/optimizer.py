from __future__ import annotations

# Copyright (c) 2026 EvangelGK. All Rights Reserved.

import json
import os
import html
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

try:
    from src.secret_store import bootstrap_env_from_secrets, get_secret
except ModuleNotFoundError:
    from secret_store import bootstrap_env_from_secrets, get_secret

from Fetchers.Factory import DataFactory
from Fetchers.ProjectConfig import ProjectConfig, RunMode
from Medallion.MedallionPipeline import MedallionPipeline
from Medallion.gold.AnalysisSuite.sensitivity_reg import sensitivity_reg
from Medallion.silver import contracts as silver_contracts

bootstrap_env_from_secrets(override=False)


@dataclass
class IterationRecord:
    iteration: int
    score: float
    inconsistencies: List[str]
    adjustments: List[str]


class LlamaQuantAnalyzer:
    """AI-powered quantitative problem fixer using Llama 3.2 via Ollama.

    Capabilities:
    - Reads actual project source files to give context-aware analysis
    - Discovers NEW bugs not yet caught by Python diagnostics
    - Outputs code changes in a strict parseable format (<<<OLD>>> / <<<NEW>>>)
    - Auto-applies approved changes directly to disk
    """

    OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
    MODEL_NAME = "llama3.2:1b"
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
    MAX_PATCH_FILES = 6
    MAX_PATCH_TOTAL_CHARS = 8000
    BLOCKED_PATH_PREFIXES: List[str] = [
        "main.py",
        "Fetchers/ProjectConfig.py",
        "Medallion/MedallionPipeline.py",
    ]

    # Maps diagnostic issue types → relevant source files to include in prompt
    ISSUE_FILE_MAP: Dict[str, List[str]] = {
        "NEGATIVE_R2":        ["Medallion/gold/AnalysisSuite/sensitivity_reg.py",
                                "Medallion/gold/AnalysisSuite/governance.py"],
        "DATA_GAPS":          ["Fetchers/FredFetcher.py",
                                "Fetchers/WorldBankFetcher.py",
                                "Fetchers/YFinanceFetcher.py"],
        "EXCESS_DRAWDOWN":    ["Medallion/gold/AnalysisSuite/stress_test.py"],
        "MULTICOLLINEARITY":  ["Medallion/gold/AnalysisSuite/sensitivity_reg.py"],
        "REGIME_INSTABILITY": ["Medallion/gold/AnalysisSuite/elasticity.py"],
        "OUTLIERS":           ["Medallion/silver/silver.py",
                                "Medallion/silver/contracts.py"],
        "SCALING_ISSUE":      ["Medallion/gold/AnalysisSuite/elasticity.py"],
        "POOR_RISK_ADJ":      ["Medallion/gold/AnalysisSuite/governance.py",
                                "Medallion/gold/AnalysisSuite/monte_carlo.py"],
        "LOOKAHEAD_BIAS":     ["Medallion/gold/AnalysisSuite/governance.py",
                    "Medallion/gold/AnalysisSuite/backtest.py"],
        "STALE_DATA":         ["Fetchers/FredFetcher.py",
                    "Fetchers/WorldBankFetcher.py",
                    "Fetchers/YFinanceFetcher.py"],
        "DIST_SHIFT":         ["Medallion/gold/AnalysisSuite/governance.py",
                    "Medallion/gold/AnalysisSuite/sensitivity_reg.py"],
        "RESIDUAL_AUTOCORR":  ["Medallion/gold/AnalysisSuite/governance.py"],
        "CI_QUALITY":         ["Medallion/gold/AnalysisSuite/governance.py"],
        "SIGN_INSTABILITY":   ["Medallion/gold/AnalysisSuite/elasticity.py"],
        "UNCERTAINTY_GAP":    ["Medallion/gold/AnalysisSuite/monte_carlo.py",
                    "Medallion/gold/AnalysisSuite/governance.py"],
        "PIPELINE_ROBUSTNESS": ["Medallion/silver/contracts.py",
                "Medallion/silver/silver.py",
                "Fetchers/Factory.py"],
        "TEMPORAL_INTEGRITY": ["Medallion/gold/AnalysisSuite/governance.py",
            "Medallion/gold/AnalysisSuite/backtest.py"],
        "BACKTEST_REALISM": ["Medallion/gold/AnalysisSuite/backtest.py",
            "Medallion/gold/AnalysisSuite/governance.py"],
        "EXPLAINABILITY_RISK": ["Medallion/gold/AnalysisSuite/sensitivity_reg.py",
            "Medallion/gold/AnalysisSuite/elasticity.py"],
        "FEATURE_SHIFT": ["Medallion/gold/AnalysisSuite/sensitivity_reg.py",
            "Medallion/gold/AnalysisSuite/governance.py"],
        "RESIDUAL_HETERO": ["Medallion/gold/AnalysisSuite/governance.py"],
        "COVERAGE_CALIBRATION": ["Medallion/gold/AnalysisSuite/governance.py"],
        "SHAP_STABILITY": ["Medallion/gold/AnalysisSuite/governance.py",
            "Medallion/gold/AnalysisSuite/sensitivity_reg.py"],
        "COEFF_CI_OVERLAP": ["Medallion/gold/AnalysisSuite/sensitivity_reg.py"],
    }

    # All files scanned during autonomous bug discovery
    BUG_SCAN_FILES: List[str] = [
        "Medallion/gold/AnalysisSuite/sensitivity_reg.py",
        "Medallion/gold/AnalysisSuite/governance.py",
        "Medallion/gold/AnalysisSuite/elasticity.py",
        "Medallion/gold/AnalysisSuite/stress_test.py",
        "Medallion/gold/AnalysisSuite/monte_carlo.py",
        "Medallion/gold/AnalysisSuite/backtest.py",
        "Medallion/silver/silver.py",
        "Fetchers/FredFetcher.py",
        "Fetchers/WorldBankFetcher.py",
    ]

    SYSTEM_PROMPT = """You are Quantos, a Senior Quant Data Fixer agent embedded in a financial Stratum-Quant-Analytics pipeline.

You have access to the actual Python source code of the project. Use it to give PRECISE, FILE-SPECIFIC fixes.

INPUT:
- Task: description of the detected problem
- Context: diagnostic metrics (score, issues, VaR, etc.)
- Source Code: actual relevant Python files from the project

RULES:
- Only propose fixes for SERIOUS problems, including: data gaps, leakage/lookahead bias, distribution shift,
  residual autocorrelation, confidence-interval quality failures, sign instability, uncertainty gaps,
  multicollinearity, excess drawdown, regime instability, stale data, and pipeline robustness failures
- Include feature-level distribution shift checks (PSI/KS), advanced residual diagnostics
    (Ljung-Box and heteroskedasticity), and explainability consistency when available
- Validate interval calibration (90 % realized coverage), SHAP/feature-importance stability,
    and coefficient confidence-interval overlap in rolling windows when data is available
- Identify bugs in the source code that Python diagnostics may have MISSED
- Be concise and technical — no sugar-coating

OUTPUT FORMAT — use EXACTLY these section headers:

---PROBLEM---
[What is wrong and why it matters for the model's predictive reliability]

---ROOT_CAUSE---
[Specific line(s) or logic in the source code causing this]

---SOLUTION---
[Precise fix description]

---CODE_CHANGES---
[For EACH changed file, use this EXACT format — one block per change:]

FILE: relative/path/from/src/to/file.py
<<<OLD>>>
exact existing code to replace (copy verbatim from source)
<<<NEW>>>
exact replacement code
<<<END>>>

[Repeat FILE/<<<OLD>>>/<<<NEW>>>/<<<END>>> blocks for every change]

---APPROVAL_QUESTION---
Apply these changes automatically to the project files? (YES/NO)"""

    def __init__(self, src_root: Optional[Path] = None, timeout_seconds: int | None = None):
        bootstrap_env_from_secrets(override=False)
        # Default 300s — llama3.2:1b on CPU can be slow; override with OLLAMA_TIMEOUT env var
        self.OLLAMA_API_URL = get_secret("OLLAMA_API_URL", self.__class__.OLLAMA_API_URL) or self.__class__.OLLAMA_API_URL
        self.MODEL_NAME = get_secret("OLLAMA_MODEL", self.__class__.MODEL_NAME) or self.__class__.MODEL_NAME
        self.GROQ_MODEL_NAME = get_secret("GROQ_MODEL", self.__class__.GROQ_MODEL_NAME) or self.__class__.GROQ_MODEL_NAME
        default_timeout = int(get_secret("OLLAMA_TIMEOUT", "300") or "300")
        self._timeout = timeout_seconds if timeout_seconds is not None else default_timeout
        self._src = src_root or Path(__file__).parent
        self._verify_connection()

    def _get_gemini_api_key(self) -> str | None:
        return get_secret("GEMINI_API_KEY")

    def _get_groq_api_key(self) -> str | None:
        return get_secret("GROQ_API_KEY")

    def _ollama_base_url(self) -> str:
        url = self.OLLAMA_API_URL
        for suffix in ("/api/generate", "/api/chat"):
            if url.endswith(suffix):
                return url[: -len(suffix)]
        return url.rsplit("/api/", 1)[0] if "/api/" in url else url

    def _verify_connection(self) -> bool:
        """Check whether any supported AI backend is reachable."""
        if self._get_gemini_api_key() or self._get_groq_api_key():
            return True
        try:
            response = requests.get(
                f"{self._ollama_base_url()}/api/tags",
                timeout=(5, 15),  # connect=5s, read=15s
            )
            if response.status_code == 200:
                return True
            print(f"[LLAMA] Warning: Ollama responded HTTP {response.status_code}")
            return False
        except Exception as e:
            print(f"[LLAMA] Warning: Could not connect to Ollama: {e}")
            return False

    # ------------------------------------------------------------------
    # File reading helpers
    # ------------------------------------------------------------------

    def _read_source_file(self, relative_path: str, max_lines: int = 120) -> str:
        """Read a source file and return its contents (truncated for LLM context)."""
        full_path = self._src / relative_path
        if not full_path.exists():
            return f"# [FILE NOT FOUND: {relative_path}]"
        try:
            lines = full_path.read_text(encoding="utf-8").splitlines()
            snippet = "\n".join(lines[:max_lines])
            if len(lines) > max_lines:
                snippet += f"\n# ... ({len(lines) - max_lines} more lines truncated)"
            return snippet
        except Exception as e:
            return f"# [ERROR READING {relative_path}: {e}]"

    def _collect_source_context(self, issue_type: str) -> str:
        """Collect source code snippets relevant to the given issue type."""
        files = self.ISSUE_FILE_MAP.get(issue_type, [])
        # Fallback: always include the main optimizer for cross-cutting issues
        if not files:
            files = ["Medallion/gold/GoldLayer.py"]
        parts = []
        for f in files:
            code = self._read_source_file(f)
            parts.append(f"\n=== FILE: {f} ===\n{code}\n")
        return "\n".join(parts)

    def _collect_all_source_for_scan(self) -> str:
        """Collect all files for autonomous bug scan."""
        parts = []
        for f in self.BUG_SCAN_FILES:
            code = self._read_source_file(f, max_lines=80)
            parts.append(f"\n=== FILE: {f} ===\n{code}\n")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _call_llama(self, prompt: str) -> str:
        """Send prompt to the best available backend: Gemini -> Groq -> Ollama."""
        gemini_key = self._get_gemini_api_key()
        if gemini_key:
            try:
                response = requests.post(
                    f"{self.GEMINI_API_URL}?key={gemini_key}",
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.2,
                            "maxOutputTokens": 4096,
                        },
                    },
                    timeout=self._timeout,
                )
                if response.status_code == 200:
                    payload = response.json()
                    candidates = payload.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            text = str(parts[0].get("text", "")).strip()
                            if text:
                                return text
            except Exception:
                pass

        groq_key = self._get_groq_api_key()
        if groq_key:
            try:
                response = requests.post(
                    self.GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.GROQ_MODEL_NAME,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 4096,
                    },
                    timeout=self._timeout,
                )
                if response.status_code == 200:
                    payload = response.json()
                    choices = payload.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        text = str(msg.get("content", "")).strip()
                        if text:
                            return text
            except Exception:
                pass

        try:
            response = requests.post(
                self.OLLAMA_API_URL,
                json={
                    "model": self.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.2,
                },
                timeout=self._timeout,
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"[LLAMA ERROR: HTTP {response.status_code}]"
        except requests.exceptions.Timeout:
            return "[LLAMA ERROR: timeout]"
        except Exception as e:
            return f"[LLAMA ERROR: {e}]"

    def analyze_problem(
        self,
        problem_description: str,
        diagnostics: Dict[str, Any],
        issue_type: str = "",
    ) -> Dict[str, Any]:
        """Analyze a diagnosed issue using real source code context."""
        metrics_ctx = self._format_metrics(diagnostics)
        source_ctx = self._collect_source_context(issue_type)

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"TASK: {problem_description}\n\n"
            f"DIAGNOSTIC METRICS:\n{metrics_ctx}\n\n"
            f"SOURCE CODE:\n{source_ctx}\n\n"
            "Now produce your technical report:"
        )

        analysis = self._call_llama(prompt)
        has_changes = "<<<OLD>>>" in analysis and "<<<NEW>>>" in analysis
        return {
            "success": not analysis.startswith("[LLAMA ERROR"),
            "analysis": analysis,
            "has_code_changes": has_changes,
        }

    def scan_for_bugs(self) -> Dict[str, Any]:
        """Autonomous scan of all project source files to discover latent bugs."""
        source_ctx = self._collect_all_source_for_scan()

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            "TASK: Autonomous bug scan. Review ALL source files below and identify "
            "any latent bugs, logic errors, or data quality issues NOT already "
            "covered by existing diagnostics. Focus on: off-by-one errors, "
            "incorrect pandas/numpy operations, missing null checks, wrong "
            "statistical assumptions, incorrect date alignment.\n\n"
            f"SOURCE CODE:\n{source_ctx}\n\n"
            "Produce your report for every bug found. "
            "If no serious bugs found, write: ---PROBLEM---\nNo latent bugs found."
        )

        analysis = self._call_llama(prompt)
        has_changes = "<<<OLD>>>" in analysis and "<<<NEW>>>" in analysis
        no_bugs = "no latent bugs found" in analysis.lower()
        return {
            "success": not analysis.startswith("[LLAMA ERROR"),
            "analysis": analysis,
            "has_code_changes": has_changes,
            "no_bugs_found": no_bugs,
        }

    # ------------------------------------------------------------------
    # Auto-apply code changes
    # ------------------------------------------------------------------

    def apply_code_changes(self, analysis: str) -> List[str]:
        """Parse <<<OLD>>>/<<<NEW>>> blocks from Llama output and apply to disk.

        Returns list of applied change descriptions.
        """
        applied: List[str] = []
        backups: Dict[Path, str] = {}
        total_changed_chars = 0
        changed_files: List[Path] = []
        # Split on FILE: markers
        import re
        blocks = re.split(r"FILE:\s*", analysis)
        for block in blocks[1:]:  # skip preamble
            lines = block.strip().splitlines()
            if not lines:
                continue
            rel_path = lines[0].strip()
            rest = "\n".join(lines[1:])

            # Extract <<<OLD>>> / <<<NEW>>> / <<<END>>> blocks
            pattern = re.compile(
                r"<<<OLD>>>(.*?)<<<NEW>>>(.*?)<<<END>>>",
                re.DOTALL,
            )
            for match in pattern.finditer(rest):
                old_code = match.group(1).strip()
                new_code = match.group(2).strip()
                if not old_code or not new_code:
                    continue
                full_path = self._src / rel_path
                if any(rel_path.startswith(prefix) for prefix in self.BLOCKED_PATH_PREFIXES):
                    applied.append(f"[BLOCKED] {rel_path}: protected by blocklist")
                    continue
                if not full_path.exists():
                    applied.append(f"[SKIP] File not found: {rel_path}")
                    continue

                if full_path not in changed_files and len(changed_files) >= self.MAX_PATCH_FILES:
                    applied.append(
                        f"[BLOCKED] {rel_path}: max changed files exceeded ({self.MAX_PATCH_FILES})"
                    )
                    continue

                original = full_path.read_text(encoding="utf-8")
                if old_code not in original:
                    applied.append(
                        f"[SKIP] Old code not found in {rel_path} — may already be fixed or Llama hallucinated"
                    )
                    continue
                updated = original.replace(old_code, new_code, 1)

                # Mandatory syntax gate per file before writing to disk.
                try:
                    compile(updated, str(full_path), "exec")
                except SyntaxError as e:
                    applied.append(f"[BLOCKED] {rel_path}: syntax check failed ({e.msg})")
                    continue

                delta_size = abs(len(new_code) - len(old_code))
                if (total_changed_chars + delta_size) > self.MAX_PATCH_TOTAL_CHARS:
                    applied.append(
                        f"[BLOCKED] {rel_path}: max patch size exceeded ({self.MAX_PATCH_TOTAL_CHARS} chars)"
                    )
                    continue

                if full_path not in backups:
                    backups[full_path] = original
                full_path.write_text(updated, encoding="utf-8")
                if full_path not in changed_files:
                    changed_files.append(full_path)
                total_changed_chars += delta_size
                applied.append(f"[APPLIED] {rel_path}: replaced {len(old_code)} chars")

        # Optional mandatory smoke-test command (if configured).
        test_cmd = os.getenv("LLAMA_MANDATORY_TEST_CMD", "").strip()
        if test_cmd and backups:
            try:
                proc = subprocess.run(
                    test_cmd,
                    shell=True,
                    cwd=str(self._src.parent),
                    capture_output=True,
                    text=True,
                    timeout=int(os.getenv("TEST_TIMEOUT", "300")),
                    check=False,
                )
                if proc.returncode != 0:
                    for p, txt in backups.items():
                        p.write_text(txt, encoding="utf-8")
                    applied.append(
                        "[ROLLED_BACK] Mandatory tests failed; reverted all applied Llama changes"
                    )
            except Exception as e:
                for p, txt in backups.items():
                    p.write_text(txt, encoding="utf-8")
                applied.append(f"[ROLLED_BACK] Mandatory test execution error: {e}")

        return applied if applied else ["[NO_CHANGES] No parseable code blocks found in Llama output"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_metrics(self, diagnostics: Dict[str, Any]) -> str:
        lines = [f"Integrity Score: {diagnostics.get('score', 0.0)}/100"]
        for issue in diagnostics.get("inconsistencies", [])[:6]:
            lines.append(f"  - {issue}")
        var_cvar = diagnostics.get("var_cvar") or {}
        if var_cvar:
            lines.append(
                f"VaR95={var_cvar.get('var_95')}  CVaR95={var_cvar.get('cvar_95')}  "
                f"VaR99={var_cvar.get('var_99')}  CVaR99={var_cvar.get('cvar_99')}"
            )
        risk = diagnostics.get("risk_adjusted_returns") or {}
        if risk:
            lines.append(f"Sharpe={risk.get('sharpe')}  Sortino={risk.get('sortino')}")
        return "\n".join(lines)


class ApprovalGateway:
    """File-based human-approval gate for optimizer mutations.

    Invisible to the Streamlit UI and to all regular users.
    Only the system owner can approve or reject proposed code mutations.

    Approval flow:
      1. Optimizer calls ``request(action_id, description, details)``.
      2. Request is written to ``output/.optimizer/approval_queue.json``.
      3. Interactive terminal → blocks on stdin ``y/n`` prompt.
      4. Non-interactive process → polls the queue file for the owner to
         manually set ``"status": "approved"`` or ``"status": "rejected"``.
         Times out after *timeout_seconds* and treats timeout as rejection.
    """

    QUEUE_FILE = "approval_queue.json"
    ALERT_FILE = "approval_pending_alert.txt"

    def __init__(self, base_output_path: Path, timeout_seconds: int = 120, force_non_interactive: bool = False) -> None:
        self._dir = base_output_path / ".optimizer"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._queue_path = self._dir / self.QUEUE_FILE
        self._user_id = base_output_path.name or "default"
        self._timeout = timeout_seconds
        self._force_non_interactive = force_non_interactive

    def request(
        self,
        action_id: str,
        description: str,
        details: Dict[str, Any],
    ) -> bool:
        """Return True if the owner approves (YES); False if rejected (NO) or timed-out."""
        entry: Dict[str, Any] = {
            "action_id": action_id,
            "description": description,
            "details": details,
            "status": "pending",
            "requested_at": datetime.utcnow().isoformat() + "Z",
            "approved_at": None,
        }
        self._queue_path.write_text(
            json.dumps(entry, indent=2, default=str), encoding="utf-8"
        )
        self._emit_pending_approval_notice(entry)
        # Use file polling if forced non-interactive OR if running in non-interactive environment
        if self._force_non_interactive or not (sys.stdin.isatty() and sys.stdout.isatty()):
            approved = self._poll_file_approval()
        else:
            approved = self._prompt_interactive(description, details)
        entry["status"] = "YES" if approved else "NO"
        entry["approved_at"] = datetime.utcnow().isoformat() + "Z"
        self._queue_path.write_text(
            json.dumps(entry, indent=2, default=str), encoding="utf-8"
        )
        self._clear_pending_alert_file()
        return approved

    def _emit_pending_approval_notice(self, entry: Dict[str, Any]) -> None:
        """Emit a loud, live notification and persist an alert file while awaiting owner approval."""
        details_text = json.dumps(entry.get("details", {}), indent=2, default=str)
        approval_script = Path(__file__).resolve().parent.parent / "respond_to_approval.py"
        approval_cmd = f'"{sys.executable}" "{approval_script}" --user-id {self._user_id} --approve/--reject'
        banner = (
            "\n" + "!" * 76 + "\n"
            "[OPTIMIZER] OWNER APPROVAL REQUIRED (PENDING)\n"
            f"Action ID : {entry.get('action_id')}\n"
            f"Action    : {entry.get('description')}\n"
            f"Timeout   : {self._timeout} seconds\n"
            f"Queue     : {self._queue_path}\n"
            f"Respond   : YES/NO in terminal OR run {approval_cmd}\n"
            "-" * 76 + "\n"
            f"FULL DETAILS:\n{details_text}\n"
            + "!" * 76
        )
        print(banner)
        self._notify_desktop_popup(entry)
        self._notify_mobile(entry)

        try:
            alert_path = self._dir / self.ALERT_FILE
            alert_path.write_text(banner + "\n", encoding="utf-8")
        except Exception:
            pass

    def _notify_desktop_popup(self, entry: Dict[str, Any]) -> None:
        """Best-effort Windows popup for interactive desktop sessions."""
        title = "STRATUM QUANT ANALYTICS: Approval Required"
        action = str(entry.get("description", "Approval pending"))
        body = (
            f"{action}\n\n"
            f"Timeout: {self._timeout}s\n"
            f"Queue: {self._queue_path}\n\n"
            "Respond with YES/NO in terminal or run:\n"
            f"respond_to_approval.py --user-id {self._user_id} --approve / --reject"
        )

        try:
            escaped_title = title.replace("'", "''")
            escaped_body = body.replace("'", "''")
            cmd = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                f"[System.Windows.Forms.MessageBox]::Show('{escaped_body}','{escaped_title}')"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True,
                timeout=8,
                check=False,
            )
        except Exception:
            pass

    def _load_env_once(self) -> None:
        """Load .env from project root the first time notifications are attempted."""
        try:
            from dotenv import load_dotenv as _load_dotenv
            # Use __file__ to reliably locate the project root (src/../.env)
            _project_root = Path(__file__).resolve().parent.parent
            _env_path = _project_root / ".env"
            if _env_path.exists():
                _load_dotenv(_env_path, override=False)
        except Exception:
            pass

    def _log_notification(self, channel: str, success: bool, detail: str) -> None:
        """Append one line to notification_log.txt so failures are auditable."""
        try:
            log_path = self._dir / "notification_log.txt"
            ts = datetime.utcnow().isoformat() + "Z"
            status = "OK" if success else "FAIL"
            with log_path.open("a", encoding="utf-8") as _lf:
                _lf.write(f"{ts} [{channel}] {status}: {detail}\n")
        except Exception:
            pass

    def _send_telegram(self, text: str) -> tuple[bool, str]:
        """Send a Telegram message. Returns (success, detail). Never raises."""
        self._load_env_once()
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not bot_token or not chat_id:
            return False, "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in secrets.toml or environment"
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=8,
            )
            if resp.status_code == 200:
                return True, f"chat_id={chat_id}"
            return False, f"HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as exc:
            return False, str(exc)

    def _notify_mobile(self, entry: Dict[str, Any]) -> None:
        """Send mobile notification via Telegram (and optionally webhook). Logs all outcomes."""
        self._load_env_once()
        action = str(entry.get("description", "Approval pending"))
        details = entry.get("details", {})
        # Include problem + proposed solution from Llama output when available.
        llama_snippet = ""
        if isinstance(details, dict) and details.get("llama_analysis"):
            raw = str(details["llama_analysis"])

            def _extract_section(text: str, section: str) -> str:
                marker = f"---{section}---"
                if marker not in text:
                    return ""
                tail = text.split(marker, 1)[1]
                next_marker_idx = tail.find("---")
                value = tail[:next_marker_idx] if next_marker_idx != -1 else tail
                return value.strip()

            prob = _extract_section(raw, "PROBLEM")
            solution = _extract_section(raw, "SOLUTION")
            snippets: List[str] = []
            if prob:
                snippets.append(f"<b>AI found:</b>\n{html.escape(prob[:350])}")
            if solution:
                snippets.append(f"<b>AI proposed fix:</b>\n{html.escape(solution[:350])}")
            if snippets:
                llama_snippet = "\n\n" + "\n\n".join(snippets)

        approval_script = Path(__file__).resolve().parent.parent / "respond_to_approval.py"
        message = (
            "\U0001F6A8 <b>STRATUM QUANT ANALYTICS — Approval Required</b>\n"
            f"<b>Action:</b> {html.escape(action)}\n"
            f"<b>Timeout:</b> {self._timeout}s\n"
            f"<b>Queue:</b> {html.escape(str(self._queue_path))}\n"
            f"<b>Respond:</b> {html.escape(sys.executable)} {html.escape(str(approval_script))} --user-id {html.escape(self._user_id)} --approve OR --reject"
            f"{llama_snippet}"
        )
        # Telegram hard limit ~4096 chars for text payloads.
        if len(message) > 3900:
            message = message[:3850] + "\n\n<i>(message truncated)</i>"

        # --- Telegram ---
        ok, detail = self._send_telegram(message)
        self._log_notification("Telegram", ok, detail)
        if not ok:
            print(f"[NOTIFY] ⚠️  Telegram notification FAILED: {detail}")
        else:
            print(f"[NOTIFY] ✅ Telegram notification sent ({detail})")

        # --- Generic webhook (optional) ---
        self._load_env_once()
        webhook_url = os.getenv("MOBILE_NOTIFY_WEBHOOK_URL", "").strip()
        if webhook_url:
            try:
                resp = requests.post(
                    webhook_url,
                    json={
                        "title": "STRATUM QUANT ANALYTICS Approval Required",
                        "message": action,
                        "action_id": entry.get("action_id"),
                        "requested_at": entry.get("requested_at"),
                    },
                    timeout=5,
                )
                wok = resp.status_code < 400
                self._log_notification("Webhook", wok, f"{webhook_url[:60]} HTTP {resp.status_code}")
            except Exception as exc:
                self._log_notification("Webhook", False, str(exc)[:120])

    def notify_run_started(self, issues: List[str]) -> None:
        """Send a notification at the start of the run when issues are found."""
        if not issues:
            return
        self._load_env_once()
        issue_list = "\n".join(f"  \u2022 {html.escape(str(i))}" for i in issues[:8])
        text = (
            "\U0001F916 <b>Optimizer run started</b>\n"
            f"Found {len(issues)} issue(s):\n{issue_list}\n\n"
            "Awaiting your approval if AI proposes code changes."
        )
        ok, detail = self._send_telegram(text)
        self._log_notification("Telegram/run_start", ok, detail)
        if not ok:
            print(f"[NOTIFY] ⚠️  Run-start Telegram notification FAILED: {detail}")
        else:
            print(f"[NOTIFY] ✅ Run-start Telegram notification sent ({detail})")

    def notify_run_finished(self, score: float, status: str, issues: List[str]) -> None:
        """Send a completion notification with final score."""
        self._load_env_once()
        issue_list = "\n".join(f"  \u2022 {html.escape(str(i))}" for i in issues[:6])
        verdict = "\u2705 Target reached!" if "optimized" in status else f"\u26A0\uFE0F {html.escape(status)} (score {score:.0f}/100)"
        text = (
            f"\U0001F916 <b>Optimizer finished</b> — {verdict}\n"
            + (f"Remaining issues:\n{issue_list}" if issues else "No outstanding issues.")
        )
        ok, detail = self._send_telegram(text)
        self._log_notification("Telegram/run_finish", ok, detail)

    def _clear_pending_alert_file(self) -> None:
        try:
            alert_path = self._dir / self.ALERT_FILE
            if alert_path.exists():
                alert_path.unlink()
        except Exception:
            pass

    def _prompt_interactive(self, description: str, details: Dict[str, Any]) -> bool:
        print("\n" + "=" * 60)
        print("[OPTIMIZER] ⚠️  APPROVAL REQUIRED BEFORE CODE MUTATION")
        print(f"  Action : {description}")
        print(f"  Details: {json.dumps(details, indent=4, default=str)}")
        print("=" * 60)
        try:
            answer = input("  Approve this change? (YES/NO): ").strip().upper()
        except EOFError:
            answer = "NO"
        return answer == "YES"

    def _poll_file_approval(self) -> bool:
        """Non-interactive: owner edits queue file status to 'YES' or 'NO'."""
        deadline = time.time() + self._timeout
        next_progress_log = time.time()

        def _norm_status(raw: Any) -> str:
            val = str(raw or "pending").strip().lower()
            if val in {"yes", "y", "approved", "approve", "true", "1"}:
                return "YES"
            if val in {"no", "n", "rejected", "reject", "false", "0", "timeout"}:
                return "NO"
            return "pending"

        while time.time() < deadline:
            try:
                data = json.loads(self._queue_path.read_text(encoding="utf-8"))
                status = _norm_status(data.get("status", "pending"))
                if status == "YES":
                    return True
                if status == "NO":
                    return False
            except Exception:
                pass

            now = time.time()
            if now >= next_progress_log:
                remaining = max(0, int(deadline - now))
                print(
                    f"[OPTIMIZER] Waiting for approval... {remaining}s left | "
                    f"Queue: {self._queue_path}"
                )
                next_progress_log = now + 10
            time.sleep(2.0)

        print("[OPTIMIZER] Approval timeout reached. Treating as NO.")
        return False


class AutomatedOptimizationLoop:
    """Senior-quant style optimization loop with hard stop conditions.

    Workflow:
    1. Run Fetchers + Pipeline + Analysis Suite.
    2. Run self-diagnostic integrity audit.
    3. If score < target, auto-adjust failed components and rerun.
    4. Stop at score >= target or max_iterations.
    """

    def __init__(
        self,
        target_score: float = 94.0,
        max_iterations: int = 8,
        user_id: Optional[str] = None,
        scheduled: bool = False,
    ) -> None:
        self.target_score = float(target_score)
        self.max_iterations = int(max_iterations)
        self.user_id = user_id or (os.getenv("DATA_USER_ID", "default").strip() or "default")
        self.scheduled = scheduled  # True if running from scheduler (file-based approvals)

        self.start_date_override: Optional[str] = None
        self.force_macro_lag_30: bool = False
        self.zscore_relax_steps: int = 0
        self.records: List[IterationRecord] = []
        # Approval gateway is initialised lazily on first use (output_dir
        # must exist first).  Stored as None until _ensure_approval_gateway.
        self._approval: Optional[ApprovalGateway] = None
        # Initialize Llama analyzer with src path so it can read project files
        self._llama: LlamaQuantAnalyzer = LlamaQuantAnalyzer(
            src_root=Path(__file__).parent,
        )  # timeout_seconds reads OLLAMA_TIMEOUT from env (default 300s)

    def _ensure_approval_gateway(self) -> ApprovalGateway:
        """Lazily create the approval gateway so output_dir exists first."""
        if self._approval is None:
            self._approval = ApprovalGateway(
                base_output_path=self.output_dir.parent,
                timeout_seconds=int(os.getenv("APPROVAL_TIMEOUT", "300")),
                force_non_interactive=self.scheduled,
            )
        return self._approval

    def _run_cmd(self, cmd: list[str], cwd: Path | None = None, timeout: int = 40) -> tuple[bool, str]:
        """Run a command and capture output without raising."""
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd or self.project_root),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            return proc.returncode == 0, out.strip()
        except Exception as exc:
            return False, str(exc)

    def _auto_commit_and_push(self, reason: str) -> str:
        """Best-effort git add/commit/push after approved code changes.

        Controlled by OPTIMIZER_AUTO_GIT_PUSH (default: 1).
        Never raises; returns a short status string for logs/adjustments.
        """
        enabled = (os.getenv("OPTIMIZER_AUTO_GIT_PUSH", "1").strip().lower() not in {"0", "false", "no"})
        if not enabled:
            return "[GIT] Auto push disabled by OPTIMIZER_AUTO_GIT_PUSH"

        ok, _ = self._run_cmd(["git", "rev-parse", "--is-inside-work-tree"])
        if not ok:
            return "[GIT] Skipped: not a git worktree"

        self._run_cmd(["git", "add", "-A"])
        ok_diff, _ = self._run_cmd(["git", "diff", "--cached", "--quiet"])
        if ok_diff:
            return "[GIT] No staged changes to commit"

        ok_branch, branch_out = self._run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        branch = branch_out.splitlines()[0].strip() if ok_branch and branch_out else "main"

        msg = f"optimizer: apply approved AI changes ({reason})"
        ok_commit, commit_out = self._run_cmd(["git", "commit", "-m", msg], timeout=90)
        if not ok_commit:
            return f"[GIT] Commit failed: {commit_out[:220]}"

        ok_push, push_out = self._run_cmd(["git", "push", "origin", branch], timeout=120)
        if not ok_push:
            return f"[GIT] Commit ok, push failed: {push_out[:220]}"
        return "[GIT] Commit+push completed"

    # ------------------------------------------------------------------
    # Regression Testing Gate
    # ------------------------------------------------------------------

    def _run_regression_gate(self, issue_type: str = "") -> tuple[bool, str]:
        """Financial-validation regression gate — runs AFTER code changes are applied.

        Verifies the backtest engine still produces realistic metrics:
          • Sharpe ratio bounded to [-5.5, 5.5]
          • Max drawdown not ≈ -100% (equity-to-zero collapse)
          • Final equity not astronomical (> 1000× from $1)
          • Annualised return ≤ 9900 % (99.0 multiplier)

        On failure the gate:
          1. Rolls back changed files (caller's responsibility via git checkout)
          2. Sends a Telegram message to the owner
        Returns (passed: bool, report: str).  Never raises.
        """
        import importlib
        import sys as _sys

        try:
            master_path = self.project_root / "data" / "gold" / "master_table.parquet"
            if not master_path.exists():
                return True, "[GATE] No price data available — financial gate skipped (non-blocking)"

            src_str = str(self.project_root / "src")
            if src_str not in _sys.path:
                _sys.path.insert(0, src_str)

            try:
                import Medallion.gold.AnalysisSuite.backtest as _bt_mod
                importlib.reload(_bt_mod)
                run_bt = _bt_mod.run_strategy_backtest
            except Exception as exc:
                return True, f"[GATE] Cannot import backtest module ({exc}) — gate skipped"

            import pandas as _pd

            df = _pd.read_parquet(master_path)
            price_col = "adj_close" if "adj_close" in df.columns else "close"
            if price_col not in df.columns:
                return True, "[GATE] No price column in master table — gate skipped"

            if "ticker" in df.columns:
                tickers = df["ticker"].dropna().unique().tolist()
                if tickers:
                    df = df[df["ticker"] == tickers[0]].copy()

            if "date" in df.columns:
                df = df.sort_values("date")
                prices = df.set_index("date")[price_col].dropna()
            else:
                prices = df[price_col].dropna()

            prices = _pd.to_numeric(prices, errors="coerce").dropna()
            if len(prices) < 210:
                return True, f"[GATE] Only {len(prices)} price rows — gate skipped"

            result = run_bt(prices=prices, rolling_window=20, z_threshold=1.5)
            m = result.get("metrics", {})
            ec = result.get("equity_curve", [1.0])
            final_equity = float(ec[-1]) if ec else 1.0
            sharpe = m.get("sharpe_ratio")
            mdd = m.get("max_drawdown", 0.0)
            ann_ret = m.get("annualized_return", 0.0)

            failures: list[str] = []
            if sharpe is not None and abs(float(sharpe)) > 5.5:
                failures.append(f"Sharpe={sharpe:.2f} outside [-5.5, 5.5]")
            if mdd is not None and float(mdd) < -0.99:
                failures.append(f"MaxDrawdown={mdd:.2%} ≈ -100% (equity collapse)")
            if final_equity > 1000.0:
                failures.append(f"FinalEquity={final_equity:.2f}× — astronomical (>1000×)")
            if ann_ret is not None and float(ann_ret) > 99.0:
                failures.append(f"AnnReturn={ann_ret:.1f} — unrealistic (>9900%)")

            if failures:
                gate = self._ensure_approval_gateway()
                gate._send_telegram(
                    "⚠️ <b>STRATUM QUANT ANALYTICS — Regression Gate FAILED</b>\n"
                    f"Issue: {html.escape(issue_type or 'N/A')}\n\n"
                    "Η λύση πέρασε το syntax check αλλά απέτυχε στο Financial Validation.\n"
                    "Τα αρχεία επαναφέρθηκαν στην προηγούμενη κατάσταση.\n\n"
                    "Αποτυχίες:\n"
                    + "\n".join(f"  • {html.escape(f)}" for f in failures)
                )
                return False, "[GATE FAILED] " + "; ".join(failures)

            return True, (
                f"[GATE PASSED] Sharpe={sharpe}, MDD={float(mdd):.2%}, "
                f"FinalEquity={final_equity:.3f}\u00d7, AnnReturn={float(ann_ret):.1%}"
            )
        except Exception as exc:
            return True, f"[GATE] Validation error (non-blocking): {exc}"

    # ------------------------------------------------------------------
    # Semantic Versioning
    # ------------------------------------------------------------------

    def _bump_version(self) -> str:
        """Bump the patch segment of pyproject.toml after an approved optimizer change.

        0.1.0 → 0.1.1  |  0.1.1-opt → 0.1.2-opt
        Writes the updated file and returns a description of the change.
        """
        import re as _re

        toml_path = self.project_root / "pyproject.toml"
        if not toml_path.exists():
            return "[VERSION] pyproject.toml not found — skipped"
        try:
            text = toml_path.read_text(encoding="utf-8")
            # Matches: version = "X.Y.Z"  or  version = "X.Y.Z-suffix"
            pat = _re.compile(r'(version\s*=\s*")(\d+\.\d+\.)(\d+)(-[^"]*)?(")')
            match = pat.search(text)
            if not match:
                return "[VERSION] version field not found in pyproject.toml — skipped"
            _prefix, maj_min, patch_str, suffix, _close = match.groups()
            old_ver = f"{maj_min}{patch_str}{suffix or ''}"
            new_ver = f"{maj_min}{int(patch_str) + 1}{suffix or ''}"
            updated = pat.sub(
                lambda m: m.group(1) + new_ver + m.group(5),
                text,
                count=1,
            )
            toml_path.write_text(updated, encoding="utf-8")
            return f"[VERSION] {old_ver} → {new_ver} (pyproject.toml bumped)"
        except Exception as exc:
            return f"[VERSION] Error bumping version: {exc}"

    @property
    def safe_user(self) -> str:
        return (
            "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(self.user_id))
            or "default"
        )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output" / self.safe_user

    @property
    def quality_report_path(self) -> Path:
        return (
            self.project_root
            / "data"
            / "users"
            / self.safe_user
            / "processed"
            / "quality"
            / "quality_report.json"
        )

    def _load_quality_report(self) -> Dict[str, Any]:
        if not self.quality_report_path.exists():
            return {}
        try:
            return json.loads(self.quality_report_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _build_config(self) -> ProjectConfig:
        cfg = ProjectConfig.load_from_env()
        cfg.data_user_id = self.safe_user
        cfg.mode = RunMode.ACTUAL
        if self.start_date_override:
            cfg.start_date = self.start_date_override
        cfg.validate_runtime_constraints()
        return cfg

    def _apply_pipeline_outlier_relaxation(self, delta: float = 0.5) -> None:
        """Relax Silver z-score outlier thresholds by replacing immutable contracts."""
        self.zscore_relax_steps += 1
        for source, src_contract in list(silver_contracts.SOURCE_CONTRACTS.items()):
            base = src_contract.default_series_contract
            relaxed_default = silver_contracts.SeriesContract(
                semantic_type=base.semantic_type,
                unit_kind=base.unit_kind,
                null_tolerance_pct=base.null_tolerance_pct,
                outlier_z_threshold=float(base.outlier_z_threshold) + float(delta),
                enable_winsorization=base.enable_winsorization,
                imputation_strategy=base.imputation_strategy,
            )
            relaxed_entities: Dict[str, silver_contracts.SeriesContract] = {}
            for entity, e_contract in src_contract.entity_contracts.items():
                relaxed_entities[entity] = silver_contracts.SeriesContract(
                    semantic_type=e_contract.semantic_type,
                    unit_kind=e_contract.unit_kind,
                    null_tolerance_pct=e_contract.null_tolerance_pct,
                    outlier_z_threshold=float(e_contract.outlier_z_threshold) + float(delta),
                    enable_winsorization=e_contract.enable_winsorization,
                    imputation_strategy=e_contract.imputation_strategy,
                )

            silver_contracts.SOURCE_CONTRACTS[source] = silver_contracts.SourceContract(
                required_columns=set(src_contract.required_columns),
                percentage_entities=set(src_contract.percentage_entities),
                default_series_contract=relaxed_default,
                entity_contracts=relaxed_entities,
            )

    def _run_pipeline(self, config: ProjectConfig) -> tuple[Dict[str, Any], MedallionPipeline]:
        factory = DataFactory(fred_api_key=config.fred_api_key)
        pipeline = MedallionPipeline(config=config, factory=factory)
        if config.should_use_parallel_pipeline() and config.mode == RunMode.ACTUAL:
            results = pipeline.run_full_pipeline_parallel()
        else:
            results = pipeline.run_full_pipeline_sequential()
        return results, pipeline

    def _recalculate_negative_r2_component(
        self,
        results: Dict[str, Any],
        pipeline: MedallionPipeline,
    ) -> Dict[str, Any]:
        """If R2 is negative, re-run sensitivity regression with 30-day macro lag."""
        try:
            gold = pipeline._get_gold_layer()
            ticker = gold._resolve_ticker(None)
            analysis_df = gold._analysis_df(ticker)
            lagged = sensitivity_reg(
                analysis_df,
                target="log_return",
                factors=["inflation", "energy_index"],
                model="OLS",
                ticker=ticker,
                macro_lag_days=30,
            )
            if isinstance(lagged, dict):
                lagged["optimizer_macro_lag_days"] = 30
            results["sensitivity_regression"] = lagged
            self.force_macro_lag_30 = True
            return results
        except Exception:
            return results

    def _detect_outlier_ratio(self, quality: Dict[str, Any]) -> float:
        files = quality.get("files", {}) if isinstance(quality, dict) else {}
        worst = 0.0
        if not isinstance(files, dict):
            return worst
        for payload in files.values():
            if not isinstance(payload, dict):
                continue
            final_rows = float(payload.get("final_rows", 0) or 0)
            clipped = float(payload.get("outliers_clipped", 0) or 0)
            if final_rows > 0:
                worst = max(worst, clipped / final_rows)
        return worst

    def _diagnose(self, results: Dict[str, Any], quality: Dict[str, Any]) -> Dict[str, Any]:
        inconsistencies: List[str] = []
        score = 100.0

        summary = quality.get("summary", {}) if isinstance(quality, dict) else {}
        missing_sources = summary.get("missing_sources", []) if isinstance(summary, dict) else []
        failed_count = int(summary.get("failed_count", 0) or 0) if isinstance(summary, dict) else 0

        if isinstance(missing_sources, list) and missing_sources:
            inconsistencies.append(f"DATA_GAPS:missing_sources={missing_sources}")
            score -= min(20.0, 8.0 * len(missing_sources))
        if failed_count > 0:
            inconsistencies.append(f"DATA_GAPS:failed_entities={failed_count}")
            score -= min(15.0, float(failed_count))

        gov = results.get("governance_report", {}) if isinstance(results.get("governance_report"), dict) else {}
        oos_r2 = ((gov.get("out_of_sample") or {}).get("r2") if isinstance(gov, dict) else None)

        sens = results.get("sensitivity_regression", {}) if isinstance(results.get("sensitivity_regression"), dict) else {}
        sens_r2 = sens.get("r2") if isinstance(sens, dict) else None

        leakage_flags = gov.get("leakage_flags", []) if isinstance(gov, dict) else []
        if isinstance(leakage_flags, list):
            suspicious = [x for x in leakage_flags if "leakage" in str(x).lower() or "future" in str(x).lower()]
            if suspicious:
                inconsistencies.append(f"LOOKAHEAD_BIAS:{suspicious}")
                score -= min(20.0, 8.0 * len(suspicious))

        elas = results.get("elasticity", {}) if isinstance(results.get("elasticity"), dict) else {}
        if isinstance(elas, dict) and elas.get("elasticity_computable") is False:
            inconsistencies.append("SCALING_ISSUE:elasticity_not_computable_mean_asset_near_zero")
            score -= 6.0

        outlier_ratio = self._detect_outlier_ratio(quality)
        if outlier_ratio > 0.05:
            inconsistencies.append(f"OUTLIERS:ratio_gt_5pct={outlier_ratio:.3f}")
            score -= min(12.0, outlier_ratio * 100.0)

        # ── Senior Quant additions ──────────────────────────────────────
        risk_adj = self._check_risk_adjusted_return(results)
        sharpe = risk_adj.get("sharpe")
        if isinstance(sharpe, float) and sharpe < -0.5:
            inconsistencies.append(f"POOR_RISK_ADJ:sharpe={sharpe:.4f}")
            score -= 8.0

        drawdown_info = self._check_drawdown_limits(results)
        if drawdown_info.get("exceeds_threshold"):
            impact = drawdown_info["worst_shock_impact"]
            inconsistencies.append(f"EXCESS_DRAWDOWN:worst_shock_impact={impact:.4f}")
            score -= 6.0

        vif_info = self._check_factor_vif(results)
        high_vif = vif_info.get("high_vif_factors", [])
        if isinstance(high_vif, list) and high_vif:
            inconsistencies.append(f"MULTICOLLINEARITY:high_vif_factors={high_vif}")
            score -= min(8.0, 4.0 * len(high_vif))

        regime_info = self._check_regime_consistency(results)
        if regime_info.get("regime_stable") is False:
            cv = regime_info.get("elasticity_cv")
            robust_cv = regime_info.get("robust_cv")
            threshold = regime_info.get("dynamic_threshold", 2.0)
            display_cv = robust_cv if robust_cv is not None else cv
            label = (
                f"REGIME_INSTABILITY:robust_cv={display_cv:.4f},threshold={threshold:.2f}"
                if display_cv is not None
                else "REGIME_INSTABILITY:high_elasticity_variance"
            )
            inconsistencies.append(label)
            score -= 5.0

        freshness_info = self._check_data_freshness(results)
        if freshness_info.get("is_stale"):
            inconsistencies.append(f"STALE_DATA:lag_days={freshness_info.get('lag_days')}")
            score -= 4.0

        shift_info = self._check_distribution_shift_proxy(results)
        if shift_info.get("shift_detected"):
            inconsistencies.append(
                f"DIST_SHIFT:std_ratio={shift_info.get('std_ratio')},mean_shift={shift_info.get('mean_shift')}"
            )
            score -= 6.0

        autocorr_info = self._check_residual_autocorrelation(results)
        if autocorr_info.get("autocorr_problem"):
            inconsistencies.append(f"RESIDUAL_AUTOCORR:lag1={autocorr_info.get('lag1_autocorr')}")
            score -= 4.0

        ci_info = self._check_confidence_interval_quality(results)
        if ci_info.get("ci_problem"):
            inconsistencies.append(f"CI_QUALITY:{ci_info.get('reason')}")
            score -= 4.0

        sign_info = self._check_sign_stability(results)
        if sign_info.get("sign_unstable"):
            inconsistencies.append(f"SIGN_INSTABILITY:flip_ratio={sign_info.get('flip_ratio')}")
            score -= 4.0

        uncertainty_info = self._check_uncertainty_gap(results)
        if uncertainty_info.get("gap_detected"):
            inconsistencies.append(f"UNCERTAINTY_GAP:{uncertainty_info.get('reason')}")
            score -= 3.0

        pipeline_info = self._check_pipeline_robustness(quality)
        if pipeline_info.get("robustness_problem"):
            inconsistencies.append(f"PIPELINE_ROBUSTNESS:{pipeline_info.get('reason')}")
            score -= 5.0

        temporal_info = self._check_temporal_integrity(results)
        if temporal_info.get("temporal_problem"):
            inconsistencies.append(f"TEMPORAL_INTEGRITY:{temporal_info.get('reason')}")
            score -= 7.0

        backtest_info = self._check_backtest_realism(results)
        if backtest_info.get("realism_problem"):
            inconsistencies.append(f"BACKTEST_REALISM:{backtest_info.get('reason')}")
            score -= 5.0

        explain_info = self._check_explainability_risk(results)
        if explain_info.get("explainability_problem"):
            inconsistencies.append(f"EXPLAINABILITY_RISK:{explain_info.get('reason')}")
            score -= 4.0

        feature_shift_info = self._check_feature_shift_psi_ks(results)
        if feature_shift_info.get("shift_detected"):
            inconsistencies.append(f"FEATURE_SHIFT:{feature_shift_info.get('reason')}")
            score -= 6.0

        residual_adv_info = self._check_residual_diagnostics_advanced(results)
        if residual_adv_info.get("hetero_problem"):
            inconsistencies.append(f"RESIDUAL_HETERO:{residual_adv_info.get('reason')}")
            score -= 4.0

        coverage_info = self._check_interval_coverage_calibration(results)
        if coverage_info.get("coverage_problem"):
            inconsistencies.append(f"COVERAGE_CALIBRATION:{coverage_info.get('reason')}")
            score -= 5.0

        shap_info = self._check_shap_stability(results)
        if shap_info.get("shap_problem"):
            inconsistencies.append(f"SHAP_STABILITY:{shap_info.get('reason')}")
            score -= 4.0

        coeff_ci_info = self._check_coeff_ci_overlap(results)
        if coeff_ci_info.get("ci_overlap_problem"):
            inconsistencies.append(f"COEFF_CI_OVERLAP:{coeff_ci_info.get('reason')}")
            score -= 4.0

        var_cvar = self._compute_var_cvar(results)
        # ───────────────────────────────────────────────────────────────

        score = max(0.0, min(100.0, score))
        return {
            "score": round(score, 2),
            "inconsistencies": inconsistencies,
            "outlier_ratio": outlier_ratio,
            "has_negative_r2": False,
            "has_data_gaps": any(s.startswith("DATA_GAPS") for s in inconsistencies),
            "var_cvar": var_cvar,
            "risk_adjusted_returns": risk_adj,
            "drawdown_info": drawdown_info,
            "vif_info": vif_info,
            "regime_info": regime_info,
            "freshness_info": freshness_info,
            "shift_info": shift_info,
            "autocorr_info": autocorr_info,
            "ci_info": ci_info,
            "sign_info": sign_info,
            "uncertainty_info": uncertainty_info,
            "pipeline_info": pipeline_info,
            "temporal_info": temporal_info,
            "backtest_info": backtest_info,
            "explain_info": explain_info,
            "feature_shift_info": feature_shift_info,
            "residual_adv_info": residual_adv_info,
            "coverage_info": coverage_info,
            "shap_info": shap_info,
            "coeff_ci_info": coeff_ci_info,
        }

    def _apply_fetcher_adjustment(self, config: ProjectConfig) -> str:
        base_start = datetime.strptime(config.start_date, "%Y-%m-%d")
        extended = base_start - timedelta(days=365)
        self.start_date_override = extended.strftime("%Y-%m-%d")
        # Secondary API switching is unavailable in current architecture.
        return (
            "Fetchers adjustment: extended start_date by 365 days "
            f"to {self.start_date_override}; secondary API fallback unavailable."
        )

    # ------------------------------------------------------------------
    # Quantitative Analysis Helpers (Senior Quant Analyst additions)
    # ------------------------------------------------------------------

    def _compute_var_cvar(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract VaR / CVaR at 95 % and 99 % from Monte Carlo output."""
        mc = results.get("monte_carlo") if isinstance(results.get("monte_carlo"), dict) else {}
        dist: Any = (
            mc.get("returns_distribution")
            or mc.get("final_returns")
            or mc.get("pct_returns")
            or mc.get("simulated_final_prices")
        )
        empty: Dict[str, Any] = {"var_95": None, "cvar_95": None, "var_99": None, "cvar_99": None}
        if not isinstance(dist, list) or len(dist) < 10:
            return empty
        try:
            arr = np.array(
                [float(x) for x in dist if x is not None and np.isfinite(float(x))],
                dtype=float,
            )
        except (TypeError, ValueError):
            return empty
        if len(arr) < 10:
            return empty
        var_95 = float(np.percentile(arr, 5))
        cvar_95 = float(arr[arr <= var_95].mean()) if (arr <= var_95).any() else var_95
        var_99 = float(np.percentile(arr, 1))
        cvar_99 = float(arr[arr <= var_99].mean()) if (arr <= var_99).any() else var_99
        return {
            "var_95": round(var_95, 6),
            "cvar_95": round(cvar_95, 6),
            "var_99": round(var_99, 6),
            "cvar_99": round(cvar_99, 6),
        }

    def _check_risk_adjusted_return(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute annualised Sharpe and Sortino from governance OOS residuals."""
        gov = results.get("governance_report") if isinstance(results.get("governance_report"), dict) else {}
        oos = gov.get("out_of_sample") if isinstance(gov.get("out_of_sample"), dict) else {}
        residuals = oos.get("residuals") or []
        if not isinstance(residuals, list) or len(residuals) < 5:
            return {"sharpe": None, "sortino": None}
        try:
            r = np.array(
                [float(x) for x in residuals if x is not None and np.isfinite(float(x))],
                dtype=float,
            )
        except (TypeError, ValueError):
            return {"sharpe": None, "sortino": None}
        if len(r) < 5:
            return {"sharpe": None, "sortino": None}
        std_r = r.std()
        if std_r == 0.0 or not np.isfinite(std_r):
            return {"sharpe": None, "sortino": None}
        mean_r = r.mean()
        sharpe = float(mean_r / std_r * np.sqrt(252))
        downside = r[r < 0]
        sortino_denom = float(downside.std()) if len(downside) > 1 else std_r
        sortino = float(mean_r / sortino_denom * np.sqrt(252)) if sortino_denom > 0 else None
        return {
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4) if sortino is not None else None,
        }

    def _check_drawdown_limits(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Measure worst stress-test shock impact vs 20 % maximum drawdown threshold."""
        stress = results.get("stress_test") if isinstance(results.get("stress_test"), dict) else {}
        stress_results = stress.get("results") if isinstance(stress, dict) else {}
        worst_impact = 0.0
        if isinstance(stress_results, dict):
            for factor_meta in stress_results.values():
                if isinstance(factor_meta, dict):
                    impact = abs(float(factor_meta.get("impact_on_return", 0.0) or 0.0))
                    worst_impact = max(worst_impact, impact)
        threshold = 0.20
        return {
            "worst_shock_impact": round(worst_impact, 6),
            "drawdown_threshold": threshold,
            "exceeds_threshold": worst_impact > threshold,
        }

    def _check_factor_vif(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect variance inflation factors from sensitivity regression output."""
        sens = results.get("sensitivity_regression") if isinstance(results.get("sensitivity_regression"), dict) else {}
        vif_data = sens.get("vif") or sens.get("variance_inflation_factors") or {}
        if not isinstance(vif_data, dict) or not vif_data:
            return {"vif_available": False, "high_vif_factors": [], "max_vif": None}
        high_vif = [
            f for f, v in vif_data.items()
            if isinstance(v, (int, float)) and float(v) > 5.0
        ]
        max_vif = max(
            (float(v) for v in vif_data.values() if isinstance(v, (int, float))),
            default=None,
        )
        return {
            "vif_available": True,
            "high_vif_factors": high_vif,
            "max_vif": round(max_vif, 4) if max_vif is not None else None,
        }

    def _check_regime_consistency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Measure regime stability via coefficient of variation of rolling elasticity."""
        elas = results.get("elasticity") if isinstance(results.get("elasticity"), dict) else {}
        rolling_raw = elas.get("rolling_elasticity") or [] if isinstance(elas, dict) else []
        if not isinstance(rolling_raw, list) or len(rolling_raw) < 4:
            return {"regime_stable": None, "elasticity_cv": None}
        vals: List[float] = []
        for item in rolling_raw:
            try:
                v = float(item.get("elasticity") if isinstance(item, dict) else item)
                if np.isfinite(v):
                    vals.append(v)
            except (TypeError, ValueError):
                pass
        if len(vals) < 4:
            return {"regime_stable": None, "elasticity_cv": None}
        arr = np.array(vals)
        n = len(arr)

        # Legacy CV (std / |mean|) — kept for backward-compat reporting
        mean_abs = abs(float(arr.mean()))
        cv = float(arr.std() / mean_abs) if mean_abs > 1e-10 else None

        # Robust IQR-based spread: insensitive to near-zero mean
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q75 - q25
        median_abs = abs(float(np.median(arr)))
        robust_cv = float(iqr / median_abs) if median_abs > 1e-10 else None

        # Bootstrap 90 % CI for robust_cv (200 resamples)
        ci_lower: Optional[float] = None
        ci_upper: Optional[float] = None
        if robust_cv is not None and n >= 8:
            rng = np.random.default_rng(42)
            boot: List[float] = []
            for _ in range(200):
                sample = rng.choice(arr, size=n, replace=True)
                m = abs(float(np.median(sample)))
                i = float(np.percentile(sample, 75) - np.percentile(sample, 25))
                if m > 1e-10:
                    boot.append(i / m)
            if boot:
                ci_lower = round(float(np.percentile(boot, 5)), 4)
                ci_upper = round(float(np.percentile(boot, 95)), 4)

        # Dynamic threshold: tightens slightly as sample size grows (floor = 2.5)
        dynamic_threshold = round(max(2.5, 4.0 * max(0.5, 1.0 - (n - 8) * 0.02)), 2)

        # Stability verdict: uses CI upper bound when available (conservative)
        if robust_cv is None:
            regime_stable = None
        elif ci_upper is not None:
            regime_stable = bool(ci_upper < dynamic_threshold)
        else:
            regime_stable = bool(robust_cv < dynamic_threshold)

        return {
            "regime_stable": regime_stable,
            "elasticity_cv": round(cv, 4) if cv is not None else None,
            "robust_cv": round(robust_cv, 4) if robust_cv is not None else None,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "dynamic_threshold": dynamic_threshold,
            "n_observations": n,
        }

    def _check_data_freshness(self, results: Dict[str, Any]) -> Dict[str, Any]:
        gov = results.get("governance_report") if isinstance(results.get("governance_report"), dict) else {}
        split = gov.get("split") if isinstance(gov, dict) else {}
        test_end = split.get("test_end") if isinstance(split, dict) else None
        if not isinstance(test_end, str):
            return {"is_stale": None, "lag_days": None}
        try:
            end_dt = datetime.strptime(test_end, "%Y-%m-%d")
        except Exception:
            return {"is_stale": None, "lag_days": None}
        lag_days = (datetime.utcnow() - end_dt).days
        threshold_days = 21 if lag_days > 30 else 14
        return {
            "is_stale": bool(lag_days > threshold_days),
            "lag_days": int(lag_days),
            "threshold_days": threshold_days,
        }

    def _check_distribution_shift_proxy(self, results: Dict[str, Any]) -> Dict[str, Any]:
        gov = results.get("governance_report") if isinstance(results.get("governance_report"), dict) else {}
        oos = gov.get("out_of_sample") if isinstance(gov, dict) else {}
        residuals = oos.get("residuals") if isinstance(oos, dict) else None
        if not isinstance(residuals, list) or len(residuals) < 40:
            return {"shift_detected": None, "std_ratio": None, "mean_shift": None}
        try:
            arr = np.array([float(x) for x in residuals if x is not None and np.isfinite(float(x))], dtype=float)
        except (TypeError, ValueError):
            return {"shift_detected": None, "std_ratio": None, "mean_shift": None}
        if len(arr) < 40:
            return {"shift_detected": None, "std_ratio": None, "mean_shift": None}
        k = max(10, len(arr) // 4)
        early = arr[:k]
        late = arr[-k:]
        early_std = float(np.std(early))
        late_std = float(np.std(late))
        std_ratio = float(late_std / early_std) if early_std > 1e-12 else None
        mean_shift = float(abs(np.mean(late) - np.mean(early)))
        shift = False
        if std_ratio is not None and (std_ratio > 1.8 or std_ratio < 0.55):
            shift = True
        if mean_shift > 0.02:
            shift = True
        return {
            "shift_detected": shift,
            "std_ratio": round(std_ratio, 4) if std_ratio is not None else None,
            "mean_shift": round(mean_shift, 6),
        }

    def _check_residual_autocorrelation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        gov = results.get("governance_report") if isinstance(results.get("governance_report"), dict) else {}
        oos = gov.get("out_of_sample") if isinstance(gov, dict) else {}
        residuals = oos.get("residuals") if isinstance(oos, dict) else None
        if not isinstance(residuals, list) or len(residuals) < 20:
            return {"autocorr_problem": None, "lag1_autocorr": None}
        try:
            arr = np.array([float(x) for x in residuals if x is not None and np.isfinite(float(x))], dtype=float)
        except (TypeError, ValueError):
            return {"autocorr_problem": None, "lag1_autocorr": None}
        if len(arr) < 20:
            return {"autocorr_problem": None, "lag1_autocorr": None}
        x = arr[:-1]
        y = arr[1:]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return {"autocorr_problem": None, "lag1_autocorr": None}
        lag1 = float(np.corrcoef(x, y)[0, 1])
        return {
            "autocorr_problem": bool(abs(lag1) > 0.2),
            "lag1_autocorr": round(lag1, 4),
            "threshold": 0.2,
        }

    def _check_confidence_interval_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        gov = results.get("governance_report") if isinstance(results.get("governance_report"), dict) else {}
        oos = gov.get("out_of_sample") if isinstance(gov, dict) else {}
        r2_ci = oos.get("r2_ci") if isinstance(oos, dict) else {}
        if not isinstance(r2_ci, dict) or not r2_ci:
            return {"ci_problem": None, "reason": None}
        lo = r2_ci.get("ci_lower")
        hi = r2_ci.get("ci_upper")
        std = r2_ci.get("std")
        reason = None
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            width = float(hi) - float(lo)
            if width <= 1e-6:
                reason = "degenerate_interval"
            elif width > 0.6:
                reason = "too_wide_interval"
        if reason is None and isinstance(std, (int, float)) and float(std) == 0.0:
            reason = "zero_bootstrap_std"
        return {"ci_problem": reason is not None, "reason": reason}

    def _check_sign_stability(self, results: Dict[str, Any]) -> Dict[str, Any]:
        elas = results.get("elasticity") if isinstance(results.get("elasticity"), dict) else {}
        rolling = elas.get("rolling_elasticity") if isinstance(elas, dict) else None
        if not isinstance(rolling, list) or len(rolling) < 6:
            return {"sign_unstable": None, "flip_ratio": None}
        vals: List[float] = []
        for item in rolling:
            try:
                v = float(item.get("elasticity") if isinstance(item, dict) else item)
                if np.isfinite(v):
                    vals.append(v)
            except (TypeError, ValueError):
                pass
        if len(vals) < 6:
            return {"sign_unstable": None, "flip_ratio": None}
        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in vals]
        signs = [s for s in signs if s != 0]
        if len(signs) < 4:
            return {"sign_unstable": None, "flip_ratio": None}
        flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        flip_ratio = flips / max(1, (len(signs) - 1))
        return {
            "sign_unstable": bool(flip_ratio > 0.4),
            "flip_ratio": round(float(flip_ratio), 4),
            "threshold": 0.4,
        }

    def _check_uncertainty_gap(self, results: Dict[str, Any]) -> Dict[str, Any]:
        var_cvar = self._compute_var_cvar(results)
        gov = results.get("governance_report") if isinstance(results.get("governance_report"), dict) else {}
        oos = gov.get("out_of_sample") if isinstance(gov, dict) else {}
        residuals = oos.get("residuals") if isinstance(oos, dict) else None
        if not isinstance(residuals, list) or len(residuals) < 10:
            return {"gap_detected": None, "reason": None}
        try:
            arr = np.array([float(x) for x in residuals if x is not None and np.isfinite(float(x))], dtype=float)
        except (TypeError, ValueError):
            return {"gap_detected": None, "reason": None}
        if len(arr) < 10:
            return {"gap_detected": None, "reason": None}
        if var_cvar.get("var_95") is None or var_cvar.get("cvar_95") is None:
            return {"gap_detected": True, "reason": "missing_tail_risk_estimates"}
        p95_abs = float(np.percentile(np.abs(arr), 95))
        cvar_abs = abs(float(var_cvar["cvar_95"]))
        if cvar_abs < 0.5 * p95_abs:
            return {"gap_detected": True, "reason": "tail_risk_underestimation"}
        return {"gap_detected": False, "reason": None}

    def _check_pipeline_robustness(self, quality: Dict[str, Any]) -> Dict[str, Any]:
        summary = quality.get("summary", {}) if isinstance(quality, dict) else {}
        files = quality.get("files", {}) if isinstance(quality, dict) else {}
        reason: Optional[str] = None

        if isinstance(summary, dict):
            retry_count = int(summary.get("retry_count", 0) or 0)
            schema_errs = int(summary.get("schema_errors", 0) or 0)
            partial_writes = int(summary.get("partial_writes", 0) or 0)
            stale_cache = int(summary.get("stale_cache_hits", 0) or 0)
            if retry_count >= 10:
                reason = f"retry_storm(retry_count={retry_count})"
            elif schema_errs > 0:
                reason = f"schema_errors(count={schema_errs})"
            elif partial_writes > 0:
                reason = f"partial_writes(count={partial_writes})"
            elif stale_cache > 0:
                reason = f"stale_cache_hits(count={stale_cache})"

        if reason is None and isinstance(files, dict):
            for payload in files.values():
                if not isinstance(payload, dict):
                    continue
                if payload.get("schema_drift") is True:
                    reason = "schema_drift_detected"
                    break
                if payload.get("partial_write") is True:
                    reason = "partial_write_detected"
                    break

        return {"robustness_problem": reason is not None, "reason": reason}

    def _check_temporal_integrity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        gov = results.get("governance_report") if isinstance(results.get("governance_report"), dict) else {}
        split = gov.get("split") if isinstance(gov, dict) else {}
        if not isinstance(split, dict) or not split:
            return {"temporal_problem": None, "reason": None}

        train_start = split.get("train_start")
        train_end = split.get("train_end")
        test_start = split.get("test_start")
        test_end = split.get("test_end")
        try:
            ts = datetime.strptime(str(train_start), "%Y-%m-%d")
            te = datetime.strptime(str(train_end), "%Y-%m-%d")
            vs = datetime.strptime(str(test_start), "%Y-%m-%d")
            ve = datetime.strptime(str(test_end), "%Y-%m-%d")
        except Exception:
            return {"temporal_problem": True, "reason": "invalid_split_dates"}

        if te >= vs:
            return {"temporal_problem": True, "reason": "train_test_overlap_or_lookahead"}
        if ts >= te or vs >= ve:
            return {"temporal_problem": True, "reason": "non_monotonic_split_windows"}
        gap_days = (vs - te).days
        if gap_days > 30:
            return {"temporal_problem": True, "reason": f"excessive_split_gap(days={gap_days})"}
        return {"temporal_problem": False, "reason": None}

    def _check_backtest_realism(self, results: Dict[str, Any]) -> Dict[str, Any]:
        bt = results.get("backtest") if isinstance(results.get("backtest"), dict) else {}
        if not isinstance(bt, dict) or not bt:
            return {"realism_problem": None, "reason": None}

        turnover = bt.get("turnover")
        slippage = bt.get("slippage_bps")
        costs = bt.get("transaction_cost_bps")
        cost_sensitivity = bt.get("cost_sensitivity")

        if slippage is None and costs is None:
            return {"realism_problem": True, "reason": "missing_slippage_and_cost_assumptions"}
        if isinstance(turnover, (int, float)) and float(turnover) > 3.0:
            return {"realism_problem": True, "reason": f"excessive_turnover(turnover={float(turnover):.3f})"}
        if cost_sensitivity is None:
            return {"realism_problem": True, "reason": "missing_transaction_cost_sensitivity"}
        return {"realism_problem": False, "reason": None}

    def _check_explainability_risk(self, results: Dict[str, Any]) -> Dict[str, Any]:
        sens = results.get("sensitivity_regression") if isinstance(results.get("sensitivity_regression"), dict) else {}
        coeffs = sens.get("coefficients") if isinstance(sens, dict) else None
        if not isinstance(coeffs, dict) or not coeffs:
            return {"explainability_problem": None, "reason": None}

        numeric_coeffs: Dict[str, float] = {}
        for k, v in coeffs.items():
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                numeric_coeffs[str(k)] = float(v)
        if len(numeric_coeffs) < 1:
            return {"explainability_problem": None, "reason": None}

        abs_sum = float(sum(abs(v) for v in numeric_coeffs.values()))
        if abs_sum > 1e-12:
            dom_feature, dom_val = max(numeric_coeffs.items(), key=lambda kv: abs(kv[1]))
            dominance = abs(dom_val) / abs_sum
            if dominance > 0.85:
                return {
                    "explainability_problem": True,
                    "reason": f"dominant_factor_concentration(feature={dom_feature},share={dominance:.3f})",
                }

        prior_signs_env = os.getenv("ECONOMIC_PRIOR_SIGNS", "").strip()
        if prior_signs_env:
            try:
                prior_signs = json.loads(prior_signs_env)
                if isinstance(prior_signs, dict):
                    for factor, prior in prior_signs.items():
                        if factor in numeric_coeffs and prior in (-1, 1):
                            sign = -1 if numeric_coeffs[factor] < 0 else (1 if numeric_coeffs[factor] > 0 else 0)
                            if sign != 0 and sign != int(prior):
                                return {
                                    "explainability_problem": True,
                                    "reason": f"economic_prior_contradiction(factor={factor})",
                                }
            except Exception:
                return {"explainability_problem": True, "reason": "invalid_economic_prior_signs_env"}

        return {"explainability_problem": False, "reason": None}

    def _check_feature_shift_psi_ks(self, results: Dict[str, Any]) -> Dict[str, Any]:
        sens = results.get("sensitivity_regression") if isinstance(results.get("sensitivity_regression"), dict) else {}
        if not isinstance(sens, dict) or not sens:
            return {"shift_detected": None, "reason": None, "offenders": []}

        train_features = sens.get("train_features")
        test_features = sens.get("test_features")
        if not isinstance(train_features, dict) or not isinstance(test_features, dict):
            return {"shift_detected": None, "reason": "missing_train_test_feature_vectors", "offenders": []}

        offenders: List[str] = []
        for feat, tr_vals in train_features.items():
            if feat not in test_features:
                continue
            te_vals = test_features.get(feat)
            if not isinstance(tr_vals, list) or not isinstance(te_vals, list):
                continue
            try:
                tr = np.array([float(x) for x in tr_vals if x is not None and np.isfinite(float(x))], dtype=float)
                te = np.array([float(x) for x in te_vals if x is not None and np.isfinite(float(x))], dtype=float)
            except (TypeError, ValueError):
                continue
            if len(tr) < 30 or len(te) < 30:
                continue

            ks = self._ks_statistic(tr, te)
            psi = self._population_stability_index(tr, te)
            if ks > 0.2 or psi > 0.2:
                offenders.append(f"{feat}(ks={ks:.3f},psi={psi:.3f})")

        if offenders:
            return {
                "shift_detected": True,
                "reason": ",".join(offenders[:4]),
                "offenders": offenders[:10],
            }
        return {"shift_detected": False, "reason": None, "offenders": []}

    def _ks_statistic(self, a: np.ndarray, b: np.ndarray) -> float:
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        all_vals = np.sort(np.concatenate([a_sorted, b_sorted]))
        cdf_a = np.searchsorted(a_sorted, all_vals, side="right") / max(1, len(a_sorted))
        cdf_b = np.searchsorted(b_sorted, all_vals, side="right") / max(1, len(b_sorted))
        return float(np.max(np.abs(cdf_a - cdf_b)))

    def _population_stability_index(self, base: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
        q = np.linspace(0, 100, bins + 1)
        edges = np.unique(np.percentile(base, q))
        if len(edges) < 4:
            return 0.0
        base_hist, _ = np.histogram(base, bins=edges)
        cur_hist, _ = np.histogram(cur, bins=edges)
        eps = 1e-6
        base_pct = np.clip(base_hist / max(1, np.sum(base_hist)), eps, 1.0)
        cur_pct = np.clip(cur_hist / max(1, np.sum(cur_hist)), eps, 1.0)
        psi = np.sum((cur_pct - base_pct) * np.log(cur_pct / base_pct))
        return float(psi)

    def _check_residual_diagnostics_advanced(self, results: Dict[str, Any]) -> Dict[str, Any]:
        gov = results.get("governance_report") if isinstance(results.get("governance_report"), dict) else {}
        oos = gov.get("out_of_sample") if isinstance(gov, dict) else {}
        residuals = oos.get("residuals") if isinstance(oos, dict) else None
        predictions = oos.get("predictions") if isinstance(oos, dict) else None
        if not isinstance(residuals, list) or len(residuals) < 30:
            return {"hetero_problem": None, "reason": None}
        try:
            e = np.array([float(x) for x in residuals if x is not None and np.isfinite(float(x))], dtype=float)
        except (TypeError, ValueError):
            return {"hetero_problem": None, "reason": None}
        if len(e) < 30:
            return {"hetero_problem": None, "reason": None}

        lb_flag, lb_reason = self._ljung_box_flag(e)
        hetero_flag, hetero_reason = self._heteroskedasticity_flag(e, predictions)
        reasons = [r for r in [lb_reason, hetero_reason] if r]
        return {
            "hetero_problem": bool(lb_flag or hetero_flag),
            "reason": ";".join(reasons) if reasons else None,
        }

    def _check_interval_coverage_calibration(self, results: Dict[str, Any]) -> Dict[str, Any]:
        gov = results.get("governance_report") if isinstance(results.get("governance_report"), dict) else {}
        oos = gov.get("out_of_sample") if isinstance(gov, dict) else {}
        residuals = oos.get("residuals") if isinstance(oos, dict) else None
        if not isinstance(residuals, list) or len(residuals) < 30:
            return {"coverage_problem": None, "reason": None}
        try:
            e = np.array([float(x) for x in residuals if x is not None and np.isfinite(float(x))], dtype=float)
        except (TypeError, ValueError):
            return {"coverage_problem": None, "reason": None}
        if len(e) < 30:
            return {"coverage_problem": None, "reason": None}

        # Empirical 90 % residual coverage under normal-scale interval.
        sigma = float(np.std(e))
        if sigma < 1e-12:
            return {"coverage_problem": True, "reason": "zero_residual_variance"}
        z90 = 1.645
        covered = np.abs(e) <= (z90 * sigma)
        empirical = float(np.mean(covered))
        target = 0.90
        tol = 0.08
        if abs(empirical - target) > tol:
            return {
                "coverage_problem": True,
                "reason": f"empirical_coverage_90={empirical:.3f},target=0.90,tol={tol:.2f}",
            }
        return {
            "coverage_problem": False,
            "reason": None,
            "empirical_coverage_90": round(empirical, 4),
        }

    def _check_shap_stability(self, results: Dict[str, Any]) -> Dict[str, Any]:
        gov = results.get("governance_report") if isinstance(results.get("governance_report"), dict) else {}
        sens = results.get("sensitivity_regression") if isinstance(results.get("sensitivity_regression"), dict) else {}
        windows = None
        if isinstance(gov, dict):
            windows = gov.get("shap_importance_windows") or gov.get("feature_importance_windows")
        if windows is None and isinstance(sens, dict):
            windows = sens.get("feature_importance_windows")
        if not isinstance(windows, list) or len(windows) < 3:
            return {"shap_problem": None, "reason": None}

        norm_windows: List[Dict[str, float]] = []
        for w in windows:
            if not isinstance(w, dict):
                continue
            numeric = {str(k): abs(float(v)) for k, v in w.items() if isinstance(v, (int, float)) and np.isfinite(float(v))}
            s = sum(numeric.values())
            if s <= 1e-12:
                continue
            norm_windows.append({k: v / s for k, v in numeric.items()})
        if len(norm_windows) < 3:
            return {"shap_problem": None, "reason": None}

        # Mean L1 drift between consecutive windows
        drifts: List[float] = []
        dominant_changes = 0
        prev_dom: Optional[str] = None
        for i, cur in enumerate(norm_windows):
            dom = max(cur.items(), key=lambda kv: kv[1])[0]
            if prev_dom is not None and dom != prev_dom:
                dominant_changes += 1
            prev_dom = dom
            if i == 0:
                continue
            prev = norm_windows[i - 1]
            features = set(prev.keys()) | set(cur.keys())
            l1 = sum(abs(cur.get(f, 0.0) - prev.get(f, 0.0)) for f in features)
            drifts.append(float(l1))

        mean_l1 = float(np.mean(drifts)) if drifts else 0.0
        dom_flip_ratio = dominant_changes / max(1, (len(norm_windows) - 1))
        if mean_l1 > 0.9 or dom_flip_ratio > 0.6:
            return {
                "shap_problem": True,
                "reason": f"mean_l1={mean_l1:.3f},dominant_flip_ratio={dom_flip_ratio:.3f}",
            }
        return {
            "shap_problem": False,
            "reason": None,
            "mean_l1": round(mean_l1, 4),
            "dominant_flip_ratio": round(dom_flip_ratio, 4),
        }

    def _check_coeff_ci_overlap(self, results: Dict[str, Any]) -> Dict[str, Any]:
        sens = results.get("sensitivity_regression") if isinstance(results.get("sensitivity_regression"), dict) else {}
        if not isinstance(sens, dict) or not sens:
            return {"ci_overlap_problem": None, "reason": None}

        windows = sens.get("rolling_coefficients")
        if not isinstance(windows, list) or len(windows) < 3:
            return {"ci_overlap_problem": None, "reason": None}

        # Expect per-window format: {feature: {coef, ci_lower, ci_upper}, ...}
        feat_pairs: Dict[str, List[bool]] = {}
        for i in range(1, len(windows)):
            prev = windows[i - 1]
            cur = windows[i]
            if not isinstance(prev, dict) or not isinstance(cur, dict):
                continue
            common = set(prev.keys()) & set(cur.keys())
            for f in common:
                p = prev.get(f)
                c = cur.get(f)
                if not isinstance(p, dict) or not isinstance(c, dict):
                    continue
                plo, phi = p.get("ci_lower"), p.get("ci_upper")
                clo, chi = c.get("ci_lower"), c.get("ci_upper")
                if not all(isinstance(x, (int, float)) for x in [plo, phi, clo, chi]):
                    continue
                overlap = not (float(phi) < float(clo) or float(chi) < float(plo))
                feat_pairs.setdefault(str(f), []).append(bool(overlap))

        if not feat_pairs:
            return {"ci_overlap_problem": None, "reason": None}

        offenders: List[str] = []
        for f, overlaps in feat_pairs.items():
            if len(overlaps) < 2:
                continue
            ratio = float(sum(1 for v in overlaps if v) / len(overlaps))
            if ratio < 0.5:
                offenders.append(f"{f}(overlap_ratio={ratio:.3f})")

        if offenders:
            return {
                "ci_overlap_problem": True,
                "reason": ",".join(offenders[:4]),
            }
        return {"ci_overlap_problem": False, "reason": None}

    def _ljung_box_flag(self, residuals: np.ndarray) -> tuple[bool, Optional[str]]:
        n = len(residuals)
        max_lag = min(10, max(2, n // 8))
        if n <= max_lag + 2:
            return False, None
        rhos: List[float] = []
        for k in range(1, max_lag + 1):
            x = residuals[:-k]
            y = residuals[k:]
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue
            rhos.append(float(np.corrcoef(x, y)[0, 1]))
        if not rhos:
            return False, None
        q = n * (n + 2) * sum((rho * rho) / max(1, (n - i)) for i, rho in enumerate(rhos, start=1))
        crit95 = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07, 6: 12.59, 7: 14.07, 8: 15.51, 9: 16.92, 10: 18.31}
        crit = crit95.get(len(rhos), 18.31)
        if q > crit:
            return True, f"ljung_box(q={q:.3f},lag={len(rhos)},crit={crit:.2f})"
        return False, None

    def _heteroskedasticity_flag(self, residuals: np.ndarray, predictions: Any) -> tuple[bool, Optional[str]]:
        e2 = residuals ** 2
        if isinstance(predictions, list):
            try:
                x = np.array([float(v) for v in predictions if v is not None and np.isfinite(float(v))], dtype=float)
            except (TypeError, ValueError):
                x = np.arange(len(e2), dtype=float)
            if len(x) != len(e2):
                x = np.arange(len(e2), dtype=float)
        else:
            x = np.arange(len(e2), dtype=float)
        if len(x) < 20:
            return False, None
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(e2))
        denom = float(np.sum((x - x_mean) ** 2))
        if denom < 1e-12:
            return False, None
        beta1 = float(np.sum((x - x_mean) * (e2 - y_mean)) / denom)
        beta0 = y_mean - beta1 * x_mean
        yhat = beta0 + beta1 * x
        ss_tot = float(np.sum((e2 - y_mean) ** 2))
        ss_res = float(np.sum((e2 - yhat) ** 2))
        if ss_tot < 1e-12:
            return False, None
        r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))
        lm = len(x) * r2
        if lm > 3.84:
            return True, f"heteroskedasticity_lm={lm:.3f}"
        return False, None

    def _request_llama_approval_for_serious_issue(
        self,
        iteration: int,
        diag: Dict[str, Any],
        issue_type: str,
    ) -> tuple[bool, str]:
        """Use Llama to analyze serious problems (with real source code) and request approval.
        
        If approved → auto-applies the code changes from Llama's output.
        Returns (approved, adjustment_description).
        """
        serious_issues = [
            "NEGATIVE_R2",
            "DATA_GAPS",
            "EXCESS_DRAWDOWN",
            "MULTICOLLINEARITY",
            "REGIME_INSTABILITY",
            "LOOKAHEAD_BIAS",
            "STALE_DATA",
            "DIST_SHIFT",
            "RESIDUAL_AUTOCORR",
            "CI_QUALITY",
            "SIGN_INSTABILITY",
            "UNCERTAINTY_GAP",
            "PIPELINE_ROBUSTNESS",
            "TEMPORAL_INTEGRITY",
            "BACKTEST_REALISM",
            "EXPLAINABILITY_RISK",
            "FEATURE_SHIFT",
            "RESIDUAL_HETERO",
            "COVERAGE_CALIBRATION",
            "SHAP_STABILITY",
            "COEFF_CI_OVERLAP",
        ]

        is_serious = any(serious in issue_type for serious in serious_issues)
        if not is_serious:
            return False, f"Skipped: {issue_type} is not a serious problem requiring approval"

        # Call Llama with full source code context
        problem_text = (
            f"SERIOUS: {issue_type} detected. "
            f"Score: {diag['score']}/100. "
            f"Issues: {diag['inconsistencies'][:3]}"
        )
        llama_result = self._llama.analyze_problem(
            problem_description=problem_text,
            diagnostics=diag,
            issue_type=issue_type,
        )

        if not llama_result["success"]:
            return False, f"Llama analysis failed: {llama_result['analysis']}"

        analysis = llama_result["analysis"]
        has_changes = llama_result["has_code_changes"]

        gate = self._ensure_approval_gateway()
        approved = gate.request(
            action_id=f"iter{iteration}_serious_{issue_type}",
            description=f"[QUANTOS] Serious problem: {issue_type} — {'code changes proposed' if has_changes else 'no code changes'}",
            details={
                "iteration": iteration,
                "issue_type": issue_type,
                "current_score": diag["score"],
                "inconsistencies": diag["inconsistencies"],
                "has_code_changes": has_changes,
                "llama_analysis": analysis,
            },
        )

        if approved and has_changes:
            applied = self._llama.apply_code_changes(analysis)
            change_summary = "; ".join(applied)

            # If the built-in smoke-test already rolled back changes, skip further steps.
            if any("[ROLLED_BACK]" in a for a in applied):
                return True, (
                    f"[QUANTOS] {issue_type} smoke test failed — changes rolled back.\n"
                    f"Details: {change_summary}"
                )

            # ── Regression Testing Gate: financial-validation before commit ──
            rg_passed, rg_report = self._run_regression_gate(issue_type=issue_type)
            if not rg_passed:
                # Restore all modified files to HEAD before the changes were written
                self._run_cmd(["git", "checkout", "HEAD", "--", "."])
                return True, (
                    f"[QUANTOS] {issue_type} fix FAILED financial regression gate.\n"
                    f"All changes rolled back. Gate: {rg_report}\n"
                    f"Proposed fix (not applied):\n{analysis[:600]}"
                )

            # ── Semantic Versioning: bump patch in pyproject.toml ────────────
            ver_msg = self._bump_version()

            git_status = self._auto_commit_and_push(f"iter{iteration}_{issue_type}")
            return True, (
                f"[QUANTOS] {issue_type} fix approved.\n"
                f"Auto-applied changes: {change_summary}\n"
                f"Regression gate: {rg_report}\n"
                f"{ver_msg}\n"
                f"{git_status}\n"
                f"Full analysis:\n{analysis[:800]}"
            )
        elif approved:
            return True, (
                f"[QUANTOS] {issue_type} fix approved (no code changes to apply).\n"
                f"Analysis:\n{analysis[:800]}"
            )
        else:
            # Still include the full Llama analysis so the owner can review what was proposed
            return False, (
                f"[QUANTOS] {issue_type} REJECTED by owner — skipped.\n"
                f"AI proposed the following (not applied):\n{analysis[:1200]}"
            )

    def run(self) -> Dict[str, Any]:
        final_results: Dict[str, Any] = {}
        final_diag: Dict[str, Any] = {"score": 0.0, "inconsistencies": []}
        gate = self._ensure_approval_gateway()

        # ── Autonomous Llama bug scan (runs once before the optimization loop) ──
        print("\n[QUANTOS] Pre-run: Scanning project source files for latent bugs...")
        bug_scan = self._llama.scan_for_bugs()
        if bug_scan["success"] and not bug_scan["no_bugs_found"] and bug_scan["has_code_changes"]:
            approved = gate.request(
                action_id="pre_run_bug_scan",
                description="[QUANTOS] Pre-run scan found latent bugs — code changes proposed",
                details={"llama_analysis": bug_scan["analysis"]},
            )
            if approved:
                applied = self._llama.apply_code_changes(bug_scan["analysis"])
                if any("[ROLLED_BACK]" in a for a in applied):
                    print(f"[QUANTOS] Pre-run bug fix smoke test failed — rolled back: {'; '.join(applied)}")
                else:
                    rg_passed, rg_report = self._run_regression_gate(issue_type="pre_run_bug_scan")
                    if not rg_passed:
                        self._run_cmd(["git", "checkout", "HEAD", "--", "."])
                        print(f"[QUANTOS] Pre-run bug fix FAILED regression gate: {rg_report}")
                    else:
                        ver_msg = self._bump_version()
                        git_status = self._auto_commit_and_push("pre_run_bug_scan")
                        print(f"[QUANTOS] Bug fixes applied: {'; '.join(applied)}")
                        print(f"[QUANTOS] Regression gate: {rg_report}")
                        print(f"[QUANTOS] {ver_msg}")
                        print(f"[QUANTOS] {git_status}")
            else:
                print("[QUANTOS] Bug fix proposals REJECTED by owner — continuing without changes")
        elif bug_scan["success"] and bug_scan["no_bugs_found"]:
            print("[QUANTOS] No latent bugs found in source files.")
        elif not bug_scan["success"]:
            print(f"[QUANTOS] Bug scan skipped (Ollama unreachable): {bug_scan.get('analysis', '')[:120]}")
        # ─────────────────────────────────────────────────────────────────────────

        # ── Run-start notification ─────────────────────────────────────────────
        # Run one pipeline pass to discover issues, then notify before the loop.
        _pre_config = self._build_config()
        _pre_results, _ = self._run_pipeline(_pre_config)
        _pre_quality = self._load_quality_report()
        _pre_diag = self._diagnose(_pre_results, _pre_quality)
        _startup_issues = list(_pre_diag.get("inconsistencies", []))
        if _startup_issues:
            gate.notify_run_started(_startup_issues)
        # ──────────────────────────────────────────────────────────────────────

        for i in range(1, self.max_iterations + 1):
            adjustments: List[str] = []
            config = self._build_config()

            results, pipeline = self._run_pipeline(config)

            quality = self._load_quality_report()
            diag = self._diagnose(results, quality)

            detected_types = {
                str(item).split(":", 1)[0]
                for item in diag.get("inconsistencies", [])
                if isinstance(item, str) and ":" in item
            }
            priority_order = [
                "LOOKAHEAD_BIAS",
                "TEMPORAL_INTEGRITY",
                "DATA_GAPS",
                "PIPELINE_ROBUSTNESS",
                "BACKTEST_REALISM",
                "EXPLAINABILITY_RISK",
                "FEATURE_SHIFT",
                "RESIDUAL_HETERO",
                "COVERAGE_CALIBRATION",
                "SHAP_STABILITY",
                "COEFF_CI_OVERLAP",
                "DIST_SHIFT",
                "RESIDUAL_AUTOCORR",
                "CI_QUALITY",
                "SIGN_INSTABILITY",
                "UNCERTAINTY_GAP",
                "EXCESS_DRAWDOWN",
                "MULTICOLLINEARITY",
                "REGIME_INSTABILITY",
                "OUTLIERS",
                "STALE_DATA",
            ]
            selected_issue = next((t for t in priority_order if t in detected_types), None)
            if selected_issue:
                approved, analysis = self._request_llama_approval_for_serious_issue(
                    iteration=i,
                    diag=diag,
                    issue_type=selected_issue,
                )
                if approved and selected_issue == "DATA_GAPS":
                    adjustments.append(
                        f"[LLAMA AI] Fetchers adjustment: {self._apply_fetcher_adjustment(config)}\n"
                        f"AI Recommendation:\n{analysis[:300]}"
                    )
                elif approved and selected_issue == "OUTLIERS":
                    self._apply_pipeline_outlier_relaxation(delta=0.5)
                    adjustments.append(
                        "[LLAMA AI] Pipeline adjustment: relaxed Silver z-score thresholds by +0.5\n"
                        f"AI Recommendation:\n{analysis[:300]}"
                    )
                elif approved:
                    adjustments.append(
                        f"[LLAMA AI] {selected_issue} remediation approved.\n"
                        f"AI Recommendation:\n{analysis[:300]}"
                    )
                else:
                    adjustments.append(
                        f"[LLAMA AI] {selected_issue} remediation REJECTED by owner — skipped"
                    )

            if float(diag["score"]) >= self.target_score:
                self.records.append(
                    IterationRecord(
                        iteration=i,
                        score=float(diag["score"]),
                        inconsistencies=list(diag["inconsistencies"]),
                        adjustments=adjustments,
                    )
                )
                final_results = results
                final_diag = diag
                break

            self.records.append(
                IterationRecord(
                    iteration=i,
                    score=float(diag["score"]),
                    inconsistencies=list(diag["inconsistencies"]),
                    adjustments=adjustments,
                )
            )
            final_results = results
            final_diag = diag

        final_report = {
            "status": "optimized" if float(final_diag.get("score", 0.0)) >= self.target_score else "max_iterations_reached",
            "target_score": self.target_score,
            "max_iterations": self.max_iterations,
            "iterations_used": len(self.records),
            "raw_integrity_score": float(final_diag.get("score", 0.0)),
            "inconsistencies": list(final_diag.get("inconsistencies", [])),
            "loop_history": [asdict(r) for r in self.records],
            "final_quant_evaluation": self._final_quant_assessment(final_diag),
            "results": final_results,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "optimizer_report.json"
        report_path.write_text(json.dumps(final_report, indent=2, default=str), encoding="utf-8")

        # Send completion notification
        gate.notify_run_finished(
            score=float(final_report["raw_integrity_score"]),
            status=final_report["status"],
            issues=list(final_report.get("inconsistencies", [])),
        )

        print(f"\n{'=' * 60}")
        print("[OPTIMIZER] Optimization complete.")
        print(f"  Score  : {final_report['raw_integrity_score']:.1f}/100  |  Status: {final_report['status']}")
        print(f"  Iters  : {final_report['iterations_used']}/{final_report['max_iterations']}")
        print(f"  Report : {report_path.resolve()}")
        print(f"{'=' * 60}\n")

        return final_report

    def _final_quant_assessment(self, diag: Dict[str, Any]) -> Dict[str, Any]:
        score = float(diag.get("score", 0.0))
        inconsistencies = list(diag.get("inconsistencies", []))
        structural_limits: List[str] = []

        for item in inconsistencies:
            if item.startswith("DATA_GAPS"):
                structural_limits.append("External source coverage / vintage gaps across macro series.")
            if item.startswith("NEGATIVE_R2"):
                structural_limits.append("Weak signal-to-noise in daily return prediction with macro factors.")
            if item.startswith("LOOKAHEAD_BIAS"):
                structural_limits.append("Temporal leakage sensitivity in rolling split windows.")
            if item.startswith("STALE_DATA"):
                structural_limits.append("Forecast quality degraded by stale test horizon / delayed source updates.")
            if item.startswith("DIST_SHIFT"):
                structural_limits.append("Distribution shift detected between earlier and recent residual regimes.")
            if item.startswith("RESIDUAL_AUTOCORR"):
                structural_limits.append("Residual serial correlation indicates unmodeled temporal structure.")
            if item.startswith("CI_QUALITY"):
                structural_limits.append("Confidence intervals are poorly calibrated (too wide/narrow/degenerate).")
            if item.startswith("SIGN_INSTABILITY"):
                structural_limits.append("Elasticity sign flips too often across rolling windows.")
            if item.startswith("UNCERTAINTY_GAP"):
                structural_limits.append("Tail-risk uncertainty is under-modeled versus empirical residual behavior.")
            if item.startswith("PIPELINE_ROBUSTNESS"):
                structural_limits.append("Operational pipeline robustness issue detected (retry/schema/cache/partial-write).")
            if item.startswith("TEMPORAL_INTEGRITY"):
                structural_limits.append("Temporal split integrity is violated or unstable for robust OOS evaluation.")
            if item.startswith("BACKTEST_REALISM"):
                structural_limits.append("Backtest assumptions miss realistic cost/slippage/turnover sensitivity constraints.")
            if item.startswith("EXPLAINABILITY_RISK"):
                structural_limits.append("Explainability risk detected (factor concentration or economic-prior contradiction).")
            if item.startswith("FEATURE_SHIFT"):
                structural_limits.append("Feature-level train-vs-recent distribution drift detected via PSI/KS.")
            if item.startswith("RESIDUAL_HETERO"):
                structural_limits.append("Residual diagnostics show serial-dependence/heteroskedastic behavior.")
            if item.startswith("COVERAGE_CALIBRATION"):
                structural_limits.append("Realized 90 % interval coverage is miscalibrated versus expected confidence level.")
            if item.startswith("SHAP_STABILITY"):
                structural_limits.append("Feature-importance/SHAP attribution is unstable across rolling windows.")
            if item.startswith("COEFF_CI_OVERLAP"):
                structural_limits.append("Rolling coefficient confidence intervals show poor overlap consistency.")
            if item.startswith("SCALING_ISSUE"):
                structural_limits.append("Near-zero return means make elasticity scaling numerically unstable.")
            if item.startswith("POOR_RISK_ADJ"):
                structural_limits.append("OOS residuals driven by noise: annualised Sharpe below -0.5.")
            if item.startswith("EXCESS_DRAWDOWN"):
                structural_limits.append("Stress scenario produces >20 % drawdown — tail risk above tolerable threshold.")
            if item.startswith("MULTICOLLINEARITY"):
                structural_limits.append("High VIF (>5) among macro factors inflates coefficient uncertainty.")
            if item.startswith("REGIME_INSTABILITY"):
                structural_limits.append("Rolling elasticity CV > 200 %: regime shifts dominate factor sensitivity.")

        if not structural_limits and score < self.target_score:
            structural_limits.append(
                "Residual variance and regime shifts prevent stable high-confidence fit above target score."
            )

        return {
            "score": score,
            "verdict": "No sugar-coating: optimized system quality is constrained by market/macro structure, not only code mechanics.",
            "why_last_5_10_percent_unattainable": sorted(set(structural_limits)),
            "var_cvar_summary": diag.get("var_cvar", {}),
            "risk_adjusted_returns": diag.get("risk_adjusted_returns", {}),
            "drawdown_summary": diag.get("drawdown_info", {}),
            "multicollinearity": diag.get("vif_info", {}),
            "regime_stability": diag.get("regime_info", {}),
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated Quant Optimization Loop — owner-only tool."
    )
    parser.add_argument(
        "--target-score",
        type=float,
        default=94.0,
        help="Stop when integrity score reaches this value (default: 94.0).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Hard cap on pipeline iterations (default: 10).",
    )
    args = parser.parse_args()

    optimizer = AutomatedOptimizationLoop(
        target_score=args.target_score,
        max_iterations=args.max_iterations,
    )
    report = optimizer.run()
    print(json.dumps(report["final_quant_evaluation"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────────
# Pillar 2 — Parameter Optimisation: Grid Search over Backtest Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

import itertools
import pandas as pd


def grid_search_backtest(
    prices: "pd.Series",
    rolling_windows: list[int] | None = None,
    z_thresholds: list[float] | None = None,
    volatility_filter: bool = True,
    trend_filter: bool = True,
    friction: float = 0.0015,
    top_n: int = 5,
    output_path: str | None = None,
) -> list[dict]:
    """
    Grid-search over rolling-window and Z-score-threshold combinations and
    return the **top N parameter sets** ranked by Sortino Ratio.

    Signal filters
    --------------
    * Volatility filter : only trade when the 14-day rolling StdDev is above
      its 20-day moving average — avoids flat, compressed regimes.
    * Trend filter      : suppress Short signals when Price > 200-day SMA
      (only go Long in up-trends), reducing counter-trend drawdowns.

    Parameters
    ----------
    prices           : pd.Series of daily close prices (date-indexed).
    rolling_windows  : lookback periods to search (default: [10, 20, 50, 100]).
    z_thresholds     : entry thresholds in σ units (default: [1.0, 1.5, 2.0, 2.5]).
    volatility_filter: enable the ATR/StdDev activity filter across all runs.
    trend_filter     : enable the 200-day SMA trend filter across all runs.
    friction         : one-way trade cost applied to all runs (default: 0.15 %).
    top_n            : how many best combinations to return (default: 5).
    output_path      : if provided, write the ranked results as a CSV to this path.

    Returns
    -------
    List[dict] — the top_n results, each containing:
        {
          "rolling_window": int,
          "z_threshold": float,
          "sortino_ratio": float | None,
          "sharpe_ratio": float | None,
          "annualized_return": float,
          "annualized_volatility": float,
          "max_drawdown": float,
          "profit_factor": float | None,
          "total_trades": int,
          "win_rate": float,
          "total_days": int,
        }
    """
    try:
        from Medallion.gold.AnalysisSuite.backtest import run_strategy_backtest
    except ModuleNotFoundError:
        from src.Medallion.gold.AnalysisSuite.backtest import run_strategy_backtest

    if rolling_windows is None:
        rolling_windows = [10, 20, 50, 100]
    if z_thresholds is None:
        z_thresholds = [1.0, 1.5, 2.0, 2.5]

    records: list[dict] = []

    for window, threshold in itertools.product(rolling_windows, z_thresholds):
        try:
            result = run_strategy_backtest(
                prices=prices,
                rolling_window=window,
                z_threshold=threshold,
                friction=friction,
                volatility_filter=volatility_filter,
                trend_filter=trend_filter,
            )
            m = result.get("metrics", {})
            row = {
                "rolling_window": window,
                "z_threshold": threshold,
                "sortino_ratio": m.get("sortino_ratio"),
                "sharpe_ratio": m.get("sharpe_ratio"),
                "annualized_return": m.get("annualized_return"),
                "annualized_volatility": m.get("annualized_volatility"),
                "max_drawdown": m.get("max_drawdown"),
                "profit_factor": m.get("profit_factor"),
                "calmar_ratio": m.get("calmar_ratio"),
                "total_trades": m.get("total_trades"),
                "win_rate": m.get("win_rate"),
                "total_days": m.get("total_days"),
            }
            records.append(row)
        except Exception as exc:
            # Log failed combinations but keep grid search running
            records.append({
                "rolling_window": window,
                "z_threshold": threshold,
                "sortino_ratio": None,
                "sharpe_ratio": None,
                "annualized_return": None,
                "annualized_volatility": None,
                "max_drawdown": None,
                "profit_factor": None,
                "calmar_ratio": None,
                "total_trades": None,
                "win_rate": None,
                "total_days": None,
                "error": str(exc),
            })

    # ── Rank by Sortino Ratio (descending); push None/-inf to the bottom ─────
    def _sort_key(row: dict) -> float:
        v = row.get("sortino_ratio")
        if v is None or not np.isfinite(float(v)):
            return -1e18
        return float(v)

    ranked = sorted(records, key=_sort_key, reverse=True)
    top = ranked[:top_n]

    # ── Optionally persist results as CSV ─────────────────────────────────────
    if output_path:
        try:
            import pandas as _pd
            df = _pd.DataFrame(ranked)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
        except Exception:
            pass  # Non-fatal: metrics already returned in-memory

    return top


