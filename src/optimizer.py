from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from Fetchers.Factory import DataFactory
from Fetchers.ProjectConfig import ProjectConfig, RunMode
from Medallion.MedallionPipeline import MedallionPipeline
from Medallion.gold.AnalysisSuite.sensitivity_reg import sensitivity_reg
from Medallion.silver import contracts as silver_contracts


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

    SYSTEM_PROMPT = """You are a Senior Quant Data Fixer AI Agent embedded in a financial scenario-planner pipeline.

You have access to the actual Python source code of the project. Use it to give PRECISE, FILE-SPECIFIC fixes.

INPUT:
- Task: description of the detected problem
- Context: diagnostic metrics (score, issues, VaR, etc.)
- Source Code: actual relevant Python files from the project

RULES:
- Only propose fixes for SERIOUS problems (negative R², data gaps, multicollinearity, drawdown >20%, regime instability)
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

    def __init__(self, src_root: Optional[Path] = None, timeout_seconds: int = 120):
        self._timeout = timeout_seconds
        self._src = src_root or Path(__file__).parent
        self._verify_connection()

    def _verify_connection(self) -> bool:
        try:
            response = requests.post(
                self.OLLAMA_API_URL,
                json={"model": self.MODEL_NAME, "prompt": "ping", "stream": False},
                timeout=5,
            )
            return response.status_code == 200
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
        """Send prompt to Llama and return raw response text."""
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
                if not full_path.exists():
                    applied.append(f"[SKIP] File not found: {rel_path}")
                    continue
                original = full_path.read_text(encoding="utf-8")
                if old_code not in original:
                    applied.append(
                        f"[SKIP] Old code not found in {rel_path} — may already be fixed or Llama hallucinated"
                    )
                    continue
                updated = original.replace(old_code, new_code, 1)
                full_path.write_text(updated, encoding="utf-8")
                applied.append(f"[APPLIED] {rel_path}: replaced {len(old_code)} chars")

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
        banner = (
            "\n" + "!" * 76 + "\n"
            "[OPTIMIZER] OWNER APPROVAL REQUIRED (PENDING)\n"
            f"Action ID : {entry.get('action_id')}\n"
            f"Action    : {entry.get('description')}\n"
            f"Timeout   : {self._timeout} seconds\n"
            f"Queue     : {self._queue_path}\n"
            "Respond   : YES/NO in terminal OR use respond_to_approval.py --approve/--reject\n"
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
        title = "Scenario Planner: Approval Required"
        action = str(entry.get("description", "Approval pending"))
        body = (
            f"{action}\n\n"
            f"Timeout: {self._timeout}s\n"
            f"Queue: {self._queue_path}\n\n"
            "Respond with YES/NO in terminal or run:\n"
            "respond_to_approval.py --approve / --reject"
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

    def _notify_mobile(self, entry: Dict[str, Any]) -> None:
        """Best-effort mobile notifications via webhook or Telegram (if env vars are set)."""
        action = str(entry.get("description", "Approval pending"))
        message = (
            "[Scenario Planner] Approval required\n"
            f"Action: {action}\n"
            f"Timeout: {self._timeout}s\n"
            f"Queue: {self._queue_path}"
        )

        # Generic webhook target (recommended for mobile via ntfy/IFTTT/Make/Zapier)
        webhook_url = os.getenv("MOBILE_NOTIFY_WEBHOOK_URL", "").strip()
        if webhook_url:
            try:
                requests.post(
                    webhook_url,
                    json={
                        "title": "Scenario Planner Approval Required",
                        "message": message,
                        "action_id": entry.get("action_id"),
                        "requested_at": entry.get("requested_at"),
                    },
                    timeout=5,
                )
            except Exception:
                pass

        # Telegram fallback (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if bot_token and chat_id:
            try:
                telegram_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                requests.post(
                    telegram_url,
                    json={"chat_id": chat_id, "text": message},
                    timeout=5,
                )
            except Exception:
                pass

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
        while time.time() < deadline:
            try:
                data = json.loads(self._queue_path.read_text(encoding="utf-8"))
                status = data.get("status", "pending")
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
            timeout_seconds=120,
        )

    def _ensure_approval_gateway(self) -> ApprovalGateway:
        """Lazily create the approval gateway so output_dir exists first."""
        if self._approval is None:
            self._approval = ApprovalGateway(
                base_output_path=self.output_dir.parent,
                timeout_seconds=120,
                force_non_interactive=False,
            )
        return self._approval

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
            description=f"[LLAMA AI] Serious problem: {issue_type} — {'code changes proposed' if has_changes else 'no code changes'}",
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
            return True, (
                f"[LLAMA AI] {issue_type} fix approved.\n"
                f"Auto-applied changes: {change_summary}\n"
                f"Full analysis:\n{analysis[:400]}"
            )
        elif approved:
            return True, (
                f"[LLAMA AI] {issue_type} fix approved (no code changes to apply).\n"
                f"Analysis:\n{analysis[:400]}"
            )
        else:
            return False, f"[LLAMA AI] {issue_type} fix REJECTED by owner — skipped"

    def run(self) -> Dict[str, Any]:
        final_results: Dict[str, Any] = {}
        final_diag: Dict[str, Any] = {"score": 0.0, "inconsistencies": []}
        gate = self._ensure_approval_gateway()

        # ── Autonomous Llama bug scan (runs once before the optimization loop) ──
        print("\n[LLAMA AI] Pre-run: Scanning project source files for latent bugs...")
        bug_scan = self._llama.scan_for_bugs()
        if bug_scan["success"] and not bug_scan["no_bugs_found"] and bug_scan["has_code_changes"]:
            approved = gate.request(
                action_id="pre_run_bug_scan",
                description="[LLAMA AI] Pre-run scan found latent bugs — code changes proposed",
                details={"llama_analysis": bug_scan["analysis"]},
            )
            if approved:
                applied = self._llama.apply_code_changes(bug_scan["analysis"])
                print(f"[LLAMA AI] Bug fixes applied: {'; '.join(applied)}")
            else:
                print("[LLAMA AI] Bug fix proposals REJECTED by owner — continuing without changes")
        elif bug_scan["success"] and bug_scan["no_bugs_found"]:
            print("[LLAMA AI] No latent bugs found in source files.")
        elif not bug_scan["success"]:
            print(f"[LLAMA AI] Bug scan skipped (Ollama unreachable): {bug_scan.get('analysis', '')[:120]}")
        # ─────────────────────────────────────────────────────────────────────────

        for i in range(1, self.max_iterations + 1):
            adjustments: List[str] = []
            config = self._build_config()

            results, pipeline = self._run_pipeline(config)

            quality = self._load_quality_report()
            diag = self._diagnose(results, quality)

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

            if diag["has_data_gaps"]:
                # Use Llama AI to analyze and request approval for serious data gap issue
                approved, analysis = self._request_llama_approval_for_serious_issue(
                    iteration=i,
                    diag=diag,
                    issue_type="DATA_GAPS",
                )
                if approved:
                    adjustments.append(f"[LLAMA AI] Fetchers adjustment: {self._apply_fetcher_adjustment(config)}\nAI Recommendation:\n{analysis[:300]}")
                else:
                    adjustments.append("[LLAMA AI] DATA_GAPS fix REJECTED by owner — skipped")

            if float(diag.get("outlier_ratio", 0.0)) > 0.05:
                # Use Llama AI to analyze and request approval for outlier issue
                approved, analysis = self._request_llama_approval_for_serious_issue(
                    iteration=i,
                    diag=diag,
                    issue_type="OUTLIERS",
                )
                if approved:
                    self._apply_pipeline_outlier_relaxation(delta=0.5)
                    adjustments.append(f"[LLAMA AI] Pipeline adjustment: relaxed Silver z-score thresholds by +0.5\nAI Recommendation:\n{analysis[:300]}")
                else:
                    adjustments.append("[LLAMA AI] Outlier relaxation REJECTED by owner — skipped")

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
