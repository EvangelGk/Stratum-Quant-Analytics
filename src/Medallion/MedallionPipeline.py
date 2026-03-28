import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from exceptions.MedallionExceptions import (
    ComplianceViolationError,
    ParallelExecutionError,
)
from Fetchers.Factory import DataFactory
from Fetchers.ProjectConfig import ProjectConfig
from logger.Catalog import catalog
from logger.Messages.MedallionMess import (
    BRONZE_START,
    BRONZE_SUCCESS,
    GOLD_SEQUENTIAL_MODE,
    GOLD_START,
    GOLD_SUCCESS,
    SILVER_START,
    SILVER_SUCCESS,
)

from .bronze import BronzeLayer
from .gold.GoldLayer import GoldLayer
from .silver.silver import SilverLayer


class MedallionPipeline:
    """
    Orchestrates the entire Medallion Architecture with parallel execution.
    """

    def __init__(self, config: ProjectConfig, factory: DataFactory):
        self.config = config
        self.factory = factory
        self.project_root = Path(__file__).parents[1]
        user_id = getattr(self.config, "data_user_id", "default")
        safe_user = (
            "".join(
                ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(user_id)
            )
            or "default"
        )
        self.data_path = self.project_root / ".." / "data" / "users" / safe_user
        self.raw_path = self.data_path / "raw"
        self.processed_path = self.data_path / "processed"
        self.gold_path = self.data_path / "gold"
        self.checkpoint_path = self.data_path / "pipeline_checkpoint.json"

        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.gold_path.mkdir(parents=True, exist_ok=True)

        self.bronze = BronzeLayer(config, factory, base_path=str(self.raw_path))
        self.silver = SilverLayer(config)
        self.gold: Optional[GoldLayer] = None
        self._stage_durations: Dict[str, float] = {}
        self._stage_success: Dict[str, bool] = {}

        self.bronze.base_path = str(self.raw_path)
        self.silver.raw_path = self.raw_path
        self.silver.processed_path = self.processed_path
        # dead_letter_path is computed in SilverLayer.__init__ before the above
        # override runs, so it would point to the non-user-scoped path. Fix it here.
        self.silver.dead_letter_path = (
            self.processed_path / "quality" / "dead_letter.jsonl"
        )

    def _get_gold_layer(self) -> GoldLayer:
        if self.gold is None:
            self.gold = GoldLayer(self.config)
            self.gold.processed_path = self.processed_path
            self.gold.gold_path = self.gold_path
            self.gold.governance_path = self.gold_path / "governance"
            self.gold.governance_path.mkdir(parents=True, exist_ok=True)
            self.gold.initialize_data()  # load data only after paths are finalised
        return self.gold

    def _reset_stage_tracking(self) -> None:
        self._stage_durations = {}
        self._stage_success = {}

    def _load_checkpoint(self) -> Dict[str, Any]:
        if not self.checkpoint_path.exists():
            return {}
        try:
            return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_checkpoint(self, stage: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        payload = self._load_checkpoint()
        payload[stage] = {
            "status": status,
            "updated_at": time.time(),
            "details": details or {},
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _is_stage_done(self, stage: str) -> bool:
        payload = self._load_checkpoint()
        if payload.get(stage, {}).get("status") != "success":
            return False
        if stage == "bronze":
            return self.raw_path.exists() and any(self.raw_path.glob("**/*.parquet"))
        if stage == "silver":
            return self.processed_path.exists() and any(self.processed_path.glob("**/*.parquet"))
        if stage == "gold":
            return (self.gold_path / "master_table.parquet").exists()
        return False

    def _run_stage_with_retry(self, stage_name: str, stage_fn: Any) -> None:
        attempts = max(1, int(getattr(self.config, "pipeline_stage_retry_attempts", 1)))
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                stage_fn()
                self._write_checkpoint(stage_name, "success", {"attempt": attempt})
                return
            except Exception as exc:
                last_error = exc
                self._write_checkpoint(
                    stage_name,
                    "failed",
                    {"attempt": attempt, "error": str(exc)},
                )
                if attempt >= attempts:
                    raise
        if last_error is not None:
            raise last_error

    def run_bronze(self) -> None:
        print(BRONZE_START)
        stage_start = time.time()
        success = False
        error_message = ""
        try:
            with tqdm(total=1, desc="Bronze Layer") as pbar:
                summary = self.bronze.run()
                pbar.update(1)
            failed = (
                int(summary.get("fail_count", 0)) if isinstance(summary, dict) else 0
            )
            missing_sources = (
                summary.get("missing_sources", []) if isinstance(summary, dict) else []
            )
            # Only hard-fail when an entire source category produced zero data.
            # Individual task failures (one bad ticker, one stale series) are
            # logged as warnings so the pipeline continues with available data.
            if missing_sources:
                error_message = (
                    f"bronze_integrity_violation: entire source(s) failed to "
                    f"produce any data: {missing_sources}"
                )
                raise ComplianceViolationError(error_message)
            if failed > 0:
                print(
                    f"Bronze partial failures: {failed} task(s) failed but pipeline "
                    "continues with available data."
                )
            success = True
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            self._stage_durations["bronze"] = time.time() - stage_start
            self._stage_success["bronze"] = success
            catalog.log_operation(
                "pipeline_stage",
                "medallion",
                {
                    "stage": "bronze",
                    "duration_seconds": self._stage_durations["bronze"],
                    "success": success,
                },
                {"error": error_message or None},
                "Bronze stage completed" if success else "Bronze stage failed",
            )
        print(BRONZE_SUCCESS)

    def run_silver(self) -> None:
        print(SILVER_START)
        stage_start = time.time()
        success = False
        error_message = ""
        try:
            with tqdm(total=1, desc="Silver Layer") as pbar:
                summary = self.silver.run()
                pbar.update(1)
            failed = summary.get("failed_count", 0) if isinstance(summary, dict) else 0
            missing_sources = (
                summary.get("missing_sources", []) if isinstance(summary, dict) else []
            )
            success = failed == 0
            if not success or missing_sources:
                error_message = (
                    f"{failed} entit{'y' if failed == 1 else 'ies'} failed: "
                    f"{summary.get('failed_entities', [])}, "
                    f"missing_sources={missing_sources}"
                )
                success = False
                if bool(getattr(self.config, "silver_hard_fail", True)):
                    raise ParallelExecutionError(
                        "Silver hard-fail enabled and data quality guardrails were "
                        "violated: "
                        f"{error_message}"
                    )
                print(f"Silver soft-fail mode active: {error_message}")
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            self._stage_durations["silver"] = time.time() - stage_start
            self._stage_success["silver"] = success
            catalog.log_operation(
                "pipeline_stage",
                "medallion",
                {
                    "stage": "silver",
                    "duration_seconds": self._stage_durations["silver"],
                    "success": success,
                },
                {"error": error_message or None},
                "Silver stage completed" if success else "Silver stage failed",
            )
        if success:
            print(SILVER_SUCCESS)

    def run_gold(self) -> None:
        print(GOLD_START)
        stage_start = time.time()
        success = False
        error_message = ""
        try:
            with tqdm(total=1, desc="Gold Layer") as pbar:
                self._get_gold_layer().create_master_table()
                pbar.update(1)
            success = True
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            self._stage_durations["gold"] = time.time() - stage_start
            self._stage_success["gold"] = success
            catalog.log_operation(
                "pipeline_stage",
                "medallion",
                {
                    "stage": "gold",
                    "duration_seconds": self._stage_durations["gold"],
                    "success": success,
                },
                {"error": error_message or None},
                "Gold stage completed" if success else "Gold stage failed",
            )
        print(GOLD_SUCCESS)

    def _emit_sla_snapshot(self) -> None:
        durations = [d for d in self._stage_durations.values() if d > 0]
        if not durations:
            return
        p95_index = max(0, min(len(durations) - 1, int(0.95 * (len(durations) - 1))))
        sorted_durations = sorted(durations)
        p95_latency = sorted_durations[p95_index]
        total_duration = sum(durations)
        total_stages = max(len(self._stage_success), 1)
        success_count = sum(1 for v in self._stage_success.values() if v)
        failure_count = total_stages - success_count
        success_rate = success_count / total_stages
        error_rate = failure_count / total_stages
        throughput = len(durations) / total_duration if total_duration > 0 else 0.0
        catalog.log_sla_snapshot(
            component="medallion",
            p95_latency_seconds=float(p95_latency),
            error_rate=float(error_rate),
            success_rate=float(success_rate),
            throughput_ops_per_sec=float(throughput),
        )
        catalog.log_slo_window("medallion", 300, "pipeline_stage")
        catalog.log_slo_window("medallion", 3600, "pipeline_stage")

    def run_full_pipeline_parallel(self) -> Dict[str, Any]:
        """
        Run pipeline stages with dependency-safe ordering.
        Only analytics fan-out remains parallel inside GoldLayer.
        """
        try:
            self._reset_stage_tracking()
            resume = bool(getattr(self.config, "pipeline_resume_from_checkpoint", False))
            with tqdm(total=4, desc="Full Pipeline") as pbar:
                if resume and self._is_stage_done("bronze"):
                    print("[resume] Skipping Bronze stage (checkpoint valid).")
                else:
                    self._run_stage_with_retry("bronze", self.run_bronze)
                pbar.update(1)
                if resume and self._is_stage_done("silver"):
                    print("[resume] Skipping Silver stage (checkpoint valid).")
                else:
                    self._run_stage_with_retry("silver", self.run_silver)
                pbar.update(1)

                if not self._stage_success.get("silver"):
                    print("Silver stage failed. Halting pipeline.")
                    return {}

                if resume and self._is_stage_done("gold"):
                    print("[resume] Skipping Gold stage (checkpoint valid).")
                else:
                    self._run_stage_with_retry("gold", self.run_gold)
                pbar.update(1)

                # Gold internal methods handle analysis parallelism.
                results = self._get_gold_layer().run_all_analyses_parallel()
                pbar.update(1)
            self._emit_sla_snapshot()
            return results
        except ParallelExecutionError as e:
            print(f"Parallel execution error: {e}. Falling back to sequential.")
            return self.run_full_pipeline_sequential()
        except Exception:
            # Surface architectural failures instead of masking them by fallback.
            raise

    def health_check(self) -> Dict[str, bool]:
        """Perform health checks on data and system."""
        checks = {}
        checks["raw_data_exists"] = self.raw_path.exists() and any(
            self.raw_path.iterdir()
        )
        checks["processed_data_exists"] = self.processed_path.exists() and any(
            self.processed_path.iterdir()
        )
        checks["gold_data_exists"] = (self.gold_path / "master_table.parquet").exists()
        checks["config_valid"] = bool(
            hasattr(self.config, "macro_series_map") and hasattr(self.config, "worldbank_indicator_map")
        )
        return checks

    def run_full_pipeline_sequential(self) -> Dict[str, Any]:
        print(GOLD_SEQUENTIAL_MODE)
        self._reset_stage_tracking()
        resume = bool(getattr(self.config, "pipeline_resume_from_checkpoint", False))
        with tqdm(total=4, desc="Full Pipeline") as pbar:
            if resume and self._is_stage_done("bronze"):
                print("[resume] Skipping Bronze stage (checkpoint valid).")
            else:
                self._run_stage_with_retry("bronze", self.run_bronze)
            pbar.update(1)
            if resume and self._is_stage_done("silver"):
                print("[resume] Skipping Silver stage (checkpoint valid).")
            else:
                self._run_stage_with_retry("silver", self.run_silver)
            pbar.update(1)

            if not self._stage_success.get("silver"):
                print("Silver stage failed. Halting pipeline.")
                return {}

            if resume and self._is_stage_done("gold"):
                print("[resume] Skipping Gold stage (checkpoint valid).")
            else:
                self._run_stage_with_retry("gold", self.run_gold)
            pbar.update(1)
            results = self._get_gold_layer().run_all_analyses()
            pbar.update(1)
        self._emit_sla_snapshot()
        return results


__all__ = ["MedallionPipeline"]
