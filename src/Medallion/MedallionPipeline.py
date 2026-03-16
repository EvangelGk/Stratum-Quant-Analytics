import time
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from exceptions.MedallionExceptions import ParallelExecutionError
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
		self.data_path = self.project_root / ".." / "data"
		self.raw_path = self.data_path / "raw"
		self.processed_path = self.data_path / "processed"
		self.gold_path = self.data_path / "gold"

		self.raw_path.mkdir(parents=True, exist_ok=True)
		self.processed_path.mkdir(parents=True, exist_ok=True)
		self.gold_path.mkdir(parents=True, exist_ok=True)

		self.bronze = BronzeLayer(config, factory)
		self.silver = SilverLayer(config)
		self.gold: Optional[GoldLayer] = None
		self._stage_durations: Dict[str, float] = {}
		self._stage_success: Dict[str, bool] = {}

		self.bronze.base_path = str(self.raw_path)
		self.silver.raw_path = self.raw_path
		self.silver.processed_path = self.processed_path

	def _get_gold_layer(self) -> GoldLayer:
		if self.gold is None:
			self.gold = GoldLayer(self.config)
			self.gold.processed_path = self.processed_path
			self.gold.gold_path = self.gold_path
		return self.gold

	def _reset_stage_tracking(self) -> None:
		self._stage_durations = {}
		self._stage_success = {}

	def run_bronze(self) -> None:
		print(BRONZE_START)
		stage_start = time.time()
		success = False
		error_message = ""
		try:
			with tqdm(total=1, desc="Bronze Layer") as pbar:
				self.bronze.run()
				pbar.update(1)
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
				self.silver.run()
				pbar.update(1)
			success = True
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
			with tqdm(total=4, desc="Full Pipeline") as pbar:
				self.run_bronze()
				pbar.update(1)
				self.run_silver()
				pbar.update(1)
				self.run_gold()
				pbar.update(1)

				# Gold internal methods handle analysis parallelism.
				results = self._get_gold_layer().run_all_analyses_parallel()
				pbar.update(1)
			self._emit_sla_snapshot()
			return results
		except ParallelExecutionError as e:
			print(f"Parallel execution error: {e}. Falling back to sequential.")
			return self.run_full_pipeline_sequential()
		except Exception as e:
			print(
				f"Unexpected error in parallel execution: {e}. "
				"Falling back to sequential."
			)
			return self.run_full_pipeline_sequential()

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
			hasattr(self.config, "fred_api_key") and self.config.fred_api_key
		)
		return checks

	def run_full_pipeline_sequential(self) -> Dict[str, Any]:
		print(GOLD_SEQUENTIAL_MODE)
		self._reset_stage_tracking()
		with tqdm(total=4, desc="Full Pipeline") as pbar:
			self.run_bronze()
			pbar.update(1)
			self.run_silver()
			pbar.update(1)
			self.run_gold()
			pbar.update(1)
			results = self._get_gold_layer().run_all_analyses()
			pbar.update(1)
		self._emit_sla_snapshot()
		return results


__all__ = ["MedallionPipeline"]
