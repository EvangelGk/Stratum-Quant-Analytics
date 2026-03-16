import concurrent.futures
from pathlib import Path
from typing import Any, Dict

from Fetchers.Factory import DataFactory
from Fetchers.ProjectConfig import ProjectConfig

from .bronze import BronzeLayer
from .gold.GoldLayer import GoldLayer
from .silver.silver import SilverLayer


class MedallionPipeline:
    """
    Orchestrates the entire Medallion Architecture with parallel execution.
    Handles Bronze (Ingestion), Silver (Transformation), Gold (Analytics).
    """

    def __init__(self, config: ProjectConfig, factory: DataFactory):
        self.config = config
        self.factory = factory
        # Centralized paths using Pathlib for cross-platform compatibility
        self.project_root = Path(__file__).parents[1]  # src/
        self.data_path = self.project_root / ".." / "data"  # Relative to project root
        self.raw_path = self.data_path / "raw"
        self.processed_path = self.data_path / "processed"
        self.gold_path = self.data_path / "gold"

        # Ensure directories exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.gold_path.mkdir(parents=True, exist_ok=True)

        # Initialize layers with shared paths
        self.bronze = BronzeLayer(config, factory)
        self.silver = SilverLayer(config)
        self.gold = GoldLayer(config)

        # Override paths in layers for consistency
        self.bronze.base_path = str(self.raw_path)
        self.silver.raw_path = self.raw_path
        self.silver.processed_path = self.processed_path
        self.gold.processed_path = self.processed_path
        self.gold.gold_path = self.gold_path

    def run_bronze(self) -> None:
        """Run Bronze layer (parallel fetching)."""
        self.bronze.run()

    def run_silver(self) -> None:
        """Run Silver layer (parallel transformation)."""
        self.silver.run()

    def run_gold(self) -> None:
        """Run Gold layer (create master table)."""
        self.gold.create_master_table()

    def run_full_pipeline_parallel(self) -> Dict[str, Any]:
        """
        Run the entire pipeline in parallel where possible.
        Bronze and Silver can run partially parallel, Gold depends on them.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit Bronze and Silver in parallel (if no dependencies)
            bronze_future = executor.submit(self.run_bronze)
            silver_future = executor.submit(self.run_silver)

            # Wait for Bronze and Silver to complete
            concurrent.futures.wait([bronze_future, silver_future])

            # Then run Gold (depends on processed data)
            gold_future = executor.submit(self.run_gold)
            gold_future.result()  # Wait for completion

            # Finally, run analyses in parallel
            analyses_future = executor.submit(self.gold.run_all_analyses_parallel)
            results = analyses_future.result()

        return results

    def run_full_pipeline_sequential(self) -> Dict[str, Any]:
        """Run the pipeline sequentially for debugging."""
        self.run_bronze()
        self.run_silver()
        self.run_gold()
        return self.gold.run_all_analyses_parallel()
