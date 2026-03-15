import concurrent.futures
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
from Fetchers.ProjectConfig import ProjectConfig
from Fetchers.Factory import DataFactory
from .bronze import BronzeLayer
from .silver.silver import SilverLayer
from .gold.GoldLayer import GoldLayer
from exceptions.MedallionExceptions import ParallelExecutionError

class MedallionPipeline:
    """
    Orchestrates the entire Medallion Architecture with parallel execution and progress tracking.
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
        self.gold = GoldLayer(config)

        self.bronze.base_path = str(self.raw_path)
        self.silver.raw_path = self.raw_path
        self.silver.processed_path = self.processed_path
        self.gold.processed_path = self.processed_path
        self.gold.gold_path = self.gold_path

    def run_bronze(self) -> None:
        with tqdm(total=1, desc="Bronze Layer") as pbar:
            self.bronze.run()
            pbar.update(1)

    def run_silver(self) -> None:
        with tqdm(total=1, desc="Silver Layer") as pbar:
            self.silver.run()
            pbar.update(1)

    def run_gold(self) -> None:
        with tqdm(total=1, desc="Gold Layer") as pbar:
            self.gold.create_master_table()
            pbar.update(1)

    def run_full_pipeline_parallel(self) -> Dict[str, Any]:
        """
        Run the entire pipeline in parallel where possible.
        Falls back to sequential on failure.
        """
        try:
            with tqdm(total=4, desc="Full Pipeline") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    bronze_future = executor.submit(self.run_bronze)
                    silver_future = executor.submit(self.run_silver)
                    concurrent.futures.wait([bronze_future, silver_future])
                    pbar.update(2)
                    gold_future = executor.submit(self.run_gold)
                    gold_future.result()
                    pbar.update(1)
                    analyses_future = executor.submit(self.gold.run_all_analyses_parallel)
                    results = analyses_future.result()
                    pbar.update(1)
            return results
        except ParallelExecutionError as e:
            print(f"Parallel execution error: {e}. Falling back to sequential.")
            return self.run_full_pipeline_sequential()
        except Exception as e:
            print(f"Unexpected error in parallel execution: {e}. Falling back to sequential.")
            return self.run_full_pipeline_sequential()

    def health_check(self) -> Dict[str, bool]:
        """Perform health checks on data and system."""
        checks = {}
        checks['raw_data_exists'] = self.raw_path.exists() and any(self.raw_path.iterdir())
        checks['processed_data_exists'] = self.processed_path.exists() and any(self.processed_path.iterdir())
        checks['gold_data_exists'] = (self.gold_path / "master_table.parquet").exists()
        checks['config_valid'] = hasattr(self.config, 'fred_api_key') and self.config.fred_api_key
        return checks

    def run_full_pipeline_sequential(self) -> Dict[str, Any]:
        with tqdm(total=4, desc="Full Pipeline") as pbar:
            self.run_bronze()
            pbar.update(1)
            self.run_silver()
            pbar.update(1)
            self.run_gold()
            pbar.update(1)
            results = self.gold.run_all_analyses_parallel()
            pbar.update(1)
        return results