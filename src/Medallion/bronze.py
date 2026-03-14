import pandas as pd
import os
from Fetchers.Factory import DataFactory
from Fetchers.ProjectConfig import ProjectConfig
import json
from datetime import datetime
import concurrent.futures
import threading
import logging
import time
import random

class BronzeLayer:
    def __init__(self, config: ProjectConfig, factory: DataFactory):
        self.config = config
        self.factory = factory
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
        os.makedirs(self.base_path, exist_ok=True)
        self.catalog = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info(f"Starting ingestion in {self.config.mode.value} mode...")
        targets = self.config.get_targets()
        
        # Prepare tasks for parallel execution
        tasks = []
        
        # 1. Financials (yFinance)
        y_fetcher = self.factory.get_fetcher("yfinance")
        for ticker in targets:
            tasks.append((y_fetcher, (ticker, self.config.start_date, self.config.end_date), f"{ticker.lower()}_financials", "yfinance"))
        
        # 2. Macro Data (FRED)
        fred_fetcher = self.factory.get_fetcher("fred")
        macro_map = {"CPIAUCSL": "inflation", "PNRGINDEXM": "energy_index"}
        for series_id, filename in macro_map.items():
            tasks.append((fred_fetcher, (series_id, self.config.start_date, self.config.end_date), filename, "fred"))
        
        # 3. World Bank
        wb_fetcher = self.factory.get_fetcher("worldbank")
        wb_map = {"NY.GDP.MKTP.KD.ZG": "gdp_growth", "EG.USE.PCAP.KG.OE": "energy_usage"}
        for indicator, filename in wb_map.items():
            tasks.append((wb_fetcher, (indicator, "WLD", self.config.start_date, self.config.end_date), filename, "worldbank"))
        
        # Execute tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self._fetch_and_save, fetcher, params, filename, source) for fetcher, params, filename, source in tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Raise any exceptions
                except Exception as e:
                    self.logger.error(f"Task failed: {e}")
        
        # Catalog is now written after each successful save, no need to write at the end

    def _fetch_and_save(self, fetcher, params, filename, source):
        for attempt in range(4):  # Retry up to 4 times (initial + 3 retries)
            try:
                time.sleep(random.uniform(1, 3))  # Random delay to avoid rate limits
                df = fetcher.fetch(*params)
                self._process_and_save(df, filename, source)
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < 3:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {filename}: {e}. Retrying...")
                    time.sleep(random.uniform(1, 3))  # Additional delay before retry
                else:
                    self.logger.error(f"Failed to fetch {filename} after 4 attempts: {e}")
                    raise  # Re-raise after max retries

    def _process_and_save(self, df: pd.DataFrame, filename: str, source: str):
        """Saves the dataframe in parquet format
        to the specified directory."""
        if df is None or df.empty:
            self.logger.warning(f"No data for {filename}")
            return

        df['ingested_at'] = datetime.now().isoformat()
        df['source_system'] = source
        
        # Create source-specific directory
        source_path = os.path.join(self.base_path, source)
        os.makedirs(source_path, exist_ok=True)
        
        full_path = os.path.join(source_path, f"{filename}.parquet")
            
        # Index=False to avoid saving unnecessary columns
        df.to_parquet(full_path, index=False, engine='pyarrow', compression='snappy')
        self.logger.info(f"Successfully saved: {full_path}")
        
        # Update the catalog with metadata
        with self.lock:
            self.catalog[filename] = {
                "path": full_path,
                "rows": len(df),
                "source": source,
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        self.logger.info(f"Saved: {filename}")
        self._write_catalog()  # Write catalog after each successful save

    def _write_catalog(self):
        """Creates the manifest file in the data/raw directory."""
        catalog_path = os.path.join(self.base_path, "catalog.json")
        with open(catalog_path, 'w', encoding='utf-8') as f:
            json.dump(self.catalog, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Data Catalog updated at {catalog_path}")