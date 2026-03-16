import concurrent.futures
import json
import logging
import os
import random
import shutil
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

from exceptions.FetchersExceptions import FetcherError, RateLimitError, TimeoutError
from Fetchers.Factory import DataFactory
from Fetchers.ProjectConfig import ProjectConfig
from logger.Messages.FetchersMess import (
    FETCHER_COMPLETION,
    FETCHER_START,
)


class BronzeLayer:
    """
    Bronze Layer for data ingestion from multiple sources.
    Handles parallel fetching, retry logic, validation, and catalog management.
    """

    def __init__(self, config: ProjectConfig, factory: DataFactory):
        self.config = config
        self.factory = factory
        self.base_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/raw")
        )
        os.makedirs(self.base_path, exist_ok=True)
        self.catalog: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.success_count = 0
        self.fail_count = 0
        self.provider_limits = {
            "yfinance": 60,
            "fred": 30,
            "worldbank": 20,
        }
        self.provider_state: Dict[str, Dict[str, float]] = {
            source: {
                "last_request_ts": 0.0,
                "consecutive_failures": 0.0,
                "circuit_open_until": 0.0,
            }
            for source in self.provider_limits.keys()
        }
        self.circuit_breaker_threshold = 3

    def run(self) -> None:
        """
        Runs the data ingestion process in parallel.
        Prepares tasks, executes them with retries, and logs summary.
        """
        print(FETCHER_START.format(source="multiple sources"))
        self.logger.info(f"Starting ingestion in {self.config.mode.value} mode...")
        targets = self.config.get_targets()

        # Prepare tasks for parallel execution
        tasks: List[Tuple[Any, Tuple[str, ...], str, str]] = []

        # 1. Financials (yFinance)
        y_fetcher = self.factory.get_fetcher("yfinance")
        for ticker in targets:
            tasks.append(
                (
                    y_fetcher,
                    (ticker, self.config.start_date, self.config.end_date),
                    f"{ticker.lower()}_financials",
                    "yfinance",
                )
            )

        # 2. Macro Data (FRED)
        fred_fetcher = self.factory.get_fetcher("fred")
        macro_map = {"CPIAUCSL": "inflation", "PNRGINDEXM": "energy_index"}
        for series_id, filename in macro_map.items():
            tasks.append(
                (
                    fred_fetcher,
                    (series_id, self.config.start_date, self.config.end_date),
                    filename,
                    "fred",
                )
            )

        # 3. World Bank
        wb_fetcher = self.factory.get_fetcher("worldbank")
        wb_map = {
            "NY.GDP.MKTP.KD.ZG": "gdp_growth",
            "EG.USE.PCAP.KG.OE": "energy_usage",
        }
        for indicator, filename in wb_map.items():
            tasks.append(
                (
                    wb_fetcher,
                    (indicator, "WLD", self.config.start_date, self.config.end_date),
                    filename,
                    "worldbank",
                )
            )

        # Execute tasks in parallel
        max_workers = min(self.config.max_workers, len(tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._fetch_and_save, fetcher, params, filename, source)
                for fetcher, params, filename, source in tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Raise any exceptions
                except (FetcherError, TimeoutError, RateLimitError) as e:
                    self.logger.error(f"Fetcher error in task: {e}")
                    self.fail_count += 1
                except Exception as e:
                    self.logger.error(f"Unexpected error in task: {e}")
                    self.fail_count += 1

        # Log summary
        total_files = self.success_count + self.fail_count
        success_rate = (
            (self.success_count / total_files * 100) if total_files > 0 else 0
        )
        print(
            FETCHER_COMPLETION.format(
                total_files=total_files, success_rate=f"{success_rate:.1f}"
            )
        )
        self.logger.info(
            "Ingestion completed. Success: %s, Failures: %s",
            self.success_count,
            self.fail_count,
        )

    def _fetch_and_save(
        self, fetcher: Any, params: Tuple[str, ...], filename: str, source: str
    ) -> None:
        """
        Fetches data with retry logic and saves it.
        Includes random delays to avoid rate limits.
        """
        for attempt in range(self.config.max_retries):
            try:
                self._wait_for_circuit(source)
                self._acquire_rate_slot(source)
                delay = random.uniform(
                    self.config.retry_delay_min, self.config.retry_delay_max
                )
                time.sleep(delay)  # Random delay to avoid rate limits
                df = fetcher.fetch(*params)
                self._process_and_save(df, filename, source)
                self._record_provider_success(source)
                self.success_count += 1
                break  # Success, exit retry loop
            except FetcherError as e:
                self._record_provider_failure(source)
                if attempt < self.config.max_retries - 1:
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {filename}: {e}. Retrying..."
                    )
                else:
                    self.logger.error(
                        (
                            f"Failed to fetch {filename} after "
                            f"{self.config.max_retries} attempts: {e}"
                        )
                    )
                    raise  # Re-raise after max retries
            except TimeoutError as e:
                self._record_provider_failure(source)
                if attempt < self.config.max_retries - 1:
                    self.logger.warning(
                        f"Timeout on attempt {attempt + 1} for {filename}: {e}."
                        " Retrying..."
                    )
                else:
                    raise
            except RateLimitError as e:
                self._record_provider_failure(source)
                self.logger.warning(
                    f"Rate limit hit for {filename}: {e}. Waiting longer..."
                )
                time.sleep(60)  # Extra wait for rate limit
            except Exception as e:
                self._record_provider_failure(source)
                self.logger.error(f"Unexpected error for {filename}: {e}")
                raise

    def _acquire_rate_slot(self, source: str) -> None:
        requests_per_minute = self.provider_limits.get(source, 20)
        min_interval = 60.0 / max(requests_per_minute, 1)
        with self.lock:
            state = self.provider_state.setdefault(
                source,
                {
                    "last_request_ts": 0.0,
                    "consecutive_failures": 0.0,
                    "circuit_open_until": 0.0,
                },
            )
            now = time.time()
            elapsed = now - state["last_request_ts"]
            wait_for = max(0.0, min_interval - elapsed)

        if wait_for > 0:
            time.sleep(wait_for)

        with self.lock:
            self.provider_state[source]["last_request_ts"] = time.time()

    def _wait_for_circuit(self, source: str) -> None:
        with self.lock:
            state = self.provider_state.setdefault(
                source,
                {
                    "last_request_ts": 0.0,
                    "consecutive_failures": 0.0,
                    "circuit_open_until": 0.0,
                },
            )
            open_until = state["circuit_open_until"]
        now = time.time()
        if open_until > now:
            sleep_for = open_until - now
            self.logger.warning(
                "Circuit breaker active for %s. Cooling down for %.2fs",
                source,
                sleep_for,
            )
            time.sleep(sleep_for)

    def _record_provider_success(self, source: str) -> None:
        with self.lock:
            state = self.provider_state[source]
            state["consecutive_failures"] = 0.0
            state["circuit_open_until"] = 0.0

    def _record_provider_failure(self, source: str) -> None:
        with self.lock:
            state = self.provider_state[source]
            state["consecutive_failures"] += 1.0
            failures = int(state["consecutive_failures"])
            if failures >= self.circuit_breaker_threshold:
                cooldown = min(15.0 * failures, 300.0)
                state["circuit_open_until"] = time.time() + cooldown

    def _process_and_save(self, df: pd.DataFrame, filename: str, source: str) -> None:
        """
        Processes and saves the dataframe to Parquet.
        Includes validation, cleanup on failure, and catalog update.
        """
        if df is None or df.empty:
            self.logger.warning(f"No data for {filename}")
            return

        # Basic validation: check for required columns based on source
        expected_columns = self._get_expected_columns(source)
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(
                f"Data for {filename} missing required columns: {expected_columns}"
            )

        df["ingested_at"] = datetime.now().isoformat()
        df["source_system"] = source

        # Create source-specific directory
        source_path = os.path.join(self.base_path, source)
        os.makedirs(source_path, exist_ok=True)

        full_path = os.path.join(source_path, f"{filename}.parquet")

        try:
            # Index=False to avoid saving unnecessary columns
            df.to_parquet(
                full_path, index=False, engine="pyarrow", compression="snappy"
            )
            self.logger.info(f"Successfully saved: {full_path}")

            # Update the catalog with metadata
            with self.lock:
                self.catalog[filename] = {
                    "path": full_path,
                    "rows": len(df),
                    "source": source,
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            self.logger.info(f"Saved: {filename}")
            self._write_catalog()  # Write catalog after each successful save
        except FetcherError as e:
            # Cleanup on failure
            if os.path.exists(full_path):
                os.remove(full_path)
            self.logger.error(f"Fetcher error for {filename}: {e}")
            raise
        except Exception as e:
            # Cleanup on failure
            if os.path.exists(full_path):
                os.remove(full_path)
            self.logger.error(f"Unexpected error for {filename}: {e}")
            raise

    def _get_expected_columns(self, source: str) -> List[str]:
        """
        Returns the expected columns for each source.
        """
        if source == "yfinance":
            return ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        elif source == "fred":
            return ["Date", "Value"]
        elif source == "worldbank":
            return ["economy", "Date", "Value"]
        else:
            return []

    def _write_catalog(self) -> None:
        """
        Creates the manifest file in the data/raw directory with backup.
        """
        catalog_path = os.path.join(self.base_path, "catalog.json")
        backup_path = catalog_path + ".bak"
        if os.path.exists(catalog_path):
            shutil.copy(catalog_path, backup_path)
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(self.catalog, f, indent=4, ensure_ascii=False, sort_keys=True)
        self.logger.info(f"Data Catalog updated at {catalog_path}")
