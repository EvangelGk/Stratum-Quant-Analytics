import concurrent.futures
import json
import logging
import os
import random
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
        self.source_success_count: Dict[str, int] = {}
        self.source_fail_count: Dict[str, int] = {}
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

    def run(self) -> Dict[str, Any]:
        """
        Runs the data ingestion process in parallel.
        Prepares tasks, executes them with retries, and logs summary.
        """
        print(FETCHER_START.format(source="multiple sources"))
        self.logger.info(f"Starting ingestion in {self.config.mode.value} mode...")
        targets = self.config.get_targets()
        self.success_count = 0
        self.fail_count = 0
        self.source_success_count = {k: 0 for k in self.provider_limits.keys()}
        self.source_fail_count = {k: 0 for k in self.provider_limits.keys()}

        # Prepare tasks for parallel execution
        tasks: List[Tuple[Any, Tuple[str, ...], str, str]] = []
        source_expected_counts: Dict[str, int] = {k: 0 for k in self.provider_limits}

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
            source_expected_counts["yfinance"] += 1

        # 2. Macro Data (FRED)
        fred_fetcher = self.factory.get_fetcher("fred")
        macro_map = dict(getattr(self.config, "macro_series_map", {}))
        for series_id, filename in macro_map.items():
            tasks.append(
                (
                    fred_fetcher,
                    (series_id, self.config.start_date, self.config.end_date),
                    filename,
                    "fred",
                )
            )
            source_expected_counts["fred"] += 1

        # 3. World Bank
        wb_fetcher = self.factory.get_fetcher("worldbank")
        wb_map = dict(getattr(self.config, "worldbank_indicator_map", {}))
        for indicator, filename in wb_map.items():
            tasks.append(
                (
                    wb_fetcher,
                    (indicator, "WLD", self.config.start_date, self.config.end_date),
                    filename,
                    "worldbank",
                )
            )
            source_expected_counts["worldbank"] += 1

        # Execute tasks in parallel
        max_workers = min(self.config.max_workers, len(tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_source = {
                executor.submit(
                    self._fetch_and_save, fetcher, params, filename, source
                ): source
                for fetcher, params, filename, source in tasks
            }
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    future.result()  # Raise any exceptions
                except (FetcherError, TimeoutError, RateLimitError) as e:
                    self.logger.error(f"Fetcher error in task: {e}")
                    with self.lock:
                        self.fail_count += 1
                        self.source_fail_count[source] = (
                            self.source_fail_count.get(source, 0) + 1
                        )
                except Exception as e:
                    self.logger.error(f"Unexpected error in task: {e}")
                    with self.lock:
                        self.fail_count += 1
                        self.source_fail_count[source] = (
                            self.source_fail_count.get(source, 0) + 1
                        )

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
        missing_sources = [
            source
            for source, expected in source_expected_counts.items()
            if expected > 0 and self.source_success_count.get(source, 0) == 0
        ]
        return {
            "total_files": total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "source_expected": source_expected_counts,
            "source_success": self.source_success_count,
            "source_fail": self.source_fail_count,
            "missing_sources": missing_sources,
        }

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
                with self.lock:
                    self.success_count += 1
                    self.source_success_count[source] = (
                        self.source_success_count.get(source, 0) + 1
                    )
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

        df = self._normalize_source_columns(df, source)

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

            # Update the catalog with metadata and persist atomically under lock
            with self.lock:
                self.catalog[filename] = {
                    "path": full_path,
                    "rows": len(df),
                    "source": source,
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                self._write_catalog()  # Serialized: called inside lock
            self.logger.info(f"Saved: {filename}")
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

    def _normalize_source_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Normalize common provider column aliases to Bronze canonical schema."""
        if df.empty:
            return df

        work_df = df.copy()

        if source == "yfinance" and isinstance(work_df.columns, pd.MultiIndex):
            level0 = [str(col[0]) for col in work_df.columns]
            level1 = [str(col[1]) for col in work_df.columns]
            known = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            work_df.columns = (
                level0
                if len(known.intersection(level0)) >= len(known.intersection(level1))
                else level1
            )

        alias_map: Dict[str, str] = {}
        if source == "yfinance":
            alias_map = {
                "date": "Date",
                "datetime": "Date",
                "index": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adj close": "Adj Close",
                "adjclose": "Adj Close",
                "adjusted close": "Adj Close",
                "volume": "Volume",
            }
        elif source == "fred":
            alias_map = {
                "date": "Date",
                "index": "Date",
                "value": "Value",
            }
        elif source == "worldbank":
            alias_map = {
                "economy": "economy",
                "country": "economy",
                "date": "Date",
                "time": "Date",
                "value": "Value",
            }

        renamed = {}
        for col in work_df.columns:
            key = str(col).strip().lower().replace("_", " ")
            key = " ".join(key.split())
            renamed[col] = alias_map.get(key, str(col))
        work_df = work_df.rename(columns=renamed)

        # Provider regressions may create duplicate canonical columns.
        # Collapse by keeping the first non-null value row-wise.
        if work_df.columns.duplicated().any():
            dedup = pd.DataFrame(index=work_df.index)
            for name in list(dict.fromkeys(work_df.columns)):
                subset = work_df.loc[:, work_df.columns == name]
                dedup[name] = subset.bfill(axis=1).iloc[:, 0]
            work_df = dedup

        if source == "yfinance" and "Adj Close" not in work_df.columns:
            if "Close" in work_df.columns:
                work_df["Adj Close"] = work_df["Close"]

        return work_df

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
        Atomically persist the in-memory catalog to disk.
        Writes to a tmp file first, then os.replace() so readers never see
        a partially-written file.  Must be called with self.lock held.
        """
        catalog_path = os.path.join(self.base_path, "catalog.json")
        tmp_path = catalog_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.catalog, f, indent=4, ensure_ascii=False, sort_keys=True)
        os.replace(tmp_path, catalog_path)  # atomic on POSIX; best-effort on Windows
        self.logger.info(f"Data Catalog updated at {catalog_path}")
