import concurrent.futures
import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import pandera.errors

from Fetchers.ProjectConfig import ProjectConfig
from src.exceptions.MedallionExceptions import (
    CatalogNotFoundError,
    ComplianceViolationError,
    DataValidationError,
    FileSaveError,
    ImputationError,
    OutlierDetectionError,
    StandardizationError,
)

# Import Schemas
from .schema import financials_schema, macro_schema, worldbank_schema


class SilverLayer:
    """
    Corporate-Grade Silver Layer.
    Responsibility: Transform Bronze (Raw) to Silver (Trusted) data via
    Standardization, Imputation, and Schema Validation.
    """

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.silver_run_id = str(uuid.uuid4())  # Unique ID for this run

        # Use Pathlib for absolute safety on Windows/Linux
        self.root_path = Path(__file__).parents[2]
        self.raw_path = self.root_path / "data" / "raw"
        self.processed_path = self.root_path / "data" / "processed"

        # Ensure directory exists
        self.processed_path.mkdir(parents=True, exist_ok=True)

        # Mapping sources to their Schemas
        self.schema_map = {
            "yfinance": financials_schema,
            "fred": macro_schema,
            "worldbank": worldbank_schema,
        }

        self.lock = threading.Lock()
        self.quality_reports: Dict[str, Dict[str, Any]] = {}

    def run(self) -> None:
        """Orchestrates the processing of all catalog files in parallel."""
        self.logger.info("--- Silver Layer: Processing Started ---")
        catalog = self._load_catalog()

        # Prepare tasks for parallel execution
        tasks: List[Tuple[str, Dict[str, Any]]] = [
            (filename, info) for filename, info in catalog.items()
        ]

        # Execute tasks in parallel
        max_workers = min(self.config.max_workers, len(tasks))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(self._process_entity, filename, info)
                for filename, info in tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Raise any exceptions
                except (
                    DataValidationError,
                    ImputationError,
                    StandardizationError,
                    FileSaveError,
                    OutlierDetectionError,
                    ComplianceViolationError,
                ) as e:
                    self.logger.error(f"Processing error in task: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error in task: {e}")

        # Generate and save quality report
        self._generate_quality_report()

    def _process_entity(self, filename: str, info: Dict[str, Any]) -> None:
        """Processes a single entity with audit trail and quality tracking."""
        try:
            self.logger.info(f"Processing entity: {filename} from {info['source']}")

            # 1. Load Data
            df = pd.read_parquet(info["path"])
            initial_rows = len(df)
            initial_nulls = df.isnull().sum().sum()

            # 2. Standardization (Units, Names, Dates)
            df, unit_normalized, temporal_aligned = self._standardize(
                df, info["source"]
            )

            # 3. Strategic Imputation with Winsorization
            df, imputed_count, outliers_clipped = self._impute(df, info["source"])

            # 4. Pandera Validation (The Ultimate Quality Gate)
            try:
                validated_df = self.schema_map[info["source"]].validate(df)
            except pandera.errors.SchemaError as e:
                raise DataValidationError(
                    f"Schema validation failed for {filename}: {e}"
                ) from e

            # 5. Add Audit Columns
            validated_df = self._add_audit_columns(
                validated_df,
                filename,
                info["source"],
                imputed_count,
                initial_rows,
                initial_nulls,
                outliers_clipped,
            )

            # 6. Persistent Storage
            self._save_to_silver(validated_df, filename, info["source"])

            # 7. Update Quality Report
            with self.lock:
                self.quality_reports[filename] = {
                    "source": info["source"],
                    "initial_rows": initial_rows,
                    "final_rows": len(validated_df),
                    "initial_nulls": int(initial_nulls),
                    "final_nulls": validated_df.isnull().sum().sum(),
                    "imputed_count": imputed_count,
                    "outliers_clipped": outliers_clipped,
                    "unit_normalized": unit_normalized,
                    "temporal_aligned": temporal_aligned,
                    "processed_at": datetime.now().isoformat(),
                    "status": "success",
                }

        except (
            DataValidationError,
            ImputationError,
            StandardizationError,
            FileSaveError,
            OutlierDetectionError,
            ComplianceViolationError,
        ) as e:
            with self.lock:
                self.quality_reports[filename] = {
                    "source": info["source"],
                    "error": str(e),
                    "processed_at": datetime.now().isoformat(),
                    "status": "failed",
                }
        except Exception as e:
            with self.lock:
                self.quality_reports[filename] = {
                    "source": info["source"],
                    "error": str(e),
                    "processed_at": datetime.now().isoformat(),
                    "status": "failed",
                }
            raise

    def _standardize(
        self, df: pd.DataFrame, source: str
    ) -> Tuple[pd.DataFrame, bool, bool]:
        """Unifies structure and measurement units with business alignment."""
        try:
            unit_normalized = False
            temporal_aligned = False

            # Snake_case columns (Standardization)
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]

            # Temporal Alignment: Force all dates to Month-End for joins
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df["date"] = df["date"] + pd.offsets.MonthEnd(0)
                temporal_aligned = True

            # Unit Normalization: Convert percentages to decimals automatically
            if source in ["fred", "worldbank"] and "value" in df.columns:
                # Detect if values are in percentage format and convert to decimals
                if df["value"].max() > 1.0:
                    df["value"] = df["value"] / 100.0
                    unit_normalized = True

            return df, unit_normalized, temporal_aligned
        except Exception as e:
            raise StandardizationError(
                f"Failed to standardize data for source {source}: {e}"
            ) from e

    def _impute(self, df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, int, int]:
        """Handle missing values with Quantile Winsorization
        and Z-Score outlier detection."""
        try:
            if df.empty:
                return df, 0, 0

            total_cells = len(df) * len(df.columns)
            initial_nulls = df.isnull().sum().sum()
            null_percentage = (initial_nulls / total_cells) * 100

            # Compliance Check: Reject if >30% nulls
            if null_percentage > 30:
                raise ComplianceViolationError(
                    f"Data quality violation: {null_percentage:.2f}% nulls"
                    "exceed 30% threshold for {source}"
                )

            # Outlier Detection with Z-Score and Winsorization
            numeric_cols = df.select_dtypes(include=[float, int]).columns
            outliers_clipped = 0
            for col in numeric_cols:
                if col in df.columns and not df[col].empty:
                    series = df[col].dropna()
                    if len(series) > 0:
                        # Z-Score check: |z| > 3.5
                        z_scores = np.abs((series - series.mean()) / series.std())
                        outliers = z_scores > 3.5
                        outliers_clipped += int(outliers.sum())

                        # Quantile Winsorization at P1 and P99
                        lower_bound = series.quantile(0.01)
                        upper_bound = series.quantile(0.99)
                        df[col] = np.clip(series, lower_bound, upper_bound)

            # Vectorized Imputation: Forward Fill for time series
            df = df.sort_values("date").reset_index(drop=True)
            df = df.ffill().bfill()

            final_nulls = df.isnull().sum().sum()
            imputed_count = initial_nulls - final_nulls

            # Update quality report with outlier info
            return df, imputed_count, outliers_clipped
        except ComplianceViolationError:
            raise
        except Exception as e:
            raise ImputationError(
                f"Failed to impute data for source {source}: {e}"
            ) from e

    def _add_audit_columns(
        self,
        df: pd.DataFrame,
        filename: str,
        source: str,
        imputed_count: int,
        initial_rows: int,
        initial_nulls: int,
        outliers_clipped: int,
    ) -> pd.DataFrame:
        """Adds audit trail columns to the DataFrame."""
        df = df.copy()
        df["processed_at"] = datetime.now().isoformat()
        df["silver_run_id"] = self.silver_run_id
        df["schema_version"] = "1.0"
        df["imputed_count"] = imputed_count
        df["outliers_clipped"] = outliers_clipped
        df["initial_rows"] = initial_rows
        df["initial_nulls"] = initial_nulls
        df["quality_score"] = (
            1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        ) * 100  # Percentage of non-null values
        return df

    def _save_to_silver(self, df: pd.DataFrame, filename: str, source: str) -> None:
        """Saves data organized by source with ZSTD compression
        and memory optimization."""
        try:
            output_dir = self.processed_path / source
            output_dir.mkdir(parents=True, exist_ok=True)

            # Memory Optimization: Convert string columns to category
            for col in df.select_dtypes(include=["object", "str"]):
                df[col] = df[col].astype("category")

            file_path = output_dir / f"{filename}_silver.parquet"
            df.to_parquet(
                file_path, index=False, compression="zstd"
            )  # ZSTD for better compression
            self.logger.info(
                f"Silver Asset Secured: {file_path} | Rows: {len(df)} |"
                "Memory Optimized"
            )
        except Exception as e:
            raise FileSaveError(
                f"Failed to save Silver data for {filename}: {e}"
            ) from e

    def _load_catalog(self) -> Dict[str, Any]:
        try:
            catalog_file = self.raw_path / "catalog.json"
            if not catalog_file.exists():
                raise CatalogNotFoundError(
                    f"Catalog not found at {catalog_file}. Run Bronze Layer first."
                )
            with open(catalog_file, "r") as f:
                return cast(Dict[str, Any], json.load(f))
        except CatalogNotFoundError:
            raise
        except Exception as e:
            raise CatalogNotFoundError(f"Failed to load catalog: {e}") from e

    def _generate_quality_report(self) -> None:
        """Generates and saves a comprehensive quality report."""
        report_path = self.processed_path / "quality_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "silver_run_id": self.silver_run_id,
                    "run_timestamp": datetime.now().isoformat(),
                    "files": self.quality_reports,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
        self.logger.info(f"Quality Report generated at {report_path}")

        # Log summary
        total_files = len(self.quality_reports)
        successful = sum(
            1 for r in self.quality_reports.values() if r.get("status") == "success"
        )
        failed = total_files - successful
        total_imputed = sum(
            r.get("imputed_count", 0)
            for r in self.quality_reports.values()
            if r.get("status") == "success"
        )
        total_outliers = sum(
            r.get("outliers_clipped", 0)
            for r in self.quality_reports.values()
            if r.get("status") == "success"
        )
        self.logger.info(
            f"Silver Processing Summary: {successful}/{total_files} files successful,"
            f" {failed} failed, {total_imputed} values imputed,"
            f" {total_outliers} outliers clipped."
        )
