import concurrent.futures
import hashlib
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

from exceptions.MedallionExceptions import (
    CatalogNotFoundError,
    ComplianceViolationError,
    DataValidationError,
    FileSaveError,
    ImputationError,
    OutlierDetectionError,
    SchemaMismatchError,
    StandardizationError,
)
from Fetchers.ProjectConfig import ProjectConfig

# Import Schemas
from .contracts import EXPECTED_SOURCES, SOURCE_CONTRACTS, get_series_contract
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
        self.user_id = getattr(self.config, "data_user_id", "default")

        # Use Pathlib for absolute safety on Windows/Linux
        self.root_path = Path(__file__).parents[3]
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
        self.dead_letter_path = self.processed_path / "quality" / "dead_letter.jsonl"

    def run(self) -> Dict[str, Any]:
        """Orchestrates the processing of all catalog files in parallel.

        Returns a summary dict with keys:
            success_count, failed_count, failed_entities
        """
        self.logger.info("--- Silver Layer: Processing Started ---")
        catalog = self._load_catalog()
        if not catalog:
            raise DataValidationError("Silver hard stop: raw catalog is empty. No entities to process.")

        # Prepare tasks for parallel execution
        tasks: List[Tuple[str, Dict[str, Any]]] = [(filename, info) for filename, info in catalog.items()]
        if not tasks:
            raise DataValidationError("Silver hard stop: no tasks resolved from catalog.")

        # Execute tasks in parallel
        max_workers = min(self.config.max_workers, len(tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_entity, filename, info) for filename, info in tasks]
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
                    SchemaMismatchError,
                ) as e:
                    self.logger.error(f"Processing error in task: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error in task: {e}")

        # Generate and save quality report
        self._generate_quality_report()

        # Build outcome summary from quality_reports
        failed_entities: List[str] = [k for k, v in self.quality_reports.items() if v.get("status") == "failed"]
        # Build source success tracking only for sources that actually appeared
        # in the catalog.  Sources that had zero tasks (e.g. FRED when the API
        # key is absent) are intentionally absent and must not be flagged as
        # missing — that would cause a false hard-fail.
        catalog_sources = {info.get("source") for info in catalog.values() if info.get("source")}
        source_success: Dict[str, int] = {src: 0 for src in catalog_sources}
        for report in self.quality_reports.values():
            src = report.get("source")
            if src in source_success and report.get("status") == "success":
                source_success[src] += 1
        missing_sources = [s for s, count in source_success.items() if count == 0]
        success_count = len(self.quality_reports) - len(failed_entities)
        return {
            "success_count": success_count,
            "failed_count": len(failed_entities),
            "failed_entities": failed_entities,
            "source_success": source_success,
            "missing_sources": missing_sources,
        }

    def _process_entity(self, filename: str, info: Dict[str, Any]) -> None:
        """Processes a single entity with audit trail and quality tracking."""
        try:
            self.logger.info(f"Processing entity: {filename} from {info['source']}")

            # 1. Load Data
            df = pd.read_parquet(info["path"])
            initial_rows = len(df)
            initial_nulls = df.isnull().sum().sum()
            input_hash = self._hash_input_frame(df)

            # Shift-left Data Contract checks before heavy transformations.
            self._preflight_contract_checks(df, info["source"], filename, info)

            # 2. Standardization (Units, Names, Dates)
            df, unit_normalized, temporal_aligned = self._standardize(df, info["source"], filename)

            # 3. Strategic Imputation with Winsorization
            df, imputed_count, outliers_clipped, max_col_null_pct, warnings = self._impute(df, info["source"], filename)

            # 4. Pandera Validation (The Ultimate Quality Gate)
            try:
                validated_df = self.schema_map[info["source"]].validate(df)
            except pandera.errors.SchemaError as e:
                raise DataValidationError(f"Schema validation failed for {filename}: {e}") from e

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
                    "initial_rows": int(initial_rows),
                    "final_rows": int(len(validated_df)),
                    "initial_nulls": int(initial_nulls),
                    "final_nulls": int(validated_df.isnull().sum().sum()),
                    "imputed_count": int(imputed_count),
                    "outliers_clipped": int(outliers_clipped),
                    "max_col_null_pct": float(max_col_null_pct),
                    "input_hash": input_hash,
                    "input_path": str(info.get("path", "")),
                    "unit_normalized": unit_normalized,
                    "temporal_aligned": temporal_aligned,
                    "warnings": warnings,
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
            SchemaMismatchError,
        ) as e:
            error_type = type(e).__name__
            short_msg = str(e)[:200]
            print(f"[Silver] ENTITY FAILED: {filename} ({info['source']}) — {error_type}: {short_msg}")
            with self.lock:
                self.quality_reports[filename] = {
                    "source": info["source"],
                    "error": str(e),
                    "processed_at": datetime.now().isoformat(),
                    "status": "failed",
                }
            self._append_dead_letter(filename, info, e)
        except Exception as e:
            error_type = type(e).__name__
            short_msg = str(e)[:200]
            print(f"[Silver] UNEXPECTED FAILURE: {filename} ({info['source']}) — {error_type}: {short_msg}")
            with self.lock:
                self.quality_reports[filename] = {
                    "source": info["source"],
                    "error": str(e),
                    "processed_at": datetime.now().isoformat(),
                    "status": "failed",
                }
            self._append_dead_letter(filename, info, e)
            raise

    def _standardize(self, df: pd.DataFrame, source: str, entity_name: str) -> Tuple[pd.DataFrame, bool, bool]:
        """Unifies structure and measurement units with business alignment."""
        try:
            canonical_entity = entity_name.split("__", 1)[0]
            unit_normalized = False
            temporal_aligned = False

            # Snake_case columns (Standardization)
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]

            # Temporal alignment by source:
            # - yfinance keeps daily granularity
            # - macro sources (fred/worldbank) align to month-end
            if "date" in df.columns:
                if source == "worldbank":
                    # World Bank often arrives as yearly values; if date is numeric
                    # year-like, convert explicitly to year-end to avoid epoch parsing.
                    year_like = pd.to_numeric(df["date"], errors="coerce")
                    if year_like.notna().all() and year_like.between(1800, 2200).all():
                        df["date"] = pd.to_datetime(
                            year_like.astype(int).astype(str) + "-12-31",
                            errors="coerce",
                        )
                    else:
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                else:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                if source in {"fred", "worldbank"}:
                    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
                else:
                    df["date"] = df["date"].dt.normalize()
                temporal_aligned = True

            # Unit normalization is intentionally explicit to avoid corrupting
            # absolute/index indicators (e.g., CPI indexes, energy usage levels).
            contract = SOURCE_CONTRACTS.get(source)
            percentage_entities = contract.percentage_entities if contract else set()
            series_contract = get_series_contract(source, canonical_entity)
            if (
                source in ["fred", "worldbank"]
                and "value" in df.columns
                and canonical_entity in percentage_entities
                and series_contract.unit_kind == "percentage"
            ):
                if df["value"].max() > 1.0:
                    df["value"] = df["value"] / 100.0
                    unit_normalized = True

            return df, unit_normalized, temporal_aligned
        except Exception as e:
            raise StandardizationError(f"Failed to standardize data for source {source}: {e}") from e

    def _impute(self, df: pd.DataFrame, source: str, entity_name: str) -> Tuple[pd.DataFrame, int, int, float, List[str]]:
        """Handle missing values with Quantile Winsorization
        and Z-Score outlier detection."""
        try:
            if df.empty:
                return df, 0, 0, 0.0, []

            initial_nulls = df.isnull().sum().sum()
            warnings: List[str] = []
            canonical_entity = entity_name.split("__", 1)[0]
            series_contract = get_series_contract(source, canonical_entity)

            # Compliance Check: Reject if ANY numeric column exceeds 30% nulls.
            # Checking per-column prevents dense columns from masking sparse ones.
            numeric_null_pct = df.select_dtypes(include=[float, int]).isnull().mean() * 100
            worst_col = numeric_null_pct.idxmax() if not numeric_null_pct.empty else None
            max_col_null_pct = float(numeric_null_pct.max()) if worst_col is not None else 0.0
            dynamic_threshold = self._resolve_dynamic_null_threshold(source, canonical_entity, base_override=series_contract.null_tolerance_pct)
            severe_threshold = min(
                100.0,
                dynamic_threshold + float(getattr(self.config, "silver_warn_to_fail_buffer", 15.0)),
            )
            # Small samples are common in unit tests and pilot runs; enforce
            # the strict null-threshold only when we have enough observations.
            if len(df) >= 10 and max_col_null_pct > severe_threshold:
                raise ComplianceViolationError(
                    "Data quality violation: "
                    f"column '{worst_col}' has {max_col_null_pct:.2f}%"
                    f" nulls, exceeding severe threshold {severe_threshold:.2f}%"
                    f" for {entity_name} ({source})"
                )
            if len(df) >= 10 and dynamic_threshold < max_col_null_pct <= severe_threshold:
                warnings.append(f"elevated_missingness:{worst_col}:{max_col_null_pct:.2f}%>{dynamic_threshold:.2f}%")

            # Outlier Detection with Z-Score and Winsorization
            numeric_cols = df.select_dtypes(include=[float, int]).columns
            outliers_clipped = 0
            for col in numeric_cols:
                if col in df.columns and not df[col].empty:
                    series = df[col].dropna()
                    if len(series) > 0:
                        # Z-Score check: |z| > 3.5
                        std = series.std()
                        if std == 0 or pd.isna(std):
                            continue
                        z_scores = np.abs((series - series.mean()) / std)
                        outliers = z_scores > series_contract.outlier_z_threshold
                        outliers_clipped += int(outliers.sum())

                        # Quantile Winsorization at P1 and P99
                        if series_contract.enable_winsorization:
                            lower_bound = series.quantile(0.01)
                            upper_bound = series.quantile(0.99)
                            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

            # Vectorized Imputation: Forward Fill for time series
            df = df.sort_values("date").reset_index(drop=True)
            if series_contract.imputation_strategy == "ffill_bfill":
                df = df.ffill().bfill()

            final_nulls = df.isnull().sum().sum()
            imputed_count = initial_nulls - final_nulls
            outlier_ratio = outliers_clipped / max(len(df), 1)
            warn_ratio = float(getattr(self.config, "silver_outlier_warning_ratio", 0.1))
            if outlier_ratio > warn_ratio:
                warnings.append(f"high_outlier_intensity:{outlier_ratio:.3f}>{warn_ratio:.3f}")

            # Update quality report with outlier info
            return df, imputed_count, outliers_clipped, max_col_null_pct, warnings
        except ComplianceViolationError:
            raise
        except Exception as e:
            raise ImputationError(f"Failed to impute data for source {source}: {e}") from e

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
        # Integrity-focused quality score: penalize starting null density,
        # imputation burden, and outlier clipping intensity.
        base_cells = max(initial_rows * max(len(df.columns), 1), 1)
        initial_null_ratio = max(0.0, min(1.0, initial_nulls / base_cells))
        imputation_ratio = max(0.0, min(1.0, imputed_count / max(initial_rows, 1)))
        outlier_ratio = max(0.0, min(1.0, outliers_clipped / max(initial_rows, 1)))
        null_penalty = min(initial_null_ratio * 100.0, 40.0)
        imputation_penalty = min(imputation_ratio * 100.0, 30.0)
        outlier_penalty = min(outlier_ratio * 100.0, 20.0)
        df["quality_score"] = max(
            0.0,
            100.0 - (null_penalty + imputation_penalty + outlier_penalty),
        )
        return df

    def _save_to_silver(self, df: pd.DataFrame, filename: str, source: str) -> None:
        """Saves data organized by source with ZSTD compression
        and memory optimization."""
        try:
            output_dir = self.processed_path / source
            output_dir.mkdir(parents=True, exist_ok=True)

            # pandas 2.x may return StringDtype columns from parquet reads.
            # pyarrow rejects numpy StringDtype on write → cast to object first.
            for col in df.columns:
                if isinstance(df[col].dtype, pd.StringDtype):
                    df[col] = df[col].astype(object)
            # Memory Optimization: Convert plain object columns to category
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype("category")

            file_path = output_dir / f"{filename}_silver.parquet"
            df.to_parquet(file_path, index=False, compression="zstd")  # ZSTD for better compression
            self.logger.info(f"Silver Asset Secured: {file_path} | Rows: {len(df)} |Memory Optimized")
        except Exception as e:
            raise FileSaveError(f"Failed to save Silver data for {filename}: {e}") from e

    def _load_catalog(self) -> Dict[str, Any]:
        try:
            catalog_file = self.raw_path / "catalog.json"
            if not catalog_file.exists():
                raise CatalogNotFoundError(f"Catalog not found at {catalog_file}. Run Bronze Layer first.")
            with open(catalog_file, "r") as f:
                return cast(Dict[str, Any], json.load(f))
        except CatalogNotFoundError:
            raise
        except Exception as e:
            raise CatalogNotFoundError(f"Failed to load catalog: {e}") from e

    def _hash_input_frame(self, df: pd.DataFrame) -> str:
        """Create deterministic lineage hash for raw input frame."""
        payload = df.to_json(orient="split", date_format="iso")
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _preflight_contract_checks(self, df: pd.DataFrame, source: str, filename: str, info: Dict[str, Any]) -> None:
        """Shift-left data contract validation and hard-stop guardrails."""
        if df is None or df.empty:
            raise ComplianceViolationError(f"Silver hard stop: '{filename}' from {source} is empty.")

        min_rows = int(getattr(self.config, "silver_min_rows", 10))
        if len(df) < min_rows:
            raise ComplianceViolationError(f"Silver hard stop: '{filename}' has only {len(df)} rows (< minimum {min_rows}).")

        expected_rows = int(info.get("rows", 0) or 0)
        min_ratio = float(getattr(self.config, "silver_min_rows_ratio", 0.1))
        if expected_rows > 0:
            required_rows = max(1, int(expected_rows * min_ratio))
            if len(df) < required_rows:
                raise ComplianceViolationError(
                    f"Silver hard stop: '{filename}' has {len(df)} rows; expected at least {required_rows} from Bronze catalog baseline ({expected_rows})."
                )

        normalized = {str(c).lower().replace(" ", "_") for c in df.columns}
        contract = SOURCE_CONTRACTS.get(source)
        expected_columns = contract.required_columns if contract else set()
        if expected_columns and not expected_columns.issubset(normalized):
            missing = sorted(expected_columns - normalized)
            raise SchemaMismatchError(f"Schema drift detected for '{filename}' ({source}). Missing columns after normalization: {missing}.")

    def _append_dead_letter(self, filename: str, info: Dict[str, Any], exc: Exception) -> None:
        """Persist rejected entities for triage/replay workflows."""
        self.dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "silver_run_id": self.silver_run_id,
            "user_id": self.user_id,
            "entity": filename,
            "source": info.get("source"),
            "input_path": str(info.get("path", "")),
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        with self.dead_letter_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _resolve_dynamic_null_threshold(self, source: str, entity_name: str, base_override: float | None = None) -> float:
        """Compute adaptive null threshold from historical quality metrics."""
        base = float(base_override) if base_override is not None else float(getattr(self.config, "silver_base_null_threshold", 30.0))
        history_file = self.processed_path / "quality_history.jsonl"
        if not history_file.exists():
            return base

        window = int(getattr(self.config, "silver_dynamic_threshold_window", 20))
        samples: List[float] = []
        try:
            with history_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if payload.get("source") != source:
                        continue
                    if payload.get("entity") not in {entity_name, None, ""}:
                        continue
                    value = payload.get("max_col_null_pct")
                    if isinstance(value, (int, float)):
                        samples.append(float(value))
        except (OSError, json.JSONDecodeError):
            return base

        if not samples:
            return base
        recent = samples[-window:]
        # Dynamic rule: mean + 2*std capped in [base, 70].
        # The upper bound is 70% (not 90%) to prevent the threshold from
        # eroding into meaninglessness after a run with unusually high nulls.
        mean = float(np.mean(recent))
        std = float(np.std(recent))
        adaptive = max(base, min(70.0, mean + (2.0 * std)))
        return adaptive

    def _append_quality_history(self) -> None:
        """Persist run quality metrics for trend monitoring over time."""
        history_file = self.processed_path / "quality_history.jsonl"
        ts = datetime.now().isoformat()
        with history_file.open("a", encoding="utf-8") as f:
            for entity, metrics in self.quality_reports.items():
                row = {
                    "timestamp": ts,
                    "silver_run_id": self.silver_run_id,
                    "user_id": self.user_id,
                    "entity": entity,
                    "source": metrics.get("source"),
                    "status": metrics.get("status"),
                    "initial_rows": metrics.get("initial_rows"),
                    "final_rows": metrics.get("final_rows"),
                    "initial_nulls": metrics.get("initial_nulls"),
                    "final_nulls": metrics.get("final_nulls"),
                    "imputed_count": metrics.get("imputed_count", 0),
                    "outliers_clipped": metrics.get("outliers_clipped", 0),
                    "max_col_null_pct": float(metrics.get("max_col_null_pct", 0.0)),
                }
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _generate_quality_report(self) -> None:
        """Generates and saves a comprehensive quality report."""
        quality_dir = self.processed_path / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        report_path = quality_dir / "quality_report.json"
        legacy_report_path = self.processed_path / "quality_report.json"
        expected_sources = EXPECTED_SOURCES
        source_success: Dict[str, int] = {k: 0 for k in expected_sources}
        source_failed: Dict[str, int] = {k: 0 for k in expected_sources}
        for report in self.quality_reports.values():
            source = report.get("source")
            status = report.get("status")
            if source in expected_sources:
                if status == "success":
                    source_success[source] += 1
                elif status == "failed":
                    source_failed[source] += 1
        missing_sources = [s for s in expected_sources if source_success[s] == 0]
        payload = {
            "silver_run_id": self.silver_run_id,
            "user_id": self.user_id,
            "run_timestamp": datetime.now().isoformat(),
            "files": self.quality_reports,
            "summary": {
                "source_success": source_success,
                "source_failed": source_failed,
                "missing_sources": missing_sources,
            },
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False, default=str)
        # Backward compatibility for UIs/scripts that still read processed/quality_report.json
        with open(legacy_report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False, default=str)
        self.logger.info(f"Quality Report generated at {report_path}")

        # Log summary
        total_files = len(self.quality_reports)
        successful = sum(1 for r in self.quality_reports.values() if r.get("status") == "success")
        failed = total_files - successful
        total_imputed = sum(r.get("imputed_count", 0) for r in self.quality_reports.values() if r.get("status") == "success")
        total_outliers = sum(r.get("outliers_clipped", 0) for r in self.quality_reports.values() if r.get("status") == "success")
        self.logger.info(
            f"Silver Processing Summary: {successful}/{total_files} files successful,"
            f" {failed} failed, {total_imputed} values imputed,"
            f" {total_outliers} outliers clipped."
        )
        self._append_quality_history()
