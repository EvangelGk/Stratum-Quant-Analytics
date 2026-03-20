import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _load_project_symbol(module_name: str, symbol_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, symbol_name)
    except Exception:
        return None


ProjectConfig = _load_project_symbol("Fetchers.ProjectConfig", "ProjectConfig")
EXPECTED_SOURCES = _load_project_symbol(
    "Medallion.silver.contracts", "EXPECTED_SOURCES"
) or {"yfinance", "fred", "worldbank"}
SOURCE_CONTRACTS = _load_project_symbol(
    "Medallion.silver.contracts", "SOURCE_CONTRACTS"
) or {}


class ScenarioAuditor:
    """Independent but integrated audit judge for the full Scenario Planner system."""

    def __init__(
        self,
        gold_path: Optional[str] = None,
        project_root: Optional[str] = None,
        user_id: Optional[str] = None,
        allowed_gap_days: int = 7,
    ) -> None:
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        load_dotenv(self.project_root / ".env")
        self.user_id = self._resolve_user_id(user_id)
        self.data_root = self.project_root / "data" / "users" / self.user_id
        self.raw_path = self.data_root / "raw"
        self.processed_path = self.data_root / "processed"
        self.gold_dir = self.data_root / "gold"
        self.logs_dir = self.project_root / "logs"
        self.output_dir = self.project_root / "output" / self.user_id
        self.quality_report_path = self.processed_path / "quality" / "quality_report.json"
        self.quality_history_path = self.processed_path / "quality_history.jsonl"
        self.governance_dir = self.gold_dir / "governance"
        self.gold_contract_path = self.gold_dir / "master_table_contract.json"
        self.gold_path = (
            Path(gold_path)
            if gold_path
            else self.gold_dir / "master_table.parquet"
        )
        self.logger = logging.getLogger("ScenarioAuditor")
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        self.config = self._load_config()
        self.expected_tickers = self._resolve_expected_tickers()
        self.expected_macro = self._resolve_expected_macro_map()
        self.expected_worldbank = self._resolve_expected_worldbank_map()
        self.expected_worldbank_economies = self._resolve_expected_worldbank_economies()
        env_allowed_gap = os.getenv("AUDITOR_ALLOWED_GAP_DAYS", "").strip()
        try:
            parsed_gap = int(env_allowed_gap) if env_allowed_gap else int(allowed_gap_days)
        except ValueError:
            parsed_gap = int(allowed_gap_days)
        self.allowed_gap_days = max(1, parsed_gap)

    def run_audit(self) -> Dict[str, Any]:
        """Run end-to-end quality, breadth, output and strictness checks."""
        report: Dict[str, Any] = {
            "status": "CRITICAL",
            "decision_ready": False,
            "project_root": str(self.project_root),
            "user_id": self.user_id,
            "gold_path": str(self.gold_path),
            "checks": {},
            "meta": {
                "expected_tickers": self.expected_tickers,
                "expected_macro_series": list(self.expected_macro.values()),
                "expected_worldbank_series": list(self.expected_worldbank.values()),
                "expected_worldbank_economies": self.expected_worldbank_economies,
            },
        }

        if not self.gold_path.exists():
            self.logger.error("Gold Table not found. Run the pipeline first.")
            report["error"] = "File Missing"
            self._print_summary(report)
            return report

        df = pd.read_parquet(self.gold_path)
        report["row_count"] = int(len(df))
        report["column_count"] = int(len(df.columns))

        report["checks"]["integration"] = self._check_sources(df)
        report["checks"]["density"] = self._check_density(df)
        report["checks"]["statistics"] = self._check_stats(df)
        report["checks"]["continuity"] = self._check_continuity(df)
        report["checks"]["survivorship"] = self._check_survivorship(df)
        report["checks"]["outputs"] = self._check_outputs()
        report["checks"]["thresholds"] = self._check_threshold_design()
        report["checks"]["governance"] = self._check_governance()

        failed_checks = [
            name for name, result in report["checks"].items() if not result.get("passed", False)
        ]
        warn_checks = [
            name
            for name, result in report["checks"].items()
            if result.get("status") == "warn"
        ]

        report["failed_checks"] = failed_checks
        report["warning_checks"] = warn_checks
        report["decision_ready"] = len(failed_checks) == 0
        report["status"] = self._overall_status(report)
        report["auditor_judgement"] = self._build_auditor_judgement(report)

        self._print_summary(report)
        return report

    def _resolve_user_id(self, explicit_user_id: Optional[str]) -> str:
        if explicit_user_id:
            return explicit_user_id
        return os.getenv("DATA_USER_ID", "default").strip() or "default"

    def _load_config(self) -> Any:
        if ProjectConfig is None:
            return None
        try:
            return ProjectConfig.load_from_env()
        except Exception:
            return None

    def _resolve_expected_tickers(self) -> List[str]:
        if self.config is not None:
            try:
                return list(self.config.get_targets())
            except Exception:
                pass
        raw = os.getenv("TARGET_TICKERS", "").strip()
        if raw:
            return [token.strip().upper() for token in raw.split(",") if token.strip()]
        mode = os.getenv("ENVIRONMENT", "sample").strip().lower()
        return ["AAPL", "F"] if mode == "sample" else ["AAPL", "TSLA", "MSFT", "WMT", "XOM"]

    def _resolve_expected_macro_map(self) -> Dict[str, str]:
        if self.config is not None and getattr(self.config, "macro_series_map", None):
            return dict(self.config.macro_series_map)
        return {
            "CPIAUCSL": "inflation",
            "PNRGINDEXM": "energy_index",
            "UNRATE": "unemployment_rate",
            "FEDFUNDS": "fed_funds_rate",
            "DGS10": "us10y_treasury_yield",
            "UMCSENT": "consumer_sentiment",
            "INDPRO": "industrial_production",
        }

    def _resolve_expected_worldbank_map(self) -> Dict[str, str]:
        if self.config is not None and getattr(self.config, "worldbank_indicator_map", None):
            return dict(self.config.worldbank_indicator_map)
        return {
            "NY.GDP.MKTP.KD.ZG": "gdp_growth",
            "EG.USE.PCAP.KG.OE": "energy_usage",
            "FP.CPI.TOTL.ZG": "inflation_wb",
            "SL.UEM.TOTL.ZS": "unemployment_wb",
            "NE.TRD.GNFS.ZS": "trade_openness",
        }

    def _resolve_expected_worldbank_economies(self) -> List[str]:
        if self.config is not None and getattr(self.config, "worldbank_economies", None):
            return list(self.config.worldbank_economies)
        raw = os.getenv("WORLDBANK_ECONOMIES", "").strip()
        if raw:
            return [token.strip().upper() for token in raw.split(",") if token.strip()]
        return ["WLD"]

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _latest_governance_file(self) -> Optional[Path]:
        if not self.governance_dir.exists():
            return None

        # Prefer the stable aggregated file written by _finalize_governance() at
        # the end of every pipeline run — it consolidates the worst-case outcome
        # across ALL tickers rather than returning an arbitrary per-ticker file.
        agg_file = self.governance_dir / "governance_decision_current_run.json"
        if agg_file.exists():
            return agg_file

        files = sorted(self.governance_dir.glob("governance_decision_*.json"))
        if not files:
            return None

        latest_run_id = self._latest_run_id()
        if latest_run_id:
            matching: List[Path] = []
            for file_path in files:
                payload = self._read_json(file_path)
                if str(payload.get("run_id", "")).strip() == latest_run_id:
                    matching.append(file_path)
            if matching:
                return matching[-1]

        return files[-1]

    def _latest_run_id(self) -> Optional[str]:
        if not self.logs_dir.exists():
            return None
        session_files = sorted(self.logs_dir.glob("session_summary_*.json"))
        if not session_files:
            return None
        latest = self._read_json(session_files[-1])
        run_id = (latest.get("session_info") or {}).get("run_id")
        return str(run_id).strip() if run_id else None

    def _check_sources(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check source integration breadth, coverage and whether all configured entities arrive."""
        catalog = self._read_json(self.raw_path / "catalog.json")
        source_counts = {source: 0 for source in EXPECTED_SOURCES}
        entity_names_by_source: Dict[str, List[str]] = {source: [] for source in EXPECTED_SOURCES}
        worldbank_economies_seen: set[str] = set()
        for entity, meta in catalog.items():
            source = str(meta.get("source", ""))
            if source in source_counts:
                source_counts[source] += 1
                entity_names_by_source[source].append(entity)
                if source == "worldbank":
                    if "__" in entity:
                        worldbank_economies_seen.add(entity.split("__", 1)[1].upper())
                    else:
                        worldbank_economies_seen.add("WLD")

        contract = self._read_json(self.gold_contract_path)
        contract_expected = contract.get("expected", {}) if isinstance(contract, dict) else {}
        contract_observed = contract.get("observed", {}) if isinstance(contract, dict) else {}

        yfinance_expected = contract_expected.get("yfinance", ["close", "log_return", "ticker", "volume"])
        fred_expected = contract_expected.get("fred", list(self.expected_macro.values()))
        worldbank_expected = contract_expected.get("worldbank", list(self.expected_worldbank.values()))

        yfinance_cols = [c for c in yfinance_expected if c in df.columns]
        fred_cols = [c for c in fred_expected if c in df.columns]
        worldbank_cols = [c for c in worldbank_expected if c in df.columns]

        source_presence: Dict[str, float] = {}
        source_cell_fill: Dict[str, float] = {}
        source_row_masks: Dict[str, pd.Series] = {}
        for source, cols in {
            "yfinance": yfinance_cols,
            "fred": fred_cols,
            "worldbank": worldbank_cols,
        }.items():
            if cols and len(df) > 0:
                row_mask = df[cols].notnull().any(axis=1)
                coverage = float(row_mask.mean() * 100.0)
                cell_fill = float(df[cols].notnull().mean().mean() * 100.0)
            else:
                row_mask = pd.Series([False] * len(df), index=df.index)
                coverage = 0.0
                cell_fill = 0.0
            source_presence[source] = round(coverage, 2)
            source_cell_fill[source] = round(cell_fill, 2)
            source_row_masks[source] = row_mask

        if len(df) > 0:
            active_source_counts = sum(mask.astype(int) for mask in source_row_masks.values())
            rows_with_all_sources_pct = float((active_source_counts == len(source_row_masks)).mean() * 100.0)
            avg_active_source_groups = float(active_source_counts.mean())
        else:
            rows_with_all_sources_pct = 0.0
            avg_active_source_groups = 0.0

        overall_integration = {
            "rows_with_all_sources_pct": round(rows_with_all_sources_pct, 2),
            "avg_active_source_groups_per_row": round(avg_active_source_groups, 2),
            "max_source_groups_per_row": len(source_row_masks),
        }

        breadth = {
            "yfinance": {
                "expected": len(self.expected_tickers),
                "observed": source_counts.get("yfinance", 0),
                "ratio": self._safe_ratio(source_counts.get("yfinance", 0), len(self.expected_tickers)),
                "entities": sorted(entity_names_by_source.get("yfinance", [])),
            },
            "fred": {
                "expected": len(self.expected_macro),
                "observed": source_counts.get("fred", 0),
                "ratio": self._safe_ratio(source_counts.get("fred", 0), len(self.expected_macro)),
                "entities": sorted(entity_names_by_source.get("fred", [])),
            },
            "worldbank": {
                "expected": len(self.expected_worldbank) * max(1, len(self.expected_worldbank_economies)),
                "observed": source_counts.get("worldbank", 0),
                "ratio": self._safe_ratio(
                    source_counts.get("worldbank", 0),
                    len(self.expected_worldbank) * max(1, len(self.expected_worldbank_economies)),
                ),
                "entities": sorted(entity_names_by_source.get("worldbank", [])),
                "economies_expected": self.expected_worldbank_economies,
                "economies_seen": sorted(worldbank_economies_seen),
                "economy_coverage_ratio": self._safe_ratio(
                    len(worldbank_economies_seen),
                    max(1, len(self.expected_worldbank_economies)),
                ),
            },
        }

        issues: List[str] = []
        missing_expected = {
            "yfinance": sorted([c for c in yfinance_expected if c not in df.columns]),
            "fred": sorted([c for c in fred_expected if c not in df.columns]),
            "worldbank": sorted([c for c in worldbank_expected if c not in df.columns]),
        }
        for source in EXPECTED_SOURCES:
            if breadth[source]["ratio"] < 0.75:
                issues.append(f"{source}_breadth_low")
            if source_presence[source] < 70.0:
                issues.append(f"{source}_row_coverage_low")
            if missing_expected.get(source):
                issues.append(f"{source}_schema_contract_mismatch")

        passed = len(issues) == 0
        status = "pass" if passed else ("warn" if len(issues) <= 2 else "fail")
        return {
            "passed": passed,
            "status": status,
            "metrics": source_presence,
            "cell_fill_pct": source_cell_fill,
            "overall": overall_integration,
            "breadth": breadth,
            "schema_contract": {
                "contract_exists": bool(contract),
                "expected": contract_expected,
                "observed": contract_observed,
                "missing_expected_columns": missing_expected,
            },
            "issues": issues,
            "interpretation": (
                "Checks whether the system ingests all three sources and captures most configured entities, not just a narrow corner of each source."
            ),
        }

    def _check_density(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check null density, dead columns and suspiciously sparse outputs."""
        if df.empty:
            return {
                "passed": False,
                "status": "fail",
                "null_pct": None,
                "row_non_null_pct": None,
                "zero_var_count": 0,
                "zero_var_columns": [],
                "all_zero_columns": [],
                "issues": ["no_rows_in_gold_table"],
                "interpretation": (
                    "No analytical rows were produced, so density cannot support decision use."
                ),
            }

        total_cells = max(df.shape[0] * df.shape[1], 1)
        null_pct = float((df.isnull().sum().sum() / total_cells) * 100.0)
        numeric_df = df.select_dtypes(include=[np.number])
        audit_columns = {
            "quality_score",
            "imputed_count",
            "outliers_clipped",
            "initial_rows",
            "initial_nulls",
        }
        candidate_cols = [c for c in numeric_df.columns if c not in audit_columns]
        zero_var_cols = [col for col in candidate_cols if numeric_df[col].dropna().std() == 0]
        all_zero_cols = [
            col
            for col in candidate_cols
            if not numeric_df[col].dropna().empty and float(numeric_df[col].abs().sum()) == 0.0
        ]
        row_non_null_pct = float(df.notnull().any(axis=1).mean() * 100.0) if len(df) else 0.0
        issues: List[str] = []

        if len(candidate_cols) == 0:
            issues.append("no_numeric_signal_columns")
        if not np.isfinite(null_pct) or not np.isfinite(row_non_null_pct):
            issues.append("density_metrics_invalid")
        if null_pct >= 25.0:
            issues.append("null_density_high")
        if row_non_null_pct <= 90.0:
            issues.append("row_coverage_low")
        if len(zero_var_cols) > 2:
            issues.append("too_many_zero_variance_columns")

        passed = len(issues) == 0
        status = "pass"
        if not passed:
            status = "fail" if ("density_metrics_invalid" in issues or "no_numeric_signal_columns" in issues or "null_density_high" in issues) else "warn"
        return {
            "passed": passed,
            "status": status,
            "null_pct": round(null_pct, 2),
            "row_non_null_pct": round(row_non_null_pct, 2),
            "zero_var_count": len(zero_var_cols),
            "zero_var_columns": zero_var_cols[:10],
            "all_zero_columns": all_zero_cols[:10],
            "issues": issues,
            "metrics": {
                "null_pct": round(null_pct, 2),
                "row_non_null_pct": round(row_non_null_pct, 2),
                "zero_var_count": len(zero_var_cols),
            },
            "interpretation": (
                "Checks whether the analytical table is dense enough to support modelling and whether numeric features are effectively dead."
            ),
        }

    def _check_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check whether produced values are statistically plausible for decision support."""
        issues: List[str] = []
        metrics: Dict[str, Any] = {}
        signals_evaluated = 0

        if "log_return" in df.columns:
            series = pd.to_numeric(df["log_return"], errors="coerce").dropna()
            if not series.empty:
                signals_evaluated += 1
                max_abs = float(series.abs().max())
                std = float(series.std()) if pd.notna(series.std()) else 0.0
                median = float(series.median())
                mad = float((series - median).abs().median())
                robust_floor = 0.5
                robust_dynamic_limit = max(robust_floor, median + 8.0 * max(mad, 1e-9))
                metrics["log_return_max_abs"] = round(max_abs, 6)
                metrics["log_return_std"] = round(std, 6)
                metrics["log_return_dynamic_limit"] = round(robust_dynamic_limit, 6)
                if max_abs > max(1.0, robust_dynamic_limit):
                    issues.append("extreme_log_return_detected")

        if "close" in df.columns and (pd.to_numeric(df["close"], errors="coerce") <= 0).any():
            issues.append("non_positive_prices_found")

        if "volume" in df.columns and (pd.to_numeric(df["volume"], errors="coerce") < 0).any():
            issues.append("negative_volume_found")

        if "quality_score" in df.columns:
            quality_mean = float(pd.to_numeric(df["quality_score"], errors="coerce").mean())
            if np.isfinite(quality_mean):
                signals_evaluated += 1
                metrics["avg_quality_score"] = round(quality_mean, 2)
                if quality_mean < 60.0:
                    issues.append("low_average_quality_score")

        if signals_evaluated == 0:
            issues.append("insufficient_statistical_signal")

        passed = len(issues) == 0
        status = "pass" if passed else ("fail" if "insufficient_statistical_signal" in issues else "warn")
        return {
            "passed": passed,
            "status": status,
            "issues": issues,
            "metrics": metrics,
            "interpretation": (
                "Checks whether prices, returns and quality scores look economically plausible rather than purely syntactically valid."
            ),
        }

    def check_temporal_continuity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess temporal continuity using business-day gaps and configurable threshold."""
        if "date" not in df.columns:
            return {
                "passed": False,
                "max_gap": None,
                "median_gap": None,
                "error": "No date column",
            }

        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"])
        if work.empty:
            return {
                "passed": False,
                "max_gap": None,
                "median_gap": None,
                "error": "No valid dates",
            }

        # Focus continuity on daily market series where ticker semantics apply.
        # Macro/worldbank joins can be sparse by design and should not dominate this check.
        if "ticker" in work.columns:
            work = work[work["ticker"].notna()].copy()
            work["ticker"] = work["ticker"].astype(str)
        else:
            work["ticker"] = "ALL"

        if "close" in work.columns:
            work = work[pd.to_numeric(work["close"], errors="coerce").notna()].copy()

        if work.empty:
            return {
                "passed": False,
                "max_gap": None,
                "median_gap": None,
                "error": "No ticker-level market rows for continuity",
            }

        duplicate_rows = int(work.duplicated(subset=["date", "ticker"]).sum())
        work = work.drop_duplicates(subset=["date", "ticker"]).sort_values(
            ["ticker", "date"]
        )

        per_group: Dict[str, Dict[str, Any]] = {}
        failed_groups: List[str] = []
        all_business_gaps: List[int] = []

        def _detect_cadence(dates: List[pd.Timestamp]) -> str:
            if len(dates) < 3:
                return "business_day"
            month_deltas = [
                (dates[i].year - dates[i - 1].year) * 12 + (dates[i].month - dates[i - 1].month)
                for i in range(1, len(dates))
            ]
            valid_month_deltas = [delta for delta in month_deltas if delta >= 0]
            if valid_month_deltas and float(np.median(valid_month_deltas)) >= 1.0:
                return "monthly"
            return "business_day"

        for ticker, group_df in work.groupby("ticker"):
            dates = group_df["date"].sort_values().tolist()
            if len(dates) < 2:
                per_group[str(ticker)] = {
                    "max_gap": 0,
                    "median_gap": 0,
                    "allowed_gap": self.allowed_gap_days,
                    "cadence": "unknown",
                }
                continue

            cadence = _detect_cadence(dates)
            if cadence == "monthly":
                # Month-based gap: consecutive monthly observations count as 1.
                gaps = [
                    max(
                        (dates[i].year - dates[i - 1].year) * 12
                        + (dates[i].month - dates[i - 1].month),
                        0,
                    )
                    for i in range(1, len(dates))
                ]
                allowed_gap = 1
            else:
                # Business-day gap: Friday -> Monday counts as 1 day.
                gaps = [
                    max(len(pd.bdate_range(start=dates[i - 1], end=dates[i])) - 1, 0)
                    for i in range(1, len(dates))
                ]
                allowed_gap = self.allowed_gap_days
            if gaps:
                all_business_gaps.extend(gaps)
            max_gap = int(max(gaps)) if gaps else 0
            median_gap = float(np.median(gaps)) if gaps else 0.0
            per_group[str(ticker)] = {
                "max_gap": max_gap,
                "median_gap": round(median_gap, 2),
                "allowed_gap": allowed_gap,
                "cadence": cadence,
            }
            if max_gap > allowed_gap:
                failed_groups.append(str(ticker))

        max_gap_all = int(max(all_business_gaps)) if all_business_gaps else 0
        median_gap_all = float(np.median(all_business_gaps)) if all_business_gaps else 0.0
        cadence_modes = [
            payload.get("cadence")
            for payload in per_group.values()
            if isinstance(payload, dict) and payload.get("cadence")
        ]
        dominant_cadence = max(set(cadence_modes), key=cadence_modes.count) if cadence_modes else "unknown"
        overall_allowed_gap = 1 if dominant_cadence == "monthly" else self.allowed_gap_days
        warnings: List[str] = []
        if duplicate_rows > 0:
            warnings.append(
                f"deduplicated_rows:{duplicate_rows} based on ['date','ticker']"
            )
        if median_gap_all == 0:
            warnings.append(
                "median_gap_is_zero; verify duplicate timestamps or intraday consolidation"
            )

        # Median gap at zero with deduplications strongly suggests duplicate timestamp noise.
        duplicate_signal = duplicate_rows > 0 and median_gap_all == 0
        passed = len(failed_groups) == 0 and not duplicate_signal
        return {
            "passed": passed,
            "max_gap": max_gap_all,
            "median_gap": round(median_gap_all, 2),
            "allowed_gap": overall_allowed_gap,
            "failed_groups": failed_groups,
            "per_group": per_group,
            "warnings": warnings,
            "duplicate_rows_removed": duplicate_rows,
            "duplicate_signal": duplicate_signal,
            "cadence": dominant_cadence,
        }

    def _check_continuity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compatibility wrapper around temporal continuity check."""
        continuity = self.check_temporal_continuity(df)
        passed = bool(continuity.get("passed", False))
        status = "pass" if passed else "warn"
        continuity["status"] = status
        continuity["metrics"] = continuity.get("per_group", {})
        continuity["interpretation"] = (
            "Evaluates continuity with business-day gaps and configurable threshold. "
            "A weekend-only break is not treated as a major gap."
        )
        return continuity

    def _check_outputs(self) -> Dict[str, Any]:
        """Check whether outputs are non-empty, non-zero and usable for downstream decisions."""
        summary_path = self.output_dir / "analysis_results.json"
        summary = self._read_json(summary_path)
        result_keys = list(summary.get("result_keys", [])) if summary else []
        results = summary.get("results", {}) if isinstance(summary, dict) else {}

        blocked = []
        nullish = []
        usable = []
        for key, value in results.items():
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
            if isinstance(value, str) and value.startswith("blocked_by_governance_gate"):
                blocked.append(key)
            elif value in (None, "", [], {}):
                nullish.append(key)
            else:
                usable.append(key)

        passed = bool(result_keys) and len(usable) >= 2
        status = "pass" if passed else ("warn" if result_keys else "fail")
        return {
            "passed": passed,
            "status": status,
            "summary_exists": summary_path.exists(),
            "result_key_count": len(result_keys),
            "usable_outputs": usable,
            "blocked_outputs": blocked,
            "nullish_outputs": nullish,
            "interpretation": (
                "Checks whether the project emits decision-support artifacts rather than empty or fully blocked placeholders."
            ),
        }

    def _check_survivorship(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Flag survivorship bias risks from delisting/stale coverage and prolonged zero-volume spans."""
        if df.empty or "ticker" not in df.columns:
            return {
                "passed": True,
                "status": "warn",
                "issues": ["survivorship_check_skipped_missing_ticker"],
                "interpretation": [
                    "Ticker column missing; survivorship bias cannot be assessed in this run."
                ],
            }

        work = df.copy()
        work["date"] = pd.to_datetime(work.get("date"), errors="coerce")
        work = work.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
        if work.empty:
            return {
                "passed": True,
                "status": "warn",
                "issues": ["survivorship_check_skipped_no_valid_rows"],
                "interpretation": [
                    "No valid ticker/date rows available for survivorship diagnostics."
                ],
            }

        max_date = work["date"].max()
        stale_tickers: List[str] = []
        zero_volume_tickers: List[str] = []
        zero_volume_spans: Dict[str, int] = {}

        for ticker, group in work.groupby("ticker"):
            ticker_name = str(ticker)
            last_date = group["date"].max()
            if pd.notna(last_date):
                stale_gap = int(len(pd.bdate_range(last_date, max_date)) - 1)
                if stale_gap > 20:
                    stale_tickers.append(f"{ticker_name}:{stale_gap}bdays")

            if "volume" not in group.columns:
                continue
            volume = pd.to_numeric(group["volume"], errors="coerce").fillna(0.0)
            zero_mask = volume <= 0.0
            run_lengths: List[int] = []
            current = 0
            for is_zero in zero_mask.tolist():
                if is_zero:
                    current += 1
                elif current > 0:
                    run_lengths.append(current)
                    current = 0
            if current > 0:
                run_lengths.append(current)

            max_run = max(run_lengths) if run_lengths else 0
            if max_run > 10:
                zero_volume_tickers.append(ticker_name)
                zero_volume_spans[ticker_name] = int(max_run)

        issues: List[str] = []
        if stale_tickers:
            issues.append("possible_delisting_or_stale_coverage")
        if zero_volume_tickers:
            issues.append("prolonged_zero_volume_detected")

        severe_issue_count = len(stale_tickers) + len(zero_volume_tickers)
        severe = severe_issue_count >= max(3, int(max(len(work["ticker"].unique()), 1) * 0.5))

        return {
            "passed": not severe,
            "status": "fail" if severe else ("warn" if issues else "pass"),
            "issues": issues,
            "metrics": {
                "stale_ticker_count": len(stale_tickers),
                "zero_volume_ticker_count": len(zero_volume_tickers),
                "max_zero_volume_streak_days": max(zero_volume_spans.values()) if zero_volume_spans else 0,
            },
            "stale_tickers": stale_tickers,
            "zero_volume_streaks": zero_volume_spans,
            "interpretation": [
                "Flags survivorship risk where names disappear before dataset end-date.",
                "Flags tickers with >10 consecutive zero-volume days that can bias return estimates.",
            ],
        }

    def _check_threshold_design(self) -> Dict[str, Any]:
        """Check whether thresholds are dynamic or fixed and whether strictness is balanced."""
        latest_governance = self._read_json(self._latest_governance_file()) if self._latest_governance_file() else {}
        gate = latest_governance.get("gate", {}) if isinstance(latest_governance, dict) else {}
        report = latest_governance.get("report", {}) if isinstance(latest_governance, dict) else {}

        silver_dynamic = bool(self.quality_history_path.exists()) and bool(
            getattr(self.config, "silver_dynamic_threshold_window", 0) > 1 if self.config else True
        )
        contract_driven = bool(SOURCE_CONTRACTS)
        stationarity_context = gate.get("stationarity_context", {}) if isinstance(gate, dict) else {}
        walk_forward_context = gate.get("walk_forward_context", {}) if isinstance(gate, dict) else {}
        governance_stationarity_dynamic = (
            stationarity_context.get("applied_min_stationary_ratio")
            != stationarity_context.get("base_min_stationary_ratio")
        )
        governance_walk_forward_dynamic = (
            walk_forward_context.get("applied_min_walk_forward_r2")
            != walk_forward_context.get("base_min_walk_forward_r2")
            or bool(walk_forward_context.get("metric_unstable"))
            or isinstance((report.get("walk_forward") or {}).get("clipped_avg_r2"), (int, float))
        )

        strictness_findings: List[str] = []
        if self.config is not None and getattr(self.config, "governance_hard_fail", True):
            strictness_findings.append("governance_hard_fail_enabled")
        if self.config is not None and getattr(self.config, "silver_hard_fail", True):
            strictness_findings.append("silver_hard_fail_enabled")
        if not silver_dynamic:
            strictness_findings.append("silver_null_threshold_history_not_active")
        if not governance_stationarity_dynamic:
            strictness_findings.append("stationarity_threshold_not_adapted_recently")

        passed = contract_driven and (silver_dynamic or governance_walk_forward_dynamic)
        return {
            "passed": passed,
            "status": "pass" if passed else "warn",
            "dynamic_thresholds": {
                "silver_dynamic_null_thresholds": silver_dynamic,
                "series_contract_thresholds": contract_driven,
                "governance_stationarity_dynamic": governance_stationarity_dynamic,
                "governance_walk_forward_dynamic": governance_walk_forward_dynamic,
            },
            "strictness_findings": strictness_findings,
            "interpretation": (
                "Reports whether the system uses adaptive thresholds or relies mostly on static hard-coded limits."
            ),
        }

    def _check_governance(self) -> Dict[str, Any]:
        """Check whether governance decisions are realistic for data regime and risk."""
        latest_file = self._latest_governance_file()
        if latest_file is None:
            return {
                "passed": False,
                "status": "warn",
                "error": "No governance decision artifact found",
            }

        payload = self._read_json(latest_file)
        gate = payload.get("gate", {}) if isinstance(payload, dict) else {}
        report = payload.get("report", {}) if isinstance(payload, dict) else {}
        reasons = list(gate.get("reasons", [])) if isinstance(gate, dict) else []
        walk_forward = report.get("walk_forward", {}) if isinstance(report, dict) else {}
        oos = report.get("out_of_sample", {}) if isinstance(report, dict) else {}

        clipped_avg_r2 = walk_forward.get("clipped_avg_r2")
        avg_r2 = walk_forward.get("avg_r2")
        median_r2 = walk_forward.get("median_r2")
        model_risk_score = report.get("model_risk_score")
        oos_r2_ci = oos.get("r2_ci", {}) or {}
        wf_r2_ci_lower = walk_forward.get("r2_ci_lower")
        wf_r2_ci_upper = walk_forward.get("r2_ci_upper")
        passed_gate = bool(gate.get("passed", False))
        severity = str(gate.get("severity", "unknown"))

        likely_over_strict = False
        reasoning: List[str] = []
        publication_lag_ok = True
        publication_lag_findings: List[str] = []

        dq_context = report.get("data_quality_context", {}) if isinstance(report, dict) else {}
        transformations = dq_context.get("transformations", {}) if isinstance(dq_context, dict) else {}
        if isinstance(transformations, dict):
            for feature, meta in transformations.items():
                if not isinstance(meta, dict):
                    continue
                source = str(meta.get("source", "")).lower()
                if source not in {"fred", "worldbank"}:
                    continue
                pub_lag = int(meta.get("publication_lag_days", meta.get("lag_days", 0)) or 0)
                transform_method = str(meta.get("transformation", "")).lower()
                if pub_lag < 45:
                    publication_lag_ok = False
                    publication_lag_findings.append(
                        f"{feature}:publication_lag_days={pub_lag}<45"
                    )
                if not transform_method.startswith("release_"):
                    publication_lag_ok = False
                    publication_lag_findings.append(
                        f"{feature}:transformation={transform_method or 'unknown'}(expected release_* after lag->ffill)"
                    )

        if any("r2_metric_alert_walk_forward_below_threshold" in reason for reason in reasons):
            if isinstance(clipped_avg_r2, (int, float)) and isinstance(avg_r2, (int, float)):
                if float(avg_r2) < -5.0 and float(clipped_avg_r2) > -2.0:
                    likely_over_strict = True
                    reasoning.append("raw_walk_forward_avg_r2_is_extreme_but_clipped_metric_is_much_healthier")
        if any("stationarity_ratio_below_threshold" in reason for reason in reasons):
            stationarity_context = gate.get("stationarity_context", {})
            considered = int(stationarity_context.get("considered_series", 0) or 0)
            if considered <= 3:
                likely_over_strict = True
                reasoning.append("stationarity_ratio_decision_based_on_few_series")
        if isinstance(model_risk_score, (int, float)) and float(model_risk_score) < 0.6 and not passed_gate:
            likely_over_strict = True
            reasoning.append("gate_blocked_even_though_model_risk_score_is_below_fail_band")

        theoretically_sound = True
        theory_notes = [
            "Using governance to block advanced analyses is defensible when predictive diagnostics are poor.",
            "Stationarity, leakage, drift and walk-forward checks are valid analysis-governance dimensions.",
        ]
        if isinstance(oos.get("r2"), (int, float)) and float(oos["r2"]) < -0.25:
            theory_notes.append("Out-of-sample explanatory power is weak, so caution is appropriate.")
        if likely_over_strict:
            theory_notes.append("Current block may be stricter than necessary for mixed-frequency exploratory analysis.")
        if publication_lag_ok:
            theory_notes.append("Look-ahead bias guardrail detected: macro features carry publication lag metadata.")
        else:
            theory_notes.append("Look-ahead bias risk: one or more macro features do not satisfy 45-day publication lag.")

        passed = passed_gate or likely_over_strict
        return {
            "passed": passed,
            "status": "pass" if passed_gate else ("warn" if likely_over_strict else "fail"),
            "gate_passed": passed_gate,
            "severity": severity,
            "reasons": reasons,
            "metrics": {
                "out_of_sample_r2": oos.get("r2"),
                "out_of_sample_r2_ci_lower": oos_r2_ci.get("ci_lower"),
                "out_of_sample_r2_ci_upper": oos_r2_ci.get("ci_upper"),
                "out_of_sample_r2_ci_confidence": oos_r2_ci.get("confidence"),
                "walk_forward_avg_r2": avg_r2,
                "walk_forward_median_r2": median_r2,
                "walk_forward_clipped_avg_r2": clipped_avg_r2,
                "walk_forward_r2_ci_lower": wf_r2_ci_lower,
                "walk_forward_r2_ci_upper": wf_r2_ci_upper,
                "model_risk_score": model_risk_score,
                "publication_lag_compliant": publication_lag_ok,
            },
            "likely_over_strict": likely_over_strict,
            "publication_lag_findings": publication_lag_findings,
            "theoretically_sound": theoretically_sound,
            "reasoning": reasoning,
            "interpretation": theory_notes,
        }

    def _safe_ratio(self, observed: int, expected: int) -> float:
        if expected <= 0:
            return 1.0
        return round(observed / expected, 4)

    def _overall_status(self, report: Dict[str, Any]) -> str:
        failed = report.get("failed_checks", [])
        warns = report.get("warning_checks", [])
        if failed:
            return "CRITICAL"
        if warns:
            return "WARN"
        return "PASS"

    def _build_auditor_judgement(self, report: Dict[str, Any]) -> Dict[str, Any]:
        governance = report["checks"].get("governance", {})
        thresholds = report["checks"].get("thresholds", {})
        integration = report["checks"].get("integration", {})
        outputs = report["checks"].get("outputs", {})

        summary_lines = [
            f"Overall status: {report.get('status')}",
            f"Decision ready: {report.get('decision_ready')}",
            "This auditor is independent from the pipeline runtime but integrated with its artifacts, contracts and governance outputs.",
        ]
        if integration.get("issues"):
            summary_lines.append(f"Integration issues: {integration.get('issues')}")
        if outputs.get("blocked_outputs"):
            summary_lines.append(
                f"Blocked analyses: {outputs.get('blocked_outputs')}"
            )
        if governance.get("likely_over_strict"):
            summary_lines.append(
                "Governance appears methodologically valid but may be conservative for the current mixed-frequency regime."
            )
        if governance.get("publication_lag_findings"):
            summary_lines.append(
                f"Look-ahead lag findings: {governance.get('publication_lag_findings')}"
            )
        survivorship = report["checks"].get("survivorship", {})
        if survivorship.get("issues"):
            summary_lines.append(
                f"Survivorship flags: {survivorship.get('issues')}"
            )
        if thresholds.get("dynamic_thresholds", {}).get("silver_dynamic_null_thresholds"):
            summary_lines.append("Silver null thresholds are dynamic/history-aware.")
        else:
            summary_lines.append("Silver null thresholds look mostly static right now.")

        return {
            "is_information_reasonable": report.get("status") != "CRITICAL",
            "can_support_decisions": report.get("decision_ready", False),
            "summary": summary_lines,
        }

    def _print_summary(self, report: Dict[str, Any]) -> None:
        status_icon = {"PASS": "✅", "WARN": "⚠️", "CRITICAL": "❌"}
        check_icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
        print("\n" + "=" * 60)
        print("SCENARIO PLANNER - SYSTEM AUDIT REPORT")
        print("=" * 60)
        overall_status = str(report.get("status", "UNKNOWN")).upper()
        print(
            f"Status: {status_icon.get(overall_status, 'ℹ️')} {overall_status}"
        )
        print(f"Decision Ready: {report.get('decision_ready', False)}")
        print(f"User: {report.get('user_id', 'unknown')}")
        print(f"Rows: {report.get('row_count', 0)} | Columns: {report.get('column_count', 0)}")
        for name, result in report.get("checks", {}).items():
            mark = "PASS" if result.get("passed") else "FAIL"
            status = result.get("status", "info")
            status_norm = str(status).lower()
            icon = check_icon.get(status_norm, "ℹ️")
            print(f"- {name.capitalize():<12} -> {icon} {mark} ({status})")
        print("=" * 60)
        for line in report.get("auditor_judgement", {}).get("summary", []):
            print(f"* {line}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    auditor = ScenarioAuditor()
    auditor.run_audit()
