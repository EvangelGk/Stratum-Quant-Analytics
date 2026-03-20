import concurrent.futures
import json
import logging
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from exceptions.MedallionExceptions import AnalysisError
from logger.Catalog import catalog
from logger.Messages.DirectionsMess import (
    LIVE_STEP_7_RESULTS_GENERATION,
)
from logger.Messages.MainMess import (
    ANALYSIS_CORRELATION_MATRIX,
    ANALYSIS_ELASTICITY,
    ANALYSIS_LAG_ANALYSIS,
    ANALYSIS_MONTE_CARLO,
    ANALYSIS_SENSITIVITY_REGRESSION,
    ANALYSIS_STRESS_TEST,
)

from .AnalysisSuite.auto_ml import auto_ml_regression
from .AnalysisSuite.correl_mtrx import correl_mtrx
from .AnalysisSuite.elasticity import elasticity
from .AnalysisSuite.forecasting import forecasting
from .AnalysisSuite.governance import governance_report
from .AnalysisSuite.lag import lag_analysis
from .AnalysisSuite.monte_carlo import monte_carlo
from .AnalysisSuite.sensitivity_reg import sensitivity_reg
from .AnalysisSuite.stress_test import stress_test


class GoldLayer:
    """
    The Crown Jewel of the Pipeline.
    Responsibility: Feature Engineering & Unified Analytical View.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        _root = Path(__file__).parents[3]
        self.processed_path = _root / "data" / "processed"
        self.gold_path = _root / "data" / "gold"
        self.gold_path.mkdir(parents=True, exist_ok=True)
        self.governance_path = self.gold_path / "governance"
        self.governance_path.mkdir(parents=True, exist_ok=True)
        self.df: pd.DataFrame = pd.DataFrame()  # deferred; call initialize_data()

    def initialize_data(self) -> None:
        """Explicitly load (or build) the master table after paths are finalised.

        The pipeline calls this after overriding processed_path / gold_path so
        that the correct directories are always used regardless of launch context.
        """
        self.gold_path.mkdir(parents=True, exist_ok=True)
        self.governance_path = self.gold_path / "governance"
        self.governance_path.mkdir(parents=True, exist_ok=True)
        self.df = self._load_or_create_master_table()

    def _load_or_create_master_table(self) -> pd.DataFrame:
        """
        Load master table if exists, else create it.
        """
        master_file = self.gold_path / "master_table.parquet"
        if master_file.exists():
            self.logger.info("Loading existing master table...")
            return pd.read_parquet(master_file)
        else:
            return self.create_master_table()

    def create_master_table(self) -> pd.DataFrame:
        """
        Denormalizes Silver data into a single 'Feature Store'.
        Implements Log-Returns transformation for statistical normality.
        """
        self.logger.info("Building Master Analytical Table...")

        # 1. Load Financials
        financial_files = list((self.processed_path / "yfinance").glob("*.parquet"))
        if not financial_files:
            raise ValueError("No financial data files found in processed/yfinance")
        dfs: List[pd.DataFrame] = []
        for f in financial_files:
            fin_df = pd.read_parquet(f)
            if "ticker" not in fin_df.columns:
                # Example: aapl_financials_silver.parquet -> AAPL
                derived_ticker = f.stem.replace("_financials_silver", "").upper()
                fin_df["ticker"] = derived_ticker
            else:
                fin_df["ticker"] = fin_df["ticker"].astype(str).str.upper()
            dfs.append(fin_df)
        master_df = pd.concat(dfs, ignore_index=True)

        if "close" not in master_df.columns:
            raise ValueError("Financial dataset missing required 'close' column")
        if "date" not in master_df.columns:
            raise ValueError("Financial dataset missing required 'date' column")

        # 2. Master Feature: Log Returns (The Senior Standard)
        # Formula: ln(P_t / P_{t-1})
        master_df["log_return"] = master_df.groupby("ticker")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )

        # 3. Join Macro Data (FRED)
        fred_files = list((self.processed_path / "fred").glob("*.parquet"))
        macro_columns: List[str] = []
        for f in fred_files:
            col_name = f.stem.replace("_silver", "")
            macro_df = pd.read_parquet(f).rename(columns={"value": col_name})
            master_df = pd.merge(
                master_df,
                macro_df[["date", col_name]],
                on="date",
                how="left",
            )
            macro_columns.append(col_name)

        # 4. Join World Bank Data
        wb_files = list((self.processed_path / "worldbank").glob("*.parquet"))
        for f in wb_files:
            col_name = f.stem.replace("_silver", "")
            wb_df = pd.read_parquet(f).rename(columns={"value": col_name})
            master_df = pd.merge(
                master_df,
                wb_df[["date", col_name]],
                on="date",
                how="left",
            )
            macro_columns.append(col_name)

        # 5. Forward-Fill Macro and World Bank Data
        master_df = master_df.sort_values(["ticker", "date"]).copy()
        if macro_columns:
            master_df[macro_columns] = (
                master_df.groupby("ticker", group_keys=False)[macro_columns]
                .ffill()
                .bfill()
            )

        # Save the "Analytical Base Table" with optional encryption
        table = pa.Table.from_pandas(master_df)
        pq.write_table(
            table, self.gold_path / "master_table.parquet", compression="zstd"
        )
        return master_df

    def _resolve_ticker(self, ticker: Optional[str]) -> Optional[str]:
        if ticker:
            return ticker
        if "ticker" not in self.df.columns:
            return None
        tickers = self.df["ticker"].dropna().unique().tolist()
        return tickers[0] if tickers else None

    def _resolve_random_seed(self) -> Optional[int]:
        enforce = bool(getattr(self.config, "enforce_reproducibility", True))
        if not enforce:
            return None
        seed = getattr(self.config, "random_seed", 42)
        return int(seed) if seed is not None else None

    def _evaluate_governance_gate(
        self,
        report: Optional[Dict[str, Any]],
        ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        profile = self._resolve_governance_profile(ticker=ticker)
        hard_fail = bool(profile["hard_fail"])
        gate_reasons: List[str] = []
        gate: Dict[str, Any] = {
            "schema_version": "governance-gate.v1",
            "hard_fail": hard_fail,
            "passed": True,
            "regime": profile["regime"],
            "severity": "pass",
            "reasons": gate_reasons,
        }

        if not report:
            if hard_fail:
                gate["passed"] = False
                gate_reasons.append("governance_report_unavailable")
            return gate

        status = report.get("status")
        if status == "insufficient_data":
            # Not enough samples for strict governance statistics: warn-only.
            gate_reasons.append("insufficient_data_for_governance_checks")
            return gate

        reasons: List[str] = []
        oos_r2 = report.get("out_of_sample", {}).get("r2")
        min_r2 = float(profile["min_r2"])
        if isinstance(oos_r2, (float, int)) and float(oos_r2) < min_r2:
            reasons.append(
                f"out_of_sample_r2_below_threshold:{oos_r2:.4f}<{min_r2:.4f}"
            )

        normalized_shift = report.get("stability", {}).get("normalized_mean_shift")
        max_shift = float(profile["max_normalized_shift"])
        if (
            isinstance(normalized_shift, (float, int))
            and float(normalized_shift) > max_shift
        ):
            reasons.append(
                "normalized_mean_shift_above_threshold:"
                f"{float(normalized_shift):.4f}>{max_shift:.4f}"
            )

        leakage_flags = report.get("leakage_flags", []) or []
        max_leakage = int(profile["max_leakage_flags"])
        if len(leakage_flags) > max_leakage:
            reasons.append(
                f"leakage_flags_above_threshold:{len(leakage_flags)}>{max_leakage}"
            )

        stationarity = report.get("stationarity", {}) or {}
        status_entries = [v for v in stationarity.values() if isinstance(v, dict)]
        stationary_count = sum(
            1 for v in status_entries if v.get("is_stationary") is True
        )
        considered = sum(
            1 for v in status_entries if v.get("is_stationary") is not None
        )
        if considered > 0:
            ratio = stationary_count / considered
            min_ratio = float(profile["min_stationary_ratio"])
            # Dynamic tolerance: when few series are testable, keep a softer floor.
            adaptive_min_ratio = (
                max(0.25, min_ratio - 0.10) if considered < 4 else min_ratio
            )
            if ratio < adaptive_min_ratio:
                reasons.append(
                    "stationarity_ratio_below_threshold:"
                    f"{ratio:.4f}<{adaptive_min_ratio:.4f}"
                )
            gate["stationarity_context"] = {
                "considered_series": considered,
                "stationary_series": stationary_count,
                "applied_min_stationary_ratio": adaptive_min_ratio,
                "base_min_stationary_ratio": min_ratio,
            }

        walk_forward = report.get("walk_forward", {}) or {}
        min_walk_forward_r2 = float(profile["min_walk_forward_r2"])
        walk_forward_avg_r2 = walk_forward.get("avg_r2")
        walk_forward_status = str(walk_forward.get("status", "unknown"))
        windows_requested = int(walk_forward.get("windows_requested", 0) or 0)
        windows_completed = int(walk_forward.get("windows_completed", 0) or 0)
        adaptive_min_walk_forward_r2 = min_walk_forward_r2
        # Dynamic tolerance: fewer completed windows -> less strict threshold.
        if windows_requested > 0 and windows_completed < windows_requested:
            adaptive_min_walk_forward_r2 = min(min_walk_forward_r2, -1.0)

        walk_forward_unstable = False
        if isinstance(walk_forward_avg_r2, (float, int)):
            # Extremely negative R2 often indicates unstable denominator/noisy slices.
            walk_forward_unstable = float(walk_forward_avg_r2) < -5.0

        if (
            isinstance(walk_forward_avg_r2, (float, int))
            and float(walk_forward_avg_r2) < adaptive_min_walk_forward_r2
        ):
            # If walk-forward metric is numerically unstable but OOS is acceptable,
            # surface as warning context (not hard fail reason).
            if not (
                walk_forward_unstable
                and isinstance(oos_r2, (float, int))
                and float(oos_r2) >= min_r2
            ):
                reasons.append(
                    "walk_forward_avg_r2_below_threshold:"
                    f"{float(walk_forward_avg_r2):.4f}"
                    f"<{adaptive_min_walk_forward_r2:.4f}"
                )
        gate["walk_forward_context"] = {
            "status": walk_forward_status,
            "windows_requested": windows_requested,
            "windows_completed": windows_completed,
            "applied_min_walk_forward_r2": adaptive_min_walk_forward_r2,
            "base_min_walk_forward_r2": min_walk_forward_r2,
            "metric_unstable": walk_forward_unstable,
        }

        model_risk_score = report.get("model_risk_score")
        max_model_risk = float(profile["max_model_risk_score"])
        if (
            isinstance(model_risk_score, (float, int))
            and float(model_risk_score) > max_model_risk
        ):
            reasons.append(
                "model_risk_score_above_threshold:"
                f"{float(model_risk_score):.4f}>{max_model_risk:.4f}"
            )

        if isinstance(model_risk_score, (float, int)):
            score = float(model_risk_score)
            warn_thr = float(profile["model_risk_warn_threshold"])
            fail_thr = float(profile["model_risk_fail_threshold"])
            if score >= fail_thr:
                gate["severity"] = "fail"
            elif score >= warn_thr:
                gate["severity"] = "warn"
            else:
                gate["severity"] = "pass"

            report.setdefault("risk_band", {})
            report["risk_band"] = {
                "label": gate["severity"],
                "warn_threshold": warn_thr,
                "fail_threshold": fail_thr,
                "score": score,
            }

        if hard_fail and reasons:
            gate["passed"] = False
        gate_reasons.extend(reasons)
        return gate

    def _resolve_governance_profile(
        self, ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        regime = str(getattr(self.config, "governance_regime", "normal")).lower()
        if regime not in {"normal", "stress", "crisis"}:
            regime = "normal"

        profile: Dict[str, Any] = {
            "regime": regime,
            "hard_fail": bool(getattr(self.config, "governance_hard_fail", True)),
            "min_r2": float(getattr(self.config, "governance_min_r2", -0.25)),
            "max_normalized_shift": float(
                getattr(self.config, "governance_max_normalized_shift", 2.5)
            ),
            "max_leakage_flags": int(
                getattr(self.config, "governance_max_leakage_flags", 1)
            ),
            "min_stationary_ratio": float(
                getattr(self.config, "governance_min_stationary_ratio", 0.4)
            ),
            "min_walk_forward_r2": float(
                getattr(self.config, "governance_min_walk_forward_r2", -0.25)
            ),
            "max_model_risk_score": float(
                getattr(self.config, "governance_max_model_risk_score", 0.6)
            ),
            "model_risk_warn_threshold": float(
                getattr(self.config, "governance_model_risk_warn_threshold", 0.4)
            ),
            "model_risk_fail_threshold": float(
                getattr(self.config, "governance_model_risk_fail_threshold", 0.6)
            ),
        }

        if ticker:
            overrides_map: Dict[str, Any] = (
                getattr(self.config, "governance_ticker_overrides", {}) or {}
            )
            ticker_overrides: Dict[str, Any] = overrides_map.get(ticker, {})
            _float_keys = {
                "min_r2",
                "max_normalized_shift",
                "min_stationary_ratio",
                "min_walk_forward_r2",
                "max_model_risk_score",
                "model_risk_warn_threshold",
                "model_risk_fail_threshold",
            }
            for k, v in ticker_overrides.items():
                if k in _float_keys and k in profile:
                    profile[k] = float(v)
                elif k == "max_leakage_flags" and k in profile:
                    profile[k] = int(v)
                elif k == "hard_fail" and k in profile:
                    profile[k] = bool(v)

        warn_threshold = float(profile["model_risk_warn_threshold"])
        fail_threshold = float(profile["model_risk_fail_threshold"])
        max_shift = float(profile["max_normalized_shift"])
        max_model_risk = float(profile["max_model_risk_score"])
        min_r2 = float(profile["min_r2"])

        if fail_threshold < warn_threshold:
            (
                warn_threshold,
                fail_threshold,
            ) = (
                fail_threshold,
                warn_threshold,
            )

        if regime == "stress":
            max_shift *= 1.2
            max_model_risk = min(1.0, max_model_risk + 0.05)
        elif regime == "crisis":
            max_shift *= 1.4
            max_model_risk = min(1.0, max_model_risk + 0.1)
            min_r2 -= 0.05

        profile["model_risk_warn_threshold"] = warn_threshold
        profile["model_risk_fail_threshold"] = fail_threshold
        profile["max_normalized_shift"] = max_shift
        profile["max_model_risk_score"] = max_model_risk
        profile["min_r2"] = min_r2

        return profile

    def _export_governance_decision(
        self,
        gate: Dict[str, Any],
        report: Optional[Dict[str, Any]],
        run_mode: str,
    ) -> None:
        from logger.Catalog import catalog

        context = catalog.get_run_context()
        payload: Dict[str, Any] = {
            "schema_version": "governance-decision.v1",
            "generated_at": datetime.now().isoformat(),
            "run_mode": run_mode,
            "run_id": context.get("run_id"),
            "correlation_id": context.get("correlation_id"),
            "gate": gate,
            "report": report,
        }
        decision_file = (
            self.governance_path / f"governance_decision_{int(time.time() * 1000)}.json"
        )
        with decision_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def read_governance_history(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load all governance decision artifacts, sorted oldest-first."""
        files: List[Path] = sorted(
            self.governance_path.glob("governance_decision_*.json")
        )
        if limit is not None:
            files = files[-limit:]
        history: List[Dict[str, Any]] = []
        for f in files:
            try:
                with f.open("r", encoding="utf-8") as fh:
                    history.append(json.load(fh))
            except (json.JSONDecodeError, OSError):
                pass
        return history

    def governance_trend_summary(self, last_n: int = 20) -> Dict[str, Any]:
        """Compute trend statistics over the last_n governance decisions."""
        history = self.read_governance_history(limit=last_n)
        if not history:
            return {"status": "no_history", "count": 0}

        passed_count = sum(
            1 for h in history if h.get("gate", {}).get("passed") is True
        )
        severity_counts: Dict[str, int] = {}
        risk_scores: List[float] = []
        walk_r2s: List[float] = []
        for h in history:
            sev = h.get("gate", {}).get("severity", "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            report = h.get("report") or {}
            score = report.get("model_risk_score")
            if isinstance(score, (float, int)):
                risk_scores.append(float(score))
            wf_r2 = (report.get("walk_forward") or {}).get("avg_r2")
            if isinstance(wf_r2, (float, int)):
                walk_r2s.append(float(wf_r2))

        total = len(history)
        pass_rate = passed_count / total if total else 0.0
        direction = "stable"
        if len(risk_scores) >= 4:
            mid = len(risk_scores) // 2
            first_avg = sum(risk_scores[:mid]) / mid
            second_avg = sum(risk_scores[mid:]) / (len(risk_scores) - mid)
            delta = second_avg - first_avg
            if delta > 0.05:
                direction = "deteriorating"
            elif delta < -0.05:
                direction = "improving"

        return {
            "status": "ok",
            "count": total,
            "pass_rate": round(pass_rate, 4),
            "severity_distribution": severity_counts,
            "avg_model_risk_score": (
                round(sum(risk_scores) / len(risk_scores), 4) if risk_scores else None
            ),
            "worst_walk_forward_r2": (round(min(walk_r2s), 4) if walk_r2s else None),
            "direction": direction,
        }

    def _blocked_results(self, reason: str) -> Dict[str, Any]:
        return {
            "elasticity": reason,
            "lag_analysis": reason,
            "monte_carlo": reason,
            "stress_test": reason,
            "sensitivity_regression": reason,
            "forecasting": reason,
            "auto_ml": reason,
        }

    def run_all_analyses(
        self,
        ticker: Optional[str] = None,
        macro_factor: str = "inflation",
        lags: int = 3,
        shock_map: Optional[Dict[str, float]] = None,
        target: str = "log_return",
        factors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run all analyses and return results in a dictionary.
        """
        results = {}
        selected_ticker = self._resolve_ticker(ticker)
        random_seed = self._resolve_random_seed()
        resolved_factors = factors or ["inflation", "energy_index"]
        try:
            start_time = time.time()
            results["correlation_matrix"] = correl_mtrx(self.df)
            duration = time.time() - start_time
            if results["correlation_matrix"] is not None:
                if isinstance(results["correlation_matrix"], pd.DataFrame):
                    rows = len(results["correlation_matrix"].index)
                    cols = len(results["correlation_matrix"].columns)
                else:
                    rows = 0
                    cols = 0
                catalog.log_analysis_operation(
                    "correlation_matrix",
                    None,
                    {"rows": rows, "columns": cols},
                    duration,
                    True,
                )
                print(ANALYSIS_CORRELATION_MATRIX.format(rows=rows, columns=cols))
        except AnalysisError as e:
            catalog.log_error(
                "gold_layer", "AnalysisError", str(e), "correlation_matrix"
            )
            self.logger.error(f"Analysis error in correlation matrix: {e}")
            results["correlation_matrix"] = None
        except Exception as e:
            catalog.log_error(
                "gold_layer", "UnexpectedError", str(e), "correlation_matrix"
            )
            self.logger.error(f"Unexpected error in correlation matrix: {e}")
            results["correlation_matrix"] = None

        try:
            results["governance_report"] = governance_report(
                self.df,
                target=target,
                factors=resolved_factors,
                random_seed=random_seed,
                reproducibility_enforced=bool(
                    getattr(self.config, "enforce_reproducibility", True)
                ),
                walk_forward_windows=int(
                    getattr(self.config, "governance_walk_forward_windows", 4)
                ),
            )
            gate = self._evaluate_governance_gate(
                results["governance_report"], ticker=selected_ticker
            )
            results["governance_gate"] = gate
            self._export_governance_decision(
                gate, results.get("governance_report"), "sequential"
            )
            if not gate.get("passed", True):
                blocked_reason = f"blocked_by_governance_gate:{gate.get('reasons', [])}"
                results.update(self._blocked_results(blocked_reason))
                return results
        except AnalysisError as e:
            self.logger.error(f"Analysis error in governance report: {e}")
            results["governance_report"] = None
            gate = self._evaluate_governance_gate(None, ticker=selected_ticker)
            results["governance_gate"] = gate
            self._export_governance_decision(gate, None, "sequential")
            if not gate.get("passed", True):
                blocked_reason = f"blocked_by_governance_gate:{gate.get('reasons', [])}"
                results.update(self._blocked_results(blocked_reason))
                return results
        except Exception as e:
            self.logger.error(f"Unexpected error in governance report: {e}")
            results["governance_report"] = None
            gate = self._evaluate_governance_gate(None, ticker=selected_ticker)
            results["governance_gate"] = gate
            self._export_governance_decision(gate, None, "sequential")
            if not gate.get("passed", True):
                blocked_reason = f"blocked_by_governance_gate:{gate.get('reasons', [])}"
                results.update(self._blocked_results(blocked_reason))
                return results

        try:
            if "log_return" in self.df.columns and macro_factor in self.df.columns:
                results["elasticity"] = elasticity(self.df, "log_return", macro_factor)
                if results["elasticity"] is not None:
                    print(
                        ANALYSIS_ELASTICITY.format(
                            elasticity_value=f"{results['elasticity']:.4f}"
                        )
                    )
            else:
                results["elasticity"] = "Required columns not available"
        except AnalysisError as e:
            self.logger.error(f"Analysis error in elasticity: {e}")
            results["elasticity"] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in elasticity: {e}")
            results["elasticity"] = None

        try:
            results["lag_analysis"] = lag_analysis(self.df, macro_factor, lags)
            if results["lag_analysis"] is not None and isinstance(
                results["lag_analysis"], dict
            ):
                best_lag = max(results["lag_analysis"], key=results["lag_analysis"].get)
                print(
                    ANALYSIS_LAG_ANALYSIS.format(
                        factor=macro_factor,
                        best_lag=best_lag,
                        correlation=f"{results['lag_analysis'][best_lag]:.4f}",
                    )
                )
        except AnalysisError as e:
            self.logger.error(f"Analysis error in lag analysis: {e}")
            results["lag_analysis"] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in lag analysis: {e}")
            results["lag_analysis"] = None

        try:
            if selected_ticker:
                results["monte_carlo"] = monte_carlo(
                    self.df, selected_ticker, random_state=random_seed
                )
                if results["monte_carlo"] is not None:
                    print(
                        ANALYSIS_MONTE_CARLO.format(
                            iterations=results["monte_carlo"].shape[1]
                            if hasattr(results["monte_carlo"], "shape")
                            else "N/A",
                            ticker=selected_ticker,
                            days=results["monte_carlo"].shape[0]
                            if hasattr(results["monte_carlo"], "shape")
                            else "N/A",
                            min_price=f"{results['monte_carlo'].min():.2f}"
                            if hasattr(results["monte_carlo"], "min")
                            else "N/A",
                            max_price=f"{results['monte_carlo'].max():.2f}"
                            if hasattr(results["monte_carlo"], "max")
                            else "N/A",
                        )
                    )
            else:
                results["monte_carlo"] = "Ticker not specified"
        except AnalysisError as e:
            self.logger.error(f"Analysis error in monte carlo: {e}")
            results["monte_carlo"] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in monte carlo: {e}")
            results["monte_carlo"] = None

        try:
            if shock_map:
                results["stress_test"] = stress_test(self.df, shock_map)
                if results["stress_test"] is not None:
                    print(
                        ANALYSIS_STRESS_TEST.format(
                            shock_details=str(shock_map),
                            max_drawdown="N/A",  # Could calculate if needed
                        )
                    )
            else:
                results["stress_test"] = "Shock map not provided"
        except AnalysisError as e:
            self.logger.error(f"Analysis error in stress test: {e}")
            results["stress_test"] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in stress test: {e}")
            results["stress_test"] = None

        try:
            results["sensitivity_regression"] = sensitivity_reg(
                self.df, target, factors, "OLS"
            )
            if results["sensitivity_regression"] is not None:
                if isinstance(results["sensitivity_regression"], str):
                    print(
                        ANALYSIS_SENSITIVITY_REGRESSION.format(
                            model_type="OLS",
                            top_factors="N/A",
                            coefficients="N/A",
                            r_squared="N/A",
                        )
                    )
                elif isinstance(results["sensitivity_regression"], dict):
                    print(
                        ANALYSIS_SENSITIVITY_REGRESSION.format(
                            model_type="Ridge",
                            top_factors=str(
                                list(
                                    results["sensitivity_regression"]
                                    .get("coefficients", {})
                                    .keys()
                                )
                            ),
                            coefficients=str(
                                list(
                                    results["sensitivity_regression"]
                                    .get("coefficients", {})
                                    .values()
                                )
                            ),
                            r_squared="N/A",
                        )
                    )
                else:
                    print(
                        ANALYSIS_SENSITIVITY_REGRESSION.format(
                            model_type="Unknown",
                            top_factors="N/A",
                            coefficients="N/A",
                            r_squared="N/A",
                        )
                    )
        except AnalysisError as e:
            self.logger.error(f"Analysis error in sensitivity regression: {e}")
            results["sensitivity_regression"] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in sensitivity regression: {e}")
            results["sensitivity_regression"] = None

        return results

    def run_all_analyses_parallel(
        self,
        ticker: Optional[str] = None,
        macro_factor: str = "inflation",
        lags: int = 3,
        shock_map: Optional[Dict[str, float]] = None,
        target: str = "log_return",
        factors: Optional[List[str]] = None,
        max_workers: int = 4,
        regression_model: str = "OLS",
        include_auto_ml: bool = False,
    ) -> Dict[str, Any]:
        """
        Run independent analyses in parallel.

        Uses a thread pool to avoid process-pickling overhead and to keep
        execution stable across different environments.
        """
        print(LIVE_STEP_7_RESULTS_GENERATION)
        results: Dict[str, Any] = {}
        safe_factors = factors or ["inflation", "energy_index"]
        worker_count = max(1, min(max_workers, 8))
        selected_ticker = self._resolve_ticker(ticker)
        random_seed = self._resolve_random_seed()

        # Run governance gate before advanced analyses.
        try:
            results["governance_report"] = governance_report(
                self.df,
                target,
                safe_factors,
                "date",
                0.2,
                24,
                random_seed,
                bool(getattr(self.config, "enforce_reproducibility", True)),
                int(getattr(self.config, "governance_walk_forward_windows", 4)),
            )
            gate = self._evaluate_governance_gate(
                results["governance_report"], ticker=selected_ticker
            )
            results["governance_gate"] = gate
            self._export_governance_decision(
                gate, results.get("governance_report"), "parallel"
            )
            if not gate.get("passed", True):
                blocked_reason = f"blocked_by_governance_gate:{gate.get('reasons', [])}"
                results.update(self._blocked_results(blocked_reason))
                results["correlation_matrix"] = correl_mtrx(self.df)
                return results
        except Exception:
            gate = self._evaluate_governance_gate(
                results.get("governance_report"), ticker=selected_ticker
            )
            results["governance_gate"] = gate
            self._export_governance_decision(
                gate, results.get("governance_report"), "parallel"
            )
            if not gate.get("passed", True):
                blocked_reason = f"blocked_by_governance_gate:{gate.get('reasons', [])}"
                results.update(self._blocked_results(blocked_reason))
                results["correlation_matrix"] = correl_mtrx(self.df)
                return results

        # Define tasks as partial functions
        tasks: Dict[str, Callable[[], Any]] = {
            "correlation_matrix": partial(correl_mtrx, self.df),
            "lag_analysis": partial(lag_analysis, self.df, macro_factor, lags),
            "sensitivity_regression": partial(
                sensitivity_reg, self.df, target, safe_factors, regression_model
            ),
            "forecasting": partial(
                forecasting, self.df, target, 10
            ),  # Forecast 10 steps for target column
        }

        if include_auto_ml:
            tasks["auto_ml"] = partial(
                auto_ml_regression, self.df, target, safe_factors, random_seed
            )

        if "log_return" in self.df.columns and macro_factor in self.df.columns:
            tasks["elasticity"] = partial(
                elasticity, self.df, "log_return", macro_factor
            )
        else:
            results["elasticity"] = "Required columns not available"

        if selected_ticker:
            tasks["monte_carlo"] = partial(
                monte_carlo, self.df, selected_ticker, 252, 10000, random_seed
            )
        else:
            results["monte_carlo"] = "Ticker not specified"

        if shock_map:
            tasks["stress_test"] = partial(stress_test, self.df, shock_map)
        else:
            results["stress_test"] = "Shock map not provided"

        # Run parallel tasks
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:
            future_to_key: Dict[concurrent.futures.Future[Any], str] = {
                executor.submit(task): key for key, task in tasks.items()
            }
            submitted_futures = list(future_to_key.keys())
            try:
                iterator = concurrent.futures.as_completed(submitted_futures)
                for future in iterator:
                    key = future_to_key[future]
                    try:
                        results[key] = future.result()
                    except AnalysisError as e:
                        self.logger.error(f"Analysis error in {key}: {e}")
                        results[key] = None
                    except Exception as e:
                        self.logger.error(f"Unexpected error in {key}: {e}")
                        results[key] = None
            except AttributeError:
                # Test doubles may not implement internal Future synchronization fields.
                for future in submitted_futures:
                    key = future_to_key[future]
                    try:
                        results[key] = future.result()
                    except AnalysisError as e:
                        self.logger.error(f"Analysis error in {key}: {e}")
                        results[key] = None
                    except Exception as e:
                        self.logger.error(f"Unexpected error in {key}: {e}")
                        results[key] = None

        return results
