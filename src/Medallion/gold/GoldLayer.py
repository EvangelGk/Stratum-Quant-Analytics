import concurrent.futures
import logging
import time
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
from .AnalysisSuite.sesnsitivity_reg import sensitivity_reg
from .AnalysisSuite.stress_test import stress_test


class GoldLayer:
    """
    The Crown Jewel of the Pipeline.
    Responsibility: Feature Engineering & Unified Analytical View.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.processed_path = Path("./data/processed")
        self.gold_path = Path("./data/gold")
        self.gold_path.mkdir(parents=True, exist_ok=True)
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
        dfs = [pd.read_parquet(f) for f in financial_files]
        master_df = pd.concat(dfs, ignore_index=True)

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
            macro_df = pd.read_parquet(f).rename(
                columns={"value": col_name}
            )
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
            wb_df = pd.read_parquet(f).rename(
                columns={"value": col_name}
            )
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
        self, report: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        hard_fail = bool(getattr(self.config, "governance_hard_fail", False))
        gate_reasons: List[str] = []
        gate: Dict[str, Any] = {
            "hard_fail": hard_fail,
            "passed": True,
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
        min_r2 = float(getattr(self.config, "governance_min_r2", -0.25))
        if isinstance(oos_r2, (float, int)) and float(oos_r2) < min_r2:
            reasons.append(f"out_of_sample_r2_below_threshold:{oos_r2:.4f}<{min_r2:.4f}")

        normalized_shift = report.get("stability", {}).get("normalized_mean_shift")
        max_shift = float(getattr(self.config, "governance_max_normalized_shift", 2.5))
        if (
            isinstance(normalized_shift, (float, int))
            and float(normalized_shift) > max_shift
        ):
            reasons.append(
                "normalized_mean_shift_above_threshold:"
                f"{float(normalized_shift):.4f}>{max_shift:.4f}"
            )

        leakage_flags = report.get("leakage_flags", []) or []
        max_leakage = int(getattr(self.config, "governance_max_leakage_flags", 1))
        if len(leakage_flags) > max_leakage:
            reasons.append(
                "leakage_flags_above_threshold:"
                f"{len(leakage_flags)}>{max_leakage}"
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
            min_ratio = float(
                getattr(self.config, "governance_min_stationary_ratio", 0.4)
            )
            if ratio < min_ratio:
                reasons.append(
                    "stationarity_ratio_below_threshold:"
                    f"{ratio:.4f}<{min_ratio:.4f}"
                )

        walk_forward = report.get("walk_forward", {}) or {}
        min_walk_forward_r2 = float(
            getattr(self.config, "governance_min_walk_forward_r2", -0.25)
        )
        walk_forward_avg_r2 = walk_forward.get("avg_r2")
        if (
            isinstance(walk_forward_avg_r2, (float, int))
            and float(walk_forward_avg_r2) < min_walk_forward_r2
        ):
            reasons.append(
                "walk_forward_avg_r2_below_threshold:"
                f"{float(walk_forward_avg_r2):.4f}<{min_walk_forward_r2:.4f}"
            )

        model_risk_score = report.get("model_risk_score")
        max_model_risk = float(
            getattr(self.config, "governance_max_model_risk_score", 0.6)
        )
        if (
            isinstance(model_risk_score, (float, int))
            and float(model_risk_score) > max_model_risk
        ):
            reasons.append(
                "model_risk_score_above_threshold:"
                f"{float(model_risk_score):.4f}>{max_model_risk:.4f}"
            )

        if hard_fail and reasons:
            gate["passed"] = False
        gate_reasons.extend(reasons)
        return gate

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
            gate = self._evaluate_governance_gate(results["governance_report"])
            results["governance_gate"] = gate
            if not gate.get("passed", True):
                blocked_reason = f"blocked_by_governance_gate:{gate.get('reasons', [])}"
                results.update(self._blocked_results(blocked_reason))
                return results
        except AnalysisError as e:
            self.logger.error(f"Analysis error in governance report: {e}")
            results["governance_report"] = None
            gate = self._evaluate_governance_gate(None)
            results["governance_gate"] = gate
            if not gate.get("passed", True):
                blocked_reason = (
                    "blocked_by_governance_gate:"
                    f"{gate.get('reasons', [])}"
                )
                results.update(self._blocked_results(blocked_reason))
                return results
        except Exception as e:
            self.logger.error(f"Unexpected error in governance report: {e}")
            results["governance_report"] = None
            gate = self._evaluate_governance_gate(None)
            results["governance_gate"] = gate
            if not gate.get("passed", True):
                blocked_reason = (
                    "blocked_by_governance_gate:"
                    f"{gate.get('reasons', [])}"
                )
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
                best_lag = max(results["lag_analysis"], key=results["lag_analysis"]
                .get)
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
            gate = self._evaluate_governance_gate(results["governance_report"])
            results["governance_gate"] = gate
            if not gate.get("passed", True):
                blocked_reason = f"blocked_by_governance_gate:{gate.get('reasons', [])}"
                results.update(self._blocked_results(blocked_reason))
                results["correlation_matrix"] = correl_mtrx(self.df)
                return results
        except Exception:
            gate = self._evaluate_governance_gate(results.get("governance_report"))
            results["governance_gate"] = gate
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
