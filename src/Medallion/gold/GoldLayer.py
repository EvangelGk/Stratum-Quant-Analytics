import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict
import concurrent.futures
from functools import partial
import pyarrow as pa
import pyarrow.parquet as pq

from .AnalysisSuite.elasticity import elasticity
from .AnalysisSuite.lag import lag_analysis
from .AnalysisSuite.correl_mtrx import correl_mtrx
from .AnalysisSuite.monte_carlo import monte_carlo
from .AnalysisSuite.stress_test import stress_test
from .AnalysisSuite.sesnsitivity_reg import sensitivity_reg
from .AnalysisSuite.forecasting import forecasting
from .AnalysisSuite.auto_ml import auto_ml_regression
from exceptions.MedallionExceptions import AnalysisError, ParallelExecutionError
from logger.Messages.MainMess import (
    ANALYSIS_CORRELATION_MATRIX, ANALYSIS_ELASTICITY, ANALYSIS_LAG_ANALYSIS,
    ANALYSIS_MONTE_CARLO, ANALYSIS_STRESS_TEST, ANALYSIS_SENSITIVITY_REGRESSION,
    ANALYSIS_FORECASTING, ANALYSIS_AUTO_ML
)
from logger.Messages.DirectionsMess import (
    LIVE_STEP_6_ANALYSIS_PROGRESS, LIVE_STEP_7_RESULTS_GENERATION
)
from logger.Catalog import catalog
import time

class GoldLayer:
    """
    The Crown Jewel of the Pipeline. 
    Responsibility: Feature Engineering & Unified Analytical View.
    """
    def __init__(self, config):
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
        master_df['log_return'] = master_df.groupby('ticker')['close'].transform(
            lambda x: np.log(x / x.shift(1))
        )

        # 3. Join Macro Data (FRED)
        fred_files = list((self.processed_path / "fred").glob("*.parquet"))
        for f in fred_files:
            macro_df = pd.read_parquet(f).rename(columns={'value': f.stem.replace('_silver', '')})
            master_df = pd.merge(master_df, macro_df[['date', f.stem.replace('_silver', '')]], on='date', how='left')

        # 4. Join World Bank Data
        wb_files = list((self.processed_path / "worldbank").glob("*.parquet"))
        for f in wb_files:
            wb_df = pd.read_parquet(f).rename(columns={'value': f.stem.replace('_silver', '')})
            master_df = pd.merge(master_df, wb_df[['date', f.stem.replace('_silver', '')]], on='date', how='left')

        # 5. Forward-Fill Macro and World Bank (γιατί τα δεδομένα δεν αλλάζουν καθημερινά)
        master_df = master_df.sort_values(['ticker', 'date']).ffill()
        
        # Save the "Analytical Base Table" with optional encryption
        table = pa.Table.from_pandas(master_df)
        pq.write_table(table, self.gold_path / "master_table.parquet", compression='zstd')
        return master_df

    def run_all_analyses(self, ticker: str = None, macro_factor: str = 'inflation', lags: int = 3, shock_map: Dict[str, float] = None, target: str = 'log_return', factors: List[str] = None):
        """
        Run all analyses and return results in a dictionary.
        """
        results = {}
        try:
            start_time = time.time()
            results['correlation_matrix'] = correl_mtrx(self.df)
            duration = time.time() - start_time
            if results['correlation_matrix'] is not None:
                rows = len(results['correlation_matrix'].index) if hasattr(results['correlation_matrix'], 'index') else 0
                cols = len(results['correlation_matrix'].columns) if hasattr(results['correlation_matrix'], 'columns') else 0
                catalog.log_analysis_operation("correlation_matrix", None,
                                             {"rows": rows, "columns": cols}, duration, True)
                print(ANALYSIS_CORRELATION_MATRIX.format(rows=rows, columns=cols))
        except AnalysisError as e:
            catalog.log_error("gold_layer", "AnalysisError", str(e), "correlation_matrix")
            self.logger.error(f"Analysis error in correlation matrix: {e}")
            results['correlation_matrix'] = None
        except Exception as e:
            catalog.log_error("gold_layer", "UnexpectedError", str(e), "correlation_matrix")
            self.logger.error(f"Unexpected error in correlation matrix: {e}")
            results['correlation_matrix'] = None

        try:
            if ticker:
                results['elasticity'] = elasticity(self.df, 'log_return', macro_factor)
                if results['elasticity'] is not None:
                    print(ANALYSIS_ELASTICITY.format(elasticity_value=f"{results['elasticity']:.4f}"))
            else:
                results['elasticity'] = "Ticker not specified"
        except AnalysisError as e:
            self.logger.error(f"Analysis error in elasticity: {e}")
            results['elasticity'] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in elasticity: {e}")
            results['elasticity'] = None

        try:
            results['lag_analysis'] = lag_analysis(self.df, macro_factor, lags)
            if results['lag_analysis'] is not None and isinstance(results['lag_analysis'], dict):
                best_lag = max(results['lag_analysis'], key=results['lag_analysis'].get)
                print(ANALYSIS_LAG_ANALYSIS.format(
                    factor=macro_factor,
                    best_lag=best_lag,
                    correlation=f"{results['lag_analysis'][best_lag]:.4f}"
                ))
        except AnalysisError as e:
            self.logger.error(f"Analysis error in lag analysis: {e}")
            results['lag_analysis'] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in lag analysis: {e}")
            results['lag_analysis'] = None

        try:
            if ticker:
                results['monte_carlo'] = monte_carlo(self.df, ticker)
                if results['monte_carlo'] is not None:
                    print(ANALYSIS_MONTE_CARLO.format(
                        iterations=results['monte_carlo'].shape[1] if hasattr(results['monte_carlo'], 'shape') else 'N/A',
                        ticker=ticker,
                        days=results['monte_carlo'].shape[0] if hasattr(results['monte_carlo'], 'shape') else 'N/A',
                        min_price=f"{results['monte_carlo'].min():.2f}" if hasattr(results['monte_carlo'], 'min') else 'N/A',
                        max_price=f"{results['monte_carlo'].max():.2f}" if hasattr(results['monte_carlo'], 'max') else 'N/A'
                    ))
            else:
                results['monte_carlo'] = "Ticker not specified"
        except AnalysisError as e:
            self.logger.error(f"Analysis error in monte carlo: {e}")
            results['monte_carlo'] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in monte carlo: {e}")
            results['monte_carlo'] = None

        try:
            if shock_map:
                results['stress_test'] = stress_test(self.df, shock_map)
                if results['stress_test'] is not None:
                    print(ANALYSIS_STRESS_TEST.format(
                        shock_details=str(shock_map),
                        max_drawdown="N/A"  # Could calculate if needed
                    ))
            else:
                results['stress_test'] = "Shock map not provided"
        except AnalysisError as e:
            self.logger.error(f"Analysis error in stress test: {e}")
            results['stress_test'] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in stress test: {e}")
            results['stress_test'] = None

        try:
            results['sensitivity_regression'] = sensitivity_reg(self.df, target, factors)
            if results['sensitivity_regression'] is not None:
                if isinstance(results['sensitivity_regression'], str):
                    print(ANALYSIS_SENSITIVITY_REGRESSION.format(
                        model_type="OLS",
                        top_factors="N/A",
                        coefficients="N/A",
                        r_squared="N/A"
                    ))
                elif isinstance(results['sensitivity_regression'], dict):
                    print(ANALYSIS_SENSITIVITY_REGRESSION.format(
                        model_type="Ridge",
                        top_factors=str(list(results['sensitivity_regression'].get('coefficients', {}).keys())),
                        coefficients=str(list(results['sensitivity_regression'].get('coefficients', {}).values())),
                        r_squared="N/A"
                    ))
                else:
                    print(ANALYSIS_SENSITIVITY_REGRESSION.format(
                        model_type="Unknown",
                        top_factors="N/A",
                        coefficients="N/A",
                        r_squared="N/A"
                    ))
        except AnalysisError as e:
            self.logger.error(f"Analysis error in sensitivity regression: {e}")
            results['sensitivity_regression'] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in sensitivity regression: {e}")
            results['sensitivity_regression'] = None

        return results

    def run_all_analyses_parallel(self, ticker: str = None, macro_factor: str = 'inflation', lags: int = 3, shock_map: Dict[str, float] = None, target: str = 'log_return', factors: List[str] = None, max_workers: int = 4, regression_model: str = 'OLS'):
        """
        Run all analyses in parallel using multiprocessing for better performance.
        """
        print(LIVE_STEP_7_RESULTS_GENERATION)
        results = {}
        
        # Define tasks as partial functions
        tasks = {
            'correlation_matrix': partial(correl_mtrx, self.df),
            'lag_analysis': partial(lag_analysis, self.df, macro_factor, lags),
            'sensitivity_regression': partial(sensitivity_reg, self.df, target, factors, regression_model),
            'forecasting': partial(forecasting, self.df, target, 10),  # Forecast 10 steps for target column
            'auto_ml': partial(auto_ml_regression, self.df, target, factors),
        }
        
        if ticker:
            tasks['elasticity'] = partial(elasticity, self.df, 'log_return', macro_factor)
            tasks['monte_carlo'] = partial(monte_carlo, self.df, ticker)
        else:
            results['elasticity'] = "Ticker not specified"
            results['monte_carlo'] = "Ticker not specified"
        
        if shock_map:
            tasks['stress_test'] = partial(stress_test, self.df, shock_map)
        else:
            results['stress_test'] = "Shock map not provided"
        
        # Run parallel tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {executor.submit(task): key for key, task in tasks.items()}
            for future in concurrent.futures.as_completed(future_to_key):
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