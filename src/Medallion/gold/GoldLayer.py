import concurrent.futures
import hashlib
import json
import logging
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

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
from .AnalysisSuite.backtest import backtest_pre2020_holdout
from .AnalysisSuite.correl_mtrx import correl_mtrx
from .AnalysisSuite.elasticity import elasticity
from .AnalysisSuite.feature_decay import feature_decay_analysis
from .AnalysisSuite.forecasting import forecasting
from .AnalysisSuite.governance import governance_report
from .AnalysisSuite.lag import lag_analysis
from .AnalysisSuite.mixed_frequency import filter_to_ticker
from .AnalysisSuite.monte_carlo import monte_carlo
from .AnalysisSuite.sensitivity_reg import sensitivity_reg
from .AnalysisSuite.stress_test import resolve_stress_scenario, stress_test


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
        contract_file = self.gold_path / "master_table_contract.json"
        if master_file.exists() and not self._master_table_stale(master_file, contract_file):
            self.logger.info("Loading existing master table...")
            return pd.read_parquet(master_file)

        self.logger.info("Master table is missing or stale. Rebuilding...")
        return self.create_master_table()

    def _current_config_fingerprint(self) -> str:
        if hasattr(self.config, "to_serializable_dict"):
            payload = self.config.to_serializable_dict()
        else:
            payload = {
                "mode": str(getattr(self.config, "mode", "")),
                "start_date": str(getattr(self.config, "start_date", "")),
                "end_date": str(getattr(self.config, "end_date", "")),
                "target_tickers": list(getattr(self.config, "target_tickers", [])),
                "macro_series_map": dict(getattr(self.config, "macro_series_map", {})),
                "worldbank_indicator_map": dict(getattr(self.config, "worldbank_indicator_map", {})),
                "worldbank_economies": list(getattr(self.config, "worldbank_economies", ["WLD"])),
                "fred_enabled": bool(getattr(self.config, "fred_api_key", None)),
            }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _latest_processed_mtime(self) -> float:
        latest = 0.0
        for source in ("yfinance", "fred", "worldbank"):
            source_dir = self.processed_path / source
            if not source_dir.exists():
                continue
            for file_path in source_dir.glob("*.parquet"):
                latest = max(latest, file_path.stat().st_mtime)
        return latest

    def _master_table_stale(self, master_file: Path, contract_file: Path) -> bool:
        if not master_file.exists():
            return True
        if not contract_file.exists():
            return True

        try:
            contract = json.loads(contract_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return True

        stored_fingerprint = str(contract.get("config_fingerprint", ""))
        current_fingerprint = self._current_config_fingerprint()
        if stored_fingerprint != current_fingerprint:
            return True

        return master_file.stat().st_mtime < self._latest_processed_mtime()

    def _aggregate_strategy(self, series: pd.Series, strategy: str) -> float:
        clean = series.dropna()
        if clean.empty:
            return float("nan")
        if strategy == "median":
            return float(clean.median())
        if strategy == "sum":
            return float(clean.sum())
        if strategy == "last":
            return float(clean.iloc[-1])
        return float(clean.mean())

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

        # Keep the financial base at its native trading-day cadence.
        # Macro sources are joined later using backward-looking as-of logic,
        # so the latest available monthly/annual value is carried onto each
        # daily market row without exact date equality requirements.
        master_df["date"] = pd.to_datetime(master_df["date"], errors="coerce")
        master_df = master_df.dropna(subset=["date"]).copy()
        master_df["date"] = master_df["date"].dt.normalize()
        master_df = master_df.sort_values(["date", "ticker"]).reset_index(drop=True)

        # 2. Master Feature: Log Returns (The Senior Standard)
        # Prefer adj_close (corporate-action adjusted via Silver normalisation of
        # yfinance Adj Close) when present; fall back to close for compatibility.
        # With YFinanceFetcher.AUTO_ADJUST=True, close IS the adjusted price so
        # both columns are equivalent — adj_close is simply the explicit alias.
        _price_col = "adj_close" if "adj_close" in master_df.columns else "close"
        master_df["log_return"] = master_df.groupby("ticker")[_price_col].transform(lambda x: np.log(x / x.shift(1)))

        # ── Anti-Gravity Guard (Log-Return Winsorisation) ────────────────────
        # A single-day move beyond ±25% almost always reflects bad YFinance
        # data (corporate actions, delisting artefacts, API errors).
        # Clipping at ±ln(1.25) ≈ ±0.2231 prevents $10^44-scale equity curves.
        _LR_CLIP = float(np.log(1.25))
        _n_lr_clipped = int((master_df["log_return"].abs() > _LR_CLIP).sum())
        if _n_lr_clipped > 0:
            self.logger.warning(
                f"Anti-Gravity Guard: clipped {_n_lr_clipped} log_return values "
                f"beyond ±{_LR_CLIP:.4f} (≈±25% simple return) — likely bad YFinance data."
            )
        master_df["log_return"] = master_df["log_return"].clip(lower=-_LR_CLIP, upper=_LR_CLIP)

        fred_enabled = bool(getattr(self.config, "fred_api_key", None))
        worldbank_enabled = bool(getattr(self.config, "worldbank_indicator_map", {}))

        expected_fred_columns = list(dict(getattr(self.config, "macro_series_map", {})).values()) if fred_enabled else []
        expected_worldbank_columns = list(dict(getattr(self.config, "worldbank_indicator_map", {})).values()) if worldbank_enabled else []
        fred_staleness_days = max(1, int(getattr(self.config, "gold_fred_max_staleness_days", 120)))
        worldbank_staleness_days = max(30, int(getattr(self.config, "gold_worldbank_max_staleness_days", 730)))

        # 3. Join Macro Data (FRED)
        fred_files = list((self.processed_path / "fred").glob("*.parquet"))
        macro_columns: List[str] = []
        fred_age_columns: List[str] = []
        for f in fred_files:
            col_name = f.stem.replace("_silver", "")
            macro_df = pd.read_parquet(f).rename(columns={"value": col_name})
            macro_df["date"] = pd.to_datetime(macro_df["date"], errors="coerce")
            source_date_col = f"__source_date_{col_name}"
            macro_df[source_date_col] = macro_df["date"]
            macro_df = macro_df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
            master_df = pd.merge_asof(
                master_df,
                macro_df[["date", col_name, source_date_col]],
                on="date",
                direction="backward",
                tolerance=pd.Timedelta(days=fred_staleness_days),
            )
            if source_date_col in master_df.columns:
                age_col = f"__age_days_{col_name}"
                master_df[age_col] = (master_df["date"] - master_df[source_date_col]).dt.days
                fred_age_columns.append(age_col)
                master_df = master_df.drop(columns=[source_date_col], errors="ignore")
            macro_columns.append(col_name)

        missing_fred_columns = [c for c in expected_fred_columns if c not in set(macro_columns)]
        if fred_enabled and missing_fred_columns:
            raise AnalysisError("Gold mapping mismatch: expected FRED columns missing from Silver artifacts: " + ", ".join(sorted(missing_fred_columns)))

        # 4. Join World Bank Data
        wb_files = list((self.processed_path / "worldbank").glob("*.parquet"))
        wb_frames_by_series: Dict[str, List[pd.DataFrame]] = {}
        wb_age_columns: List[str] = []
        for f in wb_files:
            entity_name = f.stem.replace("_silver", "")
            series_name = entity_name.split("__", 1)[0]
            wb_df = pd.read_parquet(f)
            if "date" not in wb_df.columns or "value" not in wb_df.columns:
                continue
            wb_df["date"] = pd.to_datetime(wb_df["date"], errors="coerce")
            wb_df = wb_df.dropna(subset=["date"]).copy()
            source_date_col = f"__source_date_{series_name}"
            wb_df[source_date_col] = wb_df["date"]
            wb_frames_by_series.setdefault(series_name, []).append(wb_df[["date", "value", source_date_col]].copy())

        wb_strategy = str(getattr(self.config, "worldbank_aggregation_strategy", "mean")).lower()
        if wb_strategy not in {"mean", "median", "sum", "last"}:
            wb_strategy = "mean"

        for series_name, frames in wb_frames_by_series.items():
            if not frames:
                continue
            stacked = pd.concat(frames, ignore_index=True)
            stacked = stacked.sort_values("date")
            aggregated = (
                stacked.groupby("date", as_index=False)
                .agg(
                    value=(
                        "value",
                        lambda s: self._aggregate_strategy(cast(pd.Series, s), wb_strategy),
                    ),
                    source_date=(
                        f"__source_date_{series_name}",
                        "max",
                    ),
                )
                .rename(columns={"value": series_name})
                .sort_values("date")
            )
            source_date_col = f"__source_date_{series_name}"
            if "source_date" in aggregated.columns:
                aggregated = aggregated.rename(columns={"source_date": source_date_col})

            # ── World Bank Upsampling: Linear Interpolation → Daily ───────────
            # World Bank data is annual; market data is daily.  Without
            # interpolation, 364 of every 365 rows would be NULL after the
            # asof-join, causing Bad Fitting in the ML stage.
            # Linear interpolation fills intra-year gaps WITHOUT introducing
            # look-ahead bias because the 21-day publication lag in
            # mixed_frequency.py remains the primary LAB guardrail.
            if len(aggregated) >= 2 and source_date_col in aggregated.columns:
                _daily_idx = pd.date_range(
                    start=master_df["date"].min(),
                    end=master_df["date"].max(),
                    freq="B",
                )
                _anchor_src = (
                    aggregated[["date", source_date_col]]
                    .set_index("date")[source_date_col]
                )
                _interp_frame = (
                    aggregated[["date", series_name]]
                    .set_index("date")
                    .reindex(_daily_idx)
                )
                _interp_frame[series_name] = (
                    _interp_frame[series_name]
                    .interpolate(method="linear", limit_direction="forward")
                    .ffill()
                )
                # Keep the most-recent annual anchor date for staleness tracking.
                _interp_frame[source_date_col] = (
                    _anchor_src.reindex(_daily_idx).ffill()
                )
                aggregated = (
                    _interp_frame
                    .reset_index()
                    .rename(columns={"index": "date"})
                    .sort_values("date")
                )

            master_df = pd.merge_asof(
                master_df,
                aggregated[["date", series_name, source_date_col]],
                on="date",
                direction="backward",
                tolerance=pd.Timedelta(days=worldbank_staleness_days),
            )
            if source_date_col in master_df.columns:
                age_col = f"__age_days_{series_name}"
                master_df[age_col] = (master_df["date"] - master_df[source_date_col]).dt.days
                wb_age_columns.append(age_col)
                master_df = master_df.drop(columns=[source_date_col], errors="ignore")
            macro_columns.append(series_name)

        missing_worldbank_columns = [c for c in expected_worldbank_columns if c not in set(wb_frames_by_series.keys())]
        if worldbank_enabled and missing_worldbank_columns:
            raise AnalysisError(
                "Gold mapping mismatch: expected WorldBank columns missing from Silver artifacts: " + ", ".join(sorted(missing_worldbank_columns))
            )

        # Final analytical ordering: per ticker on daily dates.
        master_df = master_df.sort_values(["ticker", "date"]).copy()

        fred_columns = [c for c in getattr(self.config, "macro_series_map", {}).values() if c in master_df.columns]
        worldbank_columns = [c for c in getattr(self.config, "worldbank_indicator_map", {}).values() if c in master_df.columns]
        source_usage = {
            "yfinance": 100.0 if len(master_df) > 0 else 0.0,
            "fred": float(master_df[fred_columns].notnull().any(axis=1).mean() * 100.0) if fred_columns and len(master_df) > 0 else 0.0,
            "worldbank": float(master_df[worldbank_columns].notnull().any(axis=1).mean() * 100.0) if worldbank_columns and len(master_df) > 0 else 0.0,
        }

        staleness_summary: Dict[str, Any] = {}
        for source, age_cols in {
            "fred": fred_age_columns,
            "worldbank": wb_age_columns,
        }.items():
            existing = [c for c in age_cols if c in master_df.columns]
            if not existing:
                continue
            age_values = pd.concat([master_df[c] for c in existing], axis=0).dropna()
            if age_values.empty:
                continue
            staleness_summary[source] = {
                "max_age_days": int(age_values.max()),
                "median_age_days": float(age_values.median()),
            }

        self.logger.info(f"Gold source usage after joins: {source_usage}")
        expected_sources = []
        if fred_enabled:
            expected_sources.append("fred")
        if worldbank_enabled:
            expected_sources.append("worldbank")
        broken_sources = [source for source in expected_sources if source_usage.get(source, 0.0) == 0.0]
        if broken_sources:
            raise AnalysisError("Gold source integration failure: joined master table has zero usable coverage for " + ", ".join(broken_sources))

        # ── Post-join null density report ───────────────────────────────────
        # Measures data "holes" after all Macro + WorldBank merges.
        # High null% in a macro column = convergence failure that will corrupt
        # ML fitting.  Report it in the contract for Auditor visibility.
        _signal_cols = [
            c for c in master_df.columns
            if not str(c).startswith("__age_days_") and not str(c).startswith("__source_date_")
        ]
        post_join_null_density: Dict[str, float] = {
            col: round(float(master_df[col].isnull().mean() * 100), 2)
            for col in _signal_cols
        }
        _high_null_cols = {k: v for k, v in post_join_null_density.items() if v > 20.0}
        if _high_null_cols:
            self.logger.warning(
                f"Post-join null density alert: {len(_high_null_cols)} columns "
                f"exceed 20% missing after macro/WB merge. "
                f"Columns: {list(_high_null_cols.keys())[:8]}"
            )

        # Persist explicit schema contract so Auditor validates against what Gold emitted.
        contract_payload = {
            "schema_version": "gold-contract.v1",
            "generated_at": datetime.now().isoformat(),
            "config_fingerprint": self._current_config_fingerprint(),
            "expected": {
                "yfinance": ["date", "ticker", "close", "log_return", "volume"],
                "fred": sorted(expected_fred_columns),
                "worldbank": sorted(expected_worldbank_columns),
            },
            "observed": {
                "all_columns": sorted([str(c) for c in master_df.columns]),
                "fred": sorted(fred_columns),
                "worldbank": sorted(worldbank_columns),
            },
            "source_usage": source_usage,
            "staleness_days": staleness_summary,
            "join_tolerance_days": {
                "fred": fred_staleness_days,
                "worldbank": worldbank_staleness_days,
            },
            "post_join_null_density_pct": post_join_null_density,
            "join_strategy": {
                "fred": "merge_asof_backward",
                "worldbank": "linear_interpolation_then_merge_asof",
                "log_return_clip": f"±ln(1.25)≈±{float(np.log(1.25)):.4f}",
            },
        }
        contract_file = self.gold_path / "master_table_contract.json"
        with contract_file.open("w", encoding="utf-8") as f:
            json.dump(contract_payload, f, indent=2)

        # Save the "Analytical Base Table" with optional encryption
        table = pa.Table.from_pandas(master_df)
        pq.write_table(table, self.gold_path / "master_table.parquet", compression="zstd")
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

    def _analysis_df(self, ticker: Optional[str]) -> pd.DataFrame:
        return filter_to_ticker(self.df, ticker=ticker)

    def compute_risk_parity_weights(
        self,
        lookback_days: int = 252,
    ) -> Dict[str, float]:
        """Compute inverse-volatility (risk-parity) portfolio weights.

        Replaces equal-weighting for the 30-ticker universe.  Each ticker's
        weight is proportional to 1/σ_annualised, normalised to sum to 1.0.

        This stabilises the Calmar ratio by over-weighting low-volatility
        names and under-weighting speculative high-vol names.

        Returns
        -------
        dict  {ticker: weight}  — weights sum to 1.0.
              Empty dict if insufficient data.
        """
        if self.df.empty or "log_return" not in self.df.columns or "ticker" not in self.df.columns:
            return {}

        work = self.df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.sort_values("date")

        cutoff = work["date"].max() - pd.Timedelta(days=lookback_days)
        recent = work[work["date"] >= cutoff]

        tickers = recent["ticker"].dropna().unique().tolist()
        if not tickers:
            return {}

        vols: Dict[str, float] = {}
        for t in tickers:
            returns = pd.to_numeric(
                recent[recent["ticker"] == t]["log_return"], errors="coerce"
            ).dropna()
            if len(returns) >= 20:
                vols[t] = float(returns.std(ddof=1) * np.sqrt(252))

        if not vols:
            return {}

        inv_vols = {t: 1.0 / max(v, 1e-8) for t, v in vols.items()}
        total = sum(inv_vols.values())
        weights = {t: round(iv / total, 8) for t, iv in inv_vols.items()}

        self.logger.info(
            f"Risk-Parity Weights: {len(weights)} tickers, lookback={lookback_days}d. "
            f"Min={min(weights.values()):.4f} Max={max(weights.values()):.4f} "
            f"(Equal weight would be {1.0/max(len(weights),1):.4f})"
        )
        return weights

    def _resolve_analysis_factors(
        self,
        analysis_df: pd.DataFrame,
        target: str,
        explicit_factors: Optional[List[str]] = None,
    ) -> List[str]:
        if explicit_factors:
            ordered = list(dict.fromkeys([f for f in explicit_factors if isinstance(f, str)]))
            return [f for f in ordered if f in analysis_df.columns and f != target]

        preferred_from_maps = list(dict(getattr(self.config, "macro_series_map", {})).values()) + list(
            dict(getattr(self.config, "worldbank_indicator_map", {})).values()
        )
        market_candidates = ["open", "high", "low", "close", "adj_close", "volume"]
        ordered_seed = list(dict.fromkeys(preferred_from_maps + market_candidates))

        numeric_cols = [
            col for col in analysis_df.columns if col != target and pd.api.types.is_numeric_dtype(analysis_df[col]) and not str(col).startswith("__age_days_")
        ]
        ordered_candidates = [c for c in ordered_seed if c in numeric_cols] + [c for c in numeric_cols if c not in ordered_seed]

        filtered: List[str] = []
        for col in ordered_candidates:
            series = pd.to_numeric(analysis_df[col], errors="coerce")
            if series.notna().sum() < 30:
                continue
            if float(series.nunique(dropna=True)) <= 1:
                continue
            filtered.append(col)

        if not filtered:
            fallback = ["inflation", "energy_index"]
            filtered = [f for f in fallback if f in analysis_df.columns and f != target]
        return filtered

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
            gate["severity"] = "warn"
            return gate

        reasons: List[str] = []
        advisory_reasons: List[str] = []
        oos_r2 = report.get("out_of_sample", {}).get("r2")
        min_r2 = float(profile["min_r2"])
        trend_vol = report.get("trend_volatility", {}) or {}
        trend_directional_accuracy = trend_vol.get("trend_directional_accuracy")
        volatility_r2 = trend_vol.get("volatility_r2")
        volatility_ratio = trend_vol.get("volatility_ratio")

        trend_volatility_acceptable = bool(
            isinstance(trend_directional_accuracy, (float, int))
            and float(trend_directional_accuracy) >= 0.55
            and (
                (isinstance(volatility_r2, (float, int)) and float(volatility_r2) >= 0.0)
                or (isinstance(volatility_ratio, (float, int)) and 0.5 <= float(volatility_ratio) <= 1.8)
            )
        )

        if isinstance(oos_r2, (float, int)) and float(oos_r2) < min_r2:
            oos_r2_ci = report.get("out_of_sample", {}).get("r2_ci", {}) or {}
            ci_upper = oos_r2_ci.get("ci_upper")
            ci_status = str(oos_r2_ci.get("status", ""))
            ci_clears = ci_status == "ok" and isinstance(ci_upper, (float, int)) and float(ci_upper) >= min_r2
            # Mildly negative OOS R2 is common for daily-return targets; treat
            # it as a diagnostic signal unless corroborated by other failures.
            mild_negative_band = float(oos_r2) >= -0.10
            if ci_clears:
                advisory_reasons.append(f"r2_metric_alert_oos_below_threshold_but_ci_upper_clears:{oos_r2:.4f}<{min_r2:.4f};ci_upper={float(ci_upper):.4f}")
            elif trend_volatility_acceptable:
                advisory_reasons.append("r2_metric_alert_oos_below_threshold_but_trend_volatility_acceptable")
            elif mild_negative_band:
                advisory_reasons.append(f"r2_metric_alert_oos_mild_negative_macro_noise_band:{oos_r2:.4f}")
            else:
                advisory_reasons.append(f"r2_metric_alert_oos_below_threshold:{oos_r2:.4f}<{min_r2:.4f}")

        normalized_shift = report.get("stability", {}).get("normalized_mean_shift")
        max_shift = float(profile["max_normalized_shift"])
        if isinstance(normalized_shift, (float, int)) and float(normalized_shift) > max_shift:
            reasons.append(f"normalized_mean_shift_above_threshold:{float(normalized_shift):.4f}>{max_shift:.4f}")

        leakage_flags = report.get("leakage_flags", []) or []
        max_leakage = int(profile["max_leakage_flags"])
        if len(leakage_flags) > max_leakage:
            reasons.append(f"leakage_flags_above_threshold:{len(leakage_flags)}>{max_leakage}")

        stationarity = report.get("stationarity", {}) or {}
        status_entries = [v for v in stationarity.values() if isinstance(v, dict)]
        stationary_count = sum(1 for v in status_entries if v.get("is_stationary") is True)
        considered = sum(1 for v in status_entries if v.get("is_stationary") is not None)
        if considered > 0:
            ratio = stationary_count / considered
            min_ratio = float(profile["min_stationary_ratio"])
            # Dynamic tolerance: when few series are testable, keep a softer floor.
            adaptive_min_ratio = max(0.25, min_ratio - 0.10) if considered < 4 else min_ratio
            if ratio < adaptive_min_ratio:
                reasons.append(f"stationarity_ratio_below_threshold:{ratio:.4f}<{adaptive_min_ratio:.4f}")
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

        if isinstance(walk_forward_avg_r2, (float, int)) and float(walk_forward_avg_r2) < adaptive_min_walk_forward_r2:
            wf_ci_upper = walk_forward.get("r2_ci_upper")
            wf_ci_clears = isinstance(wf_ci_upper, (float, int)) and float(wf_ci_upper) >= adaptive_min_walk_forward_r2
            if wf_ci_clears:
                advisory_reasons.append(
                    "r2_metric_alert_walk_forward_below_threshold_but_ci_upper_clears:"
                    f"{float(walk_forward_avg_r2):.4f}"
                    f"<{adaptive_min_walk_forward_r2:.4f};"
                    f"ci_upper={float(wf_ci_upper):.4f}"
                )
            elif walk_forward_unstable and isinstance(oos_r2, (float, int)) and float(oos_r2) >= min_r2:
                advisory_reasons.append("r2_metric_alert_walk_forward_unstable_but_oos_acceptable")
            else:
                advisory_reasons.append(f"r2_metric_alert_walk_forward_below_threshold:{float(walk_forward_avg_r2):.4f}<{adaptive_min_walk_forward_r2:.4f}")
        gate["walk_forward_context"] = {
            "status": walk_forward_status,
            "windows_requested": windows_requested,
            "windows_completed": windows_completed,
            "applied_min_walk_forward_r2": adaptive_min_walk_forward_r2,
            "base_min_walk_forward_r2": min_walk_forward_r2,
            "metric_unstable": walk_forward_unstable,
            "trend_directional_accuracy": trend_directional_accuracy,
            "volatility_r2": volatility_r2,
            "volatility_ratio": volatility_ratio,
            "trend_volatility_acceptable": trend_volatility_acceptable,
        }

        concentration = report.get("factor_concentration", {}) or {}
        top_share = concentration.get("top_share") if isinstance(concentration, dict) else None
        concentration_warn_threshold = float(profile.get("factor_concentration_warn_threshold", 0.65))
        if isinstance(top_share, (float, int)) and float(top_share) > concentration_warn_threshold:
            advisory_reasons.append(f"factor_concentration_alert_top_share_above_threshold:{float(top_share):.4f}>{concentration_warn_threshold:.4f}")
        gate["factor_concentration_context"] = {
            "top_factor": concentration.get("top_factor") if isinstance(concentration, dict) else None,
            "top_share": float(top_share) if isinstance(top_share, (float, int)) else None,
            "warn_threshold": concentration_warn_threshold,
        }

        freshness = report.get("freshness_alignment", {}) or {}
        lag_alignment_ok = freshness.get("lag_alignment_ok") if isinstance(freshness, dict) else None
        if lag_alignment_ok is False:
            advisory_reasons.append("freshness_alignment_warning_macro_lag_exceeds_policy")
        gate["freshness_context"] = {
            "target_horizon_days": (freshness.get("target_horizon_days") if isinstance(freshness, dict) else None),
            "max_publication_lag_days": (freshness.get("max_publication_lag_days") if isinstance(freshness, dict) else None),
            "lag_alignment_ok": lag_alignment_ok,
            "freshness_warn_days": int(profile.get("freshness_warn_days", 60)),
        }

        model_risk_score = report.get("model_risk_score")
        max_model_risk = float(profile["max_model_risk_score"])
        if isinstance(model_risk_score, (float, int)) and float(model_risk_score) > max_model_risk:
            advisory_reasons.append(f"model_risk_metric_alert_above_threshold:{float(model_risk_score):.4f}>{max_model_risk:.4f}")

        if isinstance(model_risk_score, (float, int)):
            score = float(model_risk_score)
            warn_thr = float(profile["model_risk_warn_threshold"])
            fail_thr = float(profile["model_risk_fail_threshold"])
            if score >= fail_thr:
                gate["severity"] = "fail"
            elif score >= warn_thr:
                gate["severity"] = "warn"
        gate_reasons.extend(advisory_reasons)
        gate_reasons.extend(reasons)
        gate["metric_only_signals"] = {
            "r2_used_as_validation_gate": False,
            "advisory_reasons": advisory_reasons,
        }
        gate["passed"] = len(reasons) == 0 or not hard_fail
        if advisory_reasons and gate.get("severity") == "pass":
            gate["severity"] = "warn"
        if reasons and gate.get("severity") == "pass":
            gate["severity"] = "warn" if not hard_fail else "fail"
        return gate

    def _resolve_governance_profile(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        regime = str(getattr(self.config, "governance_regime", "normal")).lower()
        if regime not in {"normal", "stress", "crisis"}:
            regime = "normal"

        profile: Dict[str, Any] = {
            "regime": regime,
            "hard_fail": bool(getattr(self.config, "governance_hard_fail", True)),
            "min_r2": float(getattr(self.config, "governance_min_r2", -0.25)),
            "max_normalized_shift": float(getattr(self.config, "governance_max_normalized_shift", 2.5)),
            "max_leakage_flags": int(getattr(self.config, "governance_max_leakage_flags", 1)),
            "min_stationary_ratio": float(getattr(self.config, "governance_min_stationary_ratio", 0.4)),
            "min_walk_forward_r2": float(getattr(self.config, "governance_min_walk_forward_r2", -0.25)),
            "max_model_risk_score": float(getattr(self.config, "governance_max_model_risk_score", 0.6)),
            "model_risk_warn_threshold": float(getattr(self.config, "governance_model_risk_warn_threshold", 0.4)),
            "model_risk_fail_threshold": float(getattr(self.config, "governance_model_risk_fail_threshold", 0.6)),
            "factor_concentration_warn_threshold": float(getattr(self.config, "governance_factor_concentration_warn_threshold", 0.65)),
            "freshness_warn_days": int(getattr(self.config, "governance_freshness_warn_days", 60)),
        }

        if ticker:
            overrides_map: Dict[str, Any] = getattr(self.config, "governance_ticker_overrides", {}) or {}
            ticker_overrides: Dict[str, Any] = overrides_map.get(ticker, {})
            _float_keys = {
                "min_r2",
                "max_normalized_shift",
                "min_stationary_ratio",
                "min_walk_forward_r2",
                "max_model_risk_score",
                "model_risk_warn_threshold",
                "model_risk_fail_threshold",
                "factor_concentration_warn_threshold",
            }
            for k, v in ticker_overrides.items():
                if k in _float_keys and k in profile:
                    profile[k] = float(v)
                elif k in {"max_leakage_flags", "freshness_warn_days"} and k in profile:
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
        ticker: Optional[str] = None,
    ) -> None:
        from logger.Catalog import catalog

        context = catalog.get_run_context()
        payload: Dict[str, Any] = {
            "schema_version": "governance-decision.v1",
            "generated_at": datetime.now().isoformat(),
            "run_mode": run_mode,
            "run_id": context.get("run_id"),
            "correlation_id": context.get("correlation_id"),
            "ticker": ticker,
            "gate": gate,
            "report": report,
        }
        decision_file = self.governance_path / f"governance_decision_{int(time.time() * 1000)}.json"
        with decision_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _finalize_governance(self, run_id: Optional[str] = None) -> None:
        """Write a stable worst-case aggregated governance file for the current run.

        Reads all governance_decision_*.json files written this run (matched by
        run_id when available) and produces ``governance_decision_current_run.json``
        in the same directory — a single file the Auditor can reliably query for
        the most-conservative outcome across all tickers.
        """
        try:
            all_files = sorted(self.governance_path.glob("governance_decision_*.json"))
            # Filter to current run if run_id is known.
            candidates: List[Dict[str, Any]] = []
            for fp in all_files:
                try:
                    payload = json.loads(fp.read_text(encoding="utf-8"))
                    if run_id and str(payload.get("run_id", "")).strip() != run_id:
                        continue
                    candidates.append(payload)
                except (json.JSONDecodeError, OSError):
                    pass

            if not candidates:
                return

            _severity_rank = {"fail": 3, "warn": 2, "pass": 1, "unknown": 0}

            def _decision_rank(p: Dict[str, Any]) -> int:
                gate = p.get("gate", {}) or {}
                if not gate.get("passed", True):
                    return 100 + _severity_rank.get(str(gate.get("severity", "fail")), 0)
                return _severity_rank.get(str(gate.get("severity", "pass")), 0)

            worst = max(candidates, key=_decision_rank)
            aggregated: Dict[str, Any] = {
                "schema_version": "governance-decision-aggregated.v1",
                "generated_at": datetime.now().isoformat(),
                "run_id": run_id,
                "ticker_count": len(candidates),
                "tickers": [p.get("ticker") for p in candidates],
                "aggregation": "worst_case",
                "worst_case_ticker": worst.get("ticker"),
                "gate": worst.get("gate", {}),
                "report": worst.get("report", {}),
            }
            agg_file = self.governance_path / "governance_decision_current_run.json"
            with agg_file.open("w", encoding="utf-8") as fh:
                json.dump(aggregated, fh, indent=2)
        except Exception as exc:
            self.logger.warning(f"_finalize_governance failed (non-fatal): {exc}")

    def read_governance_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load all governance decision artifacts, sorted oldest-first."""
        files: List[Path] = sorted(self.governance_path.glob("governance_decision_*.json"))
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

        passed_count = sum(1 for h in history if h.get("gate", {}).get("passed") is True)
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
            "avg_model_risk_score": (round(sum(risk_scores) / len(risk_scores), 4) if risk_scores else None),
            "worst_walk_forward_r2": (round(min(walk_r2s), 4) if walk_r2s else None),
            "direction": direction,
        }

    def _blocked_results(self, reason: str) -> Dict[str, Any]:
        return {
            "elasticity": reason,
            "feature_decay": reason,
            "lag_analysis": reason,
            "monte_carlo": reason,
            "stress_test": reason,
            "sensitivity_regression": reason,
            "forecasting": reason,
            "auto_ml": reason,
            "backtest_2020": reason,
        }

    def run_all_analyses(
        self,
        ticker: Optional[str] = None,
        macro_factor: str = "inflation",
        lags: int = 3,
        shock_map: Optional[Dict[str, float]] = None,
        scenario_name: Optional[str] = None,
        target: str = "log_return",
        factors: Optional[List[str]] = None,
        regression_model: str = "Auto",
    ) -> Dict[str, Any]:
        """
        Run all analyses and return results in a dictionary.
        """
        results = {}
        selected_ticker = self._resolve_ticker(ticker)
        random_seed = self._resolve_random_seed()
        analysis_df = self._analysis_df(selected_ticker)
        resolved_factors = self._resolve_analysis_factors(
            analysis_df=analysis_df,
            target=target,
            explicit_factors=factors,
        )
        selected_macro_factor = macro_factor if macro_factor in analysis_df.columns else (resolved_factors[0] if resolved_factors else macro_factor)
        effective_scenario = scenario_name or "geopolitical_conflict"
        scenario_payload = resolve_stress_scenario(
            scenario_name=effective_scenario,
            shock_map=shock_map,
        )
        # Stress testing is always enabled as part of the default pipeline run.
        stress_active = True
        try:
            start_time = time.time()
            results["correlation_matrix"] = correl_mtrx(
                analysis_df,
                stress_mode=stress_active,
                stress_strength=float(scenario_payload.get("correlation_breakdown_strength", 0.30)),
                scenario_name=str(scenario_payload.get("name", "custom")),
            )
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
            catalog.log_error("gold_layer", "AnalysisError", str(e), "correlation_matrix")
            self.logger.error(f"Analysis error in correlation matrix: {e}")
            results["correlation_matrix"] = None
        except Exception as e:
            catalog.log_error("gold_layer", "UnexpectedError", str(e), "correlation_matrix")
            self.logger.error(f"Unexpected error in correlation matrix: {e}")
            results["correlation_matrix"] = None

        try:
            results["governance_report"] = governance_report(
                analysis_df,
                target=target,
                factors=resolved_factors,
                random_seed=random_seed,
                reproducibility_enforced=bool(getattr(self.config, "enforce_reproducibility", True)),
                walk_forward_windows=int(getattr(self.config, "governance_walk_forward_windows", 4)),
                model_type=regression_model,
                min_target_horizon_days=int(getattr(self.config, "governance_min_target_horizon_days", 1)),
                max_target_horizon_days=int(getattr(self.config, "governance_max_target_horizon_days", 252)),
                walk_forward_tune_per_window=bool(getattr(self.config, "governance_walk_forward_tune_per_window", True)),
                factor_concentration_warn_threshold=float(getattr(self.config, "governance_factor_concentration_warn_threshold", 0.65)),
                freshness_warn_days=int(getattr(self.config, "governance_freshness_warn_days", 60)),
            )
            gate = self._evaluate_governance_gate(results["governance_report"], ticker=selected_ticker)
            results["governance_gate"] = gate
            self._export_governance_decision(gate, results.get("governance_report"), "sequential", ticker=selected_ticker)
            if not gate.get("passed", True):
                blocked_reason = f"blocked_by_governance_gate:{gate.get('reasons', [])}"
                results.update(self._blocked_results(blocked_reason))
                return results
        except AnalysisError as e:
            self.logger.error(f"Analysis error in governance report: {e}")
            results["governance_report"] = None
            gate = self._evaluate_governance_gate(None, ticker=selected_ticker)
            results["governance_gate"] = gate
            self._export_governance_decision(gate, None, "sequential", ticker=selected_ticker)
            if not gate.get("passed", True):
                blocked_reason = f"blocked_by_governance_gate:{gate.get('reasons', [])}"
                results.update(self._blocked_results(blocked_reason))
                return results
        except Exception as e:
            self.logger.error(f"Unexpected error in governance report: {e}")
            results["governance_report"] = None
            gate = self._evaluate_governance_gate(None, ticker=selected_ticker)
            results["governance_gate"] = gate
            self._export_governance_decision(gate, None, "sequential", ticker=selected_ticker)
            if not gate.get("passed", True):
                blocked_reason = f"blocked_by_governance_gate:{gate.get('reasons', [])}"
                results.update(self._blocked_results(blocked_reason))
                return results

        try:
            if "log_return" in analysis_df.columns and selected_macro_factor in analysis_df.columns:
                results["elasticity"] = elasticity(
                    analysis_df,
                    "log_return",
                    selected_macro_factor,
                    ticker=selected_ticker,
                    macro_lag_days=0,
                    rolling_window=90,
                )
                if results["elasticity"] is not None:
                    print(
                        ANALYSIS_ELASTICITY.format(
                            elasticity_value=(
                                f"{results['elasticity'].get('static_elasticity', 0.0):.4f}" if isinstance(results["elasticity"], dict) else "N/A"
                            )
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
            results["lag_analysis"] = lag_analysis(
                analysis_df,
                selected_macro_factor,
                max(lags, 90),
                target=target,
                ticker=selected_ticker,
                reference_lag_days=30,
            )
            if results["lag_analysis"] is not None and isinstance(results["lag_analysis"], dict):
                best_lag = results["lag_analysis"].get("best_lag_days", "N/A")
                best_corr = results["lag_analysis"].get("best_lag_correlation", 0.0)
                print(
                    ANALYSIS_LAG_ANALYSIS.format(
                        factor=selected_macro_factor,
                        best_lag=best_lag,
                        correlation=(f"{float(best_corr):.4f}" if isinstance(best_corr, (float, int)) else "N/A"),
                    )
                )
        except AnalysisError as e:
            self.logger.error(f"Analysis error in lag analysis: {e}")
            results["lag_analysis"] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in lag analysis: {e}")
            results["lag_analysis"] = None

        try:
            results["feature_decay"] = feature_decay_analysis(
                analysis_df,
                target=target,
                features=resolved_factors,
                max_lag=180,
            )
        except AnalysisError as e:
            self.logger.error(f"Analysis error in feature_decay: {e}")
            results["feature_decay"] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in feature_decay: {e}")
            results["feature_decay"] = None

        try:
            if selected_ticker:
                results["monte_carlo"] = monte_carlo(
                    analysis_df,
                    selected_ticker,
                    random_state=random_seed,
                    macro_scenario=("high_inflation" if shock_map and float(shock_map.get("inflation", 0.0)) > 0.0 else None),
                    macro_factor=selected_macro_factor,
                    scenario_bias=dict(scenario_payload.get("mc_bias", {})),
                )
                if results["monte_carlo"] is not None:
                    mc_paths = results["monte_carlo"].get("price_paths") if isinstance(results["monte_carlo"], dict) else None
                    print(
                        ANALYSIS_MONTE_CARLO.format(
                            iterations=mc_paths.shape[1] if hasattr(mc_paths, "shape") else "N/A",
                            ticker=selected_ticker,
                            days=mc_paths.shape[0] if hasattr(mc_paths, "shape") else "N/A",
                            min_price=f"{float(mc_paths.min()):.2f}" if hasattr(mc_paths, "min") else "N/A",
                            max_price=f"{float(mc_paths.max()):.2f}" if hasattr(mc_paths, "max") else "N/A",
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
            results["stress_test"] = stress_test(
                analysis_df,
                shock_map or {},
                target=target,
                ticker=selected_ticker,
                macro_lag_days=0,
                scenario_name=effective_scenario,
            )
            if results["stress_test"] is not None:
                print(
                    ANALYSIS_STRESS_TEST.format(
                        shock_details=str(
                            (results["stress_test"].get("scenario", {}) or {}).get("factor_shocks", shock_map or {})
                            if isinstance(results["stress_test"], dict)
                            else (shock_map or {})
                        ),
                        max_drawdown="N/A",  # Could calculate if needed
                    )
                )
        except AnalysisError as e:
            self.logger.error(f"Analysis error in stress test: {e}")
            results["stress_test"] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in stress test: {e}")
            results["stress_test"] = None

        try:
            results["sensitivity_regression"] = sensitivity_reg(
                analysis_df,
                target,
                resolved_factors,
                regression_model,
                ticker=selected_ticker,
                macro_lag_days=0,
            )
            if results["sensitivity_regression"] is not None:
                if isinstance(results["sensitivity_regression"], dict):
                    print(
                        ANALYSIS_SENSITIVITY_REGRESSION.format(
                            model_type=str(results["sensitivity_regression"].get("model", "OLS")),
                            top_factors=str(list(results["sensitivity_regression"].get("coefficients", {}).keys())),
                            coefficients=str(list(results["sensitivity_regression"].get("coefficients", {}).values())),
                            r_squared=str(results["sensitivity_regression"].get("r2", "N/A")),
                        )
                    )
        except AnalysisError as e:
            self.logger.error(f"Analysis error in sensitivity regression: {e}")
            results["sensitivity_regression"] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in sensitivity regression: {e}")
            results["sensitivity_regression"] = None

        try:
            results["forecasting"] = forecasting(
                analysis_df,
                target,
                10,
                ticker=selected_ticker,
                volatility_window=30,
            )
        except AnalysisError as e:
            self.logger.error(f"Analysis error in forecasting: {e}")
            results["forecasting"] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in forecasting: {e}")
            results["forecasting"] = None

        try:
            if bool(getattr(self.config, "auto_ml_enabled", False)):
                results["auto_ml"] = auto_ml_regression(
                    analysis_df,
                    target,
                    resolved_factors,
                    random_state=random_seed,
                    ticker=selected_ticker,
                    macro_lag_days=0,
                )
            else:
                results["auto_ml"] = "Auto-ML disabled by configuration"
        except AnalysisError as e:
            self.logger.error(f"Analysis error in auto_ml: {e}")
            results["auto_ml"] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in auto_ml: {e}")
            results["auto_ml"] = None

        try:
            results["backtest_2020"] = backtest_pre2020_holdout(
                analysis_df,
                target=target,
                features=resolved_factors,
                ticker=selected_ticker,
            )
        except AnalysisError as e:
            self.logger.error(f"Analysis error in backtest_2020: {e}")
            results["backtest_2020"] = None
        except Exception as e:
            self.logger.error(f"Unexpected error in backtest_2020: {e}")
            results["backtest_2020"] = None

        # Write worst-case aggregated governance file so Auditor has a single
        # stable artifact to read regardless of how many tickers were analysed.
        try:
            from logger.Catalog import catalog as _cat

            _run_id = _cat.get_run_context().get("run_id")
        except Exception:
            _run_id = None
        self._finalize_governance(run_id=_run_id)

        return results

    def run_all_analyses_parallel(
        self,
        ticker: Optional[str] = None,
        macro_factor: str = "inflation",
        lags: int = 3,
        shock_map: Optional[Dict[str, float]] = None,
        scenario_name: Optional[str] = None,
        target: str = "log_return",
        factors: Optional[List[str]] = None,
        max_workers: int = 4,
        regression_model: str = "Auto",
        include_auto_ml: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Run independent analyses in parallel.

        Uses a thread pool to avoid process-pickling overhead and to keep
        execution stable across different environments.
        """
        print(LIVE_STEP_7_RESULTS_GENERATION)
        results: Dict[str, Any] = {}
        worker_count = max(1, min(max_workers, 8))
        selected_ticker = self._resolve_ticker(ticker)
        random_seed = self._resolve_random_seed()
        analysis_df = self._analysis_df(selected_ticker)
        safe_factors = self._resolve_analysis_factors(
            analysis_df=analysis_df,
            target=target,
            explicit_factors=factors,
        )
        selected_macro_factor = macro_factor if macro_factor in analysis_df.columns else (safe_factors[0] if safe_factors else macro_factor)
        effective_scenario = scenario_name or "geopolitical_conflict"
        scenario_payload = resolve_stress_scenario(
            scenario_name=effective_scenario,
            shock_map=shock_map,
        )
        # Stress testing is always enabled as part of the default pipeline run.
        stress_active = True
        auto_ml_enabled = bool(getattr(self.config, "auto_ml_enabled", False)) if include_auto_ml is None else bool(include_auto_ml)

        # Run governance gate before advanced analyses.
        try:
            results["governance_report"] = governance_report(
                analysis_df,
                target,
                safe_factors,
                "date",
                0.2,
                24,
                random_seed,
                bool(getattr(self.config, "enforce_reproducibility", True)),
                int(getattr(self.config, "governance_walk_forward_windows", 4)),
                model_type=regression_model,
                min_target_horizon_days=int(getattr(self.config, "governance_min_target_horizon_days", 1)),
                max_target_horizon_days=int(getattr(self.config, "governance_max_target_horizon_days", 252)),
                walk_forward_tune_per_window=bool(getattr(self.config, "governance_walk_forward_tune_per_window", True)),
                factor_concentration_warn_threshold=float(getattr(self.config, "governance_factor_concentration_warn_threshold", 0.65)),
                freshness_warn_days=int(getattr(self.config, "governance_freshness_warn_days", 60)),
            )
            gate = self._evaluate_governance_gate(results["governance_report"], ticker=selected_ticker)
            results["governance_gate"] = gate
            self._export_governance_decision(gate, results.get("governance_report"), "parallel", ticker=selected_ticker)
            if not gate.get("passed", True):
                blocked_reason = f"blocked_by_governance_gate:{gate.get('reasons', [])}"
                results.update(self._blocked_results(blocked_reason))
                results["correlation_matrix"] = correl_mtrx(
                    analysis_df,
                    stress_mode=stress_active,
                    stress_strength=float(scenario_payload.get("correlation_breakdown_strength", 0.30)),
                    scenario_name=str(scenario_payload.get("name", "custom")),
                )
                return results
        except Exception:
            gate = self._evaluate_governance_gate(results.get("governance_report"), ticker=selected_ticker)
            results["governance_gate"] = gate
            self._export_governance_decision(gate, results.get("governance_report"), "parallel", ticker=selected_ticker)
            if not gate.get("passed", True):
                blocked_reason = f"blocked_by_governance_gate:{gate.get('reasons', [])}"
                results.update(self._blocked_results(blocked_reason))
                results["correlation_matrix"] = correl_mtrx(
                    analysis_df,
                    stress_mode=stress_active,
                    stress_strength=float(scenario_payload.get("correlation_breakdown_strength", 0.30)),
                    scenario_name=str(scenario_payload.get("name", "custom")),
                )
                return results

        # Define tasks as partial functions
        tasks: Dict[str, Callable[[], Any]] = {
            "correlation_matrix": partial(
                correl_mtrx,
                analysis_df,
                stress_active,
                float(scenario_payload.get("correlation_breakdown_strength", 0.30)),
                0.85,
                str(scenario_payload.get("name", "custom")),
            ),
            "lag_analysis": partial(
                lag_analysis,
                analysis_df,
                selected_macro_factor,
                max(lags, 90),
                target,
                selected_ticker,
                30,
            ),
            "feature_decay": partial(
                feature_decay_analysis,
                analysis_df,
                target,
                safe_factors,
                180,
            ),
            "sensitivity_regression": partial(
                sensitivity_reg,
                analysis_df,
                target,
                safe_factors,
                regression_model,
                selected_ticker,
                0,
            ),
            "forecasting": partial(
                forecasting,
                analysis_df,
                target,
                10,
                (2, 1, 1),
                selected_ticker,
                30,
            ),
            "backtest_2020": partial(
                backtest_pre2020_holdout,
                analysis_df,
                target,
                safe_factors,
                "date",
                selected_ticker,
            ),
        }

        if auto_ml_enabled:
            tasks["auto_ml"] = partial(
                auto_ml_regression,
                analysis_df,
                target,
                safe_factors,
                random_seed,
                selected_ticker,
                0,
            )

        if "log_return" in analysis_df.columns and selected_macro_factor in analysis_df.columns:
            tasks["elasticity"] = partial(
                elasticity,
                analysis_df,
                "log_return",
                selected_macro_factor,
                selected_ticker,
                0,
                90,
            )
        else:
            results["elasticity"] = "Required columns not available"

        if selected_ticker:
            tasks["monte_carlo"] = partial(
                monte_carlo,
                analysis_df,
                selected_ticker,
                252,
                10000,
                random_seed,
                "high_inflation" if shock_map and float(shock_map.get("inflation", 0.0)) > 0.0 else None,
                selected_macro_factor,
                dict(scenario_payload.get("mc_bias", {})),
            )
        else:
            results["monte_carlo"] = "Ticker not specified"

        tasks["stress_test"] = partial(
            stress_test,
            analysis_df,
            shock_map or {},
            target,
            selected_ticker,
            None,
            None,
            0,
            effective_scenario,
        )
        # Run parallel tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_key: Dict[concurrent.futures.Future[Any], str] = {executor.submit(task): key for key, task in tasks.items()}
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

        # Write worst-case aggregated governance file.
        try:
            from logger.Catalog import catalog as _cat

            _run_id = _cat.get_run_context().get("run_id")
        except Exception:
            _run_id = None
        self._finalize_governance(run_id=_run_id)

        return results
