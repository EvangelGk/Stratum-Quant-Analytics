import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


class RunMode(Enum):
    SAMPLE = "sample"
    ACTUAL = "actual"


@dataclass
class ProjectConfig:
    """
    Central Configuration Hub.
    Διαχειρίζεται τα API keys και τα περιβάλλοντα εκτέλεσης.
    """

    fred_api_key: str
    mode: RunMode = RunMode.SAMPLE
    start_date: str = "2016-01-01"
    end_date: str = "2026-12-31"
    max_workers: int = 10
    max_retries: int = 4
    retry_delay_min: float = 1.0
    retry_delay_max: float = 3.0
    random_seed: Optional[int] = 42
    enforce_reproducibility: bool = True
    governance_hard_fail: bool = True
    governance_min_r2: float = -0.25
    governance_max_normalized_shift: float = 2.5
    governance_max_leakage_flags: int = 1
    governance_min_stationary_ratio: float = 0.4
    governance_walk_forward_windows: int = 4
    governance_min_walk_forward_r2: float = -0.25
    governance_max_model_risk_score: float = 0.6
    governance_regime: str = "normal"
    governance_model_risk_warn_threshold: float = 0.4
    governance_model_risk_fail_threshold: float = 0.6
    governance_ticker_overrides: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    data_user_id: str = "default"
    silver_hard_fail: bool = True
    silver_min_rows: int = 10
    silver_min_rows_ratio: float = 0.1
    silver_base_null_threshold: float = 30.0
    silver_dynamic_threshold_window: int = 20
    macro_series_map: Dict[str, str] = field(
        default_factory=lambda: {
            "CPIAUCSL": "inflation",
            "PNRGINDEXM": "energy_index",
        }
    )
    worldbank_indicator_map: Dict[str, str] = field(
        default_factory=lambda: {
            "NY.GDP.MKTP.KD.ZG": "gdp_growth",
            "EG.USE.PCAP.KG.OE": "energy_usage",
        }
    )

    @staticmethod
    def _parse_positive_int(raw_value: str, field_name: str, default: int) -> int:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    @staticmethod
    def _parse_non_negative_float(
        raw_value: str, field_name: str, default: float
    ) -> float:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return default
        return value if value >= 0 else default

    @staticmethod
    def _parse_float(raw_value: Optional[str], default: float) -> float:
        if raw_value is None:
            return default
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _validate_iso_date(date_value: str, field_name: str, default: str) -> str:
        try:
            datetime.strptime(date_value, "%Y-%m-%d")
            return date_value
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_optional_int(
        raw_value: Optional[str], default: Optional[int]
    ) -> Optional[int]:
        if raw_value is None or raw_value == "":
            return default
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_bool(raw_value: Optional[str], default: bool) -> bool:
        if raw_value is None:
            return default
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    @classmethod
    def load_from_env(cls) -> "ProjectConfig":
        load_dotenv()

        # validate and load FRED API key
        key = os.getenv("FRED_API_KEY")
        if not key:
            # error missing FRED API key
            raise ValueError("CRITICAL ERROR: FRED_API_KEY not found in .env file.")

        # get the mode (default to 'sample' if not found)
        env_mode = os.getenv("ENVIRONMENT", "sample").strip().lower()
        mode = RunMode.ACTUAL if env_mode == "actual" else RunMode.SAMPLE

        # get dates and other configs
        start_date = cls._validate_iso_date(
            os.getenv("START_DATE", "2016-01-01"), "START_DATE", "2016-01-01"
        )
        end_date = cls._validate_iso_date(
            os.getenv("END_DATE", "2026-12-31"), "END_DATE", "2026-12-31"
        )
        max_workers = cls._parse_positive_int(
            os.getenv("MAX_WORKERS", "10"), "MAX_WORKERS", 10
        )
        max_retries = cls._parse_positive_int(
            os.getenv("MAX_RETRIES", "4"), "MAX_RETRIES", 4
        )
        retry_delay_min = cls._parse_non_negative_float(
            os.getenv("RETRY_DELAY_MIN", "1.0"), "RETRY_DELAY_MIN", 1.0
        )
        retry_delay_max = cls._parse_non_negative_float(
            os.getenv("RETRY_DELAY_MAX", "3.0"), "RETRY_DELAY_MAX", 3.0
        )
        random_seed = cls._parse_optional_int(os.getenv("RANDOM_SEED"), 42)
        enforce_reproducibility = cls._parse_bool(
            os.getenv("ENFORCE_REPRODUCIBILITY"), True
        )
        governance_hard_fail = cls._parse_bool(os.getenv("GOVERNANCE_HARD_FAIL"), True)
        governance_min_r2 = cls._parse_float(os.getenv("GOVERNANCE_MIN_R2"), -0.25)
        governance_max_normalized_shift = cls._parse_non_negative_float(
            os.getenv("GOVERNANCE_MAX_NORMALIZED_SHIFT", "2.5"),
            "GOVERNANCE_MAX_NORMALIZED_SHIFT",
            2.5,
        )
        governance_max_leakage_flags = cls._parse_positive_int(
            os.getenv("GOVERNANCE_MAX_LEAKAGE_FLAGS", "1"),
            "GOVERNANCE_MAX_LEAKAGE_FLAGS",
            1,
        )
        governance_min_stationary_ratio = cls._parse_non_negative_float(
            os.getenv("GOVERNANCE_MIN_STATIONARY_RATIO", "0.4"),
            "GOVERNANCE_MIN_STATIONARY_RATIO",
            0.4,
        )
        governance_walk_forward_windows = cls._parse_positive_int(
            os.getenv("GOVERNANCE_WALK_FORWARD_WINDOWS", "4"),
            "GOVERNANCE_WALK_FORWARD_WINDOWS",
            4,
        )
        governance_min_walk_forward_r2 = cls._parse_float(
            os.getenv("GOVERNANCE_MIN_WALK_FORWARD_R2"), -0.25
        )
        governance_max_model_risk_score = cls._parse_non_negative_float(
            os.getenv("GOVERNANCE_MAX_MODEL_RISK_SCORE", "0.6"),
            "GOVERNANCE_MAX_MODEL_RISK_SCORE",
            0.6,
        )
        governance_regime = os.getenv("GOVERNANCE_REGIME", "normal").strip().lower()
        if governance_regime not in {"normal", "stress", "crisis"}:
            governance_regime = "normal"
        governance_model_risk_warn_threshold = cls._parse_non_negative_float(
            os.getenv("GOVERNANCE_MODEL_RISK_WARN_THRESHOLD", "0.4"),
            "GOVERNANCE_MODEL_RISK_WARN_THRESHOLD",
            0.4,
        )
        governance_model_risk_fail_threshold = cls._parse_non_negative_float(
            os.getenv("GOVERNANCE_MODEL_RISK_FAIL_THRESHOLD", "0.6"),
            "GOVERNANCE_MODEL_RISK_FAIL_THRESHOLD",
            0.6,
        )
        data_user_id = os.getenv("DATA_USER_ID", "default").strip() or "default"
        silver_hard_fail = cls._parse_bool(os.getenv("SILVER_HARD_FAIL"), True)
        silver_min_rows = cls._parse_positive_int(
            os.getenv("SILVER_MIN_ROWS", "10"), "SILVER_MIN_ROWS", 10
        )
        silver_min_rows_ratio = cls._parse_non_negative_float(
            os.getenv("SILVER_MIN_ROWS_RATIO", "0.1"),
            "SILVER_MIN_ROWS_RATIO",
            0.1,
        )
        silver_min_rows_ratio = min(max(silver_min_rows_ratio, 0.0), 1.0)
        silver_base_null_threshold = cls._parse_non_negative_float(
            os.getenv("SILVER_BASE_NULL_THRESHOLD", "30.0"),
            "SILVER_BASE_NULL_THRESHOLD",
            30.0,
        )
        silver_base_null_threshold = min(max(silver_base_null_threshold, 0.0), 100.0)
        silver_dynamic_threshold_window = cls._parse_positive_int(
            os.getenv("SILVER_DYNAMIC_THRESHOLD_WINDOW", "20"),
            "SILVER_DYNAMIC_THRESHOLD_WINDOW",
            20,
        )
        if governance_model_risk_fail_threshold < governance_model_risk_warn_threshold:
            (
                governance_model_risk_warn_threshold,
                governance_model_risk_fail_threshold,
            ) = (
                governance_model_risk_fail_threshold,
                governance_model_risk_warn_threshold,
            )
        governance_min_stationary_ratio = min(
            max(governance_min_stationary_ratio, 0.0), 1.0
        )
        if retry_delay_max < retry_delay_min:
            retry_delay_min, retry_delay_max = retry_delay_max, retry_delay_min

        raw_ticker_overrides = os.getenv("GOVERNANCE_TICKER_OVERRIDES", "").strip()
        governance_ticker_overrides: Dict[str, Dict[str, float]] = {}
        if raw_ticker_overrides:
            try:
                parsed = json.loads(raw_ticker_overrides)
                if isinstance(parsed, dict):
                    governance_ticker_overrides = {
                        k: {ik: float(iv) for ik, iv in v.items()}
                        for k, v in parsed.items()
                        if isinstance(v, dict)
                    }
            except (ValueError, TypeError, KeyError):
                pass

        macro_series_map = {
            "CPIAUCSL": "inflation",
            "PNRGINDEXM": "energy_index",
        }
        worldbank_indicator_map = {
            "NY.GDP.MKTP.KD.ZG": "gdp_growth",
            "EG.USE.PCAP.KG.OE": "energy_usage",
        }
        raw_macro_map = os.getenv("MACRO_SERIES_MAP", "").strip()
        if raw_macro_map:
            try:
                parsed = json.loads(raw_macro_map)
                if isinstance(parsed, dict):
                    macro_series_map = {str(k): str(v) for k, v in parsed.items()}
            except (ValueError, TypeError):
                pass

        raw_wb_map = os.getenv("WORLDBANK_INDICATOR_MAP", "").strip()
        if raw_wb_map:
            try:
                parsed = json.loads(raw_wb_map)
                if isinstance(parsed, dict):
                    worldbank_indicator_map = {
                        str(k): str(v) for k, v in parsed.items()
                    }
            except (ValueError, TypeError):
                pass

        cfg = cls(
            fred_api_key=key,
            mode=mode,
            start_date=start_date,
            end_date=end_date,
            max_workers=max_workers,
            max_retries=max_retries,
            retry_delay_min=retry_delay_min,
            retry_delay_max=retry_delay_max,
            random_seed=random_seed,
            enforce_reproducibility=enforce_reproducibility,
            governance_hard_fail=governance_hard_fail,
            governance_min_r2=governance_min_r2,
            governance_max_normalized_shift=governance_max_normalized_shift,
            governance_max_leakage_flags=governance_max_leakage_flags,
            governance_min_stationary_ratio=governance_min_stationary_ratio,
            governance_walk_forward_windows=governance_walk_forward_windows,
            governance_min_walk_forward_r2=governance_min_walk_forward_r2,
            governance_max_model_risk_score=governance_max_model_risk_score,
            governance_regime=governance_regime,
            governance_model_risk_warn_threshold=(governance_model_risk_warn_threshold),
            governance_model_risk_fail_threshold=(governance_model_risk_fail_threshold),
            governance_ticker_overrides=governance_ticker_overrides,
            data_user_id=data_user_id,
            silver_hard_fail=silver_hard_fail,
            silver_min_rows=silver_min_rows,
            silver_min_rows_ratio=silver_min_rows_ratio,
            silver_base_null_threshold=silver_base_null_threshold,
            silver_dynamic_threshold_window=silver_dynamic_threshold_window,
            macro_series_map=macro_series_map,
            worldbank_indicator_map=worldbank_indicator_map,
        )
        cfg.validate_runtime_constraints()
        return cfg

    def validate_runtime_constraints(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("Invalid configuration: START_DATE must be <= END_DATE")
        if self.max_workers < 1:
            raise ValueError("Invalid configuration: MAX_WORKERS must be >= 1")
        if self.max_retries < 1:
            raise ValueError("Invalid configuration: MAX_RETRIES must be >= 1")
        if self.retry_delay_min > self.retry_delay_max:
            raise ValueError(
                "Invalid configuration: RETRY_DELAY_MIN cannot exceed RETRY_DELAY_MAX"
            )
        if self.silver_min_rows < 1:
            raise ValueError("Invalid configuration: SILVER_MIN_ROWS must be >= 1")
        if not (0.0 <= self.silver_min_rows_ratio <= 1.0):
            raise ValueError(
                "Invalid configuration: SILVER_MIN_ROWS_RATIO must be in [0, 1]"
            )
        if not (0.0 <= self.silver_base_null_threshold <= 100.0):
            raise ValueError(
                "Invalid configuration: SILVER_BASE_NULL_THRESHOLD must be in [0, 100]"
            )
        if self.silver_dynamic_threshold_window < 1:
            raise ValueError(
                "Invalid configuration: SILVER_DYNAMIC_THRESHOLD_WINDOW must be >= 1"
            )
        if not self.macro_series_map:
            raise ValueError("Invalid configuration: MACRO_SERIES_MAP cannot be empty")
        if not self.worldbank_indicator_map:
            raise ValueError(
                "Invalid configuration: WORLDBANK_INDICATOR_MAP cannot be empty"
            )

    def get_targets(self) -> List[str]:
        """Επιστρέφει τα tickers βάσει του mode."""
        if self.mode == RunMode.SAMPLE:
            return ["AAPL", "F"]
        return ["AAPL", "TSLA", "MSFT", "WMT", "XOM"]

    def should_use_parallel_pipeline(self) -> bool:
        """Enable the parallel orchestration path only for full (actual) runs."""
        return self.mode == RunMode.ACTUAL

    def to_serializable_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "max_workers": self.max_workers,
            "max_retries": self.max_retries,
            "retry_delay_min": self.retry_delay_min,
            "retry_delay_max": self.retry_delay_max,
            "random_seed": self.random_seed,
            "enforce_reproducibility": self.enforce_reproducibility,
            "governance_hard_fail": self.governance_hard_fail,
            "governance_min_r2": self.governance_min_r2,
            "governance_max_normalized_shift": self.governance_max_normalized_shift,
            "governance_max_leakage_flags": self.governance_max_leakage_flags,
            "governance_min_stationary_ratio": self.governance_min_stationary_ratio,
            "governance_walk_forward_windows": self.governance_walk_forward_windows,
            "governance_min_walk_forward_r2": self.governance_min_walk_forward_r2,
            "governance_max_model_risk_score": self.governance_max_model_risk_score,
            "governance_regime": self.governance_regime,
            "governance_model_risk_warn_threshold": (
                self.governance_model_risk_warn_threshold
            ),
            "governance_model_risk_fail_threshold": (
                self.governance_model_risk_fail_threshold
            ),
            "governance_ticker_overrides": self.governance_ticker_overrides,
            "data_user_id": self.data_user_id,
            "silver_hard_fail": self.silver_hard_fail,
            "silver_min_rows": self.silver_min_rows,
            "silver_min_rows_ratio": self.silver_min_rows_ratio,
            "silver_base_null_threshold": self.silver_base_null_threshold,
            "silver_dynamic_threshold_window": self.silver_dynamic_threshold_window,
            "macro_series_map": self.macro_series_map,
            "worldbank_indicator_map": self.worldbank_indicator_map,
            "targets": self.get_targets(),
        }
