import os
from dataclasses import dataclass
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
        governance_hard_fail = cls._parse_bool(
            os.getenv("GOVERNANCE_HARD_FAIL"), True
        )
        governance_min_r2 = cls._parse_float(
            os.getenv("GOVERNANCE_MIN_R2"), -0.25
        )
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
        governance_min_stationary_ratio = min(
            max(governance_min_stationary_ratio, 0.0), 1.0
        )
        if retry_delay_max < retry_delay_min:
            retry_delay_min, retry_delay_max = retry_delay_max, retry_delay_min

        return cls(
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
            "targets": self.get_targets(),
        }
