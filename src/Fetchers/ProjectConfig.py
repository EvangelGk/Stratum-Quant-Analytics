import os
from dataclasses import dataclass
from enum import Enum
from typing import List

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

    @classmethod
    def load_from_env(cls) -> "ProjectConfig":
        load_dotenv()

        # validate and load FRED API key
        key = os.getenv("FRED_API_KEY")
        if not key:
            # error missing FRED API key
            raise ValueError("CRITICAL ERROR: FRED_API_KEY not found in .env file.")

        # get the mode (default to 'sample' if not found)
        env_mode = os.getenv("ENVIRONMENT", "sample").lower()
        mode = RunMode.ACTUAL if env_mode == "actual" else RunMode.SAMPLE

        # get dates and other configs
        start_date = os.getenv("START_DATE", "2016-01-01")
        end_date = os.getenv("END_DATE", "2026-12-31")
        max_workers = int(os.getenv("MAX_WORKERS", "10"))
        max_retries = int(os.getenv("MAX_RETRIES", "4"))
        retry_delay_min = float(os.getenv("RETRY_DELAY_MIN", "1.0"))
        retry_delay_max = float(os.getenv("RETRY_DELAY_MAX", "3.0"))

        return cls(
            fred_api_key=key,
            mode=mode,
            start_date=start_date,
            end_date=end_date,
            max_workers=max_workers,
            max_retries=max_retries,
            retry_delay_min=retry_delay_min,
            retry_delay_max=retry_delay_max,
        )

    def get_targets(self) -> List[str]:
        """Επιστρέφει τα tickers βάσει του mode."""
        if self.mode == RunMode.SAMPLE:
            return ["AAPL", "F"]
        return ["AAPL", "TSLA", "MSFT", "WMT", "XOM"]
