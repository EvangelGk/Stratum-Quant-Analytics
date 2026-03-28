import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import diskcache as dc
import pandas as pd

"""
Abstract Base Class setting the interface for every data source.
Includes caching for performance.
"""


class BaseFetcher(ABC):
    def __init__(self) -> None:
        default_cache = Path(__file__).parents[2] / "data" / "cache"
        cache_root = Path(os.getenv("SCENARIO_PLANNER_CACHE_DIR", str(default_cache)))
        cache_root.mkdir(parents=True, exist_ok=True)
        self.cache = dc.Cache(str(cache_root))

    @abstractmethod
    def fetch(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        :param identifier: Ticker (yfinance) ή Series ID (FRED)
        """
        pass

    def _get_cached(self, key: str) -> Optional[pd.DataFrame]:
        return self.cache.get(key)

    def _set_cached(self, key: str, data: pd.DataFrame, expire: int = 86400) -> None:  # 24 hours
        self.cache.set(key, data, expire=expire)
