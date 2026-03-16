from abc import ABC, abstractmethod
from pathlib import Path

import diskcache as dc
import pandas as pd

"""
Abstract Base Class setting the interface for every data source.
Includes caching for performance.
"""


class BaseFetcher(ABC):
    def __init__(self):
        self.cache = dc.Cache(
            str(Path(__file__).parents[1] / "cache")
        )  # Cache in src/cache

    @abstractmethod
    def fetch(self, identifier: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        :param identifier: Ticker (yfinance) ή Series ID (FRED)
        """
        pass

    def _get_cached(self, key: str) -> pd.DataFrame or None:
        return self.cache.get(key)

    def _set_cached(
        self, key: str, data: pd.DataFrame, expire: int = 86400
    ):  # 24 hours
        self.cache.set(key, data, expire=expire)
