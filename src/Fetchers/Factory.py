from typing import Optional

from .BaseFetcher import BaseFetcher
from .FredFetcher import FredFetcher
from .WorldBankFetcher import WorldBankFetcher
from .YFinanceFetcher import YFinanceFetcher


class DataFactory:
    """
    Factory class which orchestrates the creation of Fetchers
    and ensures proper instantiation.
    """

    def __init__(self, fred_api_key: Optional[str] = None) -> None:
        self._fetchers = {
            "yfinance": YFinanceFetcher(),
            "fred": FredFetcher(api_key=fred_api_key) if fred_api_key else None,
            "worldbank": WorldBankFetcher(),
        }

    def get_fetcher(self, source: str) -> BaseFetcher:
        fetcher = self._fetchers.get(source.lower())
        if not fetcher:
            raise ValueError(f"Source '{source}' is not supported.")
        return fetcher
