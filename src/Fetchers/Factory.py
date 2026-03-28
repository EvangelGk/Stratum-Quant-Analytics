from typing import Optional

from exceptions.FetchersExceptions import MissingAPIKeyError

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
        self._disabled_reasons = {}
        if not fred_api_key:
            self._disabled_reasons["fred"] = "FRED fetcher disabled: missing FRED_API_KEY credential"
        self._fetchers = {
            "yfinance": YFinanceFetcher(),
            "fred": FredFetcher(api_key=fred_api_key) if fred_api_key else None,
            "worldbank": WorldBankFetcher(),
        }

    def get_fetcher(self, source: str) -> BaseFetcher:
        source_key = source.lower()
        if source_key not in self._fetchers:
            raise ValueError(f"Source '{source}' is not supported.")
        fetcher = self._fetchers.get(source_key)
        if not fetcher:
            reason = self._disabled_reasons.get(source_key, "source disabled")
            if source_key == "fred":
                raise MissingAPIKeyError(reason)
            raise ValueError(f"Source '{source}' unavailable: {reason}")
        return fetcher
