import hashlib
import json
from typing import Optional

import pandas as pd
from fredapi import Fred

from exceptions.FetchersExceptions import MissingAPIKeyError
from .BaseFetcher import BaseFetcher


class FredFetcher(BaseFetcher):
    CACHE_SCHEMA_VERSION = "v2"
    # FRED releases are monthly; a 7-day TTL avoids unnecessary re-fetches
    # while still picking up revised vintage data within the week.
    CACHE_TTL_SECONDS: int = 86400 * 7

    def __init__(self, api_key: Optional[str]):
        super().__init__()
        if not api_key:
            raise MissingAPIKeyError("FredFetcher requires a valid API key.")
        self.fred = Fred(api_key=api_key)

    # Fetching data from FRED with caching.
    def fetch(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_profile = {
            "schema": self.CACHE_SCHEMA_VERSION,
            "series_id": str(series_id),
            "start": str(start_date),
            "end": str(end_date),
            "normalizer": "fred-v1",
        }
        cache_hash = hashlib.sha256(
            json.dumps(cache_profile, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        key = f"fred_{self.CACHE_SCHEMA_VERSION}_{series_id}_{cache_hash}"
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        data = self.fred.get_series(
            series_id, observation_start=start_date, observation_end=end_date
        )
        df = data.to_frame(name="value").reset_index()
        df.columns = ["Date", "Value"]
        self._set_cached(key, df, expire=self.CACHE_TTL_SECONDS)
        return df
