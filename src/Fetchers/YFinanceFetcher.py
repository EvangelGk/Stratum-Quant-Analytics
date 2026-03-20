import pandas as pd

from exceptions.FetchersExceptions import APIError, CacheError

from .BaseFetcher import BaseFetcher


class YFinanceFetcher(BaseFetcher):
    import yfinance as yf
    CACHE_SCHEMA_VERSION = "v3"

    def __init__(self) -> None:
        super().__init__()

    # Fetching data from Yahoo Finance with caching.
    def fetch(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        key = (
            f"yfinance_{self.CACHE_SCHEMA_VERSION}_"
            f"{ticker}_{start_date}_{end_date}"
        )
        try:
            cached = self._get_cached(key)
            if cached is not None:
                try:
                    return self._normalize_downloaded_frame(cached)
                except APIError:
                    # Cached payload may come from an older schema version.
                    self.cache.delete(key)
        except Exception as e:
            raise CacheError(f"Cache read error: {e}") from e

        try:
            data = self.yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            df = self._normalize_downloaded_frame(data)
        except Exception as e:
            raise APIError(f"YFinance API error: {e}") from e

        try:
            self._set_cached(key, df)
        except Exception as e:
            raise CacheError(f"Cache write error: {e}") from e

        return df

    def _normalize_downloaded_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize yfinance output shape into Bronze-required OHLCV columns."""
        if data is None or data.empty:
            return pd.DataFrame()

        df = data.copy()

        # Newer yfinance versions can return MultiIndex columns
        # with either OHLCV on level-0 or level-1.
        if isinstance(df.columns, pd.MultiIndex):
            level0 = [str(col[0]) for col in df.columns]
            level1 = [str(col[1]) for col in df.columns]
            known = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            if len(known.intersection(level0)) >= len(known.intersection(level1)):
                df.columns = level0
            else:
                df.columns = level1

        # Standardize column names that might vary by provider version.
        alias_map = {
            "date": "Date",
            "datetime": "Date",
            "index": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj close": "Adj Close",
            "adjclose": "Adj Close",
            "adjusted close": "Adj Close",
            "volume": "Volume",
        }
        normalized_cols = {}
        for col in df.columns:
            key = str(col).strip().lower().replace("_", " ")
            key = " ".join(key.split())
            normalized_cols[col] = alias_map.get(key, str(col))
        df = df.rename(columns=normalized_cols)

        df = df.reset_index()
        if "Datetime" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        if "index" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"index": "Date"})
        if "date" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"date": "Date"})

        # Last-resort Date inference for unnamed datetime columns.
        if "Date" not in df.columns and not df.empty:
            first_col = df.columns[0]
            maybe_dates = pd.to_datetime(df[first_col], errors="coerce")
            if maybe_dates.notna().any():
                df = df.rename(columns={first_col: "Date"})

        # Ensure mandatory columns exist for Bronze validation.
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

        # Some provider payloads may include duplicate OHLCV columns.
        # Keep a single canonical column per field by taking the first non-null value.
        df = self._collapse_duplicate_columns(df)

        required_order = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]
        missing = [col for col in required_order if col not in df.columns]
        if missing:
            raise APIError(f"Normalized yfinance data missing columns: {missing}")

        return df[required_order]

    def _collapse_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.columns.duplicated().any():
            return df

        work = df.copy()
        for col_name in list(dict.fromkeys(work.columns)):
            duplicated_positions = [
                idx for idx, name in enumerate(work.columns) if name == col_name
            ]
            if len(duplicated_positions) <= 1:
                continue
            dup_block = work.iloc[:, duplicated_positions]
            # Row-wise first non-null across duplicate columns.
            collapsed = dup_block.bfill(axis=1).iloc[:, 0]
            work = work.drop(columns=[col_name])
            work[col_name] = collapsed

        # Preserve deterministic order after reconstruction.
        ordered_unique = list(dict.fromkeys(df.columns))
        return work[ordered_unique]
