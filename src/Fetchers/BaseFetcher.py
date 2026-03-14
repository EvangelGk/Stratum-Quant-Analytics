from abc import ABC, abstractmethod
import pandas as pd

"""
Abstract Base Class setting the interface for every data source.
"""
class BaseFetcher(ABC):
    @abstractmethod
    def fetch(self, identifier: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        :param identifier: Ticker (yfinance) ή Series ID (FRED)
        """
        pass 
