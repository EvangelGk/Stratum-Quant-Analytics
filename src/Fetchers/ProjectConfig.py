import os
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum

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

    @classmethod
    def load_from_env(cls):
        # load .env file
        load_dotenv()
        
        # validate and load FRED API key
        key = os.getenv("FRED_API_KEY")
        if not key:
            # error missing FRED API key
            raise ValueError("CRITICAL ERROR: FRED_API_KEY not found in .env file.")
        
        # get the mode (default to 'sample' if not found)
        env_mode = os.getenv("ENVIRONMENT", "sample").lower()
        mode = RunMode.ACTUAL if env_mode == "actual" else RunMode.SAMPLE
        
        return cls(fred_api_key=key, mode=mode)

    def get_targets(self):
        """Επιστρέφει τα tickers βάσει του mode."""
        if self.mode == RunMode.SAMPLE:
            return ["AAPL", "F"]
        return ["AAPL", "TSLA", "MSFT", "WMT", "XOM"]