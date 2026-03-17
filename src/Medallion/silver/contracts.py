from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set


@dataclass(frozen=True)
class SourceContract:
    required_columns: Set[str]
    percentage_entities: Set[str]


SOURCE_CONTRACTS: Dict[str, SourceContract] = {
    "yfinance": SourceContract(
        required_columns={"date", "close", "volume", "source_system", "ingested_at"},
        percentage_entities=set(),
    ),
    "fred": SourceContract(
        required_columns={"date", "value", "source_system"},
        percentage_entities={"inflation", "unemployment_rate"},
    ),
    "worldbank": SourceContract(
        required_columns={"date", "value", "economy", "source_system"},
        percentage_entities={"gdp_growth"},
    ),
}


EXPECTED_SOURCES = {"yfinance", "fred", "worldbank"}
