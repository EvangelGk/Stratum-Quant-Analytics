from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set


@dataclass(frozen=True)
class SeriesContract:
    semantic_type: str
    unit_kind: str
    null_tolerance_pct: float
    outlier_z_threshold: float
    enable_winsorization: bool = True
    imputation_strategy: str = "ffill_bfill"


@dataclass(frozen=True)
class SourceContract:
    required_columns: Set[str]
    percentage_entities: Set[str]
    default_series_contract: SeriesContract
    entity_contracts: Dict[str, SeriesContract]


SOURCE_CONTRACTS: Dict[str, SourceContract] = {
    "yfinance": SourceContract(
        required_columns={"date", "close", "volume", "source_system", "ingested_at"},
        percentage_entities=set(),
        default_series_contract=SeriesContract(
            semantic_type="price_series",
            unit_kind="absolute_price",
            null_tolerance_pct=10.0,
            outlier_z_threshold=4.0,
            enable_winsorization=True,
            imputation_strategy="ffill_bfill",
        ),
        entity_contracts={},
    ),
    "fred": SourceContract(
        required_columns={"date", "value", "source_system"},
        percentage_entities={"inflation", "unemployment_rate"},
        default_series_contract=SeriesContract(
            semantic_type="macro_level",
            unit_kind="index_or_level",
            null_tolerance_pct=25.0,
            outlier_z_threshold=4.5,
            enable_winsorization=True,
            imputation_strategy="ffill_bfill",
        ),
        entity_contracts={
            "inflation": SeriesContract(
                semantic_type="macro_rate",
                unit_kind="percentage",
                null_tolerance_pct=20.0,
                outlier_z_threshold=4.0,
                enable_winsorization=True,
                imputation_strategy="ffill_bfill",
            ),
            "energy_index": SeriesContract(
                semantic_type="macro_index",
                unit_kind="index_or_level",
                null_tolerance_pct=25.0,
                outlier_z_threshold=5.0,
                enable_winsorization=False,
                imputation_strategy="ffill_bfill",
            ),
        },
    ),
    "worldbank": SourceContract(
        required_columns={"date", "value", "economy", "source_system"},
        percentage_entities={"gdp_growth"},
        default_series_contract=SeriesContract(
            semantic_type="macro_structural",
            unit_kind="level",
            null_tolerance_pct=45.0,
            outlier_z_threshold=5.5,
            enable_winsorization=False,
            imputation_strategy="ffill_bfill",
        ),
        entity_contracts={
            "gdp_growth": SeriesContract(
                semantic_type="macro_growth",
                unit_kind="percentage",
                null_tolerance_pct=40.0,
                outlier_z_threshold=5.0,
                enable_winsorization=False,
                imputation_strategy="ffill_bfill",
            ),
            "energy_usage": SeriesContract(
                semantic_type="macro_usage",
                unit_kind="level",
                null_tolerance_pct=50.0,
                outlier_z_threshold=6.0,
                enable_winsorization=False,
                imputation_strategy="ffill_bfill",
            ),
        },
    ),
}


EXPECTED_SOURCES = {"yfinance", "fred", "worldbank"}

# ---------------------------------------------------------------------------
# Gold-layer schema contract
# Single source of truth for column names that GoldLayer writes and
# Auditor validates. Both sides reference this dict rather than hard-coding
# strings independently.
# ---------------------------------------------------------------------------
GOLD_COLUMN_MAP: Dict[str, str] = {
    # yfinance-derived
    "date": "date",
    "ticker": "ticker",
    "close": "close",
    "log_return": "log_return",
    "volume": "volume",
    # quality / audit metadata columns always present in master_table
    "quality_score": "quality_score",
    "imputed_count": "imputed_count",
    "outliers_clipped": "outliers_clipped",
}

# Columns that GoldLayer guarantees to be present regardless of macro/WB config
GOLD_REQUIRED_COLUMNS: frozenset = frozenset(
    GOLD_COLUMN_MAP[k] for k in ("date", "ticker", "close", "log_return", "volume")
)


def get_series_contract(source: str, entity_name: Optional[str] = None) -> SeriesContract:
    contract = SOURCE_CONTRACTS.get(source)
    if contract is None:
        return SeriesContract(
            semantic_type="unknown",
            unit_kind="unknown",
            null_tolerance_pct=30.0,
            outlier_z_threshold=4.5,
            enable_winsorization=True,
            imputation_strategy="ffill_bfill",
        )
    if entity_name and entity_name in contract.entity_contracts:
        return contract.entity_contracts[entity_name]
    return contract.default_series_contract
