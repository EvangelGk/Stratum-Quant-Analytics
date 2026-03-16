import pytest

from src.Medallion.bronze import BronzeLayer
from src.Medallion.silver.silver import SilverLayer


@pytest.fixture
def silver_layer(dummy_config, tmp_path):
    """Function-scoped SilverLayer with isolated tmp directories per test."""
    layer = SilverLayer(dummy_config)
    layer.raw_path = tmp_path / "raw"
    layer.processed_path = tmp_path / "processed"
    layer.raw_path.mkdir(parents=True, exist_ok=True)
    layer.processed_path.mkdir(parents=True, exist_ok=True)
    return layer


@pytest.fixture
def bronze_layer(dummy_config, tmp_path):
    """Function-scoped BronzeLayer with isolated tmp directory per test."""
    layer = BronzeLayer(dummy_config, factory=None)
    layer.base_path = str(tmp_path)
    return layer
