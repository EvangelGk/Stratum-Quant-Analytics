import sys
from pathlib import Path

import pytest

# Ensure the `src` directory is on sys.path so that project imports work during tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.Fetchers.ProjectConfig import ProjectConfig  # noqa: E402


class _DummyConfig(ProjectConfig):
    """Shared lightweight ProjectConfig for unit tests (no real API key needed)."""

    def __init__(self) -> None:
        super().__init__(fred_api_key="dummy")
        self.max_workers = 1
        self.max_retries = 1
        self.retry_delay_min = 0.0
        self.retry_delay_max = 0.0


@pytest.fixture(scope="module")
def dummy_config() -> _DummyConfig:
    """One DummyConfig per test module — shared safely across tests in that module."""
    return _DummyConfig()
