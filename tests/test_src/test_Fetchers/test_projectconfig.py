import pytest

from src.Fetchers.ProjectConfig import ProjectConfig, RunMode


def test_load_from_env_missing_key(monkeypatch):
    # Ensure dotenv does not reload any stored API keys
    monkeypatch.setattr("Fetchers.ProjectConfig.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(ValueError):
        ProjectConfig.load_from_env()


def test_load_from_env_success(monkeypatch):
    monkeypatch.setattr("Fetchers.ProjectConfig.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    monkeypatch.setenv("ENVIRONMENT", "actual")
    cfg = ProjectConfig.load_from_env()
    assert cfg.fred_api_key == "dummy"
    assert cfg.mode == RunMode.ACTUAL
    assert cfg.start_date == "2016-01-01"
    assert cfg.end_date == "2026-12-31"
    assert cfg.max_workers > 0
