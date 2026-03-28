import pytest

from src.Fetchers.ProjectConfig import ProjectConfig, RunMode


def test_load_from_env_missing_key(monkeypatch):
    # Ensure dotenv does not reload any stored API keys
    monkeypatch.setattr("src.Fetchers.ProjectConfig.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(ValueError):
        ProjectConfig.load_from_env()


def test_default_macro_map_includes_vix(monkeypatch):
    monkeypatch.delenv("MACRO_SERIES_MAP", raising=False)
    monkeypatch.delenv("WORLDBANK_INDICATOR_MAP", raising=False)
    monkeypatch.setenv("FRED_API_KEY", "")
    cfg = ProjectConfig.load_from_env()
    assert cfg.macro_series_map.get("VIXCLS") == "vix_index"


def test_load_from_env_success(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    monkeypatch.setenv("ENVIRONMENT", "actual")
    monkeypatch.setenv("DATA_USER_ID", "user_alpha")
    monkeypatch.setenv("SILVER_HARD_FAIL", "true")
    monkeypatch.setenv("SILVER_MIN_ROWS", "15")
    monkeypatch.setenv("SILVER_MIN_ROWS_RATIO", "0.2")
    monkeypatch.setenv("SILVER_BASE_NULL_THRESHOLD", "35")
    monkeypatch.setenv("SILVER_DYNAMIC_THRESHOLD_WINDOW", "25")
    monkeypatch.setenv("MACRO_SERIES_MAP", '{"CPIAUCSL":"inflation"}')
    monkeypatch.setenv("WORLDBANK_INDICATOR_MAP", '{"NY.GDP.MKTP.KD.ZG":"gdp_growth"}')
    cfg = ProjectConfig.load_from_env()
    assert cfg.fred_api_key == "dummy"
    assert cfg.mode == RunMode.ACTUAL
    assert cfg.start_date == "2016-01-01"
    assert cfg.end_date == "2026-12-31"
    assert cfg.max_workers > 0
    assert cfg.data_user_id == "user_alpha"
    assert cfg.silver_hard_fail is True
    assert cfg.silver_min_rows == 15
    assert cfg.silver_min_rows_ratio == 0.2
    assert cfg.silver_base_null_threshold == 35.0
    assert cfg.silver_dynamic_threshold_window == 25
    assert cfg.macro_series_map == {"CPIAUCSL": "inflation"}
    assert cfg.worldbank_indicator_map == {"NY.GDP.MKTP.KD.ZG": "gdp_growth"}
