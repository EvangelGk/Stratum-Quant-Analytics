from __future__ import annotations

# Copyright (c) 2026 EvangelGK. All Rights Reserved.

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _load_dotenv_once() -> None:
    """Load project .env once so getenv fallback works consistently in Streamlit."""
    if load_dotenv is None:
        return
    env_path = _project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _scalar_to_str(value: object) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return None


@lru_cache(maxsize=1)
def _streamlit_secrets_map() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for key in st.secrets.keys():
                sval = _scalar_to_str(st.secrets[key])
                if sval is not None:
                    out[str(key)] = sval
    except Exception:
        return {}
    return out


def _streamlit_secret_value(key: str) -> str | None:
    return _streamlit_secrets_map().get(key)


@lru_cache(maxsize=1)
def _load_local_secrets_toml() -> dict[str, object]:
    if tomllib is None:
        return {}
    path = _project_root() / ".streamlit" / "secrets.toml"
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _local_toml_value(key: str) -> str | None:
    data = _load_local_secrets_toml()
    if key in data:
        return _scalar_to_str(data[key])
    return None


def get_secret(key: str, default: str | None = None) -> str | None:
    """Resolve a setting from Streamlit secrets, local secrets.toml, then environment."""
    _load_dotenv_once()
    for candidate in (_streamlit_secret_value(key), _local_toml_value(key), os.getenv(key)):
        if candidate is not None and str(candidate).strip() != "":
            return str(candidate)
    return default


def bootstrap_env_from_secrets(
    override: bool = False,
    only_keys: Iterable[str] | None = None,
) -> None:
    """Populate os.environ from secrets for legacy getenv-based callsites."""
    _load_dotenv_once()
    keys = set(only_keys or [])
    sources: dict[str, str] = {}

    # Streamlit secrets source
    for key, sval in _streamlit_secrets_map().items():
        if keys and key not in keys:
            continue
        sources[key] = sval

    # Local .streamlit/secrets.toml source
    data = _load_local_secrets_toml()
    for key, raw in data.items():
        sval = _scalar_to_str(raw)
        if sval is None:
            continue
        skey = str(key)
        if keys and skey not in keys:
            continue
        sources.setdefault(skey, sval)

    for key, value in sources.items():
        if override or not os.getenv(key):
            os.environ[key] = value
