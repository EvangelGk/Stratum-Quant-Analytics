from __future__ import annotations

# Copyright (c) 2026 EvangelGK. All Rights Reserved.

import os
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _scalar_to_str(value: object) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return None


def _streamlit_secret_value(key: str) -> str | None:
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            return _scalar_to_str(st.secrets[key])
    except Exception:
        return None
    return None


def _local_toml_value(key: str) -> str | None:
    if tomllib is None:
        return None
    path = _project_root() / ".streamlit" / "secrets.toml"
    if not path.exists():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if key in data:
        return _scalar_to_str(data[key])
    return None


def get_secret(key: str, default: str | None = None) -> str | None:
    """Resolve a setting from Streamlit secrets, local secrets.toml, then environment."""
    for candidate in (_streamlit_secret_value(key), _local_toml_value(key), os.getenv(key)):
        if candidate is not None and str(candidate).strip() != "":
            return str(candidate)
    return default


def bootstrap_env_from_secrets(
    override: bool = False,
    only_keys: Iterable[str] | None = None,
) -> None:
    """Populate os.environ from secrets for legacy getenv-based callsites."""
    keys = set(only_keys or [])
    sources: dict[str, str] = {}

    # Streamlit secrets source
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for key in st.secrets.keys():
                sval = _scalar_to_str(st.secrets[key])
                if sval is None:
                    continue
                if keys and key not in keys:
                    continue
                sources[key] = sval
    except Exception:
        pass

    # Local .streamlit/secrets.toml source
    if tomllib is not None:
        path = _project_root() / ".streamlit" / "secrets.toml"
        if path.exists():
            try:
                data = tomllib.loads(path.read_text(encoding="utf-8"))
                for key, raw in data.items():
                    sval = _scalar_to_str(raw)
                    if sval is None:
                        continue
                    if keys and key not in keys:
                        continue
                    sources.setdefault(str(key), sval)
            except Exception:
                pass

    for key, value in sources.items():
        if override or not os.getenv(key):
            os.environ[key] = value
