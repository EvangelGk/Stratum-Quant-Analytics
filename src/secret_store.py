from __future__ import annotations

# Copyright (c) 2026 EvangelGK. All Rights Reserved.
import os
import threading
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


# Thread-safe caches for secret sources that may change while the app is running.
_env_cache_lock = threading.Lock()
_env_cache_signature: tuple[int, int] | None = None
_toml_cache_lock = threading.Lock()
_toml_cache: dict[str, object] = {}
_toml_cache_signature: tuple[int, int] | None = None
_secret_source_errors: dict[str, str] = {}
_injected_secret_keys: set[str] = set()


def _load_dotenv_once() -> None:
    """Reload project .env when the file changes so getenv fallback stays current."""
    global _env_cache_signature  # noqa: PLW0603
    if load_dotenv is None:
        return
    env_path = _project_root() / ".env"
    if not env_path.exists():
        return
    try:
        stat = env_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        _secret_source_errors["dotenv"] = str(exc)
        return
    with _env_cache_lock:
        if _env_cache_signature == signature:
            return
        try:
            load_dotenv(env_path, override=True)
            _env_cache_signature = signature
            _secret_source_errors.pop("dotenv", None)
        except Exception as exc:
            _secret_source_errors["dotenv"] = str(exc)


def _scalar_to_str(value: object) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return None


def _streamlit_secrets_map() -> dict[str, str]:
    """Read Streamlit secrets fresh each call — Streamlit manages its own cache."""
    out: dict[str, str] = {}
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for key in st.secrets.keys():
                sval = _scalar_to_str(st.secrets[key])
                if sval is not None:
                    out[str(key)] = sval
        _secret_source_errors.pop("streamlit", None)
    except Exception as exc:
        _secret_source_errors["streamlit"] = str(exc)
        return {}
    return out


def _streamlit_secret_value(key: str) -> str | None:
    return _streamlit_secrets_map().get(key)


def _load_local_secrets_toml() -> dict[str, object]:
    """Read .streamlit/secrets.toml, re-parsing only when the file changes on disk."""
    global _toml_cache, _toml_cache_signature  # noqa: PLW0603
    if tomllib is None:
        return {}
    path = _project_root() / ".streamlit" / "secrets.toml"
    if not path.exists():
        _secret_source_errors.pop("secrets.toml", None)
        return {}
    try:
        stat = path.stat()
        current_signature = (stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        _secret_source_errors["secrets.toml"] = str(exc)
        return {}
    with _toml_cache_lock:
        if current_signature != _toml_cache_signature:
            try:
                _toml_cache = tomllib.loads(path.read_text(encoding="utf-8"))
                _toml_cache_signature = current_signature
                _secret_source_errors.pop("secrets.toml", None)
            except Exception as exc:
                _toml_cache = {}
                _toml_cache_signature = current_signature
                _secret_source_errors["secrets.toml"] = str(exc)
        return dict(_toml_cache)


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
    global _injected_secret_keys  # noqa: PLW0603
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
        _injected_secret_keys.add(key)

    removable_keys = {key for key in _injected_secret_keys if (not keys or key in keys) and key not in sources}
    for key in removable_keys:
        os.environ.pop(key, None)
        _injected_secret_keys.discard(key)
