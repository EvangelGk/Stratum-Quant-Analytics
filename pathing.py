from __future__ import annotations

from pathlib import Path


def sanitize_profile_key(raw: str | None, default: str = "default") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(raw or "").strip())
    return cleaned or default


def build_output_dir(project_root: Path, user_key: str | None) -> Path:
    return Path(project_root) / "output" / sanitize_profile_key(user_key)


def build_user_data_dir(project_root: Path, user_key: str | None) -> Path:
    return Path(project_root) / "data" / "users" / sanitize_profile_key(user_key)


def output_path_diagnostics(active_output_dir: Path) -> tuple[str, str]:
    active_dir = Path(active_output_dir)
    return (
        f"Searching in: {active_dir.parent}",
        f"🔍 Active OUTPUT_DIR: {active_dir}",
    )