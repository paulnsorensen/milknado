from __future__ import annotations

import re
from pathlib import Path


def slugify(value: str, max_length: int | None = None) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug if max_length is None else slug[:max_length].rstrip("-")


def validate_hint_path(
    path: str,
    project_root: Path,
    *,
    label: str = "path",
) -> None:
    if not path.strip():
        raise ValueError(f"{label} {path!r} is empty")
    candidate = Path(path.strip())
    resolved = project_root.resolve() / candidate if not candidate.is_absolute() else candidate
    if not resolved.resolve().is_relative_to(project_root.resolve()):
        raise ValueError(f"{label} {path!r} escapes project root {project_root}")


class TrustedGlobalPath(str):
    """Absolute path originating in the trusted user-global config."""


def trust_global_path(path: str, base_dir: Path) -> TrustedGlobalPath:
    candidate = Path(path)
    resolved = (base_dir / candidate if not candidate.is_absolute() else candidate).resolve()
    return TrustedGlobalPath(str(resolved))


def resolve_project_path(
    path: str,
    project_root: Path,
    *,
    label: str,
    allow_outside: bool = False,
) -> Path:
    """Resolve a config path and confine untrusted values to ``project_root``."""
    root = project_root.resolve()
    candidate = Path(path)
    resolved = (root / candidate if not candidate.is_absolute() else candidate).resolve()
    if not allow_outside and not resolved.is_relative_to(root):
        raise ValueError(f"{label} {path!r} escapes project_root '{project_root}'")
    return resolved


def normalize_hint_paths(files: list[str], project_root: Path) -> list[str]:
    """Absolute -> relative when under project_root; relative paths resolved against root.

    Rejects any path escaping the root (ValueError, fail fast).
    Non-existent paths allowed (hints may name files the task will create).
    Dedupes preserving order.
    """
    root = project_root.resolve()
    seen: list[str] = []
    seen_set: set[str] = set()
    for raw in files:
        validate_hint_path(raw, root)
        path = Path(raw.strip())
        resolved = (root / path if not path.is_absolute() else path).resolve()
        rel = resolved.relative_to(root)
        rel_str = rel.as_posix()
        if rel_str not in seen_set:
            seen_set.add(rel_str)
            seen.append(rel_str)
    return seen
