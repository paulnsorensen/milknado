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
