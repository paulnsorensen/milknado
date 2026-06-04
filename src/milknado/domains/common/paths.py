from __future__ import annotations

from pathlib import Path


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
        if not raw.strip():
            raise ValueError(f"path {raw!r} is empty")
        p = Path(raw.strip())
        resolved = (root / p if not p.is_absolute() else p).resolve()
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            raise ValueError(f"path {raw!r} escapes project root {project_root}") from None
        rel_str = rel.as_posix()
        if rel_str not in seen_set:
            seen_set.add(rel_str)
            seen.append(rel_str)
    return seen
