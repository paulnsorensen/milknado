from __future__ import annotations

import re
from pathlib import Path


def slugify(value: str, max_length: int | None = None) -> str:
    """Lexically normalize text to a lowercase hyphenated slug.

    Blank or punctuation-only values return an empty string. When max_length is
    set, truncation is deterministic but does not make the result unique:
    callers that need a unique filename or branch must disambiguate collisions.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug if max_length is None else slug[:max_length].rstrip("-")


def validate_hint_path(
    path: str,
    project_root: Path,
    *,
    label: str = "path",
) -> None:
    """Validate an untrusted, whitespace-trimmed hint by resolving and confining it.

    This performs filesystem resolution and root confinement, but is validation
    at one instant; it does not make a later filesystem operation race-free.
    """
    normalized = path.strip()
    if not normalized:
        raise ValueError(f"{label} {path!r} is empty")
    resolve_project_path(normalized, project_root, label=label)


class TrustedGlobalPath(str):
    """Absolute path originating in the trusted user-global config."""


def trust_global_path(path: str, base_dir: Path) -> TrustedGlobalPath:
    """Resolve a user-global config path and explicitly mark it as trusted.

    This is the only trust-elevation helper: its result may resolve outside a
    project root when consumed by resolve_project_path.
    """
    candidate = Path(path)
    resolved = (base_dir / candidate if not candidate.is_absolute() else candidate).resolve()
    return TrustedGlobalPath(str(resolved))


def resolve_project_path(
    path: str | TrustedGlobalPath,
    project_root: Path,
    *,
    label: str,
) -> Path:
    """Resolve a path and confine untrusted values to project_root.

    Filesystem resolution catches absolute, .., and symlink escapes.
    TrustedGlobalPath is the explicit capability that permits an external
    global-config path. The returned path is safe only at resolution time;
    callers requiring race-free use must open it with confined descriptors.
    """
    root = project_root.resolve()
    candidate = Path(path)
    resolved = (root / candidate if not candidate.is_absolute() else candidate).resolve()
    if not isinstance(path, TrustedGlobalPath) and not resolved.is_relative_to(root):
        raise ValueError(f"{label} {path!r} escapes project_root '{project_root}'")
    return resolved


def normalize_hint_paths(files: list[str], project_root: Path) -> list[str]:
    """Resolve, confine, and canonically dedupe untrusted project-local hints.

    Input whitespace is trimmed; missing targets are valid because a hint may
    name a file to create. Results are root-relative POSIX paths in first-seen
    order. Resolution validates the path at that instant, not a later use.
    """
    root = project_root.resolve()
    seen: list[str] = []
    seen_set: set[str] = set()
    for raw in files:
        validate_hint_path(raw, root)
        resolved = resolve_project_path(raw.strip(), root, label="hint path")
        rel_str = resolved.relative_to(root).as_posix()
        if rel_str not in seen_set:
            seen_set.add(rel_str)
            seen.append(rel_str)
    return seen
