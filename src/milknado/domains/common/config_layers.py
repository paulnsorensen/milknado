"""Origin tracking for recursive configuration layering."""

from __future__ import annotations

from typing import Any

OriginMap = dict[tuple[str, ...], str]


def record_origins(
    origins: OriginMap, values: dict[str, Any], source: str, prefix: tuple[str, ...] = ()
) -> None:
    """Record the layer that supplied every leaf in values."""
    for key, value in values.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            record_origins(origins, value, source, path)
        else:
            origins[path] = source


def remove_origin_prefix(origins: OriginMap, prefix: tuple[str, ...]) -> None:
    """Discard attribution for a subtree that is being wholesale replaced."""
    for path in tuple(origins):
        if path[: len(prefix)] == prefix:
            del origins[path]


def origin_for(origins: OriginMap, path: tuple[str, ...]) -> str:
    """Return the nearest explicit origin, falling back to built-in defaults."""
    for end in range(len(path), 0, -1):
        if source := origins.get(path[:end]):
            return source
    return "default"
