"""Generic deep-merge helper for nested dict/list config structures."""

from __future__ import annotations

import json
from typing import Any, Literal

ListMode = Literal["replace", "merge_dedup"]


def deep_merge(
    base: dict[str, Any], override: dict[str, Any], *, list_mode: ListMode = "replace"
) -> dict[str, Any]:
    """Deep-merge ``override`` over ``base``, returning a new dict.

    Dicts recurse-merge. Lists replace (``list_mode="replace"``, the default)
    or extend with JSON-dedup (``list_mode="merge_dedup"``). Other values
    replace.
    """
    out = dict(base)
    for key, val in override.items():
        existing = out.get(key)
        if isinstance(val, dict) and isinstance(existing, dict):
            out[key] = deep_merge(existing, val, list_mode=list_mode)
        elif list_mode == "merge_dedup" and isinstance(val, list) and isinstance(existing, list):
            seen = {json.dumps(item, sort_keys=True) for item in existing}
            out[key] = existing + [
                item for item in val if json.dumps(item, sort_keys=True) not in seen
            ]
        else:
            out[key] = val
    return out
