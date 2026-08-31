"""Generic deep-merge helper for nested dict/list config structures."""

from __future__ import annotations

import json
from typing import Literal, cast

ListMode = Literal["replace", "merge_dedup"]


def deep_merge(
    base: dict[str, object], override: dict[str, object], *, list_mode: ListMode = "replace"
) -> dict[str, object]:
    """Deep-merge ``override`` over ``base``, returning a new dict.

    Dicts recurse-merge. Lists replace (``list_mode="replace"``, the default)
    or extend with JSON-dedup (``list_mode="merge_dedup"``). Other values
    replace.
    """
    out = dict(base)
    for key, val in override.items():
        existing = out.get(key)
        if isinstance(val, dict) and isinstance(existing, dict):
            out[key] = deep_merge(
                cast(dict[str, object], existing),
                cast(dict[str, object], val),
                list_mode=list_mode,
            )
        elif list_mode == "merge_dedup" and isinstance(val, list) and isinstance(existing, list):
            existing_items = cast(list[object], existing)
            override_items = cast(list[object], val)
            seen = {json.dumps(item, sort_keys=True) for item in existing_items}
            out[key] = existing_items + [
                item for item in override_items if json.dumps(item, sort_keys=True) not in seen
            ]
        else:
            out[key] = val
    return out
