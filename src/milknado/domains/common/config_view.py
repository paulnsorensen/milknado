"""Serializable resolved and attributed views of layered configuration."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import msgspec

from milknado.domains.common.config import LoadedConfig
from milknado.domains.common.config_layers import origin_for
from milknado.domains.common.flavor_codec import FlavorOverride
from milknado.domains.common.flavor_profile import resolve_flavor_profile


def resolved_view(details: LoadedConfig, flavor: str | None = None) -> dict[str, object]:
    """Return the runtime configuration that dispatch code actually consumes."""
    value = resolve_flavor_profile(details.config, flavor) if flavor else details.config
    builtins = cast(object, msgspec.to_builtins(value, enc_hook=_encode_builtin))
    return cast(dict[str, object], _json_value(builtins))


def explain_view(details: LoadedConfig, flavor: str | None = None) -> dict[str, object]:
    """Return resolved values annotated with origins captured during layering."""
    if flavor:
        return _explain_flavor(details, flavor)
    return {
        key: _annotate(value, _config_path(details, key), details.origins)
        for key, value in resolved_view(details).items()
    }


def _explain_flavor(details: LoadedConfig, flavor: str) -> dict[str, object]:
    profile = resolved_view(details, flavor)
    override = details.config.flavors.get(flavor)
    root_paths = {
        "execution_agent": ("execution_agent",),
        "quality_gates": ("quality_gates",),
        "brief_prepend": ("prompts", "worker_brief_prepend"),
        "worker_agent_type": ("worker_agent_type",),
        "loop_mode": ("loop_mode",),
        "max_iterations": ("max_iterations",),
        "max_turns": ("max_turns",),
        "worktree": ("worktree",),
    }
    override_keys = {
        "execution_agent": "execution_agent",
        "quality_gates": "quality_gates",
        "brief_prepend": "brief_prepend",
        "worker_agent_type": "agent_type",
        "loop_mode": "loop_mode",
        "max_iterations": "max_iterations",
        "max_turns": "max_turns",
        "worktree": "worktree",
    }
    return {
        key: _annotate(
            value,
            _flavor_path(override, flavor, key, root_paths, override_keys),
            details.origins,
            flavor,
        )
        for key, value in profile.items()
    }


def _flavor_path(
    override: FlavorOverride | None,
    flavor: str,
    key: str,
    root_paths: dict[str, tuple[str, ...]],
    override_keys: dict[str, str],
) -> tuple[str, ...]:
    raw_key = override_keys.get(key)
    if override and raw_key and getattr(override, raw_key) is not None:
        return ("flavor", flavor, raw_key)
    return root_paths.get(key, (key,))


def _config_path(details: LoadedConfig, key: str) -> tuple[str, ...]:
    if key == "worker_tools":
        return ("worker", "tools")
    if key == "flavors":
        return ("flavor",)
    if key == "planning_prompt_prepend":
        return _prompt_path(details, "planning_prepend")
    if key == "worker_brief_prepend":
        return _prompt_path(details, "worker_brief_prepend")
    if key == "execution_agent" and origin_for(details.origins, (key,)) == "default":
        return ("worker", "tools", details.config.agent_family)
    if key == "planning_agent" and origin_for(details.origins, (key,)) == "default":
        return ("agent_family",)
    return (key,)


def _prompt_path(details: LoadedConfig, key: str) -> tuple[str, ...]:
    if origin_for(details.origins, ("prompts", key)) != "default":
        return ("prompts", key)
    return ("prompts", f"{key}_path")


def _annotate(
    value: object,
    path: tuple[str, ...],
    origins: dict[tuple[str, ...], str],
    flavor: str | None = None,
) -> object:
    if isinstance(value, dict):
        values = cast(dict[str, object], value)
        return {
            key: _annotate(item, (*path, key), origins, flavor) for key, item in values.items()
        }
    source = origin_for(origins, path)
    if flavor and source != "default":
        source = f"{source} (flavor:{flavor})"
    return {"value": value, "source": source}


def _encode_builtin(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported type: {type(value).__name__}")


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        values = cast(dict[object, object], value)
        return {str(key): _json_value(item) for key, item in values.items()}
    if isinstance(value, (frozenset, set)):
        values = cast(frozenset[object] | set[object], value)
        return [_json_value(item) for item in sorted(values, key=str)]
    if isinstance(value, (list, tuple)):
        values = cast(list[object] | tuple[object, ...], value)
        return [_json_value(item) for item in values]
    return value
