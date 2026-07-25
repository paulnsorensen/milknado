from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

from milknado.domains.common.agent_argv import (
    DEFAULT_PLANNING_AGENT_BY_FAMILY,
    resolve_execution_agent_command,
    resolve_planning_agent_command,
    resolve_worker_tools,
)
from milknado.domains.common.config_layers import (
    OriginMap,
    record_origins,
    remove_origin_prefix,
)
from milknado.domains.common.flavor_codec import (
    FlavorOverride,
    Gate,
)
from milknado.domains.common.flavor_codec import (
    absolutize_global_flavor_paths as _absolutize_global_flavor_paths,
)
from milknado.domains.common.flavor_codec import (
    coerce_tool_list as _coerce_single_tool_list,
)
from milknado.domains.common.flavor_codec import (
    parse_flavor_tables as _parse_flavor_tables,
)
from milknado.domains.common.flavor_codec import (
    parse_gates as _parse_gates,
)
from milknado.domains.common.flavor_codec import (
    serialize_flavor_tables as _serialize_flavor_tables,
)
from milknado.domains.common.flavor_codec import (
    serialize_gates as _serialize_gates,
)
from milknado.domains.common.flavor_codec import (
    validate_loop_mode as _validated_loop_mode,
)
from milknado.domains.common.flavor_codec import (
    validate_positive_int as _validated_positive_int,
)
from milknado.domains.common.merge import deep_merge
from milknado.domains.common.paths import (
    TrustedGlobalPath,
    resolve_project_path,
    trust_global_path,
)
from milknado.domains.common.types import BUILTIN_FLAVORS

_logger = logging.getLogger(__name__)

# Keys that only make sense per-project. If set in the global config they are
# ignored with a warning rather than crashing.
_LOCAL_ONLY_KEYS = ("project_root", "db_path", "plugins")

# Native Workflow ("ultracode") backend defaults. Coordinator-side knobs that the
# subprocess dispatcher never reads; resolved per-flavor by resolve_flavor_profile.
DEFAULT_WORKER_AGENT_TYPE = "milknado:milknado-worker"
DEFAULT_LOOP_MODE = "redispatch"
DEFAULT_MAX_ITERATIONS = 8
DEFAULT_MAX_TURNS = 60
_LOOP_MODES = ("redispatch", "single")


@dataclass(frozen=True)
class MilknadoConfig:
    """Runtime config loaded from ``milknado.toml``."""

    agent_family: str = "claude"
    planning_agent: str = "claude --model opus -p --dangerously-skip-permissions"
    planning_validation_hook: str | None = None
    plan_reviewer_agent: str | None = None
    plan_review_max_rounds: int = 3
    execution_agent: str = resolve_execution_agent_command("claude")
    quality_gates: tuple[Gate, ...] | None = None
    worktree: bool = True
    worktree_pattern: str = "milknado-{node_id}-{slug}"
    concurrency_limit: int = 4
    project_root: Path = Path(".")
    db_path: Path = Path(".milknado/milknado.db")
    plugins: tuple[str, ...] = ()
    stall_threshold_seconds: int = 300
    dispatch_max_retries: int = 2
    dispatch_backoff_seconds: float = 5.0
    protected_branches: tuple[str, ...] = ("main", "master")
    completion_timeout_seconds: float | None = None
    eta_sample_size: int = 10
    worker_tools: dict[str, tuple[str, ...]] = field(default_factory=dict)
    flavors: dict[str, FlavorOverride] = field(default_factory=dict)
    flavor_registry: frozenset[str] = field(default_factory=lambda: BUILTIN_FLAVORS)
    planning_prompt_prepend: str | None = None
    worker_brief_prepend: str | None = None
    commit_footer: str | None = None
    # Native Workflow ("ultracode") backend defaults — coordinator-side only.
    worker_agent_type: str = DEFAULT_WORKER_AGENT_TYPE
    loop_mode: str = DEFAULT_LOOP_MODE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_turns: int = DEFAULT_MAX_TURNS


def default_config(project_root: Path) -> MilknadoConfig:
    return MilknadoConfig(
        agent_family="claude",
        planning_agent=resolve_planning_agent_command("claude"),
        execution_agent=resolve_execution_agent_command("claude"),
        project_root=project_root,
        db_path=project_root / ".milknado" / "milknado.db",
    )


def global_config_path() -> Path:
    """Return the canonical path of the user-global milknado config.

    Honors ``$XDG_CONFIG_HOME``; falls back to ``~/.config``. The file may not
    exist — callers must check.
    """
    base = os.environ.get("XDG_CONFIG_HOME", "").strip() or str(Path.home() / ".config")
    return Path(base) / "milknado" / "milknado.toml"


@dataclass(frozen=True)
class LoadedConfig:
    """Effective runtime config with source attribution captured during layering."""

    config: MilknadoConfig
    raw: dict[str, Any]
    origins: OriginMap


def load_config(path: Path, *, include_global: bool = True) -> MilknadoConfig:
    """Load config from path using the defaults → global → local cascade."""
    return load_config_details(path, include_global=include_global).config


def load_config_details(path: Path, *, include_global: bool = True) -> LoadedConfig:
    """Load config and preserve each supplied key's layer of origin."""
    local_raw = _read_milknado_section(path)
    inherit_global, replaced_flavors = _extract_inheritance(local_raw, local=True)
    raw: dict[str, Any] = {}
    origins: OriginMap = {}

    if include_global and inherit_global:
        global_path = global_config_path()
        if global_path.exists():
            global_raw = _read_milknado_section(global_path)
            _extract_inheritance(global_raw, local=False)
            _warn_local_only_keys(global_raw, global_path)
            for key in _LOCAL_ONLY_KEYS:
                global_raw.pop(key, None)
            _absolutize_global_prompt_paths(global_raw, global_path.parent)
            _absolutize_global_flavor_paths(global_raw, global_path.parent)
            _clear_prompts_alternate_keys(global_raw, local_raw)
            raw = _merge(raw, global_raw)
            record_origins(origins, global_raw, f"global:{global_path}")

    for flavor in replaced_flavors:
        raw.get("flavor", {}).pop(flavor, None)
        remove_origin_prefix(origins, ("flavor", flavor))
    raw = _merge(raw, local_raw)
    record_origins(origins, local_raw, f"local:{path}")
    return LoadedConfig(_build_config(raw, project_root=path.parent), raw, origins)


def _extract_inheritance(raw: dict[str, Any], *, local: bool) -> tuple[bool, tuple[str, ...]]:
    """Validate and remove layer-control metadata before normal merging."""
    inherit_global = raw.pop("inherit_global", True)
    if not isinstance(inherit_global, bool):
        raise ValueError("[milknado] inherit_global must be a boolean")
    replaced: list[str] = []
    flavors = raw.get("flavor")
    if isinstance(flavors, dict):
        for name, flavor in flavors.items():
            if isinstance(flavor, dict) and "inherit" in flavor:
                inherit = flavor.pop("inherit")
                if not isinstance(inherit, bool):
                    raise ValueError(f"[milknado.flavor.{name}] inherit must be a boolean")
                if local and not inherit:
                    replaced.append(name)
    return inherit_global, tuple(replaced)


def save_config(config: MilknadoConfig, path: Path) -> None:
    milknado = _serialize_milknado_core(config)

    prompts = _serialize_prompts(config)
    if prompts:
        milknado["prompts"] = prompts

    tools = _serialize_worker_tools(config)
    if tools:
        milknado["worker"] = {"tools": tools}

    flavor_tables = _serialize_flavor_tables(config.flavors)
    if flavor_tables:
        milknado["flavor"] = flavor_tables

    path.write_bytes(tomli_w.dumps({"milknado": milknado}).encode())


def _should_suppress_execution_agent(config: MilknadoConfig) -> bool:
    """True when execution_agent is a derived artifact of worker_tools, not user intent.

    When the family has a worker_tools single-list override AND the in-memory
    execution_agent is exactly the command that override would derive, the
    execution_agent is a derived artifact, not user intent — emitting it would
    shadow worker_tools on reload. But an explicit execution_agent that
    differs from the derived command IS user intent and must be preserved,
    so only suppress the field when it matches the derived value.
    """
    family_tools = config.worker_tools.get(config.agent_family)
    if family_tools is None:
        return False
    derived_execution_agent = resolve_execution_agent_command(
        config.agent_family,
        tools=resolve_worker_tools(config.agent_family, list(family_tools)),
    )
    return config.execution_agent == derived_execution_agent


def _serialize_milknado_core(config: MilknadoConfig) -> dict[str, Any]:
    milknado: dict[str, Any] = {
        "agent_family": config.agent_family,
        "planning_agent": config.planning_agent,
        "planning_validation_hook": config.planning_validation_hook or "",
    }
    if not _should_suppress_execution_agent(config):
        milknado["execution_agent"] = config.execution_agent
    if config.quality_gates is not None:
        milknado["quality_gates"] = _serialize_gates(config.quality_gates)
    milknado.update(
        {
            "worktree": config.worktree,
            "worktree_pattern": config.worktree_pattern,
            "concurrency_limit": config.concurrency_limit,
            "db_path": os.path.relpath(config.db_path, config.project_root),
            "plugins": list(config.plugins),
            "stall_threshold_seconds": config.stall_threshold_seconds,
            "dispatch_max_retries": config.dispatch_max_retries,
            "dispatch_backoff_seconds": config.dispatch_backoff_seconds,
            "protected_branches": list(config.protected_branches),
            "eta_sample_size": config.eta_sample_size,
            "worker_agent_type": config.worker_agent_type,
            "loop_mode": config.loop_mode,
            "max_iterations": config.max_iterations,
            "max_turns": config.max_turns,
            "plan_review_max_rounds": config.plan_review_max_rounds,
        }
    )
    if config.completion_timeout_seconds is not None:
        milknado["completion_timeout_seconds"] = config.completion_timeout_seconds
    if config.commit_footer is not None:
        milknado["commit_footer"] = config.commit_footer
    if config.plan_reviewer_agent is not None:
        milknado["plan_reviewer_agent"] = config.plan_reviewer_agent
    return milknado


def _serialize_prompts(config: MilknadoConfig) -> dict[str, str]:
    prompts: dict[str, str] = {}
    if config.planning_prompt_prepend is not None:
        prompts["planning_prepend"] = config.planning_prompt_prepend
    if config.worker_brief_prepend is not None:
        prompts["worker_brief_prepend"] = config.worker_brief_prepend
    return prompts


def _serialize_worker_tools(config: MilknadoConfig) -> dict[str, list[str]]:
    """Save worker.tools as single-list per family (raw, may contain "...").

    An explicit empty list is meaningful (no tools, replacing the family
    default), so it must survive the round trip.
    """
    return {fam: list(tool_list) for fam, tool_list in sorted(config.worker_tools.items())}


def detect_project_gates(project_root: Path) -> tuple[Gate, ...] | None:
    """Probe for language markers and return sane gates, else None.

    First match wins (checked in order). Returns None when no marker is found
    — the caller should inform the user that gates are unconfigured (fail-closed).
    """
    if (project_root / "pyproject.toml").exists():
        return (
            Gate("uv run pytest"),
            Gate("uv run ruff check"),
            Gate("uv run ty check"),
        )
    if (project_root / "Cargo.toml").exists():
        return (
            Gate("cargo test"),
            Gate("cargo clippy -- -D warnings"),
        )
    if (project_root / "package.json").exists():
        return (Gate("npm test"),)
    if (project_root / "go.mod").exists():
        return (
            Gate("go test ./..."),
            Gate("go vet ./..."),
        )
    if (project_root / "project.godot").exists():
        return (Gate("godot --headless --quit", fail_on_stdout="SCRIPT ERROR|^ERROR:"),)
    return None


def _read_milknado_section(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    section = data.get("milknado", data)
    if not isinstance(section, dict):
        raise ValueError(f"{path}: [milknado] is not a table")
    return dict(section)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` over ``base``. Scalars and lists replace; tables merge."""
    return deep_merge(base, override, list_mode="replace")


_PROMPT_PREPEND_SLOTS: tuple[str, ...] = ("planning_prepend", "worker_brief_prepend")


def _absolutize_global_prompt_paths(global_raw: dict[str, Any], base_dir: Path) -> None:
    """Resolve relative prompt ``_path`` values in the GLOBAL config against the
    global config directory.

    Prompt paths are otherwise resolved in ``_build_config`` against the local
    project root (``path.parent``) after the layered merge, so a relative global
    ``planning_prepend_path = "team.md"`` would wrongly look for the file next to
    the project's ``milknado.toml`` instead of next to the global config. Rewrite
    them to absolute here, before the merge, so the global base dir is honoured.
    Non-string values are left untouched for ``_load_prompt_prepend`` to reject.
    """
    prompts = global_raw.get("prompts")
    if not isinstance(prompts, dict):
        return
    for slot in _PROMPT_PREPEND_SLOTS:
        key = f"{slot}_path"
        value = prompts.get(key)
        if isinstance(value, str):
            prompts[key] = trust_global_path(value, base_dir)


def _clear_prompts_alternate_keys(global_raw: dict[str, Any], local_raw: dict[str, Any]) -> None:
    """Treat `<key>` and `<key>_path` as one logical slot during layered merge.

    If the local layer sets either form of a prompt-prepend slot, drop both
    forms from the global layer so the deep-merge doesn't leave a stale
    alternate behind that would later trip the mutual-exclusion check.
    """
    g_prompts = global_raw.get("prompts")
    l_prompts = local_raw.get("prompts")
    if not isinstance(g_prompts, dict) or not isinstance(l_prompts, dict):
        return
    for slot in _PROMPT_PREPEND_SLOTS:
        local_has = slot in l_prompts or f"{slot}_path" in l_prompts
        if local_has:
            g_prompts.pop(slot, None)
            g_prompts.pop(f"{slot}_path", None)


def _warn_local_only_keys(global_raw: dict[str, Any], path: Path) -> None:
    leaked = [k for k in _LOCAL_ONLY_KEYS if k in global_raw]
    if leaked:
        _logger.warning(
            "global milknado config %s sets per-project keys (%s); ignoring",
            path,
            ", ".join(leaked),
        )


def _validated_family(raw: dict[str, Any]) -> str:
    family = str(raw.get("agent_family", "claude")).strip().lower()
    if family not in DEFAULT_PLANNING_AGENT_BY_FAMILY:
        allowed = ", ".join(sorted(DEFAULT_PLANNING_AGENT_BY_FAMILY))
        raise ValueError(f"Invalid agent_family '{family}'. Expected one of: {allowed}")
    return family


def _resolve_agents(
    raw: dict[str, Any], family: str, worker_tools: dict[str, tuple[str, ...]]
) -> tuple[str, str]:
    planning_agent_raw = raw.get("planning_agent")
    execution_agent_raw = raw.get("execution_agent")
    planning_agent = resolve_planning_agent_command(
        family,
        planning_agent=(str(planning_agent_raw) if planning_agent_raw is not None else None),
    )
    # Default execution_agent is built from the family default + the worker-tool
    # override. An explicit `execution_agent` in TOML wins outright — it's the
    # opt-out for users who want full control of the worker invocation.
    family_tools = worker_tools.get(family)
    execution_agent = resolve_execution_agent_command(
        family,
        execution_agent=(str(execution_agent_raw) if execution_agent_raw is not None else None),
        tools=(
            resolve_worker_tools(family, list(family_tools)) if family_tools is not None else None
        ),
    )
    return planning_agent, execution_agent


def _scalar_config_kwargs(raw: dict[str, Any], project_root: Path) -> dict[str, Any]:
    planning_validation_hook_raw = raw.get("planning_validation_hook")
    return {
        "planning_validation_hook": (
            str(planning_validation_hook_raw).strip() if planning_validation_hook_raw else None
        ),
        "quality_gates": _parse_gates(raw.get("quality_gates"), "[milknado] quality_gates"),
        "worktree": _validated_bool(raw.get("worktree"), True, "[milknado] worktree"),
        "worktree_pattern": raw.get("worktree_pattern", "milknado-{node_id}-{slug}"),
        "concurrency_limit": raw.get("concurrency_limit", 4),
        "project_root": project_root,
        "db_path": _validated_db_path(project_root, raw.get("db_path", ".milknado/milknado.db")),
        "plugins": tuple(raw.get("plugins", [])),
        "stall_threshold_seconds": int(raw.get("stall_threshold_seconds", 300)),
        "dispatch_max_retries": int(raw.get("dispatch_max_retries", 2)),
        "dispatch_backoff_seconds": float(raw.get("dispatch_backoff_seconds", 5.0)),
        "protected_branches": tuple(raw.get("protected_branches", ["main", "master"])),
        "completion_timeout_seconds": (
            float(raw["completion_timeout_seconds"])
            if "completion_timeout_seconds" in raw
            else None
        ),
        "eta_sample_size": int(raw.get("eta_sample_size", 10)),
        "worker_agent_type": _validated_str(
            raw.get("worker_agent_type"), DEFAULT_WORKER_AGENT_TYPE, "[milknado] worker_agent_type"
        ),
        "loop_mode": _validated_loop_mode(raw.get("loop_mode", DEFAULT_LOOP_MODE), "[milknado]"),
        "max_iterations": _validated_positive_int(
            raw.get("max_iterations", DEFAULT_MAX_ITERATIONS), "[milknado] max_iterations"
        ),
        "max_turns": _validated_positive_int(
            raw.get("max_turns", DEFAULT_MAX_TURNS), "[milknado] max_turns"
        ),
        "commit_footer": _validated_optional_str(
            raw.get("commit_footer"), "[milknado] commit_footer"
        ),
        "plan_reviewer_agent": _validated_optional_str(
            raw.get("plan_reviewer_agent"), "[milknado] plan_reviewer_agent"
        ),
        "plan_review_max_rounds": _validated_positive_int(
            raw.get("plan_review_max_rounds", 3), "[milknado] plan_review_max_rounds"
        ),
    }


def _build_config(raw: dict[str, Any], *, project_root: Path) -> MilknadoConfig:
    family = _validated_family(raw)
    worker_tools = _parse_worker_tools(raw.get("worker"))
    flavors = _parse_flavor_tables(raw.get("flavor"), project_root)
    planning_agent, execution_agent = _resolve_agents(raw, family, worker_tools)

    kwargs = _scalar_config_kwargs(raw, project_root)
    kwargs.update(
        agent_family=family,
        planning_agent=planning_agent,
        execution_agent=execution_agent,
        worker_tools=worker_tools,
        flavors=flavors,
        flavor_registry=BUILTIN_FLAVORS | flavors.keys(),
        planning_prompt_prepend=_load_prompt_prepend(
            raw.get("prompts"), "planning_prepend", project_root
        ),
        worker_brief_prepend=_load_prompt_prepend(
            raw.get("prompts"), "worker_brief_prepend", project_root
        ),
    )
    return MilknadoConfig(**kwargs)


def _validated_bool(value: Any, default: bool, ctx: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{ctx} must be a boolean; got {type(value).__name__}")
    return value


def _validated_str(value: Any, default: str, ctx: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{ctx} must be a string; got {type(value).__name__}")
    return value


def _validated_optional_str(value: Any, ctx: str) -> str | None:
    """Return a string config value or None, rejecting non-string types instead of coercing."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{ctx} must be a string; got {type(value).__name__}")
    return value


def _parse_worker_tools(worker_raw: Any) -> dict[str, tuple[str, ...]]:
    """Parse [milknado.worker.tools] in single-list grammar.

    Each family key maps to a list (possibly containing one \"...\" sentinel).
    The stored tuple is the raw list; sentinel expansion happens at resolution
    time (in resolve_worker_tools / resolve_execution_agent_command).
    """
    if worker_raw is None:
        return {}
    if not isinstance(worker_raw, dict):
        raise ValueError("[milknado.worker] must be a table")
    tools_raw = worker_raw.get("tools")
    if tools_raw is None:
        return {}
    if not isinstance(tools_raw, dict):
        raise ValueError("[milknado.worker.tools] must be a table")
    out: dict[str, tuple[str, ...]] = {}
    for fam, tool_list in tools_raw.items():
        if not isinstance(fam, str):
            raise ValueError(f"[milknado.worker.tools] family keys must be strings, got {fam!r}")
        out[fam] = _coerce_single_tool_list(tool_list, f"[milknado.worker.tools.{fam}]")
    return out


def _load_prompt_prepend(
    prompts_raw: Any,
    base_key: str,
    project_root: Path,
) -> str | None:
    if prompts_raw is None:
        return None
    if not isinstance(prompts_raw, dict):
        raise ValueError("[milknado.prompts] must be a table")
    inline = prompts_raw.get(base_key)
    path_value = prompts_raw.get(f"{base_key}_path")
    if inline is not None and not isinstance(inline, str):
        raise ValueError(f"[milknado.prompts] {base_key} must be a string")
    if path_value is not None and not isinstance(path_value, str):
        raise ValueError(f"[milknado.prompts] {base_key}_path must be a string")
    if inline is not None and path_value is not None:
        raise ValueError(
            f"[milknado.prompts] {base_key} and {base_key}_path are mutually exclusive"
        )
    if inline:
        text = str(inline).strip()
        return text or None
    if path_value:
        resolved = resolve_project_path(
            str(path_value),
            project_root,
            label=f"[milknado.prompts] {base_key}_path",
            allow_outside=isinstance(path_value, TrustedGlobalPath),
        )
        if not resolved.exists():
            raise FileNotFoundError(
                f"[milknado.prompts] {base_key}_path does not exist: {resolved}"
            )
        text = resolved.read_text(encoding="utf-8").strip()
        return text or None
    return None


def _validated_db_path(project_root: Path, raw: str) -> Path:
    db_path = project_root / Path(raw)
    if not db_path.resolve().is_relative_to(project_root.resolve()):
        raise ValueError(f"db_path '{raw}' escapes project_root '{project_root}'")
    return db_path
