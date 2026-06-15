from __future__ import annotations

import logging
import os
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w

if TYPE_CHECKING:
    from milknado.domains.common.types import TaskFlavor

from milknado.domains.common.agent_argv import (
    _ALLOWED_WORKER_EXECUTABLES,
    DEFAULT_PLANNING_AGENT_BY_FAMILY,
    resolve_execution_agent_command,
    resolve_planning_agent_command,
    resolve_worker_tools,
)

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
class FlavorOverride:
    """Per-flavor config knobs loaded from [milknado.flavor.<flavor>] TOML tables.

    All fields are optional; absent = inherit from global config.
    ``brief_prepend`` holds the resolved text (inline or loaded from path(s)).
    ``quality_gates = ()`` means skip gates; ``None`` means inherit.
    ``agent_type`` / ``loop_mode`` / ``max_iterations`` / ``max_turns`` drive the
    native Workflow backend; ``None`` inherits the global default.
    """

    execution_agent: str | None = None
    tools: tuple[str, ...] | None = None  # may contain one "..." sentinel
    brief_prepend: str | None = None  # resolved text
    quality_gates: tuple[str, ...] | None = None
    agent_type: str | None = None
    loop_mode: str | None = None
    max_iterations: int | None = None
    max_turns: int | None = None


@dataclass(frozen=True)
class MilknadoConfig:
    """Runtime config loaded from ``milknado.toml``."""

    agent_family: str = "claude"
    planning_agent: str = "claude --model opus -p --dangerously-skip-permissions"
    planning_validation_hook: str | None = None
    execution_agent: str = resolve_execution_agent_command("claude")
    quality_gates: tuple[str, ...] = ("uv run pytest", "uv run ruff check", "uv run ty check")
    worktree_pattern: str = "milknado-{node_id}-{slug}"
    concurrency_limit: int = 4
    project_root: Path = Path(".")
    db_path: Path = Path(".milknado/milknado.db")
    plugins: tuple[str, ...] = ()
    stall_threshold_seconds: int = 300
    dispatch_max_retries: int = 2
    dispatch_backoff_seconds: float = 5.0
    protected_branches: tuple[str, ...] = ("main", "master")
    completion_timeout_seconds: float = 1800.0
    eta_sample_size: int = 10
    worker_tools: dict[str, tuple[str, ...]] = field(default_factory=dict)
    flavors: dict[TaskFlavor, FlavorOverride] = field(default_factory=dict)
    planning_prompt_prepend: str | None = None
    worker_brief_prepend: str | None = None
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


def load_config(path: Path, *, include_global: bool = True) -> MilknadoConfig:
    """Load config from ``path``, optionally merged under the global config.

    Load order (last write wins): built-in defaults → global → ``path``.

    ``include_global=False`` skips the global lookup — used by tests and any
    caller that must be reproducible regardless of the host's user config.
    """
    raw: dict[str, Any] = {}
    local_raw = _read_milknado_section(path)
    if include_global:
        g = global_config_path()
        if g.exists():
            global_raw = _read_milknado_section(g)
            _warn_local_only_keys(global_raw, g)
            for k in _LOCAL_ONLY_KEYS:
                global_raw.pop(k, None)
            _absolutize_global_prompt_paths(global_raw, g.parent)
            _absolutize_global_flavor_paths(global_raw, g.parent)
            _clear_prompts_alternate_keys(global_raw, local_raw)
            raw = _merge(raw, global_raw)
    raw = _merge(raw, local_raw)
    return _build_config(raw, project_root=path.parent)


def save_config(config: MilknadoConfig, path: Path) -> None:
    # When the family has a worker_tools single-list override AND the in-memory
    # execution_agent is exactly the command that override would derive, the
    # execution_agent is a derived artifact, not user intent — emitting it would
    # shadow worker_tools on reload. But an explicit execution_agent that
    # differs from the derived command IS user intent and must be preserved,
    # so only suppress the field when it matches the derived value.
    family_tools = config.worker_tools.get(config.agent_family)
    suppress_execution_agent = False
    if family_tools is not None:
        derived_execution_agent = resolve_execution_agent_command(
            config.agent_family,
            tools=resolve_worker_tools(config.agent_family, list(family_tools)),
        )
        suppress_execution_agent = config.execution_agent == derived_execution_agent

    milknado: dict[str, Any] = {
        "agent_family": config.agent_family,
        "planning_agent": config.planning_agent,
        "planning_validation_hook": config.planning_validation_hook or "",
    }
    if not suppress_execution_agent:
        milknado["execution_agent"] = config.execution_agent
    milknado.update(
        {
            "quality_gates": list(config.quality_gates),
            "worktree_pattern": config.worktree_pattern,
            "concurrency_limit": config.concurrency_limit,
            "db_path": os.path.relpath(config.db_path, config.project_root),
            "plugins": list(config.plugins),
            "stall_threshold_seconds": config.stall_threshold_seconds,
            "dispatch_max_retries": config.dispatch_max_retries,
            "dispatch_backoff_seconds": config.dispatch_backoff_seconds,
            "protected_branches": list(config.protected_branches),
            "completion_timeout_seconds": config.completion_timeout_seconds,
            "eta_sample_size": config.eta_sample_size,
            "worker_agent_type": config.worker_agent_type,
            "loop_mode": config.loop_mode,
            "max_iterations": config.max_iterations,
            "max_turns": config.max_turns,
        }
    )

    prompts: dict[str, str] = {}
    if config.planning_prompt_prepend is not None:
        prompts["planning_prepend"] = config.planning_prompt_prepend
    if config.worker_brief_prepend is not None:
        prompts["worker_brief_prepend"] = config.worker_brief_prepend
    if prompts:
        milknado["prompts"] = prompts

    # Save worker.tools as single-list per family (raw, may contain "...").
    # An explicit empty list is meaningful (no tools, replacing the family
    # default), so it must survive the round trip.
    tools: dict[str, list[str]] = {
        fam: list(tool_list) for fam, tool_list in sorted(config.worker_tools.items())
    }
    if tools:
        milknado["worker"] = {"tools": tools}

    # Save [flavor.*] tables.
    flavor_tables: dict[str, dict[str, Any]] = {}
    for flavor_key, fo in config.flavors.items():
        from milknado.domains.common.types import TaskFlavor

        flavor_name = flavor_key.value if isinstance(flavor_key, TaskFlavor) else str(flavor_key)
        entry: dict[str, Any] = {}
        if fo.execution_agent is not None:
            entry["execution_agent"] = fo.execution_agent
        if fo.tools is not None:
            entry["tools"] = list(fo.tools)
        if fo.brief_prepend is not None:
            entry["brief_prepend"] = fo.brief_prepend
        if fo.quality_gates is not None:
            entry["quality_gates"] = list(fo.quality_gates)
        if fo.agent_type is not None:
            entry["agent_type"] = fo.agent_type
        if fo.loop_mode is not None:
            entry["loop_mode"] = fo.loop_mode
        if fo.max_iterations is not None:
            entry["max_iterations"] = fo.max_iterations
        if fo.max_turns is not None:
            entry["max_turns"] = fo.max_turns
        if entry:
            flavor_tables[flavor_name] = entry
    if flavor_tables:
        milknado["flavor"] = flavor_tables

    path.write_bytes(tomli_w.dumps({"milknado": milknado}).encode())


def _read_milknado_section(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    section = data.get("milknado", data)
    if not isinstance(section, dict):
        raise ValueError(f"{path}: [milknado] is not a table")
    return dict(section)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` over ``base``. Scalars and lists replace; tables merge."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


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
        if isinstance(value, str) and not Path(value).is_absolute():
            prompts[key] = str((base_dir / value).resolve())


def _absolutize_global_flavor_paths(global_raw: dict[str, Any], base_dir: Path) -> None:
    """Resolve relative flavor brief_prepend_path values in the global config."""
    flavor = global_raw.get("flavor")
    if not isinstance(flavor, dict):
        return
    for _flavor_name, flavor_raw in flavor.items():
        if not isinstance(flavor_raw, dict):
            continue
        value = flavor_raw.get("brief_prepend_path")
        if isinstance(value, str) and not Path(value).is_absolute():
            flavor_raw["brief_prepend_path"] = str((base_dir / value).resolve())
        elif isinstance(value, list):
            flavor_raw["brief_prepend_path"] = [
                str((base_dir / v).resolve())
                if isinstance(v, str) and not Path(v).is_absolute()
                else v
                for v in value
            ]


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


def _build_config(raw: dict[str, Any], *, project_root: Path) -> MilknadoConfig:
    family = str(raw.get("agent_family", "claude")).strip().lower()
    if family not in DEFAULT_PLANNING_AGENT_BY_FAMILY:
        allowed = ", ".join(sorted(DEFAULT_PLANNING_AGENT_BY_FAMILY))
        raise ValueError(
            f"Invalid agent_family '{family}'. Expected one of: {allowed}",
        )

    planning_agent_raw = raw.get("planning_agent")
    planning_validation_hook_raw = raw.get("planning_validation_hook")
    execution_agent_raw = raw.get("execution_agent")

    worker_tools = _parse_worker_tools(raw.get("worker"))
    flavors = _parse_flavor_tables(raw.get("flavor"), project_root)
    planning_prompt_prepend = _load_prompt_prepend(
        raw.get("prompts"),
        "planning_prepend",
        project_root,
    )
    worker_brief_prepend = _load_prompt_prepend(
        raw.get("prompts"),
        "worker_brief_prepend",
        project_root,
    )

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

    return MilknadoConfig(
        agent_family=family,
        planning_agent=planning_agent,
        planning_validation_hook=(
            str(planning_validation_hook_raw).strip() if planning_validation_hook_raw else None
        ),
        execution_agent=execution_agent,
        quality_gates=tuple(
            raw.get(
                "quality_gates",
                ["uv run pytest", "uv run ruff check", "uv run ty check"],
            )
        ),
        worktree_pattern=raw.get("worktree_pattern", "milknado-{node_id}-{slug}"),
        concurrency_limit=raw.get("concurrency_limit", 4),
        project_root=project_root,
        db_path=_validated_db_path(project_root, raw.get("db_path", ".milknado/milknado.db")),
        plugins=tuple(raw.get("plugins", [])),
        stall_threshold_seconds=int(raw.get("stall_threshold_seconds", 300)),
        dispatch_max_retries=int(raw.get("dispatch_max_retries", 2)),
        dispatch_backoff_seconds=float(raw.get("dispatch_backoff_seconds", 5.0)),
        protected_branches=tuple(raw.get("protected_branches", ["main", "master"])),
        completion_timeout_seconds=float(raw.get("completion_timeout_seconds", 1800.0)),
        eta_sample_size=int(raw.get("eta_sample_size", 10)),
        worker_tools=worker_tools,
        flavors=flavors,
        planning_prompt_prepend=planning_prompt_prepend,
        worker_brief_prepend=worker_brief_prepend,
        worker_agent_type=str(raw.get("worker_agent_type", DEFAULT_WORKER_AGENT_TYPE)),
        loop_mode=_validated_loop_mode(raw.get("loop_mode", DEFAULT_LOOP_MODE), "[milknado]"),
        max_iterations=_validated_positive_int(
            raw.get("max_iterations", DEFAULT_MAX_ITERATIONS), "[milknado] max_iterations"
        ),
        max_turns=_validated_positive_int(
            raw.get("max_turns", DEFAULT_MAX_TURNS), "[milknado] max_turns"
        ),
    )


def _validated_loop_mode(value: Any, ctx: str) -> str:
    """Coerce a loop_mode value to one of the allowed modes, else raise."""
    mode = str(value)
    if mode not in _LOOP_MODES:
        raise ValueError(f"{ctx} loop_mode must be one of {list(_LOOP_MODES)}; got {mode!r}")
    return mode


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


def _parse_flavor_tables(
    flavor_raw: Any,
    project_root: Path,
) -> dict[TaskFlavor, FlavorOverride]:
    """Parse [milknado.flavor.*] tables into {TaskFlavor: FlavorOverride}."""
    from milknado.domains.common.types import TaskFlavor

    if flavor_raw is None:
        return {}
    if not isinstance(flavor_raw, dict):
        raise ValueError("[milknado.flavor] must be a table")

    out: dict[TaskFlavor, FlavorOverride] = {}
    for flavor_name, entry in flavor_raw.items():
        try:
            flavor = TaskFlavor(flavor_name)
        except ValueError:
            raise ValueError(
                f"[milknado.flavor] unknown flavor key {flavor_name!r}; "
                f"expected one of {sorted(f.value for f in TaskFlavor)}"
            ) from None
        if not isinstance(entry, dict):
            raise ValueError(f"[milknado.flavor.{flavor_name}] must be a table")
        fo = _parse_flavor_entry(entry, flavor_name, project_root)
        out[flavor] = fo
    return out


def _parse_flavor_entry(
    entry: dict[str, Any],
    flavor_name: str,
    project_root: Path,
) -> FlavorOverride:
    """Parse one [milknado.flavor.<flavor>] table."""
    ctx = f"[milknado.flavor.{flavor_name}]"
    # execution_agent
    execution_agent_raw = entry.get("execution_agent")
    execution_agent: str | None = None
    if execution_agent_raw is not None:
        if not isinstance(execution_agent_raw, str):
            raise ValueError(f"{ctx} execution_agent must be a string")
        argv = shlex.split(execution_agent_raw)
        if argv:
            exe = Path(argv[0]).name
            if exe not in _ALLOWED_WORKER_EXECUTABLES:
                raise ValueError(
                    f"{ctx} execution_agent must start with one of "
                    f"{sorted(_ALLOWED_WORKER_EXECUTABLES)!r}; got {exe!r}"
                )
        execution_agent = execution_agent_raw

    # tools
    tools_raw = entry.get("tools")
    tools: tuple[str, ...] | None = None
    if tools_raw is not None:
        tools = _coerce_single_tool_list(tools_raw, f"{ctx} tools")

    # brief_prepend / brief_prepend_path
    brief_prepend = _load_flavor_brief(entry, flavor_name, project_root)

    # quality_gates
    quality_gates_raw = entry.get("quality_gates")
    quality_gates: tuple[str, ...] | None = None
    if quality_gates_raw is not None:
        if not isinstance(quality_gates_raw, list):
            raise ValueError(f"{ctx} quality_gates must be a list of strings (use [] to skip)")
        for i, g in enumerate(quality_gates_raw):
            if not isinstance(g, str) or not g:
                raise ValueError(f"{ctx} quality_gates[{i}] must be a non-empty string")
        quality_gates = tuple(quality_gates_raw)

    # Native Workflow backend knobs (None = inherit global).
    agent_type_raw = entry.get("agent_type")
    if agent_type_raw is not None and not isinstance(agent_type_raw, str):
        raise ValueError(f"{ctx} agent_type must be a string")
    loop_mode_raw = entry.get("loop_mode")
    loop_mode = _validated_loop_mode(loop_mode_raw, ctx) if loop_mode_raw is not None else None
    max_iterations = _validated_positive_int(entry.get("max_iterations"), f"{ctx} max_iterations")
    max_turns = _validated_positive_int(entry.get("max_turns"), f"{ctx} max_turns")

    return FlavorOverride(
        execution_agent=execution_agent,
        tools=tools,
        brief_prepend=brief_prepend,
        quality_gates=quality_gates,
        agent_type=agent_type_raw,
        loop_mode=loop_mode,
        max_iterations=max_iterations,
        max_turns=max_turns,
    )


def _validated_positive_int(value: Any, ctx: str) -> int | None:
    """Coerce an optional positive-int config value; None passes through."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{ctx} must be an integer")
    if value < 1:
        raise ValueError(f"{ctx} must be >= 1; got {value}")
    return value


def _load_flavor_brief(
    entry: dict[str, Any],
    flavor_name: str,
    project_root: Path,
) -> str | None:
    """Resolve brief_prepend or brief_prepend_path for a flavor entry."""
    ctx = f"[milknado.flavor.{flavor_name}]"
    inline = entry.get("brief_prepend")
    path_value = entry.get("brief_prepend_path")
    if inline is not None and not isinstance(inline, str):
        raise ValueError(f"{ctx} brief_prepend must be a string")
    if inline is not None and path_value is not None:
        raise ValueError(f"{ctx} brief_prepend and brief_prepend_path are mutually exclusive")
    if inline is not None:
        text = str(inline).strip()
        return text or None
    if path_value is not None:
        # path_value may be a string or a list of strings.
        if isinstance(path_value, str):
            paths = [path_value]
        elif isinstance(path_value, list):
            paths = path_value
        else:
            raise ValueError(f"{ctx} brief_prepend_path must be a string or list of strings")
        parts: list[str] = []
        for p in paths:
            if not isinstance(p, str):
                raise ValueError(f"{ctx} brief_prepend_path entries must be strings")
            resolved = Path(p)
            if not resolved.is_absolute():
                resolved = project_root / resolved
            if not resolved.exists():
                raise FileNotFoundError(f"{ctx} brief_prepend_path does not exist: {resolved}")
            parts.append(resolved.read_text(encoding="utf-8").strip())
        text = "\n\n".join(p for p in parts if p)
        return text or None
    return None


def _coerce_single_tool_list(value: Any, ctx: str) -> tuple[str, ...]:
    """Validate a single-list tool entry (list of non-empty strings, at most one \"...\")."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{ctx} must be a list of strings, got {type(value).__name__}")
    items: list[str] = []
    sentinel_count = 0
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{ctx}[{i}] must be a non-empty string, got {item!r}")
        if item == "...":
            sentinel_count += 1
            if sentinel_count > 1:
                raise ValueError(f'{ctx} may contain at most one "..." sentinel, found multiple')
        items.append(item)
    return tuple(items)


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
        resolved = Path(str(path_value))
        if not resolved.is_absolute():
            resolved = project_root / resolved
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
