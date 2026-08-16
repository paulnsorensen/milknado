from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import msgspec
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
    FlavorTable,
    Gate,
    absolutize_global_flavor_paths,
    coerce_tool_list,
    normalize_flavor_table,
    normalize_gates,
    serialize_flavor_tables,
    serialize_gates,
    validate_loop_mode,
    validate_positive_int,
)
from milknado.domains.common.merge import deep_merge
from milknado.domains.common.paths import resolve_project_path, trust_global_path
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


class MilknadoConfig(msgspec.Struct, frozen=True, kw_only=True):
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
    worker_tools: dict[str, tuple[str, ...]] = msgspec.field(default_factory=dict)
    flavors: dict[str, FlavorOverride] = msgspec.field(default_factory=dict)
    flavor_registry: frozenset[str] = msgspec.field(default_factory=lambda: BUILTIN_FLAVORS)
    planning_prompt_prepend: str | None = None
    worker_brief_prepend: str | None = None
    commit_footer: str | None = None
    worker_agent_type: str = DEFAULT_WORKER_AGENT_TYPE
    loop_mode: str = DEFAULT_LOOP_MODE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_turns: int = DEFAULT_MAX_TURNS


_PROMPT_PREPEND_SLOTS: tuple[str, ...] = ("planning_prepend", "worker_brief_prepend")


class PromptsTable(msgspec.Struct, frozen=True, kw_only=True):
    """Schema for the ``[milknado.prompts]`` TOML table."""

    planning_prepend: str | None = None  # noqa: V107 - read via getattr over _PROMPT_PREPEND_SLOTS
    planning_prepend_path: str | None = None  # noqa: V107 - read via _PROMPT_PREPEND_SLOTS
    worker_brief_prepend: str | None = None
    worker_brief_prepend_path: str | None = None  # noqa: V107 - read via _PROMPT_PREPEND_SLOTS

    def __post_init__(self) -> None:
        for slot in _PROMPT_PREPEND_SLOTS:
            if getattr(self, slot) is not None and getattr(self, f"{slot}_path") is not None:
                raise ValueError(
                    f"[milknado.prompts] {slot} and {slot}_path are mutually exclusive"
                )

    def prepend(self, slot: str, project_root: Path) -> str | None:
        """Return the inline text for one slot, or load it from its ``_path`` file."""
        inline = getattr(self, slot)
        if inline:
            return inline.strip() or None
        path_value = getattr(self, f"{slot}_path")
        if not path_value:
            return None
        resolved = resolve_project_path(
            path_value,
            project_root,
            label=f"[milknado.prompts] {slot}_path",
        )
        if not resolved.exists():
            raise FileNotFoundError(f"[milknado.prompts] {slot}_path does not exist: {resolved}")
        return resolved.read_text(encoding="utf-8").strip() or None


class WorkerTable(msgspec.Struct, frozen=True, kw_only=True):
    """Schema for the ``[milknado.worker]`` TOML table."""

    tools: dict[str, tuple[str, ...]] = msgspec.field(default_factory=dict)


class _MilknadoInheritanceControls(msgspec.Struct, frozen=True, kw_only=True):
    inherit_global: bool = True


class _FlavorInheritanceControls(msgspec.Struct, frozen=True, kw_only=True):
    inherit: bool = True


class MilknadoSection(msgspec.Struct, frozen=True, kw_only=True):
    """Schema for the merged ``[milknado]`` TOML table, before derivation."""

    agent_family: str = "claude"
    planning_agent: str | None = None
    execution_agent: str | None = None
    planning_validation_hook: str | None = None
    plan_reviewer_agent: str | None = None
    plan_review_max_rounds: int = 3
    quality_gates: tuple[Gate, ...] | None = None
    worktree: bool = True
    worktree_pattern: str = "milknado-{node_id}-{slug}"
    concurrency_limit: int = 4
    db_path: str = ".milknado/milknado.db"
    plugins: tuple[str, ...] = ()
    stall_threshold_seconds: int = 300
    dispatch_max_retries: int = 2
    dispatch_backoff_seconds: float = 5.0
    protected_branches: tuple[str, ...] = ("main", "master")
    completion_timeout_seconds: float | None = None
    eta_sample_size: int = 10
    commit_footer: str | None = None
    worker_agent_type: str = DEFAULT_WORKER_AGENT_TYPE
    loop_mode: str = DEFAULT_LOOP_MODE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_turns: int = DEFAULT_MAX_TURNS
    prompts: PromptsTable = msgspec.field(default_factory=PromptsTable)
    worker: WorkerTable = msgspec.field(default_factory=WorkerTable)
    flavor: dict[str, FlavorTable] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_positive_int(self.plan_review_max_rounds, "[milknado] plan_review_max_rounds")
        validate_positive_int(self.max_iterations, "[milknado] max_iterations")
        validate_positive_int(self.max_turns, "[milknado] max_turns")
        validate_loop_mode(self.loop_mode)


T = TypeVar("T")


def _typed_convert(value: Any, model: type[T]) -> T:
    return msgspec.convert(value, type=model, strict=True)


def _validate_section_scalars(section: dict[str, Any]) -> None:
    if "worktree" in section and not isinstance(section["worktree"], bool):
        raise msgspec.ValidationError("[milknado] worktree must be a boolean")
    for key in ("commit_footer", "plan_reviewer_agent"):
        value = section.get(key)
        if value is not None and not isinstance(value, str):
            raise msgspec.ValidationError(f"[milknado] {key} must be a string")
    for key in ("worker_agent_type", "loop_mode"):
        if key in section and not isinstance(section[key], str):
            raise msgspec.ValidationError(f"[milknado] {key} must be a string")
    if "loop_mode" in section:
        validate_loop_mode(section["loop_mode"])
    for key in ("plan_review_max_rounds", "max_iterations", "max_turns"):
        if key in section:
            validate_positive_int(section[key], f"[milknado] {key}")


def _normalize_section(raw: Any) -> Any:
    if not isinstance(raw, dict):
        return raw
    normalized = dict(raw)
    family = str(normalized.get("agent_family", "claude")).strip().lower()
    if family not in DEFAULT_PLANNING_AGENT_BY_FAMILY:
        allowed = ", ".join(sorted(DEFAULT_PLANNING_AGENT_BY_FAMILY))
        raise msgspec.ValidationError(f"Invalid agent_family. Expected one of: {allowed}")
    normalized["agent_family"] = family
    _validate_section_scalars(normalized)
    for key in ("prompts", "worker", "flavor"):
        value = normalized.get(key)
        if value is not None and not isinstance(value, dict):
            raise msgspec.ValidationError(f"[milknado.{key}] must be a table")
    hook = normalized.get("planning_validation_hook")
    normalized["planning_validation_hook"] = str(hook).strip() if hook else None
    for key in ("prompts", "worker", "flavor"):
        if normalized.get(key) is None:
            normalized[key] = {}
    if "quality_gates" in normalized:
        normalized["quality_gates"] = normalize_gates(
            normalized["quality_gates"], "[milknado] quality_gates"
        )
    prompts = normalized.get("prompts")
    if isinstance(prompts, dict):
        for key, value in prompts.items():
            if value is not None and not isinstance(value, str):
                raise msgspec.ValidationError(f"[milknado.prompts] {key} must be a string")
    worker = normalized.get("worker")
    if isinstance(worker, dict) and "tools" in worker:
        worker = dict(worker)
        tools = worker["tools"]
        if not isinstance(tools, dict):
            raise msgspec.ValidationError("[milknado.worker.tools] must be a table")
        worker["tools"] = {
            family: coerce_tool_list(value, f"[milknado.worker.tools.{family}]")
            for family, value in tools.items()
        }
        normalized["worker"] = worker
    flavors = normalized.get("flavor")
    if isinstance(flavors, dict):
        for name, entry in flavors.items():
            if not isinstance(name, str) or not name:
                raise msgspec.ValidationError(
                    "[milknado.flavor] flavor keys must be non-empty strings"
                )
            if not isinstance(entry, dict):
                raise msgspec.ValidationError(f"[milknado.flavor.{name}] must be a table")
        normalized["flavor"] = {
            name: normalize_flavor_table(entry) for name, entry in flavors.items()
        }
    return normalized


def decode_milknado_section(raw: Any) -> MilknadoSection:
    """Validate an already-parsed TOML section through its typed boundary."""
    try:
        return _typed_convert(_normalize_section(raw), MilknadoSection)
    except msgspec.ValidationError:
        raise
    except ValueError as exc:
        raise msgspec.ValidationError(str(exc)) from exc


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
            absolutize_global_flavor_paths(global_raw, global_path.parent)
            _clear_prompts_alternate_keys(global_raw, local_raw)
            raw = _merge(raw, global_raw)
            record_origins(origins, global_raw, f"global:{global_path}")

    for flavor in replaced_flavors:
        raw.get("flavor", {}).pop(flavor, None)
        remove_origin_prefix(origins, ("flavor", flavor))
    raw = _merge(raw, local_raw)
    record_origins(origins, local_raw, f"local:{path}")
    return LoadedConfig(_build_config(raw, project_root=path.parent), raw, origins)


def _decode_inheritance_controls(raw: dict[str, Any]) -> _MilknadoInheritanceControls:
    if "inherit_global" in raw and not isinstance(raw["inherit_global"], bool):
        raise msgspec.ValidationError("[milknado] inherit_global must be a boolean")
    return _typed_convert(raw, _MilknadoInheritanceControls)


def _decode_flavor_inheritance_controls(raw: dict[str, Any]) -> _FlavorInheritanceControls:
    if "inherit" in raw and not isinstance(raw["inherit"], bool):
        raise msgspec.ValidationError("inherit must be a boolean")
    return _typed_convert(raw, _FlavorInheritanceControls)


def _extract_inheritance(raw: dict[str, Any], *, local: bool) -> tuple[bool, tuple[str, ...]]:
    """Validate and remove layer-control metadata before normal merging."""
    controls = _decode_inheritance_controls(raw)
    raw.pop("inherit_global", None)
    replaced: list[str] = []
    flavors = raw.get("flavor")
    if isinstance(flavors, dict):
        for name, flavor in flavors.items():
            if isinstance(flavor, dict) and "inherit" in flavor:
                flavor_controls = _decode_flavor_inheritance_controls(flavor)
                flavor.pop("inherit")
                if local and not flavor_controls.inherit:
                    replaced.append(name)
    return controls.inherit_global, tuple(replaced)


def save_config(config: MilknadoConfig, path: Path) -> None:
    milknado = _serialize_milknado_core(config)

    prompts = _serialize_prompts(config)
    if prompts:
        milknado["prompts"] = prompts

    tools = _serialize_worker_tools(config)
    if tools:
        milknado["worker"] = {"tools": tools}

    flavor_tables = serialize_flavor_tables(config.flavors)
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
        milknado["quality_gates"] = serialize_gates(config.quality_gates)
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
            Gate(command="uv run pytest"),
            Gate(command="uv run ruff check"),
            Gate(command="uv run ty check"),
        )
    if (project_root / "Cargo.toml").exists():
        return (
            Gate(command="cargo test"),
            Gate(command="cargo clippy -- -D warnings"),
        )
    if (project_root / "package.json").exists():
        return (Gate(command="npm test"),)
    if (project_root / "go.mod").exists():
        return (
            Gate(command="go test ./..."),
            Gate(command="go vet ./..."),
        )
    if (project_root / "project.godot").exists():
        return (Gate(command="godot --headless --quit", fail_on_stdout="SCRIPT ERROR|^ERROR:"),)
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


def _absolutize_global_prompt_paths(global_raw: dict[str, Any], base_dir: Path) -> None:
    """Resolve relative prompt ``_path`` values in the GLOBAL config against the
    global config directory.

    Prompt paths are otherwise resolved in ``_build_config`` against the local
    project root (``path.parent``) after the layered merge, so a relative global
    ``planning_prepend_path = "team.md"`` would wrongly look for the file next to
    the project's ``milknado.toml`` instead of next to the global config. Rewrite
    them to absolute here, before the merge, so the global base dir is honoured.
    Non-string values are left untouched for the schema to reject.
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


def _reject_omp_tool_translation(section: MilknadoSection, family: str) -> None:
    """omp has no tool-list-to-flag translation; only an explicit command works."""
    if family != "omp":
        return
    if section.worker.tools.get("omp") is not None:
        raise ValueError("[milknado.worker.tools.omp] is not supported; set execution_agent")
    if not section.execution_agent:
        raise ValueError("agent_family 'omp' requires an explicit execution_agent")
    if any(table.tools is not None for table in section.flavor.values()):
        raise ValueError("[milknado.flavor.*].tools is not supported for agent_family 'omp'")


def _resolve_agents(section: MilknadoSection, family: str) -> tuple[str, str]:
    """Derive the planning/execution commands from explicit values or family defaults."""
    planning_agent = resolve_planning_agent_command(
        family,
        planning_agent=section.planning_agent,
    )
    family_tools = section.worker.tools.get(family)
    execution_agent = resolve_execution_agent_command(
        family,
        execution_agent=section.execution_agent,
        tools=(
            resolve_worker_tools(family, list(family_tools)) if family_tools is not None else None
        ),
    )
    return planning_agent, execution_agent


def _build_config(raw: dict[str, Any], *, project_root: Path) -> MilknadoConfig:
    section = decode_milknado_section(raw)
    family = section.agent_family
    flavors = {
        name: table.to_override(name, project_root) for name, table in section.flavor.items()
    }
    _reject_omp_tool_translation(section, family)
    planning_agent, execution_agent = _resolve_agents(section, family)
    return MilknadoConfig(
        agent_family=family,
        planning_agent=planning_agent,
        planning_validation_hook=section.planning_validation_hook,
        plan_reviewer_agent=section.plan_reviewer_agent,
        plan_review_max_rounds=section.plan_review_max_rounds,
        execution_agent=execution_agent,
        quality_gates=section.quality_gates,
        worktree=section.worktree,
        worktree_pattern=section.worktree_pattern,
        concurrency_limit=section.concurrency_limit,
        project_root=project_root,
        db_path=resolve_project_path(section.db_path, project_root, label="db_path"),
        plugins=section.plugins,
        stall_threshold_seconds=section.stall_threshold_seconds,
        dispatch_max_retries=section.dispatch_max_retries,
        dispatch_backoff_seconds=section.dispatch_backoff_seconds,
        protected_branches=section.protected_branches,
        completion_timeout_seconds=section.completion_timeout_seconds,
        eta_sample_size=section.eta_sample_size,
        worker_tools=section.worker.tools,
        flavors=flavors,
        flavor_registry=BUILTIN_FLAVORS | flavors.keys(),
        planning_prompt_prepend=section.prompts.prepend("planning_prepend", project_root),
        worker_brief_prepend=section.prompts.prepend("worker_brief_prepend", project_root),
        commit_footer=section.commit_footer,
        worker_agent_type=section.worker_agent_type,
        loop_mode=section.loop_mode,
        max_iterations=section.max_iterations,
        max_turns=section.max_turns,
    )
