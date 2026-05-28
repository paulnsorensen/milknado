from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from milknado.domains.common.agent_argv import (
    DEFAULT_PLANNING_AGENT_BY_FAMILY,
    resolve_execution_agent_command,
    resolve_planning_agent_command,
    resolve_worker_tools,
)

_logger = logging.getLogger(__name__)

# Keys that only make sense per-project. If set in the global config they are
# ignored with a warning rather than crashing.
_LOCAL_ONLY_KEYS = ("project_root", "db_path", "plugins")


@dataclass(frozen=True)
class WorkerToolsOverride:
    """Per-family worker tool overlay loaded from TOML.

    Semantics, in order:
      - if ``allow`` is set, it replaces the family default entirely
      - else the family default is concatenated with ``extend``
      - ``deny`` is removed from the final list (always last)
    """

    allow: tuple[str, ...] | None = None
    extend: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()


@dataclass(frozen=True)
class MilknadoConfig:
    """Runtime config loaded from ``milknado.toml``."""

    agent_family: str = "claude"
    planning_agent: str = "claude --model opus -p --dangerously-skip-permissions"
    planning_validation_hook: str | None = None
    execution_agent: str = "claude --model sonnet -p --dangerously-skip-permissions"
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
    worker_tools: dict[str, WorkerToolsOverride] = field(default_factory=dict)
    planning_prompt_prepend: str | None = None
    worker_brief_prepend: str | None = None


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
    if include_global:
        g = global_config_path()
        if g.exists():
            global_raw = _read_milknado_section(g)
            _warn_local_only_keys(global_raw, g)
            for k in _LOCAL_ONLY_KEYS:
                global_raw.pop(k, None)
            raw = _merge(raw, global_raw)
    raw = _merge(raw, _read_milknado_section(path))
    return _build_config(raw, project_root=path.parent)


def save_config(config: MilknadoConfig, path: Path) -> None:
    lines = [
        "[milknado]",
        f'agent_family = "{config.agent_family}"',
        f'planning_agent = "{_escape_toml_string(config.planning_agent)}"',
        (
            "planning_validation_hook = "
            f'"{_escape_toml_string(config.planning_validation_hook or "")}"'
        ),
        f'execution_agent = "{_escape_toml_string(config.execution_agent)}"',
        f"quality_gates = {list(config.quality_gates)}",
        f'worktree_pattern = "{config.worktree_pattern}"',
        f"concurrency_limit = {config.concurrency_limit}",
        f'db_path = "{config.db_path.relative_to(config.project_root)}"',
        f"plugins = {list(config.plugins)}",
        f"stall_threshold_seconds = {config.stall_threshold_seconds}",
        f"dispatch_max_retries = {config.dispatch_max_retries}",
        f"dispatch_backoff_seconds = {config.dispatch_backoff_seconds}",
        f"protected_branches = {list(config.protected_branches)}",
        f"completion_timeout_seconds = {config.completion_timeout_seconds}",
        f"eta_sample_size = {config.eta_sample_size}",
    ]
    path.write_text("\n".join(lines) + "\n")


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
    # overlay. An explicit `execution_agent` in TOML wins outright — it's the
    # opt-out for users who want full control of the worker invocation.
    family_override = worker_tools.get(family)
    if family_override is not None:
        tools_for_family: tuple[str, ...] | None = resolve_worker_tools(
            family,
            allow=family_override.allow,
            extend=family_override.extend,
            deny=family_override.deny,
        )
    else:
        tools_for_family = None
    execution_agent = resolve_execution_agent_command(
        family,
        execution_agent=(str(execution_agent_raw) if execution_agent_raw is not None else None),
        tools=tools_for_family,
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
        db_path=project_root / Path(raw.get("db_path", ".milknado/milknado.db")),
        plugins=tuple(raw.get("plugins", [])),
        stall_threshold_seconds=int(raw.get("stall_threshold_seconds", 300)),
        dispatch_max_retries=int(raw.get("dispatch_max_retries", 2)),
        dispatch_backoff_seconds=float(raw.get("dispatch_backoff_seconds", 5.0)),
        protected_branches=tuple(raw.get("protected_branches", ["main", "master"])),
        completion_timeout_seconds=float(raw.get("completion_timeout_seconds", 1800.0)),
        eta_sample_size=int(raw.get("eta_sample_size", 10)),
        worker_tools=worker_tools,
        planning_prompt_prepend=planning_prompt_prepend,
        worker_brief_prepend=worker_brief_prepend,
    )


def _parse_worker_tools(worker_raw: Any) -> dict[str, WorkerToolsOverride]:
    if worker_raw is None:
        return {}
    if not isinstance(worker_raw, dict):
        raise ValueError("[milknado.worker] must be a table")
    tools_raw = worker_raw.get("tools")
    if tools_raw is None:
        return {}
    if not isinstance(tools_raw, dict):
        raise ValueError("[milknado.worker.tools] must be a table")
    out: dict[str, WorkerToolsOverride] = {}
    for family, override in tools_raw.items():
        if not isinstance(override, dict):
            raise ValueError(
                f"[milknado.worker.tools.{family}] must be a table with allow/extend/deny"
            )
        unknown = set(override) - {"allow", "extend", "deny"}
        if unknown:
            raise ValueError(f"[milknado.worker.tools.{family}] unknown keys: {sorted(unknown)}")
        allow = override.get("allow")
        extend = override.get("extend") or ()
        deny = override.get("deny") or ()
        out[family] = WorkerToolsOverride(
            allow=tuple(allow) if allow is not None else None,
            extend=tuple(extend),
            deny=tuple(deny),
        )
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
    if inline and path_value:
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


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
