from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from milknado.domains.common.agent_argv import (
    DEFAULT_PLANNING_AGENT_BY_FAMILY,
    resolve_execution_agent_command,
    resolve_planning_agent_command,
)


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
    pr_stack: bool = False


def default_config(project_root: Path) -> MilknadoConfig:
    return MilknadoConfig(
        agent_family="claude",
        planning_agent=resolve_planning_agent_command("claude"),
        execution_agent=resolve_execution_agent_command("claude"),
        project_root=project_root,
        db_path=project_root / ".milknado" / "milknado.db",
    )


def load_config(path: Path) -> MilknadoConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    milknado = data.get("milknado", data)
    project_root = path.parent

    family = str(milknado.get("agent_family", "claude")).strip().lower()
    if family not in DEFAULT_PLANNING_AGENT_BY_FAMILY:
        allowed = ", ".join(sorted(DEFAULT_PLANNING_AGENT_BY_FAMILY))
        raise ValueError(
            f"Invalid agent_family '{family}'. Expected one of: {allowed}",
        )
    planning_agent_raw = milknado.get("planning_agent")
    planning_validation_hook_raw = milknado.get("planning_validation_hook")
    execution_agent_raw = milknado.get("execution_agent")
    planning_agent = resolve_planning_agent_command(
        family,
        planning_agent=(str(planning_agent_raw) if planning_agent_raw is not None else None),
    )
    execution_agent = resolve_execution_agent_command(
        family,
        execution_agent=(str(execution_agent_raw) if execution_agent_raw is not None else None),
    )

    return MilknadoConfig(
        agent_family=family,
        planning_agent=planning_agent,
        planning_validation_hook=(
            str(planning_validation_hook_raw).strip() if planning_validation_hook_raw else None
        ),
        execution_agent=execution_agent,
        quality_gates=tuple(
            milknado.get(
                "quality_gates",
                ["uv run pytest", "uv run ruff check", "uv run ty check"],
            )
        ),
        worktree_pattern=milknado.get("worktree_pattern", "milknado-{node_id}-{slug}"),
        concurrency_limit=milknado.get("concurrency_limit", 4),
        project_root=project_root,
        db_path=_validated_db_path(project_root, milknado.get("db_path", ".milknado/milknado.db")),
        plugins=tuple(milknado.get("plugins", [])),
        stall_threshold_seconds=int(milknado.get("stall_threshold_seconds", 300)),
        dispatch_max_retries=int(milknado.get("dispatch_max_retries", 2)),
        dispatch_backoff_seconds=float(milknado.get("dispatch_backoff_seconds", 5.0)),
        protected_branches=tuple(milknado.get("protected_branches", ["main", "master"])),
        completion_timeout_seconds=float(milknado.get("completion_timeout_seconds", 1800.0)),
        eta_sample_size=int(milknado.get("eta_sample_size", 10)),
        pr_stack=bool(milknado.get("pr_stack", False)),
    )


def save_config(config: MilknadoConfig, path: Path) -> None:
    data = {
        "milknado": {
            "agent_family": config.agent_family,
            "planning_agent": config.planning_agent,
            "planning_validation_hook": config.planning_validation_hook or "",
            "execution_agent": config.execution_agent,
            "quality_gates": list(config.quality_gates),
            "worktree_pattern": config.worktree_pattern,
            "concurrency_limit": config.concurrency_limit,
            "db_path": str(config.db_path.relative_to(config.project_root)),
            "plugins": list(config.plugins),
            "stall_threshold_seconds": config.stall_threshold_seconds,
            "dispatch_max_retries": config.dispatch_max_retries,
            "dispatch_backoff_seconds": config.dispatch_backoff_seconds,
            "protected_branches": list(config.protected_branches),
            "completion_timeout_seconds": config.completion_timeout_seconds,
            "eta_sample_size": config.eta_sample_size,
            "pr_stack": config.pr_stack,
        }
    }
    path.write_bytes(tomli_w.dumps(data).encode())


def _escape_toml_string(value: str) -> str:
    """Escape backslashes and double-quotes for TOML inline string values."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _validated_db_path(project_root: Path, raw: str) -> Path:
    db_path = project_root / Path(raw)
    if not db_path.resolve().is_relative_to(project_root.resolve()):
        raise ValueError(f"db_path '{raw}' escapes project_root '{project_root}'")
    return db_path
