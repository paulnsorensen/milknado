from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

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
    execution_agent_raw = milknado.get("execution_agent")
    planning_agent = resolve_planning_agent_command(
        family,
        planning_agent=(
            str(planning_agent_raw)
            if planning_agent_raw is not None
            else None
        ),
    )
    execution_agent = resolve_execution_agent_command(
        family,
        execution_agent=(
            str(execution_agent_raw)
            if execution_agent_raw is not None
            else None
        ),
    )

    return MilknadoConfig(
        agent_family=family,
        planning_agent=planning_agent,
        execution_agent=execution_agent,
        quality_gates=tuple(milknado.get(
            "quality_gates",
            ["uv run pytest", "uv run ruff check", "uv run ty check"],
        )),
        worktree_pattern=milknado.get("worktree_pattern", "milknado-{node_id}-{slug}"),
        concurrency_limit=milknado.get("concurrency_limit", 4),
        project_root=project_root,
        db_path=project_root / Path(milknado.get("db_path", ".milknado/milknado.db")),
        plugins=tuple(milknado.get("plugins", [])),
        stall_threshold_seconds=int(milknado.get("stall_threshold_seconds", 300)),
        dispatch_max_retries=int(milknado.get("dispatch_max_retries", 2)),
        dispatch_backoff_seconds=float(milknado.get("dispatch_backoff_seconds", 5.0)),
        protected_branches=tuple(milknado.get("protected_branches", ["main", "master"])),
        completion_timeout_seconds=float(milknado.get("completion_timeout_seconds", 1800.0)),
        eta_sample_size=int(milknado.get("eta_sample_size", 10)),
    )


def save_config(config: MilknadoConfig, path: Path) -> None:
    lines = [
        "[milknado]",
        f'agent_family = "{config.agent_family}"',
        f'planning_agent = "{_escape_toml_string(config.planning_agent)}"',
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


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
