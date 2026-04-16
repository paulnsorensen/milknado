from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common.agent_argv import (
    DEFAULT_EXECUTION_AGENT_BY_FAMILY,
    DEFAULT_PLANNING_AGENT_BY_FAMILY,
    resolve_planning_agent_command,
    resolve_execution_agent_command,
)


@dataclass(frozen=True)
class MilknadoConfig:
    """Runtime config loaded from ``milknado.toml``.

    ``agent_family`` selects default planning/execution model family commands.
    ``planning_agent`` and ``execution_agent`` can be overridden per project.
    """

    agent_family: str = "claude"
    planning_agent: str = "claude --model opus -p --dangerously-skip-permissions"
    execution_agent: str = "claude --model sonnet -p --dangerously-skip-permissions"
    quality_gates: tuple[str, ...] = ("uv run pytest", "uv run ruff check", "uv run ty check")
    worktree_pattern: str = "milknado-{node_id}-{slug}"
    concurrency_limit: int = 4
    project_root: Path = Path(".")
    db_path: Path = Path(".milknado/milknado.db")
    plugins: tuple[str, ...] = ()


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
    ]
    path.write_text("\n".join(lines) + "\n")


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
