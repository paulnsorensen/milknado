from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common.agent_argv import (
    normalize_preset,
    resolve_execution_agent_command,
)


@dataclass(frozen=True)
class MilknadoConfig:
    """Runtime config loaded from ``milknado.toml``.

    ``agent_preset`` selects built-in print-mode CLIs for planning + ralphify
    execution. Use ``custom`` and set ``agent_command`` for anything else.
    """

    agent_preset: str = "custom"
    agent_command: str = "claude"
    quality_gates: tuple[str, ...] = ("uv run pytest", "uv run ruff check", "uv run ty check")
    worktree_pattern: str = "milknado-{node_id}-{slug}"
    concurrency_limit: int = 4
    project_root: Path = Path(".")
    db_path: Path = Path(".milknado/milknado.db")
    plugins: tuple[str, ...] = ()


def default_config(project_root: Path) -> MilknadoConfig:
    from milknado.domains.common.agent_argv import EXECUTION_AGENT_BY_PRESET

    return MilknadoConfig(
        agent_preset="claude",
        agent_command=EXECUTION_AGENT_BY_PRESET["claude"],
        project_root=project_root,
        db_path=project_root / ".milknado" / "milknado.db",
    )


def load_config(path: Path) -> MilknadoConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    milknado = data.get("milknado", data)
    project_root = path.parent

    preset_raw = milknado.get("agent_preset", "custom")
    preset = normalize_preset(
        str(preset_raw) if preset_raw is not None else None,
    )
    has_cmd_key = "agent_command" in milknado
    file_cmd = milknado.get("agent_command", "claude")
    file_cmd_s = str(file_cmd) if file_cmd is not None else "claude"

    agent_command = resolve_execution_agent_command(
        preset,
        agent_command=file_cmd_s,
        agent_command_key_present=has_cmd_key,
    )

    return MilknadoConfig(
        agent_preset=preset,
        agent_command=agent_command,
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
        f'agent_preset = "{config.agent_preset}"',
        f'agent_command = "{_escape_toml_string(config.agent_command)}"',
        f"quality_gates = {list(config.quality_gates)}",
        f'worktree_pattern = "{config.worktree_pattern}"',
        f"concurrency_limit = {config.concurrency_limit}",
        f'db_path = "{config.db_path.relative_to(config.project_root)}"',
        f"plugins = {list(config.plugins)}",
    ]
    path.write_text("\n".join(lines) + "\n")


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
