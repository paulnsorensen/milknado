from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MilknadoConfig:
    agent_command: str = "claude"
    quality_gates: tuple[str, ...] = ("uv run pytest", "uv run ruff check", "uv run ty check")
    worktree_pattern: str = "milknado-{node_id}-{slug}"
    concurrency_limit: int = 4
    project_root: Path = Path(".")
    db_path: Path = Path(".milknado/milknado.db")
    plugins: tuple[str, ...] = ()


def default_config(project_root: Path) -> MilknadoConfig:
    return MilknadoConfig(
        project_root=project_root,
        db_path=project_root / ".milknado" / "milknado.db",
    )


def load_config(path: Path) -> MilknadoConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    milknado = data.get("milknado", data)
    project_root = path.parent

    return MilknadoConfig(
        agent_command=milknado.get("agent_command", "claude"),
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
        f'agent_command = "{config.agent_command}"',
        f"quality_gates = {list(config.quality_gates)}",
        f'worktree_pattern = "{config.worktree_pattern}"',
        f"concurrency_limit = {config.concurrency_limit}",
        f'db_path = "{config.db_path.relative_to(config.project_root)}"',
        f"plugins = {list(config.plugins)}",
    ]
    path.write_text("\n".join(lines) + "\n")
