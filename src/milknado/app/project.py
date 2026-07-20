"""Application layer — project open: root resolution and graph/session opening."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import (
    MilknadoConfig,
    PluginHook,
    default_config,
    load_config,
)
from milknado.domains.graph import MikadoGraph
from milknado.plugins import discover_entry_point_plugins, load_plugins

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenProject:
    graph: MikadoGraph
    config: MilknadoConfig
    plugins: tuple[PluginHook, ...]


def load_project_config(root: Path) -> MilknadoConfig:
    root = root.resolve()
    config_path = root / "milknado.toml"
    return load_config(config_path) if config_path.exists() else default_config(root)


def load_project_plugins(config: MilknadoConfig) -> tuple[PluginHook, ...]:
    return (*load_plugins(config.plugins), *discover_entry_point_plugins())


def open_project_graph(
    config: MilknadoConfig,
    plugins: tuple[PluginHook, ...] = (),
) -> MikadoGraph:
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    return MikadoGraph(config.db_path, plugins=plugins)


def open_project(root: Path, config: MilknadoConfig | None = None) -> OpenProject:
    config = config or load_project_config(root)
    plugins = load_project_plugins(config)
    return OpenProject(open_project_graph(config, plugins), config, plugins)


def _bounded(text: str, limit: int = 500) -> str:
    return text.strip()[:limit]


def _worktree_main_checkout(cwd: Path) -> Path | None:
    """If cwd is a linked git worktree, return the main checkout root; else None."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError) as exc:
        _logger.warning(
            "git root discovery failed cwd=%s error=%s detail=%s",
            cwd,
            type(exc).__name__,
            _bounded(str(exc)),
        )
        return None
    if result.returncode != 0:
        _logger.warning(
            "git root discovery failed cwd=%s returncode=%s stderr=%s",
            cwd,
            result.returncode,
            _bounded(result.stderr),
        )
        return None
    common_dir = result.stdout.strip()
    # An absolute path means we're in a linked worktree.
    if not os.path.isabs(common_dir):
        return None  # relative == main checkout (e.g. ".git") — no hop needed
    return Path(common_dir).parent.resolve()


def resolve_project_root(explicit: str | None) -> Path:
    requested = Path(explicit).expanduser().resolve() if explicit and explicit.strip() else None
    injected = os.environ.get("MILKNADO_PROJECT_ROOT", "").strip()
    worker = any(
        os.environ.get(name, "").strip() for name in ("MILKNADO_NODE_ID", "MILKNADO_RUN_ID")
    )
    if worker:
        if not injected:
            raise ValueError("worker identity requires MILKNADO_PROJECT_ROOT")
        canonical = Path(injected).expanduser().resolve()
        if requested is not None and requested != canonical:
            raise ValueError(
                f"worker project root {requested} does not match injected root {canonical}"
            )
        return canonical
    if requested is not None:
        return requested
    if injected:
        return Path(injected).expanduser().resolve()
    cwd = Path.cwd().resolve()
    main = _worktree_main_checkout(cwd)
    return main if main is not None else cwd


def require_worker_run(run_id: str) -> None:
    injected = os.environ.get("MILKNADO_RUN_ID", "").strip()
    if injected and run_id != injected:
        raise ValueError(f"worker run {run_id!r} does not match injected run {injected!r}")


def open_graph(root: Path):  # noqa: ANN201
    project = open_project(root)
    return project.graph, project.config
