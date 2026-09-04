"""Application layer — project open: root resolution and graph/session opening.

Hosts the project-open policy the CLI and MCP entry surfaces both invoke: turning
a project_root argument (explicit, injected, or cwd) into a canonical path, and
opening the graph/config session. Entry modules import these so "project open"
routes through ``milknado.app.*`` rather than living inline at the boundary.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common import MilknadoConfig
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)


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
    # An absolute path means we're in a linked worktree: the common dir lives
    # inside the main checkout's .git. The main checkout root is its parent
    # (common_dir ends in .git, so parent is the main checkout).
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


def open_graph(root: Path) -> tuple[MikadoGraph, MilknadoConfig]:
    from milknado.project import open_project

    project = open_project(root)
    return project.graph, project.config
