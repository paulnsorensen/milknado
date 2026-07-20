"""Shared FastMCP instance and project-root/graph helpers for the MCP tools.

Lives in its own module so the tool modules (`mcp_server`, `mcp_todo`) can both
register against the same `mcp` instance without importing each other — which
would form an import cycle.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, TypedDict

from fastmcp import FastMCP

from milknado.domains.common import NodeKind, NodeStatus

_logger = logging.getLogger(__name__)


Kind = Literal["roadmap", "goal", "task"]
Flavor = str  # validated at runtime against BUILTIN_FLAVORS ∪ TOML-declared names (ADR-004)
TodoStatus = Literal["pending", "in_progress", "blocked", "done"]


class RunDict(TypedDict):
    """Canonical superset returned by run start, poll, list, and cancel tools."""

    run_id: str | None
    node_id: int | None
    status: str | None
    exit_code: int | None
    timed_out: bool | None
    error: str | None
    detail: str | None
    rebased: bool | None
    pid: int | None
    log_path: str | None
    summary: str | None
    result: str | None
    worktree_preserved: str | None


def build_run_dict(state: object) -> RunDict:
    source = state if isinstance(state, dict) else vars(state)
    return RunDict(
        run_id=source.get("run_id"),
        node_id=source.get("node_id"),
        status=source.get("status"),
        exit_code=source.get("exit_code"),
        timed_out=source.get("timed_out"),
        error=source.get("error"),
        detail=source.get("detail"),
        rebased=source.get("rebased"),
        pid=source.get("pid"),
        log_path=source.get("log_path"),
        summary=source.get("summary"),
        result=source.get("result"),
        worktree_preserved=source.get("worktree_preserved"),
    )


mcp = FastMCP(
    "Milknado",
    instructions=(
        "Mikado graph execution tools: graph CRUD, batch planning, worker dispatch, "
        "detached ralph runs, run polling/cancel, and roadmap import/export. "
        "Set MILKNADO_PROJECT_ROOT or pass project_root to target a repo.\n\n"
        "Ephemeral cloud containers may be reclaimed on timeout. Before going idle, "
        "persist in-progress work to your branch; if you need run state, force-add "
        "`.milknado/milknado.db` plus its `-wal`/`-shm` sidecars, then remove "
        "ignored state (`.milknado/`, `*.db`, `ralphs/`, `.claude/`, `.context/`, "
        "`.venv/`, etc.) before opening the final PR."
    ),
)


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


def open_graph(root: Path):
    from milknado.project import open_project

    project = open_project(root)
    return project.graph, project.config


_TODO_STATUS_MAP = {
    "pending": NodeStatus.PENDING,
    "in_progress": NodeStatus.RUNNING,
    "blocked": NodeStatus.BLOCKED,
    "done": NodeStatus.DONE,
}


def _parse_kind(value: str) -> NodeKind:
    try:
        return NodeKind(value)
    except ValueError as exc:
        valid = sorted(k.value for k in NodeKind)
        raise ValueError(f"invalid kind {value!r}; expected one of {valid}") from exc


def _parse_todo_status(value: str) -> NodeStatus:
    if value not in _TODO_STATUS_MAP:
        raise ValueError(f"invalid status {value!r}; expected one of {sorted(_TODO_STATUS_MAP)}")
    return _TODO_STATUS_MAP[value]


def _parse_flavor(value: str, registry: frozenset[str]) -> str:
    if value not in registry:
        raise ValueError(f"invalid flavor {value!r}; expected one of {sorted(registry)}")
    return value
