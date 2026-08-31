"""Shared FastMCP instance and project-root/graph helpers for the MCP tools.

Lives in its own module so the tool modules (`mcp.server`, `mcp.todo`) can both
register against the same `mcp` instance without importing each other — which
would form an import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, NotRequired, cast

from fastmcp import FastMCP
from typing_extensions import TypedDict

from milknado.app.project import (
    open_graph as _open_graph,
)
from milknado.app.project import (
    require_worker_run,
    resolve_project_root,
)
from milknado.domains.common import MilknadoConfig, NodeKind, NodeStatus
from milknado.domains.graph import MikadoGraph

__all__ = [
    "Flavor",
    "Kind",
    "RunDict",
    "Response",
    "BatchPlanResponse",
    "GraphNodeSummary",
    "GraphSummaryResponse",
    "NodeSummary",
    "PlanApplyResponse",
    "TodoStatus",
    "build_run_dict",
    "mcp",
    "parse_flavor",
    "parse_kind",
    "parse_todo_status",
    "open_graph",
    "require_worker_run",
    "resolve_project_root",
]


Kind = Literal["roadmap", "goal", "task"]
Flavor = str  # validated at runtime against BUILTIN_FLAVORS ∪ TOML-declared names (ADR-004)
TodoStatus = Literal["pending", "in_progress", "blocked", "done"]
Response = dict[str, object]


class NodeSummary(TypedDict):
    id: int
    kind: str
    status: str
    description: str
    flavor: NotRequired[str]
    children: NotRequired[list[NodeSummary]]


class GraphNodeSummary(TypedDict):
    id: int
    status: str
    description: str


class GraphSummaryResponse(TypedDict):
    nodes: list[GraphNodeSummary]


class BatchSummary(TypedDict):
    index: int
    change_ids: list[str]
    depends_on: list[int]
    oversized: bool


class SpreadSummary(TypedDict):
    symbol: dict[str, str]
    spread: int


class BatchPlanResponse(TypedDict):
    batches: list[BatchSummary]
    spread_report: list[SpreadSummary]
    solver_status: str


class PlanApplyResponse(TypedDict):
    nodes_created: list[int]
    graph_summary: GraphSummaryResponse


def open_graph(root: Path) -> tuple[MikadoGraph, MilknadoConfig]:
    return _open_graph(root)


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
    if isinstance(state, dict):
        source = cast(Mapping[str, object], state)
    else:
        source = cast(Mapping[str, object], vars(state))
    return cast(
        RunDict,
        cast(object, {key: source.get(key) for key in RunDict.__annotations__}),
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


_TODO_STATUS_MAP = {
    "pending": NodeStatus.PENDING,
    "in_progress": NodeStatus.RUNNING,
    "blocked": NodeStatus.BLOCKED,
    "done": NodeStatus.DONE,
}


def parse_kind(value: str) -> NodeKind:
    try:
        return NodeKind(value)
    except ValueError as exc:
        valid = sorted(k.value for k in NodeKind)
        raise ValueError(f"invalid kind {value!r}; expected one of {valid}") from exc


def parse_todo_status(value: str) -> NodeStatus:
    if value not in _TODO_STATUS_MAP:
        raise ValueError(f"invalid status {value!r}; expected one of {sorted(_TODO_STATUS_MAP)}")
    return _TODO_STATUS_MAP[value]


def parse_flavor(value: str, registry: frozenset[str]) -> str:
    if value not in registry:
        raise ValueError(f"invalid flavor {value!r}; expected one of {sorted(registry)}")
    return value


_parse_kind = parse_kind
_parse_todo_status = parse_todo_status
_parse_flavor = parse_flavor
