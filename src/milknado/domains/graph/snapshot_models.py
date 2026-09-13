"""Immutable graph snapshot records shared by observers and app sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from milknado.domains.common.session import SessionEvent, SessionView
from milknado.domains.common.types import MikadoEdge, MikadoNode
from milknado.domains.graph._goal_claims import GoalClaim
from milknado.domains.graph._run_persistence import NodeReviewRecord, RunRecord
from milknado.domains.graph.commands import CommandReceipt

_T = TypeVar("_T")
SnapshotState = Literal["loaded", "missing", "not_loaded", "not_stored"]


@dataclass(frozen=True, slots=True)
class SnapshotPage(Generic[_T]):
    items: tuple[_T, ...] | None
    offset: int
    limit: int
    total: int | None
    has_more: bool  # noqa: V107 - consumed by paged snapshot clients
    state: SnapshotState = "loaded"


@dataclass(frozen=True, slots=True)
class SnapshotValue(Generic[_T]):
    value: _T | None
    state: SnapshotState


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    nodes: tuple[MikadoNode, ...]
    edges: tuple[MikadoEdge, ...]
    root_ids: tuple[int, ...]  # noqa: V107 - consumed by graph tree clients


@dataclass(frozen=True, slots=True)
class NodeSessionSnapshot:
    run_id: str
    session: SessionView | None
    state: SnapshotState
    event_history: SnapshotPage[SessionEvent]  # noqa: V107 - consumed by detail clients


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    path: str
    content: SnapshotValue[str]


@dataclass(frozen=True, slots=True)
class NodeDetailSnapshot:
    node: MikadoNode
    description: str
    parent: MikadoNode | None
    children: SnapshotPage[MikadoNode]
    ancestors: SnapshotPage[MikadoNode]
    prerequisite_ids: SnapshotPage[int]
    dependent_ids: SnapshotPage[int]
    reverse_dependents: SnapshotPage[MikadoNode]
    owned_files: SnapshotPage[str]
    runs: SnapshotPage[RunRecord]
    reviews: SnapshotPage[NodeReviewRecord]
    sessions: SnapshotPage[NodeSessionSnapshot]
    receipts: SnapshotPage[CommandReceipt]
    goal_claim: SnapshotValue[GoalClaim]  # noqa: V107 - consumed by detail clients
    artifacts: SnapshotPage[ArtifactSnapshot]


@dataclass(frozen=True, slots=True)
class NodeDetailResponse:
    node_id: int
    request_generation: int
    detail: NodeDetailSnapshot | None

    def matches(self, node_id: int, request_generation: int) -> bool:
        return self.node_id == node_id and self.request_generation == request_generation
