"""Immutable graph snapshot records shared by observers and app sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from milknado.domains.common.session import SessionView
from milknado.domains.common.types import MikadoEdge, MikadoNode
from milknado.domains.graph._goal_claims import GoalClaim
from milknado.domains.graph._run_persistence import NodeReviewRecord, RunRecord

_T = TypeVar("_T")
SnapshotState = Literal["loaded", "not_loaded", "not_stored", "not_retained"]


@dataclass(frozen=True, slots=True)
class SnapshotPage(Generic[_T]):
    items: tuple[_T, ...] | None
    offset: int
    limit: int
    total: int | None
    has_more: bool
    state: SnapshotState = "loaded"


@dataclass(frozen=True, slots=True)
class SnapshotValue(Generic[_T]):
    value: _T | None
    state: SnapshotState


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    nodes: tuple[MikadoNode, ...]
    edges: tuple[MikadoEdge, ...]
    root_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NodeSessionSnapshot:
    run_id: str
    session: SessionView | None
    state: SnapshotState


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
    goal_claim: SnapshotValue[GoalClaim]
    artifacts: SnapshotPage[ArtifactSnapshot]


@dataclass(frozen=True, slots=True)
class NodeDetailResponse:
    node_id: int
    request_generation: int
    detail: NodeDetailSnapshot | None

    def matches(self, node_id: int, request_generation: int) -> bool:
        return self.node_id == node_id and self.request_generation == request_generation
