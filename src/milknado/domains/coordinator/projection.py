from __future__ import annotations

import sqlite3
from typing import Literal, cast

import msgspec

from milknado.domains.common import MikadoEdge, MikadoNode
from milknado.domains.coordinator.journal import snapshot_control_history
from milknado.domains.coordinator.model import (
    ControlRecord,
    CoordinatorSession,
    EntityLink,
    ProviderBinding,
)
from milknado.domains.coordinator.persistence import (
    get_coordinator,
    links_for_session,
    provider_bindings_for_session,
)
from milknado.domains.coordinator.plans import PlanProposalRecord, list_proposals
from milknado.domains.graph import (
    ExecutionGroup,
    GoalReviewRecord,
    MikadoGraph,
    RunRecord,
    subtree_post_order,
)
from milknado.loop.sessions import runtime_capabilities


class CoordinatorStatus(msgspec.Struct, frozen=True):
    goal_id: int
    provider: str
    status: str
    recovery: str | None


def read_coordinator_status(conn: sqlite3.Connection) -> tuple[CoordinatorStatus, ...]:
    exists = cast(
        tuple[int] | None,
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'coordinator_sessions'"
        ).fetchone(),
    )
    if exists is None:
        return ()
    rows = cast(
        list[tuple[int, str, str, str | None]],
        conn.execute(
            "SELECT c.goal_id, c.provider, n.status, "
            + "(SELECT status FROM coordinator_events WHERE session_id = c.id "
            + "AND kind = 'recovery' ORDER BY seq DESC LIMIT 1) "
            + "FROM coordinator_sessions AS c JOIN nodes AS n ON n.id = c.goal_id "
            + "ORDER BY c.created_at DESC LIMIT 10"
        ).fetchall(),
    )
    return tuple(CoordinatorStatus(*row) for row in rows)


class ProviderTurnState(msgspec.Struct, frozen=True):
    provider_family: str  # noqa: V107
    provider_session_id: str
    turn_id: str
    status: str


class CoordinatorSnapshot(msgspec.Struct, frozen=True):
    session: CoordinatorSession
    goal: MikadoNode
    nodes: tuple[MikadoNode, ...]
    edges: tuple[MikadoEdge, ...]
    links: tuple[EntityLink, ...]
    groups: tuple[ExecutionGroup, ...]
    runs: tuple[RunRecord, ...]
    reviews: tuple[GoalReviewRecord, ...]
    proposals: tuple[PlanProposalRecord, ...]  # noqa: V107
    provider_bindings: tuple[ProviderBinding, ...]  # noqa: V107
    provider_turns: tuple[ProviderTurnState, ...]  # noqa: V107
    recovery: tuple[ControlRecord, ...]
    capability_floor: dict[str, str]  # noqa: V107
    native_actions: tuple[str, ...]
    unsupported_actions: tuple[str, ...]
    events: tuple[ControlRecord, ...]
    cursor: int


def _goal_nodes(graph: MikadoGraph, goal_id: int) -> tuple[MikadoNode, ...]:
    root = graph.get_node(goal_id)
    if root is None:
        raise ValueError("coordinator graph node does not exist")
    ordered = reversed(subtree_post_order(graph.get_children_map(), root))
    node_ids = dict.fromkeys(node.id for node in ordered)
    return tuple(graph.get_nodes(node_ids))


def _turns(conn: sqlite3.Connection, session_id: str) -> tuple[ProviderTurnState, ...]:
    rows = cast(
        list[tuple[str, str, str, str]],
        conn.execute(
            "SELECT provider_family, provider_session_id, turn_id, status "
            + "FROM coordinator_turn_events WHERE coordinator_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall(),
    )
    states = {
        (family, provider_id, turn_id): status for family, provider_id, turn_id, status in rows
    }
    return tuple(ProviderTurnState(*key, status) for key, status in states.items())


def read_coordinator_snapshot(
    graph: MikadoGraph, conn: sqlite3.Connection, session_id: str, cursor: int
) -> CoordinatorSnapshot:
    if cursor < 0:
        raise ValueError("cursor must not be negative")
    with graph.synchronization_lock:
        _ = conn.execute("BEGIN")
        try:
            return _project_snapshot(graph, conn, session_id, cursor)
        finally:
            conn.rollback()


def _project_snapshot(
    graph: MikadoGraph, conn: sqlite3.Connection, session_id: str, cursor: int
) -> CoordinatorSnapshot:
    session = get_coordinator(conn, session_id)
    if session is None:
        raise KeyError(session_id)
    nodes = _goal_nodes(graph, session.goal_id)
    node_ids = {node.id for node in nodes}
    edges = tuple(
        edge
        for edge in graph.get_graph_snapshot().edges
        if edge.parent_id in node_ids and edge.child_id in node_ids
    )
    links = links_for_session(conn, session_id)
    events, recovery, latest = snapshot_control_history(conn, session_id, cursor)
    capabilities = runtime_capabilities(cast(Literal["claude", "codex"], session.provider))
    return CoordinatorSnapshot(
        session=session,
        goal=nodes[0],
        nodes=nodes,
        edges=edges,
        links=links,
        groups=tuple(
            group
            for link in links
            if link.kind == "execution_group"
            if (group := graph.groups.get(link.entity_id)) is not None
        ),
        runs=tuple(
            run
            for link in links
            if link.kind == "run"
            if (run := graph.runs.get(link.entity_id)) is not None
        ),
        reviews=tuple(
            review
            for link in links
            if link.kind == "approval"
            if (review := graph.get_goal_review(int(link.entity_id))) is not None
        ),
        proposals=list_proposals(conn, session_id),
        provider_bindings=provider_bindings_for_session(conn, session_id),
        provider_turns=_turns(conn, session_id),
        recovery=recovery,
        capability_floor={str(name): str(support) for name, support in capabilities.floor.items()},
        native_actions=tuple(sorted(capabilities.native_actions)),
        unsupported_actions=tuple(sorted(capabilities.unsupported_actions)),
        events=events,
        cursor=latest,
    )
