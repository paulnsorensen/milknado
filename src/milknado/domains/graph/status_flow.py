"""Graph status-flow helpers: state-machine stepping and bulk subtree transitions."""

from __future__ import annotations

import json

from milknado.domains.common import VALID_TRANSITIONS, MikadoNode, NodeStatus
from milknado.domains.common.errors import InvalidTransition

VERIFY_ROLE = "verify"
CLAIM_ROLE = "claim"


def latest_verify_ok(graph, run_id: str) -> bool:  # noqa: ANN001
    body = graph.latest_run_message(run_id, VERIFY_ROLE)
    if not body:
        return False
    try:
        return json.loads(body).get("ok") is True
    except (ValueError, TypeError, AttributeError):
        return False


def assert_done_verified(graph, node: MikadoNode) -> None:  # noqa: ANN001
    if node.run_id is None or graph.get_run(node.run_id) is None:
        return
    native = (
        graph.latest_run_message(node.run_id, CLAIM_ROLE) is not None
        or graph.latest_run_message(node.run_id, VERIFY_ROLE) is not None
    )
    if native and not latest_verify_ok(graph, node.run_id):
        raise ValueError(
            f"node {node.id} cannot be marked done: milknado_node_verify(run_id="
            f"{node.run_id!r}) has not returned ok=True. Run milknado_node_verify first."
        )


def todo_status_steps(current: NodeStatus, target: NodeStatus) -> list[NodeStatus]:
    """The ordered state-machine steps moving `current` to `target`.

    The facade exposes pending/in_progress/blocked/done, but the underlying
    state machine only allows BLOCKED -> PENDING and forbids PENDING -> DONE
    directly. Step through PENDING/RUNNING as needed so every reachable facade
    status is reachable. Setting a node to its current status yields no steps.
    """
    if current == target:
        return []
    steps: list[NodeStatus] = []
    if current == NodeStatus.BLOCKED and target in (NodeStatus.RUNNING, NodeStatus.DONE):
        steps.append(NodeStatus.PENDING)
        current = NodeStatus.PENDING
    if target == NodeStatus.DONE and current == NodeStatus.PENDING:
        steps.append(NodeStatus.RUNNING)
    steps.append(target)
    return steps


def validate_todo_status(node: MikadoNode, target: NodeStatus) -> None:
    """Raise InvalidTransition if moving `node` to `target` is illegal, without
    mutating anything. Walks the same step sequence `apply_todo_status` would
    take against VALID_TRANSITIONS so a bulk caller can pre-flight a whole
    subtree before any write commits."""
    state = node.status
    for step in todo_status_steps(node.status, target):
        if step not in VALID_TRANSITIONS.get(state, set()):
            raise InvalidTransition(
                node_id=node.id,
                current=state,
                target=step,
                valid_targets=tuple(VALID_TRANSITIONS.get(state, set())),
            )
        state = step


def apply_todo_status(graph, node: MikadoNode, target: NodeStatus) -> None:  # noqa: ANN001
    """Move a node to one of the todo facade's statuses.

    Steps through PENDING/RUNNING as needed (see `todo_status_steps`). Setting
    a node to its current status is a no-op. Reopening a DONE node is
    intentionally disallowed and still raises InvalidTransition.
    """
    if target == NodeStatus.DONE:
        assert_done_verified(graph, node)
    dispatch = {
        NodeStatus.PENDING: graph.mark_pending,
        NodeStatus.RUNNING: graph.mark_running,
        NodeStatus.DONE: graph.mark_done,
        NodeStatus.BLOCKED: graph.mark_blocked,
    }
    for step in todo_status_steps(node.status, target):
        dispatch[step](node.id)


def subtree_post_order(
    children_map: dict[int, list[MikadoNode]], root: MikadoNode
) -> list[MikadoNode]:
    """Nodes in root's subtree, children before parents, each visited once."""
    visited: set[int] = set()
    ordered: list[MikadoNode] = []

    def visit(node: MikadoNode) -> None:
        if node.id in visited:
            return
        visited.add(node.id)
        for child in children_map.get(node.id, []):
            visit(child)
        ordered.append(node)

    visit(root)
    return ordered
