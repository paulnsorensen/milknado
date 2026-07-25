"""Graph status-flow helpers: state-machine stepping and bulk subtree transitions."""

from __future__ import annotations

import json

from milknado.domains.common import VALID_TRANSITIONS, MikadoNode, NodeStatus
from milknado.domains.common.errors import InvalidTransition

VERIFY_ROLE = "verify"
CLAIM_ROLE = "claim"
VERIFY_REQUIRED_MESSAGE = (
    "node {node_id} cannot be marked done: milknado_node_verify(run_id={run_id!r}) "
    "has not returned ok=True. Run milknado_node_verify first."
)


def latest_verify_ok(graph, run_id: str) -> bool:  # noqa: ANN001
    """Accept only a JSON object with an explicit boolean ok=true."""
    body = graph.latest_run_message(run_id, VERIFY_ROLE)
    if not body:
        return False
    try:
        deposit = json.loads(body)
    except (ValueError, TypeError):
        return False
    return isinstance(deposit, dict) and deposit.get("ok") is True


def assert_done_verified(graph, node: MikadoNode) -> None:  # noqa: ANN001
    if node.run_id is None or graph.get_run(node.run_id) is None:
        return
    native = (
        graph.latest_run_message(node.run_id, CLAIM_ROLE) is not None
        or graph.latest_run_message(node.run_id, VERIFY_ROLE) is not None
    )
    if native and not latest_verify_ok(graph, node.run_id):
        raise ValueError(VERIFY_REQUIRED_MESSAGE.format(node_id=node.id, run_id=node.run_id))


def todo_status_steps(current: NodeStatus, target: NodeStatus) -> list[NodeStatus]:
    """Return state-machine steps for one todo facade request."""
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
    """Preflight every state-machine step without mutating anything."""
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
    """Apply a preflighted facade transition through the graph's atomic operation."""
    graph.set_todo_status(node.id, target)


def subtree_post_order(
    children_map: dict[int, list[MikadoNode]], root: MikadoNode
) -> list[MikadoNode]:
    """Return descendants before parents, deduplicated and safe for cycles/depth."""
    visited: set[int] = {root.id}
    ordered: list[MikadoNode] = []
    stack: list[tuple[MikadoNode, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            ordered.append(node)
            continue
        stack.append((node, True))
        for child in reversed(children_map.get(node.id, [])):
            if child.id not in visited:
                visited.add(child.id)
                stack.append((child, False))
    return ordered
