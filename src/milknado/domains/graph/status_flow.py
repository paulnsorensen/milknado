"""Graph status-flow helpers: state-machine stepping and bulk subtree transitions."""

from __future__ import annotations

from milknado.domains.common import VALID_TRANSITIONS, MikadoNode, NodeStatus
from milknado.domains.common.errors import InvalidTransition

VERIFY_ROLE = "verify"
CLAIM_ROLE = "claim"


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
