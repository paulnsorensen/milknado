from __future__ import annotations

from milknado.domains.common import MikadoNode

from .graph import MikadoGraph


def walk_ancestors(graph: MikadoGraph, node_id: int) -> list[MikadoNode]:
    """Return nodes from `node_id` (leaf) up to the root (no parents).

    Follows the first parent encountered at each step, giving a deterministic
    single-path walk even in diamond-shaped graphs.  The returned list starts
    with the node at `node_id` and ends with the root.

    Raises:
        ValueError: if `node_id` does not exist in the graph.
    """
    start = graph.get_node(node_id)
    if start is None:
        raise ValueError(f"Node {node_id} not found")

    path: list[MikadoNode] = [start]
    visited: set[int] = {start.id}

    current = start
    while current.parent_id is not None:
        if current.parent_id in visited:
            break  # cycle guard (graph enforces acyclicity, but be safe)
        parent = graph.get_node(current.parent_id)
        if parent is None:
            break
        path.append(parent)
        visited.add(parent.id)
        current = parent

    return path
