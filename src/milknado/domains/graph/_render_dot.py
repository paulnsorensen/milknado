"""Pure DOT renderer for the live Mikado execution graph."""

from __future__ import annotations

from milknado.domains.common import MikadoNode, NodeKind, NodeStatus

_STATUS_STYLES = {
    NodeStatus.PENDING.value: ("#e2e8f0", "#475569", "#0f172a"),
    NodeStatus.RUNNING.value: ("#dbeafe", "#2563eb", "#172554"),
    NodeStatus.DONE.value: ("#dcfce7", "#16a34a", "#14532d"),
    NodeStatus.BLOCKED.value: ("#fef3c7", "#d97706", "#451a03"),
    NodeStatus.FAILED.value: ("#fee2e2", "#dc2626", "#450a0a"),
}

_ARCHIVED_FILL, _ARCHIVED_STROKE, _ARCHIVED_TEXT = "#f1f5f9", "#cbd5e1", "#94a3b8"

_SHAPE_BY_KIND = {
    NodeKind.ROADMAP: "box3d",
    NodeKind.GOAL: "folder",
    NodeKind.TASK: "box",
}


def _escape_label(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    for char in ("\n", "\r", "\t"):
        escaped = escaped.replace(char, " ")
    return escaped


def _node_line(node: MikadoNode) -> str:
    label = _escape_label(f"#{node.id} {node.description}")
    shape = _SHAPE_BY_KIND[node.kind]
    if node.archived_at is None:
        style = "filled"
        fill, stroke, text = _STATUS_STYLES[node.status.value]
    else:
        style = "filled,dashed"
        fill, stroke, text = _ARCHIVED_FILL, _ARCHIVED_STROKE, _ARCHIVED_TEXT
    return (
        f'    "{node.id}" [label="{label}", shape="{shape}", style="{style}", '
        f'fillcolor="{fill}", color="{stroke}", fontcolor="{text}"];'
    )


def render_dot(
    nodes: list[MikadoNode],
    children_map: dict[int, list[MikadoNode]],
) -> str:
    """Render a deterministic Graphviz DOT digraph of the live Mikado graph."""
    node_ids = {node.id for node in nodes}
    lines = [
        "digraph mikado {",
        '    rankdir="TB";',
        '    node [style="filled", fontname="sans-serif"];',
    ]
    lines.extend(_node_line(node) for node in sorted(nodes, key=lambda node: node.id))
    edges = sorted(
        {
            (parent_id, child.id)
            for parent_id, children in children_map.items()
            for child in children
            if parent_id in node_ids and child.id in node_ids
        }
    )
    lines.extend(f'    "{parent_id}" -> "{child_id}";' for parent_id, child_id in edges)
    lines.append("}")
    return "\n".join(lines) + "\n"
