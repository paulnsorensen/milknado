"""Pure unit tests for the DOT renderer of the live Mikado graph."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from milknado.domains.common import MikadoNode, NodeKind, NodeStatus
from milknado.domains.graph import render_dot


def _node(
    node_id: int,
    description: str,
    *,
    kind: NodeKind = NodeKind.TASK,
    status: NodeStatus = NodeStatus.PENDING,
) -> MikadoNode:
    return MikadoNode(id=node_id, description=description, status=status, kind=kind)


def test_header_lines_and_default_node_style() -> None:
    dot = render_dot([], {})
    lines = dot.splitlines()
    assert lines[0] == "digraph mikado {"
    assert lines[1] == '    rankdir="TB";'
    assert lines[2] == '    node [style="filled", fontname="sans-serif"];'


def test_done_goal_node_renders_folder_shape_and_done_palette() -> None:
    node = _node(1, "ship it", kind=NodeKind.GOAL, status=NodeStatus.DONE)
    dot = render_dot([node], {})
    assert (
        '    "1" [label="#1 ship it", shape="folder", style="filled", '
        'fillcolor="#dcfce7", color="#16a34a", fontcolor="#14532d"];'
    ) in dot


def test_roadmap_and_task_shapes() -> None:
    roadmap = _node(1, "r", kind=NodeKind.ROADMAP)
    task = _node(2, "t", kind=NodeKind.TASK)
    dot = render_dot([roadmap, task], {})
    assert 'shape="box3d"' in dot
    assert 'shape="box"' in dot


def test_failed_and_running_status_colors() -> None:
    failed = _node(1, "f", status=NodeStatus.FAILED)
    running = _node(2, "r", status=NodeStatus.RUNNING)
    dot = render_dot([failed, running], {})
    assert 'fillcolor="#fee2e2"' in dot
    assert 'fillcolor="#dbeafe"' in dot


def test_archived_node_uses_dashed_dimmed_style_regardless_of_status() -> None:
    node = dataclasses.replace(
        _node(1, "old", status=NodeStatus.DONE), archived_at=datetime.now(UTC)
    )
    dot = render_dot([node], {})
    assert (
        '    "1" [label="#1 old", shape="box", style="filled,dashed", '
        'fillcolor="#f1f5f9", color="#cbd5e1", fontcolor="#94a3b8"];'
    ) in dot


def test_edges_rendered_sorted_and_deterministic() -> None:
    parent = _node(1, "p")
    child_b = _node(3, "b")
    child_a = _node(2, "a")
    nodes = [parent, child_a, child_b]
    children_map = {1: [child_b, child_a]}
    dot = render_dot(nodes, children_map)
    assert '    "1" -> "2";\n    "1" -> "3";' in dot
    assert dot == render_dot(nodes, children_map)


def test_edge_dropped_when_endpoint_missing_from_node_set() -> None:
    parent = _node(1, "p")
    dangling_child = _node(2, "c")
    dot = render_dot([parent], {1: [dangling_child]})
    assert "->" not in dot


def test_label_escapes_quotes_and_collapses_newlines() -> None:
    node = _node(1, 'say "hi"\nthere')
    dot = render_dot([node], {})
    assert 'label="#1 say \\"hi\\" there"' in dot


def test_label_doubles_backslashes_before_quote_escaping() -> None:
    node = _node(1, r"C:\node")
    dot = render_dot([node], {})
    assert r'label="#1 C:\\node"' in dot


def test_label_collapses_carriage_returns_and_tabs() -> None:
    node = _node(1, "a\r\nb\tc")
    dot = render_dot([node], {})
    assert "\r" not in dot
    assert "\t" not in dot
    assert 'label="#1 a  b c"' in dot


def test_empty_graph_is_a_valid_digraph() -> None:
    dot = render_dot([], {})
    assert dot.startswith("digraph mikado {\n")
    assert dot.endswith("}\n")
