from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from rich.style import Style
from textual.app import App, ComposeResult
from typing_extensions import override

from milknado.app.graph_panels import GraphTree
from milknado.app.graph_view import GraphTreeEntry
from milknado.domains.common import MikadoEdge, MikadoNode, NodeKind
from milknado.domains.graph import GraphSnapshot

_CREATED = datetime(2026, 9, 12, tzinfo=UTC)


class _TreeApp(App[None]):
    snapshot: GraphSnapshot
    selected_node_id: int

    def __init__(self, snapshot: GraphSnapshot, selected_node_id: int = 2) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.selected_node_id = selected_node_id

    @override
    def compose(self) -> ComposeResult:
        yield GraphTree()

    def on_mount(self) -> None:
        self.query_one(GraphTree).update_graph(self.snapshot, self.selected_node_id)


@pytest.mark.asyncio
async def test_graph_tree_preserves_reference_identity_on_graph_update() -> None:
    root = MikadoNode(1, "root", kind=NodeKind.GOAL, created_at=_CREATED)
    child = MikadoNode(2, "primary child", parent_id=1, created_at=_CREATED)
    other_root = MikadoNode(3, "other root", kind=NodeKind.GOAL, created_at=_CREATED)
    snapshot = GraphSnapshot(
        (root, child, other_root),
        (MikadoEdge(1, 2), MikadoEdge(3, 2)),
        (1, 3),
    )
    app = _TreeApp(snapshot)
    async with app.run_test(size=(80, 24)) as pilot:
        tree = app.query_one(GraphTree)
        reference = tree.root.children[1].children[0]
        _ = tree.move_cursor(reference, animate=False)
        await pilot.pause()
        assert tree.cursor_node is reference

        updated = GraphSnapshot(
            (replace(root, description="updated root"), child, other_root),
            snapshot.edges,
            snapshot.root_ids,
        )
        tree.update_graph(updated, 2)
        await pilot.pause()

        assert tree.cursor_node is not None
        assert tree.cursor_node.data == GraphTreeEntry(2, reference_parent_id=3)


@pytest.mark.asyncio
async def test_graph_tree_keeps_reference_when_selection_refreshes() -> None:
    root = MikadoNode(1, "root", kind=NodeKind.GOAL, created_at=_CREATED)
    child = MikadoNode(2, "primary child", parent_id=1, created_at=_CREATED)
    other_root = MikadoNode(3, "other root", kind=NodeKind.GOAL, created_at=_CREATED)
    snapshot = GraphSnapshot(
        (root, child, other_root),
        (MikadoEdge(1, 2), MikadoEdge(3, 2)),
        (1, 3),
    )
    app = _TreeApp(snapshot, selected_node_id=1)
    async with app.run_test(size=(80, 24)):
        tree = app.query_one(GraphTree)
        reference = tree.root.children[1].children[0]
        _ = tree.move_cursor(reference, animate=False)
        tree.update_graph(snapshot, 2)

        assert tree.cursor_node is reference


@pytest.mark.asyncio
async def test_graph_tree_ellipsizes_labels_to_native_width() -> None:
    root = MikadoNode(
        1,
        "A description that exceeds the narrow graph tree width",
        kind=NodeKind.GOAL,
        created_at=_CREATED,
    )
    snapshot = GraphSnapshot((root,), (), (1,))
    app = _TreeApp(snapshot)
    async with app.run_test(size=(40, 15)):
        tree = app.query_one(GraphTree)
        label = tree.render_label(tree.root.children[0], Style(), Style())

        assert label.plain.endswith("…")
        assert len(label.plain) <= 36


@pytest.mark.asyncio
async def test_graph_tree_does_not_move_viewport_on_unchanged_snapshot() -> None:
    root = MikadoNode(1, "root", kind=NodeKind.GOAL, created_at=_CREATED)
    children = tuple(
        MikadoNode(index, f"child {index}", parent_id=1, created_at=_CREATED)
        for index in range(2, 24)
    )
    snapshot = GraphSnapshot(
        (root, *children),
        tuple(MikadoEdge(1, child.id) for child in children),
        (1,),
    )
    app = _TreeApp(snapshot)
    async with app.run_test(size=(80, 8)) as pilot:
        tree = app.query_one(GraphTree)
        await pilot.pause()
        tree.scroll_to(y=8, animate=False)
        await pilot.pause()
        offset = tree.scroll_offset.y

        tree.update_graph(snapshot, 2)

        assert offset > 0
        assert tree.scroll_offset.y == offset


@pytest.mark.asyncio
async def test_graph_tree_keeps_visible_goal_collapsed_and_expands_new_goal() -> None:
    root = MikadoNode(1, "root", kind=NodeKind.GOAL, created_at=_CREATED)
    child = MikadoNode(2, "child", parent_id=1, created_at=_CREATED)
    snapshot = GraphSnapshot((root, child), (MikadoEdge(1, 2),), (1,))
    app = _TreeApp(snapshot)
    async with app.run_test(size=(80, 24)):
        tree = app.query_one(GraphTree)
        _ = tree.root.children[0].collapse()
        new_goal = MikadoNode(3, "new goal", kind=NodeKind.GOAL, created_at=_CREATED)

        tree.update_graph(GraphSnapshot((root, child, new_goal), snapshot.edges, (1, 3)), 2)

        assert not tree.root.children[0].is_expanded
        assert tree.root.children[1].is_expanded
