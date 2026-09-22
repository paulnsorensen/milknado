"""Native Textual graph tree controls shared by run and watch."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar, final

from rich.style import Style
from rich.text import Text
from textual.binding import Binding, BindingType
from textual.widgets import Tree
from textual.widgets.tree import TreeNode
from typing_extensions import override

from milknado.app.graph_view import (
    GraphTreeEntry,
    GraphTreeProjection,
    project_graph,
    tree_label,
)
from milknado.domains.graph import GraphSnapshot


@final
class GraphTree(Tree[GraphTreeEntry]):
    """Focusable native tree with stable node data and preserved view state."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("j", "cursor_down", show=False),
        Binding("k", "cursor_up", show=False),
    ]

    def __init__(self) -> None:
        super().__init__("Graph", id="graph-tree")
        self._projection: GraphTreeProjection | None = None
        self._snapshot: GraphSnapshot | None = None
        self._nodes_by_id: dict[int, TreeNode[GraphTreeEntry]] = {}
        self._nodes_by_key: dict[str, TreeNode[GraphTreeEntry]] = {}
        self._selected_node_id: int | None = None

    @staticmethod
    def _label_depth(node: TreeNode[GraphTreeEntry]) -> int:
        depth = 1
        parent = node.parent
        while parent is not None and parent.parent is not None:
            depth += 1
            parent = parent.parent
        return depth

    @override
    def render_label(
        self, node: TreeNode[GraphTreeEntry], base_style: Style, style: Style
    ) -> Text:
        label = super().render_label(node, base_style, style)
        depth = self._label_depth(node)
        width = self.size.width - (4 + (depth - 1) * 6)
        if node.allow_expand:
            width -= 2
        label.truncate(max(1, width), overflow="ellipsis")
        return label

    def update_graph(self, snapshot: GraphSnapshot, selected_node_id: int | None) -> None:
        if self._projection is not None and snapshot == self._snapshot:
            selected = self.cursor_node.data if self.cursor_node is not None else None
            if selected_node_id != self._selected_node_id and (
                selected is None or selected.node_id != selected_node_id
            ):
                self._select_node(selected_node_id)
            self._selected_node_id = selected_node_id
            return
        expanded = self._expanded_keys()
        known_keys = {
            data.key for node in self._walk(self.root) if (data := node.data) is not None
        }
        selected = self._selected_entry()
        scroll = self.scroll_offset
        root_expanded = self._projection is None or self.root.is_expanded
        projection = project_graph(snapshot)
        self.root.remove_children()
        _ = self.root.expand() if root_expanded else self.root.collapse()
        self._nodes_by_id = {}
        self._nodes_by_key = {}
        self._projection = projection
        self._snapshot = snapshot
        for entry in projection.roots:
            self._add_entry(self.root, entry, expanded, known_keys)
        self._selected_node_id = selected_node_id
        _ = self.call_after_refresh(self._restore_selection, selected, selected_node_id)
        _ = self.call_after_refresh(self.scroll_to, x=scroll.x, y=scroll.y, animate=False)

    def _add_entry(
        self,
        parent: TreeNode[GraphTreeEntry],
        entry: GraphTreeEntry,
        expanded: set[str],
        known_keys: set[str],
    ) -> None:
        assert self._projection is not None
        node = self._projection.nodes[entry.node_id]
        children = self._projection.children.get(entry, ())
        if entry.is_reference:
            tree_node = parent.add_leaf(
                Text(tree_label(entry, node), no_wrap=True, overflow="ellipsis"), entry
            )
        else:
            tree_node = parent.add(
                Text(tree_label(entry, node), no_wrap=True, overflow="ellipsis"),
                entry,
                expand=entry.key in expanded
                or (parent is self.root and entry.key not in known_keys),
                allow_expand=bool(children),
            )
        self._nodes_by_key[entry.key] = tree_node
        if not entry.is_reference:
            self._nodes_by_id[entry.node_id] = tree_node
        for child in children:
            self._add_entry(tree_node, child, expanded, known_keys)

    def _expanded_keys(self) -> set[str]:
        return {
            data.key
            for node in self._walk(self.root)
            if node.is_expanded and (data := node.data) is not None
        }

    def _selected_entry(self) -> GraphTreeEntry | None:
        node = self.cursor_node
        return node.data if node is not None else None

    @staticmethod
    def _walk(node: TreeNode[GraphTreeEntry]) -> Iterator[TreeNode[GraphTreeEntry]]:
        for child in node.children:
            yield child
            yield from GraphTree._walk(child)

    def _restore_selection(self, entry: GraphTreeEntry | None, node_id: int | None) -> None:
        if entry is None and self.cursor_node is not None and self.cursor_node.data is not None:
            return
        if not self._select_entry(entry):
            self._select_node(node_id)

    def _select_entry(self, entry: GraphTreeEntry | None) -> bool:
        node = self._nodes_by_key.get(entry.key) if entry is not None else None
        if node is None:
            return False
        _ = self.move_cursor(node, animate=False)
        return True

    def _select_node(self, node_id: int | None) -> None:
        node = self._nodes_by_id.get(node_id) if node_id is not None else None
        if node is not None:
            _ = self.move_cursor(node, animate=False)
