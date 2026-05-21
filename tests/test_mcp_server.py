from __future__ import annotations

import os
from pathlib import Path

import pytest

from milknado.mcp_server import (
    _open_graph,
    _project_root,
    milknado_todo_add,
    milknado_todo_next,
    milknado_todo_set_status,
    milknado_todo_tree,
)


def _call(tool, **kwargs):
    """Invoke a fastmcp-wrapped tool function with the underlying Python callable."""
    fn = getattr(tool, "fn", tool)
    return fn(**kwargs)


class TestMcpServer:
    def test_open_graph_creates_default_db_dir(self, tmp_path: Path) -> None:
        graph, _cfg = _open_graph(tmp_path)
        try:
            assert graph.get_all_nodes() == []
        finally:
            graph.close()
        assert (tmp_path / ".milknado" / "milknado.db").exists()

    def test_open_graph_and_add_node(self, tmp_path: Path) -> None:
        graph, _cfg = _open_graph(tmp_path)
        try:
            node = graph.add_node("first task")
            nodes = graph.get_all_nodes()
        finally:
            graph.close()
        assert node.id == 1
        assert len(nodes) == 1
        assert nodes[0].description == "first task"

    def test_project_root_precedence(self, tmp_path: Path, monkeypatch) -> None:
        explicit = tmp_path / "explicit"
        from_env = tmp_path / "from-env"
        monkeypatch.setenv("MILKNADO_PROJECT_ROOT", str(from_env))

        assert _project_root(str(explicit)) == explicit.resolve()
        assert _project_root(None) == from_env.resolve()

    def test_project_root_falls_back_to_cwd(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("MILKNADO_PROJECT_ROOT", raising=False)
        old = Path.cwd()
        os.chdir(tmp_path)
        try:
            assert _project_root(None) == tmp_path.resolve()
        finally:
            os.chdir(old)


class TestTodoTools:
    def test_add_and_tree_returns_forest(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal = _call(milknado_todo_add, description="ship v1", kind="goal", project_root=root)
        _call(
            milknado_todo_add,
            description="wire the schema",
            kind="task",
            parent_id=goal["id"],
            project_root=root,
        )
        _call(milknado_todo_add, description="write docs", kind="task", project_root=root)

        tree = _call(milknado_todo_tree, project_root=root)
        assert len(tree) == 2
        ship = next(n for n in tree if n["description"] == "ship v1")
        assert ship["kind"] == "goal"
        assert ship["status"] == "pending"
        assert [c["description"] for c in ship["children"]] == ["wire the schema"]
        assert ship["children"][0]["kind"] == "task"

    def test_tree_with_explicit_root_id(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal = _call(milknado_todo_add, description="g", kind="goal", project_root=root)
        _call(
            milknado_todo_add,
            description="t",
            kind="task",
            parent_id=goal["id"],
            project_root=root,
        )
        subtree = _call(milknado_todo_tree, project_root=root, root_id=goal["id"])
        assert len(subtree) == 1
        assert subtree[0]["id"] == goal["id"]
        assert len(subtree[0]["children"]) == 1

    def test_set_status_pending_to_done_collapses_through_running(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", kind="task", project_root=root)
        updated = _call(
            milknado_todo_set_status,
            node_id=task["id"],
            status="done",
            project_root=root,
        )
        assert updated["status"] == "done"

    def test_set_status_in_progress_then_done(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", kind="task", project_root=root)
        running = _call(
            milknado_todo_set_status,
            node_id=task["id"],
            status="in_progress",
            project_root=root,
        )
        assert running["status"] == "running"
        done = _call(
            milknado_todo_set_status,
            node_id=task["id"],
            status="done",
            project_root=root,
        )
        assert done["status"] == "done"

    def test_set_status_invalid_value_raises(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", kind="task", project_root=root)
        with pytest.raises(ValueError, match="invalid status"):
            _call(
                milknado_todo_set_status,
                node_id=task["id"],
                status="bogus",
                project_root=root,
            )

    def test_set_status_unknown_node_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(
                milknado_todo_set_status,
                node_id=999,
                status="done",
                project_root=str(tmp_path),
            )

    def test_add_invalid_kind_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid kind"):
            _call(
                milknado_todo_add,
                description="x",
                kind="nope",
                project_root=str(tmp_path),
            )

    def test_tree_unknown_root_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(milknado_todo_tree, project_root=str(tmp_path), root_id=42)

    def test_next_returns_ready_leaf_filtered_by_kind(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal = _call(milknado_todo_add, description="g", kind="goal", project_root=root)
        task = _call(
            milknado_todo_add,
            description="leaf",
            kind="task",
            parent_id=goal["id"],
            project_root=root,
        )
        nxt = _call(milknado_todo_next, kind="task", project_root=root)
        assert nxt is not None
        assert nxt["id"] == task["id"]

    def test_next_returns_none_when_nothing_ready(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="solo", kind="task", project_root=root)
        _call(
            milknado_todo_set_status,
            node_id=task["id"],
            status="done",
            project_root=root,
        )
        assert _call(milknado_todo_next, kind="task", project_root=root) is None

    def test_node_kind_persists_round_trip(self, tmp_path: Path) -> None:
        graph, _cfg = _open_graph(tmp_path)
        try:
            from milknado.domains.common import NodeKind

            node = graph.add_node("roadmap-node", kind=NodeKind.ROADMAP)
            reloaded = graph.get_node(node.id)
            assert reloaded is not None
            assert reloaded.kind == NodeKind.ROADMAP
        finally:
            graph.close()
