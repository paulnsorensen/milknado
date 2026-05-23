from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from milknado.mcp_server import open_graph, resolve_project_root
from milknado.mcp_todo import (
    milknado_todo_add,
    milknado_todo_brief,
    milknado_todo_next,
    milknado_todo_run,
    milknado_todo_run_poll,
    milknado_todo_run_start,
    milknado_todo_set_status,
    milknado_todo_tree,
)


def _wait_for_terminal(run_id: str, project_root: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = _call(milknado_todo_run_poll, run_id=run_id, project_root=project_root)
        if last["status"] in ("done", "failed"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish; last state={last}")


def _call(tool, **kwargs):
    """Invoke a fastmcp-wrapped tool function with the underlying Python callable."""
    fn = getattr(tool, "fn", tool)
    return fn(**kwargs)


class TestMcpServer:
    def test_open_graph_creates_default_db_dir(self, tmp_path: Path) -> None:
        graph, _cfg = open_graph(tmp_path)
        try:
            assert graph.get_all_nodes() == []
        finally:
            graph.close()
        assert (tmp_path / ".milknado" / "milknado.db").exists()

    def test_open_graph_and_add_node(self, tmp_path: Path) -> None:
        graph, _cfg = open_graph(tmp_path)
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

        assert resolve_project_root(str(explicit)) == explicit.resolve()
        assert resolve_project_root(None) == from_env.resolve()

    def test_project_root_falls_back_to_cwd(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("MILKNADO_PROJECT_ROOT", raising=False)
        old = Path.cwd()
        os.chdir(tmp_path)
        try:
            assert resolve_project_root(None) == tmp_path.resolve()
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
        graph, _cfg = open_graph(tmp_path)
        try:
            from milknado.domains.common import NodeKind

            node = graph.add_node("roadmap-node", kind=NodeKind.ROADMAP)
            reloaded = graph.get_node(node.id)
            assert reloaded is not None
            assert reloaded.kind == NodeKind.ROADMAP
        finally:
            graph.close()


class TestTodoBriefAndRun:
    def test_brief_includes_goal_chain_and_done_siblings(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        roadmap = _call(
            milknado_todo_add, description="ship product", kind="roadmap", project_root=root
        )
        goal = _call(
            milknado_todo_add,
            description="ship v1",
            kind="goal",
            parent_id=roadmap["id"],
            project_root=root,
        )
        done_sib = _call(
            milknado_todo_add,
            description="schema migration",
            kind="task",
            parent_id=goal["id"],
            project_root=root,
        )
        _call(milknado_todo_set_status, node_id=done_sib["id"], status="done", project_root=root)
        task = _call(
            milknado_todo_add,
            description="wire MCP tools",
            kind="task",
            parent_id=goal["id"],
            project_root=root,
        )

        result = _call(milknado_todo_brief, node_id=task["id"], project_root=root)
        brief = result["brief"]
        assert "# Task: wire MCP tools" in brief
        assert "ship product" in brief
        assert "ship v1" in brief
        assert "schema migration" in brief
        assert "(no file hints registered)" in brief

    def test_brief_with_file_ownership_lists_files(self, tmp_path: Path) -> None:
        graph, _cfg = open_graph(tmp_path)
        try:
            from milknado.domains.common import NodeKind

            node = graph.add_node("touch foo", kind=NodeKind.TASK)
            graph.set_file_ownership(node.id, ["src/foo.py", "src/bar.py"])
        finally:
            graph.close()

        result = _call(milknado_todo_brief, node_id=node.id, project_root=str(tmp_path))
        assert "src/foo.py" in result["brief"]
        assert "src/bar.py" in result["brief"]
        assert set(result["files"]) == {"src/foo.py", "src/bar.py"}

    def test_brief_unknown_node_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(milknado_todo_brief, node_id=42, project_root=str(tmp_path))

    def test_run_success_marks_done_and_writes_log(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="echo task", kind="task", project_root=root)
        result = _call(
            milknado_todo_run,
            node_id=task["id"],
            worker_cmd="cat",
            timeout_seconds=10,
            project_root=root,
        )
        assert result["status"] == "done"
        assert result["exit_code"] == 0
        assert result["timed_out"] is False
        assert "# Task: echo task" in result["summary"]
        assert Path(result["log_path"]).exists()

    def test_run_nonzero_exit_marks_failed(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="bad task", kind="task", project_root=root)
        result = _call(
            milknado_todo_run,
            node_id=task["id"],
            worker_cmd="false",
            timeout_seconds=10,
            project_root=root,
        )
        assert result["status"] == "failed"
        assert result["exit_code"] != 0

    def test_run_timeout_marks_failed(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="slow", kind="task", project_root=root)
        result = _call(
            milknado_todo_run,
            node_id=task["id"],
            worker_cmd="sleep 5",
            timeout_seconds=1,
            project_root=root,
        )
        assert result["status"] == "failed"
        assert result["timed_out"] is True

    def test_run_unknown_node_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(
                milknado_todo_run,
                node_id=99,
                worker_cmd="cat",
                timeout_seconds=5,
                project_root=str(tmp_path),
            )

    def test_run_uses_env_worker_cmd_when_arg_omitted(self, tmp_path: Path, monkeypatch) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", kind="task", project_root=root)
        monkeypatch.setenv("MILKNADO_WORKER_CMD", "cat")
        result = _call(
            milknado_todo_run,
            node_id=task["id"],
            timeout_seconds=10,
            project_root=root,
        )
        assert result["status"] == "done"
        assert "# Task: t" in result["summary"]


class TestTodoAsyncRun:
    def test_start_returns_run_id_and_marks_running(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="async-cat", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd="cat",
            timeout_seconds=10,
            project_root=root,
        )
        assert started["status"] == "running"
        assert started["run_id"].startswith(f"node-{task['id']}-")
        tree = _call(milknado_todo_tree, project_root=root)
        assert tree[0]["status"] == "running"

    def test_poll_reconciles_to_done_after_worker_finishes(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="async-done", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd="cat",
            timeout_seconds=10,
            project_root=root,
        )
        final = _wait_for_terminal(started["run_id"], root)
        assert final["status"] == "done"
        assert final["exit_code"] == 0
        assert final["timed_out"] is False
        assert "# Task: async-done" in final["summary"]
        tree = _call(milknado_todo_tree, project_root=root)
        assert tree[0]["status"] == "done"

    def test_poll_reconciles_to_failed_on_nonzero(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="async-fail", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd="false",
            timeout_seconds=10,
            project_root=root,
        )
        final = _wait_for_terminal(started["run_id"], root)
        assert final["status"] == "failed"
        assert final["exit_code"] != 0
        tree = _call(milknado_todo_tree, project_root=root)
        assert tree[0]["status"] == "failed"

    def test_poll_reconciles_to_failed_on_timeout(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="async-slow", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd="sleep 30",
            timeout_seconds=1,
            project_root=root,
        )
        final = _wait_for_terminal(started["run_id"], root, timeout=5.0)
        assert final["status"] == "failed"
        assert final["timed_out"] is True

    def test_start_refuses_when_node_already_running(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="x", kind="task", project_root=root)
        _call(
            milknado_todo_set_status, node_id=task["id"], status="in_progress", project_root=root
        )
        with pytest.raises(ValueError, match="already running"):
            _call(
                milknado_todo_run_start,
                node_id=task["id"],
                worker_cmd="cat",
                timeout_seconds=10,
                project_root=root,
            )

    def test_start_unknown_node_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(
                milknado_todo_run_start,
                node_id=42,
                worker_cmd="cat",
                timeout_seconds=10,
                project_root=str(tmp_path),
            )

    def test_poll_unknown_run_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(
                milknado_todo_run_poll,
                run_id="node-1-20260101T000000Z-abcd",
                project_root=str(tmp_path),
            )

    @pytest.mark.parametrize(
        "bad_run_id",
        [
            "../../etc/passwd",
            "node-1-doesnotexist",
            "",
            "node-1-20260101T000000Z-../escape",
            "../../../something.state",
        ],
    )
    def test_poll_rejects_malformed_run_id(self, tmp_path: Path, bad_run_id: str) -> None:
        with pytest.raises(ValueError, match="invalid run_id format"):
            _call(
                milknado_todo_run_poll,
                run_id=bad_run_id,
                project_root=str(tmp_path),
            )

    def test_poll_returns_running_while_worker_in_flight(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="async-sleep", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd="sleep 2",
            timeout_seconds=10,
            project_root=root,
        )
        immediate = _call(milknado_todo_run_poll, run_id=started["run_id"], project_root=root)
        assert immediate["status"] == "running"
        assert immediate["exit_code"] is None
        _wait_for_terminal(started["run_id"], root, timeout=5.0)

    def test_async_worker_spawn_failure_reaches_failed(self, tmp_path: Path) -> None:
        """Bad worker_cmd must not leave the state file stuck on 'running'."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="bad-cmd", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd="nonexistent-binary-xyz-12345",
            timeout_seconds=10,
            project_root=root,
        )
        final = _wait_for_terminal(started["run_id"], root, timeout=3.0)
        assert final["status"] == "failed"
        assert final["exit_code"] == -1
        assert "error" in final
        tree = _call(milknado_todo_tree, project_root=root)
        assert tree[0]["status"] == "failed"

    def test_start_reconciles_orphaned_terminal_runs(self, tmp_path: Path) -> None:
        """If a worker finished but nobody polled, the next start should reconcile
        and not be locked out by the RUNNING guard."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="orphan", kind="task", project_root=root)
        # First run: succeeds, but we never poll → node stays RUNNING in the graph.
        first = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd="cat",
            timeout_seconds=10,
            project_root=root,
        )
        # Wait for the worker thread to write the terminal state file.
        # Read directly rather than via _wait_for_terminal — that helper polls
        # through the MCP tool, which would auto-reconcile and defeat the test.
        state_path = Path(first["state_path"])
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state["status"] in ("done", "failed"):
                    break
            time.sleep(0.05)
        # Sanity: graph still thinks the node is running (no poll yet).
        tree = _call(milknado_todo_tree, project_root=root)
        assert tree[0]["status"] == "running"
        # Second start: should reconcile the orphan, then start a fresh run.
        second = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd="cat",
            timeout_seconds=10,
            project_root=root,
        )
        assert second["run_id"] != first["run_id"]
        _wait_for_terminal(second["run_id"], root, timeout=3.0)


class TestSchemaMigration:
    def test_mikado_graph_migrates_pre_kind_db(self, tmp_path: Path) -> None:
        """A SQLite DB created before the `kind` column must be migrated by
        MikadoGraph.__init__'s ensure_schema so that add_node's INSERT
        (which includes `kind`) works on upgrade."""
        import sqlite3

        from milknado.domains.common import NodeKind

        # Pre-populate the standard milknado db path with a pre-`kind` schema.
        db_dir = tmp_path / ".milknado"
        db_dir.mkdir()
        db_path = db_dir / "milknado.db"
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                parent_id INTEGER,
                worktree_path TEXT,
                branch_name TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                oversized INTEGER NOT NULL DEFAULT 0,
                batch_index INTEGER
            );
        """)
        conn.execute(
            "INSERT INTO nodes (description, created_at) VALUES (?, ?)",
            ("legacy", "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()

        # Open MikadoGraph — its __init__ runs create_tables + ensure_schema,
        # which must migrate the `kind` column onto the existing table.
        graph, _cfg = open_graph(tmp_path)
        try:
            # The legacy row reads back as TASK (default).
            all_nodes = graph.get_all_nodes()
            legacy_node = next(n for n in all_nodes if n.description == "legacy")
            assert legacy_node.kind == NodeKind.TASK
            # And a fresh add_node with a non-default kind must succeed
            # (this is the INSERT that would fail without the migration).
            node = graph.add_node("post-migration", kind=NodeKind.GOAL)
            reloaded = graph.get_node(node.id)
            assert reloaded is not None
            assert reloaded.kind == NodeKind.GOAL
        finally:
            graph.close()
