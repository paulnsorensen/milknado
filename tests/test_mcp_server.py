from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

import pytest

from milknado._mcp_core import mcp
from milknado.domains.common.errors import InvalidTransition
from milknado.domains.dispatch import reconcile_node_status
from milknado.mcp_run import (
    milknado_todo_run,
    milknado_todo_run_poll,
    milknado_todo_run_start,
)
from milknado.mcp_server import (
    milknado_graph_summary,
    open_graph,
    resolve_project_root,
)
from milknado.mcp_todo import (
    milknado_get_node,
    milknado_todo_brief,
    milknado_todo_next,
    milknado_todo_tree,
)
from milknado.mcp_todo_mutate import (
    milknado_delete_node,
    milknado_edit_node,
    milknado_move_node,
    milknado_set_subtree_status,
    milknado_todo_add,
    milknado_todo_set_status,
    milknado_track_follow_up,
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


def _seed_run(
    root: Path,
    *,
    run_id: str,
    node_id: int,
    status: str = "running",
    started_at: str | None = None,
    timeout_seconds: int | None = 10,
    log_path: str | None = None,
    ended_at: str | None = None,
    exit_code: int | None = None,
    timed_out: bool = False,
    error: str | None = None,
    pid: int | None = None,
) -> None:
    """Insert a runs row directly (replaces writing a .state.json sidecar).

    Ensures a node row with `node_id` exists first so the runs FK is satisfied,
    then inserts the run row at the requested status. The graph is closed so its
    WAL is checkpointed and a subsequent open on the same db sees the row.
    """
    from datetime import UTC, datetime

    graph, _cfg = open_graph(root)
    try:
        conn = graph._conn
        conn.execute(
            "INSERT OR IGNORE INTO nodes (id, description, status, created_at) "
            "VALUES (?, ?, 'running', ?)",
            (node_id, f"seeded-{node_id}", datetime.now(UTC).isoformat()),
        )
        started_at = started_at or datetime.now(UTC).isoformat()
        log_path = log_path or str(root / ".milknado" / "runs" / f"{run_id}.log")
        graph.start_run(run_id, node_id, log_path, started_at, timeout_seconds, pid)
        if status != "running":
            graph.finish_run(
                run_id,
                status=status,
                exit_code=exit_code,
                timed_out=timed_out,
                ended_at=ended_at or datetime.now(UTC).isoformat(),
                error=error,
            )
        conn.commit()
    finally:
        graph.close()


def _read_run(root: Path, run_id: str) -> dict | None:
    """Read a runs row back (replaces reading a .state.json sidecar)."""
    graph, _cfg = open_graph(root)
    try:
        return graph.get_run(run_id)
    finally:
        graph.close()


def _run_status(graph, run_id: str) -> str:  # noqa: ANN001
    """Fetch a run's status, asserting the row exists first. Keeps the
    stale-sweep assertions from masking a missing row (get_run -> None) as a raw
    TypeError instead of a clear 'run vanished' failure."""
    row = graph.get_run(run_id)
    assert row is not None, f"run {run_id!r} not found"
    return row["status"]


@pytest.fixture()
def worker_stub(tmp_path_factory, monkeypatch):
    """Install a `claude` passthrough executable on PATH so run-mechanics tests
    can drive the dispatcher with the worker-cmd allowlist enforced.

    The worker_cmd allowlist (`runner._validate_worker_argv`) only admits real
    agent CLIs (claude/codex/...). These tests exercise dispatch plumbing, not a
    live agent, so they route their throwaway commands (`cat`, `false`, `sleep`)
    through a stub named `claude` that just `exec "$@"` — stdin, exit code, and
    timeout behaviour all pass through unchanged.
    """
    bindir = tmp_path_factory.mktemp("worker-stub-bin")
    stub = bindir / "claude"
    stub.write_text('#!/bin/sh\nexec "$@"\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")

    def _cmd(command: str) -> str:
        return f"claude {command}"

    return _cmd


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

    def test_set_status_blocked_to_done_routes_through_pending(self, tmp_path: Path) -> None:
        """The facade exposes blocked->done, but the state machine only allows
        BLOCKED->PENDING; the tool must step through pending/running to land done."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", kind="task", project_root=root)
        _call(milknado_todo_set_status, node_id=task["id"], status="blocked", project_root=root)
        done = _call(
            milknado_todo_set_status, node_id=task["id"], status="done", project_root=root
        )
        assert done["status"] == "done"

    def test_set_status_blocked_to_in_progress_routes_through_pending(
        self, tmp_path: Path
    ) -> None:
        """BLOCKED→in_progress steps BLOCKED→PENDING→RUNNING without raising."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", project_root=root)
        _call(milknado_todo_set_status, node_id=task["id"], status="blocked", project_root=root)
        result = _call(
            milknado_todo_set_status, node_id=task["id"], status="in_progress", project_root=root
        )
        assert result["status"] == "running"

    def test_set_status_idempotent_set_is_noop(self, tmp_path: Path) -> None:
        """Re-setting a node to its current status must not raise InvalidTransition."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", kind="task", project_root=root)
        _call(milknado_todo_set_status, node_id=task["id"], status="blocked", project_root=root)
        again = _call(
            milknado_todo_set_status, node_id=task["id"], status="blocked", project_root=root
        )
        assert again["status"] == "blocked"

    def test_set_status_reopening_done_node_raises(self, tmp_path: Path) -> None:
        """Reopening a completed node stays disallowed by the state machine."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", kind="task", project_root=root)
        _call(milknado_todo_set_status, node_id=task["id"], status="done", project_root=root)
        with pytest.raises(Exception):  # noqa: B017 - InvalidTransition from the state machine
            _call(
                milknado_todo_set_status,
                node_id=task["id"],
                status="in_progress",
                project_root=root,
            )

    def test_tree_materialises_nested_structure(self, tmp_path: Path) -> None:
        """The single-scan children map must preserve multi-level nesting."""
        root = str(tmp_path)
        goal = _call(milknado_todo_add, description="goal", kind="goal", project_root=root)
        t1 = _call(
            milknado_todo_add,
            description="t1",
            kind="task",
            parent_id=goal["id"],
            project_root=root,
        )
        _call(
            milknado_todo_add,
            description="t1a",
            kind="task",
            parent_id=t1["id"],
            project_root=root,
        )
        tree = _call(milknado_todo_tree, project_root=root)
        assert len(tree) == 1
        assert tree[0]["description"] == "goal"
        assert [c["description"] for c in tree[0]["children"]] == ["t1"]
        assert [c["description"] for c in tree[0]["children"][0]["children"]] == ["t1a"]

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

    def test_brief_instructs_worker_to_track_follow_ups(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="do thing", project_root=root)
        result = _call(milknado_todo_brief, node_id=task["id"], project_root=root)
        assert "milknado_track_follow_up" in result["brief"]

    def test_run_rejects_disallowed_worker_cmd(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="worker_cmd must start with"):
            _call(
                milknado_todo_run,
                node_id=1,
                worker_cmd="cat",
                timeout_seconds=5,
                project_root=str(tmp_path),
            )

    def test_run_success_marks_done_and_writes_log(self, tmp_path: Path, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="echo task", kind="task", project_root=root)
        result = _call(
            milknado_todo_run,
            node_id=task["id"],
            worker_cmd=worker_stub("cat"),
            timeout_seconds=10,
            project_root=root,
        )
        assert result["status"] == "done"
        assert result["exit_code"] == 0
        assert result["timed_out"] is False
        assert "# Task: echo task" in result["summary"]
        assert Path(result["log_path"]).exists()

    def test_run_injects_node_id_into_worker_env(self, tmp_path: Path, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="env task", kind="task", project_root=root)
        result = _call(
            milknado_todo_run,
            node_id=task["id"],
            worker_cmd=worker_stub("sh -c 'echo NODE=$MILKNADO_NODE_ID'"),
            timeout_seconds=10,
            project_root=root,
        )
        assert f"NODE={task['id']}" in result["summary"]

    def test_run_nonzero_exit_marks_failed(self, tmp_path: Path, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="bad task", kind="task", project_root=root)
        result = _call(
            milknado_todo_run,
            node_id=task["id"],
            worker_cmd=worker_stub("false"),
            timeout_seconds=10,
            project_root=root,
        )
        assert result["status"] == "failed"
        assert result["exit_code"] != 0

    def test_run_timeout_marks_failed(self, tmp_path: Path, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="slow", kind="task", project_root=root)
        result = _call(
            milknado_todo_run,
            node_id=task["id"],
            worker_cmd=worker_stub("sleep 5"),
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
                worker_cmd="claude",
                timeout_seconds=5,
                project_root=str(tmp_path),
            )

    def test_run_reruns_failed_node_to_done(self, tmp_path: Path, worker_stub) -> None:
        """A synchronous re-run of a FAILED node normalises to RUNNING before
        recording the new terminal status, so the second run lands DONE instead
        of raising InvalidTransition on the FAILED -> DONE mark."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="syncretry", kind="task", project_root=root)
        failed = _call(
            milknado_todo_run,
            node_id=task["id"],
            worker_cmd=worker_stub("sh -c 'exit 1'"),
            timeout_seconds=10,
            project_root=root,
        )
        assert failed["status"] == "failed"
        done = _call(
            milknado_todo_run,
            node_id=task["id"],
            worker_cmd=worker_stub("cat"),
            timeout_seconds=10,
            project_root=root,
        )
        assert done["status"] == "done"


class TestTodoAsyncRun:
    def test_start_rejects_disallowed_worker_cmd(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="worker_cmd must start with"):
            _call(
                milknado_todo_run_start,
                node_id=1,
                worker_cmd="false",
                timeout_seconds=5,
                project_root=str(tmp_path),
            )

    def test_start_returns_run_id_and_marks_running(self, tmp_path: Path, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="async-cat", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=worker_stub("cat"),
            timeout_seconds=10,
            project_root=root,
        )
        assert started["status"] == "running"
        assert started["run_id"].startswith(f"node-{task['id']}-")
        tree = _call(milknado_todo_tree, project_root=root)
        assert tree[0]["status"] == "running"

    def test_poll_reconciles_to_done_after_worker_finishes(
        self, tmp_path: Path, worker_stub
    ) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="async-done", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=worker_stub("cat"),
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

    def test_async_run_injects_node_id_into_worker_env(self, tmp_path: Path, worker_stub) -> None:
        # The async path threads node_id through base_state into _execute; verify
        # the worker subprocess sees MILKNADO_NODE_ID just like the sync path does.
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="async-env", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=worker_stub("sh -c 'echo NODE=$MILKNADO_NODE_ID'"),
            timeout_seconds=10,
            project_root=root,
        )
        final = _wait_for_terminal(started["run_id"], root)
        assert final["status"] == "done"
        assert f"NODE={task['id']}" in final["summary"]

    def test_poll_reconciles_to_failed_on_nonzero(self, tmp_path: Path, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="async-fail", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=worker_stub("false"),
            timeout_seconds=10,
            project_root=root,
        )
        final = _wait_for_terminal(started["run_id"], root)
        assert final["status"] == "failed"
        assert final["exit_code"] != 0
        tree = _call(milknado_todo_tree, project_root=root)
        assert tree[0]["status"] == "failed"

    def test_poll_reconciles_to_failed_on_timeout(self, tmp_path: Path, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="async-slow", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=worker_stub("sleep 30"),
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
                worker_cmd="claude",
                timeout_seconds=10,
                project_root=root,
            )

    def test_start_unknown_node_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(
                milknado_todo_run_start,
                node_id=42,
                worker_cmd="claude",
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

    def test_poll_returns_running_while_worker_in_flight(
        self, tmp_path: Path, worker_stub
    ) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="async-sleep", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=worker_stub("sleep 2"),
            timeout_seconds=10,
            project_root=root,
        )
        immediate = _call(milknado_todo_run_poll, run_id=started["run_id"], project_root=root)
        assert immediate["status"] == "running"
        assert immediate["exit_code"] is None
        _wait_for_terminal(started["run_id"], root, timeout=5.0)

    def test_async_worker_spawn_failure_reaches_failed(self, tmp_path: Path) -> None:
        """A worker_cmd that passes the allowlist but can't be spawned (binary
        missing on disk) must not leave the state file stuck on 'running'. Uses
        an allowlisted basename (`claude`) at a path that does not exist, so the
        FileNotFoundError surfaces in the async worker thread, not at validation."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="bad-cmd", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=str(tmp_path / "nonexistent" / "claude"),
            timeout_seconds=10,
            project_root=root,
        )
        final = _wait_for_terminal(started["run_id"], root, timeout=3.0)
        assert final["status"] == "failed"
        assert final["exit_code"] == -1
        assert "error" in final
        tree = _call(milknado_todo_tree, project_root=root)
        assert tree[0]["status"] == "failed"

    def test_start_reconciles_orphaned_terminal_runs(self, tmp_path: Path, worker_stub) -> None:
        """If a worker finished but nobody polled, the next start should reconcile
        and not be locked out by the RUNNING guard. A 'failed' orphan reconciles to
        FAILED, which the atomic claim can re-acquire for a retry (a 'done' orphan
        reconciles to DONE and is correctly refused — a completed node is not
        silently re-run)."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="orphan", kind="task", project_root=root)
        # First run: finishes (failed), but we never poll → node stays RUNNING.
        first = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=worker_stub("sh -c 'exit 1'"),
            timeout_seconds=10,
            project_root=root,
        )
        # Wait for the worker thread to write the terminal run row. Read the row
        # directly rather than via _wait_for_terminal — that helper polls through
        # the MCP tool, which would auto-reconcile and defeat the test.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            state = _read_run(Path(root), first["run_id"])
            if state is not None and state["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        # Sanity: graph still thinks the node is running (no poll yet).
        tree = _call(milknado_todo_tree, project_root=root)
        assert tree[0]["status"] == "running"
        # Second start: should reconcile the orphan, then start a fresh run.
        second = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=worker_stub("sh -c 'exit 1'"),
            timeout_seconds=10,
            project_root=root,
        )
        assert second["run_id"] != first["run_id"]
        _wait_for_terminal(second["run_id"], root, timeout=3.0)

    def test_fail_stale_running_runs_flips_orphaned_run(self, tmp_path: Path) -> None:
        """A 'running' run past its timeout (worker thread vanished without writing
        a terminal state) is flipped to failed."""
        from datetime import UTC, datetime, timedelta

        from milknado.domains.dispatch.reconcile import fail_stale_running_runs

        root = Path(tmp_path)
        run_id = "node-7-20200101T000000Z-abcd"
        stale_started = (datetime.now(UTC) - timedelta(seconds=600)).isoformat()
        _seed_run(root, run_id=run_id, node_id=7, started_at=stale_started, timeout_seconds=10)
        graph, _cfg = open_graph(root)
        try:
            flipped = fail_stale_running_runs(graph, 7)
            assert len(flipped) == 1
            assert flipped[0]["status"] == "failed"
            assert _run_status(graph, run_id) == "failed"
        finally:
            graph.close()

    def test_fail_stale_running_runs_leaves_fresh_run_untouched(self, tmp_path: Path) -> None:
        """A 'running' run still within its timeout window must not be flipped."""
        root = Path(tmp_path)
        run_id = "node-8-20200101T000000Z-beef"
        _seed_run(root, run_id=run_id, node_id=8, timeout_seconds=600)
        graph, _cfg = open_graph(root)
        try:
            from milknado.domains.dispatch.reconcile import fail_stale_running_runs

            assert fail_stale_running_runs(graph, 8) == []
            assert _run_status(graph, run_id) == "running"
        finally:
            graph.close()

    def _seed_stale_running(
        self, root: Path, run_id: str, node_id: int, *, pid: int | None
    ) -> None:
        """Seed a 'running' run aged well past its timeout. `pid=None` models the
        async-worker path (no pid recorded); an int models the detached-ralph
        path, which records its process pid."""
        from datetime import UTC, datetime, timedelta

        stale_started = (datetime.now(UTC) - timedelta(seconds=7200)).isoformat()
        _seed_run(
            root,
            run_id=run_id,
            node_id=node_id,
            started_at=stale_started,
            timeout_seconds=1800,
            pid=pid,
        )

    def test_fail_stale_does_not_flip_live_detached_run(self, tmp_path: Path) -> None:
        """#54: a detached-ralph run still 'running' past timeout+grace but whose
        recorded pid is ALIVE (slow, e.g. wedged in `git rebase`) must NOT be
        force-failed — that thrashes a run about to write 'done'."""
        import os

        from milknado.domains.dispatch.reconcile import fail_stale_running_runs

        root = Path(tmp_path)
        run_id = "node-9-20200101T000000Z-aaaa"
        # This test process is a guaranteed-alive pid.
        self._seed_stale_running(root, run_id, 9, pid=os.getpid())
        graph, _cfg = open_graph(root)
        try:
            assert fail_stale_running_runs(graph, 9) == [], "a live runner was wrongly flipped"
            assert _run_status(graph, run_id) == "running"
        finally:
            graph.close()

    def test_fail_stale_flips_dead_detached_run(self, tmp_path: Path) -> None:
        """#54: a detached run whose recorded pid is DEAD is genuinely orphaned and
        must still be flipped to failed (the liveness guard cleans up, not hides)."""
        from milknado.domains.dispatch.reconcile import fail_stale_running_runs

        root = Path(tmp_path)
        run_id = "node-10-20200101T000000Z-bbbb"
        # PID 2**31-1 is unassigned on Linux — os.kill(_, 0) -> ProcessLookupError.
        self._seed_stale_running(root, run_id, 10, pid=2**31 - 1)
        graph, _cfg = open_graph(root)
        try:
            flipped = fail_stale_running_runs(graph, 10)
            assert len(flipped) == 1
            assert flipped[0]["status"] == "failed"
            assert _run_status(graph, run_id) == "failed"
        finally:
            graph.close()

    def test_fail_stale_flips_pidless_async_run(self, tmp_path: Path) -> None:
        """#54 regression guard: the async-worker path records NO pid; its orphan
        sweep (server crash → daemon thread vanished) must stay unchanged — a
        pid-less stale run still flips to failed."""
        from milknado.domains.dispatch.reconcile import fail_stale_running_runs

        root = Path(tmp_path)
        run_id = "node-11-20200101T000000Z-cccc"
        self._seed_stale_running(root, run_id, 11, pid=None)
        graph, _cfg = open_graph(root)
        try:
            flipped = fail_stale_running_runs(graph, 11)
            assert len(flipped) == 1
            assert flipped[0]["status"] == "failed"
            assert _run_status(graph, run_id) == "failed"
        finally:
            graph.close()

    def test_fail_stale_mixed_kinds_flips_only_the_dead_one(self, tmp_path: Path) -> None:
        """#54 cross-namespace guard: BOTH a detached-ralph run (records a pid) and
        a pid-less async run exist for the same node. The sweep must flip only the
        genuinely-dead one — a live ralph run for the node must survive, not be
        cross-failed."""
        import os

        from milknado.domains.dispatch.reconcile import fail_stale_running_runs

        root = Path(tmp_path)
        live = "node-12-20200101T000000Z-1111"
        dead = "node-12-20200102T000000Z-2222"
        self._seed_stale_running(root, live, 12, pid=os.getpid())
        self._seed_stale_running(root, dead, 12, pid=None)
        graph, _cfg = open_graph(root)
        try:
            flipped = fail_stale_running_runs(graph, 12)
            assert [f["run_id"] for f in flipped] == [dead]
            assert _run_status(graph, live) == "running"
            assert _run_status(graph, dead) == "failed"
        finally:
            graph.close()

    def test_start_releases_node_locked_by_stale_run(self, tmp_path: Path, worker_stub) -> None:
        """run_start sweeps a stale 'running' run so a node stuck on RUNNING
        (worker thread vanished) is not locked out forever."""
        from datetime import UTC, datetime, timedelta

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="stale", kind="task", project_root=root)
        graph, _cfg = open_graph(Path(root))
        try:
            graph.mark_running(task["id"])
        finally:
            graph.close()
        run_id = f"node-{task['id']}-20200101T000000Z-dead"
        stale_started = (datetime.now(UTC) - timedelta(seconds=600)).isoformat()
        _seed_run(
            Path(root),
            run_id=run_id,
            node_id=task["id"],
            started_at=stale_started,
            timeout_seconds=10,
        )
        # Must NOT raise "already running" — the stale run is reconciled first.
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=worker_stub("sh -c 'exit 1'"),
            timeout_seconds=10,
            project_root=root,
        )
        assert started["run_id"] != run_id
        final = _wait_for_terminal(started["run_id"], root, timeout=3.0)
        assert final["status"] == "failed"

    def test_run_start_reruns_failed_node_to_done(self, tmp_path: Path, worker_stub) -> None:
        """A FAILED node re-run via run_start is normalised to RUNNING, so a
        successful retry reconciles to DONE instead of raising InvalidTransition
        in run_poll (FAILED -> DONE is not a valid direct transition)."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="retry", kind="task", project_root=root)
        first = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=worker_stub("sh -c 'exit 1'"),
            timeout_seconds=10,
            project_root=root,
        )
        assert _wait_for_terminal(first["run_id"], root, timeout=3.0)["status"] == "failed"
        second = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=worker_stub("cat"),
            timeout_seconds=10,
            project_root=root,
        )
        assert _wait_for_terminal(second["run_id"], root, timeout=3.0)["status"] == "done"
        assert _call(milknado_todo_tree, project_root=root)[0]["status"] == "done"

    def test_concurrent_run_start_spawns_exactly_one_worker(
        self, tmp_path: Path, worker_stub
    ) -> None:
        """Two concurrent run_start calls on the same PENDING node must result in
        exactly one worker being spawned. Without the dispatch lock, both callers
        can pass the RUNNING status check simultaneously and each spawn a worker —
        this is the race #38 guards against."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="concurrent", kind="task", project_root=root)
        # Use a slow worker so it's still running if the lock leaks two starters.
        slow_cmd = worker_stub("sleep 10")
        results: list[dict] = []
        errors: list[str] = []
        result_lock = threading.Lock()
        barrier = threading.Barrier(2)

        def _start() -> None:
            barrier.wait()  # maximise overlap on the critical section
            try:
                r = _call(
                    milknado_todo_run_start,
                    node_id=task["id"],
                    worker_cmd=slow_cmd,
                    timeout_seconds=30,
                    project_root=root,
                )
                with result_lock:
                    results.append(r)
            except ValueError as exc:
                with result_lock:
                    errors.append(str(exc))

        threads = [threading.Thread(target=_start) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 1, (
            f"expected exactly 1 successful dispatch, got {len(results)} "
            f"(race: duplicate workers spawned). errors={errors}"
        )
        assert len(errors) == 1, f"expected exactly 1 'already running' error, got errors={errors}"
        assert "already running" in errors[0]

    def test_latest_terminal_run_picks_most_recent_by_ended_at(self, tmp_path: Path) -> None:
        """When a node has multiple terminal runs, latest_terminal_run returns the
        one with the highest ended_at timestamp. The older 'done' must not beat the
        newer 'failed' — the most recent outcome is authoritative for reconcile."""
        from milknado.domains.dispatch import latest_terminal_run
        from milknado.domains.dispatch.reconcile import find_terminal_runs_for_node

        root = Path(tmp_path)
        node_id = 42
        for run_id, status, ended_at in [
            ("node-42-20200101T000000Z-aaaa", "done", "2020-01-01T00:00:10+00:00"),
            ("node-42-20200101T000001Z-bbbb", "failed", "2020-01-01T00:00:20+00:00"),
        ]:
            _seed_run(
                root,
                run_id=run_id,
                node_id=node_id,
                status=status,
                ended_at=ended_at,
                exit_code=0 if status == "done" else 1,
            )
        graph, _cfg = open_graph(root)
        try:
            terminal_runs = find_terminal_runs_for_node(graph, node_id)
            winner = latest_terminal_run(terminal_runs)
        finally:
            graph.close()
        assert winner is not None
        assert winner["status"] == "failed", "newer run (failed) must win over older run (done)"
        assert winner["run_id"] == "node-42-20200101T000001Z-bbbb"

    def test_find_terminal_runs_filters_by_run_id(self, tmp_path: Path) -> None:
        """The fence must be applied BEFORE picking the latest run: a stale run with
        a later ended_at must not mask the current owner's terminal run. Filtering
        by run_id first leaves only the owner's run for latest_terminal_run."""
        from milknado.domains.dispatch import latest_terminal_run
        from milknado.domains.dispatch.reconcile import find_terminal_runs_for_node

        root = Path(tmp_path)
        node_id = 7
        owner = "node-7-20200101T000000Z-own0"
        stale = "node-7-20200101T000001Z-stal"
        for run_id, status, ended_at in [
            (owner, "done", "2020-01-01T00:00:10+00:00"),
            (stale, "failed", "2020-01-01T00:01:00+00:00"),  # later ended_at, but not the owner
        ]:
            _seed_run(
                root,
                run_id=run_id,
                node_id=node_id,
                status=status,
                ended_at=ended_at,
                exit_code=0 if status == "done" else 1,
            )
        graph, _cfg = open_graph(root)
        try:
            fenced = find_terminal_runs_for_node(graph, node_id, run_id=owner)
            assert [r["run_id"] for r in fenced] == [owner]
            latest = latest_terminal_run(fenced)
            assert latest is not None
            assert latest["status"] == "done", (
                "the owner's run wins once the stale later-ended_at run is fenced out"
            )
            unfenced = find_terminal_runs_for_node(graph, node_id)
            assert len(unfenced) == 2, "no fence returns both terminal runs"
        finally:
            graph.close()

    def test_find_terminal_runs_isolates_node_id(self, tmp_path: Path) -> None:
        """Node scoping is the runs query's `WHERE node_id = ?`: node 1's runs must
        not pick up node 12's terminal run. Replaces the old glob-prefix isolation
        — the db keys runs on the integer node_id directly."""
        from milknado.domains.dispatch.reconcile import find_terminal_runs_for_node

        root = Path(tmp_path)
        for run_id, node_id in [
            ("node-1-20200101T000000Z-aaaa", 1),
            ("node-12-20200101T000000Z-bbbb", 12),
        ]:
            _seed_run(
                root,
                run_id=run_id,
                node_id=node_id,
                status="done",
                ended_at="2020-01-01T00:00:10+00:00",
                exit_code=0,
            )
        graph, _cfg = open_graph(root)
        try:
            result = find_terminal_runs_for_node(graph, 1)
        finally:
            graph.close()
        assert [r["run_id"] for r in result] == ["node-1-20200101T000000Z-aaaa"], (
            "node 1's query must not pick up node 12's terminal run"
        )

    def test_find_terminal_runs_excludes_non_terminal_runs(self, tmp_path: Path) -> None:
        """The status filter is the terminality gate: a still-'running' run for the
        node must be excluded, only done/failed runs returned. Guards against the
        terminal-status filter regressing into letting a live run reconcile."""
        from milknado.domains.dispatch.reconcile import find_terminal_runs_for_node

        root = Path(tmp_path)
        _seed_run(
            root,
            run_id="node-5-20200101T000000Z-done",
            node_id=5,
            status="done",
            ended_at="2020-01-01T00:00:10+00:00",
            exit_code=0,
        )
        _seed_run(root, run_id="node-5-20200101T000001Z-runn", node_id=5, status="running")
        graph, _cfg = open_graph(root)
        try:
            result = find_terminal_runs_for_node(graph, 5)
        finally:
            graph.close()
        assert [r["run_id"] for r in result] == ["node-5-20200101T000000Z-done"], (
            "a 'running' run must not be returned as a terminal run"
        )

    def test_reconcile_node_status_is_fenced_on_run_id(self, tmp_path: Path) -> None:
        """A reconcile carrying a stale run_id must not clobber a node now RUNNING
        under a newer run; only the matching run_id finalizes it. Closes the TOCTOU
        where two starters reconcile the same orphan and the loser marks the fresh
        owner done/failed."""
        from milknado.domains.common import NodeStatus
        from milknado.domains.dispatch import reconcile_node_status
        from milknado.domains.dispatch._runstate import now_iso
        from milknado.domains.graph import MikadoGraph

        g = MikadoGraph(Path(tmp_path) / "g.db")
        try:
            node_id = g.add_node("reconcile-fence").id
            g.claim_node(node_id, "run-new", now=now_iso())
            reconcile_node_status(g, node_id, "done", run_id="run-stale")
            node_stale = g.get_node(node_id)
            assert node_stale is not None
            assert node_stale.status == NodeStatus.RUNNING, (
                "a stale run cannot finalize a newer owner"
            )
            reconcile_node_status(g, node_id, "done", run_id="run-new")
            node_final = g.get_node(node_id)
            assert node_final is not None
            assert node_final.status == NodeStatus.DONE, "the current owner's run finalizes"
        finally:
            g.close()

    def test_latest_terminal_run_returns_none_for_empty(self, tmp_path: Path) -> None:
        from milknado.domains.dispatch import latest_terminal_run

        assert latest_terminal_run([]) is None

    def test_start_sets_run_id_on_graph_node(self, tmp_path: Path, worker_stub) -> None:
        """Async dispatch sets run_id on the graph node so reconcile/queries can
        find the run without re-scanning the runs dir (finding #40)."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="run-id-check", kind="task", project_root=root)
        started = _call(
            milknado_todo_run_start,
            node_id=task["id"],
            worker_cmd=worker_stub("cat"),
            timeout_seconds=10,
            project_root=root,
        )
        graph, _cfg = open_graph(Path(root))
        try:
            node = graph.get_node(task["id"])
        finally:
            graph.close()
        assert node is not None
        assert node.run_id == started["run_id"]


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


class TestTrackFollowUp:
    def test_inserts_node_and_returns_summary(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        result = _call(milknado_track_follow_up, description="add retry", project_root=root)
        assert result["description"] == "add retry"
        assert result["kind"] == "task"
        tree = _call(milknado_todo_tree, project_root=root)
        assert any(n["description"] == "add retry" for n in tree)

    def test_defaults_parent_to_worker_nodes_parent_as_sibling(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # #124: the auto-parented follow-up lands under the worker node's PARENT
        # (a sibling), never under the worker node itself — the worker's exit-0
        # reconcile drives its node to done, and children are prerequisites, so
        # a child here would leave a done node holding unmet work.
        root = str(tmp_path)
        goal = _call(milknado_todo_add, description="goal", kind="goal", project_root=root)
        task = _call(
            milknado_todo_add, description="task", parent_id=goal["id"], project_root=root
        )
        monkeypatch.setenv("MILKNADO_NODE_ID", str(task["id"]))
        follow_up = _call(
            milknado_track_follow_up, description="discovered work", project_root=root
        )
        detail = _call(milknado_get_node, node_id=follow_up["id"], project_root=root)
        assert detail["parent_id"] == goal["id"]
        task_detail = _call(milknado_get_node, node_id=task["id"], project_root=root)
        assert follow_up["id"] not in task_detail["prerequisite_ids"]

    def test_root_level_worker_node_creates_root_level_follow_up(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # A worker node with no parent has no goal to attach the sibling under;
        # the follow-up becomes a new root-level node (same as unset env).
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="rootless task", project_root=root)
        monkeypatch.setenv("MILKNADO_NODE_ID", str(task["id"]))
        follow_up = _call(
            milknado_track_follow_up, description="discovered work", project_root=root
        )
        detail = _call(milknado_get_node, node_id=follow_up["id"], project_root=root)
        assert detail["parent_id"] is None

    def test_follow_up_during_run_never_gates_the_completing_node(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """#124 regression: worker on a running node registers a follow-up, then
        exit-0 reconcile drives the node to done. The follow-up must not be a
        prerequisite (child) of the done node — children gate readiness, so the
        inversion would mark work complete while holding an unmet prerequisite —
        and the follow-up itself must stay dispatchable under the same goal."""
        root = str(tmp_path)
        goal = _call(milknado_todo_add, description="goal", kind="goal", project_root=root)
        task = _call(
            milknado_todo_add, description="task", parent_id=goal["id"], project_root=root
        )
        graph, _cfg = open_graph(resolve_project_root(root))
        try:
            assert graph.claim_node(task["id"], "run-124", now="2026-06-03T00:00:00+00:00")
            monkeypatch.setenv("MILKNADO_NODE_ID", str(task["id"]))
            follow_up = _call(
                milknado_track_follow_up,
                description="discovered apply-work",
                project_root=root,
            )
            reconcile_node_status(graph, task["id"], "done", "run-124")
        finally:
            graph.close()
        done_node = _call(milknado_get_node, node_id=task["id"], project_root=root)
        assert done_node["status"] == "done"
        assert done_node["prerequisite_ids"] == []
        sibling = _call(milknado_get_node, node_id=follow_up["id"], project_root=root)
        assert sibling["status"] == "pending"
        assert sibling["parent_id"] == goal["id"]
        # The discovered work must actually be dispatchable — not just pending:
        # the scheduler should hand it out as the next runnable node.
        next_node = _call(milknado_todo_next, project_root=root)
        assert next_node is not None
        assert next_node["id"] == follow_up["id"]

    def test_blank_node_id_env_creates_root_level_node(self, tmp_path: Path, monkeypatch) -> None:
        root = str(tmp_path)
        monkeypatch.setenv("MILKNADO_NODE_ID", "")
        result = _call(milknado_track_follow_up, description="rootish", project_root=root)
        roots = _call(milknado_todo_tree, project_root=root)
        assert any(n["id"] == result["id"] for n in roots)

    def test_explicit_parent_overrides_worker_node_id_env(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # An explicit parent_id must win over MILKNADO_NODE_ID — the env default
        # only fills in for the omitted case (parent_id is None).
        root = str(tmp_path)
        env_parent = _call(milknado_todo_add, description="env parent", project_root=root)
        explicit = _call(milknado_todo_add, description="explicit parent", project_root=root)
        monkeypatch.setenv("MILKNADO_NODE_ID", str(env_parent["id"]))
        child = _call(
            milknado_track_follow_up,
            description="discovered",
            parent_id=explicit["id"],
            project_root=root,
        )
        under_explicit = _call(milknado_todo_tree, project_root=root, root_id=explicit["id"])[0][
            "children"
        ]
        under_env = _call(milknado_todo_tree, project_root=root, root_id=env_parent["id"])[0][
            "children"
        ]
        assert [c["id"] for c in under_explicit] == [child["id"]]
        assert under_env == []

    def test_malformed_node_id_env_raises(self, tmp_path: Path, monkeypatch) -> None:
        # A corrupt MILKNADO_NODE_ID fails loud rather than silently dropping the
        # parent link and creating a stray root node.
        root = str(tmp_path)
        monkeypatch.setenv("MILKNADO_NODE_ID", "not-an-int")
        with pytest.raises(ValueError, match="invalid literal for int"):
            _call(milknado_track_follow_up, description="orphan", project_root=root)

    def test_stale_valid_node_id_env_raises_and_persists_nothing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # A well-formed but stale MILKNADO_NODE_ID (points at a deleted node) must
        # fail loud BEFORE inserting — never leave a committed stray node behind
        # when the parent edge would fail its FK.
        root = str(tmp_path)
        gone = _call(milknado_todo_add, description="will be deleted", project_root=root)
        _call(milknado_delete_node, node_id=gone["id"], project_root=root)
        monkeypatch.setenv("MILKNADO_NODE_ID", str(gone["id"]))
        with pytest.raises(ValueError, match="not found"):
            _call(milknado_track_follow_up, description="orphan", project_root=root)
        assert _call(milknado_todo_tree, project_root=root) == []

    def test_invalid_kind_raises(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        with pytest.raises(ValueError, match="invalid kind"):
            _call(milknado_track_follow_up, description="x", kind="bogus", project_root=root)


class TestDeleteAndEditTools:
    def test_delete_leaf_returns_count_and_removes_node(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", project_root=root)
        result = _call(milknado_delete_node, node_id=task["id"], project_root=root)
        assert result == {"deleted": 1}
        assert _call(milknado_todo_tree, project_root=root) == []

    def test_delete_node_with_children_without_cascade_raises(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        parent = _call(milknado_todo_add, description="p", kind="goal", project_root=root)
        _call(milknado_todo_add, description="c", parent_id=parent["id"], project_root=root)
        with pytest.raises(ValueError, match="cascade"):
            _call(milknado_delete_node, node_id=parent["id"], project_root=root)

    def test_delete_cascade_removes_subtree(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        parent = _call(milknado_todo_add, description="p", kind="goal", project_root=root)
        _call(milknado_todo_add, description="c", parent_id=parent["id"], project_root=root)
        result = _call(milknado_delete_node, node_id=parent["id"], cascade=True, project_root=root)
        assert result == {"deleted": 2}
        assert _call(milknado_todo_tree, project_root=root) == []

    def test_edit_description_updates_it(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="old", project_root=root)
        result = _call(
            milknado_edit_node, node_id=task["id"], description="new", project_root=root
        )
        assert result["description"] == "new"

    def test_edit_kind_updates_it(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", project_root=root)
        result = _call(milknado_edit_node, node_id=task["id"], kind="goal", project_root=root)
        assert result["kind"] == "goal"

    def test_edit_with_no_fields_raises(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", project_root=root)
        with pytest.raises(ValueError, match="nothing to edit"):
            _call(milknado_edit_node, node_id=task["id"], project_root=root)

    def test_edit_invalid_kind_raises(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", project_root=root)
        with pytest.raises(ValueError, match="invalid kind"):
            _call(milknado_edit_node, node_id=task["id"], kind="bogus", project_root=root)

    def test_delete_unknown_node_raises(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            _call(milknado_delete_node, node_id=999, project_root=root)

    def test_todo_set_status_raises_when_node_missing_after_update(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit ValueError fires where assert would be stripped under python -O."""
        from milknado.domains.graph import MikadoGraph

        root = str(tmp_path)
        node = _call(milknado_todo_add, description="task", project_root=root)
        nid = node["id"]

        original = MikadoGraph.get_node
        calls = {"n": 0}

        def mock_get_node(self, nid_arg: int):
            calls["n"] += 1
            if calls["n"] >= 2:
                return None
            return original(self, nid_arg)

        monkeypatch.setattr(MikadoGraph, "get_node", mock_get_node)
        with pytest.raises(ValueError, match="not found after status update"):
            _call(milknado_todo_set_status, node_id=nid, status="in_progress", project_root=root)

    def test_edit_node_raises_when_node_missing_after_update(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit ValueError fires where assert would be stripped under python -O."""
        from milknado.domains.graph import MikadoGraph

        root = str(tmp_path)
        node = _call(milknado_todo_add, description="task", project_root=root)
        nid = node["id"]

        monkeypatch.setattr(MikadoGraph, "get_node", lambda self, n: None)
        with pytest.raises(ValueError, match="not found after edit"):
            _call(milknado_edit_node, node_id=nid, description="new desc", project_root=root)


class TestGetNode:
    def test_get_node_returns_summary_parent_and_prerequisites(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal = _call(milknado_todo_add, description="g", kind="goal", project_root=root)
        prereq_a = _call(
            milknado_todo_add, description="a", parent_id=goal["id"], project_root=root
        )
        prereq_b = _call(
            milknado_todo_add, description="b", parent_id=goal["id"], project_root=root
        )

        result = _call(milknado_get_node, node_id=goal["id"], project_root=root)
        assert result["id"] == goal["id"]
        assert result["kind"] == "goal"
        assert result["status"] == "pending"
        assert result["description"] == "g"
        assert result["parent_id"] is None
        assert sorted(result["prerequisite_ids"]) == sorted([prereq_a["id"], prereq_b["id"]])

    def test_get_node_leaf_has_empty_prerequisites_and_parent_set(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal = _call(milknado_todo_add, description="g", kind="goal", project_root=root)
        leaf = _call(milknado_todo_add, description="t", parent_id=goal["id"], project_root=root)

        result = _call(milknado_get_node, node_id=leaf["id"], project_root=root)
        assert result["parent_id"] == goal["id"]
        assert result["prerequisite_ids"] == []

    def test_get_node_unknown_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(milknado_get_node, node_id=999, project_root=str(tmp_path))


class TestTreeMaxDepth:
    def _three_levels(self, root: str) -> dict[str, int]:
        goal = _call(milknado_todo_add, description="goal", kind="goal", project_root=root)
        child = _call(
            milknado_todo_add, description="child", parent_id=goal["id"], project_root=root
        )
        grandchild = _call(
            milknado_todo_add, description="grandchild", parent_id=child["id"], project_root=root
        )
        return {"goal": goal["id"], "child": child["id"], "grandchild": grandchild["id"]}

    def test_max_depth_zero_returns_root_only(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        self._three_levels(root)
        tree = _call(milknado_todo_tree, project_root=root, max_depth=0)
        assert len(tree) == 1
        assert tree[0]["description"] == "goal"
        assert tree[0]["children"] == []

    def test_max_depth_one_returns_root_and_direct_children(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        self._three_levels(root)
        tree = _call(milknado_todo_tree, project_root=root, max_depth=1)
        assert [c["description"] for c in tree[0]["children"]] == ["child"]
        assert tree[0]["children"][0]["children"] == []

    def test_max_depth_none_returns_full_subtree(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        self._three_levels(root)
        tree = _call(milknado_todo_tree, project_root=root)
        assert tree[0]["children"][0]["children"][0]["description"] == "grandchild"

    def test_max_depth_with_explicit_root_id(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        ids = self._three_levels(root)
        subtree = _call(milknado_todo_tree, project_root=root, root_id=ids["goal"], max_depth=1)
        assert [c["description"] for c in subtree[0]["children"]] == ["child"]
        assert subtree[0]["children"][0]["children"] == []


class TestGraphSummaryFilters:
    def test_status_filter_lists_only_matching(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        done = _call(milknado_todo_add, description="done-task", project_root=root)
        _call(milknado_todo_add, description="pending-task", project_root=root)
        _call(milknado_todo_set_status, node_id=done["id"], status="done", project_root=root)

        summary = _call(milknado_graph_summary, project_root=root, status="done")
        assert any("done-task" in n["description"] for n in summary["nodes"])
        assert all("pending-task" not in n["description"] for n in summary["nodes"])

    def test_kind_filter_lists_only_matching(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        _call(milknado_todo_add, description="the-goal", kind="goal", project_root=root)
        _call(milknado_todo_add, description="the-task", kind="task", project_root=root)

        summary = _call(milknado_graph_summary, project_root=root, kind="goal")
        assert any("the-goal" in n["description"] for n in summary["nodes"])
        assert all("the-task" not in n["description"] for n in summary["nodes"])

    def test_no_match_returns_empty_graph_sentinel(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        _call(milknado_todo_add, description="t", kind="task", project_root=root)
        assert _call(milknado_graph_summary, project_root=root, kind="roadmap") == {"nodes": []}

    def test_unfiltered_lists_all(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        _call(milknado_todo_add, description="one", project_root=root)
        _call(milknado_todo_add, description="two", project_root=root)
        summary = _call(milknado_graph_summary, project_root=root)
        descriptions = [n["description"] for n in summary["nodes"]]
        assert "one" in descriptions
        assert "two" in descriptions

    def test_status_and_kind_filters_are_anded(self, tmp_path: Path) -> None:
        """Both filters together match only nodes satisfying status AND kind."""
        root = str(tmp_path)
        done_goal = _call(
            milknado_todo_add, description="done-goal", kind="goal", project_root=root
        )
        _call(milknado_todo_add, description="pending-goal", kind="goal", project_root=root)
        done_task = _call(
            milknado_todo_add, description="done-task", kind="task", project_root=root
        )
        _call(milknado_todo_set_status, node_id=done_goal["id"], status="done", project_root=root)
        _call(milknado_todo_set_status, node_id=done_task["id"], status="done", project_root=root)

        summary = _call(milknado_graph_summary, project_root=root, status="done", kind="goal")
        descriptions = [n["description"] for n in summary["nodes"]]
        assert "done-goal" in descriptions
        assert "pending-goal" not in descriptions  # right kind, wrong status
        assert "done-task" not in descriptions  # right status, wrong kind


class TestMoveNode:
    def test_move_to_new_parent_updates_parent_and_edges(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal_a = _call(milknado_todo_add, description="a", kind="goal", project_root=root)
        goal_b = _call(milknado_todo_add, description="b", kind="goal", project_root=root)
        task = _call(milknado_todo_add, description="t", parent_id=goal_a["id"], project_root=root)

        moved = _call(
            milknado_move_node,
            node_id=task["id"],
            new_parent_id=goal_b["id"],
            project_root=root,
        )
        assert moved["id"] == task["id"]
        node = _call(milknado_get_node, node_id=task["id"], project_root=root)
        assert node["parent_id"] == goal_b["id"]
        assert (
            _call(milknado_get_node, node_id=goal_a["id"], project_root=root)["prerequisite_ids"]
            == []
        )
        assert _call(milknado_get_node, node_id=goal_b["id"], project_root=root)[
            "prerequisite_ids"
        ] == [task["id"]]

    def test_move_to_root_clears_parent(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal = _call(milknado_todo_add, description="g", kind="goal", project_root=root)
        task = _call(milknado_todo_add, description="t", parent_id=goal["id"], project_root=root)

        _call(milknado_move_node, node_id=task["id"], new_parent_id=None, project_root=root)
        node = _call(milknado_get_node, node_id=task["id"], project_root=root)
        assert node["parent_id"] is None
        assert (
            _call(milknado_get_node, node_id=goal["id"], project_root=root)["prerequisite_ids"]
            == []
        )

    def test_move_preserves_children(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        anchor = _call(milknado_todo_add, description="anchor", kind="goal", project_root=root)
        parent = _call(milknado_todo_add, description="p", kind="goal", project_root=root)
        child = _call(
            milknado_todo_add, description="c", parent_id=parent["id"], project_root=root
        )

        _call(
            milknado_move_node,
            node_id=parent["id"],
            new_parent_id=anchor["id"],
            project_root=root,
        )
        moved = _call(milknado_get_node, node_id=parent["id"], project_root=root)
        assert moved["parent_id"] == anchor["id"]
        assert moved["prerequisite_ids"] == [child["id"]]

    def test_move_under_descendant_raises_cycle(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        parent = _call(milknado_todo_add, description="p", project_root=root)
        child = _call(
            milknado_todo_add, description="c", parent_id=parent["id"], project_root=root
        )
        with pytest.raises(ValueError, match="cycle"):
            _call(
                milknado_move_node,
                node_id=parent["id"],
                new_parent_id=child["id"],
                project_root=root,
            )

    def test_move_under_self_raises_cycle(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        node = _call(milknado_todo_add, description="n", project_root=root)
        with pytest.raises(ValueError, match="cycle"):
            _call(
                milknado_move_node,
                node_id=node["id"],
                new_parent_id=node["id"],
                project_root=root,
            )

    def test_move_unknown_node_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(milknado_move_node, node_id=999, new_parent_id=None, project_root=str(tmp_path))

    def test_move_to_unknown_parent_raises(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        node = _call(milknado_todo_add, description="n", project_root=root)
        with pytest.raises(ValueError, match="not found"):
            _call(
                milknado_move_node,
                node_id=node["id"],
                new_parent_id=999,
                project_root=root,
            )


class TestSetSubtreeStatus:
    def _goal_with_two_tasks(self, root: str) -> dict[str, int]:
        goal = _call(milknado_todo_add, description="goal", kind="goal", project_root=root)
        t1 = _call(milknado_todo_add, description="t1", parent_id=goal["id"], project_root=root)
        t2 = _call(milknado_todo_add, description="t2", parent_id=goal["id"], project_root=root)
        return {"goal": goal["id"], "t1": t1["id"], "t2": t2["id"]}

    def test_marks_whole_subtree_done(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        ids = self._goal_with_two_tasks(root)
        result = _call(
            milknado_set_subtree_status, root_id=ids["goal"], status="done", project_root=root
        )
        assert result == {"updated": 3}
        for key in ("goal", "t1", "t2"):
            assert (
                _call(milknado_get_node, node_id=ids[key], project_root=root)["status"] == "done"
            )

    def test_rerun_is_noop(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        ids = self._goal_with_two_tasks(root)
        _call(milknado_set_subtree_status, root_id=ids["goal"], status="done", project_root=root)
        again = _call(
            milknado_set_subtree_status, root_id=ids["goal"], status="done", project_root=root
        )
        assert again == {"updated": 0}

    def test_counts_only_changed_nodes(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        ids = self._goal_with_two_tasks(root)
        _call(milknado_todo_set_status, node_id=ids["t1"], status="done", project_root=root)
        result = _call(
            milknado_set_subtree_status, root_id=ids["goal"], status="done", project_root=root
        )
        assert result == {"updated": 2}

    def test_illegal_transition_leaves_subtree_untouched(self, tmp_path: Path) -> None:
        # Post-order visit is [t1, t2, goal]. t1 is DONE so targeting in_progress
        # must raise; the bulk update validates the whole subtree before writing,
        # so t2 and goal must remain pending — no half-applied partial commit.
        root = str(tmp_path)
        ids = self._goal_with_two_tasks(root)
        _call(milknado_todo_set_status, node_id=ids["t1"], status="done", project_root=root)
        with pytest.raises(InvalidTransition):
            _call(
                milknado_set_subtree_status,
                root_id=ids["goal"],
                status="in_progress",
                project_root=root,
            )
        assert _call(milknado_get_node, node_id=ids["t1"], project_root=root)["status"] == "done"
        assert (
            _call(milknado_get_node, node_id=ids["t2"], project_root=root)["status"] == "pending"
        )
        assert (
            _call(milknado_get_node, node_id=ids["goal"], project_root=root)["status"] == "pending"
        )

    def test_unknown_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(
                milknado_set_subtree_status,
                root_id=999,
                status="done",
                project_root=str(tmp_path),
            )

    def test_invalid_status_raises(self, tmp_path: Path) -> None:
        # Exercises the runtime _parse_todo_status guard, not the Literal
        # schema — a schema-validated client can't send "bogus".
        root = str(tmp_path)
        node = _call(milknado_todo_add, description="n", project_root=root)
        with pytest.raises(ValueError, match="invalid status"):
            _call(
                milknado_set_subtree_status,
                root_id=node["id"],
                status="bogus",
                project_root=root,
            )

    def test_diamond_shared_descendant_counted_once(self, tmp_path: Path) -> None:
        """A node reachable via two parents in the subtree is visited once, not twice."""
        root = str(tmp_path)
        graph, _cfg = open_graph(tmp_path)
        try:
            goal = graph.add_node("goal")
            left = graph.add_node("left", parent_id=goal.id)
            right = graph.add_node("right", parent_id=goal.id)
            shared = graph.add_node("shared", parent_id=left.id)
            graph.add_edge(right.id, shared.id)
        finally:
            graph.close()

        result = _call(
            milknado_set_subtree_status, root_id=goal.id, status="done", project_root=root
        )
        # goal, left, right, shared — shared counted once despite two parents.
        assert result == {"updated": 4}

    def test_leaf_root_updates_only_itself(self, tmp_path: Path) -> None:
        """A root with no children is a one-node subtree: updated == 1."""
        root = str(tmp_path)
        leaf = _call(milknado_todo_add, description="solo", project_root=root)
        result = _call(
            milknado_set_subtree_status, root_id=leaf["id"], status="done", project_root=root
        )
        assert result == {"updated": 1}
        assert _call(milknado_get_node, node_id=leaf["id"], project_root=root)["status"] == "done"


def _advertised_param(tool_name: str, param: str) -> dict:
    """The JSON-schema fragment FastMCP advertises for one tool parameter."""
    tool = asyncio.run(mcp.get_tool(tool_name))
    assert tool is not None
    return tool.parameters["properties"][param]


def _enum_values(fragment: dict) -> list[str]:
    """Pull the enum list whether it sits on the property or under anyOf (X | None)."""
    if "enum" in fragment:
        return fragment["enum"]
    for branch in fragment.get("anyOf", []):
        if "enum" in branch:
            return branch["enum"]
    raise AssertionError(f"no enum advertised in schema fragment: {fragment}")


class TestAdvertisedEnumSchema:
    """#84: FastMCP must advertise kind/status as JSON-schema enums, not bare strings.

    This is the headline deliverable — clients discover valid values from the
    schema instead of by triggering a ValueError. If the Literal typing is lost
    (reverted to ``str``), the advertised fragment drops the enum and these fail.
    """

    def test_todo_set_status_advertises_status_enum(self) -> None:
        fragment = _advertised_param("milknado_todo_set_status", "status")
        assert _enum_values(fragment) == ["pending", "in_progress", "blocked", "done"]

    def test_todo_add_advertises_kind_enum(self) -> None:
        fragment = _advertised_param("milknado_todo_add", "kind")
        assert _enum_values(fragment) == ["roadmap", "goal", "task"]

    def test_todo_next_advertises_kind_enum(self) -> None:
        fragment = _advertised_param("milknado_todo_next", "kind")
        assert _enum_values(fragment) == ["roadmap", "goal", "task"]

    def test_track_follow_up_advertises_kind_enum(self) -> None:
        fragment = _advertised_param("milknado_track_follow_up", "kind")
        assert _enum_values(fragment) == ["roadmap", "goal", "task"]

    def test_edit_node_advertises_kind_enum(self) -> None:
        fragment = _advertised_param("milknado_edit_node", "kind")
        assert _enum_values(fragment) == ["roadmap", "goal", "task"]

    def test_graph_summary_advertises_status_and_kind_enums(self) -> None:
        status = _advertised_param("milknado_graph_summary", "status")
        kind = _advertised_param("milknado_graph_summary", "kind")
        assert _enum_values(status) == ["pending", "in_progress", "blocked", "done"]
        assert _enum_values(kind) == ["roadmap", "goal", "task"]


class TestFileHintsMcp:
    """Round-trip tests: files param on MCP tools writes hints that render in the brief."""

    def test_todo_add_with_files_renders_in_brief(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(
            milknado_todo_add,
            description="touch config",
            files=["src/config.py"],
            project_root=root,
        )
        result = _call(milknado_todo_brief, node_id=task["id"], project_root=root)
        assert "src/config.py" in result["brief"]
        assert "(no file hints registered)" not in result["brief"]

    def test_todo_add_files_none_leaves_no_hints(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(
            milknado_todo_add,
            description="no hints",
            files=None,
            project_root=root,
        )
        result = _call(milknado_todo_brief, node_id=task["id"], project_root=root)
        assert "(no file hints registered)" in result["brief"]

    def test_edit_node_with_files_renders_in_brief(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="do work", project_root=root)
        _call(
            milknado_edit_node,
            node_id=task["id"],
            description="do work",
            files=["src/work.py", "tests/test_work.py"],
            project_root=root,
        )
        result = _call(milknado_todo_brief, node_id=task["id"], project_root=root)
        assert "src/work.py" in result["brief"]
        assert "tests/test_work.py" in result["brief"]
        assert "(no file hints registered)" not in result["brief"]

    def test_edit_node_files_none_leaves_hints_untouched(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="with hints", project_root=root)
        _call(
            milknado_edit_node,
            node_id=task["id"],
            description="with hints",
            files=["src/initial.py"],
            project_root=root,
        )
        # Now edit with files=None — hints must be preserved
        _call(
            milknado_edit_node,
            node_id=task["id"],
            description="updated desc",
            files=None,
            project_root=root,
        )
        result = _call(milknado_todo_brief, node_id=task["id"], project_root=root)
        assert "src/initial.py" in result["brief"]

    def test_edit_node_files_empty_clears_hints(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="clearable", project_root=root)
        _call(
            milknado_edit_node,
            node_id=task["id"],
            description="clearable",
            files=["src/temp.py"],
            project_root=root,
        )
        # Clear with files=[]
        _call(
            milknado_edit_node,
            node_id=task["id"],
            description="clearable",
            files=[],
            project_root=root,
        )
        result = _call(milknado_todo_brief, node_id=task["id"], project_root=root)
        assert "(no file hints registered)" in result["brief"]

    def test_track_follow_up_with_files_renders_in_brief(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        follow_up = _call(
            milknado_track_follow_up,
            description="follow up work",
            files=["src/followup.py"],
            project_root=root,
        )
        result = _call(milknado_todo_brief, node_id=follow_up["id"], project_root=root)
        assert "src/followup.py" in result["brief"]
        assert "(no file hints registered)" not in result["brief"]

    def test_edit_node_files_only_no_description_or_kind(self, tmp_path: Path) -> None:
        # Verify files-only edit doesn't crash: update_node must be skipped, not called
        # with (None, None) which would raise "nothing to update" from _mutations.
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="original", project_root=root)
        result = _call(
            milknado_edit_node,
            node_id=task["id"],
            files=["src/added.py"],
            project_root=root,
        )
        # Description unchanged, file hint written
        assert result["description"] == "original"
        brief = _call(milknado_todo_brief, node_id=task["id"], project_root=root)
        assert "src/added.py" in brief["brief"]

    def test_edit_node_nothing_raises_mentions_files(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="t", project_root=root)
        with pytest.raises(ValueError, match="files"):
            _call(milknado_edit_node, node_id=task["id"], project_root=root)

    def test_edit_node_files_only_missing_node_raises_value_error(self, tmp_path: Path) -> None:
        # When only files is provided for a missing node, must raise ValueError (not
        # IntegrityError from the FK constraint on file_ownership).
        root = str(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            _call(
                milknado_edit_node,
                node_id=9999,
                files=["src/ghost.py"],
                project_root=root,
            )

    def test_todo_add_files_empty_list_leaves_no_hints(self, tmp_path: Path) -> None:
        # files=[] is different from files=None: it explicitly passes an empty list.
        # No hints should be registered (set_file_ownership with [] deletes any existing).
        root = str(tmp_path)
        task = _call(
            milknado_todo_add,
            description="empty-files",
            files=[],
            project_root=root,
        )
        result = _call(milknado_todo_brief, node_id=task["id"], project_root=root)
        assert "(no file hints registered)" in result["brief"]

    def test_brief_renders_relevant_files_heading(self, tmp_path: Path) -> None:
        # The section heading must be present so the worker brief is well-structured.
        root = str(tmp_path)
        task = _call(
            milknado_todo_add,
            description="heading check",
            files=["src/x.py"],
            project_root=root,
        )
        result = _call(milknado_todo_brief, node_id=task["id"], project_root=root)
        assert "## Relevant files" in result["brief"]
        assert "- src/x.py" in result["brief"]

    def test_todo_add_multiple_files_order_preserved_in_brief(self, tmp_path: Path) -> None:
        # Insertion order must be preserved end-to-end through the brief renderer.
        root = str(tmp_path)
        task = _call(
            milknado_todo_add,
            description="ordered hints",
            files=["alpha.py", "beta.py", "gamma.py"],
            project_root=root,
        )
        result = _call(milknado_todo_brief, node_id=task["id"], project_root=root)
        brief = result["brief"]
        assert brief.index("alpha.py") < brief.index("beta.py") < brief.index("gamma.py")

    def test_edit_node_rejected_files_leaves_description_unchanged(self, tmp_path: Path) -> None:
        # Validation must run before any write: a rejected files hint must not
        # half-apply the edit, or a retrying caller double-applies.
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="original", project_root=root)
        with pytest.raises(ValueError, match="escapes"):
            _call(
                milknado_edit_node,
                node_id=task["id"],
                description="mutated",
                files=["../outside.py"],
                project_root=root,
            )
        node = _call(milknado_get_node, node_id=task["id"], project_root=root)
        assert node["description"] == "original"


def test_main_imports_all_tool_modules() -> None:
    """main() must import every tool module — registration is an @mcp.tool() import side effect.

    Checks main()'s source for each module name (word-bounded, so mcp_todo does
    not match inside mcp_todo_mutate): the suite imports the tool modules itself,
    so registration-based checks cannot detect a module dropped from main().
    """
    import inspect
    import re
    from unittest.mock import patch

    from milknado import mcp_server

    src = inspect.getsource(mcp_server.main)
    for module in ("mcp_ralph", "mcp_run", "mcp_todo", "mcp_todo_mutate"):
        assert re.search(rf"\b{module}\b", src), (
            f"main() no longer imports {module}; its tools won't register on server startup"
        )

    with patch("milknado.mcp_server.mcp") as mock_mcp:
        mock_mcp.run.return_value = None
        mcp_server.main()

    mock_mcp.run.assert_called_once()


# ── mcp-dispatch-consolidation hardening ────────────────────────────────────


def test_mcp_tool_modules_register_expected_tool_names() -> None:
    """The four main()-imported tool modules plus mcp_server must register the expected tool set.

    Pins the sorted name list so silently dropping mcp_todo_mutate (or any other
    module from main()) is caught: the count and the names both fail.
    The mcp instance is a test-session singleton; mcp_server's two module-level
    tools (milknado_graph_summary, milknado_plan_batches) are registered at import
    time of mcp_server which the test suite itself imports, so they appear here too.
    """
    from milknado import mcp_ralph, mcp_run, mcp_todo, mcp_todo_mutate  # noqa: F401
    from milknado._mcp_core import mcp

    tools = asyncio.run(mcp.list_tools())
    names = sorted(t.name for t in tools)
    expected = [
        "milknado_delete_node",
        "milknado_deposit_result",
        "milknado_edit_node",
        "milknado_get_node",
        "milknado_graph_summary",
        "milknado_move_node",
        "milknado_plan_batches",
        "milknado_ralph_run_poll",
        "milknado_ralph_run_start",
        "milknado_run_cancel",
        "milknado_run_list",
        "milknado_set_subtree_status",
        "milknado_todo_add",
        "milknado_todo_brief",
        "milknado_todo_next",
        "milknado_todo_run",
        "milknado_todo_run_poll",
        "milknado_todo_run_start",
        "milknado_todo_set_status",
        "milknado_todo_tree",
        "milknado_track_follow_up",
    ]
    assert names == expected, f"registered tool names changed; got {names}, expected {expected}"


def test_dispatch_crust_exports_new_public_names() -> None:
    """domains.dispatch.__all__ must export the three names added by the consolidation.

    cancel_run, dispatch_node_sync, reclaim_stale_node were moved from mcp_run.py
    into the dispatch domain and added to the crust; dropping any of them from
    __all__ would silently break callers that use the public API.
    """
    from milknado.domains import dispatch

    for name in ("cancel_run", "dispatch_node_sync", "reclaim_stale_node"):
        assert name in dispatch.__all__, (
            f"{name!r} missing from domains.dispatch.__all__; must be exported by the crust"
        )
        assert hasattr(dispatch, name), (
            f"{name!r} in __all__ but not importable from milknado.domains.dispatch"
        )
