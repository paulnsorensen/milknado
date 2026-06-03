"""Tests for #82 (milknado_run_list + milknado_run_cancel) and
#83 (unified RunDict superset schema across all five run tools).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import pytest

from milknado.mcp_ralph import milknado_ralph_run_poll, milknado_ralph_run_start
from milknado.mcp_run import (
    milknado_deposit_result,
    milknado_run_cancel,
    milknado_run_list,
    milknado_todo_run_poll,
    milknado_todo_run_start,
)
from milknado.mcp_server import open_graph
from milknado.mcp_todo import milknado_todo_add

_SUPERSET_KEYS = frozenset(
    {
        "run_id",
        "node_id",
        "status",
        "exit_code",
        "timed_out",
        "rebased",
        "log_path",
        "summary",
    }
)

_STUB_RUNNER = """
import argparse
from pathlib import Path
from milknado._mcp_core import open_graph
p = argparse.ArgumentParser()
p.add_argument("--node-id", type=int)
p.add_argument("--project-root")
p.add_argument("--run-id")
p.add_argument("--timeout")
a = p.parse_args()
graph, _cfg = open_graph(Path(a.project_root))
try:
    graph.finish_run(
        a.run_id,
        status="{status}",
        exit_code={exit_code},
        timed_out=False,
        ended_at="2026-01-01T00:00:00+00:00",
        rebased={rebased},
        detail=None,
    )
finally:
    graph.close()
"""


def _call(tool, **kwargs):
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
    ended_at: str | None = None,
    exit_code: int | None = None,
    timed_out: bool = False,
    error: str | None = None,
    pid: int | None = None,
) -> None:
    """Insert a runs row directly (replaces writing a .state.json sidecar)."""
    from datetime import UTC, datetime

    graph, _cfg = open_graph(root)
    try:
        graph._conn.execute(
            "INSERT OR IGNORE INTO nodes (id, description, status, created_at) "
            "VALUES (?, ?, 'running', ?)",
            (node_id, f"seeded-{node_id}", datetime.now(UTC).isoformat()),
        )
        started_at = started_at or datetime.now(UTC).isoformat()
        log_path = str(root / ".milknado" / "runs" / f"{run_id}.log")
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
        graph._conn.commit()
    finally:
        graph.close()


def _read_run(root: Path, run_id: str) -> dict | None:
    graph, _cfg = open_graph(root)
    try:
        return graph.get_run(run_id)
    finally:
        graph.close()


def _node_running_runs(root: Path, node_id: int) -> list[dict]:
    """Return this node's 'running' run rows (replaces a runs-dir glob)."""
    graph, _cfg = open_graph(root)
    try:
        return [r for r in graph.runs_for_node(node_id) if r["status"] == "running"]
    finally:
        graph.close()


def _stub_runner_cmd(tmp_path: Path, *, status: str, rebased: bool) -> str:
    script = tmp_path / f"stub_{status}.py"
    script.write_text(
        _STUB_RUNNER.format(status=status, rebased=rebased, exit_code=0 if status == "done" else 1)
    )
    return f"{sys.executable} {script}"


@pytest.fixture()
def worker_stub(tmp_path_factory, monkeypatch):
    """Same pattern as test_mcp_server.py: a `claude` shim that exec's its args."""
    bindir = tmp_path_factory.mktemp("worker-stub-bin")
    stub = bindir / "claude"
    stub.write_text('#!/bin/sh\nexec "$@"\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")

    def _cmd(command: str) -> str:
        return f"claude {command}"

    return _cmd


def _wait_for_terminal(run_id: str, root: str, tool, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = _call(tool, run_id=run_id, project_root=root)
        if last["status"] in ("done", "failed"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish; last={last}")


# ---------------------------------------------------------------------------
# #83 — Unified RunDict superset schema
# ---------------------------------------------------------------------------


class TestUnifiedRunSchema:
    """Every run tool must return a dict that contains all superset keys."""

    def test_todo_run_start_returns_superset_schema(
        self, tmp_path: Path, monkeypatch, worker_stub
    ) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="schema-start", kind="task", project_root=root)
        monkeypatch.setenv("MILKNADO_WORKER_CMD", worker_stub("cat"))
        result = _call(milknado_todo_run_start, node_id=task["id"], project_root=root)
        missing = _SUPERSET_KEYS - result.keys()
        assert not missing, f"milknado_todo_run_start missing keys: {missing}"
        # start tools return None for fields only available post-completion
        assert result["status"] == "running"
        assert result["run_id"] is not None
        assert result["exit_code"] is None
        assert result["timed_out"] is None
        assert result["rebased"] is None
        assert result["summary"] is None

    def test_todo_run_poll_returns_superset_schema(
        self, tmp_path: Path, monkeypatch, worker_stub
    ) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="schema-poll", kind="task", project_root=root)
        monkeypatch.setenv("MILKNADO_WORKER_CMD", worker_stub("cat"))
        started = _call(milknado_todo_run_start, node_id=task["id"], project_root=root)
        final = _wait_for_terminal(started["run_id"], root, milknado_todo_run_poll)
        missing = _SUPERSET_KEYS - final.keys()
        assert not missing, f"milknado_todo_run_poll missing keys: {missing}"
        assert final["run_id"] == started["run_id"]
        assert "rebased" in final  # may be None for headless runs

    def test_ralph_run_start_returns_superset_schema(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(
            milknado_todo_add, description="schema-ralph-start", kind="task", project_root=root
        )
        result = _call(
            milknado_ralph_run_start,
            node_id=task["id"],
            runner_cmd=f"{sys.executable} -c pass",
            project_root=root,
        )
        missing = _SUPERSET_KEYS - result.keys()
        assert not missing, f"milknado_ralph_run_start missing keys: {missing}"
        assert result["status"] == "running"
        assert result["exit_code"] is None
        assert result["timed_out"] is None
        assert result["rebased"] is None
        assert result["summary"] is None

    def test_ralph_run_poll_returns_superset_schema(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        task = _call(
            milknado_todo_add, description="schema-ralph-poll", kind="task", project_root=root
        )
        started = _call(
            milknado_ralph_run_start,
            node_id=task["id"],
            runner_cmd=_stub_runner_cmd(tmp_path, status="done", rebased=True),
            project_root=root,
        )
        final = _wait_for_terminal(started["run_id"], root, milknado_ralph_run_poll)
        missing = _SUPERSET_KEYS - final.keys()
        assert not missing, f"milknado_ralph_run_poll missing keys: {missing}"
        assert final["rebased"] is True


class TestSyncRunSchema:
    """milknado_todo_run (sync) must also return the superset schema and write a state file."""

    def test_todo_run_returns_superset_schema(self, tmp_path: Path, monkeypatch) -> None:
        import milknado.domains.dispatch.runner as runner_mod

        def _stub_execute(project_root, node_id, log_path, brief, argv, timeout, run_id=None):
            log_path.write_bytes(b"ok")
            return 0, False

        monkeypatch.setattr(runner_mod, "_execute", _stub_execute)

        from milknado.mcp_run import milknado_todo_run

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="sync-schema", kind="task", project_root=root)
        result = _call(milknado_todo_run, node_id=task["id"], project_root=root)
        missing = _SUPERSET_KEYS - result.keys()
        assert not missing, f"milknado_todo_run missing keys: {missing}"
        assert result["run_id"] is not None
        assert result["rebased"] is None
        assert result["exit_code"] == 0
        assert result["timed_out"] is False

    def test_todo_run_writes_run_row(self, tmp_path: Path, monkeypatch) -> None:
        import milknado.domains.dispatch.runner as runner_mod

        def _stub_execute(project_root, node_id, log_path, brief, argv, timeout, run_id=None):
            log_path.write_bytes(b"ok")
            return 0, False

        monkeypatch.setattr(runner_mod, "_execute", _stub_execute)

        from milknado.mcp_run import milknado_todo_run

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="sync-state", kind="task", project_root=root)
        result = _call(milknado_todo_run, node_id=task["id"], project_root=root)
        state = _read_run(tmp_path, result["run_id"])
        assert state is not None, "sync run must write a terminal run row"
        assert state["run_id"] == result["run_id"]
        assert state["status"] == "done"

    def test_todo_run_sets_run_id_on_node(self, tmp_path: Path, monkeypatch) -> None:
        """#40: run_id must be threaded into the node after a sync run."""
        import milknado.domains.dispatch.runner as runner_mod

        def _stub_execute(project_root, node_id, log_path, brief, argv, timeout, run_id=None):
            log_path.write_bytes(b"ok")
            return 0, False

        monkeypatch.setattr(runner_mod, "_execute", _stub_execute)

        from milknado.mcp_run import milknado_todo_run

        root = str(tmp_path)
        task = _call(
            milknado_todo_add, description="run-id-on-node", kind="task", project_root=root
        )
        result = _call(milknado_todo_run, node_id=task["id"], project_root=root)
        graph, _cfg = open_graph(tmp_path)
        try:
            node = graph.get_node(task["id"])
            assert node is not None
            assert node.run_id == result["run_id"]
        finally:
            graph.close()


class TestSyncRunOrphanRescue:
    """#42: a sync run must write a rescuable 'running' state file BEFORE it
    blocks in the worker. A client timeout can orphan the call (server killed
    mid-run, no terminal write ever lands); without the up-front file the node
    stays RUNNING forever, since fail_stale_running_runs has nothing to find.
    The async path already gets this via start_headless_async."""

    def test_sync_run_writes_rescuable_running_state_before_worker(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import milknado.domains.dispatch.runner as runner_mod

        observed: dict = {}

        def _stub_execute(project_root, node_id, log_path, brief, argv, timeout, run_id=None):
            # Snapshot the run row WHILE the worker runs — the exact window a
            # client timeout orphans the sync call in.
            rows = _node_running_runs(Path(project_root), node_id)
            if rows:
                observed["state"] = rows[0]
            log_path.write_bytes(b"ok")
            return 0, False

        monkeypatch.setattr(runner_mod, "_execute", _stub_execute)

        from milknado.mcp_run import milknado_todo_run

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="orphan", kind="task", project_root=root)
        _call(milknado_todo_run, node_id=task["id"], project_root=root)

        state = observed.get("state")
        assert state is not None, "no 'running' run row existed while the worker ran"
        assert state["status"] == "running"
        # fail_stale_running_runs keys on both fields; missing either makes an
        # orphaned run unrescuable.
        assert state["started_at"]
        assert state["timeout_seconds"] is not None

    def test_orphaned_sync_run_is_rescuable(self, tmp_path: Path, monkeypatch) -> None:
        from datetime import UTC, datetime, timedelta

        import milknado.domains.dispatch.runner as runner_mod

        def _crash_execute(project_root, node_id, log_path, brief, argv, timeout, run_id=None):
            raise RuntimeError("server killed mid-run before terminal write")

        monkeypatch.setattr(runner_mod, "_execute", _crash_execute)

        from milknado.mcp_run import milknado_todo_run

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="orphan-stale", kind="task", project_root=root)
        with pytest.raises(RuntimeError):
            _call(milknado_todo_run, node_id=task["id"], project_root=root)

        rows = _node_running_runs(tmp_path, task["id"])
        assert rows, "orphaned sync run left no running run row to rescue"
        state = rows[0]
        assert state["status"] == "running"
        # Age the run past its timeout and the stale sweep must release the node.
        graph, _cfg = open_graph(tmp_path)
        try:
            aged = datetime.now(UTC) - timedelta(seconds=state["timeout_seconds"] + 3600)
            graph._conn.execute(
                "UPDATE runs SET started_at = ? WHERE run_id = ?",
                (aged.isoformat(), state["run_id"]),
            )
            graph._conn.commit()
            flipped = runner_mod.fail_stale_running_runs(graph, task["id"])
        finally:
            graph.close()
        assert len(flipped) == 1
        assert flipped[0]["status"] == "failed"

    def test_done_node_rerun_writes_no_running_state(self, tmp_path: Path, monkeypatch) -> None:
        """The running-state write is gated on an actual PENDING/FAILED->RUNNING
        transition. A DONE node re-run must NOT drop a 'running' run the stale
        sweep could later force-fail, which would clobber the prior DONE result."""
        import milknado.domains.dispatch.runner as runner_mod

        observed: dict = {}

        def _stub_execute(project_root, node_id, log_path, brief, argv, timeout, run_id=None):
            observed["running"] = _node_running_runs(Path(project_root), node_id)
            log_path.write_bytes(b"ok")
            return 0, False

        monkeypatch.setattr(runner_mod, "_execute", _stub_execute)

        from milknado.mcp_run import milknado_todo_run

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="done-rerun", kind="task", project_root=root)
        _call(milknado_todo_run, node_id=task["id"], project_root=root)  # -> DONE
        # Second run sees a DONE node: _ensure_running returns False, so the gate
        # must skip the running-run write — no 'running' run exists mid-run.
        _call(milknado_todo_run, node_id=task["id"], project_root=root)
        assert observed["running"] == [], "DONE-node re-run must not write a 'running' run row"


# ---------------------------------------------------------------------------
# #82 — milknado_run_list
# ---------------------------------------------------------------------------


class TestRunList:
    def test_empty_when_no_runs(self, tmp_path: Path) -> None:
        result = _call(milknado_run_list, project_root=str(tmp_path))
        assert result == []

    def test_lists_state_files_after_async_run(
        self, tmp_path: Path, monkeypatch, worker_stub
    ) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="list-test", kind="task", project_root=root)
        monkeypatch.setenv("MILKNADO_WORKER_CMD", worker_stub("cat"))
        started = _call(milknado_todo_run_start, node_id=task["id"], project_root=root)
        _wait_for_terminal(started["run_id"], root, milknado_todo_run_poll)
        runs = _call(milknado_run_list, project_root=root)
        assert len(runs) >= 1
        run_ids = [r["run_id"] for r in runs]
        assert started["run_id"] in run_ids

    def test_returns_superset_keys(self, tmp_path: Path, monkeypatch, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="list-keys", kind="task", project_root=root)
        monkeypatch.setenv("MILKNADO_WORKER_CMD", worker_stub("cat"))
        started = _call(milknado_todo_run_start, node_id=task["id"], project_root=root)
        _wait_for_terminal(started["run_id"], root, milknado_todo_run_poll)
        runs = _call(milknado_run_list, project_root=root)
        assert len(runs) >= 1
        missing = _SUPERSET_KEYS - runs[0].keys()
        assert not missing, f"run_list entry missing keys: {missing}"

    def test_sorted_newest_first(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        older_id = "node-1-20260101T000000Z-aaaa"
        newer_id = "node-1-20260101T010000Z-bbbb"
        for rid, started_at in [
            (older_id, "2026-01-01T00:00:00+00:00"),
            (newer_id, "2026-01-01T01:00:00+00:00"),
        ]:
            _seed_run(
                tmp_path,
                run_id=rid,
                node_id=1,
                status="done",
                started_at=started_at,
                ended_at=started_at,
                exit_code=0,
            )
        runs = _call(milknado_run_list, project_root=root)
        assert runs[0]["run_id"] == newer_id
        assert runs[1]["run_id"] == older_id


# ---------------------------------------------------------------------------
# #82 — milknado_run_cancel
# ---------------------------------------------------------------------------


class TestRunCancel:
    @pytest.fixture(autouse=True)
    def _fast_cancel_finalize(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No-pid states hand-written here have no live worker, so cancel falls
        back after the finalize bound. Shorten it so these tests stay fast."""
        monkeypatch.setattr("milknado.mcp_run._CANCEL_FINALIZE_TIMEOUT_SECS", 0.3)

    def test_cancel_unknown_run_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(
                milknado_run_cancel,
                run_id="node-1-20260101T000000Z-abcd",
                project_root=str(tmp_path),
            )

    def test_cancel_malformed_run_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid run_id"):
            _call(milknado_run_cancel, run_id="../etc/passwd", project_root=str(tmp_path))

    def test_cancel_terminal_run_returns_without_kill(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_id = "node-1-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=1, status="done", exit_code=0)
        killed = []
        monkeypatch.setattr(os, "killpg", lambda *a: killed.append(a))
        result = _call(milknado_run_cancel, run_id=run_id, project_root=str(tmp_path))
        assert result["status"] == "done"
        assert killed == [], "must not kill a terminal run"

    def test_cancel_signals_process_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import signal as sig_mod

        run_id = "node-1-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=1, status="running", pid=9999)
        killed: list[tuple] = []
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        _call(milknado_run_cancel, run_id=run_id, project_root=str(tmp_path))
        assert killed == [(9999, sig_mod.SIGTERM)], "must send SIGTERM to process group"

    def test_cancel_writes_failed_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_id = "node-1-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=1, status="running")
        monkeypatch.setattr(os, "killpg", lambda *a: None)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        result = _call(milknado_run_cancel, run_id=run_id, project_root=str(tmp_path))
        assert result["status"] == "failed"
        assert result["exit_code"] == -1
        on_disk = _read_run(tmp_path, run_id)
        assert on_disk["status"] == "failed"
        assert on_disk["error"] == "cancelled"

    def test_cancel_returns_superset_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_id = "node-1-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=1, status="running")
        monkeypatch.setattr(os, "killpg", lambda *a: None)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        result = _call(milknado_run_cancel, run_id=run_id, project_root=str(tmp_path))
        missing = _SUPERSET_KEYS - result.keys()
        assert not missing, f"run_cancel missing keys: {missing}"

    def test_cancel_reconciles_node_to_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="cancel-node", kind="task", project_root=root)
        node_id = task["id"]
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.mark_running(node_id)
        finally:
            graph.close()
        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=node_id, status="running")
        monkeypatch.setattr(os, "killpg", lambda *a: None)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        _call(milknado_run_cancel, run_id=run_id, project_root=root)
        graph2, _cfg2 = open_graph(tmp_path)
        try:
            node = graph2.get_node(node_id)
            assert node is not None
            assert node.status.value == "failed", "cancel must reconcile node to failed"
        finally:
            graph2.close()

    def test_cancel_prunes_worktree_and_calls_prune(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Acceptance: cancel must remove the orphaned worktree AND call git worktree prune."""
        import milknado.adapters as adapters

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="cancel-wt", kind="task", project_root=root)
        node_id = task["id"]

        orphan_wt = tmp_path / "milknado-cancel-wt"
        orphan_wt.mkdir()

        graph, _cfg = open_graph(tmp_path)
        try:
            graph.mark_running(
                node_id, worktree_path=str(orphan_wt), branch_name="milknado/cancel"
            )
        finally:
            graph.close()

        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=node_id, status="running")

        removed: list[Path] = []
        pruned: list[bool] = []

        monkeypatch.setattr(
            adapters.GitAdapter, "remove_worktree", lambda self, wt: removed.append(wt)
        )
        monkeypatch.setattr(
            adapters.GitAdapter, "prune_worktrees", lambda self: pruned.append(True)
        )
        monkeypatch.setattr(os, "killpg", lambda *a: None)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)

        _call(milknado_run_cancel, run_id=run_id, project_root=root)

        assert removed == [orphan_wt], "orphaned worktree must be removed on cancel"
        assert pruned, "git worktree prune must be called after cancel"

    def test_sync_stuck_running_cancel_reconciles_and_clears(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run stuck "running" with no pid and no live worker (crashed sync run)
        must still be cleared: cancel falls back to writing the terminal state,
        reconciles the node, and leaves no stale sentinel."""
        from milknado.domains.dispatch import cancel_path
        from milknado.domains.dispatch import runs_dir as _runs_dir

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="sync-stuck", kind="task", project_root=root)
        node_id = task["id"]
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.mark_running(node_id)
        finally:
            graph.close()
        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=node_id, status="running")

        result = _call(milknado_run_cancel, run_id=run_id, project_root=root)

        assert result["status"] == "failed"
        assert result["exit_code"] == -1
        on_disk = _read_run(tmp_path, run_id)
        assert on_disk["status"] == "failed"
        assert on_disk["error"] == "cancelled"
        graph2, _cfg2 = open_graph(tmp_path)
        try:
            node = graph2.get_node(node_id)
            assert node is not None
            assert node.status.value == "failed"
        finally:
            graph2.close()
        assert not cancel_path(_runs_dir(tmp_path), run_id).exists(), "sentinel must be cleared"

    def test_cancel_logs_when_worktree_removal_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If `git worktree remove` raises mid-cancel, _reconcile_cancel logs a
        warning and still reconciles the node + prunes — a failed removal must
        never strand the node RUNNING."""
        import milknado.adapters as adapters

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="wt-fail", kind="task", project_root=root)
        node_id = task["id"]
        orphan_wt = tmp_path / "milknado-wt-fail"
        orphan_wt.mkdir()
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.mark_running(node_id, worktree_path=str(orphan_wt), branch_name="milknado/x")
        finally:
            graph.close()
        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=node_id, status="running")

        pruned: list[bool] = []

        def boom(self, wt):
            raise RuntimeError("worktree locked")

        monkeypatch.setattr(adapters.GitAdapter, "remove_worktree", boom)
        monkeypatch.setattr(
            adapters.GitAdapter, "prune_worktrees", lambda self: pruned.append(True)
        )
        monkeypatch.setattr(os, "killpg", lambda *a: None)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)

        with caplog.at_level(logging.WARNING, logger="milknado.mcp_run"):
            result = _call(milknado_run_cancel, run_id=run_id, project_root=root)

        assert result["status"] == "failed"
        assert any("Failed to remove worktree" in r.getMessage() for r in caplog.records), (
            "a failed worktree removal must be logged"
        )
        assert pruned, "prune must still run even when removal fails"
        graph2, _cfg2 = open_graph(tmp_path)
        try:
            node = graph2.get_node(node_id)
            assert node is not None
            assert node.status.value == "failed", "removal failure must not strand node RUNNING"
        finally:
            graph2.close()


class TestAsyncCancel:
    """The regression the review flagged: cancel must genuinely stop a real
    async-headless worker mid-run, not falsely mark it failed while the worker
    keeps running and later clobbers the cancelled state."""

    def test_cancel_stops_async_worker_and_finalizes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_stub
    ) -> None:
        from milknado.domains.dispatch import cancel_path
        from milknado.domains.dispatch import runs_dir as _runs_dir

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="async-cancel", kind="task", project_root=root)
        node_id = task["id"]
        # A genuinely mid-run worker: sleeps far longer than the test, so any
        # "failed" terminal state can only come from cooperative cancellation.
        monkeypatch.setenv("MILKNADO_WORKER_CMD", worker_stub("sleep 30"))
        started = _call(milknado_todo_run_start, node_id=node_id, project_root=root)
        run_id = started["run_id"]
        assert started["status"] == "running"

        result = _call(milknado_run_cancel, run_id=run_id, project_root=root)

        # (b) final state is cooperative-cancel terminal
        assert result["status"] == "failed"
        assert result["exit_code"] == -1
        on_disk = _read_run(tmp_path, run_id)
        assert on_disk["status"] == "failed"
        assert on_disk["error"] == "cancelled"

        # (a)+(c) the worker stopped and does NOT overwrite the cancelled state:
        # had it kept running, `sleep 30` would later flip the state to done.
        time.sleep(1.0)
        after = _read_run(tmp_path, run_id)
        assert after["status"] == "failed", "worker must not clobber the cancelled state"
        assert after["error"] == "cancelled"

        # (d) node reconciled to failed under the run_id fence
        graph, _cfg = open_graph(tmp_path)
        try:
            node = graph.get_node(node_id)
            assert node is not None
            assert node.status.value == "failed"
            # the fenced FAILED transition clears run_id (mark_terminal), proving
            # reconcile went through the run_id fence rather than stranding it
            assert node.run_id is None
        finally:
            graph.close()

        # sentinel cleared after the terminal write
        assert not cancel_path(_runs_dir(tmp_path), run_id).exists(), "sentinel must be cleared"

    def test_async_worker_timeout_marks_failed_and_timed_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_stub
    ) -> None:
        """The async poll loop (`_execute_cancellable`) enforces the run's own
        timeout deadline — distinct from the sync `_execute` `communicate(timeout=)`
        path. A worker that outlives its timeout must be terminated and finalized
        `failed` / `timed_out=True`, with no cancel sentinel involved."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="a-timeout", kind="task", project_root=root)
        node_id = task["id"]
        # Worker sleeps far past its 1s timeout, so a terminal state can only come
        # from the poll loop's deadline enforcement, not from the worker exiting.
        monkeypatch.setenv("MILKNADO_WORKER_CMD", worker_stub("sleep 30"))
        started = _call(
            milknado_todo_run_start, node_id=node_id, timeout_seconds=1, project_root=root
        )
        run_id = started["run_id"]

        final = _wait_for_terminal(run_id, root, milknado_todo_run_poll, timeout=8.0)

        assert final["status"] == "failed"
        assert final["timed_out"] is True, "timeout must be recorded, not a plain fail"
        # Timeout is not a cancellation: no sentinel was written, and the terminal
        # write must not masquerade as error='cancelled'.
        on_disk = _read_run(tmp_path, run_id)
        assert on_disk.get("error") != "cancelled"

    def test_async_cancel_done_in_window_reconciles_done_not_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression (age:correctness): when the worker finishes `done` inside the
        finalize window, cancel observes a `done` state — it must reconcile the node
        `done`, not force-fail a genuinely-completed run and discard its work."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="done-in-win", kind="task", project_root=root)
        node_id = task["id"]
        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.mark_running(node_id)
            graph.set_run_id(node_id, run_id)
        finally:
            graph.close()
        _seed_run(tmp_path, run_id=run_id, node_id=node_id, status="running")

        # The worker finalizes `done` during the finalize poll window: the run row
        # flips to done and that is what `_await_cancel_finalize` returns.
        def fake_finalize(graph, rid: str) -> dict:
            graph.finish_run(
                rid, status="done", exit_code=0, timed_out=False, ended_at="2026-01-01T00:00:00Z"
            )
            return graph.get_run(rid)

        monkeypatch.setattr("milknado.mcp_run._await_cancel_finalize", fake_finalize)

        result = _call(milknado_run_cancel, run_id=run_id, project_root=root)

        assert result["status"] == "done", "a run that finished done must be reported done"
        graph2, _cfg2 = open_graph(tmp_path)
        try:
            node = graph2.get_node(node_id)
            assert node is not None
            assert node.status.value == "done", "node must reconcile done, not be force-failed"
        finally:
            graph2.close()


class TestPidCancelReconcile:
    """The pid (detached-ralph) cancel branch. The cooked change routes its node
    reconcile through the run_id-fenced `_reconcile_cancel` — the deliberate
    deviation from the spec's "keep the pid path unchanged". These lock both the
    positive reconcile and the fence that the deviation exists to preserve."""

    def _running_node_owned_by(self, tmp_path: Path, run_id: str) -> int:
        """Create a RUNNING node fenced to `run_id` (the active owner)."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="pid-cancel", kind="task", project_root=root)
        node_id = task["id"]
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.mark_running(node_id)
            graph.set_run_id(node_id, run_id)
        finally:
            graph.close()
        return node_id

    def _write_pid_state(self, tmp_path: Path, run_id: str, node_id: int) -> None:
        _seed_run(tmp_path, run_id=run_id, node_id=node_id, status="running", pid=9999)

    def test_pid_cancel_reconciles_node_to_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A detached-ralph run (has pid) cancels: SIGTERM the group, write the
        terminal state, and reconcile the node to failed through the run_id fence.
        Previously only the async branch's reconcile had a real-node test."""
        import signal as sig_mod

        run_id = "node-1-20260101T000000Z-aaaa"
        node_id = self._running_node_owned_by(tmp_path, run_id)
        self._write_pid_state(tmp_path, run_id, node_id)

        killed: list[tuple] = []
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))

        result = _call(milknado_run_cancel, run_id=run_id, project_root=str(tmp_path))

        assert killed == [(9999, sig_mod.SIGTERM)], "pid branch must SIGTERM the process group"
        assert result["status"] == "failed"
        assert result["exit_code"] == -1
        graph2, _cfg2 = open_graph(tmp_path)
        try:
            node = graph2.get_node(node_id)
            assert node is not None
            assert node.status.value == "failed", "pid cancel must reconcile the node to failed"
            # FAILED via the fenced mark_terminal clears run_id — proves the
            # reconcile went through the fence, not a bare unconditional write.
            assert node.run_id is None
        finally:
            graph2.close()

    def test_pid_cancel_reconcile_is_fenced_against_reclaim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fence the deviation preserves: a stale cancel for run-A must NOT
        clobber a node already re-claimed under a newer run-B. A regression to an
        unfenced `mark_failed` here would walk the live run-B node to FAILED."""
        stale_run = "node-1-20260101T000000Z-aaaa"
        newer_run = "node-1-20260102T000000Z-bbbb"
        node_id = self._running_node_owned_by(tmp_path, stale_run)
        # Re-own the still-RUNNING node under the newer run, then cancel the stale one.
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.set_run_id(node_id, newer_run)
        finally:
            graph.close()
        self._write_pid_state(tmp_path, stale_run, node_id)

        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", lambda *a: None)

        _call(milknado_run_cancel, run_id=stale_run, project_root=str(tmp_path))

        graph2, _cfg2 = open_graph(tmp_path)
        try:
            node = graph2.get_node(node_id)
            assert node is not None
            assert node.status.value == "running", "stale cancel must not clobber reclaimed node"
            assert node.run_id == newer_run, "newer run's ownership must survive a stale cancel"
        finally:
            graph2.close()

    def test_pid_cancel_tolerates_already_gone_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A detached-ralph run whose process group is already gone (killpg raises
        ProcessLookupError) must still finalize cleanly: write the terminal state
        and reconcile the node, not propagate the signal error."""
        run_id = "node-1-20260101T000000Z-cccc"
        node_id = self._running_node_owned_by(tmp_path, run_id)
        self._write_pid_state(tmp_path, run_id, node_id)

        def gone(*_a):
            raise ProcessLookupError("no such process")

        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", gone)

        result = _call(milknado_run_cancel, run_id=run_id, project_root=str(tmp_path))

        assert result["status"] == "failed"
        assert result["exit_code"] == -1
        on_disk = _read_run(tmp_path, run_id)
        assert on_disk["status"] == "failed"
        assert on_disk["error"] == "cancelled"
        graph2, _cfg2 = open_graph(tmp_path)
        try:
            node = graph2.get_node(node_id)
            assert node is not None
            assert node.status.value == "failed", "cancel must finalize despite a dead pgroup"
        finally:
            graph2.close()


class TestCancelFinalizeAndRace:
    """Helper-level coverage for the async-cancel finalize path: the unreadable-
    state warning and the last-poll-gap race re-read that must not clobber a
    worker's genuine terminal write."""

    def test_await_cancel_finalize_returns_none_when_run_stays_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the run row stays 'running' for the whole finalize window (the worker
        is wedged or dead), the bound elapses to None so the caller takes over the
        terminal write itself."""
        from milknado.mcp_run import _await_cancel_finalize

        monkeypatch.setattr("milknado.mcp_run._CANCEL_FINALIZE_TIMEOUT_SECS", 0.05)
        monkeypatch.setattr("milknado.mcp_run._CANCEL_FINALIZE_POLL_SECS", 0.01)

        run_id = "node-1-20260101T000000Z-wedg"
        _seed_run(tmp_path, run_id=run_id, node_id=1, status="running")
        graph, _cfg = open_graph(tmp_path)
        try:
            out = _await_cancel_finalize(graph, run_id)
        finally:
            graph.close()
        assert out is None, "a run still 'running' for the whole window must time out to None"

    def test_async_cancel_race_reread_adopts_last_gap_finalize(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The race fence: _await_cancel_finalize returns None (bound elapsed) but
        the worker wrote its terminal run row in the final poll gap. Cancel must
        re-read and ADOPT that write, not overwrite a genuine `done` with
        `cancelled`."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="race", kind="task", project_root=root)
        node_id = task["id"]
        run_id = f"node-{node_id}-20260101T000000Z-dddd"
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.mark_running(node_id)
            graph.set_run_id(node_id, run_id)
        finally:
            graph.close()
        _seed_run(tmp_path, run_id=run_id, node_id=node_id, status="running")

        def elapsed_after_done(graph, rid: str):
            graph.finish_run(
                rid, status="done", exit_code=0, timed_out=False, ended_at="2026-01-01T00:00:00Z"
            )
            return None

        monkeypatch.setattr("milknado.mcp_run._await_cancel_finalize", elapsed_after_done)

        result = _call(milknado_run_cancel, run_id=run_id, project_root=root)

        assert result["status"] == "done", "must adopt the worker's last-gap terminal write"
        on_disk = _read_run(tmp_path, run_id)
        assert on_disk["status"] == "done", "must not clobber the worker's done write"
        graph2, _cfg2 = open_graph(tmp_path)
        try:
            node = graph2.get_node(node_id)
            assert node is not None
            assert node.status.value == "done", "node reconciles to the run's real status"
        finally:
            graph2.close()


class TestTerminateWorker:
    """_terminate_worker's SIGKILL escalation: a worker that ignores SIGTERM must
    be force-killed after the grace window (runner.py:196-198)."""

    def test_terminate_worker_escalates_to_sigkill_when_sigterm_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import signal as sig_mod
        import subprocess as sp_mod

        import milknado.domains.dispatch.runner as runner_mod

        monkeypatch.setattr(runner_mod, "_CANCEL_GRACE_SECS", 0.2)
        ready = tmp_path / "handler-ready"
        # Trap+ignore SIGTERM, signal readiness, then sleep far past the grace
        # window: the only way this worker dies is the kill() escalation.
        code = (
            "import signal, time, pathlib; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"pathlib.Path({str(ready)!r}).write_text('ok'); "
            "time.sleep(30)"
        )
        proc = sp_mod.Popen([sys.executable, "-c", code])
        deadline = time.monotonic() + 5.0
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "worker did not install its SIGTERM handler in time"
        try:
            runner_mod._terminate_worker(proc)
            # _terminate_worker must have reaped the process itself — capture the
            # code BEFORE the cleanup net so a regressed escalation can't be masked.
            rc = proc.returncode
        finally:
            if proc.poll() is None:  # cleanup net; only fires if escalation regressed
                proc.kill()
                proc.wait()

        assert rc == -sig_mod.SIGKILL, (
            "a worker ignoring SIGTERM must be escalated to SIGKILL after the grace window"
        )


# ---------------------------------------------------------------------------
# #122 — milknado_deposit_result + result round-trip
# ---------------------------------------------------------------------------


class TestDepositResult:
    def test_deposit_rejects_malformed_run_id(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid run_id"):
            _call(
                milknado_deposit_result,
                run_id="../etc/passwd",
                payload="x",
                project_root=str(tmp_path),
            )

    def test_deposit_unknown_run_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(
                milknado_deposit_result,
                run_id="node-1-20260101T000000Z-abcd",
                payload="x",
                project_root=str(tmp_path),
            )

    def test_deposit_persists_and_returns_seq(self, tmp_path: Path) -> None:
        run_id = "node-1-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=1, status="running")
        first = _call(
            milknado_deposit_result, run_id=run_id, payload="hello", project_root=str(tmp_path)
        )
        assert first == {"run_id": run_id, "seq": 1}
        second = _call(
            milknado_deposit_result, run_id=run_id, payload="again", project_root=str(tmp_path)
        )
        assert second["seq"] == 2, "seq must increment per deposit"
        graph, _cfg = open_graph(tmp_path)
        try:
            assert graph.latest_run_message(run_id, "result") == "again"
        finally:
            graph.close()

    def test_todo_run_poll_returns_deposited_result(
        self, tmp_path: Path, monkeypatch, worker_stub
    ) -> None:
        """The async-run poll must surface the deposited result under `result`."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="deposit-poll", kind="task", project_root=root)
        monkeypatch.setenv("MILKNADO_WORKER_CMD", worker_stub("cat"))
        started = _call(milknado_todo_run_start, node_id=task["id"], project_root=root)
        _wait_for_terminal(started["run_id"], root, milknado_todo_run_poll)
        _call(
            milknado_deposit_result,
            run_id=started["run_id"],
            payload="the deliverable",
            project_root=root,
        )
        final = _call(milknado_todo_run_poll, run_id=started["run_id"], project_root=root)
        assert final["result"] == "the deliverable"

    def test_poll_result_is_none_without_deposit(
        self, tmp_path: Path, monkeypatch, worker_stub
    ) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="no-deposit", kind="task", project_root=root)
        monkeypatch.setenv("MILKNADO_WORKER_CMD", worker_stub("cat"))
        started = _call(milknado_todo_run_start, node_id=task["id"], project_root=root)
        final = _wait_for_terminal(started["run_id"], root, milknado_todo_run_poll)
        assert final["result"] is None, "no deposit -> result is None, not a missing key"

    def test_multipart_deliverable_round_trips_intact(self, tmp_path: Path) -> None:
        """Dogfood shape (#122): a multi-part deliverable — several before/after
        items — deposited as one payload must come back whole through the poll,
        not truncated to a 14-line log tail."""
        run_id = "node-1-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=1, status="running")
        payload = "\n".join(f"item {i}: before X{i} -> after Y{i}" for i in range(40))
        _call(milknado_deposit_result, run_id=run_id, payload=payload, project_root=str(tmp_path))
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.finish_run(
                run_id,
                status="done",
                exit_code=0,
                timed_out=False,
                ended_at="2026-01-01T00:00:00Z",
            )
            graph._conn.commit()
        finally:
            graph.close()
        final = _call(milknado_todo_run_poll, run_id=run_id, project_root=str(tmp_path))
        assert final["result"] == payload
        assert final["result"].count("\n") == 39, "every line of the deliverable must survive"
