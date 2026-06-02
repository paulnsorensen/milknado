"""Tests for #82 (milknado_run_list + milknado_run_cancel) and
#83 (unified RunDict superset schema across all five run tools).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import pytest

from milknado.mcp_ralph import milknado_ralph_run_poll, milknado_ralph_run_start
from milknado.mcp_run import (
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
        "state_path",
        "summary",
    }
)

_STUB_RUNNER = """
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--node-id", type=int)
p.add_argument("--project-root")
p.add_argument("--run-id")
p.add_argument("--state-path")
p.add_argument("--timeout")
a = p.parse_args()
sp = Path(a.state_path)
state = json.loads(sp.read_text())
state.update(status="{status}", rebased={rebased}, detail=None)
state["ended_at"] = "2026-01-01T00:00:00+00:00"
sp.write_text(json.dumps(state))
"""


def _call(tool, **kwargs):
    fn = getattr(tool, "fn", tool)
    return fn(**kwargs)


def _stub_runner_cmd(tmp_path: Path, *, status: str, rebased: bool) -> str:
    script = tmp_path / f"stub_{status}.py"
    script.write_text(_STUB_RUNNER.format(status=status, rebased=rebased))
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
        assert result["state_path"] is not None

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
        assert final["state_path"] is not None
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
        assert final["state_path"] is not None
        assert final["rebased"] is True


class TestSyncRunSchema:
    """milknado_todo_run (sync) must also return the superset schema and write a state file."""

    def test_todo_run_returns_superset_schema(self, tmp_path: Path, monkeypatch) -> None:
        import milknado.domains.dispatch.runner as runner_mod

        def _stub_execute(project_root, node_id, log_path, brief, argv, timeout):
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
        assert result["state_path"] is not None
        assert result["rebased"] is None
        assert result["exit_code"] == 0
        assert result["timed_out"] is False

    def test_todo_run_writes_state_file(self, tmp_path: Path, monkeypatch) -> None:
        import milknado.domains.dispatch.runner as runner_mod

        def _stub_execute(project_root, node_id, log_path, brief, argv, timeout):
            log_path.write_bytes(b"ok")
            return 0, False

        monkeypatch.setattr(runner_mod, "_execute", _stub_execute)

        from milknado.mcp_run import milknado_todo_run

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="sync-state", kind="task", project_root=root)
        result = _call(milknado_todo_run, node_id=task["id"], project_root=root)
        state_path = Path(result["state_path"])
        assert state_path.exists(), "sync run must write a state file"
        state = json.loads(state_path.read_text())
        assert state["run_id"] == result["run_id"]
        assert state["status"] == "done"

    def test_todo_run_sets_run_id_on_node(self, tmp_path: Path, monkeypatch) -> None:
        """#40: run_id must be threaded into the node after a sync run."""
        import milknado.domains.dispatch.runner as runner_mod

        def _stub_execute(project_root, node_id, log_path, brief, argv, timeout):
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

        def _stub_execute(project_root, node_id, log_path, brief, argv, timeout):
            # Snapshot on-disk state WHILE the worker runs — the exact window a
            # client timeout orphans the sync call in.
            for sp in runner_mod._runs_dir(project_root).glob(f"node-{node_id}-*.state.json"):
                observed["state"] = json.loads(sp.read_text())
            log_path.write_bytes(b"ok")
            return 0, False

        monkeypatch.setattr(runner_mod, "_execute", _stub_execute)

        from milknado.mcp_run import milknado_todo_run

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="orphan", kind="task", project_root=root)
        _call(milknado_todo_run, node_id=task["id"], project_root=root)

        state = observed.get("state")
        assert state is not None, "no 'running' state file existed while the worker ran"
        assert state["status"] == "running"
        # fail_stale_running_runs keys on both fields; missing either makes an
        # orphaned run unrescuable.
        assert state["started_at"]
        assert state["timeout_seconds"] is not None

    def test_orphaned_sync_run_is_rescuable(self, tmp_path: Path, monkeypatch) -> None:
        from datetime import UTC, datetime, timedelta

        import milknado.domains.dispatch.runner as runner_mod
        from milknado.domains.dispatch.runner import _runs_dir, fail_stale_running_runs

        def _crash_execute(project_root, node_id, log_path, brief, argv, timeout):
            raise RuntimeError("server killed mid-run before terminal write")

        monkeypatch.setattr(runner_mod, "_execute", _crash_execute)

        from milknado.mcp_run import milknado_todo_run

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="orphan-stale", kind="task", project_root=root)
        with pytest.raises(RuntimeError):
            _call(milknado_todo_run, node_id=task["id"], project_root=root)

        runs_dir = _runs_dir(Path(root))
        files = list(runs_dir.glob(f"node-{task['id']}-*.state.json"))
        assert files, "orphaned sync run left no state file to rescue"
        state = json.loads(files[0].read_text())
        assert state["status"] == "running"
        # Age the run past its timeout and the stale sweep must release the node.
        aged = datetime.now(UTC) - timedelta(seconds=state["timeout_seconds"] + 3600)
        state["started_at"] = aged.isoformat()
        files[0].write_text(json.dumps(state))

        flipped = fail_stale_running_runs(Path(root), task["id"])
        assert len(flipped) == 1
        assert flipped[0]["status"] == "failed"

    def test_done_node_rerun_writes_no_running_state(self, tmp_path: Path, monkeypatch) -> None:
        """The running-state write is gated on an actual PENDING/FAILED->RUNNING
        transition. A DONE node re-run must NOT drop a 'running' file the stale
        sweep could later force-fail, which would clobber the prior DONE result."""
        import milknado.domains.dispatch.runner as runner_mod

        observed: dict = {}

        def _stub_execute(project_root, node_id, log_path, brief, argv, timeout):
            files = runner_mod._runs_dir(project_root).glob(f"node-{node_id}-*.state.json")
            observed["running"] = [
                s for f in files if (s := json.loads(f.read_text())).get("status") == "running"
            ]
            log_path.write_bytes(b"ok")
            return 0, False

        monkeypatch.setattr(runner_mod, "_execute", _stub_execute)

        from milknado.mcp_run import milknado_todo_run

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="done-rerun", kind="task", project_root=root)
        _call(milknado_todo_run, node_id=task["id"], project_root=root)  # -> DONE
        # Second run sees a DONE node: _ensure_running returns False, so the gate
        # must skip the running-state write — no 'running' file exists mid-run.
        _call(milknado_todo_run, node_id=task["id"], project_root=root)
        assert observed["running"] == [], "DONE-node re-run must not write a 'running' state file"


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
        rdir = tmp_path / ".milknado" / "runs"
        rdir.mkdir(parents=True)
        older_id = "node-1-20260101T000000Z-aaaa"
        newer_id = "node-1-20260101T010000Z-bbbb"
        for rid, started_at in [
            (older_id, "2026-01-01T00:00:00+00:00"),
            (newer_id, "2026-01-01T01:00:00+00:00"),
        ]:
            (rdir / f"{rid}.state.json").write_text(
                json.dumps(
                    {
                        "run_id": rid,
                        "node_id": 1,
                        "status": "done",
                        "started_at": started_at,
                    }
                )
            )
        runs = _call(milknado_run_list, project_root=root)
        assert runs[0]["run_id"] == newer_id
        assert runs[1]["run_id"] == older_id

    def test_tolerates_corrupt_state_file(self, tmp_path: Path) -> None:
        rdir = tmp_path / ".milknado" / "runs"
        rdir.mkdir(parents=True)
        (rdir / "node-1-20260101T000000Z-dead.state.json").write_text("not-json{{{")
        result = _call(milknado_run_list, project_root=str(tmp_path))
        assert result == []


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
        rdir = tmp_path / ".milknado" / "runs"
        rdir.mkdir(parents=True)
        run_id = "node-1-20260101T000000Z-abcd"
        (rdir / f"{run_id}.state.json").write_text(
            json.dumps({"run_id": run_id, "node_id": 1, "status": "done"})
        )
        killed = []
        monkeypatch.setattr(os, "killpg", lambda *a: killed.append(a))
        result = _call(milknado_run_cancel, run_id=run_id, project_root=str(tmp_path))
        assert result["status"] == "done"
        assert killed == [], "must not kill a terminal run"

    def test_cancel_signals_process_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import signal as sig_mod

        rdir = tmp_path / ".milknado" / "runs"
        rdir.mkdir(parents=True)
        run_id = "node-1-20260101T000000Z-abcd"
        (rdir / f"{run_id}.state.json").write_text(
            json.dumps({"run_id": run_id, "node_id": None, "status": "running", "pid": 9999})
        )
        killed: list[tuple] = []
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        _call(milknado_run_cancel, run_id=run_id, project_root=str(tmp_path))
        assert killed == [(9999, sig_mod.SIGTERM)], "must send SIGTERM to process group"

    def test_cancel_writes_failed_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rdir = tmp_path / ".milknado" / "runs"
        rdir.mkdir(parents=True)
        run_id = "node-1-20260101T000000Z-abcd"
        state_path = rdir / f"{run_id}.state.json"
        state_path.write_text(json.dumps({"run_id": run_id, "node_id": None, "status": "running"}))
        monkeypatch.setattr(os, "killpg", lambda *a: None)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        result = _call(milknado_run_cancel, run_id=run_id, project_root=str(tmp_path))
        assert result["status"] == "failed"
        assert result["exit_code"] == -1
        on_disk = json.loads(state_path.read_text())
        assert on_disk["status"] == "failed"
        assert on_disk["error"] == "cancelled"

    def test_cancel_returns_superset_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rdir = tmp_path / ".milknado" / "runs"
        rdir.mkdir(parents=True)
        run_id = "node-1-20260101T000000Z-abcd"
        (rdir / f"{run_id}.state.json").write_text(
            json.dumps({"run_id": run_id, "node_id": None, "status": "running"})
        )
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
        rdir = tmp_path / ".milknado" / "runs"
        rdir.mkdir(parents=True, exist_ok=True)
        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        (rdir / f"{run_id}.state.json").write_text(
            json.dumps({"run_id": run_id, "node_id": node_id, "status": "running"})
        )
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

        rdir = tmp_path / ".milknado" / "runs"
        rdir.mkdir(parents=True, exist_ok=True)
        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        (rdir / f"{run_id}.state.json").write_text(
            json.dumps({"run_id": run_id, "node_id": node_id, "status": "running"})
        )

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
        rdir = tmp_path / ".milknado" / "runs"
        rdir.mkdir(parents=True, exist_ok=True)
        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        state_path = rdir / f"{run_id}.state.json"
        state_path.write_text(
            json.dumps({"run_id": run_id, "node_id": node_id, "status": "running"})
        )

        result = _call(milknado_run_cancel, run_id=run_id, project_root=root)

        assert result["status"] == "failed"
        assert result["exit_code"] == -1
        on_disk = json.loads(state_path.read_text())
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
        rdir = tmp_path / ".milknado" / "runs"
        rdir.mkdir(parents=True, exist_ok=True)
        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        (rdir / f"{run_id}.state.json").write_text(
            json.dumps({"run_id": run_id, "node_id": node_id, "status": "running"})
        )

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
        state_path = Path(result["state_path"])
        on_disk = json.loads(state_path.read_text())
        assert on_disk["status"] == "failed"
        assert on_disk["error"] == "cancelled"

        # (a)+(c) the worker stopped and does NOT overwrite the cancelled state:
        # had it kept running, `sleep 30` would later flip the state to done.
        time.sleep(1.0)
        after = json.loads(state_path.read_text())
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
        on_disk = json.loads(Path(final["state_path"]).read_text())
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
        rdir = tmp_path / ".milknado" / "runs"
        rdir.mkdir(parents=True, exist_ok=True)
        state_path = rdir / f"{run_id}.state.json"
        state_path.write_text(
            json.dumps({"run_id": run_id, "node_id": node_id, "status": "running"})
        )

        # The worker finalizes `done` during the finalize poll window: the state
        # flips to done and that is what `_await_cancel_finalize` returns.
        done_state = {"run_id": run_id, "node_id": node_id, "status": "done", "exit_code": 0}

        def fake_finalize(sp: Path) -> dict:
            sp.write_text(json.dumps(done_state))
            return done_state

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

    def _write_pid_state(self, tmp_path: Path, run_id: str, node_id: int) -> Path:
        rdir = tmp_path / ".milknado" / "runs"
        rdir.mkdir(parents=True, exist_ok=True)
        state_path = rdir / f"{run_id}.state.json"
        state_path.write_text(
            json.dumps({"run_id": run_id, "node_id": node_id, "status": "running", "pid": 9999})
        )
        return state_path

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
        state_path = self._write_pid_state(tmp_path, run_id, node_id)

        def gone(*_a):
            raise ProcessLookupError("no such process")

        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", gone)

        result = _call(milknado_run_cancel, run_id=run_id, project_root=str(tmp_path))

        assert result["status"] == "failed"
        assert result["exit_code"] == -1
        on_disk = json.loads(state_path.read_text())
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

    def test_await_cancel_finalize_logs_after_unreadable_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If the state file stays unreadable for the whole finalize window, the
        bound elapses to None and a warning naming the read-error count is logged,
        so a wedged/permission-denied state file is diagnosable, not silent."""
        from milknado.mcp_run import _await_cancel_finalize

        monkeypatch.setattr("milknado.mcp_run._CANCEL_FINALIZE_TIMEOUT_SECS", 0.05)
        monkeypatch.setattr("milknado.mcp_run._CANCEL_FINALIZE_POLL_SECS", 0.01)

        def boom(_sp):
            raise OSError("permission denied")

        monkeypatch.setattr("milknado.mcp_run.read_state", boom)

        with caplog.at_level(logging.WARNING, logger="milknado.mcp_run"):
            out = _await_cancel_finalize(tmp_path / "missing.state.json")

        assert out is None, "an unreadable state for the whole window must time out to None"
        assert any("unreadable state reads" in r.getMessage() for r in caplog.records), (
            "the read-error count must be logged when the bound elapses"
        )

    def test_async_cancel_race_reread_adopts_last_gap_finalize(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The race fence (mcp_run.py:452): _await_cancel_finalize returns None
        (bound elapsed) but the worker wrote its terminal state in the final poll
        gap. Cancel must re-read and ADOPT that write, not overwrite a genuine
        `done` with `cancelled`."""
        from milknado.domains.dispatch import runs_dir as _runs_dir

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
        rdir = _runs_dir(tmp_path)
        rdir.mkdir(parents=True, exist_ok=True)
        state_path = rdir / f"{run_id}.state.json"
        state_path.write_text(
            json.dumps({"run_id": run_id, "node_id": node_id, "status": "running"})
        )

        done_state = {
            "run_id": run_id,
            "node_id": node_id,
            "status": "done",
            "exit_code": 0,
            "ended_at": "2026-01-01T00:00:00+00:00",
        }

        def elapsed_after_done(sp: Path):
            sp.write_text(json.dumps(done_state))
            return None

        monkeypatch.setattr("milknado.mcp_run._await_cancel_finalize", elapsed_after_done)

        result = _call(milknado_run_cancel, run_id=run_id, project_root=root)

        assert result["status"] == "done", "must adopt the worker's last-gap terminal write"
        on_disk = json.loads(state_path.read_text())
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
