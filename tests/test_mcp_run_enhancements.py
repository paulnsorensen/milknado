"""Tests for #82 (milknado_run_list + milknado_run_cancel) and
#83 (unified RunDict superset schema across all five run tools).
"""

from __future__ import annotations

import json
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
