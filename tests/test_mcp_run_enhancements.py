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

from milknado.adapters import ProcessAdapter
from milknado.domains.common import NodeKind, RunResult, WorktreeMode
from milknado.domains.dispatch import ProcessOutcome
from milknado.mcp._core import RunDict
from milknado.mcp.ralph import milknado_run_loop_poll, milknado_run_loop_start
from milknado.mcp.run import (
    milknado_deposit_result,
    milknado_run_cancel,
    milknado_run_inline_poll,
    milknado_run_inline_start,
    milknado_run_list,
)
from milknado.mcp.server import open_graph
from milknado.mcp.todo_mutate import milknado_todo_add

_SUPERSET_KEYS = frozenset(RunDict.__annotations__)

_STUB_RUNNER = """
import argparse
from pathlib import Path
from milknado.mcp._core import open_graph
from milknado.domains.common import RunResult
p = argparse.ArgumentParser()
p.add_argument("--node-id", type=int)
p.add_argument("--project-root")
p.add_argument("--run-id")
p.add_argument("--timeout")
p.add_argument("--target-branch")
p.add_argument("--base-oid")
a = p.parse_args()
graph, _cfg = open_graph(Path(a.project_root))
try:
    graph.finish_run(
        a.run_id,
        RunResult(
            status="{status}",
            exit_code={exit_code},
            timed_out=False,
            ended_at="2026-01-01T00:00:00+00:00",
            rebased={rebased},
            detail=None,
        ),
    )
finally:
    graph.close()
"""


def _force_run_id(graph, node_id: int, run_id: str) -> None:
    """Seed a node's run_id directly; no production API sets it in isolation."""
    graph._conn.execute("UPDATE nodes SET run_id = ? WHERE id = ?", (run_id, node_id))
    graph._conn.commit()


def _init_git(root: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
        ],
        cwd=root,
        check=True,
    )


def _call(tool, **kwargs):
    fn = getattr(tool, "fn", tool)
    # run_inline{,_start} now default to ISOLATE (per-dispatch worktree). These
    # legacy tests target the shared-checkout path, so pin THIS_BRANCH unless a
    # test opts into ISOLATE explicitly.
    if fn.__name__ in ("milknado_run_inline", "milknado_run_inline_start"):
        kwargs.setdefault("worktree", WorktreeMode.THIS_BRANCH)
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
    log_path: str | None = None,
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
        log_path = log_path or str(root / ".milknado" / "runs" / f"{run_id}.log")
        graph.start_run(run_id, node_id, log_path, started_at, timeout_seconds, pid)
        if status != "running":
            graph.finish_run(
                run_id,
                RunResult(
                    status=status,
                    exit_code=exit_code,
                    timed_out=timed_out,
                    ended_at=ended_at or datetime.now(UTC).isoformat(),
                    error=error,
                ),
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
def test_run_dict_schema_matches_annotations() -> None:
    from milknado.mcp._core import build_run_dict

    assert set(build_run_dict({})) == set(RunDict.__annotations__)


def test_cancel_sentinel_uses_touch_and_missing_ok_unlink(tmp_path: Path) -> None:
    from milknado.domains.dispatch import clear_cancel, is_cancel_requested, request_cancel

    request_cancel(tmp_path, "node-1-20260101T000000Z-abcd")
    assert is_cancel_requested(tmp_path, "node-1-20260101T000000Z-abcd")
    clear_cancel(tmp_path, "node-1-20260101T000000Z-abcd")
    clear_cancel(tmp_path, "node-1-20260101T000000Z-abcd")
    assert not is_cancel_requested(tmp_path, "node-1-20260101T000000Z-abcd")


# ---------------------------------------------------------------------------


class TestUnifiedRunSchema:
    """Every run tool must return a dict that contains all superset keys."""

    def test_run_inline_start_returns_superset_schema(self, tmp_path: Path, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="schema-start", kind="task", project_root=root)
        result = _call(
            milknado_run_inline_start,
            node_id=task["id"],
            worker_cmd=worker_stub("cat"),
            project_root=root,
        )
        missing = _SUPERSET_KEYS - result.keys()
        assert not missing, f"milknado_run_inline_start missing keys: {missing}"
        # start tools return None for fields only available post-completion
        assert result["status"] == "running"
        assert result["run_id"] is not None
        assert result["exit_code"] is None
        assert result["timed_out"] is None
        assert result["rebased"] is None
        assert result["summary"] is None

    def test_run_inline_poll_returns_superset_schema(self, tmp_path: Path, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="schema-poll", kind="task", project_root=root)
        started = _call(
            milknado_run_inline_start,
            node_id=task["id"],
            worker_cmd=worker_stub("cat"),
            project_root=root,
        )
        final = _wait_for_terminal(started["run_id"], root, milknado_run_inline_poll)
        missing = _SUPERSET_KEYS - final.keys()
        assert not missing, f"milknado_run_inline_poll missing keys: {missing}"
        assert final["run_id"] == started["run_id"]
        assert "rebased" in final  # may be None for headless runs

    def test_run_loop_start_returns_superset_schema(self, tmp_path: Path) -> None:
        _init_git(tmp_path)
        root = str(tmp_path)
        task = _call(
            milknado_todo_add, description="schema-ralph-start", kind="task", project_root=root
        )
        result = _call(
            milknado_run_loop_start,
            node_id=task["id"],
            runner_cmd=f"{sys.executable} -c pass",
            project_root=root,
        )
        missing = _SUPERSET_KEYS - result.keys()
        assert not missing, f"milknado_run_loop_start missing keys: {missing}"
        assert result["status"] == "running"
        assert result["exit_code"] is None
        assert result["timed_out"] is None
        assert result["rebased"] is None
        assert result["summary"] is None

    def test_run_loop_poll_returns_superset_schema(self, tmp_path: Path) -> None:
        _init_git(tmp_path)
        root = str(tmp_path)
        task = _call(
            milknado_todo_add, description="schema-ralph-poll", kind="task", project_root=root
        )
        started = _call(
            milknado_run_loop_start,
            node_id=task["id"],
            runner_cmd=_stub_runner_cmd(tmp_path, status="done", rebased=True),
            project_root=root,
        )
        final = _wait_for_terminal(started["run_id"], root, milknado_run_loop_poll)
        missing = _SUPERSET_KEYS - final.keys()
        assert not missing, f"milknado_run_loop_poll missing keys: {missing}"
        assert final["rebased"] is True


class TestSyncRunSchema:
    """milknado_run_inline (sync) must also return the superset schema and write a state file."""

    def test_run_inline_returns_superset_schema(self, tmp_path: Path, monkeypatch) -> None:
        import milknado.domains.dispatch.runner as runner_mod

        def _stub_execute(
            project_root, node_id, log_path, brief, argv, timeout, run_id=None, **kwargs
        ):
            log_path.write_bytes(b"ok")
            return 0, False

        monkeypatch.setattr(runner_mod, "_execute", _stub_execute)

        from milknado.mcp.run import milknado_run_inline

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="sync-schema", kind="task", project_root=root)
        result = _call(milknado_run_inline, node_id=task["id"], project_root=root)
        missing = _SUPERSET_KEYS - result.keys()
        assert not missing, f"milknado_run_inline missing keys: {missing}"
        assert result["run_id"] is not None
        assert result["rebased"] is None
        assert result["exit_code"] == 0
        assert result["timed_out"] is False

    def test_run_inline_writes_run_row(self, tmp_path: Path, monkeypatch) -> None:
        import milknado.domains.dispatch.runner as runner_mod

        def _stub_execute(
            project_root, node_id, log_path, brief, argv, timeout, run_id=None, **kwargs
        ):
            log_path.write_bytes(b"ok")
            return 0, False

        monkeypatch.setattr(runner_mod, "_execute", _stub_execute)

        from milknado.mcp.run import milknado_run_inline

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="sync-state", kind="task", project_root=root)
        result = _call(milknado_run_inline, node_id=task["id"], project_root=root)
        state = _read_run(tmp_path, result["run_id"])
        assert state is not None, "sync run must write a terminal run row"
        assert state["run_id"] == result["run_id"]
        assert state["status"] == "done"

    def test_run_inline_sets_run_id_on_node(self, tmp_path: Path, monkeypatch) -> None:
        """#40: run_id must be threaded into the node after a sync run."""
        import milknado.domains.dispatch.runner as runner_mod

        def _stub_execute(
            project_root, node_id, log_path, brief, argv, timeout, run_id=None, **kwargs
        ):
            log_path.write_bytes(b"ok")
            return 0, False

        monkeypatch.setattr(runner_mod, "_execute", _stub_execute)

        from milknado.mcp.run import milknado_run_inline

        root = str(tmp_path)
        task = _call(
            milknado_todo_add, description="run-id-on-node", kind="task", project_root=root
        )
        result = _call(milknado_run_inline, node_id=task["id"], project_root=root)
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

        def _stub_execute(
            project_root, node_id, log_path, brief, argv, timeout, run_id=None, **kwargs
        ):
            # Snapshot the run row WHILE the worker runs — the exact window a
            # client timeout orphans the sync call in.
            rows = _node_running_runs(Path(project_root), node_id)
            if rows:
                observed["state"] = rows[0]
            log_path.write_bytes(b"ok")
            return 0, False

        monkeypatch.setattr(runner_mod, "_execute", _stub_execute)

        from milknado.mcp.run import milknado_run_inline

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="orphan", kind="task", project_root=root)
        _call(milknado_run_inline, node_id=task["id"], project_root=root)

        state = observed.get("state")
        assert state is not None, "no 'running' run row existed while the worker ran"
        assert state["status"] == "running"
        # fail_stale_running_runs keys on both fields; missing either makes an
        # orphaned run unrescuable.
        assert state["started_at"]
        assert state["timeout_seconds"] is not None

    def test_sync_dispatch_exception_finalizes_run_failed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """An in-process exception during sync dispatch (here a worker crash,
        simulated by _execute raising) must finalize the run and node FAILED
        before re-raising — never strand them RUNNING. The true-orphan path (an
        uncatchable kill leaving the up-front 'running' row for the stale sweep)
        stays covered by test_sync_run_writes_rescuable_running_state_before_worker
        plus the fail_stale_running_runs unit tests in test_mcp_server.py."""
        import milknado.domains.dispatch.runner as runner_mod

        def _crash_execute(
            project_root, node_id, log_path, brief, argv, timeout, run_id=None, **kwargs
        ):
            raise RuntimeError("worker crashed mid-run before terminal write")

        monkeypatch.setattr(runner_mod, "_execute", _crash_execute)

        from milknado.mcp.run import milknado_run_inline

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="crash", kind="task", project_root=root)
        with pytest.raises(RuntimeError):
            _call(milknado_run_inline, node_id=task["id"], project_root=root)

        # High-1 contract: the caught exception finalizes the node and its run
        # terminal — nothing is left RUNNING for the stale sweep to clean up.
        graph, _cfg = open_graph(tmp_path)
        try:
            node = graph.get_node(task["id"])
            assert node is not None
            assert node.status.value == "failed", "exception must fail the node, not strand it"
            runs = graph.runs_for_node(task["id"])
            assert runs, "dispatch must have created a run row before the worker ran"
            assert all(r["status"] != "running" for r in runs), "no run may be left running"
            assert any(r["status"] == "failed" for r in runs), "the crashed run must be failed"
        finally:
            graph.close()

    def test_done_node_rerun_refused_first_run_injects_run_id(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The first (PENDING->RUNNING) run has a run row, so it MUST inject
        MILKNADO_RUN_ID — the brief's deposit step keys on it (the #122 channel).
        A DONE-node re-dispatch is refused up front (High-5), so it spawns no
        second worker and captures no second env. Exercises the real
        ProcessAdapter boundary instead of stubbing process mechanics in the
        dispatch domain.
        """
        envs: list[dict[str, str]] = []

        def _capture_run(
            self,
            argv,
            cwd,
            log_path,
            stdin,
            env,
            timeout,
            **kwargs,
        ):
            envs.append(env)
            log_path.write_text("", encoding="utf-8")
            return ProcessOutcome(exit_code=0, timed_out=False, cancelled=False)

        monkeypatch.setattr(ProcessAdapter, "run", _capture_run)

        from milknado.mcp.run import milknado_run_inline

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="rerun-env", kind="task", project_root=root)
        # First run: PENDING -> RUNNING -> DONE (a run row exists).
        _call(milknado_run_inline, node_id=task["id"], project_root=root)
        # Second run sees a DONE node and is refused before any worker spawns.
        with pytest.raises(ValueError, match="already done"):
            _call(milknado_run_inline, node_id=task["id"], project_root=root)

        assert len(envs) == 1, "the refused re-dispatch must not spawn a second worker"
        assert "MILKNADO_RUN_ID" in envs[0], (
            "a real (running) run must inject MILKNADO_RUN_ID so the worker can deposit"
        )


# ---------------------------------------------------------------------------
# #82 — milknado_run_list
# ---------------------------------------------------------------------------


class TestDispatchLifecycleGuards:
    @pytest.mark.parametrize(
        ("node", "message"),
        [
            (None, "not found"),
            (type("GoalNode", (), {"kind": NodeKind.GOAL})(), "only task nodes"),
        ],
    )
    def test_sync_dispatch_rejects_missing_or_non_task_node(
        self, tmp_path: Path, node: object | None, message: str
    ) -> None:
        from milknado.domains.dispatch import SyncDispatchRequest, dispatch_node_sync

        class Graph:
            def get_node(self, node_id: int) -> object | None:
                return node

        request = SyncDispatchRequest(
            node_id=7,
            project_root=tmp_path,
            worker_cmd=None,
            timeout_seconds=1,
            default_cmd="claude",
            process=object(),  # type: ignore[arg-type]
            worktree_mode=WorktreeMode.THIS_BRANCH,
        )
        with pytest.raises(ValueError, match=message):
            dispatch_node_sync(Graph(), object(), request)  # type: ignore[arg-type]

    def test_sync_dispatch_detects_node_deleted_after_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from milknado.domains.common import NodeKind
        from milknado.domains.dispatch import SyncDispatchRequest, dispatch_node_sync, lifecycle
        from milknado.domains.dispatch.runner import RunResult as WorkerResult

        node = type("TaskNode", (), {"kind": NodeKind.TASK})()

        class Graph:
            def __init__(self) -> None:
                self.reads = 0

            def get_node(self, node_id: int) -> object | None:
                self.reads += 1
                return node if self.reads == 1 else None

            def claim_node_for_dispatch(self, *args, **kwargs) -> None:
                pass

            def start_run(self, *args, **kwargs) -> None:
                pass

            def finish_run(self, *args, **kwargs) -> None:
                pass

            def mark_terminal(self, *args, **kwargs) -> None:
                pass

        monkeypatch.setattr(lifecycle, "render_brief", lambda *args, **kwargs: "")
        monkeypatch.setattr(
            lifecycle,
            "run_headless",
            lambda *args, **kwargs: WorkerResult(
                exit_code=0,
                log_path=tmp_path / "run.log",
                summary="",
                timed_out=False,
            ),
        )
        request = SyncDispatchRequest(
            node_id=7,
            project_root=tmp_path,
            worker_cmd=None,
            timeout_seconds=1,
            default_cmd="claude",
            process=object(),  # type: ignore[arg-type]
            worktree_mode=WorktreeMode.THIS_BRANCH,
        )
        with pytest.raises(RuntimeError, match="not found after run completed"):
            dispatch_node_sync(Graph(), object(), request)  # type: ignore[arg-type]


def test_stale_sweep_logs_and_skips_malformed_started_at(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from milknado.domains.dispatch import fail_stale_running_runs

    class Graph:
        def runs_for_node(self, node_id: int) -> list[dict]:
            return [
                {
                    "run_id": "node-4-20260101T000000Z-bad1",
                    "status": "running",
                    "started_at": "not-a-timestamp",
                    "timeout_seconds": 1,
                }
            ]

    with caplog.at_level(logging.WARNING):
        assert fail_stale_running_runs(Graph(), 4) == []
    assert "malformed timestamp" in caplog.text


class TestRunList:
    def test_empty_when_no_runs(self, tmp_path: Path) -> None:
        result = _call(milknado_run_list, project_root=str(tmp_path))
        assert result == []

    def test_lists_state_files_after_async_run(self, tmp_path: Path, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="list-test", kind="task", project_root=root)
        started = _call(
            milknado_run_inline_start,
            node_id=task["id"],
            worker_cmd=worker_stub("cat"),
            project_root=root,
        )
        _wait_for_terminal(started["run_id"], root, milknado_run_inline_poll)
        runs = _call(milknado_run_list, project_root=root)
        assert len(runs) >= 1
        run_ids = [r["run_id"] for r in runs]
        assert started["run_id"] in run_ids

    def test_returns_superset_keys(self, tmp_path: Path, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="list-keys", kind="task", project_root=root)
        started = _call(
            milknado_run_inline_start,
            node_id=task["id"],
            worker_cmd=worker_stub("cat"),
            project_root=root,
        )
        _wait_for_terminal(started["run_id"], root, milknado_run_inline_poll)
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
    def _joined_cancel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from milknado.domains.dispatch import cancel as cancel_module

        monkeypatch.setattr(
            cancel_module,
            "_await_cancel_finalize",
            lambda graph, run_id: cancel_module._finalize_cancelled(graph, run_id),
        )

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
        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.mark_running(node_id, run_id=run_id)
        finally:
            graph.close()
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

        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.mark_running(
                node_id,
                worktree_path=str(orphan_wt),
                branch_name="milknado/cancel",
                run_id=run_id,
            )
        finally:
            graph.close()
        _seed_run(tmp_path, run_id=run_id, node_id=node_id, status="running")

        removed: list[Path] = []
        monkeypatch.setattr(
            adapters.GitAdapter,
            "remove_worktree",
            lambda self, wt, target="HEAD": removed.append(wt),
        )

        _call(milknado_run_cancel, run_id=run_id, project_root=root)

        assert removed == [orphan_wt], "orphaned worktree must be removed on cancel"

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
        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.mark_running(node_id, run_id=run_id)
        finally:
            graph.close()
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
        import milknado.adapters as adapters

        root = str(tmp_path)
        task = _call(milknado_todo_add, description="wt-fail", kind="task", project_root=root)
        node_id = task["id"]
        orphan_wt = tmp_path / "milknado-wt-fail"
        orphan_wt.mkdir()
        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.mark_running(
                node_id, worktree_path=str(orphan_wt), branch_name="milknado/x", run_id=run_id
            )
        finally:
            graph.close()
        _seed_run(tmp_path, run_id=run_id, node_id=node_id, status="running")

        def boom(self, wt, target="HEAD"):
            raise RuntimeError("worktree locked")

        monkeypatch.setattr(adapters.GitAdapter, "remove_worktree", boom)
        with caplog.at_level(logging.WARNING, logger="milknado.domains.dispatch.cancel"):
            result = _call(milknado_run_cancel, run_id=run_id, project_root=root)
        assert result["status"] == "failed"
        assert any("cancel preserving worktree" in r.getMessage() for r in caplog.records)

    def _cancel_repo_with_worktree(
        self, tmp_path: Path, *, unlanded: bool
    ) -> tuple[int, str, Path]:
        """A real git repo at tmp_path with a RUNNING node whose worktree either
        has all work landed (clean, no new commits) or one unlanded commit."""
        import subprocess

        def git(cwd: Path, *args: str) -> None:
            subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)

        git(tmp_path, "init", "-q", "-b", "feature", str(tmp_path))
        git(tmp_path, "config", "user.email", "t@t")
        git(tmp_path, "config", "user.name", "t")
        (tmp_path / "README.md").write_text("seed\n")
        git(tmp_path, "add", ".")
        git(tmp_path, "commit", "-qm", "seed")
        wt = tmp_path / "milknado-cancel-wt"
        git(tmp_path, "worktree", "add", "-q", "-b", "milknado/cancel", str(wt))
        if unlanded:
            (wt / "work.py").write_text("x = 1\n")
            git(wt, "add", ".")
            git(wt, "commit", "-qm", "unlanded work")

        task = _call(
            milknado_todo_add, description="cancel-git", kind="task", project_root=str(tmp_path)
        )
        node_id = task["id"]
        run_id = f"node-{node_id}-20260101T000000Z-abcd"
        graph, _cfg = open_graph(tmp_path)
        try:
            graph.mark_running(
                node_id, worktree_path=str(wt), branch_name="milknado/cancel", run_id=run_id
            )
        finally:
            graph.close()
        _seed_run(tmp_path, run_id=run_id, node_id=node_id, status="running")
        return node_id, run_id, wt

    def test_cancel_removes_landed_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Acceptance 5 (landed): a cancelled run whose worktree holds nothing
        unlanded is torn down through the fail-closed path and the node
        reconciles to failed."""
        node_id, run_id, wt = self._cancel_repo_with_worktree(tmp_path, unlanded=False)
        monkeypatch.setattr(os, "killpg", lambda *a: None)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)

        result = _call(milknado_run_cancel, run_id=run_id, project_root=str(tmp_path))

        assert result["status"] == "failed"
        assert result["worktree_preserved"] is None, (
            "a removed worktree must not report a preserved path"
        )
        assert not wt.exists(), "a landed worktree must be removed on cancel"
        graph, _cfg = open_graph(tmp_path)
        try:
            node = graph.get_node(node_id)
            assert node is not None
            assert node.status.value == "failed"
        finally:
            graph.close()

    def test_cancel_preserves_unlanded_worktree_and_reconciles_terminal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling a run whose worktree holds a commit the feature branch
        never received must NOT strand the node RUNNING or raise out of the
        tool: the fail-closed teardown refusal is caught, the worktree is
        preserved on disk (the only copy of that work), the node reconciles to
        terminal (failed), and the tool reports the preserved path."""
        node_id, run_id, wt = self._cancel_repo_with_worktree(tmp_path, unlanded=True)
        monkeypatch.setattr(os, "killpg", lambda *a: None)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)

        result = _call(milknado_run_cancel, run_id=run_id, project_root=str(tmp_path))

        assert result["status"] == "failed"
        assert result["worktree_preserved"] == str(wt), (
            "the tool must report the preserved worktree path"
        )
        assert wt.exists(), "refusal must preserve the worktree on disk"
        assert (wt / "work.py").read_text() == "x = 1\n"
        graph, _cfg = open_graph(tmp_path)
        try:
            node = graph.get_node(node_id)
            assert node is not None
            assert node.status.value == "failed", (
                "the node must reach terminal even when its worktree is preserved"
            )
        finally:
            graph.close()


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
        slow_cmd = worker_stub("sleep 30")
        started = _call(
            milknado_run_inline_start, node_id=node_id, worker_cmd=slow_cmd, project_root=root
        )
        run_id = started["run_id"]
        assert started["status"] == "running"
        deadline = time.monotonic() + 5.0
        while _read_run(tmp_path, run_id)["pid"] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _read_run(tmp_path, run_id)["pid"] is not None

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
        slow_cmd = worker_stub("sleep 30")
        started = _call(
            milknado_run_inline_start,
            node_id=node_id,
            worker_cmd=slow_cmd,
            timeout_seconds=1,
            project_root=root,
        )
        run_id = started["run_id"]

        final = _wait_for_terminal(run_id, root, milknado_run_inline_poll, timeout=8.0)

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
            _force_run_id(graph, node_id, run_id)
        finally:
            graph.close()
        _seed_run(tmp_path, run_id=run_id, node_id=node_id, status="running")

        # The worker finalizes `done` during the finalize poll window: the run row
        # flips to done and that is what `_await_cancel_finalize` returns.
        def fake_finalize(graph, rid: str) -> dict:
            graph.finish_run(
                rid,
                RunResult(
                    status="done", exit_code=0, timed_out=False, ended_at="2026-01-01T00:00:00Z"
                ),
            )
            return graph.get_run(rid)

        monkeypatch.setattr(
            "milknado.domains.dispatch.cancel._await_cancel_finalize", fake_finalize
        )

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
            _force_run_id(graph, node_id, run_id)
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
            _force_run_id(graph, node_id, newer_run)
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
        from milknado.domains.dispatch.cancel import _await_cancel_finalize

        monkeypatch.setattr("milknado.domains.dispatch.cancel._CANCEL_FINALIZE_TIMEOUT_SECS", 0.05)
        monkeypatch.setattr("milknado.domains.dispatch.cancel._CANCEL_FINALIZE_POLL_SECS", 0.01)

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
            _force_run_id(graph, node_id, run_id)
        finally:
            graph.close()
        _seed_run(tmp_path, run_id=run_id, node_id=node_id, status="running")

        def elapsed_after_done(graph, rid: str):
            graph.finish_run(
                rid,
                RunResult(
                    status="done", exit_code=0, timed_out=False, ended_at="2026-01-01T00:00:00Z"
                ),
            )
            return None

        monkeypatch.setattr(
            "milknado.domains.dispatch.cancel._await_cancel_finalize", elapsed_after_done
        )

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

    def test_pid_cancel_preserves_state_when_process_will_not_exit(self) -> None:
        from milknado.domains.dispatch.cancel import _cancel_pid_run

        class Process:
            def terminate_group(self, pid: int, timeout: float) -> bool:
                return False

        with pytest.raises(RuntimeError, match="did not exit after termination"):
            _cancel_pid_run(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                Process(),
                {"pid": 91, "node_id": 3},
                "node-3-20260101T000000Z-stuk",
            )

    def test_pid_cancel_fails_loud_when_terminal_write_is_not_confirmed(self) -> None:
        from milknado.domains.dispatch.cancel import _cancel_pid_run

        class Graph:
            def finish_run(self, run_id, result):
                return False

            def get_run(self, run_id):
                return None

        class Process:
            def terminate_group(self, pid: int, timeout: float) -> bool:
                return True

        run_id = "node-3-20260101T000000Z-gone"
        with pytest.raises(RuntimeError, match="cancellation finalization was not confirmed"):
            _cancel_pid_run(
                Graph(),
                object(),  # type: ignore[arg-type]
                Process(),
                {"pid": 91, "node_id": 3},
                run_id,
            )

    def test_async_cancel_preserves_running_state_without_exit_confirmation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from milknado.domains.dispatch import cancel as cancel_module

        class Graph:
            def get_run(self, run_id: str) -> dict:
                return {"status": "running"}

        monkeypatch.setattr(cancel_module, "_await_cancel_finalize", lambda graph, run_id: None)
        run_id = "node-3-20260101T000000Z-stuk"
        with pytest.raises(RuntimeError, match="has not confirmed worker exit"):
            cancel_module._cancel_async_run(
                Graph(),
                object(),  # type: ignore[arg-type]
                tmp_path,
                {"node_id": 3},
                run_id,
            )
        assert not (cancel_module.runs_dir(tmp_path) / f"{run_id}.cancel").exists()


class TestTerminateWorker:
    """The production process adapter escalates an ignored SIGTERM to SIGKILL."""

    def test_terminate_worker_escalates_to_sigkill_when_sigterm_ignored(
        self, tmp_path: Path
    ) -> None:
        import signal as sig_mod
        import subprocess as sp_mod

        ready = tmp_path / "handler-ready"
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
            ProcessAdapter(termination_grace=0.2)._terminate(proc)
            rc = proc.returncode
        finally:
            if proc.poll() is None:
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

    def test_deposit_rejects_mismatched_worker_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_id = "node-1-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=1, status="running")
        monkeypatch.setenv("MILKNADO_PROJECT_ROOT", str(tmp_path.resolve()))
        monkeypatch.setenv("MILKNADO_RUN_ID", "node-2-20260101T000000Z-feed")
        with pytest.raises(ValueError, match="does not match injected run"):
            _call(
                milknado_deposit_result,
                run_id=run_id,
                payload="wrong owner",
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

    def test_run_inline_poll_returns_deposited_result(self, tmp_path: Path, worker_stub) -> None:
        """The async-run poll must surface the deposited result under `result`."""
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="deposit-poll", kind="task", project_root=root)
        started = _call(
            milknado_run_inline_start,
            node_id=task["id"],
            worker_cmd=worker_stub("cat"),
            project_root=root,
        )
        _wait_for_terminal(started["run_id"], root, milknado_run_inline_poll)
        _call(
            milknado_deposit_result,
            run_id=started["run_id"],
            payload="the deliverable",
            project_root=root,
        )
        final = _call(milknado_run_inline_poll, run_id=started["run_id"], project_root=root)
        assert final["result"] == "the deliverable"

    def test_poll_result_is_none_without_deposit(self, tmp_path: Path, worker_stub) -> None:
        root = str(tmp_path)
        task = _call(milknado_todo_add, description="no-deposit", kind="task", project_root=root)
        started = _call(
            milknado_run_inline_start,
            node_id=task["id"],
            worker_cmd=worker_stub("cat"),
            project_root=root,
        )
        final = _wait_for_terminal(started["run_id"], root, milknado_run_inline_poll)
        assert final["result"] is None, "no deposit -> result is None, not a missing key"

    def test_run_inline_poll_derives_log_path_from_run_id(self, tmp_path: Path) -> None:
        """The poll must tail the log derived from the validated run_id — a
        tampered runs.log_path in the user-editable db must not cause an
        arbitrary-file read (mirrors milknado_run_loop_poll's derivation)."""
        decoy = tmp_path / "secret.txt"
        decoy.write_text("SECRET-CONTENT", encoding="utf-8")
        run_id = "node-1-20260101T000000Z-feed"
        _seed_run(tmp_path, run_id=run_id, node_id=1, status="done", exit_code=0)
        derived = tmp_path / ".milknado" / "runs" / f"{run_id}.log"
        derived.parent.mkdir(parents=True, exist_ok=True)
        derived.write_text("real log", encoding="utf-8")
        graph, _cfg = open_graph(tmp_path)
        try:
            graph._conn.execute(
                "UPDATE runs SET log_path = ? WHERE run_id = ?", (str(decoy), run_id)
            )
            graph._conn.commit()
        finally:
            graph.close()
        final = _call(milknado_run_inline_poll, run_id=run_id, project_root=str(tmp_path))
        assert final["summary"] == "real log"
        assert "SECRET-CONTENT" not in (final["summary"] or "")
        assert final["log_path"] == str(derived), "returned log_path must be the derived one"

    def test_run_loop_poll_falls_back_to_latest_iteration_log_for_executor_runs(
        self, tmp_path: Path
    ) -> None:
        """Medium: executor-owned (in-process) ralph runs never write the
        detached path's flat <run_id>.log — they write one file per
        iteration under a `.ralph-logs` directory instead. Poll must fall
        back to tailing the newest iteration file rather than silently
        returning an empty summary forever."""
        run_id = "node-1-20260101T000000Z-1234"
        log_dir = tmp_path / "milknado-1-task" / ".ralph-logs"
        log_dir.mkdir(parents=True)
        (log_dir / "0001_20260101T000000Z.log").write_text("iteration 1 output", encoding="utf-8")
        (log_dir / "0002_20260101T000100Z.log").write_text("iteration 2 output", encoding="utf-8")
        _seed_run(
            tmp_path,
            run_id=run_id,
            node_id=1,
            status="running",
            log_path=str(log_dir),
        )
        final = _call(milknado_run_loop_poll, run_id=run_id, project_root=str(tmp_path))
        assert final["summary"] == "iteration 2 output", (
            "poll must tail the newest per-iteration file, not stay empty"
        )

    def test_run_loop_poll_ignores_directory_log_path_outside_project_root(
        self, tmp_path: Path
    ) -> None:
        """The directory fallback must stay inside the project root — a
        tampered runs.log_path pointing at a directory elsewhere on disk
        must not be tailed, mirroring the flat-file derivation's guard."""
        outside_dir = tmp_path.parent / f"{tmp_path.name}-outside-ralph-logs"
        outside_dir.mkdir(exist_ok=True)
        (outside_dir / "0001_x.log").write_text("SECRET-ITERATION-OUTPUT", encoding="utf-8")
        try:
            run_id = "node-1-20260101T000000Z-5678"
            _seed_run(
                tmp_path,
                run_id=run_id,
                node_id=1,
                status="running",
                log_path=str(outside_dir),
            )
            final = _call(milknado_run_loop_poll, run_id=run_id, project_root=str(tmp_path))
            assert final["summary"] == "", "a directory outside project_root must not be tailed"
        finally:
            import shutil

            shutil.rmtree(outside_dir, ignore_errors=True)

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
                RunResult(
                    status="done",
                    exit_code=0,
                    timed_out=False,
                    ended_at="2026-01-01T00:00:00Z",
                ),
            )
            graph._conn.commit()
        finally:
            graph.close()
        final = _call(milknado_run_inline_poll, run_id=run_id, project_root=str(tmp_path))
        assert final["result"] == payload
        assert final["result"].count("\n") == 39, "every line of the deliverable must survive"


def test_finish_dispatch_rejects_lost_run_fence() -> None:
    from types import SimpleNamespace

    from milknado.domains.dispatch import lifecycle

    class Graph:
        def finish_run(self, *_args) -> bool:
            return False

    with pytest.raises(RuntimeError, match="terminal run write lost its fence"):
        lifecycle._finish_dispatch(
            Graph(), 1, "run-1", SimpleNamespace(exit_code=0, timed_out=False), "done", None
        )


def test_finish_dispatch_allows_deleted_node_after_lost_node_fence() -> None:
    from types import SimpleNamespace

    from milknado.domains.dispatch import lifecycle

    class Graph:
        def finish_run(self, *_args) -> bool:
            return True

        def mark_terminal(self, *_args) -> bool:
            return False

        def get_node(self, _node_id: int) -> None:
            return None

    lifecycle._finish_dispatch(
        Graph(), 1, "run-1", SimpleNamespace(exit_code=0, timed_out=False), "done", None
    )


def test_sync_dispatch_reports_terminal_persistence_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from milknado.domains.dispatch import lifecycle
    from milknado.domains.dispatch.lifecycle import SyncDispatchRequest

    class Graph:
        def get_node(self, _node_id: int):
            return SimpleNamespace(kind=NodeKind.TASK)

        def claim_node_for_dispatch(self, *_args, **_kwargs) -> None:
            return None

        def start_run(self, *_args, **_kwargs) -> None:
            return None

        def finish_run(self, *_args, **_kwargs) -> bool:
            return False

        def mark_terminal(self, *_args, **_kwargs) -> bool:
            return True

    monkeypatch.setattr(lifecycle, "render_brief", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        lifecycle,
        "run_headless",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("worker failed")),
    )
    request = SyncDispatchRequest(
        node_id=1,
        project_root=tmp_path,
        worker_cmd=None,
        timeout_seconds=1,
        default_cmd="claude",
        process=object(),  # type: ignore[arg-type]
        worktree_mode=WorktreeMode.THIS_BRANCH,
    )
    with pytest.raises(RuntimeError, match="terminal persistence lost its fence"):
        lifecycle.dispatch_node_sync(Graph(), object(), request)  # type: ignore[arg-type]


def test_cancel_finalize_surfaces_late_write_loss() -> None:
    from milknado.domains.dispatch import cancel

    class Graph:
        def finish_run(self, *_args) -> bool:
            return False

        def get_run(self, _run_id: str) -> dict:
            return {"run_id": _run_id, "status": "done"}

    result = cancel._finalize_cancelled(Graph(), "run-1")
    assert result["terminal_persistence"] == "late-write-lost"


def test_stale_reconcile_rejects_lost_terminal_fence() -> None:
    from datetime import UTC, datetime, timedelta

    from milknado.domains.dispatch import reconcile

    class Graph:
        def runs_for_node(self, _node_id: int) -> list[dict]:
            return [
                {
                    "run_id": "run-stale",
                    "status": "running",
                    "started_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
                    "timeout_seconds": 1,
                }
            ]

        def finish_run(self, *_args) -> bool:
            return False

    with pytest.raises(RuntimeError, match="stale terminal write lost"):
        reconcile.fail_stale_running_runs(Graph(), 1)


def test_async_worker_writes_terminal_error_sidecar_on_persistence_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import milknado.domains.dispatch.async_run as async_run

    request = async_run.AsyncRunRequest(
        project_root=tmp_path,
        node_id=1,
        brief="brief",
        worker_cmd=None,
        timeout_seconds=1,
        run_id="node-1-20260101T000000Z-async",
        default_cmd="claude",
        cwd=tmp_path,
    )
    log_path = tmp_path / ".milknado" / "runs" / f"{request.run_id}.log"
    log_path.parent.mkdir(parents=True)
    log_path.touch()

    class Graph:
        def finish_run(self, *_args) -> bool:
            raise RuntimeError("database unavailable")

        def close(self) -> None:
            return None

    class Sessions:
        def open_graph(self, _project_root: Path):
            return Graph(), None

    monkeypatch.setattr(
        async_run,
        "_run_worker_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("worker failed")),
    )
    context = async_run.AsyncWorkerContext(
        request=request,
        log_path=log_path,
        argv=("claude",),
        graph_sessions=Sessions(),
        git=object(),  # type: ignore[arg-type]
        process=object(),  # type: ignore[arg-type]
    )
    async_run._async_worker(context)
    sidecar = tmp_path / ".milknado" / "runs" / f"{request.run_id}.terminal-error"
    assert "terminal persistence raised" in sidecar.read_text(encoding="utf-8")


def test_poll_async_run_reports_terminal_error_sidecar(tmp_path: Path) -> None:
    import milknado.domains.dispatch.async_run as async_run
    from milknado.domains.dispatch._runstate import runs_dir

    run_id = "node-1-20260101T000000Z-abcd"
    rdir = runs_dir(tmp_path)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"{run_id}.terminal-error").write_text("persistence failed", encoding="utf-8")

    class Graph:
        def get_run(self, _run_id: str) -> dict:
            return {"run_id": _run_id, "status": "failed", "log_path": "tampered"}

    result = async_run.poll_async_run(Graph(), tmp_path, run_id)
    assert result["terminal_persistence"] == "degraded"
    assert result["error"] == "persistence failed"


def test_finish_dispatch_rejects_lost_node_fence_when_node_remains() -> None:
    from types import SimpleNamespace

    from milknado.domains.dispatch import lifecycle

    class Graph:
        def finish_run(self, *_args) -> bool:
            return True

        def mark_terminal(self, *_args) -> bool:
            return False

        def get_node(self, _node_id: int):
            return SimpleNamespace(id=_node_id)

    with pytest.raises(RuntimeError, match="terminal node write lost its fence"):
        lifecycle._finish_dispatch(
            Graph(), 1, "run-1", SimpleNamespace(exit_code=0, timed_out=False), "done", None
        )


def test_sync_dispatch_preserves_terminal_persistence_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from milknado.domains.dispatch import lifecycle
    from milknado.domains.dispatch.lifecycle import SyncDispatchRequest

    class Graph:
        def get_node(self, _node_id: int):
            return SimpleNamespace(kind=NodeKind.TASK)

        def claim_node_for_dispatch(self, *_args, **_kwargs) -> None:
            return None

        def start_run(self, *_args, **_kwargs) -> None:
            return None

        def finish_run(self, *_args, **_kwargs) -> bool:
            return True

        def mark_terminal(self, *_args, **_kwargs) -> bool:
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(lifecycle, "render_brief", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        lifecycle,
        "run_headless",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("worker failed")),
    )
    request = SyncDispatchRequest(
        node_id=1,
        project_root=tmp_path,
        worker_cmd=None,
        timeout_seconds=1,
        default_cmd="claude",
        process=object(),  # type: ignore[arg-type]
        worktree_mode=WorktreeMode.THIS_BRANCH,
    )
    with pytest.raises(RuntimeError, match="database unavailable"):
        lifecycle.dispatch_node_sync(Graph(), object(), request)  # type: ignore[arg-type]


def test_async_worker_reports_cancelled_run_fence_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import milknado.domains.dispatch.async_run as async_run

    request = async_run.AsyncRunRequest(
        project_root=tmp_path,
        node_id=1,
        brief="brief",
        worker_cmd=None,
        timeout_seconds=1,
        run_id="node-1-20260101T000000Z-cafe",
        default_cmd="claude",
        cwd=tmp_path,
    )
    log_path = tmp_path / ".milknado" / "runs" / f"{request.run_id}.log"
    log_path.parent.mkdir(parents=True)
    log_path.touch()

    class Graph:
        def finish_run(self, *_args) -> bool:
            return False

        def close(self) -> None:
            return None

    class Sessions:
        def open_graph(self, _project_root: Path):
            return Graph(), None

    monkeypatch.setattr(
        async_run, "_run_worker_process", lambda *_args, **_kwargs: (0, False, True)
    )
    context = async_run.AsyncWorkerContext(
        request=request,
        log_path=log_path,
        argv=("claude",),
        graph_sessions=Sessions(),
        git=object(),  # type: ignore[arg-type]
        process=object(),  # type: ignore[arg-type]
    )
    async_run._async_worker(context)
