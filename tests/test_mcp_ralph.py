from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from milknado.mcp_ralph import milknado_ralph_run_poll, milknado_ralph_run_start
from milknado.mcp_server import open_graph
from milknado.mcp_todo import milknado_todo_add, milknado_todo_tree

# A stub runner standing in for `python -m milknado._ralph_node_runner`: it honors
# the same argv the MCP tool appends and writes the requested terminal state,
# letting us exercise the start->spawn->poll plumbing without a real ralph loop.
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
        detail={detail},
    )
finally:
    graph.close()
"""


# A stub runner that echoes its inherited MILKNADO_NODE_ID into the run detail,
# letting us assert the detached spawn injects the env var (parenting fix).
_ENV_CAPTURE_RUNNER = """
import argparse, os
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
        status="done",
        exit_code=0,
        timed_out=False,
        ended_at="2026-01-01T00:00:00+00:00",
        rebased=True,
        detail=os.environ.get("MILKNADO_NODE_ID", "<unset>"),
    )
finally:
    graph.close()
"""


def _call(tool, **kwargs):
    fn = getattr(tool, "fn", tool)
    return fn(**kwargs)


def _read_run(root: Path, run_id: str) -> dict | None:
    graph, _cfg = open_graph(root)
    try:
        return graph.get_run(run_id)
    finally:
        graph.close()


def _node_runs(root: Path, node_id: int) -> list[dict]:
    graph, _cfg = open_graph(root)
    try:
        return graph.runs_for_node(node_id)
    finally:
        graph.close()


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
                timed_out=False,
                ended_at=ended_at or datetime.now(UTC).isoformat(),
            )
        graph._conn.commit()
    finally:
        graph.close()


def _stub_runner_cmd(tmp_path: Path, *, status: str, rebased: bool, detail: str = "None") -> str:
    script = tmp_path / f"stub_runner_{status}.py"
    script.write_text(
        _STUB_RUNNER.format(
            status=status,
            rebased=rebased,
            detail=detail,
            exit_code=0 if status == "done" else 1,
        )
    )
    return f"{sys.executable} {script}"


def _wait_for_terminal(run_id: str, root: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = _call(milknado_ralph_run_poll, run_id=run_id, project_root=root)
        if last["status"] in ("done", "failed"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish; last={last}")


def test_start_returns_run_id_and_writes_running_state(tmp_path: Path) -> None:
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="ralph-node", kind="task", project_root=root)
    # A do-nothing runner so the state stays "running" right after start.
    noop = f"{sys.executable} -c pass"
    started = _call(
        milknado_ralph_run_start,
        node_id=task["id"],
        runner_cmd=noop,
        project_root=root,
    )
    assert started["status"] == "running"
    assert started["run_id"].startswith(f"node-{task['id']}-")
    assert _read_run(tmp_path, started["run_id"])["status"] == "running"


def test_start_records_pid_in_running_state(tmp_path: Path) -> None:
    """The detached pid is persisted on the running run row (and returned) so
    poll/reconcile can liveness-check a loop that outlives this server."""
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="pid", kind="task", project_root=root)
    noop = f"{sys.executable} -c pass"
    started = _call(
        milknado_ralph_run_start,
        node_id=task["id"],
        runner_cmd=noop,
        project_root=root,
    )
    assert isinstance(started["pid"], int)
    state = _read_run(tmp_path, started["run_id"])
    assert state["pid"] == started["pid"]


def test_detached_runner_receives_node_id_in_env(tmp_path: Path) -> None:
    """The detached ralph runner must spawn with MILKNADO_NODE_ID set so
    milknado_track_follow_up parents follow-ups as siblings of the executing
    node (under its parent) instead of at the graph root (mirrors the headless
    dispatch path). The stub runner echoes its inherited MILKNADO_NODE_ID into
    the run detail; we assert it equals the node being executed. Fails if the
    spawn drops the var."""
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="env-node-id", kind="task", project_root=root)
    script = tmp_path / "env_capture_runner.py"
    script.write_text(_ENV_CAPTURE_RUNNER)
    started = _call(
        milknado_ralph_run_start,
        node_id=task["id"],
        runner_cmd=f"{sys.executable} {script}",
        project_root=root,
    )
    final = _wait_for_terminal(started["run_id"], root)
    assert final["status"] == "done"
    assert final["detail"] == str(task["id"])


def test_poll_reads_done_after_runner_finishes(tmp_path: Path) -> None:
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="ralph-done", kind="task", project_root=root)
    started = _call(
        milknado_ralph_run_start,
        node_id=task["id"],
        runner_cmd=_stub_runner_cmd(tmp_path, status="done", rebased=True),
        project_root=root,
    )
    final = _wait_for_terminal(started["run_id"], root)
    assert final["status"] == "done"
    assert final["rebased"] is True


def test_poll_reads_failed_with_detail(tmp_path: Path) -> None:
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="ralph-fail", kind="task", project_root=root)
    started = _call(
        milknado_ralph_run_start,
        node_id=task["id"],
        runner_cmd=_stub_runner_cmd(tmp_path, status="failed", rebased=False, detail="'boom'"),
        project_root=root,
    )
    final = _wait_for_terminal(started["run_id"], root)
    assert final["status"] == "failed"
    assert final["rebased"] is False
    assert final["detail"] == "boom"


def test_start_refuses_when_node_already_running(tmp_path: Path) -> None:
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="busy", kind="task", project_root=root)
    graph, _cfg = open_graph(tmp_path)
    try:
        graph.mark_running(task["id"])
    finally:
        graph.close()
    with pytest.raises(ValueError, match="already running"):
        _call(
            milknado_ralph_run_start,
            node_id=task["id"],
            runner_cmd=f"{sys.executable} -c pass",
            project_root=root,
        )


def test_start_unknown_node_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        _call(milknado_ralph_run_start, node_id=99, project_root=str(tmp_path))


def test_poll_unknown_run_id_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        _call(
            milknado_ralph_run_poll,
            run_id="node-1-20260101T000000Z-abcd",
            project_root=str(tmp_path),
        )


def test_poll_rejects_malformed_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid run_id"):
        _call(milknado_ralph_run_poll, run_id="../etc/passwd", project_root=str(tmp_path))


def test_start_claims_node_running_before_spawn(tmp_path: Path) -> None:
    """start now atomically claims the node RUNNING in the parent (the cross-process
    mutual-exclusion point) BEFORE spawning the detached runner, so a second
    concurrent caller loses the claim. With a no-op runner that never reaches a
    terminal write, the node stays RUNNING under the claim."""
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="pend", kind="task", project_root=root)
    started = _call(
        milknado_ralph_run_start,
        node_id=task["id"],
        runner_cmd=f"{sys.executable} -c pass",
        project_root=root,
    )
    tree = _call(milknado_todo_tree, project_root=root)
    assert tree[0]["status"] == "running"
    # the node carries the claim run_id as its fence (matches the run-state file name)
    graph, _cfg = open_graph(tmp_path)
    try:
        node = graph.get_node(task["id"])
    finally:
        graph.close()
    assert node is not None
    assert node.run_id == started["run_id"]
    assert node.pid == started["pid"]


def test_start_marks_run_failed_when_spawn_raises(tmp_path: Path) -> None:
    """A runner_cmd that cannot be spawned must not leave the run stuck at
    'running' forever: the spawn error is caught, the run row turned terminal, and
    the error re-raised so the caller sees a clean failure."""
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="badspawn", kind="task", project_root=root)
    bad_cmd = str(tmp_path / "no-such-runner-binary")
    with pytest.raises(OSError):
        _call(
            milknado_ralph_run_start,
            node_id=task["id"],
            runner_cmd=bad_cmd,
            project_root=root,
        )
    runs = _node_runs(tmp_path, task["id"])
    assert len(runs) == 1, "exactly one run row should have been written"
    state = runs[0]
    assert state["status"] == "failed"
    assert state["rebased"] is False
    assert "spawn failed" in state["detail"]
    assert state["ended_at"] is not None
    # The node is claimed RUNNING in the parent BEFORE the spawn attempt, so the
    # spawn-failure path must release that claim (fenced mark_terminal FAILED) —
    # otherwise the node is stranded RUNNING forever and no retry can re-claim it.
    from milknado.domains.common import NodeStatus

    graph, _cfg = open_graph(tmp_path)
    try:
        node = graph.get_node(task["id"])
    finally:
        graph.close()
    assert node is not None
    assert node.status == NodeStatus.FAILED, "spawn failure frees the claim, no strand RUNNING"
    assert node.run_id is None, "the fence run_id is cleared so the node is freed for re-dispatch"


def test_poll_derives_log_path_from_run_id(tmp_path: Path) -> None:
    """Poll derives the log path from the validated run_id, not the stored field,
    so a tampered log_path can't read an arbitrary file. With no log file on disk
    the summary is an empty tail rather than an error."""
    root = str(tmp_path)
    run_id = "node-1-20260101T000000Z-abcd"
    _seed_run(tmp_path, run_id=run_id, node_id=1, status="running")
    state = _call(milknado_ralph_run_poll, run_id=run_id, project_root=root)
    assert state["status"] == "running"
    assert state["summary"] == ""  # derived log file absent -> empty tail, never KeyError


def test_runner_crash_writes_detail_and_keeps_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the detached runner crashes mid-execution it must finalize the run row
    'failed' with the error under the documented 'detail' field."""
    import milknado.adapters as adapters
    from milknado import _ralph_node_runner

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("boom")

    # GitAdapter is the first adapter the runner builds; raising there exercises
    # the runner's terminal failed-write path after open_graph succeeded.
    monkeypatch.setattr(adapters, "GitAdapter", _boom)
    run_id = "node-1-20260101T000000Z-abcd"
    _seed_run(tmp_path, run_id=run_id, node_id=1, status="running")
    rc = _ralph_node_runner.main(
        ["--node-id", "1", "--project-root", str(tmp_path), "--run-id", run_id, "--timeout", "12"]
    )
    assert rc == 1
    state = _read_run(tmp_path, run_id)
    assert state["status"] == "failed"
    assert state["rebased"] is False
    assert state["detail"] == "RuntimeError: boom"
    assert state["log_path"].endswith(f"{run_id}.log")
    assert state["timeout_seconds"] == 10


def test_runner_writes_done_on_successful_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The detached runner's happy path: open the graph, build adapters, run the
    node to completion, then overwrite the state file with a terminal 'done' the
    poll reads — preserving the base schema (log_path/timeout_seconds)."""
    import milknado._mcp_core as mcp_core
    import milknado.adapters as adapters
    import milknado.domains.execution as execution
    from milknado import _ralph_node_runner
    from milknado.domains.execution.headless import HeadlessOutcome

    class _Cfg:
        execution_agent = "claude"
        quality_gates = ()
        worktree_pattern = "wt-{node}"
        flavors: dict = {}
        worker_brief_prepend = None
        agent_family = "claude"

    class _Graph:
        def __init__(self) -> None:
            self.closed = False
            self.finished: dict | None = None

        def get_node(self, node_id: int) -> None:  # noqa: ARG002
            return None

        def finish_run(self, run_id: str, **fields) -> None:  # noqa: ANN003
            self.finished = {"run_id": run_id, **fields}

        def deposit_run_message(self, *a, **k) -> int:  # noqa: ANN002, ANN003
            return 1

        def close(self) -> None:
            self.closed = True

    class _Git:
        def __init__(self, _root: object) -> None: ...

        def current_branch(self) -> str:
            return "main"

    graph = _Graph()
    monkeypatch.setattr(mcp_core, "open_graph", lambda _root: (graph, _Cfg()))
    monkeypatch.setattr(adapters, "GitAdapter", _Git)

    class _StubRalph:
        def poll_progress_events(self) -> list:
            return []

    monkeypatch.setattr(adapters, "RalphifyAdapter", lambda *a, **k: _StubRalph())
    monkeypatch.setattr(adapters, "CrgAdapter", lambda *a, **k: object())
    monkeypatch.setattr(execution, "Executor", lambda **k: object())
    monkeypatch.setattr(execution, "ExecutionConfig", lambda **k: object())
    monkeypatch.setattr(
        execution,
        "run_node_to_completion",
        lambda *a, **k: HeadlessOutcome(node_id=1, success=True, detail=None),
    )

    run_id = "node-1-20260101T000000Z-abcd"
    rc = _ralph_node_runner.main(
        ["--node-id", "1", "--project-root", str(tmp_path), "--run-id", run_id, "--timeout", "30"]
    )
    assert rc == 0
    assert graph.closed is True  # the graph handle is always released
    assert graph.finished is not None
    assert graph.finished["status"] == "done"
    assert graph.finished["rebased"] is True
    assert graph.finished["run_id"] == run_id


def test_runner_calls_stop_run_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A timed-out run must call stop_run on the ralph adapter so the underlying
    loop does not outlive its timeout as a zombie process. Exercises the full
    _ralph_node_runner.main path without monkeypatching run_node_to_completion,
    so the real CompletionTimeout handler is exercised end-to-end."""
    import milknado._mcp_core as mcp_core
    import milknado.adapters as adapters
    import milknado.domains.execution as execution
    from milknado import _ralph_node_runner
    from milknado.domains.common.errors import CompletionTimeout
    from milknado.domains.execution.executor import DispatchResult

    class _Cfg:
        execution_agent = "claude"
        quality_gates = ()
        worktree_pattern = "wt-{node}"
        flavors: dict = {}
        worker_brief_prepend = None
        agent_family = "claude"

    class _Graph:
        def __init__(self) -> None:
            self.closed = False
            self.finished: dict | None = None

        def get_node(self, node_id: int) -> None:  # noqa: ARG002
            return None

        def finish_run(self, run_id: str, **fields) -> None:  # noqa: ANN003
            self.finished = {"run_id": run_id, **fields}

        def deposit_run_message(self, *a, **k) -> int:  # noqa: ANN002, ANN003
            return 1

        def close(self) -> None:
            self.closed = True

    class _Git:
        def __init__(self, _root: object) -> None: ...

        def current_branch(self) -> str:
            return "main"

    class _StubRalph:
        def __init__(self) -> None:
            self.stopped: list[str] = []

        def wait_for_next_completion(self, active_run_ids, timeout=None):  # noqa: ANN001
            raise CompletionTimeout(active_run_ids=active_run_ids, waited_seconds=timeout or 0.0)

        def stop_run(self, run_id: str) -> None:
            self.stopped.append(run_id)

        def poll_progress_events(self) -> list:
            return []

    class _StubExecutor:
        def dispatch(self, node_id: int, config) -> DispatchResult:  # noqa: ANN001
            return DispatchResult(
                node_id=node_id, worktree=Path("/tmp/wt"), run_id=f"run-{node_id}"
            )

        def fail(self, node_id: int) -> None:
            pass

    stub_ralph = _StubRalph()
    graph = _Graph()
    monkeypatch.setattr(mcp_core, "open_graph", lambda _root: (graph, _Cfg()))
    monkeypatch.setattr(adapters, "GitAdapter", _Git)
    monkeypatch.setattr(adapters, "RalphifyAdapter", lambda *a, **k: stub_ralph)
    monkeypatch.setattr(adapters, "CrgAdapter", lambda *a, **k: object())
    monkeypatch.setattr(execution, "Executor", lambda **k: _StubExecutor())
    monkeypatch.setattr(execution, "ExecutionConfig", lambda **k: object())

    run_id = "node-1-20260101T000000Z-abcd"
    rc = _ralph_node_runner.main(
        ["--node-id", "1", "--project-root", str(tmp_path), "--run-id", run_id, "--timeout", "5"]
    )
    assert rc == 1
    assert stub_ralph.stopped == ["run-1"], "timeout must call stop_run on the ralph adapter"
    assert graph.finished is not None
    assert graph.finished["status"] == "failed"
    assert "timeout" in (graph.finished.get("detail") or "")


def test_resolve_runner_cmd_prefers_explicit_then_env_then_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner resolution order: explicit arg > MILKNADO_RALPH_RUNNER_CMD > default."""
    from milknado.mcp_ralph import _DEFAULT_RUNNER, _resolve_runner_cmd

    monkeypatch.delenv("MILKNADO_RALPH_RUNNER_CMD", raising=False)
    assert _resolve_runner_cmd("/bin/run --flag") == ["/bin/run", "--flag"]
    assert _resolve_runner_cmd("   ") == list(_DEFAULT_RUNNER)  # blank falls through
    assert _resolve_runner_cmd(None) == list(_DEFAULT_RUNNER)

    monkeypatch.setenv("MILKNADO_RALPH_RUNNER_CMD", "envrunner --x")
    assert _resolve_runner_cmd(None) == ["envrunner", "--x"]
    assert _resolve_runner_cmd("explicit-wins") == ["explicit-wins"]


def test_start_reconciles_orphaned_terminal_run_then_proceeds(tmp_path: Path) -> None:
    """A node stuck RUNNING because a finished run was never polled is reconciled
    (the loop at the top of start), freeing it so a fresh claim can spawn. A
    'failed' orphan reconciles to FAILED, which the new claim can re-acquire (a
    'done' orphan reconciles to DONE and is correctly refused — done is terminal)."""
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="orphan", kind="task", project_root=root)
    node_id = task["id"]
    # Mark the node RUNNING and drop a terminal 'failed' state file for it that no
    # one polled — exactly the orphan the start-time reconcile loop must clear.
    graph, _cfg = open_graph(tmp_path)
    try:
        graph.mark_running(node_id)
    finally:
        graph.close()
    orphan_id = f"node-{node_id}-20260101T000000Z-dead"
    _seed_run(
        tmp_path,
        run_id=orphan_id,
        node_id=node_id,
        status="failed",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T01:00:00+00:00",
        timeout_seconds=1800,
        exit_code=1,
    )
    started = _call(
        milknado_ralph_run_start,
        node_id=node_id,
        runner_cmd=f"{sys.executable} -c pass",
        project_root=root,
    )
    # Reconcile freed the node (FAILED), so the fresh claim wins and start spawns.
    assert started["status"] == "running"
    graph, _cfg = open_graph(tmp_path)
    try:
        node = graph.get_node(node_id)
    finally:
        graph.close()
    assert node is not None
    assert node.run_id == started["run_id"]  # re-claimed under the new run, not relocked


def test_start_refuses_when_node_running_with_live_pid(tmp_path: Path) -> None:
    """A node RUNNING under a live owner pid with a non-expired claim is refused —
    no second worker. (This process's own pid stands in for a live owner.)"""
    import os

    from milknado.domains.dispatch._runstate import now_iso

    root = str(tmp_path)
    task = _call(milknado_todo_add, description="live", kind="task", project_root=root)
    node_id = task["id"]
    graph, _cfg = open_graph(tmp_path)
    try:
        graph.claim_node(node_id, "node-1-20260101T000000Z-live", now=now_iso())
        graph.set_pid(node_id, "node-1-20260101T000000Z-live", os.getpid())
    finally:
        graph.close()
    with pytest.raises(ValueError, match="already running"):
        _call(
            milknado_ralph_run_start,
            node_id=node_id,
            runner_cmd=f"{sys.executable} -c pass",
            project_root=root,
        )


def test_start_reclaims_dead_pid_without_timeout_wait(tmp_path: Path) -> None:
    """A node left RUNNING by a DEAD pid is reclaimed on the next start and a fresh
    worker dispatched immediately — no 1800s timeout wait. The dead owner is
    detected by process-liveness (os.kill(pid, 0)), not by the stale-timeout sweep."""
    from milknado.domains.dispatch._runstate import now_iso

    root = str(tmp_path)
    task = _call(milknado_todo_add, description="deadpid", kind="task", project_root=root)
    node_id = task["id"]
    dead_pid = 2**31 - 1  # unreapable: no process holds it
    dead_run_id = f"node-{node_id}-20260101T000000Z-dead"
    graph, _cfg = open_graph(tmp_path)
    try:
        # Simulate a runner that claimed, recorded its pid, then died without
        # writing a terminal state — node stuck RUNNING, well within the 1800s
        # timeout (so only pid-liveness, not the stale sweep, can free it).
        graph.claim_node(node_id, dead_run_id, now=now_iso())
        graph.set_pid(node_id, dead_run_id, dead_pid)
    finally:
        graph.close()
    started = _call(
        milknado_ralph_run_start,
        node_id=node_id,
        runner_cmd=f"{sys.executable} -c pass",
        project_root=root,
    )
    assert started["status"] == "running"
    assert started["run_id"] != dead_run_id  # a fresh run won the reclaimed node
    graph, _cfg = open_graph(tmp_path)
    try:
        node = graph.get_node(node_id)
    finally:
        graph.close()
    assert node is not None
    assert node.run_id == started["run_id"]


def test_concurrent_ralph_run_start_spawns_exactly_one_worker(tmp_path: Path) -> None:
    """Two concurrent milknado_ralph_run_start calls on the same PENDING node must
    spawn exactly one detached worker; the loser raises 'already running'. The
    atomic claim_node UPDATE — not an in-process lock — is the cross-process gate."""
    import threading

    root = str(tmp_path)
    task = _call(milknado_todo_add, description="concurrent-ralph", kind="task", project_root=root)
    node_id = task["id"]
    # A no-op runner that never reaches terminal, so a leaked second claim would
    # leave the node RUNNING under two run_ids and we'd see two successes.
    noop = f"{sys.executable} -c pass"

    results: list[dict] = []
    errors: list[str] = []
    result_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def _start() -> None:
        barrier.wait()
        try:
            r = _call(
                milknado_ralph_run_start,
                node_id=node_id,
                runner_cmd=noop,
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

    assert len(results) == 1, f"expected exactly 1 successful dispatch, errors={errors}"
    assert len(errors) == 1, f"expected exactly 1 'already running', errors={errors}"
    assert "already running" in errors[0]


def test_orphan_worktree_removed_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#50: milknado_ralph_run_start must prune an orphaned worktree left by a
    killed runner before spawning a new run so git-worktree-add does not fail."""
    import milknado.adapters as adapters

    removed: list[Path] = []

    def _stub_remove(self: object, wt: Path) -> None:  # noqa: ANN001
        removed.append(wt)

    monkeypatch.setattr(adapters.GitAdapter, "remove_worktree", _stub_remove)

    root = str(tmp_path)
    task = _call(milknado_todo_add, description="orphan-wt", kind="task", project_root=root)
    node_id = task["id"]

    # Simulate a node stuck RUNNING with a worktree path that physically exists
    orphan_wt = tmp_path / "milknado-orphan-wt"
    orphan_wt.mkdir()

    graph, _cfg = open_graph(tmp_path)
    try:
        graph.mark_running(node_id, worktree_path=str(orphan_wt), branch_name="milknado/orphan")
    finally:
        graph.close()

    # Seed a terminal 'failed' run so orphan reconcile has something to act on.
    # (A 'done' orphan would reconcile to DONE — a terminal state the claim correctly
    # refuses. Use 'failed' so the node is freed to FAILED and the fresh claim wins.)
    orphan_id = f"node-{node_id}-20260101T000000Z-dead"
    _seed_run(
        tmp_path,
        run_id=orphan_id,
        node_id=node_id,
        status="failed",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T01:00:00+00:00",
        timeout_seconds=1800,
        exit_code=1,
    )

    started = _call(
        milknado_ralph_run_start,
        node_id=node_id,
        runner_cmd=f"{sys.executable} -c pass",
        project_root=root,
    )
    assert started["status"] == "running"
    assert removed == [orphan_wt], "orphaned worktree must be removed before retry"
