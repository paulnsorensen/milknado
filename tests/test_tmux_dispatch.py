"""Tmux run-substrate behaviour against a fake adapter: fail-closed dispatch,
window lifecycle, attach preconditions, and per-row reconciliation."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from milknado.adapters.tmux import TmuxDispatchError
from milknado.domains.common import NodeStatus, RunResult, WorktreeMode, pid_alive
from milknado.domains.dispatch import (
    RunWindow,
    ensure_tmux_ready,
    reconcile_run_window,
    resolve_attach_target,
    runs_dir,
)
from milknado.domains.dispatch._runstate import request_cancel
from milknado.domains.dispatch.tmux_run import (
    cleanup_run_window,
    execute_in_window,
    read_exit_code,
)
from milknado.mcp.ralph import milknado_run_loop_start
from milknado.mcp.run import (
    milknado_run_inline_poll,
    milknado_run_inline_start,
)
from milknado.mcp.server import open_graph
from milknado.mcp.todo_mutate import milknado_todo_add

RUN_ID = "node-1-20260101T000000Z-deadbeef"


def _call(tool, **kwargs):
    fn = getattr(tool, "fn", tool)
    # run_inline{,_start} now default to ISOLATE (per-dispatch worktree). These
    # legacy tmux tests target the shared-checkout path, so pin THIS_BRANCH.
    if fn.__name__ in ("milknado_run_inline", "milknado_run_inline_start"):
        kwargs.setdefault("worktree", WorktreeMode.THIS_BRANCH)
    return fn(**kwargs)


class FakeTmux:
    """In-memory stand-in for TmuxAdapter honouring its window-set semantics."""

    session_name = "milknado-proj"

    def __init__(self, *, available: bool = True, fail_session: bool = False) -> None:
        self._available = available
        self._fail_session = fail_session
        self.windows: set[str] = set()
        self.opened: list[RunWindow] = []
        self.killed: list[str] = []
        self.pane_pid = os.getpid()

    def available(self) -> bool:
        return self._available

    def ensure_session(self) -> None:
        if self._fail_session:
            raise TmuxDispatchError("no server")

    def window_exists(self, run_id: str) -> bool:
        return run_id in self.windows

    def target_for(self, run_id: str) -> str:
        return f"={self.session_name}:={run_id}"

    def kill_window(self, run_id: str) -> bool:
        if run_id not in self.windows:
            return False
        self.windows.discard(run_id)
        self.killed.append(run_id)
        return True

    def open_run_window(self, window: RunWindow) -> int:
        if window.run_id in self.windows:
            raise TmuxDispatchError(f"tmux window {window.run_id!r} already exists")
        self.windows.add(window.run_id)
        self.opened.append(window)
        return self.pane_pid


class SpawningFakeTmux(FakeTmux):
    """Fake that runs a real (reaped) subprocess honouring the window contract:
    brief consumed, output to the log, exit code to the rc file."""

    def __init__(self, script: str) -> None:
        super().__init__()
        self._script = script
        self.procs: list[subprocess.Popen] = []

    def open_run_window(self, window: RunWindow) -> int:
        super().open_run_window(window)
        cmd = self._script.format(
            brief=window.brief_path, log=window.log_path, rc=window.exit_code_path
        )
        proc = subprocess.Popen(["sh", "-c", cmd], start_new_session=True)
        self.procs.append(proc)
        # Reap promptly so pid-liveness sees the death (tmux would do this).
        threading.Thread(target=proc.wait, daemon=True).start()
        return proc.pid


# --- fail-closed preflight -------------------------------------------------


def test_ensure_tmux_ready_passes_when_available() -> None:
    ensure_tmux_ready(FakeTmux())


def test_ensure_tmux_ready_fails_closed_when_binary_missing() -> None:
    with pytest.raises(ValueError, match="not on PATH"):
        ensure_tmux_ready(FakeTmux(available=False))


def test_ensure_tmux_ready_fails_closed_when_server_cannot_start() -> None:
    with pytest.raises(ValueError, match="server could not start"):
        ensure_tmux_ready(FakeTmux(fail_session=True))


# --- loop-path dispatch (milknado_run_loop_start) ---------------------------


def _add_task(root: str) -> int:
    project = Path(root)
    if not (project / ".git").exists():
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=project, check=True)
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
            cwd=project,
            check=True,
        )
    return _call(
        milknado_todo_add,
        description="tmux-task",
        kind="task",
        project_root=root,
    )["id"]


def _node_status(root, node_id: int) -> NodeStatus:
    graph, _cfg = open_graph(Path(root))
    try:
        return graph.get_node(node_id).status
    finally:
        graph.close()


def test_loop_start_without_use_tmux_never_touches_tmux(tmp_path, monkeypatch) -> None:
    """The default detached path must not construct a tmux adapter at all."""

    def _boom(*args, **kwargs):
        raise AssertionError("TmuxAdapter constructed on the default path")

    monkeypatch.setattr("milknado.app.ralph.TmuxAdapter", _boom)
    monkeypatch.setattr("milknado.mcp.ralph.TmuxAdapter", _boom)
    root = str(tmp_path)
    started = _call(
        milknado_run_loop_start,
        node_id=_add_task(root),
        runner_cmd=f"{sys.executable} -c pass",
        project_root=root,
    )
    assert started["status"] == "running"


def test_loop_start_use_tmux_fails_closed_and_leaves_node_pending(tmp_path, monkeypatch) -> None:
    fake = FakeTmux(available=False)
    monkeypatch.setattr("milknado.app.ralph.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.mcp.ralph.TmuxAdapter", lambda root: fake)
    root = str(tmp_path)
    node_id = _add_task(root)
    with pytest.raises(ValueError, match="not on PATH"):
        _call(
            milknado_run_loop_start,
            node_id=node_id,
            runner_cmd=f"{sys.executable} -c pass",
            use_tmux=True,
            project_root=root,
        )
    assert _node_status(root, node_id) == NodeStatus.PENDING
    assert fake.opened == []


def test_loop_start_use_tmux_opens_window_named_after_run_id(tmp_path, monkeypatch) -> None:
    fake = FakeTmux()
    monkeypatch.setattr("milknado.app.ralph.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.mcp.ralph.TmuxAdapter", lambda root: fake)
    root = str(tmp_path)
    started = _call(
        milknado_run_loop_start,
        node_id=_add_task(root),
        runner_cmd=f"{sys.executable} -c pass",
        use_tmux=True,
        project_root=root,
    )
    (window,) = fake.opened
    assert window.run_id == started["run_id"]
    assert window.argv[:3] == (sys.executable, "-c", "pass")
    assert window.argv[window.argv.index("--run-id") + 1] == started["run_id"]
    assert window.env["MILKNADO_RUN_ID"] == started["run_id"]
    assert window.log_path.exists()
    # The pane pid stands in for Popen's pid: returned AND persisted on the run
    # row so pid-liveness reclaim and cancel keep working for tmux runs.
    assert started["pid"] == fake.pane_pid
    graph, _cfg = open_graph(tmp_path)
    try:
        assert graph.get_run(started["run_id"])["pid"] == fake.pane_pid
    finally:
        graph.close()


def test_loop_start_window_collision_releases_claim_as_failed(tmp_path, monkeypatch) -> None:
    """A tmux failure after the claim must release the node (mirrors the
    OSError spawn-failure guard), not strand it RUNNING."""
    fake = FakeTmux()
    monkeypatch.setattr("milknado.app.ralph.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.mcp.ralph.TmuxAdapter", lambda root: fake)

    def _collide(window: RunWindow) -> int:
        raise TmuxDispatchError(f"tmux window {window.run_id!r} already exists")

    fake.open_run_window = _collide
    root = str(tmp_path)
    node_id = _add_task(root)
    with pytest.raises(TmuxDispatchError, match="already exists"):
        _call(
            milknado_run_loop_start,
            node_id=node_id,
            runner_cmd=f"{sys.executable} -c pass",
            use_tmux=True,
            project_root=root,
        )
    assert _node_status(root, node_id) == NodeStatus.FAILED


# --- run-inline path (milknado_run_inline_start) --------------------------------


def _wait_for_terminal(root: str, run_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict | None = None
    while time.monotonic() < deadline:
        last = _call(milknado_run_inline_poll, run_id=run_id, project_root=root)
        if last["status"] in ("done", "failed"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish; last={last}")


def test_run_inline_use_tmux_fails_closed_and_leaves_node_pending(tmp_path, monkeypatch) -> None:
    fake = FakeTmux(available=False)
    monkeypatch.setattr("milknado.app.run.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.mcp.run.TmuxAdapter", lambda root: fake)
    root = str(tmp_path)
    node_id = _add_task(root)
    with pytest.raises(ValueError, match="not on PATH"):
        _call(
            milknado_run_inline_start,
            node_id=node_id,
            worker_cmd="claude",
            use_tmux=True,
            project_root=root,
        )
    assert _node_status(root, node_id) == NodeStatus.PENDING


def test_run_inline_use_tmux_success_delivers_brief_without_poll_reconcile(
    tmp_path, monkeypatch
) -> None:
    """Worker reads the staged brief; normal polling never touches tmux."""
    fake = SpawningFakeTmux("cat {brief} >> {log}; echo 0 > {rc}")
    monkeypatch.setattr("milknado.app.run.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.mcp.run.TmuxAdapter", lambda root: fake)
    root = str(tmp_path)
    node_id = _add_task(root)
    started = _call(
        milknado_run_inline_start,
        node_id=node_id,
        worker_cmd="claude",
        use_tmux=True,
        project_root=root,
    )
    final = _wait_for_terminal(root, started["run_id"])
    assert final["status"] == "done"
    (window,) = fake.opened
    assert window.env["MILKNADO_NODE_ID"] == str(node_id)
    assert window.env["MILKNADO_RUN_ID"] == started["run_id"]
    assert window.env["MILKNADO_PROJECT_ROOT"] == str(tmp_path.resolve())
    assert "tmux-task" in final["summary"]  # brief made it through the stdin redirect
    assert _node_status(root, node_id) == NodeStatus.DONE
    assert fake.killed == [], "normal polling must not reconcile tmux windows"
    # The staged brief and rc file are transient: the worker's finally clears
    # them once the terminal state is written (log stays as the diagnostic).
    rdir = runs_dir(tmp_path)
    deadline = time.monotonic() + 5.0
    while (rdir / f"{started['run_id']}.brief").exists() and time.monotonic() < deadline:
        time.sleep(0.05)  # cleanup runs in the worker thread just after finish_run
    assert not (rdir / f"{started['run_id']}.brief").exists()
    assert not (rdir / f"{started['run_id']}.rc").exists()


def test_run_inline_use_tmux_failure_preserves_window(tmp_path, monkeypatch) -> None:
    fake = SpawningFakeTmux("echo boom >> {log}; echo 7 > {rc}; exit 7")
    monkeypatch.setattr("milknado.app.run.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.mcp.run.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.app.run.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.mcp.run.TmuxAdapter", lambda root: fake)
    root = str(tmp_path)
    started = _call(
        milknado_run_inline_start,
        node_id=_add_task(root),
        worker_cmd="claude",
        use_tmux=True,
        project_root=root,
    )
    final = _wait_for_terminal(root, started["run_id"])
    assert final["status"] == "failed"
    assert final["exit_code"] == 7
    # Failed run: window preserved for inspection — reconcile must NOT kill it.
    assert started["run_id"] not in fake.killed
    assert started["run_id"] in fake.windows


def test_run_inline_use_tmux_records_pane_pid_on_run_row(tmp_path, monkeypatch) -> None:
    """A tmux pane outlives an MCP-server restart, so the pane pid must land
    on the run row for the stale sweep's pid-liveness skip and cancel."""
    fake = SpawningFakeTmux("cat {brief} > /dev/null; echo 0 > {rc}")
    monkeypatch.setattr("milknado.app.run.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.mcp.run.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.app.run.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.mcp.run.TmuxAdapter", lambda root: fake)
    root = str(tmp_path)
    started = _call(
        milknado_run_inline_start,
        node_id=_add_task(root),
        worker_cmd="claude",
        use_tmux=True,
        project_root=root,
    )
    final = _wait_for_terminal(root, started["run_id"])
    assert final["pid"] == fake.procs[0].pid


def test_run_cancel_of_tmux_run_inline_kills_pane_group(tmp_path, monkeypatch) -> None:
    """With the pane pid recorded, milknado_run_cancel takes the pid route:
    the pane's process group is signalled and the run finalizes failed."""
    from milknado.mcp.run import milknado_run_cancel

    fake = SpawningFakeTmux("sleep 30")
    monkeypatch.setattr("milknado.app.run.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.mcp.run.TmuxAdapter", lambda root: fake)
    root = str(tmp_path)
    node_id = _add_task(root)
    started = _call(
        milknado_run_inline_start,
        node_id=node_id,
        worker_cmd="claude",
        use_tmux=True,
        project_root=root,
    )
    # Wait for the worker thread to persist the pane pid on the run row.
    graph, _cfg = open_graph(tmp_path)
    try:
        deadline = time.monotonic() + 5.0
        while graph.get_run(started["run_id"]).get("pid") is None:
            assert time.monotonic() < deadline, "pane pid never recorded"
            time.sleep(0.05)
    finally:
        graph.close()
    final = _call(milknado_run_cancel, run_id=started["run_id"], project_root=root)
    assert final["status"] == "failed"
    deadline = time.monotonic() + 5.0
    while pid_alive(fake.procs[0].pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_alive(fake.procs[0].pid)
    assert _node_status(root, node_id) == NodeStatus.FAILED


def test_run_inline_use_tmux_timeout_kills_pane_group(tmp_path, monkeypatch) -> None:
    fake = SpawningFakeTmux("sleep 30")
    monkeypatch.setattr("milknado.app.run.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.mcp.run.TmuxAdapter", lambda root: fake)
    root = str(tmp_path)
    started = _call(
        milknado_run_inline_start,
        node_id=_add_task(root),
        worker_cmd="claude",
        timeout_seconds=1,
        use_tmux=True,
        project_root=root,
    )
    final = _wait_for_terminal(root, started["run_id"])
    assert final["status"] == "failed"
    assert final["timed_out"] is True
    assert not pid_alive(fake.procs[0].pid)


# --- execute_in_window (pane waiter) ----------------------------------------


def test_execute_in_window_honours_cancel_sentinel(tmp_path) -> None:
    fake = SpawningFakeTmux("sleep 30")
    rdir = runs_dir(tmp_path)
    request_cancel(rdir, RUN_ID)
    window = RunWindow(
        run_id=RUN_ID,
        argv=("sh", "-c", "sleep 30"),
        cwd=tmp_path,
        log_path=rdir / f"{RUN_ID}.log",
        exit_code_path=rdir / f"{RUN_ID}.rc",
    )
    exit_code, timed_out, cancelled = execute_in_window(fake, window, 60, rdir)
    assert (exit_code, timed_out, cancelled) == (-1, False, True)
    assert not pid_alive(fake.procs[0].pid)


def test_terminate_pane_escalates_to_sigkill(monkeypatch) -> None:
    from milknado.domains.dispatch import tmux_run

    monkeypatch.setattr(tmux_run, "_PANE_KILL_GRACE_SECS", 0.2)
    proc = subprocess.Popen(["sh", "-c", 'trap "" TERM; sleep 30'], start_new_session=True)
    threading.Thread(target=proc.wait, daemon=True).start()
    time.sleep(0.2)  # let the trap install so SIGTERM is really ignored
    tmux_run._terminate_pane(proc.pid)
    deadline = time.monotonic() + 2.0
    while pid_alive(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.05)  # SIGKILL delivery + reap are asynchronous
    assert not pid_alive(proc.pid)


def test_read_exit_code_missing_or_garbled_is_minus_one(tmp_path) -> None:
    assert read_exit_code(tmp_path / "absent.rc") == -1
    garbled = tmp_path / "garbled.rc"
    garbled.write_text("not-a-number")
    assert read_exit_code(garbled) == -1
    ok = tmp_path / "ok.rc"
    ok.write_text("7\n")
    assert read_exit_code(ok) == 7


# --- window lifecycle reconciliation ----------------------------------------


def test_cleanup_kills_completed_run_window_once(tmp_path) -> None:
    fake = FakeTmux()
    fake.windows.add(RUN_ID)
    state = {"run_id": RUN_ID, "status": "done"}
    assert cleanup_run_window(fake, state) is True
    assert fake.killed == [RUN_ID]
    # Second reconcile pass: window already gone — rediscovered per-row, never
    # double-killed or re-created.
    assert cleanup_run_window(fake, state) is False
    assert fake.killed == [RUN_ID]
    assert fake.opened == []


def test_cleanup_preserves_failed_run_window(tmp_path) -> None:
    fake = FakeTmux()
    fake.windows.add(RUN_ID)
    assert cleanup_run_window(fake, {"run_id": RUN_ID, "status": "failed"}) is False
    assert RUN_ID in fake.windows


def test_cleanup_leaves_running_run_window_alone(tmp_path) -> None:
    fake = FakeTmux()
    fake.windows.add(RUN_ID)
    assert cleanup_run_window(fake, {"run_id": RUN_ID, "status": "running"}) is False
    assert RUN_ID in fake.windows


def test_cleanup_is_noop_without_tmux_binary(tmp_path) -> None:
    fake = FakeTmux(available=False)
    fake.windows.add(RUN_ID)
    assert cleanup_run_window(fake, {"run_id": RUN_ID, "status": "done"}) is False


def test_loop_poll_does_not_reconcile_done_run_window(tmp_path, monkeypatch) -> None:
    """Normal terminal polling reads durable state without touching tmux."""
    from milknado.mcp.ralph import milknado_run_loop_poll

    run_id = "node-7-20260101T000000Z-0000feed"
    graph, _cfg = open_graph(tmp_path)
    try:
        node = graph.add_node("loop-reconcile")
        graph.start_run(run_id, node.id, f"/tmp/{run_id}.log", "2026-01-01T00:00:00+00:00", 60)
        graph.finish_run(
            run_id,
            RunResult(
                status="done", exit_code=0, timed_out=False, ended_at="2026-01-01T00:01:00+00:00"
            ),
        )
    finally:
        graph.close()
    fake = FakeTmux()
    fake.windows.add(run_id)
    monkeypatch.setattr("milknado.app.ralph.TmuxAdapter", lambda root: fake)
    monkeypatch.setattr("milknado.mcp.ralph.TmuxAdapter", lambda root: fake)
    state = _call(milknado_run_loop_poll, run_id=run_id, project_root=str(tmp_path))
    assert state["status"] == "done"
    assert fake.killed == []


def test_reconcile_run_window_never_breaks_a_poll(caplog) -> None:
    class BrokenTmux:
        def available(self) -> bool:
            raise RuntimeError("tmux exploded")

    reconcile_run_window(BrokenTmux(), {"run_id": RUN_ID, "status": "done"})
    assert "tmux window reconcile failed" in caplog.text


# --- attach preconditions ----------------------------------------------------


def _seed_run(graph, run_id: str, status: str = "running") -> None:
    node = graph.add_node("attach-target")
    graph.start_run(run_id, node.id, f"/tmp/{run_id}.log", "2026-01-01T00:00:00+00:00", 60)
    if status != "running":
        graph.finish_run(
            run_id,
            RunResult(
                status=status,
                exit_code=0 if status == "done" else 1,
                timed_out=False,
                ended_at="2026-01-01T00:01:00+00:00",
            ),
        )


def test_attach_rejects_malformed_run_id(graph) -> None:
    with pytest.raises(ValueError, match="invalid run_id format"):
        resolve_attach_target(graph, FakeTmux(), "not-a-run-id")


def test_attach_rejects_unknown_run(graph) -> None:
    with pytest.raises(ValueError, match="not found"):
        resolve_attach_target(graph, FakeTmux(), RUN_ID)


def test_attach_rejects_finished_run(graph) -> None:
    _seed_run(graph, RUN_ID, status="failed")
    with pytest.raises(ValueError, match="already finished .status=failed."):
        resolve_attach_target(graph, FakeTmux(), RUN_ID)


def test_attach_rejects_when_tmux_missing(graph) -> None:
    _seed_run(graph, RUN_ID)
    with pytest.raises(ValueError, match="tmux is not installed"):
        resolve_attach_target(graph, FakeTmux(available=False), RUN_ID)


def test_attach_rejects_non_tmux_run(graph) -> None:
    _seed_run(graph, RUN_ID)
    with pytest.raises(ValueError, match="has no tmux window"):
        resolve_attach_target(graph, FakeTmux(), RUN_ID)


def test_attach_returns_exact_match_target_for_live_tmux_run(graph) -> None:
    _seed_run(graph, RUN_ID)
    fake = FakeTmux()
    fake.windows.add(RUN_ID)
    target = resolve_attach_target(graph, fake, RUN_ID)
    assert target == f"=milknado-proj:={RUN_ID}"


# --- milknado attach CLI -------------------------------------------------------


def _seed_cli_run(tmp_path, status: str = "running") -> None:
    graph, _cfg = open_graph(tmp_path)
    try:
        _seed_run(graph, RUN_ID, status=status)
    finally:
        graph.close()


def test_cli_attach_unknown_run_exits_with_clear_message(tmp_path) -> None:
    from typer.testing import CliRunner

    from milknado.cli import app

    result = CliRunner().invoke(app, ["attach", RUN_ID, "--project-root", str(tmp_path)])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_attach_finished_run_exits_with_clear_message(tmp_path) -> None:
    from typer.testing import CliRunner

    from milknado.cli import app

    _seed_cli_run(tmp_path, status="done")
    result = CliRunner().invoke(app, ["attach", RUN_ID, "--project-root", str(tmp_path)])
    assert result.exit_code == 1
    assert "already finished" in result.output


def test_cli_attach_execs_tmux_focused_on_the_run_window(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from milknado.cli import app

    _seed_cli_run(tmp_path)
    fake = FakeTmux()
    fake.windows.add(RUN_ID)
    monkeypatch.setattr("milknado.app.run.TmuxAdapter", lambda root: fake)
    monkeypatch.delenv("TMUX", raising=False)
    execs: list[list[str]] = []
    monkeypatch.setattr("os.execvp", lambda prog, argv: execs.append(argv))
    result = CliRunner().invoke(app, ["attach", RUN_ID, "--project-root", str(tmp_path)])
    assert result.exit_code == 0
    target = f"=milknado-proj:={RUN_ID}"
    assert execs == [
        ["tmux", "select-window", "-t", target, ";", "attach-session", "-t", "=milknado-proj"]
    ]


def test_cli_attach_inside_tmux_switches_client(monkeypatch) -> None:
    from milknado.cli.run import _tmux_attach_argv

    monkeypatch.setenv("TMUX", "/tmp/sock,1,0")
    argv = _tmux_attach_argv("=milknado-proj:=w")
    assert argv[-3:] == ["switch-client", "-t", "=milknado-proj"]
