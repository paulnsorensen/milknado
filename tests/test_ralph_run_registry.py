"""Curd A (#296): ralph-dispatched runs are first-class in the runs registry."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import milknado.domains.dispatch.cancel as cancel_module
import milknado.domains.execution.executor as executor_module
from milknado.domains.common import Gate
from milknado.domains.dispatch._runstate import request_cancel, runs_dir
from milknado.domains.dispatch.cancel import cancel_run
from milknado.domains.execution import ExecutionConfig, Executor
from milknado.mcp._core import open_graph
from milknado.mcp.ralph import milknado_run_loop_poll
from milknado.mcp.run import milknado_run_cancel, milknado_run_list
from tests.test_execution import FakeCrg, FakeGit, FakeRalph


def _config(root: Path, completion_timeout_seconds: int | None = None) -> ExecutionConfig:
    return ExecutionConfig(
        execution_agent="claude",
        quality_gates=(Gate("true"),),
        worktree_pattern="milknado-{node_id}-{slug}",
        project_root=root,
        completion_timeout_seconds=completion_timeout_seconds,
    )


def _executor(graph: Any) -> Executor:
    return Executor(graph=graph, git=FakeGit(), ralph=FakeRalph(), crg=FakeCrg())


def test_dispatch_inserts_runs_row(graph: Any, tmp_path: Path) -> None:
    graph.add_node("registered ralph run")
    result = _executor(graph).dispatch(1, _config(tmp_path))

    row = graph.get_run(result.run_id)
    assert row is not None, "ralph dispatch must insert a runs row"
    assert row["node_id"] == 1
    assert row["status"] == "running"
    assert row["log_path"].endswith(".ralph-logs")


def test_verdict_message_deposit_no_longer_fk_fails(graph: Any, tmp_path: Path) -> None:
    """The #296 regression: run_messages REFERENCES runs(run_id); without the
    dispatch-time row this insert raised a FOREIGN KEY error and the verdict
    was lost."""
    graph.add_node("verdict target")
    result = _executor(graph).dispatch(1, _config(tmp_path))

    seq = graph.deposit_run_message(
        result.run_id, "node_review", '{"verdict":"reject"}', "2026-07-23T00:00:00Z"
    )
    assert seq == 1
    assert graph.latest_run_message(result.run_id, "node_review") == '{"verdict":"reject"}'


def test_run_list_includes_ralph_runs(graph: Any, tmp_path: Path) -> None:
    graph.add_node("listed ralph run")
    result = _executor(graph).dispatch(1, _config(tmp_path))

    listed = graph.recent_runs(50)
    assert result.run_id in {r["run_id"] for r in listed}


def test_cancel_marks_ralph_run_row_cancelled(graph: Any, tmp_path: Path) -> None:
    """A ralph run with a recorded pid cancels through the pid path; its runs
    row is finalized with the cancelled marker."""
    graph.add_node("cancellable ralph run")
    result = _executor(graph).dispatch(1, _config(tmp_path))
    graph.set_run_pid(result.run_id, 424242)

    class _Process:
        def terminate_group(self, pid: int, timeout: float) -> bool:  # noqa: ARG002
            return True

    final = cancel_run(
        graph,
        MagicMock(),
        _Process(),
        tmp_path,
        result.run_id,
    )
    assert final["status"] == "failed"
    assert final["error"] == "cancelled"
    row = graph.get_run(result.run_id)
    assert row is not None and row["error"] == "cancelled"


def test_cancel_pidless_ralph_run_via_sentinel(graph: Any, tmp_path: Path) -> None:
    """The real pid-less path: no artificial set_run_pid. cancel_run takes the
    async-sentinel path; the executor's cancel watcher observes the sentinel,
    force-stops the loop, and finalizes the row cancelled inside the confirm
    bound — live cancel must not stall and raise."""
    graph.add_node("cancellable pid-less ralph run")
    # live=True: the run's thread is alive from creation, so the watcher
    # stays on its poll loop and observes the sentinel (a dead thread would
    # exit it on the dead-thread branch first).
    ralph = FakeRalph(live=True)
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())
    result = executor.dispatch(1, _config(tmp_path))
    assert graph.get_run(result.run_id)["pid"] is None

    final = cancel_run(
        graph,
        MagicMock(),
        MagicMock(),
        tmp_path,
        result.run_id,
    )
    assert final["status"] == "failed"
    assert final["error"] == "cancelled"
    row = graph.get_run(result.run_id)
    assert row is not None and row["error"] == "cancelled"


def test_watcher_exits_when_run_thread_is_dead(graph: Any, tmp_path: Path) -> None:
    """Dead-thread exit branch: a run whose thread has finished makes the
    cancel watcher return on its own — the completion path owns the finalize,
    and no polling daemon thread leaks past the dispatch."""
    graph.add_node("finished ralph run")
    ralph = FakeRalph(id_prefix="watchdead")
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())
    result = executor.dispatch(1, _config(tmp_path))
    assert not ralph.runs[result.run_id].thread.is_alive()

    watcher = next(
        (t for t in threading.enumerate() if t.name == f"ralph-cancel-watch-{result.run_id}"),
        None,
    )
    if watcher is not None:  # None: it already exited — the same proof
        watcher.join(timeout=5)
        assert not watcher.is_alive(), "watcher must exit once the run thread is dead"
    row = graph.get_run(result.run_id)
    assert row is not None and row["status"] == "running"


def test_cancel_force_stop_failure_fails_closed_then_recovers(
    graph: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force-stop-failure branch: when the watcher cannot confirm the loop
    stopped, it must NOT write a falsely-terminal row — the row stays
    'running', so cancel_run's confirm bound expires and raises the
    preserving error instead of tearing a worktree out from under a live
    worker. The watcher is NOT one-shot: the sentinel stays set, it keeps
    observing, and once the transient stop failure clears it retries the
    force-stop and finalizes the row cancelled — a retried cancel then
    succeeds."""
    graph.add_node("wedged ralph run")
    ralph = FakeRalph(live=True, id_prefix="wedged")
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())
    result = executor.dispatch(1, _config(tmp_path))
    ralph.force_stop_result = False
    monkeypatch.setattr(cancel_module, "_CANCEL_FINALIZE_TIMEOUT_SECS", 0.5)

    with pytest.raises(RuntimeError, match="has not confirmed worker exit"):
        cancel_run(graph, MagicMock(), MagicMock(), tmp_path, result.run_id)

    row = graph.get_run(result.run_id)
    assert row is not None and row["status"] == "running", "no falsely-terminal row"
    assert set(ralph.force_stopped) == {result.run_id}

    # The transient failure clears: the still-observing watcher retries the
    # force-stop, kills the loop, and finalizes the row cancelled.
    ralph.force_stop_result = True
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = graph.get_run(result.run_id)
        if row is not None and row["status"] != "running":
            break
        time.sleep(0.05)
    row = graph.get_run(result.run_id)
    assert row is not None and row["error"] == "cancelled", (
        "watcher must retry the force-stop and finalize cancelled"
    )

    # A retried cancel no longer strands: the terminal row early-returns.
    final = cancel_run(graph, MagicMock(), MagicMock(), tmp_path, result.run_id)
    assert final["error"] == "cancelled"

    watcher = next(
        (t for t in threading.enumerate() if t.name == f"ralph-cancel-watch-{result.run_id}"),
        None,
    )
    if watcher is not None:  # None: it already exited — the same proof
        watcher.join(timeout=5)
        assert not watcher.is_alive(), "watcher must exit after the confirmed stop"


def test_watcher_retry_backoff_doubles_and_caps(
    graph: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A permanently wedged loop must not log at full poll rate forever:
    consecutive force-stop failures back off exponentially (doubling from
    the poll cadence) up to a documented ceiling — while the
    retry-until-stopped semantics are preserved: once the stop can confirm,
    the watcher still finalizes the row cancelled."""
    graph.add_node("wedged retry backoff")
    ralph = FakeRalph(live=True, id_prefix="backoff")
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())
    result = executor.dispatch(1, _config(tmp_path))
    ralph.force_stop_result = False

    real_sleep = time.sleep
    sleeps: list[float] = []

    def _recording_sleep(secs: float) -> None:
        sleeps.append(secs)
        real_sleep(0.001)  # record the requested cadence; never actually back off

    monkeypatch.setattr(executor_module.time, "sleep", _recording_sleep)
    request_cancel(runs_dir(tmp_path), result.run_id)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and len(ralph.force_stopped) < 10:
        real_sleep(0.01)
    assert len(ralph.force_stopped) >= 10, "watcher must keep retrying the stop"

    # The transient failure clears: retry-until-stopped is intact — the
    # watcher stops the loop and finalizes cancelled.
    ralph.force_stop_result = True
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = graph.get_run(result.run_id)
        if row is not None and row["status"] != "running":
            break
        real_sleep(0.05)
    row = graph.get_run(result.run_id)
    assert row is not None and row["error"] == "cancelled"

    # Poll-cadence sleeps are exactly _RALPH_CANCEL_POLL_SECS; only backoff
    # sleeps exceed it. They must double from the poll cadence up to the
    # ceiling and then hold there.
    poll = executor_module._RALPH_CANCEL_POLL_SECS
    cap = executor_module._RALPH_CANCEL_RETRY_BACKOFF_MAX_SECS
    backoff_sleeps = [s for s in sleeps if s > poll]
    assert backoff_sleeps[:7] == [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    assert backoff_sleeps[7:], "backoff must reach the ceiling"
    assert all(s == cap for s in backoff_sleeps[7:])


def test_completed_ralph_run_row_is_finalized(graph: Any, tmp_path: Path) -> None:
    """Executor ralph rows must not zombie as 'running': completing the node
    finalizes the runs row done, so run_list stays truthful and cancel of a
    finished run early-returns instead of stalling."""
    graph.add_node("finalized ralph run")
    executor = _executor(graph)
    result = executor.dispatch(1, _config(tmp_path))

    executor.complete(1, "main")

    row = graph.get_run(result.run_id)
    assert row is not None
    assert row["status"] == "done"
    assert row["ended_at"] is not None
    # Cancel of the terminal row is a no-op returning the stored state.
    final = cancel_run(graph, MagicMock(), MagicMock(), tmp_path, result.run_id)
    assert final["status"] == "done"


def test_dispatch_registers_row_before_starting_loop_with_recovery_metadata(
    graph: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H1 (corrected): persistence must precede execution, and the row must
    carry the real completion timeout with NO pid, mirroring the async-worker
    path — so cancel always routes through the sentinel path, never the
    process-kill path (an executor ralph loop runs in-process; its pid is the
    server's own pid)."""
    call_order: list[str] = []
    orig_graph_start_run = graph.start_run

    def spy_graph_start_run(*a: Any, **kw: Any) -> None:
        call_order.append("graph")
        return orig_graph_start_run(*a, **kw)

    monkeypatch.setattr(graph, "start_run", spy_graph_start_run)

    ralph = FakeRalph()
    orig_ralph_start_run = ralph.start_run

    def spy_ralph_start_run(*a: Any, **kw: Any) -> None:
        call_order.append("ralph")
        return orig_ralph_start_run(*a, **kw)

    monkeypatch.setattr(ralph, "start_run", spy_ralph_start_run)

    graph.add_node("ordered dispatch")
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())
    config = _config(tmp_path, completion_timeout_seconds=900)
    result = executor.dispatch(1, config)

    assert call_order == ["graph", "ralph"], "the DB insert must land before the loop starts"
    row = graph.get_run(result.run_id)
    assert row is not None
    assert row["timeout_seconds"] == 900
    assert row["pid"] is None


def test_runs_row_insert_failure_fails_closed_and_never_starts_loop(
    graph: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H1: an unregistered run is worse than a failed dispatch. A failed
    registry write must propagate (not be swallowed) and the ralph loop must
    never start on a failed registration."""
    graph.add_node("fail-closed dispatch")
    ralph = FakeRalph()
    monkeypatch.setattr(graph, "start_run", MagicMock(side_effect=RuntimeError("db down")))

    with pytest.raises(RuntimeError, match="db down"):
        Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg()).dispatch(
            1, _config(tmp_path)
        )

    assert ralph.runs_started == [], "the loop must not start when registration fails"


def test_stale_sweep_recovers_pidless_executor_row_after_timeout_elapses(
    graph: Any, tmp_path: Path
) -> None:
    """H1 (corrected): executor rows record NO pid, so fail_stale_running_runs
    has nothing to check for liveness — it sweeps the row purely on
    started_at + timeout_seconds + _STALE_GRACE_SECONDS, the same rule the
    async-worker path already relies on. While the server is alive, the
    headless completion-timeout force-stop drives a live run to terminal
    before this window elapses, so the sweep never fires on a real run."""
    from datetime import UTC, datetime, timedelta

    from milknado.domains.dispatch import fail_stale_running_runs

    graph.add_node("pidless dispatch")

    run_id = "node-1-20200101T000000Z-pidless"
    timeout = 300
    old_started = (datetime.now(UTC) - timedelta(seconds=timeout + 3600)).isoformat()

    graph.start_run(run_id, 1, str(tmp_path / "pidless.log"), old_started, timeout)
    assert graph.get_run(run_id)["pid"] is None

    flipped = fail_stale_running_runs(graph, 1)
    assert [f["run_id"] for f in flipped] == [run_id]
    assert graph.get_run(run_id)["status"] == "failed"


def test_cancel_pidless_stale_row_routes_through_sentinel_not_pid_kill(
    graph: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard for H1: an executor row with pid=None must route
    cancel through _cancel_async_run, never _cancel_pid_run — the latter
    would terminate the recorded pid's process group, which for an
    in-process executor ralph loop is the server's own pid."""
    monkeypatch.setattr(cancel_module, "_CANCEL_FINALIZE_TIMEOUT_SECS", 0.2)
    monkeypatch.setattr(cancel_module, "_CANCEL_FINALIZE_POLL_SECS", 0.05)

    def _fail_if_called(*a: Any, **kw: Any) -> None:
        raise AssertionError("_cancel_pid_run must not be called for a pid-less row")

    monkeypatch.setattr(cancel_module, "_cancel_pid_run", _fail_if_called)

    graph.add_node("pidless cancel routing")
    run_id = "node-1-20200101T000000Z-routing"
    graph.start_run(run_id, 1, str(tmp_path / "routing.log"), "2020-01-01T00:00:00Z", 300)
    assert graph.get_run(run_id)["pid"] is None

    with pytest.raises(RuntimeError, match="has not confirmed worker exit"):
        cancel_run(graph, MagicMock(), MagicMock(), tmp_path, run_id)


def test_mcp_functions_accept_executor_run_id(tmp_path: Path) -> None:
    """Blocker 1B hardening: an executor-dispatched node's run_id must pass
    the MCP tools' RUN_ID_RE gate (`node-<id>-<ts>-<hex>` shape), not just the
    domain functions the earlier tests exercise directly.

    Uses the same on-disk db path the MCP tools open (via `open_graph`)
    instead of the `graph` fixture's separate `test.db`, so the dispatch and
    the MCP calls below see the same runs table. `FakeRalph(live=True)` gives
    the run a real thread so the cancel watcher stays on its poll loop long
    enough for `milknado_run_cancel`'s sentinel-confirm wait to observe it —
    the graph must stay open for the watcher thread until after that cancel.
    """
    root = str(tmp_path)
    mcp_graph, _cfg = open_graph(tmp_path)
    try:
        mcp_graph.add_node("mcp-format run_id")
        executor = Executor(
            graph=mcp_graph, git=FakeGit(), ralph=FakeRalph(live=True), crg=FakeCrg()
        )
        result = executor.dispatch(1, _config(tmp_path))

        listed = milknado_run_list(project_root=root)
        assert result.run_id in {r["run_id"] for r in listed}

        polled = milknado_run_loop_poll(run_id=result.run_id, project_root=root)
        assert polled["run_id"] == result.run_id

        cancelled = milknado_run_cancel(run_id=result.run_id, project_root=root)
        assert cancelled["run_id"] == result.run_id
    finally:
        mcp_graph.close()


def test_cancel_stale_adopted_round_run_id_is_a_noop(tmp_path: Path) -> None:
    """Blocker 1B hardening: for an adopted (review-owned) execution, the
    node's run_id fence stays pinned to the parent/owner fence
    (`_owner_fence_by_node`) while each round registers its own distinct
    run_id in the runs table. Cancelling a round's own (still-running) run_id
    through the real MCP tool must not raise and must not touch the node's
    worktree/status it doesn't solely own — `_reconcile_cancel`'s
    `node.run_id != run_id` guard must fire and skip reconciliation."""
    from milknado.domains.dispatch._runstate import now_iso

    root = str(tmp_path)
    mcp_graph, _cfg = open_graph(tmp_path)
    try:
        mcp_graph.add_node("adopted round cancel")
        parent_fence = "parent-fence-1"
        assert mcp_graph.claim_node(1, parent_fence, now=now_iso())
        executor = Executor(
            graph=mcp_graph, git=FakeGit(), ralph=FakeRalph(live=True), crg=FakeCrg()
        )

        result = executor.dispatch(1, _config(tmp_path), parent_run_id=parent_fence)
        before_node = mcp_graph.get_node(1)
        assert before_node is not None
        assert before_node.run_id == parent_fence, "adopted dispatch keeps the parent fence"
        assert result.run_id != parent_fence, "the round gets its own distinct run_id"

        cancelled = milknado_run_cancel(run_id=result.run_id, project_root=root)
        assert cancelled["run_id"] == result.run_id

        after_node = mcp_graph.get_node(1)
        assert after_node is not None
        assert after_node.run_id == parent_fence, (
            "stale-round cancel must not touch the node's owner fence"
        )
        assert after_node.status.value == "running", (
            "stale-round cancel must not flip the node's status"
        )
        assert after_node.worktree_path is not None, (
            "stale-round cancel must not tear down a worktree it doesn't own"
        )
    finally:
        mcp_graph.close()
