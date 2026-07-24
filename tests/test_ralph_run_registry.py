"""Curd A (#296): ralph-dispatched runs are first-class in the runs registry."""

from __future__ import annotations

import logging
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
from tests.test_execution import FakeCrg, FakeGit, FakeRalph


def _config(root: Path) -> ExecutionConfig:
    return ExecutionConfig(
        execution_agent="claude",
        quality_gates=(Gate("true"),),
        worktree_pattern="milknado-{node_id}-{slug}",
        project_root=root,
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


def test_runs_row_insert_failure_does_not_kill_dispatch(
    graph: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A DB outage at dispatch must not strand the worker: the runs-row insert
    is best-effort (findings thread in memory, never through the DB), so a
    failed insert logs loud and the dispatch still returns its run_id."""
    graph.add_node("resilient dispatch")
    monkeypatch.setattr(
        graph, "start_run", MagicMock(side_effect=RuntimeError("db down"))
    )

    with caplog.at_level(logging.ERROR):
        result = _executor(graph).dispatch(1, _config(tmp_path))

    assert result.run_id, "dispatch must succeed despite the failed registry write"
    assert any("runs-row insert failed" in r.message for r in caplog.records), (
        "the failed registry write must log loud, not vanish"
    )
