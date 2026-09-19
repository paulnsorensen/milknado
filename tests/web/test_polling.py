# pyright: basic

from __future__ import annotations

from threading import Event
from time import monotonic

import pytest

from milknado.app.run_source import ExecutionSnapshot, NodeSnapshotRequest
from milknado.web.polling import PolledSnapshotSource


class Source:
    def __init__(self) -> None:
        self.count = 0
        self.closed = False

    def snapshot(self) -> ExecutionSnapshot:
        self.count += 1
        return ExecutionSnapshot(
            goal=str(self.count),
            active_runs=(),
            terminal_runs=(),
            completed=0,
            failed=0,
            stopped=0,
            available=0,
            event_lines=(),
        )

    def subscribe(self, listener):
        del listener
        return lambda: None

    def node_snapshot(self, request):
        del request
        raise NotImplementedError

    def close(self) -> None:
        self.closed = True


class FailingSource(Source):
    def snapshot(self) -> ExecutionSnapshot:
        if self.count > 0:
            raise RuntimeError("poll failed")
        return super().snapshot()


def test_polling_reports_poll_and_listener_failures_without_stopping() -> None:
    source = FailingSource()
    polled = PolledSnapshotSource(source, interval=0.01)
    events: list[ExecutionSnapshot] = []

    def failing_listener(snapshot: ExecutionSnapshot) -> None:
        raise RuntimeError("listener failed")

    polled.subscribe(events.append)
    polled.subscribe(failing_listener)
    polled.start()
    deadline = monotonic() + 1
    while monotonic() < deadline and not polled.snapshot().listener_errors:
        ready = Event()
        ready.wait(0.02)
    snapshot = polled.snapshot()
    polled.close()
    assert snapshot.listener_errors
    assert events


def test_polling_start_is_idempotent_and_snapshot_starts_lazily() -> None:
    polled = PolledSnapshotSource(Source(), interval=0.01)
    assert polled.snapshot().goal == "1"
    polled.start()
    polled.start()
    polled.subscribe(lambda snapshot: None)
    polled.close()


def test_polling_caches_and_publishes_one_snapshot_per_tick() -> None:
    source = Source()
    polled = PolledSnapshotSource(source, interval=0.01)
    events: list[str] = []
    ready = Event()

    def record(snapshot: ExecutionSnapshot) -> None:
        events.append(snapshot.goal)
        ready.set()

    unsubscribe = polled.subscribe(record)
    polled.start()
    assert ready.wait(1)
    deadline = monotonic() + 1
    while source.count < 2 and monotonic() < deadline:
        ready.wait(0.02)
    assert polled.snapshot().goal == str(source.count)
    assert len(events) >= 1
    unsubscribe()
    polled.close()
    assert source.closed


def test_polling_delegates_node_snapshot() -> None:
    polled = PolledSnapshotSource(Source())
    with pytest.raises(NotImplementedError):
        _ = polled.node_snapshot(NodeSnapshotRequest(node_id=1, request_generation=1))
