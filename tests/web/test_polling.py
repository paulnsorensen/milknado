from __future__ import annotations

from threading import Event
from time import monotonic

from milknado.app.run_source import ExecutionSnapshot
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


def test_polling_caches_and_publishes_one_snapshot_per_tick() -> None:
    source = Source()
    polled = PolledSnapshotSource(source, interval=0.01)
    events: list[str] = []
    ready = Event()
    unsubscribe = polled.subscribe(lambda snapshot: (events.append(snapshot.goal), ready.set()))
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
