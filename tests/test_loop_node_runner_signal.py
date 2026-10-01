from __future__ import annotations

import signal
from threading import Event, Thread
from time import monotonic
from typing import cast

import pytest

from milknado.app._shutdown import ShutdownIntent, ShutdownSignal
from milknado.domains.execution import NodeLoopOutcome, RunLoop
from milknado.mcp._loop_node_runner import _supervise_node  # pyright: ignore[reportPrivateUsage]


def test_detached_runner_stops_while_node_thread_blocks() -> None:
    started = Event()
    release = Event()
    intent = ShutdownIntent()

    class Driver:
        deadlines: list[float] = []

        def force_stop_active(self, deadline: float) -> bool:
            self.deadlines.append(deadline)
            return True

        def confirm_preserved_stop(self, outcome: NodeLoopOutcome) -> NodeLoopOutcome:
            return outcome

    driver = Driver()

    def run_node() -> NodeLoopOutcome:
        started.set()
        _ = release.wait(2.0)
        return NodeLoopOutcome(1, True)

    signaler = Thread(target=lambda: (started.wait(), intent.record(signal.SIGTERM, None)))
    signaler.start()
    start = monotonic()
    try:
        with pytest.raises(ShutdownSignal) as caught:
            _ = _supervise_node(cast(RunLoop, cast(object, driver)), intent, run_node, 1)
    finally:
        release.set()
        signaler.join(1.0)

    assert caught.value.signum == signal.SIGTERM
    assert monotonic() - start < 1.0
    assert driver.deadlines == [intent.deadline(8.0)]


@pytest.mark.parametrize("fails", [False, True])
def test_detached_runner_supervises_preserved_stop_retries(fails: bool) -> None:
    retrying = Event()
    release = Event()
    intent = ShutdownIntent()

    class Driver:
        def __init__(self) -> None:
            self.deadlines: list[float] = []
            self.confirmed: list[bool] = []

        def force_stop_active(self, deadline: float) -> bool:
            self.deadlines.append(deadline)
            return True

        def confirm_preserved_stop(self, outcome: NodeLoopOutcome) -> NodeLoopOutcome:
            self.confirmed.append(outcome.ownership_preserved)
            retrying.set()
            _ = release.wait(2.0)
            return outcome

    driver = Driver()

    def run_node() -> NodeLoopOutcome:
        if fails:
            raise RuntimeError("run failed after dispatch")
        return NodeLoopOutcome(1, False, ownership_preserved=True)

    def send_signals() -> None:
        assert retrying.wait(1.0)
        intent.record(signal.SIGTERM, None)
        intent.record(signal.SIGINT, None)

    signaler = Thread(target=send_signals)
    signaler.start()
    try:
        with pytest.raises(ShutdownSignal) as caught:
            _ = _supervise_node(cast(RunLoop, cast(object, driver)), intent, run_node, 1)
    finally:
        release.set()
        signaler.join(1.0)

    assert driver.confirmed == [True]
    assert caught.value.signum == signal.SIGTERM
    assert driver.deadlines == [intent.deadline(8.0)]
