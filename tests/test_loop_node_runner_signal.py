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
            _ = _supervise_node(cast(RunLoop, cast(object, driver)), intent, run_node)
    finally:
        release.set()
        signaler.join(1.0)

    assert caught.value.signum == signal.SIGTERM
    assert monotonic() - start < 1.0
    assert driver.deadlines == [intent.deadline(8.0)]
