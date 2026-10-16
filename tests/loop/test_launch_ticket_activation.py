from __future__ import annotations

import threading
import time

from milknado.loop._process_registry import WorkerRegistry


def test_stop_all_returns_promptly_while_gate_release_is_slow() -> None:
    registry = WorkerRegistry()
    ticket = registry.reserve()
    release_entered = threading.Event()
    finish_release = threading.Event()
    stop_returned = threading.Event()
    activated: list[bool] = []

    def slow_release() -> None:
        release_entered.set()
        _ = finish_release.wait(timeout=5)

    activator = threading.Thread(
        target=lambda: activated.append(ticket.activate(slow_release, lambda _: True))
    )
    stopper = threading.Thread(
        target=lambda: (registry.stop_all(time.monotonic() + 0.5), stop_returned.set())
    )
    activator.start()
    try:
        assert release_entered.wait(timeout=2)
        stopper.start()
        assert stop_returned.wait(timeout=1), "stop_all blocked behind the gate release"
    finally:
        finish_release.set()
        activator.join(timeout=5)
        if stopper.is_alive():
            stopper.join(timeout=5)
    assert activated == [False], "a stop requested during release must abort the start"
    ticket.close()


def test_activate_reports_success_when_no_stop_arrives() -> None:
    registry = WorkerRegistry()
    ticket = registry.reserve()
    assert ticket.activate(lambda: None, lambda _: True)
    ticket.close()


def test_per_run_stop_during_gate_release_aborts_the_start() -> None:
    registry = WorkerRegistry()
    ticket = registry.reserve("run-1")
    stop_results: list[bool] = []

    def release_with_concurrent_stop() -> None:
        stop_results.append(registry.stop_run_workers("run-1", time.monotonic() + 1))

    assert not ticket.activate(release_with_concurrent_stop, lambda _: True)
    assert stop_results == [False], "the stop found no live worker yet"
    ticket.close()
