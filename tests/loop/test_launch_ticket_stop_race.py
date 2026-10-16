"""A stop that lands around gate activation never lets the worker exec unobserved."""

from __future__ import annotations

import threading
import time

from milknado.loop._process_registry import WorkerRegistry


def test_stop_before_activate_never_releases_the_gate() -> None:
    registry = WorkerRegistry()
    ticket = registry.reserve()
    released: list[bool] = []

    assert not ticket.stop(time.monotonic() + 1)

    assert not ticket.activate(lambda: released.append(True), lambda _: True)
    assert released == [], "a stop requested before release must not let the worker exec"
    ticket.close()


def test_stop_before_bind_pending_closes_the_gate_and_refuses() -> None:
    registry = WorkerRegistry()
    ticket = registry.reserve()
    closed: list[bool] = []

    assert not ticket.stop(time.monotonic() + 1)

    assert not ticket.bind_pending(lambda: closed.append(True))
    assert closed == [True]
    ticket.close()


def test_stop_waits_for_in_flight_activation_then_shuts_the_worker_down() -> None:
    registry = WorkerRegistry()
    ticket = registry.reserve()
    release_entered = threading.Event()
    finish_release = threading.Event()
    shutdown_deadlines: list[float] = []
    stop_results: list[bool] = []
    activated: list[bool] = []

    def slow_release() -> None:
        release_entered.set()
        _ = finish_release.wait(timeout=5)

    def shutdown(deadline: float) -> bool:
        shutdown_deadlines.append(deadline)
        return True

    deadline = time.monotonic() + 5
    activator = threading.Thread(
        target=lambda: activated.append(ticket.activate(slow_release, shutdown))
    )
    stopper = threading.Thread(target=lambda: stop_results.append(ticket.stop(deadline)))
    activator.start()
    assert release_entered.wait(timeout=2)
    stopper.start()
    stopper.join(timeout=0.3)
    assert stopper.is_alive(), "stop must wait for the in-flight activation"
    assert shutdown_deadlines == []
    assert registry.lock.acquire(timeout=1), "the gate wait must not hold the registry lock"
    registry.lock.release()

    finish_release.set()
    activator.join(timeout=5)
    stopper.join(timeout=5)

    assert activated == [False]
    assert shutdown_deadlines == [deadline]
    assert stop_results == [True]
    ticket.close()
