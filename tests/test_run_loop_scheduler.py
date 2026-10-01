from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from threading import Event, Thread
from types import SimpleNamespace
from typing import cast

import pytest

from milknado.domains.common import ProgressEvent
from milknado.domains.common.protocols import LoopPort
from milknado.domains.execution.executor import Executor
from milknado.domains.execution.run_loop import RunLoop
from milknado.domains.execution.run_loop._scheduler import Scheduler
from milknado.domains.execution.run_loop.state import TerminalRunState
from milknado.domains.graph import MikadoGraph
from milknado.loop import RunStatus


def test_scheduler_lifecycle_records_immutable_history_and_attempts() -> None:
    scheduler = Scheduler(eta_sample_size=2)
    scheduler.admit_run("run-1", 7, 10.0)
    before = scheduler.view()
    scheduler.record_progress(ProgressEvent(run_id="run-1", work=1, total=2, message="half"))
    terminal = TerminalRunState("run-1", 7, "build", RunStatus.FAILED, (), (), 0.0)
    finished = scheduler.finish_run("run-1", terminal, 14.0)
    scheduler.record_completion(finished.duration)
    scheduler.record_failure(finished.node_id, strict=True)
    scheduler.count_outcomes(failed=1)

    after = scheduler.view()
    assert before.active[0].progress is None
    assert finished.node_id == 7 and finished.duration == 4.0
    assert after.active == ()
    assert after.terminal_runs[0].duration_seconds == 4.0
    assert after.completion_durations == (4.0,)
    assert after.failed == 1 and after.failure_triggered
    scheduler.admit_run("run-2", 7, 20.0)
    assert scheduler.view().active[0].prior_attempts == 1
    with pytest.raises(FrozenInstanceError):
        after.__setattr__("failed", 2)


def test_scheduler_deferral_stop_and_reset_sequence() -> None:
    scheduler = Scheduler(eta_sample_size=2)
    scheduler.defer_capacity()
    assert scheduler.retry_deferred(10.0, 1.0)
    assert not scheduler.retry_deferred(10.5, 1.0)
    assert scheduler.retry_deferred(11.0, 1.0)
    _ = scheduler.plan_dispatch(2, strict=False)
    scheduler.admit_run("run-1", 3, 12.0)
    scheduler.record_stop(3)
    scheduler.close_admission()

    stopped = scheduler.view()
    assert not stopped.capacity_deferred
    assert stopped.scheduling_stopped
    assert stopped.stopped_nodes == {3}
    assert stopped.stopped == 1
    assert scheduler.abandon_run("run-1") == 3
    assert scheduler.view().active == ()
    scheduler.reset_run()
    assert scheduler.view().stopped_nodes == frozenset()
    assert scheduler.view().stopped == 0
    assert scheduler.view().scheduling_stopped


@pytest.mark.parametrize(
    ("active_count", "strict_failure", "expected_available"),
    [(0, False, 2), (2, False, 0), (0, True, 0)],
)
def test_dispatch_plan_applies_capacity_and_strict_admission(
    active_count: int, strict_failure: bool, expected_available: int
) -> None:
    scheduler = Scheduler(eta_sample_size=2)
    for node_id in range(active_count):
        scheduler.admit_run(f"run-{node_id}", node_id, 0.0)
    scheduler.record_stop(9)
    if strict_failure:
        scheduler.trigger_failure()
    scheduler.defer_capacity()

    plan = scheduler.plan_dispatch(2, strict=True)

    assert plan.available == expected_available
    assert plan.stopped_nodes == {9}
    assert not scheduler.view().capacity_deferred


def test_dispatch_failure_decides_strict_batch_stop() -> None:
    scheduler = Scheduler(eta_sample_size=2)
    assert not scheduler.dispatch_failed(strict=False).stop_batch
    assert not scheduler.view().failure_triggered
    assert scheduler.dispatch_failed(strict=True).stop_batch
    assert scheduler.view().failure_triggered


def test_stop_admission_prevents_new_dispatch_plan() -> None:
    scheduler = Scheduler(eta_sample_size=2)
    scheduler.close_admission()
    assert scheduler.plan_dispatch(2, strict=False).available == 0


def test_force_stop_snapshot_survives_concurrent_abandonment() -> None:
    entered = Event()
    finished = Event()

    class PausingActive:
        def __init__(self, data: dict[str, int]) -> None:
            self.data: dict[str, int] = data

        def items(self) -> Iterator[tuple[str, int]]:
            iterator = iter(self.data.items())
            first = next(iterator)
            entered.set()
            _ = finished.wait(1.0)
            yield first
            yield from iterator

        def pop(self, run_id: str, default: None = None) -> int | None:
            return self.data.pop(run_id, default)

    class StopExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def force_stop_run(self, run_id: str, timeout: float) -> bool:
            _ = timeout
            self.calls.append(run_id)
            return True

    class StopLoop:
        def stop_active_workers(self, deadline: float) -> bool:
            _ = deadline
            return True

    executor = StopExecutor()
    worker_loop = StopLoop()
    run_loop = RunLoop(
        cast(Executor, cast(object, executor)),
        cast(MikadoGraph, object()),
        cast(LoopPort, cast(object, worker_loop)),
    )
    scheduler = run_loop._scheduler  # pyright: ignore[reportPrivateUsage]
    scheduler.admit_run("run-1", 1, 0.0)
    scheduler.admit_run("run-2", 2, 0.0)
    scheduler._active = cast(  # pyright: ignore[reportPrivateUsage]
        dict[str, int],
        cast(object, PausingActive(scheduler._active)),  # pyright: ignore[reportPrivateUsage]
    )
    results: list[bool] = []
    errors: list[Exception] = []

    def stop() -> None:
        try:
            results.append(run_loop.force_stop_active(10.0))
        except Exception as exc:
            errors.append(exc)

    def abandon() -> None:
        _ = scheduler.abandon_run("run-1")
        finished.set()

    stop_thread = Thread(target=stop)
    stop_thread.start()
    assert entered.wait(2.0)
    finish_thread = Thread(target=abandon)
    finish_thread.start()
    stop_thread.join(3.0)
    finish_thread.join(3.0)

    assert not stop_thread.is_alive() and not finish_thread.is_alive()
    assert not errors
    assert results == [True]
    assert executor.calls == ["run-1", "run-2"]


def test_force_stop_admission_does_not_wait_for_scheduling_lock() -> None:
    def stop_workers(_deadline: float) -> bool:
        return True

    run_loop = RunLoop(
        cast(Executor, object()),
        cast(MikadoGraph, object()),
        cast(LoopPort, cast(object, SimpleNamespace(stop_active_workers=stop_workers))),
    )
    result: list[bool] = []
    worker = Thread(target=lambda: result.append(run_loop.force_stop_active(10.0)))
    lock = run_loop._scheduling_lock  # pyright: ignore[reportPrivateUsage]
    assert lock.acquire()
    try:
        worker.start()
        worker.join(1.0)
        admitted = not worker.is_alive()
    finally:
        lock.release()
        worker.join(2.0)

    assert admitted
    assert result == [True]
