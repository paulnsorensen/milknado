"""Exercise shutdown failures without hiding unresolved worker ownership."""

from __future__ import annotations

import asyncio
import signal
from queue import Queue
from threading import Event, Thread
from time import monotonic
from typing import cast

import pytest
from typing_extensions import override

import milknado.app._shutdown as shutdown
from milknado.app._shutdown import ShutdownIntent, ShutdownSignal, bounded_stop, supervise
from milknado.app.run import ExecutionController
from milknado.app.run_tui import ExecutionApp, run_execution_tui
from tests.test_execution_tui import FakeController


def test_bounded_stop_propagates_callback_failure() -> None:
    deadline = monotonic() + 5
    observed: list[float] = []

    def fail_stop(received_deadline: float) -> bool:
        observed.append(received_deadline)
        raise RuntimeError("stop failed")

    with pytest.raises(RuntimeError, match="stop failed"):
        _ = bounded_stop(fail_stop, deadline)
    assert observed == [deadline]


def test_supervise_transfers_system_exit_from_run_thread() -> None:
    failure = SystemExit(23)
    outcomes: Queue[BaseException] = Queue(maxsize=1)

    def fail_run() -> None:
        raise failure

    def invoke() -> None:
        try:
            _ = supervise(fail_run, ShutdownIntent(), lambda deadline: True, "exit-run")
        except BaseException as exc:
            outcomes.put(exc)

    caller = Thread(target=invoke, daemon=True)
    caller.start()
    caller.join(timeout=1)
    assert not caller.is_alive(), "supervisor waited for an unpublished outcome"
    assert outcomes.get_nowait() is failure


def test_supervise_preserves_signal_when_cleanup_raises() -> None:
    intent = ShutdownIntent()
    started = Event()
    release = Event()
    deadlines: list[float] = []

    def run() -> str:
        started.set()
        assert release.wait(timeout=5)
        return "finished"

    def fail_stop(deadline: float) -> bool:
        deadlines.append(deadline)
        raise RuntimeError("cleanup failed")

    def send_signal() -> None:
        assert started.wait(timeout=5)
        intent.record(signal.SIGTERM, None)

    signaler = Thread(target=send_signal)
    signaler.start()
    try:
        with pytest.raises(ShutdownSignal) as caught:
            _ = supervise(run, intent, fail_stop, "blocked-run")
    finally:
        release.set()
        signaler.join(timeout=5)
    assert not signaler.is_alive()
    assert caught.value.signum == signal.SIGTERM
    assert intent.cleanup_confirmed is False
    assert deadlines == [intent.deadline(8.0)]


def test_signal_after_queued_result_wins_without_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = ShutdownIntent()
    deadlines: list[float] = []

    class SignalOnResultQueue(Queue[object]):
        @override
        def get(self, block: bool = True, timeout: float | None = None) -> object:
            outcome = super().get(block, timeout)
            if outcome == "finished":
                intent.record(signal.SIGTERM, None)
                first_deadline = intent.deadline(8.0)
                intent.record(signal.SIGINT, None)
                assert intent.deadline(8.0) == first_deadline
            return outcome

    def stop(deadline: float) -> bool:
        deadlines.append(deadline)
        return True

    monkeypatch.setattr(shutdown, "Queue", SignalOnResultQueue)
    with pytest.raises(ShutdownSignal) as caught:
        _ = supervise(lambda: "finished", intent, stop, "completed-run")
    assert caught.value.signum == signal.SIGTERM
    assert intent.cleanup_confirmed is True
    assert deadlines == [intent.deadline(8.0)]


@pytest.mark.asyncio
async def test_tui_quit_reports_failed_cleanup_after_confirmation() -> None:
    class FailingController(FakeController):
        force_stop_all_requests: int

        @override
        def force_stop_all(self, timeout: float = 8.0) -> bool:
            del timeout
            self.force_stop_all_requests += 1
            raise RuntimeError("cleanup failed")

    controller = FailingController()
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press("q")
        assert app.screen.is_modal
        await pilot.press("y")
        async with asyncio.timeout(5):
            while app.cleanup_confirmed is None:
                await asyncio.sleep(0.02)
    assert controller.force_stop_all_requests == 1
    assert app.cleanup_confirmed is False


def test_tui_entry_keeps_signal_when_cleanup_raises(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingController(FakeController):
        force_stop_all_requests: int

        @override
        def force_stop_all(self, timeout: float = 8.0) -> bool:
            del timeout
            self.force_stop_all_requests += 1
            raise RuntimeError("cleanup failed")

    controller = FailingController()
    controller.shutdown_intent.record(signal.SIGTERM, None)
    with pytest.raises(ShutdownSignal) as caught:
        _ = run_execution_tui(
            cast(ExecutionController, cast(object, controller)),
            feature_branch="feature",
        )
    assert caught.value.signum == signal.SIGTERM
    assert controller.force_stop_all_requests == 1
    assert "worker ownership remains" in capsys.readouterr().err
