from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Thread, get_ident
from time import monotonic, sleep
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionController,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
    build_execution_controller,
)
from milknado.domains.common import MilknadoConfig
from milknado.domains.execution import ExecutionConfig
from milknado.domains.execution.run_loop.state import (
    ActiveRunState,
    RunActionState,
    RunLoopState,
)
from milknado.loop import RunStatus


@dataclass
class FakeLoop:
    current_state: RunLoopState

    def __post_init__(self) -> None:
        self.listener: object | None = None
        self.run_calls: list[dict[str, object]] = []
        self.guidance: list[tuple[str, str]] = []
        self.cancelled: list[str] = []
        self.force_stops: list[tuple[str, float]] = []
        self.stop_scheduling_calls = 0

    def state(self) -> RunLoopState:
        return self.current_state

    def stop_scheduling(self) -> None:
        self.stop_scheduling_calls += 1

    def admit_stop_scheduling(self) -> None:
        pass

    def set_state_listener(self, listener: object) -> None:
        self.listener = listener

    def run(self, **kwargs: object) -> str:
        self.run_calls.append(kwargs)
        return "result"

    def publish(self, state: RunLoopState) -> None:
        self.current_state = state
        assert callable(self.listener)
        self.listener(state)

    def queue_guidance(self, run_id: str, text: str) -> bool:
        self.guidance.append((run_id, text))
        return text == "accepted"

    def cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)

    def force_stop(self, run_id: str, timeout: float) -> bool:
        self.force_stops.append((run_id, timeout))
        return True


def loop_state(*, output: tuple[str, ...] = ("last line",)) -> RunLoopState:
    return RunLoopState(
        goal="Ship controller",
        active_runs=(
            ActiveRunState(
                run_id="run-1",
                node_id=7,
                description="Build snapshots",
                status=RunStatus.RUNNING,
                progress="1/2",
                stop_requested=False,
                actions=RunActionState(cancel_reason="already stopping"),
                output=output,
                pending_guidance=("use domain barrels",),
            ),
        ),
        terminal_runs=(),
        completed=3,
        failed=1,
        stopped=0,
        available=2,
        event_lines=("dispatched run-1",),
    )


def snapshot(*, output: tuple[str, ...] = ("last line",)) -> ExecutionSnapshot:
    return ExecutionSnapshot(
        goal="Ship controller",
        active_runs=(
            ActiveRunSnapshot(
                run_id="run-1",
                node_id=7,
                description="Build snapshots",
                status=ExecutionRunStatus.RUNNING,
                progress="1/2",
                stop_requested=False,
                actions=RunActionAvailability(cancel_reason="already stopping"),
                output=output,
                pending_guidance=("use domain barrels",),
            ),
        ),
        terminal_runs=(),
        completed=3,
        failed=1,
        stopped=0,
        available=2,
        event_lines=("dispatched run-1",),
    )


def test_controller_delegates_run_and_control_ports() -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(loop, cast(ExecutionConfig, None), cast(int, None))

    assert controller.run(feature_branch="feature", strict=True, spec_text="spec") == "result"
    assert len(loop.run_calls) == 1
    run_call = loop.run_calls[0]
    process_controls = run_call.pop("process_controls")
    assert callable(process_controls)
    assert run_call == {
        "config": None,
        "feature_branch": "feature",
        "concurrency_limit": None,
        "strict": True,
        "spec_text": "spec",
        "spec_path": None,
        "interactive": False,
    }
    assert controller.queue_guidance("run-1", "accepted") is True
    assert controller.queue_guidance("run-1", "rejected") is False
    controller.cancel("run-1")
    assert controller.force_stop("run-1", timeout=2.5) is True
    controller.stop_scheduling()
    assert loop.guidance == [("run-1", "accepted"), ("run-1", "rejected")]
    assert loop.cancelled == ["run-1"]
    assert loop.force_stops == [("run-1", 2.5)]
    assert loop.stop_scheduling_calls == 1


def test_controller_subscription_delivers_replacement_snapshot_and_unsubscribes() -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(loop, cast(ExecutionConfig, None), cast(int, None))
    received: list[ExecutionSnapshot] = []

    unsubscribe = controller.subscribe(received.append)
    loop.publish(loop_state(output=("new line",)))
    unsubscribe()
    loop.publish(loop_state(output=("ignored",)))
    assert received == [snapshot(), snapshot(output=("new line",))]
    assert received[1].active_runs[0].output == ("new line",)


def test_controller_snapshot_reads_the_latest_projected_state() -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(loop, cast(ExecutionConfig, None), cast(int, None))

    loop.publish(loop_state(output=("replacement",)))

    assert controller.snapshot() == snapshot(output=("replacement",))


def test_controller_listener_failure_does_not_block_other_listeners(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(loop, cast(ExecutionConfig, None), cast(int, None))
    received: list[ExecutionSnapshot] = []

    listener_calls = 0

    def failing_listener(_snapshot: ExecutionSnapshot) -> None:
        nonlocal listener_calls
        listener_calls += 1
        if listener_calls > 1:
            raise RuntimeError("listener failed")

    controller.subscribe(failing_listener)
    controller.subscribe(received.append)
    loop.publish(loop_state(output=("replacement",)))

    assert received[-1] == snapshot(output=("replacement",))
    assert "failing_listener" in caplog.text


def test_controller_reraises_execution_failure() -> None:
    loop = FakeLoop(loop_state())
    loop.run = MagicMock(side_effect=RuntimeError("worker failed"))  # type: ignore[method-assign]
    controller = ExecutionController(loop, cast(ExecutionConfig, None), cast(int, None))

    with pytest.raises(RuntimeError, match="worker failed"):
        controller.run(feature_branch="feature")


def test_controller_builder_composes_runtime_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    import milknado.adapters as adapters
    import milknado.domains.execution as execution

    constructed: dict[str, Any] = {}

    class FakeExecutor:
        def __init__(self, **kwargs: object) -> None:
            constructed["executor"] = kwargs
            constructed["executor_instance"] = self

    class FakeRunLoop(FakeLoop):
        def __init__(self, **kwargs: object) -> None:
            constructed["loop"] = kwargs
            super().__init__(loop_state())

    monkeypatch.setattr(adapters, "GitAdapter", lambda root: ("git", root))
    monkeypatch.setattr(adapters, "CrgAdapter", lambda root: ("crg", root))
    monkeypatch.setattr(adapters, "LoopAdapter", lambda: "ralph")
    monkeypatch.setattr(execution, "Executor", FakeExecutor)
    monkeypatch.setattr(execution, "RunLoop", FakeRunLoop)
    config = MilknadoConfig(concurrency_limit=3)
    graph = object()
    root = Path("/project")

    controller = build_execution_controller(graph, config, root)

    assert isinstance(controller, ExecutionController)
    assert constructed["executor"] == {
        "graph": graph,
        "git": ("git", root),
        "ralph": "ralph",
        "crg": ("crg", root),
    }
    assert constructed["loop"] == {
        "executor": constructed["executor_instance"],
        "graph": graph,
        "ralph": "ralph",
        "config": config,
    }
    assert controller._execution_config.project_root == root
    assert controller._concurrency_limit == 3


class ThreadBoundLoop:
    def __init__(self, graph: Any) -> None:
        self._graph = graph
        self.started = Event()
        self.release = Event()
        self.calls: list[tuple[str, int]] = []
        self.listener: object | None = None

    def state(self) -> RunLoopState:
        return loop_state()

    def set_state_listener(self, listener: object) -> None:
        self.listener = listener

    def run(self, *, process_controls: Any = None, **_kwargs: object) -> str:
        self.calls.append(("run", get_ident()))
        assert self._graph.get_root() is not None
        self.started.set()
        while not self.release.is_set():
            process_controls()
            sleep(0.001)
        return "result"

    def queue_guidance(self, run_id: str, text: str) -> bool:
        assert self._graph.get_root() is not None
        self.calls.append((f"guidance:{run_id}:{text}", get_ident()))
        return True

    def cancel(self, run_id: str) -> None:
        assert self._graph.get_root() is not None
        self.calls.append((f"cancel:{run_id}", get_ident()))

    def force_stop(self, run_id: str, timeout: float) -> bool:
        assert self._graph.get_root() is not None
        self.calls.append((f"force:{run_id}:{timeout}", get_ident()))
        return True

    def admit_stop_scheduling(self) -> None:
        pass

    def publish(self, state: RunLoopState) -> None:
        assert callable(self.listener)
        self.listener(state)


def test_controller_marshals_run_and_controls_to_one_graph_safe_thread(
    graph: Any,
) -> None:
    graph.add_node("root")
    loop = ThreadBoundLoop(graph)
    controller = ExecutionController(loop, cast(ExecutionConfig, None), cast(int, None))
    result: list[str] = []

    caller = Thread(
        target=lambda: result.append(controller.run(feature_branch="feature")),
    )
    caller.start()
    assert loop.started.wait(timeout=1)

    assert controller.queue_guidance("run-1", "continue") is True
    controller.cancel("run-1")
    assert controller.force_stop("run-1", timeout=2.5) is True
    loop.release.set()
    caller.join(timeout=1)

    assert result == ["result"]
    assert {thread_id for _, thread_id in loop.calls} == {loop.calls[0][1]}


def test_controller_rejects_a_second_concurrent_run(graph: Any) -> None:
    graph.add_node("root")
    loop = ThreadBoundLoop(graph)
    controller = ExecutionController(loop, cast(ExecutionConfig, None), cast(int, None))
    runner = Thread(target=lambda: controller.run(feature_branch="feature"))
    runner.start()
    assert loop.started.wait(timeout=1)

    with pytest.raises(RuntimeError, match="already running"):
        controller.run(feature_branch="feature")

    loop.release.set()
    runner.join(timeout=1)
    assert not runner.is_alive()


class StopAdmissionLoop:
    def __init__(self) -> None:
        self.started = Event()
        self.admitted = Event()
        self.stopped = Event()
        self.listener: object | None = None

    def state(self) -> RunLoopState:
        return loop_state()

    def set_state_listener(self, listener: object) -> None:
        self.listener = listener

    def run(self, *, process_controls: Any = None, **_kwargs: object) -> str:
        self.started.set()
        assert self.admitted.wait(timeout=1)
        process_controls()
        return "result"

    def admit_stop_scheduling(self) -> None:
        self.admitted.set()

    def stop_scheduling(self) -> None:
        self.stopped.set()


def test_controller_admits_stop_before_queuing_control() -> None:
    loop = StopAdmissionLoop()
    controller = ExecutionController(loop, cast(ExecutionConfig, None), cast(int, None))
    result: list[str] = []
    runner = Thread(target=lambda: result.append(controller.run(feature_branch="feature")))
    runner.start()
    assert loop.started.wait(timeout=1)

    stop = Thread(target=controller.stop_scheduling)
    stop.start()
    assert loop.admitted.wait(timeout=1)

    stop.join(timeout=1)
    runner.join(timeout=1)
    assert not stop.is_alive()
    assert result == ["result"]
    assert loop.stopped.is_set()


class _ShutdownGateQueue(Queue[Any]):
    def __init__(self, loop: ThreadBoundLoop) -> None:
        super().__init__()
        self.put_started = Event()
        self.allow_put = Event()
        self._loop = loop

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        self.put_started.set()
        self._loop.release.set()
        assert self.allow_put.wait(timeout=2)
        super().put(item, block=block, timeout=timeout)


def test_controller_rejects_control_admitted_during_shutdown(graph: Any) -> None:
    graph.add_node("root")
    loop = ThreadBoundLoop(graph)
    controller = ExecutionController(loop, cast(ExecutionConfig, None), cast(int, None))
    controls = _ShutdownGateQueue(loop)
    controller._controls = controls
    runner = Thread(target=lambda: controller.run(feature_branch="feature"))
    runner.start()
    assert loop.started.wait(timeout=1)

    control_error: list[BaseException] = []
    control = Thread(
        target=lambda: _capture_control_error(controller, control_error),
        daemon=True,
    )
    control.start()
    assert controls.put_started.wait(timeout=1)
    deadline = monotonic() + 0.1
    while controller._running and monotonic() < deadline:
        sleep(0.001)
    controls.allow_put.set()
    control.join(timeout=1)
    runner.join(timeout=1)

    assert not control.is_alive()
    assert len(control_error) == 1
    assert isinstance(control_error[0], RuntimeError)
    assert str(control_error[0]) == "execution has finished"


def _capture_control_error(controller: ExecutionController, errors: list[BaseException]) -> None:
    try:
        controller.cancel("run-1")
    except BaseException as error:
        errors.append(error)
