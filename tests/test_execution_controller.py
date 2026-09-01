from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from operator import attrgetter
from pathlib import Path
from queue import Queue
from threading import Event, Thread, get_ident
from time import monotonic, sleep
from typing import cast
from unittest.mock import MagicMock

import pytest
from typing_extensions import override

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionController,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
    build_execution_controller,
)
from milknado.domains.common import MilknadoConfig
from milknado.domains.execution import ExecutionConfig, RunLoop
from milknado.domains.execution.run_loop.state import (
    ActiveRunState,
    RunActionState,
    RunLoopState,
)
from milknado.domains.graph import MikadoGraph
from milknado.loop import RunStatus


def _as_run_loop(value: object) -> RunLoop:
    return cast(RunLoop, value)


def _none_config() -> ExecutionConfig:
    return cast(ExecutionConfig, cast(object, None))


def _none_limit() -> int:
    return cast(int, cast(object, None))


def _as_graph(value: object) -> MikadoGraph:
    return cast(MikadoGraph, value)


@dataclass
class FakeLoop:
    current_state: RunLoopState
    listener: Callable[[RunLoopState], None] | None = None
    run_calls: list[dict[str, object]] = field(default_factory=list)
    guidance: list[tuple[str, str]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    force_stops: list[tuple[str, float]] = field(default_factory=list)
    stop_scheduling_calls: int = 0

    def state(self) -> RunLoopState:
        return self.current_state

    def stop_scheduling(self) -> None:
        self.stop_scheduling_calls += 1

    def admit_stop_scheduling(self) -> None:
        pass

    def set_state_listener(self, listener: Callable[[RunLoopState], None]) -> None:
        self.listener = listener

    def run(self, **kwargs: object) -> str:
        self.run_calls.append(kwargs)
        return "result"

    def publish(self, state: RunLoopState) -> None:
        self.current_state = state
        assert self.listener is not None
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
                elapsed_seconds=12.0,
                progress_pct=50.0,
                eta_seconds=8.0,
                attempt=1,
                max_attempts=3,
                stalled=False,
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
                elapsed_seconds=12.0,
                progress_pct=50.0,
                eta_seconds=8.0,
                attempt=1,
                max_attempts=3,
                stalled=False,
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
    controller = ExecutionController(_as_run_loop(loop), _none_config(), _none_limit())

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


def test_controller_waits_for_worker_cleanup_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import milknado.app.run as run_module

    loop = FakeLoop(loop_state())
    controller = ExecutionController(
        cast(RunLoop, cast(object, loop)),
        cast(ExecutionConfig, cast(object, None)),
        cast(int, cast(object, None)),
    )
    outcome_ready = Event()
    release_worker = Event()
    first_returned = Event()
    first_result: list[object] = []

    class GatedQueue(Queue[object]):
        @override
        def put(
            self,
            item: object,
            block: bool = True,
            timeout: float | None = None,
        ) -> None:
            super().put(item, block, timeout)
            if self.maxsize == 1 and not outcome_ready.is_set():
                outcome_ready.set()
                _ = release_worker.wait(timeout=1)

    monkeypatch.setattr(run_module, "Queue", GatedQueue)

    def run_first() -> None:
        first_result.append(controller.run(feature_branch="feature"))
        first_returned.set()

    runner = Thread(target=run_first)
    runner.start()
    assert outcome_ready.wait(timeout=1)
    try:
        assert not first_returned.wait(timeout=0.1)
    finally:
        release_worker.set()
    assert first_returned.wait(timeout=1)
    runner.join(timeout=1)

    assert first_result == ["result"]
    assert controller.run(feature_branch="feature") == "result"


def test_controller_subscription_delivers_replacement_snapshot_and_unsubscribes() -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(_as_run_loop(loop), _none_config(), _none_limit())
    received: list[ExecutionSnapshot] = []

    unsubscribe = controller.subscribe(received.append)
    loop.publish(loop_state(output=("new line",)))
    unsubscribe()
    loop.publish(loop_state(output=("ignored",)))
    assert received == [snapshot(), snapshot(output=("new line",))]
    assert received[1].active_runs[0].output == ("new line",)


def test_controller_snapshot_reads_the_latest_projected_state() -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(_as_run_loop(loop), _none_config(), _none_limit())

    loop.publish(loop_state(output=("replacement",)))

    assert controller.snapshot() == snapshot(output=("replacement",))


def test_controller_listener_failure_does_not_block_other_listeners(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(_as_run_loop(loop), _none_config(), _none_limit())
    received: list[ExecutionSnapshot] = []

    listener_calls = 0

    def failing_listener(_snapshot: ExecutionSnapshot) -> None:
        nonlocal listener_calls
        listener_calls += 1
        if listener_calls > 1:
            raise RuntimeError("listener failed")

    _ = controller.subscribe(failing_listener)
    _ = controller.subscribe(received.append)
    loop.publish(loop_state(output=("replacement",)))

    assert received[-1] == snapshot(output=("replacement",))
    assert "failing_listener" in caplog.text


def test_controller_reraises_execution_failure() -> None:
    loop = FakeLoop(loop_state())
    loop.run = MagicMock(side_effect=RuntimeError("worker failed"))  # type: ignore[method-assign]
    controller = ExecutionController(_as_run_loop(loop), _none_config(), _none_limit())

    with pytest.raises(RuntimeError, match="worker failed"):
        _ = controller.run(feature_branch="feature")


def test_controller_builder_composes_runtime_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    import milknado.adapters as adapters
    import milknado.domains.execution as execution

    constructed: dict[str, object] = {}

    class FakeExecutor:
        def __init__(self, **kwargs: object) -> None:
            constructed["executor"] = kwargs
            constructed["executor_instance"] = self

    class FakeRunLoop(FakeLoop):
        def __init__(self, **kwargs: object) -> None:
            constructed["loop"] = kwargs
            super().__init__(loop_state())

    def fake_git_adapter(root: Path) -> tuple[str, Path]:
        return ("git", root)

    def fake_crg_adapter(root: Path) -> tuple[str, Path]:
        return ("crg", root)

    def fake_loop_adapter() -> str:
        return "ralph"

    monkeypatch.setattr(adapters, "GitAdapter", fake_git_adapter)
    monkeypatch.setattr(adapters, "CrgAdapter", fake_crg_adapter)
    monkeypatch.setattr(adapters, "LoopAdapter", fake_loop_adapter)
    monkeypatch.setattr(execution, "Executor", FakeExecutor)
    monkeypatch.setattr(execution, "RunLoop", FakeRunLoop)
    config = MilknadoConfig(concurrency_limit=3)
    graph = object()
    root = Path("/project")

    controller = build_execution_controller(_as_graph(graph), config, root)

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
    execution_config = cast(ExecutionConfig, attrgetter("_execution_config")(controller))
    assert execution_config.project_root == root
    assert cast(int, attrgetter("_concurrency_limit")(controller)) == 3


class ThreadBoundLoop:
    started: Event
    release: Event

    def __init__(self, graph: MikadoGraph) -> None:
        self._graph: MikadoGraph = graph
        self.started = Event()
        self.release = Event()
        self.calls: list[tuple[str, int]] = []
        self.listener: Callable[[RunLoopState], None] | None = None

    def state(self) -> RunLoopState:
        return loop_state()

    def set_state_listener(self, listener: Callable[[RunLoopState], None]) -> None:
        self.listener = listener

    def run(self, *, process_controls: Callable[[], None] | None = None, **_kwargs: object) -> str:
        self.calls.append(("run", get_ident()))
        assert self._graph.get_root() is not None
        self.started.set()
        assert process_controls is not None
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
        assert self.listener is not None
        self.listener(state)


def test_controller_marshals_run_and_controls_to_one_graph_safe_thread(
    graph: MikadoGraph,
) -> None:
    _ = graph.add_node("root")
    loop = ThreadBoundLoop(graph)
    controller = ExecutionController(_as_run_loop(loop), _none_config(), _none_limit())
    result: list[object] = []

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


def test_controller_propagates_control_failure_from_execution_thread(
    graph: MikadoGraph,
) -> None:
    _ = graph.add_node("root")
    loop = ThreadBoundLoop(graph)
    loop.cancel = MagicMock(side_effect=RuntimeError("cancel failed"))  # type: ignore[method-assign]
    controller = ExecutionController(_as_run_loop(loop), _none_config(), _none_limit())
    runner = Thread(target=lambda: controller.run(feature_branch="feature"))
    runner.start()
    assert loop.started.wait(timeout=1)

    with pytest.raises(RuntimeError, match="cancel failed"):
        controller.cancel("run-1")

    loop.release.set()
    runner.join(timeout=1)
    assert not runner.is_alive()


def test_controller_rejects_a_second_concurrent_run(graph: MikadoGraph) -> None:
    _ = graph.add_node("root")
    loop = ThreadBoundLoop(graph)
    controller = ExecutionController(_as_run_loop(loop), _none_config(), _none_limit())
    runner = Thread(target=lambda: controller.run(feature_branch="feature"))
    runner.start()
    assert loop.started.wait(timeout=1)

    with pytest.raises(RuntimeError, match="already running"):
        _ = controller.run(feature_branch="feature")

    loop.release.set()
    runner.join(timeout=1)
    assert not runner.is_alive()


class StopAdmissionLoop:
    started: Event
    admitted: Event
    stopped: Event

    def __init__(self) -> None:
        self.started = Event()
        self.admitted = Event()
        self.stopped = Event()
        self.listener: Callable[[RunLoopState], None] | None = None

    def state(self) -> RunLoopState:
        return loop_state()

    def set_state_listener(self, listener: Callable[[RunLoopState], None]) -> None:
        self.listener = listener

    def run(self, *, process_controls: Callable[[], None] | None = None, **_kwargs: object) -> str:
        self.started.set()
        assert self.admitted.wait(timeout=1)
        assert process_controls is not None
        process_controls()
        return "result"

    def admit_stop_scheduling(self) -> None:
        self.admitted.set()

    def stop_scheduling(self) -> None:
        self.stopped.set()


def test_controller_admits_stop_before_queuing_control() -> None:
    loop = StopAdmissionLoop()
    controller = ExecutionController(_as_run_loop(loop), _none_config(), _none_limit())
    result: list[object] = []
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


class _ShutdownGateQueue(Queue[object]):
    put_started: Event
    allow_put: Event

    def __init__(self, loop: ThreadBoundLoop) -> None:
        super().__init__()
        self.put_started = Event()
        self.allow_put = Event()
        self._bound_loop: ThreadBoundLoop = loop

    @override
    def put(self, item: object, block: bool = True, timeout: float | None = None) -> None:
        self.put_started.set()
        self._bound_loop.release.set()
        assert self.allow_put.wait(timeout=2)
        super().put(item, block=block, timeout=timeout)


def test_controller_rejects_control_admitted_during_shutdown(graph: MikadoGraph) -> None:
    _ = graph.add_node("root")
    loop = ThreadBoundLoop(graph)
    controller = ExecutionController(_as_run_loop(loop), _none_config(), _none_limit())
    controls = _ShutdownGateQueue(loop)
    controls_attr = "_controls"
    setattr(cast(object, controller), controls_attr, controls)
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
    while attrgetter("_running")(controller) and monotonic() < deadline:
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


def test_run_execution_loop_passes_interactive_false(monkeypatch: pytest.MonkeyPatch) -> None:
    import milknado.adapters as adapters
    import milknado.domains.dispatch as dispatch
    import milknado.domains.execution as execution
    from milknado.app.run import run_execution_loop

    captured: dict[str, object] = {}

    class FakeRunLoop(FakeLoop):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(loop_state())

        @override
        def run(self, **kwargs: object) -> str:
            captured.update(kwargs)
            return "result"

    def no_orphans(_graph: MikadoGraph) -> None:
        pass

    def fake_git_adapter(root: Path) -> tuple[str, Path]:
        return ("git", root)

    def fake_crg_adapter(root: Path) -> tuple[str, Path]:
        return ("crg", root)

    def fake_loop_adapter() -> str:
        return "ralph"

    def fake_executor(**kwargs: object) -> object:
        _ = kwargs
        return object()

    monkeypatch.setattr(dispatch, "reconcile_orphaned_runs", no_orphans)
    monkeypatch.setattr(adapters, "GitAdapter", fake_git_adapter)
    monkeypatch.setattr(adapters, "CrgAdapter", fake_crg_adapter)
    monkeypatch.setattr(adapters, "LoopAdapter", fake_loop_adapter)
    monkeypatch.setattr(execution, "Executor", fake_executor)
    monkeypatch.setattr(execution, "RunLoop", FakeRunLoop)

    _ = run_execution_loop(
        _as_graph(object()),
        MilknadoConfig(concurrency_limit=3),
        Path("/project"),
        "feature",
        False,
    )

    assert captured["interactive"] is False
