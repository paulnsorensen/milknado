from __future__ import annotations

import asyncio
import signal
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

from milknado.app._shutdown import ShutdownIntent, ShutdownSignal
from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionController,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
)
from milknado.app.run_source import NodeSnapshotRequest
from milknado.app.run_tui import ExecutionApp
from milknado.app.watch import WatchSnapshotSource
from milknado.domains.common import (
    MilknadoConfig,
    NodeKind,
    NodeSpec,
    RunResult,
    SessionContext,
    SessionEvent,
)
from milknado.domains.execution import ExecutionConfig, RunLoop
from milknado.domains.execution.run_loop.state import (
    ActiveRunState,
    RunActionState,
    RunLoopState,
)
from milknado.domains.graph import (
    GoalReviewDecision,
    GoalReviewDecisionRequest,
    GoalReviewRecord,
    GoalReviewRequest,
    MikadoGraph,
)
from milknado.loop import RunStatus


def _as_run_loop(value: object) -> RunLoop:
    return cast(RunLoop, value)


def _none_config() -> ExecutionConfig:
    return cast(ExecutionConfig, cast(object, None))


def _none_limit() -> int:
    return cast(int, cast(object, None))


def _policy_config() -> MilknadoConfig:
    return MilknadoConfig(protected_branches=())


@dataclass
class FakeLoop:
    current_state: RunLoopState
    listener: Callable[[RunLoopState], None] | None = None
    run_calls: list[dict[str, object]] = field(default_factory=list)
    guidance: list[tuple[str, str]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    force_stops: list[tuple[str, float]] = field(default_factory=list)
    stop_scheduling_calls: int = 0
    force_stop_deadlines: list[float] = field(default_factory=list)

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

    def force_stop_active(self, deadline: float) -> bool:
        self.force_stop_deadlines.append(deadline)
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


def _request_goal_review(graph: MikadoGraph, goal_id: int, evidence: str) -> GoalReviewRecord:
    return graph.request_goal_review(
        GoalReviewRequest(
            goal_id=goal_id,
            goal_revision="sha256:goal",
            evidence=evidence,
            proposed_change="Review proposed change",
            reviewer="worker",
            assessed_at="2026-09-13T12:00:00+00:00",
        )
    )


def test_controller_delegates_run_and_control_ports() -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )

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
        "await_owner_work": False,
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


def test_force_stop_all_calls_loop_without_control_queue() -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )
    start = monotonic()

    assert controller.force_stop_all(timeout=1.0) is True
    assert len(loop.force_stop_deadlines) == 1
    assert start + 1.0 <= loop.force_stop_deadlines[0] <= monotonic() + 1.0


def test_force_stop_all_returns_at_deadline_when_cleanup_blocks() -> None:
    release = Event()

    class BlockedStop(FakeLoop):
        def force_stop_active(self, deadline: float) -> bool:
            self.force_stop_deadlines.append(deadline)
            _ = release.wait(0.3)
            return True

    loop = BlockedStop(loop_state())
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )
    start = monotonic()
    try:
        assert controller.force_stop_all(timeout=0.05) is False
        assert monotonic() - start < 0.2
    finally:
        release.set()


def test_main_thread_observes_signal_while_execution_thread_blocks() -> None:
    started = Event()
    release = Event()
    intent = ShutdownIntent()

    class BlockedLoop(FakeLoop):
        def run(self, **kwargs: object) -> str:
            del kwargs
            started.set()
            _ = release.wait(2.0)
            return "result"

    loop = BlockedLoop(loop_state())
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config(),
        shutdown_intent=intent,
    )
    signaler = Thread(target=lambda: (started.wait(), intent.record(signal.SIGTERM, None)))
    signaler.start()
    start = monotonic()
    try:
        with pytest.raises(ShutdownSignal) as caught:
            _ = controller.run(feature_branch="feature")
    finally:
        release.set()
        signaler.join(1.0)

    assert caught.value.signum == signal.SIGTERM
    assert monotonic() - start < 1.0
    assert loop.force_stop_deadlines == [intent.deadline(8.0)]


@pytest.mark.asyncio
async def test_tui_quit_force_stops_real_controller_before_exit() -> None:
    started = Event()
    released = Event()
    stopped = Event()

    class BlockingLoop(FakeLoop):
        @override
        def run(self, **kwargs: object) -> str:
            del kwargs
            started.set()
            assert released.wait(3.0)
            return "result"

        @override
        def force_stop_active(self, deadline: float) -> bool:
            self.force_stop_deadlines.append(deadline)
            stopped.set()
            released.set()
            return True

    loop = BlockingLoop(loop_state())
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )
    app = ExecutionApp(controller, feature_branch="feature")
    async with app.run_test(size=(120, 36)) as pilot:
        assert await asyncio.to_thread(started.wait, 2.0)
        await pilot.press("q", "y")
        assert await asyncio.to_thread(stopped.wait, 2.0)
        assert loop.stop_scheduling_calls == 0
        assert len(loop.force_stop_deadlines) == 1


def test_project_and_watch_snapshots_share_pending_goal_review_filter(tmp_path: Path) -> None:
    db_path = tmp_path / "milknado.db"
    graph = MikadoGraph(db_path)
    pending_goal = graph.add_node("Pause execution", spec=NodeSpec(kind=NodeKind.GOAL))
    archived_goal = graph.add_node("Archived execution", spec=NodeSpec(kind=NodeKind.GOAL))
    graph.mark_running(archived_goal.id)
    graph.mark_done(archived_goal.id)
    archived_review = _request_goal_review(graph, archived_goal.id, "Archived evidence")
    _ = graph.archive_subtree(archived_goal.id)

    accepted_goal = graph.add_node("Accepted execution", spec=NodeSpec(kind=NodeKind.GOAL))
    accepted_review = _request_goal_review(graph, accepted_goal.id, "Accepted evidence")
    rejected_goal = graph.add_node("Rejected execution", spec=NodeSpec(kind=NodeKind.GOAL))
    rejected_review = _request_goal_review(graph, rejected_goal.id, "Rejected evidence")
    graph.register_controller_master()
    _ = graph.decide_goal_review(
        GoalReviewDecisionRequest(accepted_review.review_id, GoalReviewDecision.ACCEPTED),
        decided_by="controller",
    )
    _ = graph.decide_goal_review(
        GoalReviewDecisionRequest(rejected_review.review_id, GoalReviewDecision.REJECTED),
        decided_by="controller",
    )
    pending_review = _request_goal_review(graph, pending_goal.id, "Pending evidence")

    try:
        run_snapshot = ExecutionController._project_snapshot(  # pyright: ignore[reportPrivateUsage]
            loop_state(), graph
        )
    finally:
        graph.close()

    source = WatchSnapshotSource(tmp_path, db_path)
    try:
        watch_snapshot = source.snapshot()
    finally:
        source.close()

    assert archived_review not in run_snapshot.pending_goal_reviews
    assert run_snapshot.pending_goal_reviews == watch_snapshot.pending_goal_reviews
    assert run_snapshot.pending_goal_reviews == (pending_review,)


def test_controller_refuses_protected_branch_before_run() -> None:
    from milknado.app.run import ProtectedBranchRefusal

    loop = FakeLoop(loop_state())
    config = MilknadoConfig(protected_branches=("main",))
    controller = ExecutionController(_as_run_loop(loop), _none_config(), _none_limit(), config)

    with pytest.raises(ProtectedBranchRefusal, match="protected branch"):
        _ = controller.run(feature_branch="main")
    assert loop.run_calls == []


def test_controller_waits_for_worker_cleanup_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import milknado.app._shutdown as shutdown_module

    loop = FakeLoop(loop_state())
    controller = ExecutionController(
        cast(RunLoop, cast(object, loop)),
        cast(ExecutionConfig, cast(object, None)),
        cast(int, cast(object, None)),
        _policy_config(),
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

    monkeypatch.setattr(shutdown_module, "Queue", GatedQueue)

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


def test_controller_snapshot_uses_durable_run_totals(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        node = graph.add_node("node")
        outcomes = (("failed", "worker failed"), ("failed", "cancelled"), ("done", None))
        for index, (status, error) in enumerate(outcomes):
            run_id = f"run-{index}"
            graph.runs.start(run_id, node.id, "", "2026-09-11T00:00:00+00:00", 60)
            graph.runs.finish(
                run_id,
                RunResult(
                    status=status,
                    exit_code=0 if status == "done" else 1,
                    timed_out=False,
                    ended_at="2026-09-11T00:00:01+00:00",
                    error=error,
                    detail=None,
                    rebased=None,
                ),
            )

        controller = ExecutionController(
            _as_run_loop(FakeLoop(loop_state())),
            _none_config(),
            _none_limit(),
            _policy_config(),
            graph,
        )
        snapshot = controller.snapshot()
        assert (snapshot.completed, snapshot.failed, snapshot.stopped) == (1, 1, 1)
    finally:
        graph.close()


def test_controller_subscription_delivers_replacement_snapshot_and_unsubscribes() -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )
    received: list[ExecutionSnapshot] = []

    unsubscribe = controller.subscribe(received.append)
    loop.publish(loop_state(output=("new line",)))
    unsubscribe()
    loop.publish(loop_state(output=("ignored",)))
    assert received == [snapshot(), snapshot(output=("new line",))]
    assert received[1].active_runs[0].output == ("new line",)


def test_controller_removes_listener_when_initial_replay_fails() -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )
    calls = 0

    def failing_listener(_snapshot: ExecutionSnapshot) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("initial replay failed")

    with pytest.raises(RuntimeError, match="initial replay failed"):
        _ = controller.subscribe(failing_listener)

    loop.publish(loop_state(output=("replacement",)))
    assert calls == 1


def test_controller_snapshot_reads_the_latest_projected_state() -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )

    loop.publish(loop_state(output=("replacement",)))

    assert controller.snapshot() == snapshot(output=("replacement",))


def test_controller_listener_failure_is_visible_to_other_listeners(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loop = FakeLoop(loop_state())
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )
    received: list[ExecutionSnapshot] = []

    listener_calls = 0

    def failing_listener(_snapshot: ExecutionSnapshot) -> None:
        nonlocal listener_calls
        listener_calls += 1
        if listener_calls > 1:
            raise RuntimeError(f"listener failed {listener_calls}")

    def error_listener(current: ExecutionSnapshot) -> None:
        if current.listener_errors:
            raise RuntimeError("cannot render errors")

    unsubscribe = controller.subscribe(failing_listener)
    error_unsubscribe = controller.subscribe(error_listener)
    _ = controller.subscribe(received.append)
    with caplog.at_level("ERROR", logger="milknado.app.run"):
        loop.publish(loop_state(output=("replacement",)))
        loop.publish(loop_state(output=("second",)))

    assert listener_calls == 3
    assert received[-1].active_runs[0].output == ("second",)

    listener_errors = received[-1].listener_errors
    assert controller.snapshot().listener_errors == listener_errors
    assert set(listener_errors) == {
        f"{failing_listener.__qualname__}: listener failed {listener_calls}",
        f"{error_listener.__qualname__}: cannot render errors",
    }
    assert "failed while publishing error" in caplog.text
    unsubscribe()
    error_unsubscribe()
    assert controller.snapshot().listener_errors == ()
    from milknado.app.run_view import events_text

    rendered = events_text(received[-1].event_lines, listener_errors)
    assert rendered.splitlines()[0] == f"Listener error: {listener_errors[0]}"


def test_controller_reraises_execution_failure() -> None:
    loop = FakeLoop(loop_state())
    loop.run = MagicMock(side_effect=RuntimeError("worker failed"))  # type: ignore[method-assign]
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        _ = controller.run(feature_branch="feature")


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
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )
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
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )
    runner = Thread(target=lambda: controller.run(feature_branch="feature"))
    runner.start()
    assert loop.started.wait(timeout=1)

    with pytest.raises(RuntimeError, match="cancel failed"):
        controller.cancel("run-1")

    loop.release.set()
    runner.join(timeout=1)
    assert not runner.is_alive()


def test_controller_exposes_node_snapshot_through_shared_source_contract(
    graph: MikadoGraph,
) -> None:
    node = graph.add_node("Controller detail")
    loop = FakeLoop(loop_state())
    controller = ExecutionController(
        _as_run_loop(loop),
        _none_config(),
        _none_limit(),
        _policy_config(),
        graph=graph,
    )

    response = controller.node_snapshot(NodeSnapshotRequest(node.id, request_generation=9))

    assert response.matches(node.id, 9)
    assert response.detail is not None
    assert response.detail.description == "Controller detail"


def test_controller_forwards_session_event_page(
    graph: MikadoGraph,
) -> None:
    node = graph.add_node("Controller session detail")
    _ = graph.runs.start(
        "controller-run",
        node.id,
        "controller.log",
        "2026-09-12T00:00:00+00:00",
        60,
    )
    _ = graph.sessions.start("controller-run", SessionContext(family="codex", cwd="."))
    _ = graph.sessions.append("controller-run", SessionEvent(kind="status", text="first"))
    _ = graph.sessions.append("controller-run", SessionEvent(kind="status", text="second"))
    controller = ExecutionController(
        _as_run_loop(FakeLoop(loop_state())),
        _none_config(),
        _none_limit(),
        _policy_config(),
        graph=graph,
    )

    response = controller.node_snapshot(
        NodeSnapshotRequest(node.id, request_generation=9, limit=1, session_event_page=1)
    )

    assert response.detail is not None
    sessions = response.detail.sessions.items
    assert sessions is not None and sessions[0].event_history.items is not None
    assert tuple(event.text for event in sessions[0].event_history.items) == ("first",)


def test_controller_rejects_a_second_concurrent_run(graph: MikadoGraph) -> None:
    _ = graph.add_node("root")
    loop = ThreadBoundLoop(graph)
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )
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
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )
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
    controller = ExecutionController(
        _as_run_loop(loop), _none_config(), _none_limit(), _policy_config()
    )
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
