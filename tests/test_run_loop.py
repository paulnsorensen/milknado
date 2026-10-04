import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from operator import attrgetter
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from typing import Protocol, TypeVar, cast
from unittest.mock import MagicMock, patch

import pytest

from milknado.adapters.loop import LoopAdapter
from milknado.domains.common import (
    CONTROLLER_MASTER_ENV,
    NodeKind,
    ProgressEvent,
    SessionInput,
    SessionView,
    TerminalRunOutcome,
    VerifySpecResult,
)
from milknado.domains.common.config import Gate, MilknadoConfig
from milknado.domains.common.types import NodeSpec, NodeStatus, RebaseResult
from milknado.domains.execution import (
    NO_GATES_CONFIGURED_MESSAGE,
    ExecutionConfig,
    Executor,
    RunLoop,
)
from milknado.domains.execution._models import (
    DispatchResult,
    PreservedWorkerRun,
    RebaseConflict,
)
from milknado.domains.execution.executor import WorktreeManager
from milknado.domains.execution.run_loop._scheduler import Scheduler
from milknado.domains.execution.run_loop.state import RunLoopState, summarize_description
from milknado.domains.graph import (
    ConcurrencyLimitReached,
    GoalReviewDecision,
    GoalReviewDecisionRequest,
    GoalReviewRequest,
    MikadoGraph,
)
from milknado.loop import RunStatus

T = TypeVar("T")


def _scheduler(loop: RunLoop) -> Scheduler:
    return cast(Scheduler, attrgetter("_scheduler")(loop))


def _success(loop: object) -> dict[str, bool]:
    return cast(dict[str, bool], attrgetter("_success")(loop))


def _runs(loop: object) -> dict[str, "FakeRun"]:
    return cast(dict[str, "FakeRun"], attrgetter("_runs")(loop))


def _ordinal_to_run_id(loop: object) -> dict[str, str]:
    return cast(dict[str, str], attrgetter("_ordinal_to_run_id")(loop))


def _mock_attr(value: object, name: str) -> MagicMock:
    return cast(MagicMock, getattr(value, name))


def _set_attr(target: object, name: str, value: object) -> None:
    setattr(target, name, value)


def _call_kwargs(mock: MagicMock) -> dict[str, object]:
    call = cast(object | None, attrgetter("call_args")(mock))
    if call is None:
        return {}
    return cast(dict[str, object], attrgetter("kwargs")(call))


def _mock(loop: RunLoop, name: str) -> MagicMock:
    return cast(MagicMock, getattr(loop, name))


def _worktree_manager(executor: Executor) -> WorktreeManager:
    return cast(WorktreeManager, attrgetter("_wt")(executor))


def _managed_worktrees(executor: Executor) -> dict[int, Path]:
    return cast(dict[int, Path], attrgetter("_worktrees")(_worktree_manager(executor)))


def _execute_run(
    loop: RunLoop,
    config: ExecutionConfig,
    feature_branch: str,
    concurrency_limit: int,
    timeout: float | None,
    interactive: bool,
) -> tuple[int, int, int, list[RebaseConflict], bool]:
    method = cast(
        Callable[
            [ExecutionConfig, str, int, float | None, bool],
            tuple[int, int, int, list[RebaseConflict], bool],
        ],
        attrgetter("_execute_run")(loop),
    )
    return method(config, feature_branch, concurrency_limit, timeout, interactive)


def _dispatch_batch(
    loop: RunLoop, config: ExecutionConfig, concurrency_limit: int
) -> tuple[int, int]:
    method = cast(
        Callable[[ExecutionConfig, int], tuple[int, int]],
        attrgetter("_dispatch_batch")(loop),
    )
    return method(config, concurrency_limit)


def _handle_completion_timeout(loop: RunLoop, timeout: object) -> int:
    method = cast(Callable[[object], int], attrgetter("_handle_completion_timeout")(loop))
    return method(timeout)


def _publish_state(loop: RunLoop) -> None:
    method = cast(Callable[[], None], attrgetter("_publish_state")(loop))
    method()


def _progress_before_completion(loop: object) -> list[ProgressEvent]:
    return cast(list[ProgressEvent], attrgetter("_progress_before_completion")(loop))


@dataclass
class FakeRunState:
    run_id: str = "run-1"
    status: RunStatus = RunStatus.RUNNING
    total: int = 0
    stop_requested: bool = False
    force_stop_requested: bool = False


@dataclass
class FakeRun:
    state: FakeRunState = field(default_factory=FakeRunState)


@dataclass(frozen=True)
class _FakeReview:
    approved: bool = True
    findings_md: str = ""
    error: bool = False


class FakeGit:
    def __init__(self) -> None:
        self.created: list[tuple[Path, str]] = []
        self.removed: list[Path] = []
        self.rebase_result: RebaseResult = RebaseResult(success=True)

    def branch_exists(self, branch: str) -> bool:
        _ = branch
        return False

    def create_worktree(self, path: Path, branch: str) -> Path:
        self.created.append((path, branch))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def git_common_dir(self, worktree: Path) -> Path | None:
        _ = worktree
        return None

    def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
        self.removed.append(path)
        _ = target

    def worktree_teardown_blocker(self, path: Path, target: str = "HEAD") -> str | None:
        _ = (path, target)
        return None

    def force_remove_worktree(self, path: Path) -> None:
        self.removed.append(path)

    def delete_branch(self, branch: str) -> None:
        _ = branch

    def prune_worktrees(self) -> None:
        pass

    def rebase(self, worktree: Path, onto: str) -> RebaseResult:
        _ = (worktree, onto)
        return self.rebase_result

    def current_branch(self) -> str:
        return "main"

    def resolve_ref(self, ref: str) -> str:
        return f"{ref}-oid"

    def diff_for_review(self, worktree: Path, base_oid: str) -> str:
        _ = (worktree, base_oid)
        return ""

    def compare_and_swap_ref(self, ref: str, expected_oid: str, new_oid: str) -> None:
        _ = (ref, expected_oid, new_oid)

    def squash_and_commit(self, worktree: Path, onto: str, msg: str) -> bool:
        _ = (worktree, onto, msg)
        return True

    def fast_forward(self, branch: str) -> None:
        _ = branch

    def untracked_merge_collisions(self, worktree: Path) -> tuple[str, ...]:
        _ = worktree
        return ()


class FakeCrg:
    def ensure_graph(self, project_root: Path) -> None:
        _ = project_root

    def get_impact_radius(self, files: list[str]) -> dict[str, object]:
        return {"files": files}

    def get_architecture_overview(self) -> dict[str, object]:
        return {"modules": []}

    def list_communities(
        self, sort_by: str = "size", min_size: int = 0
    ) -> list[dict[str, object]]:
        _ = (sort_by, min_size)
        return []

    def list_flows(self, sort_by: str = "criticality", limit: int = 50) -> list[dict[str, object]]:
        _ = (sort_by, limit)
        return []

    def get_bridge_nodes(self, top_n: int = 10) -> list[dict[str, object]]:
        _ = top_n
        return []

    def get_hub_nodes(self, top_n: int = 10) -> list[dict[str, object]]:
        _ = top_n
        return []


class FakeLoop:
    _run_counter: int

    def __init__(self) -> None:
        self._run_counter = 0
        self._success: dict[str, bool] = {}
        self._outcomes: dict[str, TerminalRunOutcome] = {}
        self._pending_completions: list[tuple[str, TerminalRunOutcome]] = []
        self._runs: dict[str, FakeRun] = {}
        self.output: dict[str, list[str]] = {}
        self.guidance: dict[str, tuple[str, ...]] = {}
        self._progress_before_completion: list[ProgressEvent] = []
        self.requested_stops: list[str] = []
        self.force_stops: list[tuple[str, float | None]] = []
        self.stop_active_deadlines: list[float] = []
        self._ordinal_to_run_id: dict[str, str] = {}
        self._run_id_to_ordinal: dict[str, str] = {}

    def create_run(
        self,
        agent: str,
        loop_dir: Path,
        loop_file: Path,
        quality_gates: tuple[Gate, ...] | None,
        project_root: Path | None = None,
        commit_footer: str | None = None,
        base_oid: str | None = None,
        runtime_policy: object | None = None,
        run_id: str | None = None,
        completion_probe: Callable[[], bool] | None = None,
        max_iterations: int | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> FakeRun:
        _ = (
            agent,
            loop_dir,
            loop_file,
            quality_gates,
            project_root,
            commit_footer,
            base_oid,
            runtime_policy,
            completion_probe,
            max_iterations,
            timeout,
            env,
        )
        self._run_counter += 1
        ordinal_id = f"run-{self._run_counter}"
        resolved_run_id = run_id or ordinal_id
        self._ordinal_to_run_id[ordinal_id] = resolved_run_id
        self._run_id_to_ordinal[resolved_run_id] = ordinal_id
        success = self._success.get(resolved_run_id, self._success.get(ordinal_id, True))
        outcome = self._outcomes.get(
            resolved_run_id,
            self._outcomes.get(
                ordinal_id, TerminalRunOutcome("completed" if success else "failed")
            ),
        )
        self._pending_completions.append((resolved_run_id, outcome))
        run = FakeRun(state=FakeRunState(run_id=resolved_run_id))
        self._runs[resolved_run_id] = run
        return run

    def start_run(self, run_id: str) -> None:
        _ = run_id

    def request_stop_run(self, run_id: str) -> None:
        self.requested_stops.append(run_id)
        self._runs[run_id].state.stop_requested = True

    def force_stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        self.force_stops.append((run_id, timeout))
        self._runs[run_id].state.force_stop_requested = True
        return True

    def stop_active_workers(self, deadline: float) -> bool:
        self.stop_active_deadlines.append(deadline)
        return True

    def stop_run_workers(self, graph_run_id: str, deadline: float) -> bool:
        _ = (graph_run_id, deadline)
        return True

    def stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        _ = (run_id, timeout)
        return True

    def list_runs(self) -> list[FakeRun]:
        return list(self._runs.values())

    def get_run(self, run_id: str) -> FakeRun | None:
        return self._runs.get(run_id)

    def is_run_alive(self, run_id: str) -> bool:
        _ = run_id
        return False

    def _seeded(self, mapping: dict[str, T], run_id: str, default: T) -> T:
        if run_id in mapping:
            return mapping[run_id]
        ordinal = self._run_id_to_ordinal.get(run_id)
        if ordinal is not None and ordinal in mapping:
            return mapping[ordinal]
        return default

    def get_run_stdout(self, run_id: str) -> list[str]:
        return self._seeded(self.output, run_id, [])

    def get_run_failure_detail(self, run_id: str) -> str | None:
        _ = run_id
        return None

    def get_run_output_tail(self, run_id: str, max_lines: int) -> list[str]:
        return self._seeded(self.output, run_id, [])[-max_lines:]

    def get_run_session(self, run_id: str) -> SessionView:
        _ = run_id
        return SessionView()

    def get_run_session_id(self, run_id: str) -> str | None:
        _ = run_id
        return None

    def session_input(self, run_id: str, command: SessionInput) -> bool:
        _ = run_id, command
        return False

    def get_run_guidance(self, run_id: str) -> tuple[str, ...]:
        return self._seeded(self.guidance, run_id, ())

    def queue_guidance(self, run_id: str, text: str) -> bool:
        self.guidance[run_id] = (*self.guidance.get(run_id, ()), text)
        return True

    def wait_for_next_completion(
        self,
        active_run_ids: set[str],
        timeout: float | None = None,
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]:
        _ = timeout
        if self._progress_before_completion:
            event = self._progress_before_completion.pop(0)
            resolved_run_id = self._ordinal_to_run_id.get(event.run_id, event.run_id)
            if resolved_run_id in active_run_ids:
                if resolved_run_id != event.run_id:
                    event = replace(event, run_id=resolved_run_id)
                return resolved_run_id, event
        for index, (run_id, outcome) in enumerate(self._pending_completions):
            if run_id in active_run_ids:
                _ = self._pending_completions.pop(index)
                return run_id, outcome
        raise RuntimeError("No pending completions for active runs")

    def poll_progress_events(self) -> list[ProgressEvent]:
        return []

    def run_node_review(
        self,
        agent: str,
        prompt: str,
        worktree: Path,
        project_root: Path,
        *,
        timeout_seconds: float,
        graph_run_id: str | None = None,
    ) -> _FakeReview:
        _ = (agent, prompt, worktree, project_root, timeout_seconds, graph_run_id)
        return _FakeReview()

    def verify_spec(self, spec_text: str, graph_state: str) -> VerifySpecResult:
        _ = (spec_text, graph_state)
        return VerifySpecResult(outcome="done")

    def generate_loop_md(
        self,
        brief: str,
        quality_gates: tuple[Gate, ...] | None,
        output_path: Path,
        prior_findings: str = "",
        findings_round: int | None = None,
    ) -> Path:
        _ = (brief, quality_gates, prior_findings, findings_round)
        return output_path

    def set_run_fails(self, run_id: str) -> None:
        self._success[run_id] = False

    def set_run_stopped(self, run_id: str) -> None:
        self._outcomes[run_id] = TerminalRunOutcome("stopped")


@pytest.fixture()
def config(tmp_path: Path) -> ExecutionConfig:
    from milknado.domains.common.config import Gate

    return ExecutionConfig(
        execution_agent="claude",
        quality_gates=(Gate(command="uv run pytest"),),
        worktree_pattern="milknado-{node_id}-{slug}",
        project_root=tmp_path,
    )


@pytest.fixture()
def fake_git() -> FakeGit:
    return FakeGit()


@pytest.fixture()
def fake_loop() -> FakeLoop:
    return FakeLoop()


@pytest.fixture()
def fake_crg() -> FakeCrg:
    return FakeCrg()


@pytest.fixture()
def executor(
    graph: MikadoGraph,
    fake_git: FakeGit,
    fake_loop: FakeLoop,
    fake_crg: FakeCrg,
) -> Executor:
    return Executor(graph=graph, git=fake_git, loop=fake_loop, crg=fake_crg)


def test_shutdown_intent_refuses_dispatch_before_main_stop(
    graph: MikadoGraph, executor: Executor, fake_loop: FakeLoop, config: ExecutionConfig
) -> None:
    root = graph.add_node("root")
    _ = graph.add_node("leaf", parent_id=root.id)
    run_loop = RunLoop(
        executor=executor,
        graph=graph,
        loop=fake_loop,
        shutdown_requested=lambda: True,
    )

    result = run_loop.run(config, "feature", interactive=False)

    assert result.dispatched_total == 0
    assert fake_loop.list_runs() == []


def test_force_stop_active_closes_admission_without_scheduling_lock(
    run_loop: RunLoop, fake_loop: FakeLoop
) -> None:
    lock = run_loop._scheduling_lock  # pyright: ignore[reportPrivateUsage]
    finished = Event()
    deadline = monotonic() + 1.0
    _ = lock.acquire()
    try:
        worker = Thread(
            target=lambda: (run_loop.force_stop_active(deadline), finished.set()), daemon=True
        )
        worker.start()
        assert finished.wait(0.5)
        assert fake_loop.stop_active_deadlines == [deadline]
        assert _scheduler(run_loop).view().scheduling_stopped is True
    finally:
        lock.release()


def test_state_is_bounded_and_published(
    run_loop: RunLoop,
    graph: MikadoGraph,
    fake_loop: FakeLoop,
) -> None:
    root = graph.add_node("ship controller")
    leaf = graph.add_node("build snapshots", parent_id=root.id)
    graph.mark_running(leaf.id)
    _runs(fake_loop)["run-1"] = FakeRun(state=FakeRunState(run_id="run-1", stop_requested=True))
    fake_loop.output["run-1"] = [f"line {index}" for index in range(35)]
    fake_loop.guidance["run-1"] = ("use domain barrels",)
    _scheduler(run_loop).admit_run("run-1", leaf.id, monotonic())
    _scheduler(run_loop).record_progress(
        ProgressEvent(run_id="run-1", work=1, total=2, message="building")
    )
    received: list[RunLoopState] = []
    run_loop.set_state_listener(received.append)
    _publish_state(
        run_loop,
    )

    state = received[0]
    active = state.active_runs[0]
    assert state.goal == "ship controller"
    assert state.active_runs == (active,)
    assert active.status is RunStatus.RUNNING
    assert active.progress == "building"
    assert active.stop_requested is True
    assert active.actions.cancel_reason == "stop already requested"
    assert active.actions.guidance_reason == "run is stopping"
    assert active.actions.force_stop_reason is None
    assert active.output == tuple(f"line {index}" for index in range(5, 35))
    assert active.pending_guidance == ("use domain barrels",)


def test_state_collects_configured_projection_facts(
    graph: MikadoGraph, executor: Executor, fake_loop: FakeLoop
) -> None:
    run_loop = RunLoop(
        executor=executor,
        graph=graph,
        loop=fake_loop,
        config=MilknadoConfig(dispatch_max_retries=4, stall_threshold_seconds=300),
    )
    root = graph.add_node("ship controller")
    leaf = graph.add_node("build snapshots", parent_id=root.id)
    graph.mark_running(leaf.id)
    scheduler = _scheduler(run_loop)
    scheduler.admit_run("run-1", leaf.id, 100.0)
    scheduler.record_failure(leaf.id, strict=False)
    for duration in (10.0, 20.0, 30.0):
        scheduler.record_completion(duration)
    scheduler.record_progress(ProgressEvent(run_id="run-1", work=3, total=4))

    with patch("milknado.domains.execution.run_loop.time.monotonic", return_value=105.0):
        snapshot = run_loop.state().active_runs[0]

    assert snapshot.elapsed_seconds == 5.0
    assert snapshot.eta_seconds == 15.0
    assert snapshot.progress_pct == 75.0
    assert snapshot.attempt == 2
    assert snapshot.max_attempts == 5
    assert snapshot.stalled is False


def test_state_uses_configured_stall_threshold(
    graph: MikadoGraph, executor: Executor, fake_loop: FakeLoop
) -> None:
    run_loop = RunLoop(
        executor=executor,
        graph=graph,
        loop=fake_loop,
        config=MilknadoConfig(stall_threshold_seconds=60),
    )
    root = graph.add_node("ship controller")
    leaf = graph.add_node("build snapshots", parent_id=root.id)
    graph.mark_running(leaf.id)
    _scheduler(run_loop).admit_run("run-1", leaf.id, 100.0)

    with patch("milknado.domains.execution.run_loop.time.monotonic", return_value=159.0):
        before = run_loop.state().active_runs[0]
    with patch("milknado.domains.execution.run_loop.time.monotonic", return_value=160.0):
        at_threshold = run_loop.state().active_runs[0]

    assert before.progress_pct is None
    assert before.stalled is False
    assert at_threshold.progress_pct is None
    assert at_threshold.stalled is True


def test_terminal_run_duration_seconds_from_stopped_completion(
    run_loop: RunLoop, graph: MikadoGraph, fake_loop: FakeLoop
) -> None:
    from milknado.domains.execution.run_loop._completion import (
        CompletionContext,
        handle_completion,
    )

    root = graph.add_node("ship controller")
    leaf = graph.add_node("build snapshots", parent_id=root.id)
    graph.mark_running(leaf.id)
    _scheduler(run_loop).admit_run("run-1", leaf.id, 100.0)
    _runs(fake_loop)["run-1"] = FakeRun(state=FakeRunState(run_id="run-1"))

    with patch(
        "milknado.domains.execution.run_loop._completion.time.monotonic",
        return_value=142.0,
    ):
        context = CompletionContext(
            _scheduler(run_loop),
            graph,
            run_loop._executor,  # pyright: ignore[reportPrivateUsage]
            run_loop._loop,  # pyright: ignore[reportPrivateUsage]
            run_loop._input,  # pyright: ignore[reportPrivateUsage]
            run_loop._logs,  # pyright: ignore[reportPrivateUsage]
            run_loop._strict,  # pyright: ignore[reportPrivateUsage]
        )
        _ = handle_completion(context, "run-1", "stopped", "main")

    assert run_loop.state().terminal_runs[-1].duration_seconds == 42.0


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (RunStatus.COMPLETED, "run has completed"),
        (RunStatus.FAILED, "run has failed"),
    ],
)
def test_terminal_active_run_disables_all_controls(
    run_loop: RunLoop,
    graph: MikadoGraph,
    fake_loop: FakeLoop,
    status: RunStatus,
    reason: str,
) -> None:
    root = graph.add_node("ship controller")
    leaf = graph.add_node("build snapshots", parent_id=root.id)
    _runs(fake_loop)["run-1"] = FakeRun(state=FakeRunState(run_id="run-1", status=status))
    _scheduler(run_loop).admit_run("run-1", leaf.id, monotonic())

    active = run_loop.state().active_runs[0]

    assert active.actions.cancel_reason == reason
    assert active.actions.guidance_reason == reason
    assert active.actions.force_stop_reason == reason


def test_control_queue_applies_cancel_and_force_stop(
    run_loop: RunLoop,
    graph: MikadoGraph,
    fake_loop: FakeLoop,
) -> None:
    root = graph.add_node("ship controller")
    leaf = graph.add_node("stop worker", parent_id=root.id)
    graph.mark_running(leaf.id)
    _runs(fake_loop)["run-1"] = FakeRun()
    _scheduler(run_loop).admit_run("run-1", leaf.id, monotonic())

    run_loop.cancel("run-1")
    assert fake_loop.requested_stops == ["run-1"]
    assert run_loop.force_stop("run-1", timeout=2.5) is True
    assert len(fake_loop.force_stops) == 1
    run_id, remaining = fake_loop.force_stops[0]
    assert run_id == "run-1"
    assert remaining is not None
    assert 2.4 < remaining <= 2.5
    assert run_loop.state().active_runs[0].actions.force_stop_reason == (
        "force stop already requested"
    )

    run_loop.stop_scheduling()
    assert fake_loop.requested_stops == ["run-1", "run-1"]


def test_progress_snapshot_is_published_before_terminal_completion(
    run_loop: RunLoop,
    graph: MikadoGraph,
    config: ExecutionConfig,
    fake_loop: FakeLoop,
) -> None:
    root = graph.add_node("ship controller")
    _ = graph.add_node("build snapshots", parent_id=root.id)
    _progress_before_completion(fake_loop).append(
        ProgressEvent(run_id="run-1", work=1, total=2, message="building")
    )
    received: list[RunLoopState] = []
    run_loop.set_state_listener(received.append)
    _ = run_loop.run(config, "main")

    assert any(
        snapshot.active_runs and snapshot.active_runs[0].progress == "building"
        for snapshot in received
    )


def test_state_listener_failure_logs_listener_identity(
    run_loop: RunLoop,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def failing_listener(_state: RunLoopState) -> None:
        raise RuntimeError("listener failed")

    run_loop.set_state_listener(failing_listener)

    assert run_loop.queue_guidance("run-1", "use domain barrels") is True
    assert "failing_listener" in caplog.text


@pytest.fixture()
def run_loop(
    executor: Executor,
    graph: MikadoGraph,
    fake_loop: FakeLoop,
) -> RunLoop:
    return RunLoop(executor=executor, graph=graph, loop=fake_loop)


def test_initial_dispatch_respects_a_preexisting_scheduling_stop(
    run_loop: RunLoop,
    config: ExecutionConfig,
) -> None:
    dispatch = MagicMock()
    _set_attr(run_loop, "_dispatch_batch", dispatch)
    run_loop.admit_stop_scheduling()

    _ = _execute_run(run_loop, config, "main", concurrency_limit=1, timeout=1.0, interactive=False)

    dispatch.assert_not_called()


def test_initial_dispatch_drains_pending_controls_before_scheduling(
    run_loop: RunLoop,
    config: ExecutionConfig,
) -> None:
    dispatch = MagicMock()
    _set_attr(run_loop, "_dispatch_batch", dispatch)
    dispatch.return_value = (0, 0)
    _set_attr(run_loop, "_process_controls", run_loop.stop_scheduling)

    _ = _execute_run(run_loop, config, "main", concurrency_limit=1, timeout=1.0, interactive=False)

    dispatch.assert_not_called()


def test_completion_deadline_starts_before_the_first_short_control_poll(
    run_loop: RunLoop,
    graph: MikadoGraph,
    config: ExecutionConfig,
    fake_loop: FakeLoop,
) -> None:
    from milknado.domains.common.errors import CompletionTimeout

    node = graph.add_node("active")
    graph.mark_running(node.id)
    _scheduler(run_loop).admit_run("run-1", node.id, monotonic())
    _set_attr(run_loop, "_dispatch_if_scheduling_open", MagicMock(return_value=(0, 0)))
    _set_attr(run_loop, "_handle_completion_timeout", MagicMock(return_value=1))
    control_calls = 0

    def process_controls() -> None:
        nonlocal control_calls
        control_calls += 1
        if control_calls == 3:
            _ = _scheduler(run_loop).abandon_run("run-1")

    def short_poll_timeout(
        active_run_ids: set[str], timeout: float | None = None
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]:
        raise CompletionTimeout(waited_seconds=timeout or 0.0, active_run_ids=active_run_ids)

    _set_attr(fake_loop, "wait_for_next_completion", short_poll_timeout)
    _set_attr(run_loop, "_process_controls", process_controls)

    with patch(
        "milknado.domains.execution.run_loop.time.monotonic",
        side_effect=(100.0, 100.01),
    ):
        _ = _execute_run(
            run_loop,
            config,
            "main",
            concurrency_limit=1,
            timeout=1.0,
            interactive=False,
        )

    _mock(run_loop, "_handle_completion_timeout").assert_not_called()


def _run_on_fake_clock(
    run_loop: RunLoop,
    config: ExecutionConfig,
    fake_loop: FakeLoop,
    graph: MikadoGraph,
    *,
    waits: list[tuple[float, ProgressEvent | None]],
    with_controls: bool,
    dispatches: list[tuple[int, int]],
    retries: list[tuple[int, int]],
) -> MagicMock:
    """Run the loop on a fake clock; each wait advances it, the last wait ends the run.

    Returns the mocked ``_handle_completion_timeout``.
    """
    from milknado.domains.common.errors import CompletionTimeout

    node = graph.add_node("active")
    graph.mark_running(node.id)
    _scheduler(run_loop).admit_run("run-1", node.id, monotonic())
    _scheduler(run_loop).defer_capacity()
    queued = list(dispatches)

    def dispatch() -> tuple[int, int]:
        result = queued.pop(0)
        if not queued:
            _ = _scheduler(run_loop).plan_dispatch(1, strict=False)
        return result

    def dispatch_any(*_args: object) -> tuple[int, int]:
        return dispatch()

    _set_attr(
        run_loop,
        "_dispatch_if_scheduling_open",
        MagicMock(side_effect=dispatch_any),
    )
    _set_attr(run_loop, "_retry_deferred_if_due", MagicMock(side_effect=retries))
    _set_attr(run_loop, "_handle_completion_timeout", MagicMock(return_value=1))
    if with_controls:
        _set_attr(run_loop, "_process_controls", MagicMock())
    clock = [0.0]
    pending = list(waits)

    def wait(
        active_run_ids: set[str], timeout: float | None = None
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]:
        del timeout  # signature mirrors LoopPort; the fake ignores the timeout
        advance, event = pending.pop(0)
        clock[0] += advance
        if not pending:
            _ = _scheduler(run_loop).abandon_run("run-1")
        if event is not None:
            return "run-1", event
        raise CompletionTimeout(waited_seconds=advance, active_run_ids=active_run_ids)

    _set_attr(fake_loop, "wait_for_next_completion", wait)
    with patch(
        "milknado.domains.execution.run_loop.time.monotonic",
        side_effect=lambda: clock[0],
    ):
        _ = _execute_run(
            run_loop, config, "main", concurrency_limit=1, timeout=100.0, interactive=False
        )
    return _mock(run_loop, "_handle_completion_timeout")


def test_controller_retry_admit_restarts_the_completion_deadline(
    run_loop: RunLoop,
    graph: MikadoGraph,
    config: ExecutionConfig,
    fake_loop: FakeLoop,
) -> None:
    """A node admitted by a deferred retry gets a full timeout, not the leftover of A's."""
    handler = _run_on_fake_clock(
        run_loop,
        config,
        fake_loop,
        graph,
        waits=[(60.0, None), (60.0, None)],
        with_controls=True,
        dispatches=[(0, 0)],
        retries=[(1, 0), (0, 0)],
    )

    handler.assert_not_called()


def test_no_controls_retry_admit_restarts_the_completion_deadline(
    run_loop: RunLoop,
    graph: MikadoGraph,
    config: ExecutionConfig,
    fake_loop: FakeLoop,
) -> None:
    handler = _run_on_fake_clock(
        run_loop,
        config,
        fake_loop,
        graph,
        waits=[(60.0, None), (60.0, None)],
        with_controls=False,
        dispatches=[(0, 0), (1, 0), (0, 0)],
        retries=[],
    )

    handler.assert_not_called()


def test_no_controls_progress_restarts_the_deferred_completion_deadline(
    run_loop: RunLoop,
    graph: MikadoGraph,
    config: ExecutionConfig,
    fake_loop: FakeLoop,
) -> None:
    """Progress pushes the deadline out, as when each wait had its own full timeout."""
    progress = ProgressEvent(run_id="run-1", work=1, total=2, message="building")
    handler = _run_on_fake_clock(
        run_loop,
        config,
        fake_loop,
        graph,
        waits=[(90.0, progress), (60.0, None)],
        with_controls=False,
        dispatches=[(0, 0), (0, 0)],
        retries=[],
    )

    handler.assert_not_called()


def test_unset_completion_timeout_polls_controls_without_timing_out(
    run_loop: RunLoop,
    graph: MikadoGraph,
    config: ExecutionConfig,
    fake_loop: FakeLoop,
) -> None:
    from milknado.domains.common.errors import CompletionTimeout

    node = graph.add_node("active")
    graph.mark_running(node.id)
    _scheduler(run_loop).admit_run("run-1", node.id, monotonic())
    _set_attr(run_loop, "_dispatch_if_scheduling_open", MagicMock(return_value=(0, 0)))
    _set_attr(run_loop, "_handle_completion_timeout", MagicMock(return_value=1))
    observed_timeouts: list[float | None] = []
    control_calls = 0

    def process_controls() -> None:
        nonlocal control_calls
        control_calls += 1
        if control_calls >= 4:
            _ = _scheduler(run_loop).abandon_run("run-1")

    def short_poll_timeout(
        active_run_ids: set[str], timeout: float | None = None
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]:
        observed_timeouts.append(timeout)
        raise CompletionTimeout(waited_seconds=timeout or 0.0, active_run_ids=active_run_ids)

    _set_attr(fake_loop, "wait_for_next_completion", short_poll_timeout)
    _set_attr(run_loop, "_process_controls", process_controls)

    _ = _execute_run(
        run_loop, config, "main", concurrency_limit=1, timeout=None, interactive=False
    )

    _mock(run_loop, "_handle_completion_timeout").assert_not_called()
    assert observed_timeouts
    assert all(polled == 0.1 for polled in observed_timeouts)


def test_stop_latched_before_run_skips_terminal_spec_verification(
    run_loop: RunLoop,
    graph: MikadoGraph,
    config: ExecutionConfig,
) -> None:
    root = graph.add_node("root goal")
    run_loop.stop_scheduling()

    result = run_loop.run(config, "main", spec_text="spec: do the thing")

    assert result.verify_outcome is None
    root_node = graph.get_node(root.id)
    assert root_node is not None
    assert root_node.status == NodeStatus.PENDING


def test_full_brief_remains_available_after_worker_completion(
    run_loop: RunLoop, graph: MikadoGraph, config: ExecutionConfig
) -> None:
    root = graph.add_node("Ship session views")
    brief = (
        "US-12: Preserve the complete task brief while the node list uses a short title\n\n"
        "Acceptance: Keep the selected worktree, input history, and verification rules visible."
    )
    node = graph.add_node(brief, parent_id=root.id)
    observed: list[RunLoopState] = []
    run_loop.set_state_listener(observed.append)

    result = run_loop.run(config, "main")

    assert result.completed_total == 1
    assert brief in [
        active.description for snapshot in observed for active in snapshot.active_runs
    ]
    terminal = run_loop.state().terminal_runs[0]
    assert (terminal.node_id, terminal.description, terminal.status) == (
        node.id,
        brief,
        RunStatus.COMPLETED,
    )


class TestRunLoopSingleNode:
    def test_solo_root_not_marked_done_when_undecomposed(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        # Root is never dispatched as a work node. With no non-root nodes, the
        # goal was never decomposed, so nothing was done — the structural
        # fallback must not vacuously mark it complete.
        _ = graph.add_node("root goal")
        result = run_loop.run(config, "main")

        assert result.dispatched_total == 0
        assert result.completed_total == 0
        assert result.failed_total == 0
        assert result.root_done is False

    def test_solo_root_marked_done_without_spec_after_leaf_completes(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root goal")
        _ = graph.add_node("leaf", parent_id=root.id)
        _ = run_loop.run(config, "main")

        root_node = graph.get_node(root.id)
        assert root_node is not None
        assert root_node.status == NodeStatus.DONE


class TestRunLoopParentChild:
    def test_dispatches_leaf_not_root(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        # Root is excluded from dispatch; only the leaf is dispatched. Without
        # spec_text, the structural fallback completes the root once the leaf is done.
        root = graph.add_node("root")
        leaf = graph.add_node("leaf", parent_id=root.id)

        result = run_loop.run(config, "main")

        assert result.dispatched_total == 1
        assert result.completed_total == 1
        assert result.root_done is True
        leaf_node = graph.get_node(leaf.id)
        root_node = graph.get_node(root.id)
        assert leaf_node is not None and leaf_node.status == NodeStatus.DONE
        assert root_node is not None and root_node.status == NodeStatus.DONE

    def test_leaf_done_root_marked_done_without_spec(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        _ = graph.add_node("leaf", parent_id=root.id)

        _ = run_loop.run(config, "main")

        root_node = graph.get_node(root.id)
        assert root_node is not None
        assert root_node.status == NodeStatus.DONE


class TestRunLoopStoppedOutcome:
    def test_stopped_run_retains_output_guidance_and_telemetry_without_redispatch(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        loop_adapter.set_run_stopped("run-1")
        loop_adapter.output["run-1"] = ["last worker output"]
        loop_adapter.guidance["run-1"] = ("not delivered",)
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)
        root = graph.add_node("root")
        leaf = graph.add_node("leaf", parent_id=root.id)

        with patch("milknado.domains.execution.run_loop._logger.info") as log_info:
            result = loop.run(config, "main", interactive=False)

        node = graph.get_node(leaf.id)
        assert node is not None
        assert node.status is NodeStatus.PENDING
        assert (result.dispatched_total, result.completed_total, result.failed_total) == (1, 0, 0)
        assert _scheduler(loop).view().stopped_nodes == {leaf.id}
        assert _scheduler(loop).view().active == ()
        state = loop.state()
        assert state.stopped == 1
        assert state.available == 0
        assert len(state.terminal_runs) == 1
        terminal = state.terminal_runs[0]
        real_run_id = _ordinal_to_run_id(loop_adapter)["run-1"]
        assert terminal.run_id == real_run_id
        assert terminal.status is RunStatus.STOPPED
        assert terminal.output == ("last worker output",)
        assert terminal.pending_guidance == ("not delivered",)
        assert any(
            call.args[0] == "node_stopped node_id=%d run_id=%s duration=%.1fs"
            and call.args[1:3] == (leaf.id, real_run_id)
            for call in log_info.call_args_list
        )
        telemetry = [
            json.loads(cast(str, call.args[1]))
            for call in log_info.call_args_list
            if call.args[0] == "FINAL_TELEMETRY %s"
        ]
        assert telemetry == [
            {
                "dispatched": 1,
                "completed": 0,
                "failed": 0,
                "stopped": 1,
                "conflicts": 0,
                "root_done": False,
                "strict_exit": False,
                "interrupted": False,
            }
        ]

    def test_stopped_terminal_history_is_bounded_to_the_newest_twenty(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        for index in range(1, 22):
            loop_adapter.set_run_stopped(f"run-{index}")
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)
        root = graph.add_node("root")
        for index in range(21):
            _ = graph.add_node(f"leaf-{index}", parent_id=root.id)
        _ = loop.run(config, "main", interactive=False)

        state = loop.state()
        assert state.stopped == 21
        assert len(state.terminal_runs) == 20
        assert tuple(run.run_id for run in state.terminal_runs) == tuple(
            _ordinal_to_run_id(loop_adapter)[f"run-{index}"] for index in range(2, 22)
        )


class TestRunLoopParallelLeaves:
    def test_dispatches_parallel_leaves_not_root(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        _ = graph.add_node("leaf-a", parent_id=root.id)
        _ = graph.add_node("leaf-b", parent_id=root.id)

        result = run_loop.run(config, "main")

        assert result.dispatched_total == 2
        assert result.completed_total == 2
        # Root not dispatched; structural fallback completes it once both leaves are done.
        assert result.root_done is True


class TestRunLoopConcurrencyLimit:
    def test_respects_concurrency_limit(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        _ = graph.add_node("a", parent_id=root.id)
        _ = graph.add_node("b", parent_id=root.id)
        _ = graph.add_node("c", parent_id=root.id)

        result = run_loop.run(config, "main", concurrency_limit=2)

        # 3 leaves dispatched, root excluded, then structurally completed.
        assert result.dispatched_total == 3
        assert result.completed_total == 3
        assert result.root_done is True

    def test_capacity_refusal_leaves_task_pending_without_failure(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        executor: Executor,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        task = graph.add_node("task", parent_id=root.id)
        with patch.object(executor, "dispatch", side_effect=ConcurrencyLimitReached(1, 1)):
            dispatched, failed = _dispatch_batch(run_loop, config, 1)

        assert (dispatched, failed) == (0, 0)
        node = graph.get_node(task.id)
        assert node is not None and node.status is NodeStatus.PENDING
        assert not _scheduler(run_loop).view().active

    def test_post_start_dispatch_error_keeps_cli_worker_owned(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        executor: Executor,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        task = graph.add_node("task", parent_id=root.id)
        with (
            patch.object(
                executor, "dispatch", side_effect=PreservedWorkerRun(task.id, "started-run")
            ),
            patch.object(executor, "force_stop_run", side_effect=[False, True]) as stop,
            patch.object(executor, "fail") as fail,
            patch("milknado.domains.execution.run_loop.time.sleep"),
        ):
            assert _dispatch_batch(run_loop, config, 1) == (0, 1)
        assert stop.call_count == 2
        fail.assert_called_once_with(task.id)

    def test_raced_claim_does_not_fail_other_owner(
        self, run_loop: RunLoop, graph: MikadoGraph, config: ExecutionConfig
    ) -> None:
        root = graph.add_node("root")
        task = graph.add_node("task", parent_id=root.id)
        claim = graph.claim_node

        def other_claim(*_args: object, **_kwargs: object) -> bool:
            assert claim(task.id, "other-owner", now="2026-01-01T00:00:00Z")
            return False

        with patch.object(graph, "claim_node", side_effect=other_claim):
            assert _dispatch_batch(run_loop, config, 1) == (0, 0)
        node = graph.get_node(task.id)
        assert node is not None
        assert node.status is NodeStatus.RUNNING
        assert node.run_id == "other-owner"
        assert not _scheduler(run_loop).view().active

    def test_invalid_worktree_pattern_releases_claim(
        self, graph: MikadoGraph, executor: Executor, config: ExecutionConfig
    ) -> None:
        root = graph.add_node("root")
        task = graph.add_node("task", parent_id=root.id)

        with pytest.raises(ValueError, match="outside project_root"):
            _ = executor.dispatch(task.id, replace(config, worktree_pattern="../outside"))

        node = graph.get_node(task.id)
        assert node is not None
        assert node.status is NodeStatus.PENDING
        assert node.run_id is None

    def test_detached_parent_claim_blocks_scheduler_on_same_graph(
        self,
        tmp_path: Path,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_loop: FakeLoop,
        fake_crg: FakeCrg,
    ) -> None:
        graph = MikadoGraph(tmp_path / "capacity.db", concurrency_limit=1)
        try:
            root = graph.add_node("root")
            busy = graph.add_node("detached", parent_id=root.id)
            waiting = graph.add_node("scheduled", parent_id=root.id)
            graph.claim_node_for_dispatch(busy.id, "detached-parent", now="2026-01-01T00:00:00Z")
            executor = Executor(graph=graph, git=fake_git, loop=fake_loop, crg=fake_crg)
            driver = RunLoop(executor=executor, graph=graph, loop=fake_loop)

            assert _dispatch_batch(driver, config, 4) == (0, 0)
            node = graph.get_node(waiting.id)
            assert node is not None and node.status is NodeStatus.PENDING
            assert graph.mark_terminal(busy.id, "detached-parent", NodeStatus.DONE)
            assert _dispatch_batch(driver, config, 4) == (1, 0)
            node = graph.get_node(waiting.id)
            assert node is not None and node.status is NodeStatus.RUNNING
        finally:
            graph.close()

    def test_single_node_driver_leaves_sibling_and_root_untouched(
        self, run_loop: RunLoop, graph: MikadoGraph, config: ExecutionConfig
    ) -> None:
        root = graph.add_node("root")
        selected = graph.add_node("selected", parent_id=root.id)
        sibling = graph.add_node("sibling", parent_id=root.id)

        outcome = run_loop.run_node(selected.id, config, "main", 30.0)

        assert outcome.success is True
        selected_node = graph.get_node(selected.id)
        sibling_node = graph.get_node(sibling.id)
        root_node = graph.get_node(root.id)
        assert selected_node is not None and selected_node.status is NodeStatus.DONE
        assert sibling_node is not None and sibling_node.status is NodeStatus.PENDING
        assert root_node is not None and root_node.status is NodeStatus.PENDING


class TestRunLoopFailure:
    def test_failed_leaf_marks_leaf_failed(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        # Root is not dispatched; only the leaf is. Set the leaf's run to fail.
        loop_adapter = FakeLoop()
        _success(loop_adapter)["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root")
        leaf = graph.add_node("will fail", parent_id=root.id)
        result = loop.run(config, "main")

        assert result.failed_total == 1
        assert result.root_done is False
        leaf_node = graph.get_node(leaf.id)
        assert leaf_node is not None
        assert leaf_node.status == NodeStatus.FAILED

    def test_failed_leaf_blocks_parent(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        _success(loop_adapter)["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root")
        _ = graph.add_node("leaf", parent_id=root.id)
        result = loop.run(config, "main")

        assert result.failed_total == 1
        assert result.root_done is False
        root_node = graph.get_node(root.id)
        assert root_node is not None
        assert root_node.status == NodeStatus.PENDING


class TestRunLoopResult:
    def test_empty_graph_returns_immediately(
        self,
        run_loop: RunLoop,
        config: ExecutionConfig,
    ) -> None:
        result = run_loop.run(config, "main")

        assert result.dispatched_total == 0
        assert result.completed_total == 0
        assert result.root_done is False


class TestRunLoopDispatchFailure:
    def test_dispatch_failure_marks_leaf_failed(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()

        def fail_generate(*_args: object, **_kwargs: object) -> Path:
            raise RuntimeError("loop exploded")

        loop_adapter.generate_loop_md = fail_generate
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root")
        leaf = graph.add_node("doomed", parent_id=root.id)
        result = loop.run(config, "main")

        assert result.dispatched_total == 0
        assert result.root_done is False
        leaf_node = graph.get_node(leaf.id)
        assert leaf_node is not None
        assert leaf_node.status == NodeStatus.FAILED

    def test_missing_quality_gates_are_visible_in_dispatch_failure(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        loop_adapter = FakeLoop()
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)
        root = graph.add_node("root")
        leaf = graph.add_node("missing gates", parent_id=root.id)

        result = loop.run(replace(config, quality_gates=None), "main")

        assert result.dispatched_total == 0
        assert result.failed_total == 1
        leaf_node = graph.get_node(leaf.id)
        assert leaf_node is not None
        assert leaf_node.status == NodeStatus.FAILED
        assert NO_GATES_CONFIGURED_MESSAGE in caplog.text

    def test_dispatch_failure_does_not_crash_loop(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        call_count = 0
        original_generate = loop_adapter.generate_loop_md

        def fail_first_only(
            brief: str,
            quality_gates: tuple[Gate, ...] | None,
            output_path: Path,
            prior_findings: str = "",
            findings_round: int | None = None,
        ) -> Path:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("loop exploded")
            return original_generate(
                brief, quality_gates, output_path, prior_findings, findings_round
            )

        loop_adapter.generate_loop_md = fail_first_only  # type: ignore
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root")
        _ = graph.add_node("doomed-leaf", parent_id=root.id)
        _ = graph.add_node("good-leaf", parent_id=root.id)

        result = loop.run(config, "main")

        assert result.root_done is False
        good_leaf = graph.get_node(3)
        assert good_leaf is not None
        assert good_leaf.status == NodeStatus.DONE


class TestRunLoopRebaseConflicts:
    def test_rebase_conflict_details_surfaced_in_result(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_crg: FakeCrg,
    ) -> None:
        fake_git = FakeGit()
        fake_git.rebase_result = RebaseResult(
            success=False,
            conflicting_files=("src/models.py", "src/views.py"),
            detail="CONFLICT (content): Merge conflict in src/models.py",
        )
        loop_adapter = FakeLoop()
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root")
        leaf = graph.add_node("conflicting node", parent_id=root.id)
        result = loop.run(config, "main")

        assert result.failed_total == 1
        assert len(result.rebase_conflicts) == 1
        conflict = result.rebase_conflicts[0]
        assert conflict.node_id == leaf.id
        assert conflict.conflicting_files == ("src/models.py", "src/views.py")
        assert "Merge conflict" in conflict.detail

    def test_no_conflicts_yields_empty_tuple(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        _ = graph.add_node("clean node", parent_id=root.id)
        result = run_loop.run(config, "main")

        assert result.rebase_conflicts == ()


class TestRunLoopFileConflicts:
    def test_serializes_conflicting_nodes(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_loop: FakeLoop,
        fake_crg: FakeCrg,
    ) -> None:
        executor = Executor(
            graph=graph,
            git=fake_git,
            loop=fake_loop,
            crg=fake_crg,
        )
        loop = RunLoop(executor=executor, graph=graph, loop=fake_loop)

        root = graph.add_node("root")
        a = graph.add_node("a", parent_id=root.id)
        b = graph.add_node("b", parent_id=root.id)
        graph.files.claim(a.id, ["shared.py"])
        graph.files.claim(b.id, ["shared.py"])

        result = loop.run(config, "main")

        # 2 leaves dispatched (serialized due to file conflict), root excluded,
        # then structurally completed.
        assert result.dispatched_total == 2
        assert result.completed_total == 2
        assert result.root_done is True


class TestStrictDrain:
    def test_no_new_dispatch_after_failure_in_flight_completes(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        _success(loop_adapter)["run-1"] = False  # leaf-a fails, leaf-b (run-2) succeeds

        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root")
        _ = graph.add_node("leaf-a", parent_id=root.id)
        _ = graph.add_node("leaf-b", parent_id=root.id)

        result = loop.run(config, "main", strict=True)

        # Both leaves dispatched; root never dispatched after strict-failure
        assert result.dispatched_total == 2
        assert result.failed_total == 1
        assert result.completed_total == 1
        assert result.strict_exit is True
        assert result.root_done is False

    def test_none_gates_preflight_strict_halts_after_first_fail(
        self,
        graph: MikadoGraph,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
        tmp_path: Path,
    ) -> None:
        """Strict mode + None quality_gates: preflight fails the first node and
        sets failure_triggered=True, halting dispatch of the second node."""
        none_gates_config = ExecutionConfig(
            execution_agent="claude",
            quality_gates=None,  # fail-closed
            worktree_pattern="milknado-{node_id}-{slug}",
            project_root=tmp_path,
        )

        loop_adapter = FakeLoop()
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root")
        _ = graph.add_node("leaf-a", parent_id=root.id)
        _ = graph.add_node("leaf-b", parent_id=root.id)

        result = loop.run(none_gates_config, "main", strict=True)

        # Preflight on leaf-a fails → strict stops after 1 failure; leaf-b never dispatched
        assert result.failed_total >= 1
        assert result.strict_exit is True


class TestProtectedBranchGuard:
    def test_protected_branch_refused_before_log_created(self, tmp_path: Path) -> None:
        from milknado.app.run import ProtectedBranchRefusal, ensure_dispatch_allowed
        from milknado.domains.common.config import MilknadoConfig

        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            protected_branches=("main", "master"),
        )

        with pytest.raises(ProtectedBranchRefusal, match="protected branch"):
            ensure_dispatch_allowed(cfg, "main", allow_protected=False)
        assert not any((tmp_path / ".milknado").glob("run-*.log"))

    def test_protected_branch_with_allow_protected_does_not_raise(self, tmp_path: Path) -> None:
        from milknado.app.run import ensure_dispatch_allowed
        from milknado.domains.common.config import MilknadoConfig

        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            protected_branches=("main", "master"),
        )

        ensure_dispatch_allowed(cfg, "main", allow_protected=True)

    def test_feature_branch_does_not_raise(self, tmp_path: Path) -> None:
        from milknado.app.run import ensure_dispatch_allowed
        from milknado.domains.common.config import MilknadoConfig

        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            protected_branches=("main", "master"),
        )

        ensure_dispatch_allowed(cfg, "feature-x", allow_protected=False)

    def test_second_protected_branch_is_refused(self, tmp_path: Path) -> None:
        from milknado.app.run import ProtectedBranchRefusal, ensure_dispatch_allowed
        from milknado.domains.common.config import MilknadoConfig

        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            protected_branches=("main", "master"),
        )

        with pytest.raises(ProtectedBranchRefusal, match="protected branch"):
            ensure_dispatch_allowed(cfg, "master", allow_protected=False)

    def test_detached_head_refused_even_with_allow_protected(self, tmp_path: Path) -> None:
        from milknado.app.run import ProtectedBranchRefusal, ensure_dispatch_allowed
        from milknado.domains.common.config import MilknadoConfig

        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            protected_branches=("main", "master"),
        )

        for branch in ("HEAD", ""):
            with pytest.raises(ProtectedBranchRefusal, match="detached branch"):
                ensure_dispatch_allowed(cfg, branch, allow_protected=True)


class TestOrphanCleanupTransientRetries:
    def test_ensure_clean_worktree_called_each_attempt_no_stale_accumulation(
        self,
        graph: MikadoGraph,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
        tmp_path: Path,
    ) -> None:
        from milknado.domains.common.errors import TransientDispatchError
        from milknado.domains.execution.executor import Executor as _Executor

        loop = FakeLoop()
        call_count = 0
        original_generate = loop.generate_loop_md

        def fail_thrice(
            brief: str,
            quality_gates: tuple[Gate, ...] | None,
            output_path: Path,
            prior_findings: str = "",
            findings_round: int | None = None,
        ) -> Path:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise TransientDispatchError("rate limited")
            return original_generate(
                brief, quality_gates, output_path, prior_findings, findings_round
            )

        loop.generate_loop_md = fail_thrice  # type: ignore

        executor = _Executor(graph=graph, git=fake_git, loop=loop, crg=fake_crg)

        clean_calls: list[int] = []
        worktree_sizes: list[int] = []
        original_ensure = _worktree_manager(executor).ensure_clean

        def tracked_ensure(node_id: int) -> None:
            worktree_sizes.append(len(_managed_worktrees(executor)))
            clean_calls.append(node_id)
            return original_ensure(node_id)

        _worktree_manager(executor).ensure_clean = tracked_ensure

        retry_config = ExecutionConfig(
            execution_agent="claude",
            quality_gates=(Gate(command="uv run pytest"),),
            worktree_pattern="milknado-{node_id}-{slug}",
            project_root=tmp_path,
            dispatch_max_retries=3,
            dispatch_backoff_seconds=0.0,
        )
        _ = graph.add_node("transient node")

        _ = executor.dispatch(1, retry_config)

        # Called once at the start of each _dispatch_once attempt (3 fail + 1 success = 4)
        assert len(clean_calls) == 4
        # At no point when ensure_clean fires are there stale entries
        assert all(sz == 0 for sz in worktree_sizes)
        # Final state: successful dispatch recorded, not cleared
        assert 1 in _managed_worktrees(executor)


class TestRootCompletionViaVerifySpec:
    def test_root_marked_done_by_verify_spec_after_all_leaves_done(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root goal")
        leaf = graph.add_node("leaf", parent_id=root.id)

        result = loop.run(config, "main", spec_text="spec: do the thing")

        assert result.root_done is True
        root_node = graph.get_node(root.id)
        leaf_node = graph.get_node(leaf.id)
        assert root_node is not None and root_node.status == NodeStatus.DONE
        assert leaf_node is not None and leaf_node.status == NodeStatus.DONE
        # Root was NOT dispatched — only the leaf was.
        assert result.dispatched_total == 1

    def test_root_not_dispatched_during_run(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        dispatched_ids: list[int] = []
        loop_adapter = FakeLoop()
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        original_dispatch = executor.dispatch

        def tracking_dispatch(node_id: int, cfg: ExecutionConfig) -> DispatchResult:
            dispatched_ids.append(node_id)
            return original_dispatch(node_id, cfg)

        _set_attr(executor, "dispatch", tracking_dispatch)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root goal")
        _ = graph.add_node("leaf", parent_id=root.id)
        _ = loop.run(config, "main", spec_text="spec: do the thing")

        assert root.id not in dispatched_ids

    def test_root_stays_pending_when_leaves_fail(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        _success(loop_adapter)["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root goal")
        _ = graph.add_node("leaf", parent_id=root.id)

        result = loop.run(config, "main", spec_text="spec: do the thing")

        assert result.root_done is False
        root_node = graph.get_node(root.id)
        assert root_node is not None
        assert root_node.status == NodeStatus.PENDING


class TestRootCompletionStructuralFallback:
    def test_root_marked_done_without_spec_when_all_leaves_done(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        verify_calls: list[tuple[str, str]] = []
        original_verify_spec = loop_adapter.verify_spec

        def tracking_verify_spec(spec_text: str, graph_state: str) -> VerifySpecResult:
            verify_calls.append((spec_text, graph_state))
            return original_verify_spec(spec_text, graph_state)

        _set_attr(loop_adapter, "verify_spec", tracking_verify_spec)

        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root goal")
        leaf = graph.add_node("leaf", parent_id=root.id)

        result = loop.run(config, "main")

        assert result.root_done is True
        assert result.verify_outcome is None
        root_node = graph.get_node(root.id)
        leaf_node = graph.get_node(leaf.id)
        assert root_node is not None and root_node.status == NodeStatus.DONE
        assert leaf_node is not None and leaf_node.status == NodeStatus.DONE
        assert verify_calls == []

    def test_missing_verifier_keeps_completed_leaves_pending(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        from milknado.domains.planning.planner import Planner

        loop_adapter = LoopAdapter()
        planner = MagicMock(spec=Planner)
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter, planner=planner)

        root = graph.add_node("root goal")
        leaf = graph.add_node("leaf", parent_id=root.id)
        graph.mark_running(leaf.id)
        graph.mark_done(leaf.id)

        result = loop.run(config, "main", spec_text="spec: do the thing")

        assert result.root_done is False
        assert result.verify_outcome is not None
        assert result.verify_outcome.done is False
        assert result.verify_outcome.goal_delta == (
            "verification unavailable: no agent configured"
        )
        root_node = graph.get_node(root.id)
        assert root_node is not None
        assert root_node.status == NodeStatus.PENDING
        _mock_attr(planner, "replan_with_delta").assert_not_called()

    def test_root_stays_pending_without_spec_when_leaf_not_done(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        _success(loop_adapter)["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root goal")
        _ = graph.add_node("leaf", parent_id=root.id)

        result = loop.run(config, "main")

        assert result.root_done is False
        root_node = graph.get_node(root.id)
        assert root_node is not None
        assert root_node.status == NodeStatus.PENDING

    def test_root_stays_pending_without_spec_when_failure_triggered(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        _success(loop_adapter)["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root goal")
        leaf = graph.add_node("leaf", parent_id=root.id)

        result = loop.run(config, "main", strict=True)

        assert result.root_done is False
        root_node = graph.get_node(root.id)
        leaf_node = graph.get_node(leaf.id)
        assert root_node is not None and root_node.status == NodeStatus.PENDING
        assert leaf_node is not None and leaf_node.status != NodeStatus.DONE

    def test_bare_root_not_marked_done_without_children(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root goal")

        result = loop.run(config, "main")

        assert result.root_done is False
        assert result.dispatched_total == 0
        root_node = graph.get_node(root.id)
        assert root_node is not None and root_node.status == NodeStatus.PENDING


# ---------------------------------------------------------------------------
# LoopAdapter.create_run passes log_dir
# ---------------------------------------------------------------------------


class _RunConfig(Protocol):
    log_dir: Path


class TestLoopAdapterLogDir:
    def test_create_run_passes_log_dir_under_worktree(
        self,
        tmp_path: Path,
    ) -> None:
        loop_dir = tmp_path / "wt-node-1"
        loop_dir.mkdir()
        loop_file = loop_dir / "loop.md"
        _ = loop_file.write_text("# task", encoding="utf-8")

        captured_configs: list[_RunConfig] = []

        def fake_create_run(
            config: _RunConfig, *, emitter: object, run_id: str | None = None
        ) -> FakeRun:
            assert emitter is attrgetter("_emitter")(adapter)
            captured_configs.append(config)
            return FakeRun(state=FakeRunState(run_id=run_id or "run-test"))

        from milknado.adapters.loop import LoopAdapter

        adapter = LoopAdapter(agent="claude")

        with patch.object(
            attrgetter("_manager")(adapter), "create_run", side_effect=fake_create_run
        ):
            _ = adapter.create_run(
                agent="claude",
                loop_dir=loop_dir,
                loop_file=loop_file,
                quality_gates=(),
                project_root=None,
            )

        assert len(captured_configs) == 1
        cfg = captured_configs[0]
        assert cfg.log_dir == loop_dir / ".loop-logs"


# ---------------------------------------------------------------------------
# Coverage helpers: state.py missing branches
# ---------------------------------------------------------------------------


class TestSummarizeDescriptionTruncation:
    def test_long_description_is_truncated(self) -> None:
        long_text = "A" * 100
        result = summarize_description(long_text, 80)
        assert len(result) <= 80
        assert result.endswith("…")

    def test_short_description_unchanged(self) -> None:
        result = summarize_description("short task", 80)
        assert result == "short task"


def test_execution_domain_imports_without_rich() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['rich'] = None; import milknado.domains.execution",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Coverage helpers: __init__.py missing branches
# ---------------------------------------------------------------------------


class TestHandleCompletionTimeout:
    def test_default_wait_has_no_wall_clock_limit(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        _set_attr(
            loop_adapter,
            "wait_for_next_completion",
            MagicMock(wraps=loop_adapter.wait_for_next_completion),
        )
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)
        root = graph.add_node("root")
        _ = graph.add_node("slow-leaf", parent_id=root.id)

        result = loop.run(config, "main")

        assert result.completed_total == 1
        wait_kwargs = _call_kwargs(_mock_attr(loop_adapter, "wait_for_next_completion"))
        assert wait_kwargs["timeout"] is None

    def test_configured_wait_keeps_explicit_timeout(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        from milknado.domains.common.config import MilknadoConfig

        loop_adapter = FakeLoop()
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        milknado_config = MilknadoConfig(completion_timeout_seconds=60.0)
        loop = RunLoop(
            executor=executor,
            graph=graph,
            loop=loop_adapter,
            config=milknado_config,
        )

        with patch.object(
            loop,
            "_execute_run",
            return_value=(0, 0, 0, [], False),
        ) as execute_run:
            _ = loop.run(config, "main")
        execute_run.assert_called_once_with(config, "main", 4, 60.0, True)

    def test_timeout_marks_active_nodes_failed(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        from milknado.domains.common.errors import CompletionTimeout

        loop_adapter = FakeLoop()

        # Make wait_for_next_completion raise CompletionTimeout
        def raise_timeout(
            active_run_ids: set[str], timeout: float | None = None
        ) -> tuple[str, object]:
            _ = timeout
            raise CompletionTimeout(waited_seconds=60.0, active_run_ids=active_run_ids)

        _set_attr(loop_adapter, "wait_for_next_completion", raise_timeout)

        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root")
        _ = graph.add_node("slow-leaf", parent_id=root.id)

        result = loop.run(config, "main")

        assert result.failed_total == 1
        assert result.root_done is False

    def test_timeout_strict_sets_failure_triggered(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        from milknado.domains.common.errors import CompletionTimeout

        loop_adapter = FakeLoop()
        first_call = [True]

        def raise_timeout(
            active_run_ids: set[str], timeout: float | None = None
        ) -> tuple[str, object]:
            _ = timeout
            if first_call[0]:
                first_call[0] = False
                raise CompletionTimeout(waited_seconds=60.0, active_run_ids=active_run_ids)
            return next(iter(active_run_ids)), True

        _set_attr(loop_adapter, "wait_for_next_completion", raise_timeout)

        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root")
        _ = graph.add_node("slow-leaf", parent_id=root.id)

        result = loop.run(config, "main", strict=True)

        assert result.strict_exit is True

    def test_timeout_preserves_active_ownership_until_worker_exits(
        self,
        graph: MikadoGraph,
    ) -> None:
        from milknado.domains.common.errors import CompletionTimeout

        executor = MagicMock()
        executor.force_stop_run = MagicMock(return_value=False)
        loop_adapter = FakeLoop()
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)
        _scheduler(loop).admit_run("run-1", 7, monotonic())

        failed = _handle_completion_timeout(
            loop, CompletionTimeout(waited_seconds=60.0, active_run_ids={"run-1"})
        )

        assert failed == 0
        assert [(run.run_id, run.node_id) for run in _scheduler(loop).view().active] == [
            ("run-1", 7)
        ]
        _mock_attr(executor, "fail").assert_not_called()
        _mock_attr(executor, "force_stop_run").assert_called_once_with("run-1", timeout=10.0)


class TestVerifySpecGapsPath:
    def test_gaps_outcome_calls_replan(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        from milknado.domains.common.protocols import VerifySpecResult

        loop_adapter = FakeLoop()

        def gaps_verify(spec_text: str, graph_state: str) -> VerifySpecResult:
            _ = (spec_text, graph_state)
            return VerifySpecResult(outcome="gaps", goal_delta="add feature X")

        loop_adapter.verify_spec = gaps_verify

        from milknado.domains.planning.planner import Planner

        planner = MagicMock(spec=Planner)
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter, planner=planner)

        root = graph.add_node("root")
        _ = graph.add_node("leaf", parent_id=root.id)
        _ = loop.run(config, "main", spec_text="spec text")

        _mock_attr(planner, "replan_with_delta").assert_called_once()

    def test_root_already_done_skips_verify(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        verify_calls: list[tuple[object, ...]] = []

        def done_verify(spec_text: str, graph_state: str) -> VerifySpecResult:
            verify_calls.append((spec_text, graph_state))
            return VerifySpecResult(outcome="done")

        loop_adapter.verify_spec = done_verify

        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root")
        graph.mark_running(root.id)
        graph.mark_done(root.id)
        _ = loop.run(config, "main", spec_text="spec text")

        assert len(verify_calls) == 0


class TestDispatchBatchConcurrencyFull:
    def test_no_dispatch_when_at_limit(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()
        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root")
        _ = graph.add_node("leaf-a", parent_id=root.id)
        _ = graph.add_node("leaf-b", parent_id=root.id)

        # Limit=1, two leaves pending — only first dispatched; second on next iteration
        result = loop.run(config, "main", concurrency_limit=1)

        assert result.dispatched_total == 2
        assert result.completed_total == 2


class TestKeyboardInterrupt:
    def test_keyboard_interrupt_propagates(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        loop_adapter = FakeLoop()

        def raise_interrupt(
            active_run_ids: set[str], timeout: float | None = None
        ) -> tuple[str, object]:
            _ = timeout
            _ = active_run_ids
            raise KeyboardInterrupt

        _set_attr(loop_adapter, "wait_for_next_completion", raise_interrupt)

        executor = Executor(graph=graph, git=fake_git, loop=loop_adapter, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)

        root = graph.add_node("root")
        _ = graph.add_node("leaf", parent_id=root.id)

        with pytest.raises(KeyboardInterrupt):
            _ = loop.run(config, "main")


class TestDispatchBatchDirectGuards:
    def test_strict_failure_triggered_returns_zero(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_loop: FakeLoop,
    ) -> None:

        loop = RunLoop(executor=executor, graph=graph, loop=fake_loop)
        _set_attr(loop, "_strict", True)
        _scheduler(loop).trigger_failure()

        result = _dispatch_batch(loop, config, 4)

        assert result == (0, 0)

    def test_no_capacity_returns_zero(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_loop: FakeLoop,
    ) -> None:

        loop = RunLoop(executor=executor, graph=graph, loop=fake_loop)
        # Fill active to the limit
        for node_id in range(1, 5):
            _scheduler(loop).admit_run(f"run-{node_id}", node_id, monotonic())

        result = _dispatch_batch(loop, config, 4)

        assert result == (0, 0)

    def test_dispatch_exception_increments_failed(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_loop: FakeLoop,
    ) -> None:
        from unittest.mock import MagicMock

        executor = MagicMock()
        _mock_attr(executor, "dispatch").side_effect = RuntimeError("boom")
        root = graph.add_node("root")
        leaf = graph.add_node("failing-leaf", parent_id=root.id)

        loop = RunLoop(executor=executor, graph=graph, loop=fake_loop)
        dispatched, failed = _dispatch_batch(loop, config, 4)

        assert dispatched == 0
        assert failed == 1
        assert f"✗ dispatch node {leaf.id}" in loop.state().event_lines[-1]

    def test_strict_mode_breaks_on_dispatch_exception(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_loop: FakeLoop,
    ) -> None:
        from unittest.mock import MagicMock

        executor = MagicMock()
        _mock_attr(executor, "dispatch").side_effect = RuntimeError("boom")
        root = graph.add_node("root")
        _ = graph.add_node("leaf-a", parent_id=root.id)
        _ = graph.add_node("leaf-b", parent_id=root.id)

        loop = RunLoop(executor=executor, graph=graph, loop=fake_loop)
        _set_attr(loop, "_strict", True)
        _ = _dispatch_batch(loop, config, 4)

        assert _mock_attr(executor, "dispatch").call_count == 1
        assert _scheduler(loop).view().failure_triggered is True


class TestDispatchBatchFlavoredGates:
    """AC6: a flavored node with quality_gates=() propagates empty gates to the executor."""

    def test_research_node_dispatches_with_empty_quality_gates(
        self,
        graph: MikadoGraph,
        fake_loop: FakeLoop,
        config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        from unittest.mock import MagicMock

        from milknado.domains.common.config import FlavorOverride, MilknadoConfig
        from milknado.domains.execution.run_loop import RunLoop

        milknado_cfg = MilknadoConfig(
            agent_family="claude",
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            flavors={
                "research": FlavorOverride(quality_gates=(), brief_prepend="Research only."),
            },
        )

        root = graph.add_node("root")
        _ = graph.add_node(
            "research leaf",
            parent_id=root.id,
            spec=NodeSpec(flavor="research"),
        )

        captured: list[ExecutionConfig] = []

        executor = MagicMock()

        def dispatch(node_id: int, cfg: ExecutionConfig) -> MagicMock:
            _ = node_id
            captured.append(cfg)
            return MagicMock(run_id="r1")

        _mock_attr(executor, "dispatch").side_effect = dispatch

        loop = RunLoop(executor=executor, graph=graph, loop=fake_loop, config=milknado_cfg)
        _ = _dispatch_batch(loop, config, 4)

        assert len(captured) == 1, "expected exactly one dispatch call"
        assert captured[0].quality_gates == (), (
            "flavored node with quality_gates=() must propagate empty gates to executor"
        )
        assert captured[0].brief_prepend == "Research only."

    def test_flavored_node_dispatches_with_attempt_caps(
        self,
        graph: MikadoGraph,
        fake_loop: FakeLoop,
        config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        """The CLI loop must carry the flavor's iteration and attempt caps to the executor."""
        from unittest.mock import MagicMock

        from milknado.domains.common.config import FlavorOverride, MilknadoConfig
        from milknado.domains.execution.run_loop import RunLoop

        milknado_cfg = MilknadoConfig(
            agent_family="claude",
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            flavors={
                "runner": FlavorOverride(max_iterations=3, attempt_timeout_seconds=1800),
            },
        )

        root = graph.add_node("root")
        _ = graph.add_node("runner leaf", parent_id=root.id, spec=NodeSpec(flavor="runner"))

        captured: list[ExecutionConfig] = []
        executor = MagicMock()

        def dispatch(node_id: int, cfg: ExecutionConfig) -> MagicMock:
            _ = node_id
            captured.append(cfg)
            return MagicMock(run_id="r1")

        _mock_attr(executor, "dispatch").side_effect = dispatch

        loop = RunLoop(executor=executor, graph=graph, loop=fake_loop, config=milknado_cfg)
        _ = _dispatch_batch(loop, config, 4)

        assert len(captured) == 1, "expected exactly one dispatch call"
        assert captured[0].max_iterations == 3
        assert captured[0].attempt_timeout_seconds == 1800.0
        assert captured[0].completion_timeout_seconds == 5400


def _wait_for_owner_work(loop: RunLoop, config: ExecutionConfig, concurrency_limit: int) -> int:
    method = cast(Callable[[ExecutionConfig, int], int], attrgetter("_wait_for_owner_work")(loop))
    return method(config, concurrency_limit)


class TestOwnerIdleWait:
    """An owner-attached run keeps scheduling while the graph has nothing ready."""

    @staticmethod
    def _owner_loop(
        graph: MikadoGraph, fake_loop: FakeLoop, controls: Callable[[], None]
    ) -> tuple[RunLoop, MagicMock, list[float]]:
        executor = MagicMock()

        def dispatch(node_id: int, _cfg: ExecutionConfig) -> MagicMock:
            return MagicMock(run_id=f"r{node_id}")

        _mock_attr(executor, "dispatch").side_effect = dispatch
        loop = RunLoop(executor=executor, graph=graph, loop=fake_loop)
        sleeps: list[float] = []
        _set_attr(loop, "_await_owner_work", True)
        _set_attr(loop, "_process_controls", controls)
        _set_attr(loop, "_idle_sleep", sleeps.append)
        return loop, executor, sleeps

    def test_dispatches_a_node_the_owner_readies_while_idle(
        self, graph: MikadoGraph, fake_loop: FakeLoop, config: ExecutionConfig
    ) -> None:
        root = graph.add_node("root")
        calls: list[int] = []

        def controls() -> None:
            calls.append(len(calls))
            if len(calls) == 2:
                _ = graph.add_node("late leaf", parent_id=root.id)

        loop, executor, sleeps = self._owner_loop(graph, fake_loop, controls)

        assert _wait_for_owner_work(loop, config, 4) == 1
        assert sleeps == [1.0], "the loop must sleep once before the node became ready"
        assert _mock_attr(executor, "dispatch").call_count == 1
        assert [run.node_id for run in _scheduler(loop).view().active] == [root.id + 1]

    def test_stops_waiting_when_the_owner_closes_scheduling(
        self, graph: MikadoGraph, fake_loop: FakeLoop, config: ExecutionConfig
    ) -> None:
        _ = graph.add_node("root")
        loop, executor, sleeps = self._owner_loop(graph, fake_loop, lambda: None)
        _set_attr(loop, "_process_controls", loop.stop_scheduling)

        assert _wait_for_owner_work(loop, config, 4) == 0
        assert sleeps == []
        _mock_attr(executor, "dispatch").assert_not_called()

    def test_batch_run_ends_instead_of_waiting(
        self, graph: MikadoGraph, fake_loop: FakeLoop, config: ExecutionConfig
    ) -> None:
        _ = graph.add_node("root")
        controls = MagicMock()
        loop, _executor, sleeps = self._owner_loop(graph, fake_loop, controls)
        _set_attr(loop, "_await_owner_work", False)

        assert _wait_for_owner_work(loop, config, 4) == 0
        controls.assert_not_called()
        assert sleeps == []

    def test_publishes_a_failed_idle_dispatch_before_sleeping(
        self, graph: MikadoGraph, fake_loop: FakeLoop, config: ExecutionConfig
    ) -> None:
        _ = graph.add_node("root")
        loop, _executor, sleeps = self._owner_loop(graph, fake_loop, lambda: None)
        received: list[RunLoopState] = []
        loop.set_state_listener(received.append)

        def dispatch_fails(_config: ExecutionConfig, _limit: int) -> tuple[int, int]:
            loop.stop_scheduling()
            return 0, 1

        _set_attr(loop, "_dispatch_if_scheduling_open", dispatch_fails)

        assert _wait_for_owner_work(loop, config, 4) == 0
        assert received[-1].failed == 1
        assert sleeps == [1.0]

    def test_completes_the_root_once_owner_work_is_done(
        self, graph: MikadoGraph, fake_loop: FakeLoop, config: ExecutionConfig
    ) -> None:
        root = graph.add_node("root")
        leaf = graph.add_node("finished leaf", parent_id=root.id)
        graph.mark_running(leaf.id)
        graph.mark_done(leaf.id)
        loop, executor, sleeps = self._owner_loop(graph, fake_loop, lambda: None)

        assert _wait_for_owner_work(loop, config, 4) == 0
        settled = graph.get_node(root.id)
        assert settled is not None and settled.status == NodeStatus.DONE
        assert sleeps == []
        _mock_attr(executor, "dispatch").assert_not_called()

    def test_waits_for_a_pending_goal_review_before_completing_the_root(
        self,
        graph: MikadoGraph,
        fake_loop: FakeLoop,
        config: ExecutionConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(CONTROLLER_MASTER_ENV, "external-controller-master")
        graph.register_controller_master()
        root = graph.add_node("root", spec=NodeSpec(kind=NodeKind.GOAL))
        leaf = graph.add_node("finished leaf", parent_id=root.id)
        graph.mark_running(leaf.id)
        graph.mark_done(leaf.id)
        review = graph.request_goal_review(
            GoalReviewRequest(
                goal_id=root.id,
                goal_revision="sha256:goal",
                evidence="Review evidence",
                proposed_change="Review proposed change",
                affected_node_ids=None,
                reviewer="worker",
                assessed_at="2026-09-13T12:00:00+00:00",
            )
        )
        scans: list[int] = []

        def decide_on_second_scan() -> None:
            scans.append(len(scans))
            if len(scans) == 2:
                _ = graph.decide_goal_review(
                    GoalReviewDecisionRequest(review.review_id, GoalReviewDecision.REJECTED),
                    decided_by="human",
                )

        loop, _executor, sleeps = self._owner_loop(graph, fake_loop, decide_on_second_scan)

        assert _wait_for_owner_work(loop, config, 4) == 0
        assert sleeps == [1.0]
        settled = graph.get_node(root.id)
        assert settled is not None and settled.status == NodeStatus.DONE

    def test_verifies_the_spec_once_per_idle_graph_state(
        self, graph: MikadoGraph, fake_loop: FakeLoop, config: ExecutionConfig
    ) -> None:
        root = graph.add_node("root")
        leaf = graph.add_node("finished leaf", parent_id=root.id)
        graph.mark_running(leaf.id)
        graph.mark_done(leaf.id)
        verify_calls: list[str] = []

        def gaps(spec_text: str, _graph_state: str) -> VerifySpecResult:
            verify_calls.append(spec_text)
            return VerifySpecResult(outcome="gaps")

        _set_attr(fake_loop, "verify_spec", gaps)
        controls: list[int] = []
        loop, _executor, sleeps = self._owner_loop(graph, fake_loop, lambda: None)

        def stop_on_third_scan() -> None:
            controls.append(len(controls))
            if len(controls) == 3:
                loop.stop_scheduling()

        _set_attr(loop, "_process_controls", stop_on_third_scan)
        _set_attr(loop, "_spec", ("spec: ship it", None))

        assert _wait_for_owner_work(loop, config, 4) == 0
        assert verify_calls == ["spec: ship it"]
        assert sleeps == [1.0, 1.0]
        settled = graph.get_node(root.id)
        assert settled is not None and settled.status == NodeStatus.PENDING

    def test_owner_run_reports_the_idle_spec_verification(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
        fake_loop: FakeLoop,
    ) -> None:
        executor = Executor(graph=graph, git=fake_git, loop=fake_loop, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, loop=fake_loop)
        sleeps: list[float] = []
        _set_attr(loop, "_idle_sleep", sleeps.append)
        scans: list[int] = []

        def bounded_controls() -> None:
            scans.append(len(scans))
            assert len(scans) < 50, "owner-attached run never settled the root"

        root = graph.add_node("root goal")
        _ = graph.add_node("leaf", parent_id=root.id)

        result = loop.run(
            config,
            "main",
            spec_text="spec: do the thing",
            process_controls=bounded_controls,
            interactive=False,
            await_owner_work=True,
        )

        assert result.root_done is True
        assert result.verify_outcome is not None and result.verify_outcome.done is True

    def test_execute_run_returns_once_the_owner_stops_an_idle_run(
        self, graph: MikadoGraph, fake_loop: FakeLoop, config: ExecutionConfig
    ) -> None:
        _ = graph.add_node("root")
        loop, executor, _sleeps = self._owner_loop(graph, fake_loop, lambda: None)
        _set_attr(loop, "_process_controls", loop.stop_scheduling)

        dispatched, completed, failed, conflicts, timed_out = _execute_run(
            loop, config, "main", 4, None, False
        )

        assert (dispatched, completed, failed, conflicts, timed_out) == (0, 0, 0, [], False)
        _mock_attr(executor, "dispatch").assert_not_called()
