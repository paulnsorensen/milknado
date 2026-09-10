import collections
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from operator import attrgetter
from pathlib import Path
from typing import Protocol, TypeVar, cast
from unittest.mock import MagicMock, patch

import pytest

from milknado.adapters.loop import LoopAdapter
from milknado.domains.common import (
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
    DispatchResult,
    ExecutionConfig,
    Executor,
    RunLoop,
)
from milknado.domains.execution.executor import RebaseConflict, WorktreeManager
from milknado.domains.execution.run_loop.state import (
    RunLoopState,
    TerminalRunState,
    summarize_description,
)
from milknado.domains.graph import MikadoGraph
from milknado.loop import RunStatus

T = TypeVar("T")


def _active(loop: RunLoop) -> dict[str, int]:
    return cast(dict[str, int], attrgetter("_active")(loop))


def _progress_by_run(loop: RunLoop) -> dict[str, ProgressEvent]:
    return cast(dict[str, ProgressEvent], attrgetter("_progress_by_run")(loop))


def _dispatched_at(loop: RunLoop) -> dict[str, float]:
    return cast(dict[str, float], attrgetter("_dispatched_at")(loop))


def _completion_durations(loop: RunLoop) -> collections.deque[float]:
    return cast(collections.deque[float], attrgetter("_completion_durations")(loop))


def _attempts(loop: RunLoop) -> dict[int, int]:
    return cast(dict[int, int], attrgetter("_attempts")(loop))


def _stopped_nodes(loop: RunLoop) -> set[int]:
    return cast(set[int], attrgetter("_stopped_nodes")(loop))


def _terminal_runs(loop: RunLoop) -> collections.deque[TerminalRunState]:
    return cast(collections.deque[TerminalRunState], attrgetter("_terminal_runs")(loop))


def _success(ralph: object) -> dict[str, bool]:
    return cast(dict[str, bool], attrgetter("_success")(ralph))


def _runs(ralph: object) -> dict[str, "FakeRun"]:
    return cast(dict[str, "FakeRun"], attrgetter("_runs")(ralph))


def _ordinal_to_run_id(ralph: object) -> dict[str, str]:
    return cast(dict[str, str], attrgetter("_ordinal_to_run_id")(ralph))


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


def _progress_before_completion(ralph: object) -> list[ProgressEvent]:
    return cast(list[ProgressEvent], attrgetter("_progress_before_completion")(ralph))


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


class FakeRalph:
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
        self._ordinal_to_run_id: dict[str, str] = {}
        self._run_id_to_ordinal: dict[str, str] = {}

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        quality_gates: tuple[Gate, ...] | None,
        project_root: Path | None = None,
        commit_footer: str | None = None,
        base_oid: str | None = None,
        runtime_policy: object | None = None,
        run_id: str | None = None,
        completion_probe: Callable[[], bool] | None = None,
    ) -> FakeRun:
        _ = (
            agent,
            ralph_dir,
            ralph_file,
            quality_gates,
            project_root,
            commit_footer,
            base_oid,
            runtime_policy,
            completion_probe,
        )
        self._run_counter += 1
        ordinal_id = f"run-{self._run_counter}"
        resolved_run_id = run_id or ordinal_id
        self._ordinal_to_run_id[ordinal_id] = resolved_run_id
        self._run_id_to_ordinal[resolved_run_id] = ordinal_id
        success = self._success.get(resolved_run_id, self._success.get(ordinal_id, True))
        outcome = self._outcomes.get(
            resolved_run_id,
            self._outcomes.get(ordinal_id, "completed" if success else "failed"),
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
    ) -> _FakeReview:
        _ = (agent, prompt, worktree, project_root, timeout_seconds)
        return _FakeReview()

    def verify_spec(self, spec_text: str, graph_state: str) -> VerifySpecResult:
        _ = (spec_text, graph_state)
        return VerifySpecResult(outcome="done")

    def generate_ralph_md(
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
        self._outcomes[run_id] = "stopped"


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
def fake_ralph() -> FakeRalph:
    return FakeRalph()


@pytest.fixture()
def fake_crg() -> FakeCrg:
    return FakeCrg()


@pytest.fixture()
def executor(
    graph: MikadoGraph,
    fake_git: FakeGit,
    fake_ralph: FakeRalph,
    fake_crg: FakeCrg,
) -> Executor:
    return Executor(graph=graph, git=fake_git, ralph=fake_ralph, crg=fake_crg)


def test_state_is_bounded_and_published(
    run_loop: RunLoop,
    graph: MikadoGraph,
    fake_ralph: FakeRalph,
) -> None:
    root = graph.add_node("ship controller")
    leaf = graph.add_node("build snapshots", parent_id=root.id)
    graph.mark_running(leaf.id)
    _runs(fake_ralph)["run-1"] = FakeRun(state=FakeRunState(run_id="run-1", stop_requested=True))
    fake_ralph.output["run-1"] = [f"line {index}" for index in range(35)]
    fake_ralph.guidance["run-1"] = ("use domain barrels",)
    _active(run_loop)["run-1"] = leaf.id
    _progress_by_run(run_loop)["run-1"] = ProgressEvent(
        run_id="run-1", work=1, total=2, message="building"
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


class TestActiveStateProjectedFields:
    """`_active_state` computes elapsed/progress/eta/attempt/stalled — pin these to
    the real producer so mutating the formulas fails the suite."""

    def _seed_active(self, run_loop: RunLoop, graph: MikadoGraph) -> tuple[int, str]:
        root = graph.add_node("ship controller")
        leaf = graph.add_node("build snapshots", parent_id=root.id)
        graph.mark_running(leaf.id)
        _active(run_loop)["run-1"] = leaf.id
        return leaf.id, "run-1"

    def test_elapsed_seconds_from_known_dispatch_time(
        self, run_loop: RunLoop, graph: MikadoGraph
    ) -> None:
        _, run_id = self._seed_active(run_loop, graph)
        _dispatched_at(run_loop)[run_id] = 100.0

        with patch("milknado.domains.execution.run_loop.time.monotonic", return_value=150.0):
            state = run_loop.state()

        assert state.active_runs[0].elapsed_seconds == 50.0

    def test_elapsed_seconds_defaults_to_zero_without_dispatch_time(
        self, run_loop: RunLoop, graph: MikadoGraph
    ) -> None:
        _ = self._seed_active(run_loop, graph)

        with patch("milknado.domains.execution.run_loop.time.monotonic", return_value=200.0):
            state = run_loop.state()

        assert state.active_runs[0].elapsed_seconds == 0.0

    def test_progress_pct_from_work_and_total(self, run_loop: RunLoop, graph: MikadoGraph) -> None:
        _, run_id = self._seed_active(run_loop, graph)
        _progress_by_run(run_loop)[run_id] = ProgressEvent(
            run_id=run_id, work=3, total=4, message="x"
        )

        state = run_loop.state()

        assert state.active_runs[0].progress_pct == 75.0

    def test_progress_pct_is_none_when_total_is_zero(
        self, run_loop: RunLoop, graph: MikadoGraph
    ) -> None:
        _, run_id = self._seed_active(run_loop, graph)
        _progress_by_run(run_loop)[run_id] = ProgressEvent(
            run_id=run_id, work=0, total=0, message="x"
        )

        state = run_loop.state()

        assert state.active_runs[0].progress_pct is None

    def test_eta_seconds_is_none_with_fewer_than_three_samples(
        self, run_loop: RunLoop, graph: MikadoGraph
    ) -> None:
        _ = self._seed_active(run_loop, graph)
        _completion_durations(run_loop).extend([10.0, 20.0])

        state = run_loop.state()

        assert state.active_runs[0].eta_seconds is None

    def test_eta_seconds_is_mean_minus_elapsed_with_three_or_more_samples(
        self, run_loop: RunLoop, graph: MikadoGraph
    ) -> None:
        _, run_id = self._seed_active(run_loop, graph)
        _completion_durations(run_loop).extend([10.0, 20.0, 30.0])
        _dispatched_at(run_loop)[run_id] = 100.0

        with patch("milknado.domains.execution.run_loop.time.monotonic", return_value=105.0):
            state = run_loop.state()

        assert state.active_runs[0].eta_seconds == 15.0

    def test_eta_seconds_floors_at_zero(self, run_loop: RunLoop, graph: MikadoGraph) -> None:
        _, run_id = self._seed_active(run_loop, graph)
        _completion_durations(run_loop).extend([1.0, 2.0, 3.0])
        _dispatched_at(run_loop)[run_id] = 100.0

        with patch("milknado.domains.execution.run_loop.time.monotonic", return_value=200.0):
            state = run_loop.state()

        assert state.active_runs[0].eta_seconds == 0.0

    def test_attempt_is_one_on_first_try(self, run_loop: RunLoop, graph: MikadoGraph) -> None:
        _ = self._seed_active(run_loop, graph)

        state = run_loop.state()

        assert state.active_runs[0].attempt == 1

    def test_attempt_increments_after_recorded_failure(
        self, run_loop: RunLoop, graph: MikadoGraph
    ) -> None:
        node_id, _ = self._seed_active(run_loop, graph)
        _attempts(run_loop)[node_id] = 1

        state = run_loop.state()

        assert state.active_runs[0].attempt == 2

    def test_max_attempts_from_config_dispatch_max_retries(
        self, graph: MikadoGraph, executor: Executor, fake_ralph: FakeRalph
    ) -> None:
        run_loop = RunLoop(
            executor=executor,
            graph=graph,
            ralph=fake_ralph,
            config=MilknadoConfig(dispatch_max_retries=4),
        )
        _ = self._seed_active(run_loop, graph)

        state = run_loop.state()

        assert state.active_runs[0].max_attempts == 5

    def test_stalled_false_below_threshold(
        self, graph: MikadoGraph, executor: Executor, fake_ralph: FakeRalph
    ) -> None:
        run_loop = RunLoop(
            executor=executor,
            graph=graph,
            ralph=fake_ralph,
            config=MilknadoConfig(stall_threshold_seconds=300),
        )
        _, run_id = self._seed_active(run_loop, graph)
        _dispatched_at(run_loop)[run_id] = 0.0

        with patch("milknado.domains.execution.run_loop.time.monotonic", return_value=299.0):
            state = run_loop.state()

        assert state.active_runs[0].stalled is False

    def test_stalled_true_at_or_above_threshold(
        self, graph: MikadoGraph, executor: Executor, fake_ralph: FakeRalph
    ) -> None:
        run_loop = RunLoop(
            executor=executor,
            graph=graph,
            ralph=fake_ralph,
            config=MilknadoConfig(stall_threshold_seconds=300),
        )
        _, run_id = self._seed_active(run_loop, graph)
        _dispatched_at(run_loop)[run_id] = 0.0

        with patch("milknado.domains.execution.run_loop.time.monotonic", return_value=300.0):
            state = run_loop.state()

        assert state.active_runs[0].stalled is True


def test_terminal_run_duration_seconds_from_stopped_completion(
    run_loop: RunLoop, graph: MikadoGraph, fake_ralph: FakeRalph
) -> None:
    from milknado.domains.execution.run_loop._completion import handle_completion

    root = graph.add_node("ship controller")
    leaf = graph.add_node("build snapshots", parent_id=root.id)
    graph.mark_running(leaf.id)
    _active(run_loop)["run-1"] = leaf.id
    _dispatched_at(run_loop)["run-1"] = 100.0
    _runs(fake_ralph)["run-1"] = FakeRun(state=FakeRunState(run_id="run-1"))

    with patch(
        "milknado.domains.execution.run_loop._completion.time.monotonic",
        return_value=142.0,
    ):
        _ = handle_completion(run_loop, "run-1", "stopped", "main")

    assert _terminal_runs(run_loop)[-1].duration_seconds == 42.0


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
    fake_ralph: FakeRalph,
    status: RunStatus,
    reason: str,
) -> None:
    root = graph.add_node("ship controller")
    leaf = graph.add_node("build snapshots", parent_id=root.id)
    _runs(fake_ralph)["run-1"] = FakeRun(state=FakeRunState(run_id="run-1", status=status))
    _active(run_loop)["run-1"] = leaf.id

    active = run_loop.state().active_runs[0]

    assert active.actions.cancel_reason == reason
    assert active.actions.guidance_reason == reason
    assert active.actions.force_stop_reason == reason


def test_control_queue_applies_cancel_and_force_stop(
    run_loop: RunLoop,
    graph: MikadoGraph,
    fake_ralph: FakeRalph,
) -> None:
    root = graph.add_node("ship controller")
    leaf = graph.add_node("stop worker", parent_id=root.id)
    graph.mark_running(leaf.id)
    _runs(fake_ralph)["run-1"] = FakeRun()
    _active(run_loop)["run-1"] = leaf.id

    run_loop.cancel("run-1")
    assert fake_ralph.requested_stops == ["run-1"]
    assert run_loop.force_stop("run-1", timeout=2.5) is True
    assert fake_ralph.force_stops == [("run-1", 2.5)]
    assert run_loop.state().active_runs[0].actions.force_stop_reason == (
        "force stop already requested"
    )

    run_loop.stop_scheduling()
    assert fake_ralph.requested_stops == ["run-1", "run-1"]


def test_progress_snapshot_is_published_before_terminal_completion(
    run_loop: RunLoop,
    graph: MikadoGraph,
    config: ExecutionConfig,
    fake_ralph: FakeRalph,
) -> None:
    root = graph.add_node("ship controller")
    _ = graph.add_node("build snapshots", parent_id=root.id)
    _progress_before_completion(fake_ralph).append(
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
    fake_ralph: FakeRalph,
) -> RunLoop:
    return RunLoop(executor=executor, graph=graph, ralph=fake_ralph)


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
    fake_ralph: FakeRalph,
) -> None:
    from milknado.domains.common.errors import CompletionTimeout

    node = graph.add_node("active")
    graph.mark_running(node.id)
    _active(run_loop)["run-1"] = node.id
    _set_attr(run_loop, "_dispatch_if_scheduling_open", MagicMock(return_value=(0, 0)))
    _set_attr(run_loop, "_handle_completion_timeout", MagicMock(return_value=1))
    control_calls = 0

    def process_controls() -> None:
        nonlocal control_calls
        control_calls += 1
        if control_calls == 3:
            _active(run_loop).clear()

    def short_poll_timeout(
        active_run_ids: set[str], timeout: float | None = None
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]:
        raise CompletionTimeout(waited_seconds=timeout or 0.0, active_run_ids=active_run_ids)

    _set_attr(fake_ralph, "wait_for_next_completion", short_poll_timeout)
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


def test_unset_completion_timeout_polls_controls_without_timing_out(
    run_loop: RunLoop,
    graph: MikadoGraph,
    config: ExecutionConfig,
    fake_ralph: FakeRalph,
) -> None:
    from milknado.domains.common.errors import CompletionTimeout

    node = graph.add_node("active")
    graph.mark_running(node.id)
    _active(run_loop)["run-1"] = node.id
    _set_attr(run_loop, "_dispatch_if_scheduling_open", MagicMock(return_value=(0, 0)))
    _set_attr(run_loop, "_handle_completion_timeout", MagicMock(return_value=1))
    observed_timeouts: list[float | None] = []
    control_calls = 0

    def process_controls() -> None:
        nonlocal control_calls
        control_calls += 1
        if control_calls >= 4:
            _active(run_loop).clear()

    def short_poll_timeout(
        active_run_ids: set[str], timeout: float | None = None
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]:
        observed_timeouts.append(timeout)
        raise CompletionTimeout(waited_seconds=timeout or 0.0, active_run_ids=active_run_ids)

    _set_attr(fake_ralph, "wait_for_next_completion", short_poll_timeout)
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
        ralph = FakeRalph()
        ralph.set_run_stopped("run-1")
        ralph.output["run-1"] = ["last worker output"]
        ralph.guidance["run-1"] = ("not delivered",)
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        root = graph.add_node("root")
        leaf = graph.add_node("leaf", parent_id=root.id)

        with patch("milknado.domains.execution.run_loop._logger.info") as log_info:
            result = loop.run(config, "main", interactive=False)

        node = graph.get_node(leaf.id)
        assert node is not None
        assert node.status is NodeStatus.PENDING
        assert (result.dispatched_total, result.completed_total, result.failed_total) == (1, 0, 0)
        assert _stopped_nodes(loop) == {leaf.id}
        assert _active(loop) == {}
        state = loop.state()
        assert state.stopped == 1
        assert state.available == 0
        assert len(state.terminal_runs) == 1
        terminal = state.terminal_runs[0]
        real_run_id = _ordinal_to_run_id(ralph)["run-1"]
        assert terminal.run_id == real_run_id
        assert terminal.status is RunStatus.STOPPED
        assert terminal.output == ("last worker output",)
        assert terminal.pending_guidance == ("not delivered",)
        assert any(
            call.args[0] == "node_stopped node_id=%d run_id=%s duration=%.1fs"
            and call.args[1:3] == (leaf.id, real_run_id)
            for call in log_info.call_args_list
        )
        assert any(
            call.args[0] == "FINAL_TELEMETRY %s" and '"stopped": 1' in call.args[1]
            for call in log_info.call_args_list
        )

    def test_stopped_terminal_history_is_bounded_to_the_newest_twenty(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        ralph = FakeRalph()
        for index in range(1, 22):
            ralph.set_run_stopped(f"run-{index}")
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        root = graph.add_node("root")
        for index in range(21):
            _ = graph.add_node(f"leaf-{index}", parent_id=root.id)
        _ = loop.run(config, "main", interactive=False)

        state = loop.state()
        assert state.stopped == 21
        assert len(state.terminal_runs) == 20
        assert tuple(run.run_id for run in state.terminal_runs) == tuple(
            _ordinal_to_run_id(ralph)[f"run-{index}"] for index in range(2, 22)
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


class TestRunLoopFailure:
    def test_failed_leaf_marks_leaf_failed(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        # Root is not dispatched; only the leaf is. Set the leaf's run to fail.
        ralph = FakeRalph()
        _success(ralph)["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        ralph = FakeRalph()
        _success(ralph)["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        ralph = FakeRalph()

        def fail_generate(*_args: object, **_kwargs: object) -> Path:
            raise RuntimeError("ralph exploded")

        ralph.generate_ralph_md = fail_generate
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        ralph = FakeRalph()
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
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
        ralph = FakeRalph()
        call_count = 0
        original_generate = ralph.generate_ralph_md

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
                raise RuntimeError("ralph exploded")
            return original_generate(
                brief, quality_gates, output_path, prior_findings, findings_round
            )

        ralph.generate_ralph_md = fail_first_only  # type: ignore
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        ralph = FakeRalph()
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        fake_ralph: FakeRalph,
        fake_crg: FakeCrg,
    ) -> None:
        executor = Executor(
            graph=graph,
            git=fake_git,
            ralph=fake_ralph,
            crg=fake_crg,
        )
        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph)

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
        ralph = FakeRalph()
        _success(ralph)["run-1"] = False  # leaf-a fails, leaf-b (run-2) succeeds

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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

        ralph = FakeRalph()
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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

        ralph = FakeRalph()
        call_count = 0
        original_generate = ralph.generate_ralph_md

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

        ralph.generate_ralph_md = fail_thrice  # type: ignore

        executor = _Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)

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
        ralph = FakeRalph()
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        ralph = FakeRalph()
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        original_dispatch = executor.dispatch

        def tracking_dispatch(node_id: int, cfg: ExecutionConfig) -> DispatchResult:
            dispatched_ids.append(node_id)
            return original_dispatch(node_id, cfg)

        _set_attr(executor, "dispatch", tracking_dispatch)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        ralph = FakeRalph()
        _success(ralph)["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        ralph = FakeRalph()
        verify_calls: list[tuple[str, str]] = []
        original_verify_spec = ralph.verify_spec

        def tracking_verify_spec(spec_text: str, graph_state: str) -> VerifySpecResult:
            verify_calls.append((spec_text, graph_state))
            return original_verify_spec(spec_text, graph_state)

        _set_attr(ralph, "verify_spec", tracking_verify_spec)

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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

        ralph = LoopAdapter()
        planner = MagicMock(spec=Planner)
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph, planner=planner)

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
        ralph = FakeRalph()
        _success(ralph)["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        ralph = FakeRalph()
        _success(ralph)["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        ralph = FakeRalph()
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        ralph_dir = tmp_path / "wt-node-1"
        ralph_dir.mkdir()
        ralph_file = ralph_dir / "ralph.md"
        _ = ralph_file.write_text("# task", encoding="utf-8")

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
                ralph_dir=ralph_dir,
                ralph_file=ralph_file,
                quality_gates=(),
                project_root=None,
            )

        assert len(captured_configs) == 1
        cfg = captured_configs[0]
        assert cfg.log_dir == ralph_dir / ".ralph-logs"


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
        ralph = FakeRalph()
        _set_attr(
            ralph, "wait_for_next_completion", MagicMock(wraps=ralph.wait_for_next_completion)
        )
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        root = graph.add_node("root")
        _ = graph.add_node("slow-leaf", parent_id=root.id)

        result = loop.run(config, "main")

        assert result.completed_total == 1
        assert _call_kwargs(_mock_attr(ralph, "wait_for_next_completion"))["timeout"] is None

    def test_configured_wait_keeps_explicit_timeout(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        from milknado.domains.common.config import MilknadoConfig

        ralph = FakeRalph()
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        milknado_config = MilknadoConfig(completion_timeout_seconds=60.0)
        loop = RunLoop(
            executor=executor,
            graph=graph,
            ralph=ralph,
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

        ralph = FakeRalph()

        # Make wait_for_next_completion raise CompletionTimeout
        def raise_timeout(
            active_run_ids: set[str], timeout: float | None = None
        ) -> tuple[str, object]:
            _ = timeout
            raise CompletionTimeout(waited_seconds=60.0, active_run_ids=active_run_ids)

        _set_attr(ralph, "wait_for_next_completion", raise_timeout)

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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

        ralph = FakeRalph()
        first_call = [True]

        def raise_timeout(
            active_run_ids: set[str], timeout: float | None = None
        ) -> tuple[str, object]:
            _ = timeout
            if first_call[0]:
                first_call[0] = False
                raise CompletionTimeout(waited_seconds=60.0, active_run_ids=active_run_ids)
            return next(iter(active_run_ids)), True

        _set_attr(ralph, "wait_for_next_completion", raise_timeout)

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        executor.stop_run = MagicMock(return_value=False)
        ralph = FakeRalph()
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        _set_attr(loop, "_active", {"run-1": 7})

        failed = _handle_completion_timeout(
            loop, CompletionTimeout(waited_seconds=60.0, active_run_ids={"run-1"})
        )

        assert failed == 0
        assert _active(loop) == {"run-1": 7}
        _mock_attr(executor, "fail").assert_not_called()
        _mock_attr(executor, "stop_run").assert_called_once_with("run-1", timeout=10.0)


class TestVerifySpecGapsPath:
    def test_gaps_outcome_calls_replan(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        from milknado.domains.common.protocols import VerifySpecResult

        ralph = FakeRalph()

        def gaps_verify(spec_text: str, graph_state: str) -> VerifySpecResult:
            _ = (spec_text, graph_state)
            return VerifySpecResult(outcome="gaps", goal_delta="add feature X")

        ralph.verify_spec = gaps_verify

        from milknado.domains.planning.planner import Planner

        planner = MagicMock(spec=Planner)
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph, planner=planner)

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
        ralph = FakeRalph()
        verify_calls: list[tuple[object, ...]] = []

        def done_verify(spec_text: str, graph_state: str) -> VerifySpecResult:
            verify_calls.append((spec_text, graph_state))
            return VerifySpecResult(outcome="done")

        ralph.verify_spec = done_verify

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        ralph = FakeRalph()
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        ralph = FakeRalph()

        def raise_interrupt(
            active_run_ids: set[str], timeout: float | None = None
        ) -> tuple[str, object]:
            _ = timeout
            _ = active_run_ids
            raise KeyboardInterrupt

        _set_attr(ralph, "wait_for_next_completion", raise_interrupt)

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

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
        fake_ralph: FakeRalph,
    ) -> None:

        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph)
        _set_attr(loop, "_strict", True)
        _set_attr(loop, "_failure_triggered", True)

        result = _dispatch_batch(loop, config, 4)

        assert result == (0, 0)

    def test_no_capacity_returns_zero(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_ralph: FakeRalph,
    ) -> None:

        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph)
        # Fill active to the limit
        _set_attr(loop, "_active", {"run-1": 1, "run-2": 2, "run-3": 3, "run-4": 4})

        result = _dispatch_batch(loop, config, 4)

        assert result == (0, 0)

    def test_dispatch_exception_increments_failed(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_ralph: FakeRalph,
    ) -> None:
        from unittest.mock import MagicMock

        executor = MagicMock()
        _mock_attr(executor, "dispatch").side_effect = RuntimeError("boom")
        root = graph.add_node("root")
        leaf = graph.add_node("failing-leaf", parent_id=root.id)

        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph)
        dispatched, failed = _dispatch_batch(loop, config, 4)

        assert dispatched == 0
        assert failed == 1
        assert f"✗ dispatch node {leaf.id}" in loop.state().event_lines[-1]

    def test_strict_mode_breaks_on_dispatch_exception(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_ralph: FakeRalph,
    ) -> None:
        from unittest.mock import MagicMock

        executor = MagicMock()
        _mock_attr(executor, "dispatch").side_effect = RuntimeError("boom")
        root = graph.add_node("root")
        _ = graph.add_node("leaf-a", parent_id=root.id)
        _ = graph.add_node("leaf-b", parent_id=root.id)

        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph)
        _set_attr(loop, "_strict", True)
        _ = _dispatch_batch(loop, config, 4)

        assert _mock_attr(executor, "dispatch").call_count == 1
        assert cast(bool, attrgetter("_failure_triggered")(loop)) is True


class TestDispatchBatchFlavoredGates:
    """AC6: a flavored node with quality_gates=() propagates empty gates to the executor."""

    def test_research_node_dispatches_with_empty_quality_gates(
        self,
        graph: MikadoGraph,
        fake_ralph: FakeRalph,
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

        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph, config=milknado_cfg)
        _ = _dispatch_batch(loop, config, 4)

        assert len(captured) == 1, "expected exactly one dispatch call"
        assert captured[0].quality_gates == (), (
            "flavored node with quality_gates=() must propagate empty gates to executor"
        )
        assert captured[0].brief_prepend == "Research only."
