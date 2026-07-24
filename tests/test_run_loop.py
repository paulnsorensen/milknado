import collections
import io
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from milknado.domains.common import ProgressEvent
from milknado.domains.common.types import (
    MikadoNode,
    NodeSpec,
    NodeStatus,
    RebaseResult,
)
from milknado.domains.execution import ExecutionConfig, Executor, RunLoop
from milknado.domains.execution.run_loop.display import (
    TuiState,
    _build_log_panel,
    _build_title,
    _build_worker_table,
    _render_overlay,
    _render_progress_bar,
)
from milknado.domains.execution.run_loop.state import RunLoopState
from milknado.domains.graph import MikadoGraph
from milknado.loop import RunStatus


@dataclass
class FakeRunState:
    run_id: str = "run-1"
    status: RunStatus = RunStatus.RUNNING
    stop_requested: bool = False
    force_stop_requested: bool = False


@dataclass
class FakeRun:
    state: FakeRunState = field(default_factory=FakeRunState)


class FakeGit:
    def __init__(self) -> None:
        self.created: list[tuple[Path, str]] = []
        self.removed: list[Path] = []
        self.rebase_result: RebaseResult = RebaseResult(success=True)

    def branch_exists(self, branch: str) -> bool:
        return False

    def create_worktree(self, path: Path, branch: str) -> Path:
        self.created.append((path, branch))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
        self.removed.append(path)

    def force_remove_worktree(self, path: Path) -> None:
        self.removed.append(path)

    def rebase(self, worktree: Path, onto: str) -> RebaseResult:
        return self.rebase_result

    def current_branch(self) -> str:
        return "main"

    def resolve_ref(self, ref: str) -> str:
        return f"{ref}-oid"

    def commit_all(self, worktree: Path, message: str) -> None:
        pass

    def squash_and_commit(self, worktree: Path, onto: str, msg: str) -> None:
        pass


class FakeCrg:
    def ensure_graph(self, project_root: Path) -> None:
        pass

    def get_impact_radius(self, files: list[str]) -> dict[str, Any]:
        return {"files": files}

    def get_architecture_overview(self) -> dict[str, Any]:
        return {"modules": []}

    def list_communities(
        self,
        sort_by: str = "size",
        min_size: int = 0,
    ) -> list[dict[str, Any]]:
        return []

    def list_flows(
        self,
        sort_by: str = "criticality",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return []

    def get_bridge_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return []

    def get_hub_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return []


class FakeRalph:
    def __init__(self) -> None:
        self._run_counter = 0
        self._success: dict[str, bool] = {}
        self._outcomes: dict[str, str] = {}
        self._pending_completions: list[tuple[str, str]] = []
        self._runs: dict[str, FakeRun] = {}
        self.output: dict[str, list[str]] = {}
        self.guidance: dict[str, tuple[str, ...]] = {}
        self._progress_before_completion: list[ProgressEvent] = []
        self.requested_stops: list[str] = []
        self.force_stops: list[tuple[str, float | None]] = []
        # Maps the ordinal placeholder id ("run-1", "run-2", ...) a test
        # pre-seeds before dispatch to the real (Executor-supplied,
        # node-format) run id assigned at create_run time, and back.
        self._ordinal_to_run_id: dict[str, str] = {}
        self._run_id_to_ordinal: dict[str, str] = {}

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: list[str],
        project_root: Path | None = None,
        commit_footer: str | None = None,
        base_oid: str | None = None,
        runtime_policy: Any | None = None,
        run_id: str | None = None,
    ) -> FakeRun:
        self._run_counter += 1
        ordinal_id = f"run-{self._run_counter}"
        run_id = run_id or ordinal_id
        self._ordinal_to_run_id[ordinal_id] = run_id
        self._run_id_to_ordinal[run_id] = ordinal_id
        # Tests pre-seed outcomes/success by ordinal ("run-1", "run-2", ...)
        # before the real (Executor-supplied, node-format) run id is known;
        # fall back to the ordinal key so that pre-configuration still works.
        success = self._success.get(run_id, self._success.get(ordinal_id, True))
        outcome = self._outcomes.get(
            run_id, self._outcomes.get(ordinal_id, "completed" if success else "failed")
        )
        self._pending_completions.append((run_id, outcome))
        run = FakeRun(state=FakeRunState(run_id=run_id))
        self._runs[run_id] = run
        return run

    def start_run(self, run_id: str) -> None:
        pass

    def request_stop_run(self, run_id: str) -> None:
        self.requested_stops.append(run_id)
        self._runs[run_id].state.stop_requested = True

    def force_stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        self.force_stops.append((run_id, timeout))
        self._runs[run_id].state.force_stop_requested = True
        return True

    def stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        return True

    def list_runs(self) -> list[Any]:
        return []

    def get_run(self, run_id: str) -> Any | None:
        return self._runs.get(run_id)

    def _seeded(self, mapping: dict[str, Any], run_id: str, default: Any) -> Any:
        """Look up a test-pre-seeded value by the real run id, falling back
        to the ordinal placeholder ("run-1", ...) the test seeded it under
        before the real run id was known."""
        if run_id in mapping:
            return mapping[run_id]
        ordinal = self._run_id_to_ordinal.get(run_id)
        if ordinal is not None and ordinal in mapping:
            return mapping[ordinal]
        return default

    def get_run_stdout(self, run_id: str) -> list[str]:
        return self._seeded(self.output, run_id, [])

    def get_run_output_tail(self, run_id: str, max_lines: int) -> list[str]:
        return self._seeded(self.output, run_id, [])[-max_lines:]

    def get_run_guidance(self, run_id: str) -> tuple[str, ...]:
        return self._seeded(self.guidance, run_id, ())

    def queue_guidance(self, run_id: str, text: str) -> bool:
        self.guidance[run_id] = (*self.guidance.get(run_id, ()), text)
        return True

    def get_run_failure_detail(self, run_id: str) -> str | None:
        return None

    def wait_for_next_completion(
        self,
        active_run_ids: set[str],
        timeout: float | None = None,
    ) -> tuple[str, str]:
        if self._progress_before_completion:
            event = self._progress_before_completion.pop(0)
            # Tests pre-seed this event by ordinal ("run-1") before dispatch
            # assigns the real run id; resolve through the ordinal map.
            resolved_run_id = self._ordinal_to_run_id.get(event.run_id, event.run_id)
            if resolved_run_id in active_run_ids:
                if resolved_run_id != event.run_id:
                    event = replace(event, run_id=resolved_run_id)
                return resolved_run_id, event
        for i, (run_id, outcome) in enumerate(self._pending_completions):
            if run_id in active_run_ids:
                self._pending_completions.pop(i)
                return run_id, outcome
        raise RuntimeError("No pending completions for active runs")

    def verify_spec(self, spec_text: str, graph_state: str) -> Any:
        from milknado.domains.common.protocols import VerifySpecResult

        return VerifySpecResult(outcome="done")

    def generate_ralph_md(
        self,
        node: MikadoNode,
        context: str,
        quality_gates: list[str],
        output_path: Path,
        prior_findings: str = "",
        findings_round: int | None = None,
    ) -> Path:
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
        quality_gates=(Gate("uv run pytest"),),
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
    fake_ralph._runs["run-1"] = FakeRun(state=FakeRunState(run_id="run-1", stop_requested=True))
    fake_ralph.output["run-1"] = [f"line {index}" for index in range(35)]
    fake_ralph.guidance["run-1"] = ("use domain barrels",)
    run_loop._active["run-1"] = leaf.id
    run_loop._progress_by_run["run-1"] = ProgressEvent(
        run_id="run-1", work=1, total=2, message="building"
    )
    received: list[RunLoopState] = []
    run_loop.set_state_listener(received.append)
    run_loop._publish_state()

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
    fake_ralph._runs["run-1"] = FakeRun(state=FakeRunState(run_id="run-1", status=status))
    run_loop._active["run-1"] = leaf.id

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
    fake_ralph._runs["run-1"] = FakeRun()
    run_loop._active["run-1"] = leaf.id

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
    graph.add_node("build snapshots", parent_id=root.id)
    fake_ralph._progress_before_completion.append(
        ProgressEvent(run_id="run-1", work=1, total=2, message="building")
    )
    received: list[RunLoopState] = []
    run_loop.set_state_listener(received.append)

    run_loop.run(config, "main")

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
    run_loop._dispatch_batch = dispatch
    run_loop.admit_stop_scheduling()

    run_loop._execute_run(config, "main", concurrency_limit=1, timeout=1.0, interactive=False)

    dispatch.assert_not_called()


def test_initial_dispatch_drains_pending_controls_before_scheduling(
    run_loop: RunLoop,
    config: ExecutionConfig,
) -> None:
    dispatch = MagicMock()
    run_loop._dispatch_batch = dispatch
    dispatch.return_value = (0, 0)
    run_loop._process_controls = run_loop.stop_scheduling

    run_loop._execute_run(config, "main", concurrency_limit=1, timeout=1.0, interactive=False)

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
    run_loop._active["run-1"] = node.id
    run_loop._dispatch_if_scheduling_open = MagicMock(return_value=(0, 0))
    run_loop._handle_completion_timeout = MagicMock(return_value=1)
    control_calls = 0

    def process_controls() -> None:
        nonlocal control_calls
        control_calls += 1
        if control_calls == 3:
            run_loop._active.clear()

    def short_poll_timeout(
        active_run_ids: set[str], timeout: float | None = None
    ) -> tuple[str, str]:
        raise CompletionTimeout(waited_seconds=timeout or 0.0, active_run_ids=active_run_ids)

    fake_ralph.wait_for_next_completion = short_poll_timeout  # type: ignore[method-assign]
    run_loop._process_controls = process_controls

    with patch(
        "milknado.domains.execution.run_loop.time.monotonic",
        side_effect=(100.0, 100.01),
    ):
        run_loop._execute_run(
            config,
            "main",
            concurrency_limit=1,
            timeout=1.0,
            interactive=False,
        )

    run_loop._handle_completion_timeout.assert_not_called()


def test_unset_completion_timeout_polls_controls_without_timing_out(
    run_loop: RunLoop,
    graph: MikadoGraph,
    config: ExecutionConfig,
    fake_ralph: FakeRalph,
) -> None:
    from milknado.domains.common.errors import CompletionTimeout

    node = graph.add_node("active")
    graph.mark_running(node.id)
    run_loop._active["run-1"] = node.id
    run_loop._dispatch_if_scheduling_open = MagicMock(return_value=(0, 0))
    run_loop._handle_completion_timeout = MagicMock(return_value=1)
    observed_timeouts: list[float | None] = []
    control_calls = 0

    def process_controls() -> None:
        nonlocal control_calls
        control_calls += 1
        if control_calls >= 4:
            run_loop._active.clear()

    def short_poll_timeout(
        active_run_ids: set[str], timeout: float | None = None
    ) -> tuple[str, str]:
        observed_timeouts.append(timeout)
        raise CompletionTimeout(waited_seconds=timeout or 0.0, active_run_ids=active_run_ids)

    fake_ralph.wait_for_next_completion = short_poll_timeout  # type: ignore[method-assign]
    run_loop._process_controls = process_controls

    run_loop._execute_run(config, "main", concurrency_limit=1, timeout=None, interactive=False)

    run_loop._handle_completion_timeout.assert_not_called()
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
    assert graph.get_node(root.id).status == NodeStatus.PENDING


class TestRunLoopSingleNode:
    def test_solo_root_not_dispatched(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        # Root is never dispatched as a work node; it's completed via verify_spec.
        # Without spec_text, root stays PENDING and root_done is False.
        graph.add_node("root goal")
        result = run_loop.run(config, "main")

        assert result.dispatched_total == 0
        assert result.completed_total == 0
        assert result.failed_total == 0
        assert result.root_done is False

    def test_solo_root_stays_pending_without_spec(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root goal")
        graph.add_node("leaf", parent_id=root.id)
        run_loop.run(config, "main")

        root_node = graph.get_node(root.id)
        assert root_node is not None
        assert root_node.status == NodeStatus.PENDING


class TestRunLoopParentChild:
    def test_dispatches_leaf_not_root(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        # Root is excluded from dispatch; only the leaf is dispatched.
        # Without spec_text, root stays PENDING.
        root = graph.add_node("root")
        leaf = graph.add_node("leaf", parent_id=root.id)

        result = run_loop.run(config, "main")

        assert result.dispatched_total == 1
        assert result.completed_total == 1
        assert result.root_done is False
        leaf_node = graph.get_node(leaf.id)
        root_node = graph.get_node(root.id)
        assert leaf_node is not None and leaf_node.status == NodeStatus.DONE
        assert root_node is not None and root_node.status == NodeStatus.PENDING

    def test_leaf_done_root_pending_without_spec(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        graph.add_node("leaf", parent_id=root.id)

        run_loop.run(config, "main")

        root_node = graph.get_node(root.id)
        assert root_node is not None
        assert root_node.status == NodeStatus.PENDING


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
        assert loop._stopped_nodes == {leaf.id}
        assert loop._active == {}
        state = loop.state()
        assert state.stopped == 1
        assert state.available == 0
        assert len(state.terminal_runs) == 1
        terminal = state.terminal_runs[0]
        real_run_id = ralph._ordinal_to_run_id["run-1"]
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
            graph.add_node(f"leaf-{index}", parent_id=root.id)

        loop.run(config, "main", interactive=False)

        state = loop.state()
        assert state.stopped == 21
        assert len(state.terminal_runs) == 20
        assert tuple(run.run_id for run in state.terminal_runs) == tuple(
            ralph._ordinal_to_run_id[f"run-{index}"] for index in range(2, 22)
        )


class TestRunLoopParallelLeaves:
    def test_dispatches_parallel_leaves_not_root(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.add_node("leaf-b", parent_id=root.id)

        result = run_loop.run(config, "main")

        assert result.dispatched_total == 2
        assert result.completed_total == 2
        # Root not dispatched; stays PENDING without spec_text.
        assert result.root_done is False


class TestRunLoopConcurrencyLimit:
    def test_respects_concurrency_limit(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        graph.add_node("a", parent_id=root.id)
        graph.add_node("b", parent_id=root.id)
        graph.add_node("c", parent_id=root.id)

        result = run_loop.run(config, "main", concurrency_limit=2)

        # 3 leaves dispatched, root excluded.
        assert result.dispatched_total == 3
        assert result.completed_total == 3
        assert result.root_done is False


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
        ralph._success["run-1"] = False

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
        ralph._success["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        graph.add_node("leaf", parent_id=root.id)
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
        graph: MikadoGraph,
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
        ralph.generate_ralph_md = lambda *_a, **_kw: (_ for _ in ()).throw(  # type: ignore
            RuntimeError("ralph exploded"),
        )
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

        def fail_first_only(*args: Any, **kwargs: Any) -> Path:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("ralph exploded")
            return original_generate(*args, **kwargs)

        ralph.generate_ralph_md = fail_first_only  # type: ignore
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        graph.add_node("doomed-leaf", parent_id=root.id)
        graph.add_node("good-leaf", parent_id=root.id)

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
        graph.add_node("clean node", parent_id=root.id)
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
        graph.set_file_ownership(a.id, ["shared.py"])
        graph.set_file_ownership(b.id, ["shared.py"])

        result = loop.run(config, "main")

        # 2 leaves dispatched (serialized due to file conflict), root excluded.
        assert result.dispatched_total == 2
        assert result.completed_total == 2
        assert result.root_done is False


_RICH_DESC = "US-204: split bundling\n\n## Reuse candidates\n- foo.py:123\n- bar.py:45"


def _make_tui_state(node_id: int) -> TuiState:
    return TuiState(
        tick=0,
        active={"run-1": node_id},
        logs=[],
        dispatched_at={"run-1": time.monotonic()},
        attempts={},
        progress_by_run={},
        completion_durations=[],
        stall_threshold=300.0,
        max_retries=2,
        exec_agent="claude",
    )


def _render_to_text(renderable: Any) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, width=200)
    console.print(renderable)
    return buf.getvalue()


def _snapshot(renderable: Any) -> str:
    console = Console(record=True, width=200)
    console.print(renderable)
    return console.export_text()


class TestWorkerTableDescriptionSanitization:
    def test_description_cell_is_summarized(self, graph: MikadoGraph) -> None:
        node = graph.add_node(_RICH_DESC)
        state = _make_tui_state(node.id)
        rendered = _render_to_text(_build_worker_table(state, graph))

        assert "split bundling" in rendered
        assert "##" not in rendered
        assert "Reuse" not in rendered

    def test_description_cell_has_no_raw_newlines(self, graph: MikadoGraph) -> None:
        node = graph.add_node(_RICH_DESC)
        state = _make_tui_state(node.id)
        rendered = _render_to_text(_build_worker_table(state, graph))

        assert "\n\n" not in rendered


class TestRenderOverlayPreservesRawDescription:
    """Overlay panel title uses _summarize_description (consistent with worker table)."""

    def test_overlay_title_collapses_newlines(self, graph: MikadoGraph) -> None:
        node = graph.add_node(_RICH_DESC)
        state = _make_tui_state(node.id)
        ralph = FakeRalph()
        panel = _render_overlay("run-1", state, graph, ralph)

        assert "\n" not in (panel.title or "")
        assert "split bundling" in (panel.title or "")

    def test_overlay_sanitization_is_nondestructive(self, graph: MikadoGraph) -> None:
        node = graph.add_node(_RICH_DESC)
        state = _make_tui_state(node.id)
        _build_worker_table(state, graph)

        fresh = graph.get_node(node.id)
        assert fresh is not None
        assert "##" in fresh.description
        assert "- foo.py:123" in fresh.description
        assert "- bar.py:45" in fresh.description


# ---------------------------------------------------------------------------
# US-208: headless TUI snapshot tests
# ---------------------------------------------------------------------------


class TestBuildTitle:
    def test_shows_done_count(self, graph: MikadoGraph) -> None:
        n = graph.add_node("done-node")
        graph.mark_running(n.id)
        graph.mark_done(n.id)

        text = _snapshot(_build_title({}, graph))

        assert "1 done" in text

    def test_shows_failed_count(self, graph: MikadoGraph) -> None:
        n = graph.add_node("fail-node")
        graph.mark_failed(n.id)

        text = _snapshot(_build_title({}, graph))

        assert "1 failed" in text

    def test_shows_blocked_count(self, graph: MikadoGraph) -> None:
        n = graph.add_node("blocked-node")
        graph.mark_blocked(n.id)

        text = _snapshot(_build_title({}, graph))

        assert "1 blocked" in text

    def test_all_counts_aggregate(self, graph: MikadoGraph) -> None:
        done_n = graph.add_node("done")
        graph.mark_running(done_n.id)
        graph.mark_done(done_n.id)

        fail_n = graph.add_node("fail")
        graph.mark_failed(fail_n.id)

        blocked_n = graph.add_node("blocked")
        graph.mark_blocked(blocked_n.id)

        text = _snapshot(_build_title({}, graph))

        assert "1 done" in text
        assert "1 failed" in text
        assert "1 blocked" in text


class TestRenderProgressBar:
    def test_normal_returns_spinner_frame(self) -> None:
        result = _render_progress_bar("◜", elapsed=0.0, pct=None, stall_threshold=300.0)

        assert "◜" in result
        assert "⚠" not in result

    def test_stalled_includes_warning_glyph(self) -> None:
        result = _render_progress_bar("◜", elapsed=400.0, pct=None, stall_threshold=300.0)

        assert "⚠" in result

    def test_with_pct_shows_bar_and_percentage(self) -> None:
        result = _render_progress_bar("◜", elapsed=10.0, pct=70.0, stall_threshold=300.0)

        assert "70%" in result
        assert "█" in result

    def test_completed_full_bar(self) -> None:
        result = _render_progress_bar("◜", elapsed=30.0, pct=100.0, stall_threshold=300.0)

        assert "100%" in result
        assert "░" not in result

    def test_pct_over_100_clamped_to_100(self) -> None:
        result = _render_progress_bar("◜", elapsed=10.0, pct=150.0, stall_threshold=300.0)
        # Bar chars should be exactly 10 total, percentage clamped to 100
        bar_chars = result.count("█") + result.count("░")
        assert bar_chars == 10
        assert "100%" in result

    def test_pct_negative_clamped_to_0(self) -> None:
        result = _render_progress_bar("◜", elapsed=10.0, pct=-5.0, stall_threshold=300.0)
        bar_chars = result.count("█") + result.count("░")
        assert bar_chars == 10
        assert "0%" in result

    def test_pct_50_sanity_check(self) -> None:
        result = _render_progress_bar("◜", elapsed=10.0, pct=50.0, stall_threshold=300.0)
        bar_chars = result.count("█") + result.count("░")
        assert bar_chars == 10
        assert "50%" in result


class TestBuildWorkerTableColumns:
    def test_elapsed_column_present(self, graph: MikadoGraph) -> None:
        node = graph.add_node("simple task")
        state = _make_tui_state(node.id)
        text = _snapshot(_build_worker_table(state, graph))

        assert "Elapsed" in text

    def test_eta_column_shows_unknown_when_no_history(self, graph: MikadoGraph) -> None:
        node = graph.add_node("simple task")
        state = _make_tui_state(node.id)
        text = _snapshot(_build_worker_table(state, graph))

        assert "~?" in text

    def test_files_column_shows_owned_files(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task with files")
        graph.set_file_ownership(node.id, ["src/foo.py"])
        state = _make_tui_state(node.id)
        text = _snapshot(_build_worker_table(state, graph))

        assert "src/foo.py" in text

    def test_attempt_column_empty_on_first_attempt(self, graph: MikadoGraph) -> None:
        node = graph.add_node("fresh task")
        state = _make_tui_state(node.id)
        text = _snapshot(_build_worker_table(state, graph))

        assert "1/" not in text

    def test_attempt_column_shows_ratio_on_retry(self, graph: MikadoGraph) -> None:
        node = graph.add_node("retried task")
        state = TuiState(
            tick=0,
            active={"run-1": node.id},
            logs=[],
            dispatched_at={"run-1": time.monotonic()},
            attempts={node.id: 1},
            progress_by_run={},
            completion_durations=[],
            stall_threshold=300.0,
            max_retries=2,
            exec_agent="claude",
        )
        text = _snapshot(_build_worker_table(state, graph))

        assert "2/3" in text

    def test_description_is_single_line(self, graph: MikadoGraph) -> None:
        node = graph.add_node(_RICH_DESC)
        state = _make_tui_state(node.id)
        text = _snapshot(_build_worker_table(state, graph))

        lines_with_desc = [ln for ln in text.splitlines() if "split bundling" in ln]
        assert len(lines_with_desc) == 1


class _RalphWithStdout(FakeRalph):
    def __init__(self, lines: list[str]) -> None:
        super().__init__()
        self._lines = lines

    def get_run_stdout(self, run_id: str) -> list[str]:
        return self._lines

    def get_run_output_tail(self, run_id: str, max_lines: int) -> list[str]:
        return self._lines[-max_lines:]


class TestRenderOverlayLogLines:
    def test_stdout_lines_appear_in_overlay(self, graph: MikadoGraph) -> None:
        node = graph.add_node("node-with-output")
        state = _make_tui_state(node.id)
        ralph = _RalphWithStdout(["line alpha", "line beta", "line gamma"])
        panel = _render_overlay("run-1", state, graph, ralph)
        text = _snapshot(panel)

        assert "line alpha" in text
        assert "line gamma" in text

    def test_only_last_100_stdout_lines_shown(self, graph: MikadoGraph) -> None:
        node = graph.add_node("many-lines")
        state = _make_tui_state(node.id)
        ralph = _RalphWithStdout([f"log-line-{i}" for i in range(150)])
        panel = _render_overlay("run-1", state, graph, ralph)
        text = _snapshot(panel)

        assert "log-line-149" in text
        assert "log-line-0" not in text

    def test_overlay_title_is_summarized(self, graph: MikadoGraph) -> None:
        node = graph.add_node(_RICH_DESC)
        state = _make_tui_state(node.id)
        ralph = FakeRalph()
        panel = _render_overlay("run-1", state, graph, ralph)

        assert "US-204" not in (panel.title or "")
        assert "\n" not in (panel.title or "")


# ---------------------------------------------------------------------------
# US-207: four integration paths
# ---------------------------------------------------------------------------


class TestStrictDrain:
    def test_no_new_dispatch_after_failure_in_flight_completes(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        ralph = FakeRalph()
        ralph._success["run-1"] = False  # leaf-a fails, leaf-b (run-2) succeeds

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.add_node("leaf-b", parent_id=root.id)

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
        graph.add_node("leaf-a", parent_id=root.id)
        graph.add_node("leaf-b", parent_id=root.id)

        result = loop.run(none_gates_config, "main", strict=True)

        # Preflight on leaf-a fails → strict stops after 1 failure; leaf-b never dispatched
        assert result.failed_total >= 1
        assert result.strict_exit is True


class TestProtectedBranchGuard:
    def test_protected_branch_returns_typed_refusal_before_log_created(
        self, tmp_path: Path
    ) -> None:
        from milknado.app.run import ProtectedBranchRefusal, check_protected_branch
        from milknado.domains.common.config import MilknadoConfig

        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            protected_branches=("main", "master"),
        )
        refusal = check_protected_branch(cfg, "main", allow_protected=False)

        assert refusal == ProtectedBranchRefusal(branch="main", reason="protected")
        assert not any((tmp_path / ".milknado").glob("run-*.log"))

    def test_protected_branch_with_allow_protected_does_not_raise(self, tmp_path: Path) -> None:
        from milknado.app.run import check_protected_branch
        from milknado.domains.common.config import MilknadoConfig

        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            protected_branches=("main", "master"),
        )

        # Explicit opt-in past the guard: no exception, returns None.
        assert check_protected_branch(cfg, "main", allow_protected=True) is None

    def test_unprotected_branch_does_not_raise(self, tmp_path: Path) -> None:
        from milknado.app.run import check_protected_branch
        from milknado.domains.common.config import MilknadoConfig

        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            protected_branches=("main", "master"),
        )

        assert check_protected_branch(cfg, "feature-x", allow_protected=False) is None

    def test_second_protected_branch_also_returns_typed_refusal(self, tmp_path: Path) -> None:
        from milknado.app.run import ProtectedBranchRefusal, check_protected_branch
        from milknado.domains.common.config import MilknadoConfig

        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            protected_branches=("main", "master"),
        )

        assert check_protected_branch(cfg, "master", allow_protected=False) == (
            ProtectedBranchRefusal(branch="master", reason="protected")
        )

    def test_detached_head_refused_even_with_allow_protected(self, tmp_path: Path) -> None:
        from milknado.app.run import ProtectedBranchRefusal, check_protected_branch
        from milknado.domains.common.config import MilknadoConfig

        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            protected_branches=("main", "master"),
        )

        for branch in ("HEAD", ""):
            assert check_protected_branch(cfg, branch, allow_protected=True) == (
                ProtectedBranchRefusal(branch=branch, reason="detached")
            )


class TestStalledWorkerGlyph:
    def test_stalled_worker_shows_warning_glyph_in_table(self, graph: MikadoGraph) -> None:
        node = graph.add_node("long-running task")
        past = time.monotonic() - 400.0
        state = TuiState(
            tick=0,
            active={"run-1": node.id},
            logs=[],
            dispatched_at={"run-1": past},
            attempts={},
            progress_by_run={},
            completion_durations=[],
            stall_threshold=300.0,
            max_retries=2,
            exec_agent="claude",
        )
        rendered = _render_to_text(_build_worker_table(state, graph))

        assert "⚠" in rendered


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

        def fail_thrice(*args: Any, **kwargs: Any) -> Path:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise TransientDispatchError("rate limited")
            return original_generate(*args, **kwargs)

        ralph.generate_ralph_md = fail_thrice  # type: ignore

        executor = _Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)

        clean_calls: list[int] = []
        worktree_sizes: list[int] = []
        original_ensure = executor._wt.ensure_clean

        def tracked_ensure(node_id: int) -> None:
            worktree_sizes.append(len(executor._wt._worktrees))
            clean_calls.append(node_id)
            return original_ensure(node_id)

        executor._wt.ensure_clean = tracked_ensure  # type: ignore

        retry_config = ExecutionConfig(
            execution_agent="claude",
            quality_gates=("uv run pytest",),
            worktree_pattern="milknado-{node_id}-{slug}",
            project_root=tmp_path,
            dispatch_max_retries=3,
            dispatch_backoff_seconds=0.0,
        )

        graph.add_node("transient node")
        executor.dispatch(1, retry_config)

        # Called once at the start of each _dispatch_once attempt (3 fail + 1 success = 4)
        assert len(clean_calls) == 4
        # At no point when ensure_clean fires are there stale entries
        assert all(sz == 0 for sz in worktree_sizes)
        # Final state: successful dispatch recorded, not cleared
        assert 1 in executor._wt._worktrees


class TestBuildLogPanel:
    def test_deque_maxlen_30_truncates_older_entries(self) -> None:
        logs: collections.deque[str] = collections.deque(maxlen=30)
        for i in range(40):
            logs.append(f"entry-{i}")
        text = _snapshot(_build_log_panel(logs))

        assert "entry-39" in text
        assert "entry-9" not in text

    def test_deque_within_maxlen_shows_all(self) -> None:
        logs: collections.deque[str] = collections.deque(maxlen=30)
        for i in range(20):
            logs.append(f"item-{i}")
        text = _snapshot(_build_log_panel(logs))

        assert "item-0" in text
        assert "item-19" in text

    def test_no_mid_line_cuts(self) -> None:
        complete_entry = "start:end"
        logs: collections.deque[str] = collections.deque(maxlen=30)
        logs.append(complete_entry)
        text = _snapshot(_build_log_panel(logs))

        assert "start:end" in text

    def test_empty_logs_shows_placeholder(self) -> None:
        text = _snapshot(_build_log_panel([]))

        assert "No events yet" in text


# ---------------------------------------------------------------------------
# Root completion via _maybe_verify_spec (not via dispatch)
# ---------------------------------------------------------------------------


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

        def tracking_dispatch(node_id: int, cfg: Any) -> Any:
            dispatched_ids.append(node_id)
            return original_dispatch(node_id, cfg)

        executor.dispatch = tracking_dispatch  # type: ignore
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root goal")
        graph.add_node("leaf", parent_id=root.id)

        loop.run(config, "main", spec_text="spec: do the thing")

        assert root.id not in dispatched_ids

    def test_root_stays_pending_when_leaves_fail(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        ralph = FakeRalph()
        ralph._success["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root goal")
        graph.add_node("leaf", parent_id=root.id)

        result = loop.run(config, "main", spec_text="spec: do the thing")

        assert result.root_done is False
        root_node = graph.get_node(root.id)
        assert root_node is not None
        assert root_node.status == NodeStatus.PENDING


# ---------------------------------------------------------------------------
# LoopAdapter.create_run passes log_dir
# ---------------------------------------------------------------------------


class TestLoopAdapterLogDir:
    def test_create_run_passes_log_dir_under_worktree(
        self,
        tmp_path: Path,
    ) -> None:
        from unittest.mock import MagicMock

        ralph_dir = tmp_path / "wt-node-1"
        ralph_dir.mkdir()
        ralph_file = ralph_dir / "ralph.md"
        ralph_file.write_text("# task", encoding="utf-8")

        captured_configs: list[Any] = []

        def fake_create_run(config: Any, *, emitter: Any, run_id: str | None = None) -> Any:
            assert emitter is adapter._emitter
            captured_configs.append(config)
            fake_run = MagicMock()
            fake_run.state.run_id = run_id or "run-test"
            return fake_run

        from milknado.adapters.loop import LoopAdapter

        adapter = LoopAdapter(agent="claude")

        with patch.object(adapter._manager, "create_run", side_effect=fake_create_run):
            adapter.create_run(
                agent="claude",
                ralph_dir=ralph_dir,
                ralph_file=ralph_file,
                commands=[],
                quality_gates=[],
                project_root=None,
            )

        assert len(captured_configs) == 1
        cfg = captured_configs[0]
        assert cfg.log_dir == ralph_dir / ".ralph-logs"


# ---------------------------------------------------------------------------
# Coverage helpers: display.py missing branches
# ---------------------------------------------------------------------------


class TestEtaStr:
    def test_shows_unknown_when_no_history(self) -> None:
        from milknado.domains.execution.run_loop.display import _eta_str

        assert _eta_str(None, 0.0) == "~?"

    def test_shows_estimate_when_avg_provided(self) -> None:
        from milknado.domains.execution.run_loop.display import _eta_str

        result = _eta_str(120.0, 30.0)
        assert result.startswith("~")
        assert "01:30" in result or "00:" in result  # remaining ≈ 90s

    def test_clamps_to_zero_when_elapsed_exceeds_avg(self) -> None:
        from milknado.domains.execution.run_loop.display import _eta_str

        result = _eta_str(60.0, 200.0)
        assert "00:00" in result


class TestSummarizeDescriptionTruncation:
    def test_long_description_is_truncated(self) -> None:
        from milknado.domains.execution.run_loop.display import _summarize_description

        long_text = "A" * 100
        result = _summarize_description(long_text, max_chars=80)
        assert len(result) <= 80
        assert result.endswith("…")

    def test_short_description_unchanged(self) -> None:
        from milknado.domains.execution.run_loop.display import _summarize_description

        result = _summarize_description("short task")
        assert result == "short task"


class TestBuildWorkerTableMissingNode:
    def test_row_skipped_when_node_missing_from_graph(self, graph: MikadoGraph) -> None:
        import time as _time

        from milknado.domains.execution.run_loop.display import TuiState, _build_worker_table

        # active has node_id 999 which doesn't exist in graph
        state = TuiState(
            tick=0,
            active={"run-1": 999},
            logs=[],
            dispatched_at={"run-1": _time.monotonic()},
            attempts={},
            progress_by_run={},
            completion_durations=[],
            stall_threshold=300.0,
            max_retries=2,
            exec_agent="claude",
        )
        table = _build_worker_table(state, graph)
        # Should not crash; row count is 0 (skipped)
        assert table.row_count == 0


class TestRenderOverlayMissingRunId:
    def test_returns_not_found_panel(self, graph: MikadoGraph) -> None:
        from milknado.domains.execution.run_loop.display import TuiState, _render_overlay

        state = TuiState(
            tick=0,
            active={},
            logs=[],
            dispatched_at={},
            attempts={},
            progress_by_run={},
            completion_durations=[],
            stall_threshold=300.0,
            max_retries=2,
            exec_agent="claude",
        )
        ralph = FakeRalph()
        panel = _render_overlay("unknown-run", state, graph, ralph)
        text = _snapshot(panel)
        assert "worker not found" in text or "Overlay" in text


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
        ralph.wait_for_next_completion = MagicMock(wraps=ralph.wait_for_next_completion)
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        root = graph.add_node("root")
        graph.add_node("slow-leaf", parent_id=root.id)

        result = loop.run(config, "main")

        assert result.completed_total == 1
        assert ralph.wait_for_next_completion.call_args.kwargs["timeout"] is None

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
            loop.run(config, "main")

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
        def raise_timeout(active_run_ids: set[str], timeout: float | None = None) -> tuple:
            raise CompletionTimeout(waited_seconds=60.0, active_run_ids=active_run_ids)

        ralph.wait_for_next_completion = raise_timeout  # type: ignore

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        graph.add_node("slow-leaf", parent_id=root.id)

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

        def raise_timeout(active_run_ids: set[str], timeout: float | None = None) -> tuple:
            if first_call[0]:
                first_call[0] = False
                raise CompletionTimeout(waited_seconds=60.0, active_run_ids=active_run_ids)
            return next(iter(active_run_ids)), True

        ralph.wait_for_next_completion = raise_timeout  # type: ignore

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        graph.add_node("slow-leaf", parent_id=root.id)

        result = loop.run(config, "main", strict=True)

        assert result.strict_exit is True

    def test_timeout_preserves_active_ownership_until_worker_exits(
        self,
        graph: MikadoGraph,
    ) -> None:
        from milknado.domains.common.errors import CompletionTimeout

        executor = MagicMock()
        ralph = FakeRalph()
        ralph.stop_run = MagicMock(return_value=False)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        loop._active = {"run-1": 7}

        failed = loop._handle_completion_timeout(
            CompletionTimeout(waited_seconds=60.0, active_run_ids={"run-1"})
        )

        assert failed == 0
        assert loop._active == {"run-1": 7}
        executor.fail.assert_not_called()
        ralph.stop_run.assert_called_once_with("run-1", timeout=10.0)


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
        ralph.verify_spec = lambda spec, graph_str: VerifySpecResult(  # type: ignore
            outcome="gaps", goal_delta="add feature X"
        )

        from milknado.domains.planning.planner import Planner

        planner = MagicMock(spec=Planner)
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph, planner=planner)

        root = graph.add_node("root")
        graph.add_node("leaf", parent_id=root.id)

        loop.run(config, "main", spec_text="spec text")

        planner.replan_with_delta.assert_called_once()

    def test_root_already_done_skips_verify(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        ralph = FakeRalph()
        verify_calls: list = []
        ralph.verify_spec = lambda *a: (  # ty: ignore[invalid-assignment]
            verify_calls.append(a)
            or __import__(
                "milknado.domains.common.protocols", fromlist=["VerifySpecResult"]
            ).VerifySpecResult(outcome="done")
        )

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        graph.mark_running(root.id)
        graph.mark_done(root.id)

        loop.run(config, "main", spec_text="spec text")

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
        graph.add_node("leaf-a", parent_id=root.id)
        graph.add_node("leaf-b", parent_id=root.id)

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

        def raise_interrupt(active_run_ids: set[str], timeout: float | None = None) -> tuple:
            raise KeyboardInterrupt

        ralph.wait_for_next_completion = raise_interrupt  # type: ignore

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        graph.add_node("leaf", parent_id=root.id)

        with pytest.raises(KeyboardInterrupt):
            loop.run(config, "main")


class TestDispatchBatchDirectGuards:
    def test_strict_failure_triggered_returns_zero(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_ralph: FakeRalph,
    ) -> None:
        from unittest.mock import MagicMock

        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph)
        loop._strict = True
        loop._failure_triggered = True

        live = MagicMock()
        result = loop._dispatch_batch(config, 4, live)

        assert result == (0, 0)

    def test_no_capacity_returns_zero(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_ralph: FakeRalph,
    ) -> None:
        from unittest.mock import MagicMock

        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph)
        # Fill active to the limit
        loop._active = {"run-1": 1, "run-2": 2, "run-3": 3, "run-4": 4}

        live = MagicMock()
        result = loop._dispatch_batch(config, 4, live)

        assert result == (0, 0)

    def test_dispatch_exception_increments_failed(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_ralph: FakeRalph,
    ) -> None:
        from unittest.mock import MagicMock

        executor = MagicMock()
        executor.dispatch.side_effect = RuntimeError("boom")
        root = graph.add_node("root")
        graph.add_node("failing-leaf", parent_id=root.id)

        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph)
        live = MagicMock()
        dispatched, failed = loop._dispatch_batch(config, 4, live)

        assert dispatched == 0
        assert failed == 1

    def test_strict_mode_breaks_on_dispatch_exception(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_ralph: FakeRalph,
    ) -> None:
        from unittest.mock import MagicMock

        executor = MagicMock()
        executor.dispatch.side_effect = RuntimeError("boom")
        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.add_node("leaf-b", parent_id=root.id)

        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph)
        loop._strict = True
        live = MagicMock()
        loop._dispatch_batch(config, 4, live)

        assert executor.dispatch.call_count == 1
        assert loop._failure_triggered is True


class TestRenderLiveFrameOverlayBranch:
    def test_overlay_branch_used_when_overlay_state_set(
        self,
        executor: Executor,
        graph: MikadoGraph,
        fake_ralph: FakeRalph,
    ) -> None:
        from unittest.mock import MagicMock

        from milknado.domains.execution.run_loop import RunLoop

        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph)
        loop._input.overlay_state = "run-xyz"

        live_mock = MagicMock()
        # Should call _render_overlay (not _build_layout); won't crash even with no such run_id
        loop._render_live_frame(live_mock)

        live_mock.update.assert_called_once()


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
                "research": FlavorOverride(quality_gates=()),
            },
        )

        root = graph.add_node("root")
        graph.add_node(
            "research leaf",
            parent_id=root.id,
            spec=NodeSpec(flavor="research"),
        )

        captured: list = []

        executor = MagicMock()
        executor.dispatch.side_effect = lambda node_id, cfg: (
            captured.append(cfg) or MagicMock(run_id="r1")
        )

        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph, config=milknado_cfg)
        live = MagicMock()
        loop._dispatch_batch(config, 4, live)

        assert len(captured) == 1, "expected exactly one dispatch call"
        assert captured[0].quality_gates == (), (
            "flavored node with quality_gates=() must propagate empty gates to executor"
        )
