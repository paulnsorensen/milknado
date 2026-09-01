"""US-006: dispatched_at capture and completion_duration_seconds persistence."""

from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from milknado.domains.common import ProgressEvent, TerminalRunOutcome, VerifySpecResult
from milknado.domains.common.config import Gate
from milknado.domains.common.types import NodeStatus, RebaseResult
from milknado.domains.execution import ExecutionConfig, Executor
from milknado.domains.graph import MikadoGraph
from milknado.loop import RunStatus

# ---------------------------------------------------------------------------
# Minimal fakes (mirrors test_execution.py without reimporting)
# ---------------------------------------------------------------------------


@dataclass
class _FakeRunState:
    run_id: str = "run-dur-1"
    status: RunStatus = RunStatus.RUNNING
    total: int = 0
    stop_requested: bool = False
    force_stop_requested: bool = False


@dataclass
class _FakeRun:
    state: _FakeRunState = field(default_factory=_FakeRunState)


@dataclass(frozen=True)
class _FakeReview:
    approved: bool = True
    findings_md: str = ""


class _FakeGit:
    def __init__(self) -> None:
        self.rebase_result: RebaseResult = RebaseResult(success=True)

    def branch_exists(self, branch: str) -> bool:
        _ = branch
        return False

    def create_worktree(self, path: Path, branch: str) -> Path:
        _ = branch
        return path

    def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
        _ = (path, target)

    def force_remove_worktree(self, path: Path) -> None:
        _ = path

    def worktree_teardown_blocker(self, path: Path, target: str = "HEAD") -> str | None:
        _ = (path, target)
        return None

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


class _FakeRalph:
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
    ) -> _FakeRun:
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
        return _FakeRun(state=_FakeRunState(run_id=run_id or "run-dur-1"))

    def start_run(self, run_id: str) -> None:
        _ = run_id

    def queue_guidance(self, run_id: str, text: str) -> bool:
        _ = (run_id, text)
        return True

    def request_stop_run(self, run_id: str) -> None:
        _ = run_id

    def stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        _ = (run_id, timeout)
        return True

    def force_stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        _ = (run_id, timeout)
        return True

    def list_runs(self) -> list[_FakeRun]:
        return []

    def get_run(self, run_id: str) -> _FakeRun | None:
        _ = run_id
        return None

    def is_run_alive(self, run_id: str) -> bool:
        _ = run_id
        return False

    def get_run_stdout(self, run_id: str) -> list[str]:
        _ = run_id
        return []

    def get_run_failure_detail(self, run_id: str) -> str | None:
        _ = run_id
        return None

    def get_run_output_tail(self, run_id: str, max_lines: int) -> list[str]:
        _ = (run_id, max_lines)
        return []

    def get_run_guidance(self, run_id: str) -> tuple[str, ...]:
        _ = run_id
        return ()

    def wait_for_next_completion(
        self,
        active_run_ids: set[str],
        timeout: float | None = None,
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]:
        _ = (active_run_ids, timeout)
        raise RuntimeError("not used")

    def run_node_review(
        self, agent: str, prompt: str, worktree: Path, project_root: Path
    ) -> _FakeReview:
        _ = (agent, prompt, worktree, project_root)
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


class _FakeCrg:
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph(tmp_path: Path) -> Generator[MikadoGraph, None, None]:
    g = MikadoGraph(tmp_path / "dur.db")
    yield g
    g.close()


@pytest.fixture()
def config(tmp_path: Path) -> ExecutionConfig:
    return ExecutionConfig(
        execution_agent="claude",
        quality_gates=(Gate(command="uv run pytest"),),
        worktree_pattern="milknado-{node_id}-{slug}",
        project_root=tmp_path,
    )


@pytest.fixture()
def executor(graph: MikadoGraph) -> Executor:
    return Executor(graph=graph, git=_FakeGit(), ralph=_FakeRalph(), crg=_FakeCrg())


# ---------------------------------------------------------------------------
# Tests: dispatched_at captured on dispatch
# ---------------------------------------------------------------------------


class TestDispatchedAtCapture:
    def test_dispatched_at_set_after_dispatch(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        before = datetime.now(UTC)
        _ = graph.add_node("compute thing")
        _ = executor.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.dispatched_at is not None
        assert node.dispatched_at >= before

    def test_dispatched_at_is_utc(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        _ = graph.add_node("write tests")
        _ = executor.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.dispatched_at is not None
        assert node.dispatched_at.tzinfo is not None

    def test_dispatched_at_not_set_before_dispatch(
        self,
        graph: MikadoGraph,
    ) -> None:
        _ = graph.add_node("not yet dispatched")
        node = graph.get_node(1)
        assert node is not None
        assert node.dispatched_at is None


# ---------------------------------------------------------------------------
# Tests: duration written on completion
# ---------------------------------------------------------------------------


class TestCompletionDuration:
    def test_duration_written_after_complete(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        _ = graph.add_node("feature")
        _ = executor.dispatch(1, config)
        _ = executor.complete(1, "main")
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.DONE

        durations = graph.recent_completion_durations(limit=10)
        assert len(durations) == 1
        assert durations[0] >= 0.0

    def test_duration_is_non_negative(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        _ = graph.add_node("fast task")
        _ = executor.dispatch(1, config)
        _ = executor.complete(1, "main")
        durations = graph.recent_completion_durations(limit=10)
        assert all(d >= 0.0 for d in durations)

    def test_duration_not_written_on_rebase_failure(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_git = _FakeGit()
        fake_git.rebase_result = RebaseResult(success=False)
        ex = Executor(graph=graph, git=fake_git, ralph=_FakeRalph(), crg=_FakeCrg())
        _ = graph.add_node("conflicted task")
        _ = ex.dispatch(1, config)

        # Create the worktree dir so _rebase_and_merge actually calls git.rebase
        node = graph.get_node(1)
        assert node is not None and node.worktree_path is not None
        Path(node.worktree_path).mkdir(parents=True, exist_ok=True)

        _ = ex.complete(1, "main")

        durations = graph.recent_completion_durations(limit=10)
        assert durations == []


# ---------------------------------------------------------------------------
# Tests: missing dispatched_at treated as null (no duration written)
# ---------------------------------------------------------------------------


class TestMissingDispatchedAt:
    def test_no_duration_written_when_dispatched_at_missing(
        self,
        graph: MikadoGraph,
    ) -> None:
        _ = graph.add_node("manually inserted node")
        graph.mark_running(1)
        graph.mark_done(1)

        durations = graph.recent_completion_durations(limit=10)
        assert durations == []

    def test_direct_complete_without_dispatch_skips_duration(
        self,
        executor: Executor,
        graph: MikadoGraph,
    ) -> None:
        _ = graph.add_node("bypassed dispatch")
        graph.mark_running(1)
        # complete() reads dispatched_at from the node; it's None here
        _ = executor.complete(1, "main")

        durations = graph.recent_completion_durations(limit=10)
        assert durations == []


# ---------------------------------------------------------------------------
# Tests: histogram / recent_completion_durations with 0 / 1 / many nodes
# ---------------------------------------------------------------------------


class TestRecentCompletionDurations:
    def test_empty_returns_empty_list(self, graph: MikadoGraph) -> None:
        assert graph.recent_completion_durations(limit=10) == []

    def test_single_completed_node(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        _ = graph.add_node("one")
        _ = executor.dispatch(1, config)
        _ = executor.complete(1, "main")
        durations = graph.recent_completion_durations(limit=10)
        assert len(durations) == 1

    def test_many_completed_nodes(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        for desc in ("alpha", "beta", "gamma"):
            _ = graph.add_node(desc)

        for node_id in (1, 2, 3):
            _ = executor.dispatch(node_id, config)
            _ = executor.complete(node_id, "main")

        durations = graph.recent_completion_durations(limit=10)
        assert len(durations) == 3
        assert all(isinstance(d, float) for d in durations)

    def test_limit_is_respected(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        for desc in ("a", "b", "c", "d", "e"):
            _ = graph.add_node(desc)

        for node_id in range(1, 6):
            _ = executor.dispatch(node_id, config)
            _ = executor.complete(node_id, "main")

        durations = graph.recent_completion_durations(limit=3)
        assert len(durations) == 3

    def test_excludes_nodes_without_duration(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        _ = graph.add_node("completed")
        _ = graph.add_node("pending")
        _ = executor.dispatch(1, config)
        _ = executor.complete(1, "main")

        durations = graph.recent_completion_durations(limit=10)
        assert len(durations) == 1
