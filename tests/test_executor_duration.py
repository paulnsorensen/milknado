"""US-006: dispatched_at capture and completion_duration_seconds persistence."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from milknado.domains.common.types import MikadoNode, NodeStatus, RebaseResult
from milknado.domains.execution import ExecutionConfig, Executor
from milknado.domains.graph import MikadoGraph

# ---------------------------------------------------------------------------
# Minimal fakes (mirrors test_execution.py without reimporting)
# ---------------------------------------------------------------------------


@dataclass
class _FakeRunState:
    run_id: str = "run-dur-1"


@dataclass
class _FakeRun:
    state: _FakeRunState = field(default_factory=_FakeRunState)


class _FakeGit:
    def __init__(self) -> None:
        self.rebase_result: RebaseResult = RebaseResult(success=True)

    def create_worktree(self, path: Path, branch: str) -> Path:
        return path

    def remove_worktree(self, path: Path) -> None:
        pass

    def rebase(self, worktree: Path, onto: str) -> RebaseResult:
        return self.rebase_result

    def current_branch(self) -> str:
        return "main"

    def commit_all(self, worktree: Path, message: str) -> None:
        pass

    def squash_and_commit(self, worktree: Path, onto: str, msg: str) -> None:
        pass


class _FakeRalph:
    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: list[str],
        project_root: Path | None = None,
    ) -> _FakeRun:
        return _FakeRun()

    def start_run(self, run_id: str) -> None:
        pass

    def stop_run(self, run_id: str) -> None:
        pass

    def list_runs(self) -> list[Any]:
        return []

    def get_run(self, run_id: str) -> Any | None:
        return None

    def get_run_stdout(self, run_id: str) -> list[str]:
        return []

    def wait_for_next_completion(
        self,
        active_run_ids: set[str],
        timeout: float | None = None,
    ) -> tuple[str, bool]:
        raise RuntimeError("not used")

    def poll_progress_events(self) -> list[Any]:
        return []

    def verify_spec(self, spec_text: str, graph_state: str) -> Any:
        from milknado.domains.common.protocols import VerifySpecResult

        return VerifySpecResult(outcome="done")

    def generate_ralph_md(
        self,
        node: MikadoNode,
        context: str,
        quality_gates: list[str],
        output_path: Path,
    ) -> Path:
        return output_path


class _FakeCrg:
    def ensure_graph(self, project_root: Path) -> None:
        pass

    def get_impact_radius(self, files: list[str]) -> dict[str, Any]:
        return {"files": files}

    def get_architecture_overview(self) -> dict[str, Any]:
        return {"modules": []}

    def list_communities(self, sort_by: str = "size", min_size: int = 0) -> list[dict[str, Any]]:
        return []

    def list_flows(self, sort_by: str = "criticality", limit: int = 50) -> list[dict[str, Any]]:
        return []

    def get_minimal_context(
        self,
        task: str = "",
        changed_files: list[str] | None = None,
    ) -> dict[str, Any]:
        return {}

    def get_bridge_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return []

    def get_hub_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return []

    def semantic_search_nodes(self, query: str, top_n: int = 5) -> list[dict[str, Any]]:
        return []

    def semantic_search(
        self,
        query: str,
        top_n: int = 5,
        detail_level: str = "minimal",
    ) -> list[dict[str, Any]]:
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
        quality_gates=("uv run pytest",),
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
        graph.add_node("compute thing")
        executor.dispatch(1, config)
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
        graph.add_node("write tests")
        executor.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.dispatched_at is not None
        assert node.dispatched_at.tzinfo is not None

    def test_dispatched_at_not_set_before_dispatch(
        self,
        graph: MikadoGraph,
    ) -> None:
        graph.add_node("not yet dispatched")
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
        graph.add_node("feature")
        executor.dispatch(1, config)
        executor.complete(1, "main")
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
        graph.add_node("fast task")
        executor.dispatch(1, config)
        executor.complete(1, "main")
        durations = graph.recent_completion_durations(limit=10)
        assert all(d >= 0.0 for d in durations)

    def test_duration_not_written_on_rebase_failure(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        tmp_path: Path,
    ) -> None:
        fake_git = _FakeGit()
        fake_git.rebase_result = RebaseResult(success=False)
        ex = Executor(graph=graph, git=fake_git, ralph=_FakeRalph(), crg=_FakeCrg())
        graph.add_node("conflicted task")
        ex.dispatch(1, config)

        # Create the worktree dir so _rebase_and_merge actually calls git.rebase
        node = graph.get_node(1)
        assert node is not None and node.worktree_path is not None
        Path(node.worktree_path).mkdir(parents=True, exist_ok=True)

        ex.complete(1, "main")

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
        graph.add_node("manually inserted node")
        graph.mark_running(1)
        graph.mark_done(1)

        durations = graph.recent_completion_durations(limit=10)
        assert durations == []

    def test_direct_complete_without_dispatch_skips_duration(
        self,
        executor: Executor,
        graph: MikadoGraph,
    ) -> None:
        graph.add_node("bypassed dispatch")
        graph.mark_running(1)
        # complete() reads dispatched_at from the node; it's None here
        executor.complete(1, "main")

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
        graph.add_node("one")
        executor.dispatch(1, config)
        executor.complete(1, "main")
        durations = graph.recent_completion_durations(limit=10)
        assert len(durations) == 1

    def test_many_completed_nodes(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        for desc in ("alpha", "beta", "gamma"):
            graph.add_node(desc)

        for node_id in (1, 2, 3):
            executor.dispatch(node_id, config)
            executor.complete(node_id, "main")

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
            graph.add_node(desc)

        for node_id in range(1, 6):
            executor.dispatch(node_id, config)
            executor.complete(node_id, "main")

        durations = graph.recent_completion_durations(limit=3)
        assert len(durations) == 3

    def test_excludes_nodes_without_duration(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        graph.add_node("completed")
        graph.add_node("pending")
        executor.dispatch(1, config)
        executor.complete(1, "main")

        durations = graph.recent_completion_durations(limit=10)
        assert len(durations) == 1
