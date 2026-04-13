from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from milknado.domains.common.types import MikadoNode, NodeStatus
from milknado.domains.execution import (
    ExecutionConfig,
    Executor,
    RunLoop,
)
from milknado.domains.graph import MikadoGraph


@dataclass
class FakeRun:
    id: str = "run-1"


class FakeGit:
    def __init__(self) -> None:
        self.created: list[tuple[Path, str]] = []
        self.removed: list[Path] = []
        self.rebase_result: bool = True

    def create_worktree(self, path: Path, branch: str) -> Path:
        self.created.append((path, branch))
        return path

    def remove_worktree(self, path: Path) -> None:
        self.removed.append(path)

    def rebase(self, worktree: Path, onto: str) -> bool:
        return self.rebase_result

    def current_branch(self) -> str:
        return "main"

    def commit_all(self, worktree: Path, message: str) -> None:
        pass


class FakeCrg:
    def ensure_graph(self, project_root: Path) -> None:
        pass

    def get_impact_radius(self, files: list[str]) -> dict[str, Any]:
        return {"files": files}

    def get_architecture_overview(self) -> dict[str, Any]:
        return {"modules": []}


class FakeRalph:
    def __init__(self) -> None:
        self._run_counter = 0
        self._complete_after: dict[str, int] = {}
        self._poll_counts: dict[str, int] = {}
        self._success: dict[str, bool] = {}

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: list[str],
    ) -> FakeRun:
        self._run_counter += 1
        run_id = f"run-{self._run_counter}"
        self._complete_after.setdefault(run_id, 1)
        self._poll_counts[run_id] = 0
        self._success.setdefault(run_id, True)
        return FakeRun(id=run_id)

    def start_run(self, run_id: str) -> None:
        pass

    def stop_run(self, run_id: str) -> None:
        pass

    def list_runs(self) -> list[Any]:
        return []

    def get_run(self, run_id: str) -> Any | None:
        return None

    def is_run_complete(self, run_id: str) -> bool:
        self._poll_counts[run_id] = self._poll_counts.get(run_id, 0) + 1
        return self._poll_counts[run_id] >= self._complete_after[run_id]

    def is_run_success(self, run_id: str) -> bool:
        return self._success.get(run_id, True)

    def generate_ralph_md(
        self,
        node: MikadoNode,
        context: str,
        quality_gates: list[str],
        output_path: Path,
    ) -> Path:
        return output_path

    def set_completes_after(self, run_id: str, polls: int) -> None:
        self._complete_after[run_id] = polls

    def set_run_fails(self, run_id: str) -> None:
        self._success[run_id] = False


@pytest.fixture()
def config(tmp_path: Path) -> ExecutionConfig:
    return ExecutionConfig(
        agent_command="claude",
        quality_gates=("uv run pytest",),
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


@pytest.fixture()
def run_loop(
    executor: Executor,
    graph: MikadoGraph,
    fake_ralph: FakeRalph,
) -> RunLoop:
    return RunLoop(executor=executor, graph=graph, ralph=fake_ralph)


class TestRunLoopSingleNode:
    def test_dispatches_and_completes_single_node(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        graph.add_node("root goal")
        result = run_loop.run(config, "main", poll_interval=0.0)

        assert result.dispatched_total == 1
        assert result.completed_total == 1
        assert result.failed_total == 0
        assert result.root_done is True

    def test_root_marked_done(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        graph.add_node("root goal")
        run_loop.run(config, "main", poll_interval=0.0)

        root = graph.get_node(1)
        assert root is not None
        assert root.status == NodeStatus.DONE


class TestRunLoopParentChild:
    def test_dispatches_leaf_then_parent(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        graph.add_node("leaf", parent_id=root.id)

        result = run_loop.run(config, "main", poll_interval=0.0)

        assert result.dispatched_total == 2
        assert result.completed_total == 2
        assert result.root_done is True

    def test_all_nodes_done(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        graph.add_node("leaf", parent_id=root.id)

        run_loop.run(config, "main", poll_interval=0.0)

        for node in graph.get_all_nodes():
            assert node.status == NodeStatus.DONE


class TestRunLoopParallelLeaves:
    def test_dispatches_parallel_leaves(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.add_node("leaf-b", parent_id=root.id)

        result = run_loop.run(config, "main", poll_interval=0.0)

        assert result.dispatched_total == 3
        assert result.completed_total == 3
        assert result.root_done is True


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

        result = run_loop.run(
            config, "main", concurrency_limit=2, poll_interval=0.0,
        )

        assert result.dispatched_total == 4
        assert result.completed_total == 4
        assert result.root_done is True


class TestRunLoopFailure:
    def test_failed_run_marks_node_failed(
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

        graph.add_node("will fail")
        result = loop.run(config, "main", poll_interval=0.0)

        assert result.failed_total == 1
        assert result.root_done is False
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.FAILED

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
        result = loop.run(config, "main", poll_interval=0.0)

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
        result = run_loop.run(config, "main", poll_interval=0.0)

        assert result.dispatched_total == 0
        assert result.completed_total == 0
        assert result.root_done is False


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
            graph=graph, git=fake_git, ralph=fake_ralph, crg=fake_crg,
        )
        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph)

        root = graph.add_node("root")
        a = graph.add_node("a", parent_id=root.id)
        b = graph.add_node("b", parent_id=root.id)
        graph.set_file_ownership(a.id, ["shared.py"])
        graph.set_file_ownership(b.id, ["shared.py"])

        result = loop.run(config, "main", poll_interval=0.0)

        assert result.dispatched_total == 3
        assert result.completed_total == 3
        assert result.root_done is True
