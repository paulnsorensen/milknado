from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from milknado.domains.common.types import MikadoNode, NodeStatus
from milknado.domains.execution import (
    CompletionResult,
    DispatchResult,
    ExecutionConfig,
    Executor,
    get_dispatchable_nodes,
)
from milknado.domains.graph import MikadoGraph


@dataclass
class FakeRun:
    id: str = "run-1"


class FakeGit:
    def __init__(self) -> None:
        self.created: list[tuple[Path, str]] = []
        self.removed: list[Path] = []
        self.commits: list[tuple[Path, str]] = []
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
        self.commits.append((worktree, message))


class FakeRalph:
    def __init__(self) -> None:
        self.runs_created: list[dict[str, Any]] = []
        self.runs_started: list[str] = []
        self.generated: list[Path] = []

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: list[str],
    ) -> FakeRun:
        self.runs_created.append({"agent": agent, "dir": ralph_dir})
        return FakeRun()

    def start_run(self, run_id: str) -> None:
        self.runs_started.append(run_id)

    def stop_run(self, run_id: str) -> None:
        pass

    def list_runs(self) -> list[Any]:
        return []

    def get_run(self, run_id: str) -> Any | None:
        return None

    def generate_ralph_md(
        self,
        node: MikadoNode,
        context: str,
        quality_gates: list[str],
        output_path: Path,
    ) -> Path:
        self.generated.append(output_path)
        return output_path


class FakeCrg:
    def ensure_graph(self, project_root: Path) -> None:
        pass

    def get_impact_radius(self, files: list[str]) -> dict[str, Any]:
        return {"files": files}

    def get_architecture_overview(self) -> dict[str, Any]:
        return {"modules": []}


@pytest.fixture()
def config(tmp_path: Path) -> ExecutionConfig:
    return ExecutionConfig(
        agent_command="claude",
        quality_gates=("uv run pytest",),
        worktree_pattern="milknado-{node_id}-{slug}",
        project_root=tmp_path,
    )


@pytest.fixture()
def executor(graph: MikadoGraph) -> Executor:
    return Executor(
        graph=graph, git=FakeGit(), ralph=FakeRalph(), crg=FakeCrg(),
    )


class TestGetDispatchableNodes:
    def test_empty_graph(self, graph: MikadoGraph) -> None:
        assert get_dispatchable_nodes(graph) == []

    def test_single_pending_leaf(self, graph: MikadoGraph) -> None:
        graph.add_node("root")
        assert get_dispatchable_nodes(graph) == [1]

    def test_excludes_running_nodes(self, graph: MikadoGraph) -> None:
        graph.add_node("root")
        graph.mark_running(1)
        assert get_dispatchable_nodes(graph) == []

    def test_excludes_done_nodes(self, graph: MikadoGraph) -> None:
        graph.add_node("root")
        graph.mark_running(1)
        graph.mark_done(1)
        assert get_dispatchable_nodes(graph) == []

    def test_parallel_leaves_no_conflict(
        self, graph: MikadoGraph
    ) -> None:
        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.add_node("leaf-b", parent_id=root.id)
        dispatchable = get_dispatchable_nodes(graph)
        assert 2 in dispatchable
        assert 3 in dispatchable

    def test_filters_conflicting_leaves(
        self, graph: MikadoGraph
    ) -> None:
        root = graph.add_node("root")
        a = graph.add_node("leaf-a", parent_id=root.id)
        b = graph.add_node("leaf-b", parent_id=root.id)
        graph.set_file_ownership(a.id, ["shared.py"])
        graph.set_file_ownership(b.id, ["shared.py"])
        dispatchable = get_dispatchable_nodes(graph)
        assert len(dispatchable) == 1
        assert dispatchable[0] == a.id

    def test_child_must_complete_before_parent(
        self, graph: MikadoGraph
    ) -> None:
        root = graph.add_node("root")
        child = graph.add_node("child", parent_id=root.id)
        dispatchable = get_dispatchable_nodes(graph)
        assert child.id in dispatchable
        assert root.id not in dispatchable


class TestExecutorDispatch:
    def test_creates_worktree_and_starts_run(
        self, executor: Executor, graph: MikadoGraph, config: ExecutionConfig,
    ) -> None:
        graph.add_node("extract interface")
        result = executor.dispatch(1, config)

        assert isinstance(result, DispatchResult)
        assert result.node_id == 1
        assert result.run_id == "run-1"
        assert "extract-interface" in str(result.worktree)

    def test_marks_node_running(
        self, executor: Executor, graph: MikadoGraph, config: ExecutionConfig,
    ) -> None:
        graph.add_node("do work")
        executor.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.RUNNING

    def test_stores_worktree_path(
        self, executor: Executor, graph: MikadoGraph, config: ExecutionConfig,
    ) -> None:
        graph.add_node("task")
        executor.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.worktree_path is not None
        assert "milknado-1-" in node.worktree_path

    def test_stores_branch_name(
        self, executor: Executor, graph: MikadoGraph, config: ExecutionConfig,
    ) -> None:
        graph.add_node("task")
        executor.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.branch_name is not None
        assert node.branch_name.startswith("milknado/1-")

    def test_generates_ralph_md(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_ralph = FakeRalph()
        ex = Executor(
            graph=graph, git=FakeGit(), ralph=fake_ralph, crg=FakeCrg(),
        )
        graph.add_node("build feature")
        ex.dispatch(1, config)
        assert len(fake_ralph.generated) == 1
        assert fake_ralph.generated[0].name == "RALPH.md"

    def test_dispatch_nonexistent_node_raises(
        self, executor: Executor, config: ExecutionConfig,
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            executor.dispatch(999, config)

    def test_creates_git_worktree(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_git = FakeGit()
        ex = Executor(
            graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg(),
        )
        graph.add_node("refactor auth")
        ex.dispatch(1, config)
        assert len(fake_git.created) == 1
        path, branch = fake_git.created[0]
        assert "milknado-1-" in str(path)
        assert branch.startswith("milknado/1-")

    def test_starts_ralph_run(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_ralph = FakeRalph()
        ex = Executor(
            graph=graph, git=FakeGit(), ralph=fake_ralph, crg=FakeCrg(),
        )
        graph.add_node("do it")
        ex.dispatch(1, config)
        assert fake_ralph.runs_started == ["run-1"]


class TestExecutorComplete:
    def test_marks_done_on_success(
        self, graph: MikadoGraph,
    ) -> None:
        fake_git = FakeGit()
        ex = Executor(
            graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg(),
        )
        graph.add_node("task")
        graph.mark_running(1)
        result = ex.complete(1, "main")

        assert isinstance(result, CompletionResult)
        assert result.rebased is True
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.DONE

    def test_marks_failed_on_rebase_conflict(
        self, graph: MikadoGraph, tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        fake_git.rebase_result = False
        ex = Executor(
            graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        result = ex.complete(1, "main")

        assert result.rebased is False
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.FAILED

    def test_removes_worktree_on_success(
        self, graph: MikadoGraph, tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        ex = Executor(
            graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        ex.complete(1, "main")
        assert wt in fake_git.removed

    def test_keeps_worktree_on_failure(
        self, graph: MikadoGraph, tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        fake_git.rebase_result = False
        ex = Executor(
            graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        ex.complete(1, "main")
        assert len(fake_git.removed) == 0

    def test_returns_newly_ready_nodes(
        self, graph: MikadoGraph,
    ) -> None:
        ex = Executor(
            graph=graph, git=FakeGit(), ralph=FakeRalph(), crg=FakeCrg(),
        )
        root = graph.add_node("root")
        child = graph.add_node("child", parent_id=root.id)
        graph.mark_running(child.id)
        result = ex.complete(child.id, "main")
        assert root.id in result.newly_ready

    def test_complete_nonexistent_raises(
        self, executor: Executor,
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            executor.complete(999, "main")


class TestExecutorFail:
    def test_marks_node_failed(
        self, executor: Executor, graph: MikadoGraph,
    ) -> None:
        graph.add_node("task")
        graph.mark_running(1)
        executor.fail(1)
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.FAILED


class TestSlugify:
    def test_basic_slugify(self) -> None:
        from milknado.domains.execution.executor import _slugify

        assert _slugify("Extract Payment Service") == "extract-payment-service"

    def test_truncates_long_slugs(self) -> None:
        from milknado.domains.execution.executor import _slugify

        result = _slugify("a" * 50)
        assert len(result) <= 30

    def test_strips_special_chars(self) -> None:
        from milknado.domains.execution.executor import _slugify

        assert _slugify("fix: auth (v2)") == "fix-auth-v2"


class TestBuildNodeContext:
    def test_without_files(self) -> None:
        from milknado.domains.execution.executor import _build_node_context

        result = _build_node_context("do stuff", [], FakeCrg())
        assert "# Task" in result
        assert "do stuff" in result
        assert "Impact" not in result

    def test_with_files(self) -> None:
        from milknado.domains.execution.executor import _build_node_context

        result = _build_node_context("refactor", ["auth.py"], FakeCrg())
        assert "# Task" in result
        assert "# Impact Radius" in result
        assert "# Owned Files" in result
        assert "`auth.py`" in result
