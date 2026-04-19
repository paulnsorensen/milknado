from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from milknado.domains.common.types import (
    MikadoNode,
    NodeStatus,
    RebaseResult,
)
from milknado.domains.execution import (
    CompletionResult,
    DispatchResult,
    ExecutionConfig,
    Executor,
    get_dispatchable_nodes,
)
from milknado.domains.graph import MikadoGraph


@dataclass
class FakeRunState:
    run_id: str = "run-1"


@dataclass
class FakeRun:
    state: FakeRunState = field(default_factory=FakeRunState)


class FakeGit:
    def __init__(self) -> None:
        self.created: list[tuple[Path, str]] = []
        self.removed: list[Path] = []
        self.commits: list[tuple[Path, str]] = []
        self.rebase_result: RebaseResult = RebaseResult(success=True)

    def create_worktree(self, path: Path, branch: str) -> Path:
        self.created.append((path, branch))
        return path

    def remove_worktree(self, path: Path) -> None:
        self.removed.append(path)

    def rebase(self, worktree: Path, onto: str) -> RebaseResult:
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
        execution_agent="claude",
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

    def test_stores_run_id_in_graph(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        ex = Executor(
            graph=graph, git=FakeGit(), ralph=FakeRalph(), crg=FakeCrg(),
        )
        graph.add_node("track run")
        ex.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.run_id == "run-1"

    def test_cleans_up_worktree_on_dispatch_failure(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_git = FakeGit()
        fake_ralph = FakeRalph()
        fake_ralph.create_run = lambda **_kw: (_ for _ in ()).throw(  # type: ignore[assignment]
            RuntimeError("ralph exploded"),
        )
        ex = Executor(
            graph=graph, git=fake_git, ralph=fake_ralph, crg=FakeCrg(),
        )
        graph.add_node("doomed task")
        with pytest.raises(RuntimeError, match="ralph exploded"):
            ex.dispatch(1, config)
        assert len(fake_git.removed) == 1

    def test_resets_node_to_pending_on_dispatch_failure(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_ralph = FakeRalph()
        fake_ralph.create_run = lambda **_kw: (_ for _ in ()).throw(  # type: ignore[assignment]
            RuntimeError("ralph exploded"),
        )
        ex = Executor(
            graph=graph, git=FakeGit(), ralph=fake_ralph, crg=FakeCrg(),
        )
        graph.add_node("doomed task")
        with pytest.raises(RuntimeError, match="ralph exploded"):
            ex.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.PENDING
        assert node.worktree_path is None
        assert node.branch_name is None


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
        fake_git.rebase_result = RebaseResult(success=False)
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

    def test_removes_worktree_on_rebase_failure(
        self, graph: MikadoGraph, tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        fake_git.rebase_result = RebaseResult(success=False)
        ex = Executor(
            graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        ex.complete(1, "main")
        assert wt in fake_git.removed

    def test_removes_worktree_on_commit_failure(
        self, graph: MikadoGraph, tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        fake_git.commit_all = lambda *_args: (_ for _ in ()).throw(  # type: ignore[assignment]
            RuntimeError("nothing to commit"),
        )
        ex = Executor(
            graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        result = ex.complete(1, "main")
        assert wt in fake_git.removed
        assert result.rebased is False

    def test_clears_metadata_on_rebase_failure(
        self, graph: MikadoGraph, tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        fake_git.rebase_result = RebaseResult(success=False)
        ex = Executor(
            graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt), branch_name="milknado/1-task")
        ex.complete(1, "main")
        node = graph.get_node(1)
        assert node is not None
        assert node.worktree_path is None
        assert node.branch_name is None

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

    def test_commit_message_includes_node_id_and_description(
        self, graph: MikadoGraph, tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        ex = Executor(
            graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg(),
        )
        graph.add_node("Add user authentication")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        ex.complete(1, "main")

        assert len(fake_git.commits) == 1
        _, msg = fake_git.commits[0]
        assert msg == "feat(milknado-1): Add user authentication"

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

    def test_clears_metadata_on_fail(
        self, executor: Executor, graph: MikadoGraph,
    ) -> None:
        graph.add_node("task")
        graph.mark_running(1, worktree_path="/tmp/wt", branch_name="milknado/1-task")
        executor.fail(1)
        node = graph.get_node(1)
        assert node is not None
        assert node.worktree_path is None
        assert node.branch_name is None

    def test_removes_worktree_on_fail(
        self, graph: MikadoGraph, tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        ex = Executor(
            graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt), branch_name="milknado/1-task")
        ex.fail(1)
        assert wt in fake_git.removed


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
    def test_root_only_no_why_chain(self, graph: MikadoGraph) -> None:
        """Single root node: no Why chain; Goal and Your task both present."""
        from milknado.domains.execution.executor import _build_node_context

        graph.add_node("Root goal description")
        node = graph.get_node(1)
        assert node is not None
        result = _build_node_context(node, graph, FakeCrg())
        assert "## Goal" in result
        assert "Root goal description" in result
        assert "## Your task" in result
        assert "## Why chain" not in result

    def test_leaf_depth_3_has_why_chain(self, graph: MikadoGraph) -> None:
        """Leaf at depth 3: Goal=root, Why chain=batch1+batch2, Your task=leaf."""
        from milknado.domains.execution.executor import _build_node_context

        root = graph.add_node("Root goal: refactor auth")
        batch1 = graph.add_node("Batch1: extract interfaces", parent_id=root.id)
        batch2 = graph.add_node("Batch2: update callers", parent_id=batch1.id)
        leaf = graph.add_node("Leaf: fix import in handler.py", parent_id=batch2.id)

        result = _build_node_context(leaf, graph, FakeCrg())
        assert "## Goal" in result
        assert "Root goal: refactor auth" in result
        assert "## Why chain" in result
        assert "Batch2: update callers" in result
        assert "Batch1: extract interfaces" in result
        assert "## Your task" in result
        assert "Leaf: fix import in handler.py" in result
        assert "## Files" in result
        assert "## Impact Radius" in result

    def test_crg_none_shows_degradation_marker(self, graph: MikadoGraph) -> None:
        """crg=None produces fallback Impact Radius line."""
        from milknado.domains.execution.executor import _build_node_context

        graph.add_node("some task")
        node = graph.get_node(1)
        assert node is not None
        result = _build_node_context(node, graph, None)
        assert "## Impact Radius" in result
        assert "CRG unavailable" in result

    def test_with_files_includes_file_list(self, graph: MikadoGraph) -> None:
        """Files assigned to node appear in ## Files section."""
        from milknado.domains.execution.executor import _build_node_context

        graph.add_node("refactor")
        graph.set_file_ownership(1, ["auth.py", "models.py"])
        node = graph.get_node(1)
        assert node is not None
        result = _build_node_context(node, graph, FakeCrg())
        assert "## Files" in result
        assert "`auth.py`" in result
        assert "`models.py`" in result
        assert "## Impact Radius" in result

    def test_why_chain_order_parent_first(self, graph: MikadoGraph) -> None:
        """Why chain lists parent before grandparent."""
        from milknado.domains.execution.executor import _build_node_context

        root = graph.add_node("Root")
        parent = graph.add_node("Parent node", parent_id=root.id)
        leaf = graph.add_node("Leaf node", parent_id=parent.id)

        result = _build_node_context(leaf, graph, FakeCrg())
        # Why chain shows parent (not root which is in Goal); root excluded from Why chain
        assert "## Why chain" in result
        assert "Parent node" in result
        # Root is only in Goal section, not duplicated in Why chain
        assert result.count("Root") == 1  # only in ## Goal
