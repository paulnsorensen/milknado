"""Tests for reconcile_stale_nodes — crash / interrupt recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from milknado.domains.common.types import NodeStatus
from milknado.domains.execution.reconcile import ReconcileResult, reconcile_stale_nodes
from milknado.domains.graph import MikadoGraph

# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


class FakeGit:
    def __init__(
        self,
        *,
        branches: set[str] | None = None,
        ancestors: set[tuple[str, str]] | None = None,
    ) -> None:
        self._branches: set[str] = branches or set()
        # ancestors = set of (ref, of_ref) pairs for which is_ancestor returns True
        self._ancestors: set[tuple[str, str]] = ancestors or set()
        self.removed: list[Path] = []

    def branch_exists(self, branch: str) -> bool:
        return branch in self._branches

    def is_ancestor(self, ref: str, of_ref: str) -> bool:
        return (ref, of_ref) in self._ancestors

    def remove_worktree(self, path: Path) -> None:
        self.removed.append(path)

    # satisfy remaining GitPort surface
    def create_worktree(self, path: Path, branch: str) -> Path:
        return path

    def rebase(self, worktree: Path, onto: str) -> Any:
        return None

    def current_branch(self) -> str:
        return "main"

    def commit_all(self, worktree: Path, message: str) -> None:
        pass

    def squash_and_commit(self, worktree: Path, onto: str, msg: str) -> None:
        pass

    def push_branch(self, branch: str, remote: str = "origin") -> None:
        pass


class FakeRalph:
    def __init__(self, *, run_ids_alive: set[str] | None = None) -> None:
        self._alive: set[str] = run_ids_alive or set()
        self.stopped: list[str] = []

    def get_run(self, run_id: str) -> Any | None:
        return object() if run_id in self._alive else None

    def stop_run(self, run_id: str) -> None:
        self.stopped.append(run_id)

    # unused stubs
    def create_run(self, *a: Any, **kw: Any) -> Any:
        return None

    def start_run(self, run_id: str) -> None:
        pass

    def list_runs(self) -> list[Any]:
        return []

    def get_run_stdout(self, run_id: str) -> list[str]:
        return []

    def wait_for_next_completion(self, *a: Any, **kw: Any) -> Any:
        raise NotImplementedError

    def poll_progress_events(self) -> list[Any]:
        return []

    def verify_spec(self, *a: Any, **kw: Any) -> Any:
        return None

    def generate_ralph_md(self, *a: Any, **kw: Any) -> Path:
        return Path("RALPH.md")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReconcileStaleNodes:
    def test_no_running_nodes_returns_empty_result(self, graph: MikadoGraph) -> None:
        graph.add_node("task")
        result = reconcile_stale_nodes(graph, FakeGit(), FakeRalph(), "main")
        assert result == ReconcileResult()

    def test_running_node_without_branch_resets_to_pending(self, graph: MikadoGraph) -> None:
        graph.add_node("task")
        graph.mark_running(1)
        result = reconcile_stale_nodes(graph, FakeGit(), FakeRalph(), "main")

        assert 1 in result.reset_pending
        assert 1 not in result.resolved_done
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.PENDING

    def test_running_node_with_nonexistent_branch_resets_to_pending(
        self, graph: MikadoGraph
    ) -> None:
        graph.add_node("task")
        graph.mark_running(1, branch_name="milknado/1-task")
        result = reconcile_stale_nodes(graph, FakeGit(), FakeRalph(), "main")

        assert 1 in result.reset_pending
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.PENDING

    def test_running_node_branch_exists_but_not_ancestor_resets_to_pending(
        self, graph: MikadoGraph
    ) -> None:
        graph.add_node("task")
        graph.mark_running(1, branch_name="milknado/1-task")
        git = FakeGit(branches={"milknado/1-task"})  # exists, not merged
        result = reconcile_stale_nodes(graph, git, FakeRalph(), "main")

        assert 1 in result.reset_pending
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.PENDING

    def test_running_node_branch_merged_marked_done(self, graph: MikadoGraph) -> None:
        graph.add_node("task")
        graph.mark_running(1, branch_name="milknado/1-task")
        # complete() squash-rebases the node branch ONTO feature_branch without
        # advancing feature_branch's tip, so the merged signature is
        # feature_branch (main) being an ancestor of the node branch.
        git = FakeGit(
            branches={"milknado/1-task"},
            ancestors={("main", "milknado/1-task")},
        )
        result = reconcile_stale_nodes(graph, git, FakeRalph(), "main")

        assert 1 in result.resolved_done
        assert 1 not in result.reset_pending
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.DONE

    def test_node_branch_ancestor_of_feature_branch_is_not_merged(
        self, graph: MikadoGraph
    ) -> None:
        # Regression: the reversed direction — node branch being an ancestor of
        # feature_branch — is NOT the completion signature (that would mean the
        # node added nothing on top of main). Such a node must reset to PENDING,
        # not be falsely marked DONE.
        graph.add_node("task")
        graph.mark_running(1, branch_name="milknado/1-task")
        git = FakeGit(
            branches={"milknado/1-task"},
            ancestors={("milknado/1-task", "main")},
        )
        result = reconcile_stale_nodes(graph, git, FakeRalph(), "main")

        assert 1 in result.reset_pending
        assert 1 not in result.resolved_done
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.PENDING

    def test_worktree_still_exists_prevents_merged_detection(
        self, graph: MikadoGraph, tmp_path: Path
    ) -> None:
        wt = tmp_path / "milknado-1-task"
        wt.mkdir()
        graph.add_node("task")
        graph.mark_running(1, branch_name="milknado/1-task", worktree_path=str(wt))
        # Branch would be "merged" if we didn't guard on worktree presence.
        git = FakeGit(
            branches={"milknado/1-task"},
            ancestors={("milknado/1-task", "main")},
        )
        result = reconcile_stale_nodes(graph, git, FakeRalph(), "main")

        # Worktree still exists → not safely merged → reset
        assert 1 in result.reset_pending
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.PENDING

    def test_stale_worktree_removed_on_reset(self, graph: MikadoGraph, tmp_path: Path) -> None:
        wt = tmp_path / "milknado-1-task"
        wt.mkdir()
        graph.add_node("task")
        graph.mark_running(1, worktree_path=str(wt))
        git = FakeGit()

        reconcile_stale_nodes(graph, git, FakeRalph(), "main")

        assert wt in git.removed

    def test_stop_run_called_for_active_ralph_run(self, graph: MikadoGraph) -> None:
        graph.add_node("task")
        graph.mark_running(1)
        graph.set_run_id(1, "run-xyz")
        ralph = FakeRalph(run_ids_alive={"run-xyz"})

        reconcile_stale_nodes(graph, FakeGit(), ralph, "main")

        assert "run-xyz" in ralph.stopped

    def test_stop_run_skipped_when_run_not_found(self, graph: MikadoGraph) -> None:
        graph.add_node("task")
        graph.mark_running(1)
        graph.set_run_id(1, "run-gone")
        ralph = FakeRalph(run_ids_alive=set())  # run not alive

        reconcile_stale_nodes(graph, FakeGit(), ralph, "main")

        assert "run-gone" not in ralph.stopped

    def test_stop_run_failure_does_not_abort_reconcile(self, graph: MikadoGraph) -> None:
        graph.add_node("task")
        graph.mark_running(1)
        graph.set_run_id(1, "run-boom")

        class BoomRalph(FakeRalph):
            def get_run(self, run_id: str) -> Any:
                return object()

            def stop_run(self, run_id: str) -> None:
                raise RuntimeError("connection refused")

        reconcile_stale_nodes(graph, FakeGit(), BoomRalph(), "main")  # must not raise

        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.PENDING

    def test_git_error_falls_back_to_reset(self, graph: MikadoGraph) -> None:
        graph.add_node("task")
        graph.mark_running(1, branch_name="milknado/1-task")

        class BoomGit(FakeGit):
            def branch_exists(self, branch: str) -> bool:
                raise OSError("git exploded")

        result = reconcile_stale_nodes(graph, BoomGit(), FakeRalph(), "main")

        assert 1 in result.reset_pending
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.PENDING

    def test_multiple_running_nodes_all_reconciled(self, graph: MikadoGraph) -> None:
        graph.add_node("root")
        graph.add_node("child-a", parent_id=1)
        graph.add_node("child-b", parent_id=1)
        graph.mark_running(2)
        graph.mark_running(3, branch_name="milknado/3-child-b")
        git = FakeGit(
            branches={"milknado/3-child-b"},
            ancestors={("main", "milknado/3-child-b")},
        )
        result = reconcile_stale_nodes(graph, git, FakeRalph(), "main")

        assert 2 in result.reset_pending
        assert 3 in result.resolved_done
        assert graph.get_node(2).status == NodeStatus.PENDING  # type: ignore[union-attr]
        assert graph.get_node(3).status == NodeStatus.DONE  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Coverage gaps: _clean_worktree internal paths (lines 101, 104-105)
# ---------------------------------------------------------------------------


class TestCleanWorktreeInternals:
    def test_clean_worktree_skips_when_path_does_not_exist(
        self, graph: MikadoGraph, tmp_path: Path
    ) -> None:
        """_clean_worktree returns early when the worktree path is absent — no git call."""
        nonexistent = tmp_path / "gone"
        # Make sure it really doesn't exist
        assert not nonexistent.exists()
        graph.add_node("task")
        graph.mark_running(1, worktree_path=str(nonexistent))
        git = FakeGit()

        reconcile_stale_nodes(graph, git, FakeRalph(), "main")

        # remove_worktree must NOT have been called — path was already gone
        assert git.removed == []

    def test_clean_worktree_swallows_remove_worktree_exception(
        self, graph: MikadoGraph, tmp_path: Path
    ) -> None:
        """_clean_worktree logs and does not re-raise when remove_worktree fails."""
        wt = tmp_path / "milknado-1-task"
        wt.mkdir()
        graph.add_node("task")
        graph.mark_running(1, worktree_path=str(wt))

        class BoomGit(FakeGit):
            def remove_worktree(self, path: Path) -> None:
                raise OSError("filesystem error")

        # Must not raise — reconcile continues gracefully
        result = reconcile_stale_nodes(graph, BoomGit(), FakeRalph(), "main")
        assert 1 in result.reset_pending
