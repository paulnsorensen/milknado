"""Tests for worktree-shared-graph spec (two parts).

Part 1 — worktree db anchoring:
  resolve_project_root cwd fallback detects a linked worktree via
  `git rev-parse --git-common-dir` and hops to the main checkout's root.
  Explicit project_root arg and MILKNADO_PROJECT_ROOT env are honored as given.

Part 2 — claim_goal subtree fencing:
  claim_goal(goal_id, run_id) fences a goal subtree so dispatch tools refuse
  to start a task whose ancestor goal is owned by a DIFFERENT live run.
  Dead-claimant reclaim and release-on-completion semantics mirror claim_node.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from milknado._mcp_core import resolve_project_root
from milknado.domains.common import NodeKind
from milknado.domains.dispatch._runstate import now_iso
from milknado.domains.graph import MikadoGraph

_DEAD_PID = 2**31 - 1  # no process can hold this; os.kill(_, 0) -> ProcessLookupError


def _git(cwd: str | Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def _git_worktree_setup(base: Path) -> tuple[Path, Path]:
    """Create a git repo at base/main with one commit and a linked worktree at base/wt."""
    main = base / "main"
    wt = base / "wt"
    main.mkdir()
    _git(base, "init", str(main))
    _git(main, "config", "user.email", "test@test.com")
    _git(main, "config", "user.name", "Test")
    (main / "README").write_text("hi")
    _git(main, "add", ".")
    _git(main, "commit", "-m", "init")
    _git(main, "worktree", "add", str(wt))
    return main, wt


# ---------------------------------------------------------------------------
# Part 1 — resolve_project_root worktree anchoring
# ---------------------------------------------------------------------------


class TestResolveProjectRootWorktreeAnchoring:
    """resolve_project_root hops from a linked worktree to the main checkout."""

    def test_explicit_project_root_is_honored_unchanged(self, tmp_path: Path) -> None:
        """An explicit project_root bypasses the git check entirely."""
        explicit = tmp_path / "explicit_root"
        explicit.mkdir()
        result = resolve_project_root(str(explicit))
        assert result == explicit.resolve()

    def test_env_project_root_is_honored_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """MILKNADO_PROJECT_ROOT env bypasses the git check."""
        env_root = tmp_path / "env_root"
        env_root.mkdir()
        monkeypatch.setenv("MILKNADO_PROJECT_ROOT", str(env_root))
        result = resolve_project_root(None)
        assert result == env_root.resolve()

    def test_non_git_dir_falls_back_to_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A directory that is not a git repo returns cwd unchanged."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("MILKNADO_PROJECT_ROOT", raising=False)
        result = resolve_project_root(None)
        assert result == tmp_path.resolve()

    def test_main_checkout_cwd_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """cwd is already the main checkout (git-common-dir == .git) — no hop."""
        _git(tmp_path.parent, "init", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("MILKNADO_PROJECT_ROOT", raising=False)
        result = resolve_project_root(None)
        assert result == tmp_path.resolve()

    def test_linked_worktree_cwd_hops_to_main_checkout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """cwd is a linked worktree — resolve_project_root returns the main checkout root."""
        main, wt = _git_worktree_setup(tmp_path)
        monkeypatch.chdir(wt)
        monkeypatch.delenv("MILKNADO_PROJECT_ROOT", raising=False)
        result = resolve_project_root(None)
        assert result == main.resolve(), f"expected main checkout {main.resolve()}, got {result}"

    def test_explicit_root_in_worktree_is_not_reanchored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Explicit project_root pointing at a linked worktree is returned as-is."""
        _main, wt = _git_worktree_setup(tmp_path)
        monkeypatch.delenv("MILKNADO_PROJECT_ROOT", raising=False)
        # Explicit arg pointing at the worktree — returned unchanged
        result = resolve_project_root(str(wt))
        assert result == wt.resolve()


# ---------------------------------------------------------------------------
# Part 2 — claim_goal subtree fencing
# ---------------------------------------------------------------------------


def _make_goal_with_task(graph: MikadoGraph) -> tuple[int, int]:
    """Add a goal with one task child. Returns (goal_id, task_id)."""
    goal = graph.add_node("goal", kind=NodeKind.GOAL)
    task = graph.add_node("task", parent_id=goal.id, kind=NodeKind.TASK)
    return goal.id, task.id


class TestClaimGoal:
    """claim_goal fences a goal's subtree."""

    def test_claim_goal_succeeds_on_unclaimed_goal(self, graph: MikadoGraph) -> None:
        goal_id, _ = _make_goal_with_task(graph)
        assert graph.claim_goal(goal_id, "run-A", now=now_iso()) is True

    def test_claim_goal_refuses_second_claimant(self, graph: MikadoGraph) -> None:
        goal_id, _ = _make_goal_with_task(graph)
        assert graph.claim_goal(goal_id, "run-A", now=now_iso()) is True
        assert graph.claim_goal(goal_id, "run-B", now=now_iso()) is False

    def test_claim_goal_records_run_id(self, graph: MikadoGraph) -> None:
        goal_id, _ = _make_goal_with_task(graph)
        graph.claim_goal(goal_id, "run-A", now=now_iso())
        goal = graph.get_node(goal_id)
        assert goal is not None
        assert goal.goal_run_id == "run-A"

    def test_claim_goal_only_valid_for_goal_kind(self, graph: MikadoGraph) -> None:
        task = graph.add_node("t")
        with pytest.raises(ValueError, match="goal"):
            graph.claim_goal(task.id, "run-A", now=now_iso())

    def test_claim_goal_raises_if_node_not_found(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.claim_goal(9999, "run-A", now=now_iso())

    def test_dead_claimant_is_reclaimable(self, graph: MikadoGraph) -> None:
        """When the owning run's pid is dead, the claim can be reclaimed."""
        goal_id, _ = _make_goal_with_task(graph)
        graph.claim_goal(goal_id, "run-A", now=now_iso())
        graph.set_goal_pid(goal_id, "run-A", _DEAD_PID)
        assert graph.try_reclaim_goal(goal_id, now=now_iso()) is True
        # After reclaim, a new claimant wins
        assert graph.claim_goal(goal_id, "run-B", now=now_iso()) is True

    def test_live_claimant_is_not_reclaimable(self, graph: MikadoGraph) -> None:
        goal_id, _ = _make_goal_with_task(graph)
        graph.claim_goal(goal_id, "run-A", now=now_iso())
        graph.set_goal_pid(goal_id, "run-A", os.getpid())  # this process: alive
        assert graph.try_reclaim_goal(goal_id, now=now_iso()) is False

    def test_release_goal_frees_claim(self, graph: MikadoGraph) -> None:
        goal_id, _ = _make_goal_with_task(graph)
        graph.claim_goal(goal_id, "run-A", now=now_iso())
        assert graph.release_goal(goal_id, "run-A") is True
        assert graph.claim_goal(goal_id, "run-B", now=now_iso()) is True

    def test_release_goal_is_noop_for_stale_run_id(self, graph: MikadoGraph) -> None:
        goal_id, _ = _make_goal_with_task(graph)
        graph.claim_goal(goal_id, "run-A", now=now_iso())
        assert graph.release_goal(goal_id, "run-stale") is False

    def test_release_goal_on_unclaimed_goal_is_noop(self, graph: MikadoGraph) -> None:
        goal_id, _ = _make_goal_with_task(graph)
        assert graph.release_goal(goal_id, "run-A") is False


class TestDispatchRefusalUnderClaimedGoal:
    """Dispatch tools refuse a task whose ancestor goal is claimed by a DIFFERENT live run."""

    def _seed(self, tmp_path: Path) -> tuple[Path, MikadoGraph, int, int]:
        """Create a graph with a goal->task tree; return (root, graph, goal_id, task_id)."""
        db = tmp_path / ".milknado" / "milknado.db"
        db.parent.mkdir()
        g = MikadoGraph(db)
        goal_id, task_id = _make_goal_with_task(g)
        g.close()
        return tmp_path, MikadoGraph(db), goal_id, task_id

    def test_todo_run_refuses_task_under_claimed_goal(self, tmp_path: Path) -> None:
        root, graph, goal_id, task_id = self._seed(tmp_path)
        graph.claim_goal(goal_id, "run-coord-A", now=now_iso())
        graph.set_goal_pid(goal_id, "run-coord-A", os.getpid())  # live owner
        graph.close()

        from milknado.mcp_run import milknado_todo_run

        with pytest.raises(ValueError, match="goal.*claimed|claimed.*goal"):
            milknado_todo_run(task_id, project_root=str(root))

    def test_todo_run_start_refuses_task_under_claimed_goal(self, tmp_path: Path) -> None:
        root, graph, goal_id, task_id = self._seed(tmp_path)
        graph.claim_goal(goal_id, "run-coord-A", now=now_iso())
        graph.set_goal_pid(goal_id, "run-coord-A", os.getpid())  # live owner
        graph.close()

        from milknado.mcp_run import milknado_todo_run_start

        with pytest.raises(ValueError, match="goal.*claimed|claimed.*goal"):
            milknado_todo_run_start(task_id, project_root=str(root))

    def test_ralph_run_start_refuses_task_under_claimed_goal(self, tmp_path: Path) -> None:
        root, graph, goal_id, task_id = self._seed(tmp_path)
        graph.claim_goal(goal_id, "run-coord-A", now=now_iso())
        graph.set_goal_pid(goal_id, "run-coord-A", os.getpid())  # live owner
        graph.close()

        from milknado.mcp_ralph import milknado_ralph_run_start

        with pytest.raises(ValueError, match="goal.*claimed|claimed.*goal"):
            milknado_ralph_run_start(task_id, project_root=str(root))

    def test_dispatch_allowed_when_goal_not_claimed(self, tmp_path: Path) -> None:
        """No claimed goal → dispatch proceeds normally (kind check error, not goal error)."""
        root, graph, goal_id, task_id = self._seed(tmp_path)
        graph.close()

        from milknado.mcp_run import milknado_todo_run_start

        # Should NOT raise a goal-claimed error; may raise other validation errors
        try:
            milknado_todo_run_start(task_id, project_root=str(root))
        except ValueError as exc:
            assert "claimed" not in str(exc).lower(), (
                f"unexpected goal-claim refusal with no goal claimed: {exc}"
            )

    def test_dispatch_allowed_after_dead_claimant_reclaim(self, tmp_path: Path) -> None:
        """Dead claimant → reclaimed → dispatch no longer blocked."""
        root, graph, goal_id, task_id = self._seed(tmp_path)
        graph.claim_goal(goal_id, "run-coord-A", now=now_iso())
        graph.set_goal_pid(goal_id, "run-coord-A", _DEAD_PID)
        graph.close()

        from milknado.mcp_run import milknado_todo_run_start

        # Should NOT raise a goal-claimed error; dead claimant is reclaimed on dispatch
        try:
            milknado_todo_run_start(task_id, project_root=str(root))
        except ValueError as exc:
            assert "claimed" not in str(exc).lower(), (
                f"unexpected goal-claim refusal after dead claimant: {exc}"
            )

    def test_same_run_id_is_allowed_to_dispatch(self, tmp_path: Path) -> None:
        """Same run_id claiming the goal is not a blocker (own coordinator)."""
        # This case arises when a coordinator claims a goal then dispatches tasks under it.
        # We verify the fence checks DIFFERENT run_id only.
        root, graph, goal_id, task_id = self._seed(tmp_path)
        # claim with same run id that would be passed to dispatch check internals
        graph.claim_goal(goal_id, "run-coord-A", now=now_iso())
        graph.set_goal_pid(goal_id, "run-coord-A", os.getpid())
        graph.close()

        # The check_ancestor_goal_claim helper should return None (allowed) when
        # run_id matches the caller's own run_id — exercised via the graph method directly.
        g2 = MikadoGraph(root / ".milknado" / "milknado.db")
        # Same run_id: not blocked
        assert g2.ancestor_goal_claimed_by_other(task_id, caller_run_id="run-coord-A") is None
        # Different run_id: blocked
        assert g2.ancestor_goal_claimed_by_other(task_id, caller_run_id="run-coord-B") is not None
        g2.close()
