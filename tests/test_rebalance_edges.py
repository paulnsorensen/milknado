"""Focused rebalance regression tests."""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from milknado.adapters.git import GitAdapter
from milknado.domains.common import NodeSpec
from milknado.domains.common.errors import GitOperationError
from milknado.domains.graph import MikadoGraph
from milknado.domains.graph import _rebalance as graph_rebalance
from tests.rebalance_helpers import (
    NOW,
    FakeGit,
    _archived_ids,
    _git,
    _insert_node,
    _project_with_db,
    _register_run,
    run_rebalance,
)

# ── #15: dry-run structural findings match the real run ──────────────────────


def _seed_chain_project(root: Path) -> tuple[int, ...]:
    """A DONE depth-5 chain (swept away) plus a LIVE depth-4 chain (reported)."""
    conn = _project_with_db(root)
    parent: int | None = None
    for i in range(5):
        parent = _insert_node(conn, f"done chain goal {i}", "done", kind="goal", parent_id=parent)
    live_ids: list[int] = []
    live_parent: int | None = None
    for i in range(4):
        live_parent = _insert_node(
            conn, f"live chain goal {i}", "in_progress", kind="goal", parent_id=live_parent
        )
        live_ids.append(live_parent)
    conn.commit()
    conn.close()
    return tuple(live_ids)


class TestDryRunStructuralParity:
    def test_dry_run_and_real_run_structural_reports_agree(self, tmp_path: Path) -> None:
        dry_root = tmp_path / "dry"
        dry_root.mkdir()
        real_root = tmp_path / "real"
        real_root.mkdir()
        live_chain = _seed_chain_project(dry_root)
        _seed_chain_project(real_root)

        dry = run_rebalance(dry_root, dry_run=True, git=FakeGit())
        real = run_rebalance(real_root, git=FakeGit())

        # The DONE chain is archived by the real sweep, so neither run reports
        # it; the LIVE chain survives and both runs report exactly it.
        assert dry.chains == real.chains == (live_chain,)
        assert dry.lopsided == real.lopsided == ()


# ── #15: dry-run reap plans the dirty probe the real run performs ─────────────


class TestDryRunReapProbeParity:
    def _dirty_worktree_project(self, root: Path) -> str:
        _git(root, "init", "-b", "main")
        _git(root, "commit", "--allow-empty", "-m", "init")
        wt = root / ".worktrees" / "wt-dirty"
        _git(root, "worktree", "add", "-b", "milknado/dirty", str(wt))
        (wt / "dirty.txt").write_text("uncommitted")
        conn = _project_with_db(root)
        node = _insert_node(
            conn,
            "archived dirty",
            "done",
            archived_at=NOW,
            worktree_path=str(wt),
            branch_name="milknado/dirty",
        )
        _register_run(conn, node)
        conn.commit()
        conn.close()
        return str(wt)

    def test_dry_run_plans_preservation_for_a_dirty_worktree(self, tmp_path: Path) -> None:
        dry_root = tmp_path / "dry"
        dry_root.mkdir()
        real_root = tmp_path / "real"
        real_root.mkdir()
        self._dirty_worktree_project(dry_root)
        real_wt = self._dirty_worktree_project(real_root)

        dry = run_rebalance(dry_root, dry_run=True, git=GitAdapter(dry_root))
        real = run_rebalance(real_root, git=GitAdapter(real_root))

        # Plan matches apply: the dirty worktree is preserved in both, removed
        # in neither, and no branch deletion is promised or taken.
        assert dry.reaped == real.reaped == ()
        assert dry.branches_deleted == real.branches_deleted == ()
        assert len(dry.preserved) == 1
        assert real.preserved == (real_wt,)


# ── #11/#12: transaction rollback on restructure failure ──────────────────────


class TestTransactionRollback:
    def test_restructure_failure_leaves_no_partial_archive_stamps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _project_with_db(tmp_path)
        _insert_node(conn, "done goal", "done", kind="goal")
        conn.commit()
        conn.close()

        def boom(_conn: sqlite3.Connection) -> int:
            raise RuntimeError("simulated restructure failure")

        monkeypatch.setattr(graph_rebalance, "_group_orphans", boom)
        with pytest.raises(RuntimeError, match="simulated restructure failure"):
            run_rebalance(tmp_path, git=FakeGit())

        check = sqlite3.connect(str(tmp_path / ".milknado" / "milknado.db"))
        check.row_factory = sqlite3.Row
        assert _archived_ids(check) == set()
        check.close()


class TestDeleteBranchAdapter:
    def _repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _git(repo, "commit", "--allow-empty", "-m", "init")
        return repo

    def test_deletes_merged_branch(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        _git(repo, "branch", "merged-branch")
        GitAdapter(repo).delete_branch("merged-branch")
        probe = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", "refs/heads/merged-branch"],
            cwd=repo,
            capture_output=True,
        )
        assert probe.returncode == 1  # branch gone

    def test_unmerged_branch_refusal_surfaces_nonzero_exit(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        _git(repo, "checkout", "-b", "unmerged-branch")
        _git(repo, "commit", "--allow-empty", "-m", "unmerged work")
        _git(repo, "checkout", "main")
        adapter = GitAdapter(repo)
        with pytest.raises(GitOperationError):
            adapter.delete_branch("unmerged-branch")
        assert adapter.branch_exists("unmerged-branch")  # refusal never forces


class TestGroupOrphansPreservesPrereqEdges:
    """Regression (post-merge F1): reparenting an orphan severs ONLY the
    parentage edge — prereq edges other nodes declared onto it survive."""

    def test_rebalance_preserves_dependent_prereq_edge_on_orphan(self, tmp_path: Path) -> None:
        db_path = tmp_path / ".milknado" / "milknado.db"
        db_path.parent.mkdir()
        graph = MikadoGraph(db_path)
        orphan = graph.add_node("root-level orphan task")
        dependent = graph.add_node("dependent", spec=NodeSpec(prereqs=(orphan.id,)))
        graph.close()

        report = run_rebalance(tmp_path, reap=False)

        graph = MikadoGraph(db_path)
        try:
            assert report.inbox_id is not None
            assert report.moved_tasks == 2  # both root-level tasks attach
            # The prereq edge survived the move: the dependent still lists the orphan.
            assert [n.id for n in graph.get_children(dependent.id)] == [orphan.id]
            # The dependent is NOT ready while its prereq is undone.
            assert dependent.id not in {n.id for n in graph.get_ready_nodes()}
            # The orphan now sits under the Inbox (diamond well-formedness).
            orphan_node = graph.get_node(orphan.id)
            assert orphan_node is not None
            assert orphan_node.parent_id == report.inbox_id
            assert orphan.id in {n.id for n in graph.get_children(report.inbox_id)}
        finally:
            graph.close()
