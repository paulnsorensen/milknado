"""Focused rebalance regression tests."""

from __future__ import annotations

# Structural report setup intentionally ignores database write results.
import sqlite3

from milknado.domains.graph.rebalance import (
    RebalanceReport,
    render_report,
)
from tests.rebalance_helpers import (
    NOW,
    insert_node,
    structural_report,
)

# ── #13: structural report (report-only) ─────────────────────────────────────


class TestStructuralReport:
    def _goal_chain(self, conn: sqlite3.Connection, depth: int) -> list[int]:
        ids: list[int] = []
        parent: int | None = None
        for i in range(depth):
            node_id = insert_node(
                conn, f"chain goal {i}", "in_progress", kind="goal", parent_id=parent
            )
            ids.append(node_id)
            parent = node_id
        return ids

    def test_chain_at_depth_4_is_reported(self, conn: sqlite3.Connection) -> None:
        ids = self._goal_chain(conn, 4)
        report = structural_report(conn)
        assert report.chains == (tuple(ids),)
        assert report.lopsided == ()

    def test_chain_at_depth_3_is_not_reported(self, conn: sqlite3.Connection) -> None:
        _ = self._goal_chain(conn, 3)
        assert structural_report(conn).chains == ()

    def test_chain_head_only_is_reported_for_longer_chains(self, conn: sqlite3.Connection) -> None:
        ids = self._goal_chain(conn, 5)
        report = structural_report(conn)
        # One maximal chain, not overlapping sub-chains at depths 4 and 5.
        assert report.chains == (tuple(ids),)

    def test_branching_breaks_the_chain(self, conn: sqlite3.Connection) -> None:
        g1 = insert_node(conn, "g1", "in_progress", kind="goal")
        g2 = insert_node(conn, "g2", "in_progress", kind="goal", parent_id=g1)
        g3 = insert_node(conn, "g3", "in_progress", kind="goal", parent_id=g2)
        # g3 forks into two goal children, so no single-child run reaches 4.
        _ = insert_node(conn, "g4a", "in_progress", kind="goal", parent_id=g3)
        _ = insert_node(conn, "g4b", "in_progress", kind="goal", parent_id=g3)
        assert structural_report(conn).chains == ()

    def test_goal_with_20_direct_tasks_is_reported(self, conn: sqlite3.Connection) -> None:
        goal = insert_node(conn, "fat goal", "in_progress", kind="goal")
        for i in range(20):
            _ = insert_node(conn, f"task {i}", "pending", parent_id=goal)
        assert structural_report(conn).lopsided == (goal,)

    def test_goal_with_19_direct_tasks_is_not_reported(self, conn: sqlite3.Connection) -> None:
        goal = insert_node(conn, "almost fat goal", "in_progress", kind="goal")
        for i in range(19):
            _ = insert_node(conn, f"task {i}", "pending", parent_id=goal)
        assert structural_report(conn).lopsided == ()

    def test_archived_nodes_are_excluded(self, conn: sqlite3.Connection) -> None:
        parent: int | None = None
        for i in range(4):
            parent = insert_node(
                conn,
                f"archived chain goal {i}",
                "done",
                kind="goal",
                parent_id=parent,
                archived_at=NOW,
            )
        assert structural_report(conn).chains == ()

    def test_structural_report_never_mutates(self, conn: sqlite3.Connection) -> None:
        _ = self._goal_chain(conn, 4)
        goal = insert_node(conn, "fat goal", "in_progress", kind="goal")
        for i in range(20):
            _ = insert_node(conn, f"task {i}", "pending", parent_id=goal)
        nodes_before = conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
        edges_before = conn.execute("SELECT * FROM edges ORDER BY parent_id, child_id").fetchall()
        report = structural_report(conn)
        assert report.chains and report.lopsided  # findings exist, so mutation would show
        assert conn.execute("SELECT * FROM nodes ORDER BY id").fetchall() == nodes_before
        assert (
            conn.execute("SELECT * FROM edges ORDER BY parent_id, child_id").fetchall()
            == edges_before
        )


# ── #16: report ordering and alignment ───────────────────────────────────────


class TestRenderReport:
    def _report(self) -> RebalanceReport:
        return RebalanceReport(
            archived_roots=(1, 2, 3),
            inbox_id=9,
            moved_tasks=12,
            chains=((4, 5, 6, 7),),
            lopsided=(8,),
            reaped=("/a", "/b"),
            branches_deleted=("b1",),
            preserved=(),
        )

    def test_ordering_sweep_restructure_reap(self) -> None:
        lines = render_report(self._report()).splitlines()
        assert lines[0].startswith("Sweep:")
        assert all(line.startswith("Restructure:") for line in lines[1:4])
        assert lines[4].startswith("Reap:")

    def test_counters_equal_width(self) -> None:
        rendered = render_report(self._report())
        # Widest counter is moved_tasks=12, so every counter is zero-padded to 2.
        assert "archived 03 subtree(s)" in rendered
        assert "moved 12 orphan task(s)" in rendered
        assert "found 01 deep chain(s)" in rendered
        assert "found 01 lopsided goal(s)" in rendered
        assert "removed 02 worktree(s)" in rendered
        assert "deleted 01 branch(es)" in rendered
        assert "preserved 00" in rendered

    def test_dry_run_preserves_alignment(self) -> None:
        report = self._report()
        plain = render_report(report).splitlines()
        dry = render_report(report, dry_run=True).splitlines()
        prefix = "[dry-run] Would "
        assert len(plain) == len(dry)
        for p, d in zip(plain, dry, strict=True):
            assert d == prefix + p

    def test_exact_line_strings(self) -> None:
        assert render_report(self._report()).splitlines() == [
            "Sweep: archived 03 subtree(s) (roots: 1, 2, 3)",
            "Restructure: moved 12 orphan task(s) under Inbox (id 9)",
            "Restructure: found 01 deep chain(s) (roots: 4)",
            "Restructure: found 01 lopsided goal(s) (ids: 8)",
            "Reap: removed 02 worktree(s), deleted 01 branch(es), preserved 00, "
            + "kept 00 branch(es)",
            "Reap worktrees: remove [/a, /b]; preserve [none]",
            "Reap branches: delete [b1]; keep [none]",
        ]

    def test_dry_run_exact_line_strings(self) -> None:
        assert render_report(self._report(), dry_run=True).splitlines() == [
            "[dry-run] Would Sweep: archived 03 subtree(s) (roots: 1, 2, 3)",
            "[dry-run] Would Restructure: moved 12 orphan task(s) under Inbox (id 9)",
            "[dry-run] Would Restructure: found 01 deep chain(s) (roots: 4)",
            "[dry-run] Would Restructure: found 01 lopsided goal(s) (ids: 8)",
            "[dry-run] Would Reap: removed 02 worktree(s), deleted 01 branch(es), "
            + "preserved 00, kept 00 branch(es)",
            "[dry-run] Would Reap worktrees: remove [/a, /b]; preserve [none]",
            "[dry-run] Would Reap branches: delete [b1]; keep [none]",
        ]

    def test_kept_branches_render_suffix(self) -> None:
        report = RebalanceReport(reaped=("/a",), branches_kept=("b1", "b2"))
        line = render_report(report).splitlines()[4]
        assert line == (
            "Reap: removed 1 worktree(s), deleted 0 branch(es), preserved 0, kept 2 branch(es)"
        )
        assert render_report(report).splitlines()[6] == (
            "Reap branches: delete [none]; keep [b1, b2]"
        )

    def test_dry_run_renders_would_be_inbox_as_new(self) -> None:
        report = RebalanceReport(moved_tasks=2, inbox_created=True)
        line = render_report(report, dry_run=True).splitlines()[1]
        assert line == "[dry-run] Would Restructure: moved 2 orphan task(s) under Inbox (new)"
        # A real run with no Inbox and nothing moved still says "no Inbox".
        assert "under Inbox (no Inbox)" in render_report(RebalanceReport()).splitlines()[1]

    def test_empty_report_exact_strings(self) -> None:
        assert render_report(RebalanceReport()).splitlines() == [
            "Sweep: archived 0 subtree(s) (roots: none)",
            "Restructure: moved 0 orphan task(s) under Inbox (no Inbox)",
            "Restructure: found 0 deep chain(s) (roots: none)",
            "Restructure: found 0 lopsided goal(s) (ids: none)",
            "Reap: removed 0 worktree(s), deleted 0 branch(es), preserved 0, kept 0 branch(es)",
            "Reap worktrees: remove [none]; preserve [none]",
            "Reap branches: delete [none]; keep [none]",
        ]
