"""Application orchestration for `milknado rebalance`: sweep -> restructure -> reap.

A real run wraps sweep+restructure in one write transaction (BEGIN IMMEDIATE).
A dry-run performs reads only: the plan comes from the same selection
functions without ever opening a write transaction — nothing is stamped,
moved, torn down, pruned, or even created on disk. Reap is fail-closed and
never forced: any teardown surprise (dirty, unlanded, missing, error) lands
the worktree in `preserved` and the pass continues; `git worktree prune`
always runs last in a real reap.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from milknado.adapters.git import GitAdapter
from milknado.domains.common import GitPort
from milknado.domains.graph import (
    RebalanceReport,
    find_archivable_roots,
    find_inbox_id,
    group_orphans,
    open_db,
    orphan_candidates,
    reap_eligible_ids,
    structural_report,
    sweep_archivable,
)

_DB_PATH = Path(".milknado") / "milknado.db"


def _reap_candidates(conn: sqlite3.Connection, eligible_ids: set[int]) -> list[sqlite3.Row]:
    """Reap-eligible nodes with a provisioned worktree and evidence of completion.

    Eligibility is the archived gate (`reap_eligible_ids`): only archived
    nodes' worktrees are ever torn down. Evidence = a registered run for the
    node that reached a terminal state (anything but 'running'). Nodes without
    a registered run are skipped silently: teardown only applies to
    milknado-managed worktrees.
    """
    if not eligible_ids:
        return []
    placeholders = ",".join("?" for _ in eligible_ids)
    return conn.execute(
        f"""
        SELECT n.id, n.worktree_path, n.branch_name
        FROM nodes n
        WHERE n.worktree_path IS NOT NULL
          AND n.id IN ({placeholders})
          AND EXISTS (
              SELECT 1 FROM runs r
              WHERE r.node_id = n.id AND r.status != 'running'
          )
        ORDER BY n.id
        """,  # noqa: S608
        sorted(eligible_ids),
    ).fetchall()


def _no_run_worktree_count(conn: sqlite3.Connection, eligible_ids: set[int]) -> int:
    """Eligible nodes carrying a worktree but no terminal run record.

    `_reap_candidates` skips these (no evidence the worktree is
    milknado-managed and complete); the count lets the report name them
    instead of letting them vanish silently.
    """
    if not eligible_ids:
        return 0
    placeholders = ",".join("?" for _ in eligible_ids)
    row = conn.execute(
        f"""
        SELECT COUNT(*) FROM nodes n
        WHERE n.worktree_path IS NOT NULL
          AND n.id IN ({placeholders})
          AND NOT EXISTS (
              SELECT 1 FROM runs r
              WHERE r.node_id = n.id AND r.status != 'running'
          )
        """,  # noqa: S608
        sorted(eligible_ids),
    ).fetchone()
    return int(row[0])


def _reap(
    conn: sqlite3.Connection,
    git: GitPort,
    *,
    eligible_ids: set[int],
    dry_run: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    reaped: list[str] = []
    branches: list[str] = []
    preserved: list[str] = []
    kept: list[str] = []
    for row in _reap_candidates(conn, eligible_ids):
        worktree_path = row["worktree_path"]
        branch = row["branch_name"]
        if dry_run:
            # Plan only: the merged check never runs, so no branch deletion is
            # promised — over-promising would diverge plan from apply. The
            # dirty/unlanded probe IS read-only and runs here too, so a worktree
            # the real run would preserve is planned as preserved.
            try:
                blocker = git.worktree_teardown_blocker(Path(worktree_path))
            except Exception:
                blocker = "probe failed"  # fail closed, same as the real run
            (preserved if blocker is not None else reaped).append(worktree_path)
            continue
        try:
            git.remove_worktree(Path(worktree_path))
        except Exception:
            # Fail closed: any teardown surprise (dirty, unlanded, stale/missing
            # path, git error) preserves the worktree — never force-remove it,
            # never delete its branch, and never brick the rest of the pass.
            preserved.append(worktree_path)
            continue
        reaped.append(worktree_path)
        if branch:
            try:
                git.delete_branch(branch)
            except Exception:
                # `git branch -d` refuses an unmerged head by design. The
                # worktree is gone (it stays in `reaped` — never two buckets);
                # the surviving artifact is the branch, reported by name.
                kept.append(branch)
            else:
                branches.append(branch)
    if not dry_run:
        # Prune heals stale registrations (e.g. externally removed worktrees)
        # and runs even when some teardowns were preserved.
        git.prune_worktrees()
    return tuple(reaped), tuple(branches), tuple(preserved), tuple(kept)


def rebalance(
    project_root: Path,
    *,
    dry_run: bool = False,
    sweep: bool = True,
    restructure: bool = True,
    reap: bool = True,
    git: GitPort | None = None,
) -> RebalanceReport:
    """Run the rebalance passes against <project_root>/.milknado/milknado.db.

    `git` is injectable for tests; production callers leave it None and get a
    GitAdapter rooted at project_root.
    """
    db_path = project_root / _DB_PATH
    if dry_run and not db_path.exists():
        # Mutate nothing — not even creating the database file.
        return RebalanceReport()
    conn = open_db(db_path)
    try:
        archived_roots: tuple[int, ...] = ()
        inbox_id: int | None = None
        moved_tasks = 0
        chains: tuple[tuple[int, ...], ...] = ()
        lopsided: tuple[int, ...] = ()
        eligible: set[int] = set()
        if dry_run:
            # Reads only: the same selection functions produce the plan without
            # a write transaction, so a dry-run never blocks a live writer.
            # `eligible` = archived ∪ about-to-be-archived, so the restructure
            # and reap plans reflect the post-sweep forest the real run sees.
            eligible = reap_eligible_ids(conn, sweep_pending=sweep)
            if sweep:
                archived_roots = tuple(find_archivable_roots(conn))
            if restructure:
                moved_tasks = len([i for i in orphan_candidates(conn) if i not in eligible])
                inbox_id = find_inbox_id(conn)
                if inbox_id in eligible:
                    # The Inbox itself is about to be archived; the real run
                    # find-or-creates a fresh one, so the plan must predict
                    # "(new)" rather than name the doomed id.
                    inbox_id = None
        elif sweep or restructure:
            conn.execute("BEGIN IMMEDIATE")
            try:
                archived: list[int] = []
                if sweep:
                    archived = sweep_archivable(conn)
                if restructure:
                    moved_tasks = group_orphans(conn)
                    inbox_id = find_inbox_id(conn)
                conn.commit()
                archived_roots = tuple(archived)
            except Exception:
                conn.rollback()
                raise
        if restructure:
            # Report-only; structural findings never mutate. A real run reads
            # the post-sweep forest; a dry-run excludes the about-to-be-archived
            # set so its findings match what the apply would report.
            structure = (
                structural_report(conn, exclude_ids=eligible)
                if dry_run
                else structural_report(conn)
            )
            chains, lopsided = structure.chains, structure.lopsided
        reaped: tuple[str, ...] = ()
        branches: tuple[str, ...] = ()
        preserved: tuple[str, ...] = ()
        kept: tuple[str, ...] = ()
        notes: tuple[str, ...] = ()
        if reap:
            if not dry_run:
                eligible = reap_eligible_ids(conn, sweep_pending=False)
            reaped, branches, preserved, kept = _reap(
                conn,
                git if git is not None else GitAdapter(project_root),
                eligible_ids=eligible,
                dry_run=dry_run,
            )
            skipped = _no_run_worktree_count(conn, eligible)
            if skipped:
                notes = (f"skipped {skipped} worktree(s) with no run record",)
        return RebalanceReport(
            archived_roots=archived_roots,
            inbox_id=inbox_id,
            moved_tasks=moved_tasks,
            chains=chains,
            lopsided=lopsided,
            reaped=reaped,
            branches_deleted=branches,
            preserved=preserved,
            branches_kept=kept,
            notes=notes,
        )
    finally:
        conn.close()
