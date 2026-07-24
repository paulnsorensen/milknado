"""Storage-agnostic contracts and rendering for graph rebalancing."""

from __future__ import annotations

from dataclasses import dataclass

INBOX_DESCRIPTION = "Inbox (milknado-managed)"


@dataclass(frozen=True)
class StructureReport:
    """Report-only structural findings."""

    chains: tuple[tuple[int, ...], ...] = ()
    lopsided: tuple[int, ...] = ()


@dataclass(frozen=True)
class RebalanceState:
    """Graph mutations completed by the sweep and restructure passes."""

    archived_roots: tuple[int, ...] = ()
    inbox_id: int | None = None
    inbox_created: bool = False
    moved_tasks: int = 0
    chains: tuple[tuple[int, ...], ...] = ()
    lopsided: tuple[int, ...] = ()


@dataclass(frozen=True)
class ReapTarget:
    """Archived worktree eligible for teardown."""

    node_id: int
    worktree_path: str
    branch_name: str | None = None


@dataclass(frozen=True)
class ReapFailure:
    """One handled Git failure, retained for operator telemetry."""

    operation: str
    target: str
    reason: str


@dataclass(frozen=True)
class ReapOutcome:
    """Result of processing one eligible worktree."""

    target: ReapTarget
    worktree_removed: bool = False
    branch_deleted: bool = False
    failures: tuple[ReapFailure, ...] = ()


@dataclass(frozen=True)
class RebalanceReport:
    """Outcome of one rebalance pass, ordered sweep -> restructure -> reap."""

    archived_roots: tuple[int, ...] = ()
    inbox_id: int | None = None
    inbox_created: bool = False
    moved_tasks: int = 0
    chains: tuple[tuple[int, ...], ...] = ()
    lopsided: tuple[int, ...] = ()
    reaped: tuple[str, ...] = ()
    branches_deleted: tuple[str, ...] = ()
    preserved: tuple[str, ...] = ()
    branches_kept: tuple[str, ...] = ()
    failures: tuple[ReapFailure, ...] = ()


def _counter_values(report: RebalanceReport) -> tuple[int, ...]:
    return (
        len(report.archived_roots),
        report.moved_tasks,
        len(report.chains),
        len(report.lopsided),
        len(report.reaped),
        len(report.branches_deleted),
        len(report.preserved),
        len(report.branches_kept),
    )


def _format_targets(values: tuple[str, ...]) -> str:
    return ", ".join(values) or "none"


def render_report(report: RebalanceReport, dry_run: bool = False) -> str:
    """Render fixed-order counters, exact reap targets, and handled failures."""
    counts = _counter_values(report)
    width = max(1, max(len(str(count)) for count in counts))
    archived, moved, chains, lopsided, reaped, branches, preserved, kept = (
        f"{count:0{width}d}" for count in counts
    )
    prefix = "[dry-run] Would " if dry_run else ""
    roots = _format_targets(tuple(str(root) for root in report.archived_roots))
    inbox = (
        "new"
        if report.inbox_created
        else (f"id {report.inbox_id}" if report.inbox_id is not None else "no Inbox")
    )
    chain_roots = _format_targets(tuple(str(chain[0]) for chain in report.chains))
    lopsided_ids = _format_targets(tuple(str(goal) for goal in report.lopsided))
    lines = [
        f"{prefix}Sweep: archived {archived} subtree(s) (roots: {roots})",
        f"{prefix}Restructure: moved {moved} orphan task(s) under Inbox ({inbox})",
        f"{prefix}Restructure: found {chains} deep chain(s) (roots: {chain_roots})",
        f"{prefix}Restructure: found {lopsided} lopsided goal(s) (ids: {lopsided_ids})",
        f"{prefix}Reap: removed {reaped} worktree(s), deleted {branches} branch(es), "
        f"preserved {preserved}, kept {kept} branch(es)",
        f"{prefix}Reap worktrees: remove [{_format_targets(report.reaped)}]; "
        f"preserve [{_format_targets(report.preserved)}]",
        f"{prefix}Reap branches: delete [{_format_targets(report.branches_deleted)}]; "
        f"keep [{_format_targets(report.branches_kept)}]",
    ]
    lines.extend(
        f"{prefix}Reap failure: {failure.operation} {failure.target}: {failure.reason}"
        for failure in report.failures
    )
    return "\n".join(lines)
