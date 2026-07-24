"""Application orchestration for sweep, restructure, and fenced Git reap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.adapters.git import GitAdapter
from milknado.domains.common import GitPort
from milknado.domains.graph import (
    MikadoGraph,
    ReapFailure,
    ReapOutcome,
    ReapTarget,
    RebalanceReport,
    RebalanceState,
)

_DB_PATH = Path(".milknado") / "milknado.db"


@dataclass(frozen=True)
class RebalanceOptions:
    """Pass selection for one rebalance invocation."""

    dry_run: bool = False
    sweep: bool = True
    restructure: bool = True
    reap: bool = True


def _reason(exc: Exception) -> str:
    return str(exc) or type(exc).__name__


def _plan_target(git: GitPort, target: ReapTarget) -> ReapOutcome:
    path = Path(target.worktree_path)
    try:
        blocker = git.worktree_teardown_blocker(path)
    except Exception as exc:
        failure = ReapFailure("probe_worktree", target.worktree_path, _reason(exc))
        return ReapOutcome(target, failures=(failure,))
    if blocker is not None:
        failure = ReapFailure("probe_worktree", target.worktree_path, blocker)
        return ReapOutcome(target, failures=(failure,))
    return ReapOutcome(
        target,
        worktree_removed=True,
        branch_deleted=target.branch_name is not None,
    )


def _apply_target(git: GitPort, target: ReapTarget) -> ReapOutcome:
    try:
        git.remove_worktree(Path(target.worktree_path))
    except Exception as exc:
        failure = ReapFailure("remove_worktree", target.worktree_path, _reason(exc))
        return ReapOutcome(target, failures=(failure,))
    if target.branch_name is None:
        return ReapOutcome(target, worktree_removed=True)
    try:
        git.delete_branch(target.branch_name)
    except Exception as exc:
        failure = ReapFailure("delete_branch", target.branch_name, _reason(exc))
        return ReapOutcome(target, worktree_removed=True, failures=(failure,))
    return ReapOutcome(target, worktree_removed=True, branch_deleted=True)


def _run_reap(
    graph: MikadoGraph, git: GitPort, dry_run: bool
) -> tuple[tuple[ReapOutcome, ...], tuple[ReapFailure, ...]]:
    if dry_run:
        return tuple(_plan_target(git, target) for target in graph.get_reap_targets()), ()
    outcomes = graph.reap_archived(lambda target: _apply_target(git, target))
    try:
        git.prune_worktrees()
    except Exception as exc:
        failure = ReapFailure("prune_worktrees", "repository", _reason(exc))
        return outcomes, (failure,)
    return outcomes, ()


def _build_report(
    state: RebalanceState,
    outcomes: tuple[ReapOutcome, ...],
    extra_failures: tuple[ReapFailure, ...],
) -> RebalanceReport:
    reaped = tuple(
        outcome.target.worktree_path for outcome in outcomes if outcome.worktree_removed
    )
    preserved = tuple(
        outcome.target.worktree_path for outcome in outcomes if not outcome.worktree_removed
    )
    branches_deleted = tuple(
        outcome.target.branch_name
        for outcome in outcomes
        if outcome.branch_deleted and outcome.target.branch_name is not None
    )
    branches_kept = tuple(
        outcome.target.branch_name
        for outcome in outcomes
        if outcome.worktree_removed
        and not outcome.branch_deleted
        and outcome.target.branch_name is not None
    )
    failures = tuple(failure for outcome in outcomes for failure in outcome.failures)
    return RebalanceReport(
        archived_roots=state.archived_roots,
        inbox_id=state.inbox_id,
        inbox_created=state.inbox_created,
        moved_tasks=state.moved_tasks,
        chains=state.chains,
        lopsided=state.lopsided,
        reaped=reaped,
        branches_deleted=branches_deleted,
        preserved=preserved,
        branches_kept=branches_kept,
        failures=(*failures, *extra_failures),
    )


def rebalance(
    project_root: Path,
    options: RebalanceOptions = RebalanceOptions(),
    git: GitPort | None = None,
) -> RebalanceReport:
    """Run selected passes; dry-run mutates only a migrated in-memory snapshot."""
    db_path = project_root / _DB_PATH
    if options.dry_run and not db_path.exists():
        return RebalanceReport()
    graph = MikadoGraph.open_snapshot(db_path) if options.dry_run else MikadoGraph(db_path)
    try:
        state = graph.rebalance(sweep=options.sweep, restructure=options.restructure)
        outcomes: tuple[ReapOutcome, ...] = ()
        failures: tuple[ReapFailure, ...] = ()
        if options.reap:
            active_git = git if git is not None else GitAdapter(project_root)
            outcomes, failures = _run_reap(graph, active_git, options.dry_run)
        return _build_report(state, outcomes, failures)
    finally:
        graph.close()
