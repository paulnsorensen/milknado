"""Milknado MCP rebalance tool — sweep/restructure/reap against a project root."""

from __future__ import annotations

import dataclasses
import logging

from milknado.app.rebalance import RebalanceOptions
from milknado.app.rebalance import rebalance as _rebalance_impl
from milknado.domains.graph import render_report
from milknado.mcp._core import Response, mcp, resolve_project_root

_logger = logging.getLogger(__name__)


@mcp.tool()
def milknado_rebalance(
    dry_run: bool = False,
    sweep: bool = True,
    restructure: bool = True,
    reap: bool = True,
    project_root: str = "",
) -> Response:
    """Rebalance the working tree: archive finished subtrees, regroup orphans, reap worktrees.

    Passes run in order sweep -> restructure -> reap; each can be disabled.
    dry_run performs reads only and reports what a real run would do, mutating
    nothing — not even creating the database file. Reap is fail-closed and
    never forced: teardown surprises land the worktree in `preserved`.
    Returns the RebalanceReport fields plus a rendered `report` string.
    """
    root = resolve_project_root(project_root or None)
    options = RebalanceOptions(
        dry_run=dry_run,
        sweep=sweep,
        restructure=restructure,
        reap=reap,
    )
    report = _rebalance_impl(root, options)
    _logger.info(
        "milknado_rebalance: dry_run=%s archived=%d moved=%d reaped=%d preserved=%d",
        dry_run,
        len(report.archived_roots),
        report.moved_tasks,
        len(report.reaped),
        len(report.preserved),
    )
    return {**dataclasses.asdict(report), "report": render_report(report, dry_run=dry_run)}
