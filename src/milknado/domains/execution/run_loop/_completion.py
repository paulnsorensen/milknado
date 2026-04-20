from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from milknado.domains.execution.executor import RebaseConflict
from milknado.domains.execution.run_loop._logging import ts
from milknado.domains.execution.run_loop.display import _summarize_description

if TYPE_CHECKING:
    from rich.live import Live

_logger = logging.getLogger("milknado")


def handle_completion(
    loop: Any,
    run_id: str,
    success: bool,
    feature_branch: str,
    live: Live,
) -> tuple[int, int, list[RebaseConflict]]:
    completed = failed = 0
    conflicts: list[RebaseConflict] = []

    node_id = loop._active.pop(run_id)
    if loop._input.overlay_state == run_id:
        loop._input.overlay_state = None
    node = loop._graph.get_node(node_id)
    desc = _summarize_description(node.description) if node else str(node_id)
    start = loop._dispatched_at.pop(run_id, time.monotonic())
    duration = time.monotonic() - start

    if success:
        result = loop._executor.complete(node_id, feature_branch)
        loop._completion_durations.append(duration)
        if result.rebase_conflict:
            conflicts.append(result.rebase_conflict)
            files = ", ".join(result.rebase_conflict.conflicting_files)
            live.console.print(f"[red]✗[/red] [{node_id}] {desc} — conflict: {files}")
            _logger.warning(
                "node_conflict node_id=%d files=%s",
                node_id,
                list(result.rebase_conflict.conflicting_files),
            )
            loop._logs.append(f"[{ts()}] ✗ node {node_id} conflict")
            loop._attempts[node_id] = loop._attempts.get(node_id, 0) + 1
            if loop._strict:
                loop._failure_triggered = True
            failed += 1
        else:
            live.console.print(f"[green]✓[/green] [{node_id}] {desc}")
            _logger.info("node_completed node_id=%d duration=%.1fs", node_id, duration)
            loop._logs.append(f"[{ts()}] ✓ node {node_id} in {int(duration)}s")
            completed += 1
    else:
        loop._executor.fail(node_id)
        live.console.print(f"[red]✗[/red] [{node_id}] {desc}")
        _logger.warning("node_failed node_id=%d", node_id)
        loop._logs.append(f"[{ts()}] ✗ node {node_id} failed")
        loop._attempts[node_id] = loop._attempts.get(node_id, 0) + 1
        if loop._strict:
            loop._failure_triggered = True
        failed += 1

    return completed, failed, conflicts
