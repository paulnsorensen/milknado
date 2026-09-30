"""Owner-idle root settlement for an owner-attached run."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from milknado.domains.common.types import NodeStatus
from milknado.domains.graph import GoalAdmissionDenied

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger("milknado")

IdleGraphState = tuple[tuple[int, NodeStatus], ...]


def settle_once(
    graph: MikadoGraph, settled: IdleGraphState | None, settle: Callable[[], object]
) -> IdleGraphState | None:
    """Settle once per idle graph state; retry while a goal review blocks the root.

    Returns the graph state to remember, or None so the next rescan retries.
    """
    state = tuple((node.id, node.status) for node in graph.get_all_nodes())
    if state == settled:
        return settled
    try:
        _ = settle()
    except GoalAdmissionDenied as exc:
        _logger.info("root settlement waits for a goal review decision: %s", exc)
        return None
    return state
