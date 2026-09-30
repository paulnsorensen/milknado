"""Application composition for unassociated loop-worker recovery."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from milknado.adapters import ProcessAdapter
from milknado.domains.dispatch import reap_orphaned_workers
from milknado.domains.dispatch.reap import ReapRequest
from milknado.domains.graph import (
    UnassociatedWorkers,
    default_worker_db_path,
    existing_standalone_worker_db,
)

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)


def reconcile_loop_workers(graph: MikadoGraph) -> None:
    """Recover unassociated workers from known evidence stores only."""
    deadline = time.monotonic() + 8.0
    default_path = default_worker_db_path()
    for path in dict.fromkeys((graph.db_path, default_path)):
        if path != graph.db_path and not existing_standalone_worker_db(path):
            continue
        request = ReapRequest(UnassociatedWorkers(), deadline=deadline, db_path=path)
        if not reap_orphaned_workers(graph, ProcessAdapter(), request):
            _logger.error("unassociated worker recovery unresolved: db_path=%s", path)
