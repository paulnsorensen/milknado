"""Claim a node under a host worker slot: acquire, claim, free the slot on failure."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.protocols import HostCapacityPort, SlotLease
from milknado.domains.dispatch._runstate import now_iso

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph


def claim_with_host_slot(
    graph: MikadoGraph,
    pool: HostCapacityPort | None,
    node_run: tuple[int, str],
    root: Path,
) -> SlotLease | None:
    """Take a host slot before the graph claim; free it when the claim fails.

    Return ``None`` when no pool is configured. ``HostCapacityFull`` propagates
    before the graph is touched, so a full pool never claims the node.
    """
    node_id, run_id = node_run
    lease = pool.acquire(run_id, node_id, root) if pool is not None else None
    try:
        graph.claim_node_for_dispatch(node_id, run_id, now=now_iso())
    except BaseException:
        if lease is not None:
            lease.release()
        raise
    return lease
