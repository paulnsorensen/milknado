"""Per-node host worker slot leases held by one executing supervisor."""

from __future__ import annotations

import time
from pathlib import Path

from milknado.domains.common.protocols import HostCapacityPort, SlotLease
from milknado.domains.graph import HostCapacityFull

# A detached runner re-takes the slot its launcher probed a moment earlier. The
# wait only covers that hand-off; a pool that stays full defers the node instead.
ADOPTED_SLOT_WAIT_SECONDS = 2.0
_POLL_SECONDS = 0.25


class SlotLedger:
    def __init__(self) -> None:
        self._port: HostCapacityPort | None = None
        self._leases: dict[int, SlotLease] = {}

    def use(self, port: HostCapacityPort) -> None:
        self._port = port

    def take(self, run_id: str, node_id: int, root: Path, *, wait: bool) -> None:
        """Hold a slot for the node; raise ``HostCapacityFull`` when none frees up."""
        if self._port is None or node_id in self._leases:
            return
        deadline = time.monotonic() + (ADOPTED_SLOT_WAIT_SECONDS if wait else 0.0)
        while True:
            try:
                self._leases[node_id] = self._port.acquire(run_id, node_id, root)
                return
            except HostCapacityFull:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(_POLL_SECONDS)

    def drop(self, node_id: int) -> None:
        lease = self._leases.pop(node_id, None)
        if lease is not None:
            lease.release()

    def settle(self, node_id: int, *, running: bool) -> None:
        """Release the node's slot once its run is no longer in flight."""
        if not running:
            self.drop(node_id)
