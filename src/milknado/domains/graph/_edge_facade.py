"""Synchronized edge mutation facade for MikadoGraph."""

from __future__ import annotations

import sqlite3

from milknado.domains.common import MikadoEdge
from milknado.domains.graph._analytics_facade import synchronized
from milknado.domains.graph._creation import add_edge, remove_edge

__all__ = ["_EdgeFacade"]


class _EdgeFacade:
    @property
    def _conn(self) -> sqlite3.Connection:
        raise NotImplementedError

    @synchronized
    def add_edge(self, parent_id: int, child_id: int) -> MikadoEdge:
        return add_edge(self._conn, parent_id, child_id)

    @synchronized
    def remove_edge(self, parent_id: int, child_id: int) -> bool:
        return remove_edge(self._conn, parent_id, child_id)
