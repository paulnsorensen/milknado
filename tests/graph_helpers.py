"""Shared graph test helper — a single justified private-attribute seam.

Test files across the graph cluster reach into ``MikadoGraph._conn`` to run
raw SQL assertions the public API doesn't expose. Aliasing that access here
keeps the justified pyright suppression to one line instead of scattering
file-wide directives.
"""

from __future__ import annotations

import sqlite3
from threading import RLock

from milknado.domains.graph import MikadoGraph


def graph_conn(graph: MikadoGraph) -> sqlite3.Connection:
    return graph._conn  # pyright: ignore[reportPrivateUsage]


def graph_raw_conn(graph: MikadoGraph) -> sqlite3.Connection | None:
    return graph._raw_conn  # pyright: ignore[reportPrivateUsage]


def graph_close_stack(graph: MikadoGraph) -> list[str] | None:
    return graph._close_stack  # pyright: ignore[reportPrivateUsage]


def graph_is_closed(graph: MikadoGraph) -> bool:
    return graph._closed  # pyright: ignore[reportPrivateUsage]


def graph_lock(graph: MikadoGraph) -> RLock:
    return graph._lock  # pyright: ignore[reportPrivateUsage]
