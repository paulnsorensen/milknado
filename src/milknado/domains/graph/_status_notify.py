"""Plugin status-change notification helpers for MikadoGraph.

Extracted from graph.py to keep the graph slice within its file-size budget.
These are module-level functions over a connection/plugin tuple, matching the
sibling _persistence / _mutations / _transitions modules.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from milknado.domains.common import NodeStatus

if TYPE_CHECKING:
    from milknado.domains.common import MikadoNode, PluginHook

_logger = logging.getLogger(__name__)


def node_status(
    conn: sqlite3.Connection,
    plugins: tuple[PluginHook, ...],
    node_id: int,
) -> NodeStatus | None:
    # Only the plugin-notify path needs the pre-transition status; skip the
    # extra SELECT entirely when no plugins are registered.
    if not plugins:
        return None
    row = conn.execute("SELECT status FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return NodeStatus(row[0]) if row else None


def notify_status_change(
    plugins: tuple[PluginHook, ...],
    node: MikadoNode | None,
    old_status: NodeStatus,
    new_status: NodeStatus,
) -> None:
    if not plugins or node is None:
        return
    for plugin in plugins:
        try:
            plugin.on_node_status_change(node, old_status, new_status)
        except Exception:
            _logger.exception("Plugin %s raised in on_node_status_change", plugin.meta.name)
