"""Vendored loop engine — derived from ralphify.

MIT License. Upstream: computerlovetech/ralphify.
Fork: paulnsorensen/ralphify (ec494875d0309c273c03f5f52a2cc3fabc96fa16).

Public API — import only these six names from milknado.loop:
"""

from milknado.loop._events import EventType, QueueEmitter
from milknado.loop._run_types import CompletionVerdict, RunConfig, RunStatus
from milknado.loop.manager import RunManager

__all__ = [
    "CompletionVerdict",
    "EventType",
    "QueueEmitter",
    "RunConfig",
    "RunManager",
    "RunStatus",
]
