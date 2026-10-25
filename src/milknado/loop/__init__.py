"""Milknado's loop engine, derived from ralphify.

MIT License. Upstream: computerlovetech/ralphify.
Fork source: paulnsorensen/ralphify (ec494875d0309c273c03f5f52a2cc3fabc96fa16).

Public API:
"""

from milknado.loop._events import EventType, QueueEmitter
from milknado.loop._process_adapter import ProcessAdapter
from milknado.loop._protection import loop_protection
from milknado.loop._run_types import CompletionVerdict, RunConfig, RunStatus
from milknado.loop._worker_recovery import CoordinatorWorkerRecovery
from milknado.loop.manager import RunManager
from milknado.loop.sessions import SessionChannel, create_protocol, run_session

__all__ = [
    "CompletionVerdict",
    "CoordinatorWorkerRecovery",
    "EventType",
    "ProcessAdapter",
    "QueueEmitter",
    "RunConfig",
    "RunManager",
    "RunStatus",
    "SessionChannel",
    "create_protocol",
    "loop_protection",
    "run_session",
]
