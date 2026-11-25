"""Milknado's loop engine, derived from ralphify.

MIT License. Upstream: computerlovetech/ralphify.
Fork source: paulnsorensen/ralphify (ec494875d0309c273c03f5f52a2cc3fabc96fa16).

Public API:
"""

from milknado.loop._loop_adapter import LoopAdapter
from milknado.loop._process_adapter import ProcessAdapter
from milknado.loop._protection import loop_protection
from milknado.loop._worker_recovery import CoordinatorWorkerRecovery
from milknado.loop.sessions import SessionChannel, create_protocol, run_session

__all__ = [
    "CoordinatorWorkerRecovery",
    "LoopAdapter",
    "ProcessAdapter",
    "SessionChannel",
    "create_protocol",
    "loop_protection",
    "run_session",
]
