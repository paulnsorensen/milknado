from __future__ import annotations

import queue
import shlex
from pathlib import Path

from milknado.domains.common import SessionContext, SessionInput, SessionView
from milknado.domains.graph import MikadoGraph
from milknado.loop import QueueEmitter, RunManager
from milknado.loop._events import Event, EventData
from milknado.loop.manager import ManagedRun
from milknado.loop.sessions import is_supported


class LoopSessionMixin:
    def __init__(self, agent: str = "", graph: MikadoGraph | None = None) -> None:
        self._manager: RunManager = RunManager()
        self._queue: queue.Queue[Event[EventData]] = queue.Queue()
        self._emitter: QueueEmitter = QueueEmitter(self._queue)
        self._agent: str = agent
        self._graph: MikadoGraph | None = graph

    def _session_context(
        self, agent_cmd: str, project_root: Path, base_oid: str | None
    ) -> SessionContext | None:
        command = tuple(shlex.split(agent_cmd))
        if self._graph is None or not is_supported(command):
            return None
        return SessionContext(
            family=Path(command[0]).stem.lower(),
            cwd=str(project_root),
            base_oid=base_oid or "",
        )

    def _attach_session_sink(self, run: ManagedRun) -> None:
        if self._graph is None:
            return
        session = run.state.session
        if session is None:
            raise RuntimeError("structured runs require a session channel")
        store = self._graph.sessions
        run_id = run.state.run_id
        session.set_sink(lambda event: store.append(run_id, event))

    def start_run(self, run_id: str) -> None:
        managed = self._manager.get_run(run_id)
        if managed is not None and managed.thread is None and self._graph is not None:
            context = managed.config.session_context
            if context is not None:
                self._graph.sessions.start(run_id, context)
        self._manager.start_run(run_id)

    def session_input(self, run_id: str, command: SessionInput) -> bool:
        return self._manager.session_input(run_id, command)

    def get_run_session(self, run_id: str) -> SessionView:
        return self._manager.get_run_session(run_id)

    def get_run_session_id(self, run_id: str) -> str | None:
        return self._manager.get_run_session_id(run_id)
