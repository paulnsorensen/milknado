from __future__ import annotations

import queue
import shlex
import uuid
from dataclasses import replace
from pathlib import Path

from milknado.domains.common import SessionAction, SessionContext, SessionInput, SessionView
from milknado.domains.graph import MikadoGraph, admit_session_command
from milknado.loop import QueueEmitter, RunManager
from milknado.loop._events import Event, EventData
from milknado.loop._run_types import RunConfig
from milknado.loop.manager import ManagedRun
from milknado.loop.sessions import is_supported


class LoopSessionMixin:
    def __init__(self, agent: str = "", graph: MikadoGraph | None = None) -> None:
        self._manager: RunManager = RunManager()
        self._queue: queue.Queue[Event[EventData]] = queue.Queue()
        self._emitter: QueueEmitter = QueueEmitter(self._queue)
        self._agent: str = agent
        self._graph: MikadoGraph | None = graph
        self._owner_incarnation: str = uuid.uuid4().hex
        self._command_ids: dict[tuple[str, str], str] = {}

    def _configure_session_commands(self, config: RunConfig, run_id: str) -> None:
        config.session_admitter = lambda command: self._admit_session_command(run_id, command)
        config.session_state_sink = lambda command, state: self._record_command_state(
            run_id, command, state
        )
        config.session_durable_drain = lambda: self._drain_durable_commands(run_id)

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
        commands = self._graph.commands
        run_id = run.state.run_id
        session.set_sink(lambda event: store.append(run_id, event))

        def publish_capabilities(
            context: SessionContext,
            actions: tuple[SessionAction, ...],
            invocation_id: str,
            permission_ids: tuple[str, ...],
        ) -> None:
            self._publish_session_capabilities(
                run_id, context, actions, invocation_id, permission_ids
            )

        session.set_capability_sink(
            publish_capabilities,
            on_close=lambda invocation: commands.close_owner(
                run_id, self._owner_incarnation, invocation
            ),
        )
        run.config.session_admitter = lambda command: self._admit_session_command(run_id, command)
        run.config.session_state_sink = lambda command, state: self._record_command_state(
            run_id, command, state
        )
        run.config.session_durable_drain = lambda: self._drain_durable_commands(run_id)

    def start_run(self, run_id: str) -> None:
        managed = self._manager.get_run(run_id)
        if managed is not None and managed.thread is None and self._graph is not None:
            context = managed.config.session_context
            if context is not None:
                self._graph.sessions.start(run_id, context)
        self._manager.start_run(run_id)

    def _publish_session_capabilities(  # noqa: PLR0913
        self,
        run_id: str,
        context: SessionContext,
        actions: tuple[str, ...],
        invocation_id: str,
        permission_ids: tuple[str, ...],
    ) -> None:
        del context
        if self._graph is None:
            return
        record = self._graph.runs.get(run_id)
        if record is None or record["status"] != "running":
            return
        _ = self._graph.commands.publish_capabilities(
            run_id,
            record["node_id"],
            invocation_id,
            self._owner_incarnation,
            actions,
            permission_ids,
        )

    def _admit_session_command(self, run_id: str, command: SessionInput) -> SessionInput | None:
        if self._graph is None:
            return command
        admitted = admit_session_command(
            self._graph, run_id, command, owner_incarnation=self._owner_incarnation
        )
        if admitted is None:
            return None
        self._command_ids[(run_id, command.request_id)] = admitted.command_id
        self._command_ids[(run_id, admitted.command_id)] = admitted.command_id
        return admitted

    def _drain_durable_commands(self, run_id: str) -> tuple[SessionInput, ...]:
        if self._graph is None:
            return ()
        commands = self._graph.commands.claim_pending(run_id, self._owner_incarnation)
        submitted: list[SessionInput] = []
        for command in commands:
            provider_id = (
                command.permission_id
                if command.action in {"approve", "deny"}
                else command.command_id
            ) or ""
            self._command_ids[(run_id, provider_id)] = command.command_id
            submitted.append(
                SessionInput(
                    action=command.action,
                    text=command.text,
                    request_id=provider_id,
                    command_id=command.command_id,
                )
            )
        return tuple(submitted)

    def _record_command_state(self, run_id: str, command: SessionInput, state: str) -> None:
        if self._graph is None or state in {"queued", "requested"}:
            return
        command_id = (
            command.command_id
            or self._command_ids.get((run_id, command.request_id))
            or command.request_id
        )
        record = self._graph.runs.get(run_id)
        if record is None:
            return
        status = "submitted" if state == "submitted" else state
        if status not in {"submitted", "delivered", "rejected", "unconfirmed"}:
            return
        stored = self._graph.commands.command(command_id)
        if stored is None:
            return
        if status == "submitted":
            _ = self._graph.commands.submit(stored)
        elif status == "delivered":
            _ = self._graph.commands.deliver(stored)
        elif status == "rejected":
            _ = self._graph.commands.reject(stored)
        else:
            _ = self._graph.commands.unconfirm(stored)

    def session_input(self, run_id: str, command: SessionInput) -> bool:
        return self._manager.session_input(run_id, command)

    def get_run_session(self, run_id: str) -> SessionView:
        view = self._manager.get_run_session(run_id)
        if self._graph is None:
            return view
        record = self._graph.runs.get(run_id)
        capabilities = self._graph.commands.capabilities(run_id) if record else None
        if record is None or capabilities is None:
            return replace(view, actions=(), owner_incarnation="", invocation_id="")
        if record["status"] != "running":
            return replace(
                view,
                actions=(),
                owner_incarnation=capabilities.owner_incarnation,
                invocation_id=capabilities.invocation_id,
            )
        return replace(
            view,
            actions=capabilities.actions,
            owner_incarnation=capabilities.owner_incarnation,
            invocation_id=capabilities.invocation_id,
        )

    def get_run_session_id(self, run_id: str) -> str | None:
        return self._manager.get_run_session_id(run_id)
