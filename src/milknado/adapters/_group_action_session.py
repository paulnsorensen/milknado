from __future__ import annotations

from dataclasses import dataclass

import msgspec

from milknado.domains.common import SessionInput
from milknado.domains.graph import (
    GoalAdmissionDenied,
    MikadoGraph,
    TaskAttempt,
    admit_session_command,
)
from milknado.loop.sessions import RuntimeSession, SessionChannel


@dataclass(frozen=True, slots=True)
class GroupActionSession:
    runtime: RuntimeSession
    graph: MikadoGraph
    run_id: str
    owner_incarnation: str
    invocation_id: str

    @property
    def family(self) -> str:
        return self.runtime.family

    @property
    def provider_session_id(self) -> str:
        return self.runtime.provider_session_id

    @property
    def channel(self) -> SessionChannel:
        return self.runtime.channel

    def submit_action(self, action: SessionInput) -> str:
        if self.channel.capture_incarnation() != self.runtime.incarnation:
            return "unavailable"
        bound = msgspec.structs.replace(
            action, invocation_id=action.invocation_id or self.invocation_id
        )
        try:
            admitted = admit_session_command(
                self.graph, self.run_id, bound, owner_incarnation=self.owner_incarnation
            )
        except GoalAdmissionDenied:
            return "rejected"
        return "queued" if admitted is not None else "rejected"


def drain_group_commands(
    graph: MikadoGraph, attempt: TaskAttempt, owner: str
) -> tuple[SessionInput, ...]:
    return tuple(
        SessionInput(
            action=command.action,
            text=command.text,
            request_id=command.permission_id or command.command_id,
            command_id=command.command_id,
        )
        for command in graph.commands.claim_pending(attempt.attempt_id, owner)
    )


def record_group_command(graph: MikadoGraph, command: SessionInput, state: str) -> None:
    stored = graph.commands.command(command.command_id)
    transition = {
        "submitted": graph.commands.submit,
        "delivered": graph.commands.deliver,
        "rejected": graph.commands.reject,
        "unconfirmed": graph.commands.unconfirm,
    }.get(state)
    if stored is not None and transition is not None:
        _ = transition(stored)
