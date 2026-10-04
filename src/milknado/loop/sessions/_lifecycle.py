from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from milknado.domains.common import SessionInput
from milknado.loop._agent import AgentResult, AgentRunSpec
from milknado.loop.sessions._capabilities import runtime_capabilities
from milknado.loop.sessions._channel import SessionChannel
from milknado.loop.sessions._protocol import (
    ProtocolStep,
    ProviderFamily,
    ProviderSessionIdentity,
    RecoveryReceipt,
    RuntimeRecoveryRequest,
)
from milknado.loop.sessions._runtime import run_session

ActionState = Literal["queued", "rejected", "unsupported", "unknown_session"]


@dataclass(frozen=True, slots=True)
class RuntimeRequest:
    spec: AgentRunSpec
    channel: SessionChannel
    resume: RuntimeRecoveryRequest | None = None


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    run: AgentResult | None  # noqa: V1xx
    recovery: RecoveryReceipt | None = None  # noqa: V1xx


@dataclass(frozen=True, slots=True)
class RuntimeSession:
    identity: ProviderSessionIdentity
    channel: SessionChannel

    @classmethod
    def from_step(  # noqa: V1xx
        cls, family: ProviderFamily, step: ProtocolStep, channel: SessionChannel
    ) -> RuntimeSession | None:
        identity = ProviderSessionIdentity.from_step(family, step)
        return cls(identity, channel) if identity is not None else None


@dataclass(frozen=True, slots=True)
class RuntimeActionReceipt:
    provider_session_id: str  # noqa: V1xx
    action: SessionInput  # noqa: V1xx
    state: ActionState  # noqa: V1xx


def start_or_resume(request: RuntimeRequest) -> RuntimeResult:
    """Start a session or report that native resume is not wired."""
    if request.resume is not None:
        return RuntimeResult(
            run=None,
            recovery=RecoveryReceipt(request.resume.identity, "unsupported"),
        )
    return RuntimeResult(run=run_session(request.spec, request.channel))


def submit_runtime_action(
    provider_session_id: str, action: SessionInput, session: RuntimeSession
) -> RuntimeActionReceipt:
    """Queue an action only for the matching active provider session."""
    if provider_session_id != session.identity.session_id:
        state: ActionState = "unknown_session"
    elif action.action not in runtime_capabilities(session.identity.family).native_actions:
        state = "unsupported"
    elif session.channel.submit(action):
        state = "queued"
    else:
        state = "rejected"
    return RuntimeActionReceipt(provider_session_id, action, state)
