from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
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
    incarnation: int

    @classmethod
    def from_step(  # noqa: V1xx
        cls, family: ProviderFamily, step: ProtocolStep, channel: SessionChannel
    ) -> RuntimeSession | None:
        identity = ProviderSessionIdentity.from_step(family, step)
        incarnation = channel.capture_incarnation()
        if identity is None or incarnation is None:
            return None
        return cls(identity, channel, incarnation)


@dataclass(frozen=True, slots=True)
class RuntimeActionReceipt:
    provider_session_id: str  # noqa: V1xx
    action: SessionInput  # noqa: V1xx
    state: ActionState  # noqa: V1xx


def _resume_command(command: list[str], identity: ProviderSessionIdentity) -> list[str]:
    if not command or Path(command[0]).name.lower().removesuffix(".exe") != identity.family:
        raise ValueError("resume provider family does not match command")
    if not identity.session_id or identity.session_id.startswith("-"):
        raise ValueError("resume requires a provider session id")
    args = command[1:]
    if identity.family == "claude":
        if any(arg in {"--resume", "--continue"} or arg.startswith("--resume=") for arg in args):
            raise ValueError("resume command already selects a Claude session")
        return [*command, "--resume", identity.session_id]
    if args and args[0] in {"exec", "app-server"}:
        args = args[1:]
    if args and args[0] == "resume":
        raise ValueError("resume command already selects a Codex thread")
    return [command[0], "resume", identity.session_id, *args]


def start_or_resume(request: RuntimeRequest) -> RuntimeResult:
    """Run an explicit turn against the provider's recorded session."""
    if request.resume is None:
        return RuntimeResult(run=run_session(request.spec, request.channel))
    resume = request.resume
    if request.spec.cwd is None or request.spec.cwd != resume.cwd or not resume.cwd.is_absolute():
        raise ValueError("resume worktree does not match run cwd")
    command = _resume_command(request.spec.cmd, resume.identity)
    if resume.identity.family == "codex":
        from milknado.loop.sessions._codex_policy import translate_argv

        if translate_argv(tuple(command), request.spec.cwd).cwd != resume.cwd.resolve():
            raise ValueError("resume worktree does not match Codex effective cwd")
    run = run_session(replace(request.spec, cmd=command), request.channel)
    confirmed = run.session_id == resume.identity.session_id
    outcome = "resumed" if confirmed else "unavailable"
    receipt = RecoveryReceipt(resume.identity, outcome, confirmed and run.terminal_confirmed)
    return RuntimeResult(run, receipt)


def submit_runtime_action(
    provider_session_id: str, action: SessionInput, session: RuntimeSession
) -> RuntimeActionReceipt:
    """Queue an action only for the matching active provider session."""
    if provider_session_id != session.identity.session_id:
        state: ActionState = "unknown_session"
    elif action.action not in runtime_capabilities(session.identity.family).native_actions:
        state = "unsupported"
    else:
        submitted = session.channel.submit_for_incarnation(session.incarnation, action)
        state = "unknown_session" if submitted is None else "queued" if submitted else "rejected"
    return RuntimeActionReceipt(provider_session_id, action, state)
