from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from milknado.domains.common import SessionAction, SessionEvent, SessionInput

ProviderFamily = Literal["claude", "codex"]
RecoveryOutcome = Literal["reattached", "resumed", "unknown_turn", "unavailable"]


@dataclass(frozen=True, slots=True)
class ProtocolStep:
    commands: tuple[bytes, ...] = ()
    events: tuple[SessionEvent, ...] = ()
    after_write_events: tuple[SessionEvent, ...] = ()
    done: bool = False
    result_text: str | None = None
    failed: bool = False
    interrupted: bool = False
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderSessionIdentity:
    family: ProviderFamily
    session_id: str

    @classmethod
    def from_step(
        cls, family: ProviderFamily, step: ProtocolStep
    ) -> ProviderSessionIdentity | None:
        """Use only the identity parsed from a provider protocol frame."""
        if not step.session_id:
            return None
        return cls(family=family, session_id=step.session_id)


@dataclass(frozen=True, slots=True)
class RuntimeRecoveryRequest:
    identity: ProviderSessionIdentity
    cwd: Path

    def __post_init__(self) -> None:
        if type(self.identity) is not ProviderSessionIdentity:
            raise TypeError("recovery requires provider session identity")


@dataclass(frozen=True, slots=True)
class RecoveryReceipt:
    identity: ProviderSessionIdentity
    outcome: RecoveryOutcome
    turn_confirmed: bool = False

    def __post_init__(self) -> None:
        if self.outcome == "unknown_turn" and self.turn_confirmed:
            raise ValueError("unknown turn cannot be confirmed")


class SessionProtocol(Protocol):
    command: tuple[str, ...]

    @property
    def actions(self) -> tuple[SessionAction, ...]: ...

    def start(self, prompt: str) -> ProtocolStep: ...
    def receive(self, line: bytes) -> ProtocolStep: ...
    def submit(self, command: SessionInput) -> ProtocolStep: ...
