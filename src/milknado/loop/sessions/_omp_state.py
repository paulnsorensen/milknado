from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import SessionAction
from milknado.loop.sessions._omp_wire import ChunkDecoder, rpc_command


@dataclass(slots=True)
class Pending:
    action: str
    text: str
    acknowledged: bool = False


@dataclass(frozen=True, slots=True)
class UiRequest:
    method: str
    title: str = ""
    options: tuple[str, ...] = ()


class OmpState:
    _BASE_ACTIONS: tuple[SessionAction, ...] = ("steer", "follow_up", "interrupt")
    _INTERACTIVE_UI: frozenset[str] = frozenset({"confirm", "select", "input", "editor"})
    _PASSIVE_UI: frozenset[str] = frozenset(
        {"notify", "setStatus", "setWidget", "setTitle", "set_editor_text"}
    )

    def __init__(self, argv: tuple[str, ...], cwd: Path) -> None:
        self.command: tuple[str, ...] = rpc_command(argv)
        self.cwd: Path = cwd
        self._sequence: int = 0
        self._started: bool = False
        self._active: bool = False
        self._failed: bool = False
        self._interrupt_accepted: bool = False
        self._saw_aborted: bool = False
        self._pending: dict[str, Pending] = {}
        self._queued: dict[str, tuple[str, str]] = {}
        self._ui_requests: dict[str, UiRequest] = {}
        self._resolved: set[str] = set()
        self._assistant_id: str = ""
        self._session_id: str | None = None
        self._result_text: str | None = None
        self._chunks: ChunkDecoder = ChunkDecoder()
        self._assistant_text: str = ""

    @property
    def actions(self) -> tuple[SessionAction, ...]:
        if self._ui_requests:
            return (*self._BASE_ACTIONS, "approve", "deny")
        return self._BASE_ACTIONS
