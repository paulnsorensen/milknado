"""Experimental T3 execution boundary and single-owner attempt journal."""

from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import connect
from typing import Literal, Protocol, TypeVar, cast
from uuid import uuid4

import msgspec

Status = Literal[
    "preparing",
    "queued",
    "starting",
    "running",
    "waiting",
    "completed",
    "interrupted",
    "failed",
    "cancelled",
    "rolled_back",
]
InterruptStatus = Literal[
    "interrupt_requested",
    "no_active_run",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "rolled_back",
]
Call = Callable[[str, dict[str, str | int]], object]
T = TypeVar("T")


class Unresolved(RuntimeError):
    """The spike cannot establish the run state or apply the requested transition."""


class Launch(msgspec.Struct, frozen=True):
    projectId: str
    threadId: str
    runId: str
    status: Status


class Run(msgspec.Struct, frozen=True):
    runId: str
    status: Status


class Thread(msgspec.Struct, frozen=True):
    threadId: str


class Read(msgspec.Struct, frozen=True):
    thread: Thread
    recentRuns: list[Run]


class Wait(msgspec.Struct, frozen=True):
    threadId: str
    runId: str
    status: Status
    timedOut: bool


class Interrupt(msgspec.Struct, frozen=True):
    threadId: str
    runId: str | None
    status: InterruptStatus


class Features(msgspec.Struct, frozen=True):
    cancellation: bool


class Capabilities(msgspec.Struct, frozen=True):
    features: Features


@dataclass(frozen=True)
class RunRef:
    environment: str
    thread: str
    run: str

    def arguments(self) -> dict[str, str | int]:
        return {"environmentId": self.environment, "threadId": self.thread, "runId": self.run}


class ExecutionBackend(Protocol):
    @property
    def identity(self) -> str: ...
    def start(self, task: str, request_key: str) -> RunRef: ...
    def inspect(self, ref: RunRef) -> Status: ...
    def wait(self, ref: RunRef) -> Wait: ...
    def request_cancel(self, ref: RunRef) -> InterruptStatus: ...
    def capabilities(self) -> Capabilities: ...


@dataclass(frozen=True)
class T3Backend:
    environment: str
    project: str
    call: Call

    @property
    def identity(self) -> str:
        return f"t3/{self.environment}/{self.project}"

    def _call(self, name: str, arguments: dict[str, str | int], schema: type[T]) -> T:
        return msgspec.convert(self.call(name, arguments), type=schema)

    def start(self, task: str, request_key: str) -> RunRef:
        reply = self._call(
            "t3_thread_launch",
            {
                "environmentId": self.environment,
                "projectId": self.project,
                "title": task,
                "message": task,
                "clientRequestId": request_key,
            },
            Launch,
        )
        if reply.projectId != self.project or not reply.threadId or not reply.runId:
            raise Unresolved("Launch did not bind the requested project and run")
        return RunRef(self.environment, reply.threadId, reply.runId)

    def inspect(self, ref: RunRef) -> Status:
        args = ref.arguments()
        del args["runId"]
        reply = self._call("t3_thread_read", args, Read)
        matches: list[Status] = [run.status for run in reply.recentRuns if run.runId == ref.run]
        if reply.thread.threadId != ref.thread or len(matches) != 1:
            raise Unresolved("Bound run is absent or ambiguous")
        return matches[0]

    def wait(self, ref: RunRef) -> Wait:
        reply = self._call("t3_thread_wait", ref.arguments() | {"timeoutMs": 5000}, Wait)
        if (reply.threadId, reply.runId) != (ref.thread, ref.run):
            raise Unresolved("Wait reply does not identify the bound run")
        return reply

    def request_cancel(self, ref: RunRef) -> InterruptStatus:
        if not self.capabilities().features.cancellation:
            raise Unresolved("T3 does not advertise cancellation")
        reply = self._call("t3_thread_interrupt", ref.arguments(), Interrupt)
        if (reply.threadId, reply.runId) != (ref.thread, ref.run):
            raise Unresolved("Interrupt reply does not identify the bound run")
        return reply.status

    def capabilities(self) -> Capabilities:
        return self._call(
            "orchestrator_capabilities", {"environmentId": self.environment}, Capabilities
        )


@dataclass(frozen=True)
class Attempt:
    task: str
    key: str
    ref: RunRef


@dataclass(frozen=True)
class Journal:
    path: Path
    backend: ExecutionBackend

    def initialize(self) -> None:
        with closing(connect(self.path)) as db, db:
            _ = db.execute("""CREATE TABLE IF NOT EXISTS attempt (
                task TEXT PRIMARY KEY, request_key TEXT NOT NULL UNIQUE,
                backend TEXT NOT NULL, environment TEXT, thread TEXT, run TEXT,
                accepted INTEGER NOT NULL DEFAULT 0
            )""")

    def start(self, task: str, *, needs_approval: bool = False) -> Attempt:
        if needs_approval:
            raise Unresolved("Permission approval is not supported by this spike")
        if task not in {"A", "B"}:
            raise ValueError(task)
        self._reserve(task)
        with closing(connect(self.path)) as db, db:
            _ = db.execute("BEGIN IMMEDIATE")
            row = cast(
                object,
                db.execute(
                    """SELECT request_key, backend, environment, thread, run
                    FROM attempt WHERE task=?""",
                    (task,),
                ).fetchone(),
            )
            if row is None:
                raise Unresolved("Attempt reservation disappeared")
            key, backend, environment, thread, run = msgspec.convert(
                row, type=tuple[str, str, str | None, str | None, str | None]
            )
            if backend != self.backend.identity:
                raise Unresolved("Backend changed; refusing fallback")
            if run is not None:
                if not environment or not thread:
                    raise Unresolved("Stored run is missing its environment or thread")
                return Attempt(task, key, RunRef(environment, thread, run))
            ref = self.backend.start(task, key)
            _ = db.execute(
                "UPDATE attempt SET environment=?, thread=?, run=? WHERE request_key=?",
                (ref.environment, ref.thread, ref.run, key),
            )
            return Attempt(task, key, ref)

    def _reserve(self, task: str) -> None:
        with closing(connect(self.path)) as db, db:
            _ = db.execute("BEGIN IMMEDIATE")
            if (
                task == "B"
                and not db.execute(
                    "SELECT 1 FROM attempt WHERE task='A' AND accepted=1"
                ).fetchone()
            ):
                raise Unresolved("A is not accepted")
            _ = db.execute(
                """INSERT INTO attempt(task, request_key, backend) VALUES (?, ?, ?)
                ON CONFLICT(task) DO NOTHING""",
                (task, uuid4().hex, self.backend.identity),
            )

    def accept(self, attempt: Attempt) -> None:
        self._resolve(attempt, accept=True)

    def supersede_reconciled(self, attempt: Attempt) -> None:
        """Reserve a successor after the workspace owner reconciles the completed run."""
        self._resolve(attempt, accept=False)

    def _resolve(self, attempt: Attempt, *, accept: bool) -> None:
        if self.backend.inspect(attempt.ref) != "completed":
            raise Unresolved("Expected a completed run")
        replacement = (
            (attempt.key, attempt.ref.environment, attempt.ref.thread, attempt.ref.run)
            if accept
            else (uuid4().hex, None, None, None)
        )
        with closing(connect(self.path)) as db, db:
            changed = db.execute(
                """UPDATE attempt SET accepted=?, request_key=?, environment=?, thread=?, run=?
                WHERE task=? AND request_key=? AND backend=?
                AND environment=? AND thread=? AND run=? AND accepted=0""",
                (
                    int(accept),
                    *replacement,
                    attempt.task,
                    attempt.key,
                    self.backend.identity,
                    attempt.ref.environment,
                    attempt.ref.thread,
                    attempt.ref.run,
                ),
            )
            if changed.rowcount != 1:
                raise Unresolved("Stale or already accepted attempt")
