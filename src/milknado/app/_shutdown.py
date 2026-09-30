"""Record entry-point shutdown intent without interrupting application state."""

from __future__ import annotations

import logging
import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from queue import Empty, Queue
from threading import Thread, current_thread, main_thread
from time import monotonic
from types import FrameType
from typing import TypeVar

STOP_TIMEOUT_SECONDS = 8.0


class ShutdownSignal(RuntimeError):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"received signal {signum}")


class ShutdownIntent:
    def __init__(self) -> None:
        self.signum: int | None = None
        self.started_at: float | None = None
        self.cleanup_confirmed: bool | None = None
        self._installed = False

    @property
    def requested(self) -> bool:
        return self.signum is not None

    def record(self, signum: int, frame: FrameType | None) -> None:
        del frame
        if self.signum is None:
            self.signum = signum
            self.started_at = monotonic()

    def deadline(self, timeout: float) -> float | None:
        return None if self.started_at is None else self.started_at + timeout

    def rearm(self) -> None:
        if not self._installed:
            return
        for signum in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
            if signum is not None:
                signal.signal(signum, self.record)

    @contextmanager
    def installed(self) -> Iterator[None]:
        signals = [signal.SIGINT, signal.SIGTERM]
        if hangup := getattr(signal, "SIGHUP", None):
            signals.append(hangup)
        previous = {signum: signal.getsignal(signum) for signum in signals}
        try:
            self._installed = True
            self.rearm()
            yield
        finally:
            self._installed = False
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def bounded_stop(stop: Callable[[float], bool], deadline: float) -> bool:
    outcomes: Queue[bool | Exception] = Queue(maxsize=1)

    def execute() -> None:
        try:
            outcomes.put(stop(deadline))
        except Exception as exc:  # noqa: BLE001 - Transfer the stop failure to the caller.
            outcomes.put(exc)

    Thread(target=execute, name="milknado-force-stop", daemon=True).start()
    try:
        outcome = outcomes.get(timeout=max(0.0, deadline - monotonic()))
    except Empty:
        return False
    if isinstance(outcome, Exception):
        raise outcome
    return outcome


_T = TypeVar("_T")


def supervise(
    run: Callable[[], _T],
    intent: ShutdownIntent,
    stop: Callable[[float], bool],
    thread_name: str,
) -> _T:
    outcomes: Queue[_T | Exception] = Queue(maxsize=1)

    def execute() -> None:
        try:
            outcomes.put(run())
        except Exception as exc:  # noqa: BLE001 - Transfer the run failure to the caller.
            outcomes.put(exc)

    handlers = intent.installed() if current_thread() is main_thread() else nullcontext()
    with handlers:
        worker = Thread(target=execute, name=thread_name, daemon=True)
        worker.start()
        while True:
            if signum := intent.signum:
                deadline = intent.deadline(STOP_TIMEOUT_SECONDS)
                assert deadline is not None
                try:
                    confirmed = bounded_stop(stop, deadline)
                except Exception:
                    logging.getLogger("milknado").exception(
                        "shutdown cleanup failed after signal %d", signum
                    )
                    confirmed = False
                intent.cleanup_confirmed = confirmed
                if not confirmed:
                    logging.getLogger("milknado").warning(
                        "shutdown cleanup remains unresolved after signal %d", signum
                    )
                raise ShutdownSignal(signum)
            try:
                outcome = outcomes.get(timeout=0.05)
            except Empty:
                continue
            while worker.is_alive() and not intent.requested:
                worker.join(timeout=0.05)
            if intent.requested:
                continue
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
