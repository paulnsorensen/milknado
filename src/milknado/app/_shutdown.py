"""Record entry-point shutdown intent without interrupting application state."""

from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from time import monotonic
from types import FrameType


STOP_TIMEOUT_SECONDS = 8.0


class ShutdownSignal(RuntimeError):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"received signal {signum}")


class ShutdownIntent:
    def __init__(self) -> None:
        self.signum: int | None = None
        self.started_at: float | None = None

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

    @contextmanager
    def installed(self) -> Iterator[None]:
        signals = [signal.SIGINT, signal.SIGTERM]
        if hangup := getattr(signal, "SIGHUP", None):
            signals.append(hangup)
        previous = {signum: signal.getsignal(signum) for signum in signals}
        try:
            for signum in signals:
                signal.signal(signum, self.record)
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
