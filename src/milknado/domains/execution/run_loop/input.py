from __future__ import annotations

import queue
import threading
from collections.abc import Callable


class InputController:
    """Raw-stdin key capture for TUI drill-in overlay.

    No-ops on non-tty or non-Unix environments. Key semantics:
      * Esc       → clear overlay
      * L<digits>↵ → open overlay for node with matching id
    """

    def __init__(
        self,
        on_overlay: Callable[[int], None],
        on_clear: Callable[[], None],
    ) -> None:
        self._on_overlay = on_overlay
        self._on_clear = on_clear
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._awaiting_digits: bool = False
        self._key_buffer: str = ""

    def start(self) -> None:
        import sys

        if not sys.stdin.isatty():
            return
        try:
            import select
            import termios
            import tty
        except ImportError:
            return
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        stop = self._stop

        def _read_keys() -> None:
            try:
                while not stop.is_set():
                    readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if readable:
                        ch = sys.stdin.read(1)
                        if ch:
                            self._queue.put(ch)
            finally:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:  # noqa: BLE001
                    pass

        self._thread = threading.Thread(
            target=_read_keys, daemon=True, name="milknado-input",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    def drain(self) -> None:
        try:
            while True:
                self._handle(self._queue.get_nowait())
        except queue.Empty:
            pass

    def inject(self, key: str) -> None:
        # Test seam — bypass the OS thread, drive the state machine directly.
        self._handle(key)

    def _handle(self, key: str) -> None:
        if key == "\x1b":
            self._awaiting_digits = False
            self._key_buffer = ""
            self._on_clear()
            return
        if key in ("l", "L") and not self._awaiting_digits:
            self._awaiting_digits = True
            self._key_buffer = ""
            return
        if self._awaiting_digits:
            if key.isdigit():
                self._key_buffer += key
            elif key in ("\r", "\n"):
                self._awaiting_digits = False
                if self._key_buffer:
                    self._on_overlay(int(self._key_buffer))
                self._key_buffer = ""
