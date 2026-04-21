from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field


@dataclass
class InputState:
    overlay_state: str | None = None
    awaiting_node_digits: bool = False
    key_buffer: str = ""
    input_queue: queue.Queue[str] = field(default_factory=queue.Queue)
    input_stop: threading.Event = field(default_factory=threading.Event)
    input_thread: threading.Thread | None = None


def start_input_thread(state: InputState) -> None:
    import sys

    if not sys.stdin.isatty():
        return
    try:
        import select
        import termios
        import tty
    except ImportError:
        return
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except (termios.error, OSError, ValueError):
        return
    stop = state.input_stop

    def _read_keys() -> None:
        try:
            while not stop.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if readable:
                    ch = sys.stdin.read(1)
                    if ch:
                        state.input_queue.put(ch)
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:  # noqa: BLE001
                pass

    state.input_thread = threading.Thread(
        target=_read_keys, daemon=True, name="milknado-input"
    )
    state.input_thread.start()


def stop_input_thread(state: InputState) -> None:
    state.input_stop.set()
    if state.input_thread is not None:
        state.input_thread.join(timeout=0.5)
        state.input_thread = None


def handle_key(state: InputState, key: str, active: dict[str, int]) -> None:
    if key == "\x1b":
        state.overlay_state = None
        state.awaiting_node_digits = False
        state.key_buffer = ""
        return
    if key in ("l", "L") and not state.awaiting_node_digits:
        state.awaiting_node_digits = True
        state.key_buffer = ""
        return
    if state.awaiting_node_digits:
        if key.isdigit():
            state.key_buffer += key
        elif key in ("\r", "\n"):
            state.awaiting_node_digits = False
            if state.key_buffer:
                target = int(state.key_buffer)
                for run_id, node_id in active.items():
                    if node_id == target:
                        state.overlay_state = run_id
                        break
            state.key_buffer = ""


def drain_input(state: InputState, active: dict[str, int]) -> None:
    try:
        while True:
            handle_key(state, state.input_queue.get_nowait(), active)
    except queue.Empty:
        pass
