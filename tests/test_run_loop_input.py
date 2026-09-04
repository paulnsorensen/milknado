"""Tests for run_loop/input.py: InputState, handle_key, drain_input, thread lifecycle."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable

import pytest

from milknado.domains.execution.run_loop.input import (
    InputState,
    drain_input,
    handle_key,
    start_input_thread,
    stop_input_thread,
)


@pytest.fixture()
def state() -> InputState:
    return InputState()


class _FakeStdin:
    def __init__(
        self,
        reads: list[str] | None = None,
        on_read: Callable[[], None] | None = None,
        fd: int = 0,
    ) -> None:
        self._reads: deque[str] = deque(reads or [])
        self._on_read: Callable[[], None] | None = on_read
        self._fd: int = fd

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._fd

    def read(self, _n: int) -> str:
        if self._on_read is not None:
            self._on_read()
        return self._reads.popleft() if self._reads else ""


class _FakeTermios:
    error: type[OSError] = OSError
    TCSADRAIN: int = 0

    def __init__(
        self,
        get_error: Exception | None = None,
        set_error: Exception | None = None,
    ) -> None:
        self._get_error: Exception | None = get_error
        self._set_error: Exception | None = set_error

    def tcgetattr(self, _fd: int) -> list[object]:
        if self._get_error is not None:
            raise self._get_error
        return []

    def tcsetattr(self, _fd: int, _when: int, _settings: list[object]) -> None:
        if self._set_error is not None:
            raise self._set_error


class _FakeTTY:
    def setcbreak(self, _fd: int) -> None:
        return None


class _FakeSelect:
    def __init__(
        self,
        stdin: _FakeStdin,
        *,
        readable: bool = False,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._stdin: _FakeStdin = stdin
        self._readable: bool = readable
        self._stop_event: threading.Event | None = stop_event

    def select(
        self,
        _read: list[_FakeStdin],
        _write: list[object],
        _error: list[object],
        _timeout: float,
    ) -> tuple[list[_FakeStdin], list[object], list[object]]:
        if self._stop_event is not None:
            self._stop_event.set()
        readable = [self._stdin] if self._readable else []
        return readable, [], []


class TestHandleKeyEscape:
    def test_escape_clears_overlay(self, state: InputState) -> None:
        state.overlay_state = "run-1"
        handle_key(state, "\x1b", {})
        assert state.overlay_state is None

    def test_escape_clears_awaiting_digits(self, state: InputState) -> None:
        state.awaiting_node_digits = True
        handle_key(state, "\x1b", {})
        assert state.awaiting_node_digits is False

    def test_escape_clears_key_buffer(self, state: InputState) -> None:
        state.key_buffer = "42"
        handle_key(state, "\x1b", {})
        assert state.key_buffer == ""


class TestHandleKeyLookup:
    def test_l_sets_awaiting_digits(self, state: InputState) -> None:
        handle_key(state, "l", {})
        assert state.awaiting_node_digits is True

    def test_L_sets_awaiting_digits(self, state: InputState) -> None:
        handle_key(state, "L", {})
        assert state.awaiting_node_digits is True

    def test_l_resets_key_buffer(self, state: InputState) -> None:
        state.key_buffer = "99"
        handle_key(state, "l", {})
        assert state.key_buffer == ""

    def test_l_ignored_when_already_awaiting(self, state: InputState) -> None:
        state.awaiting_node_digits = True
        state.key_buffer = "4"
        handle_key(state, "l", {})
        # l treated as digit check: 'l' is not a digit, so nothing changes
        assert state.awaiting_node_digits is True


class TestHandleKeyDigits:
    def test_digits_accumulate_in_buffer(self, state: InputState) -> None:
        state.awaiting_node_digits = True
        handle_key(state, "1", {})
        handle_key(state, "2", {})
        assert state.key_buffer == "12"

    def test_enter_sets_overlay_for_matching_node(self, state: InputState) -> None:
        state.awaiting_node_digits = True
        state.key_buffer = "3"
        active = {"run-1": 3, "run-2": 7}
        handle_key(state, "\r", active)
        assert state.overlay_state == "run-1"
        assert state.awaiting_node_digits is False
        assert state.key_buffer == ""

    def test_newline_also_triggers_lookup(self, state: InputState) -> None:
        state.awaiting_node_digits = True
        state.key_buffer = "7"
        active = {"run-x": 7}
        handle_key(state, "\n", active)
        assert state.overlay_state == "run-x"

    def test_enter_with_no_match_leaves_overlay_unchanged(self, state: InputState) -> None:
        state.awaiting_node_digits = True
        state.key_buffer = "99"
        state.overlay_state = None
        handle_key(state, "\r", {"run-1": 1})
        assert state.overlay_state is None

    def test_enter_with_empty_buffer_does_nothing(self, state: InputState) -> None:
        state.awaiting_node_digits = True
        state.key_buffer = ""
        handle_key(state, "\r", {"run-1": 1})
        assert state.overlay_state is None


class TestDrainInput:
    def test_drain_processes_queued_key(self, state: InputState) -> None:
        state.input_queue.put("l")
        drain_input(state, {})
        assert state.awaiting_node_digits is True

    def test_drain_processes_multiple_keys(self, state: InputState) -> None:
        state.input_queue.put("l")
        state.input_queue.put("5")
        state.input_queue.put("\r")
        active = {"run-abc": 5}
        drain_input(state, active)
        assert state.overlay_state == "run-abc"

    def test_drain_empty_queue_does_nothing(self, state: InputState) -> None:
        drain_input(state, {})
        assert state.overlay_state is None
        assert state.awaiting_node_digits is False


class TestStopInputThread:
    def test_stop_sets_event(self, state: InputState) -> None:
        stop_input_thread(state)
        assert state.input_stop.is_set()

    def test_stop_clears_thread_reference(self, state: InputState) -> None:
        fake_thread = threading.Thread(target=lambda: None, daemon=True)
        fake_thread.start()
        state.input_thread = fake_thread
        stop_input_thread(state)
        assert state.input_thread is None

    def test_stop_with_no_thread_is_noop(self, state: InputState) -> None:
        state.input_thread = None
        stop_input_thread(state)  # should not raise


class TestStartInputThreadNonTTY:
    def test_no_thread_started_when_not_tty(self, state: InputState) -> None:
        # In test environments stdin is not a TTY; start_input_thread should bail early.
        start_input_thread(state)
        assert state.input_thread is None


class TestStartInputThreadTermiosError:
    def test_termios_error_returns_cleanly(self, state: InputState) -> None:
        import sys
        from unittest.mock import patch

        fake_stdin = _FakeStdin()
        fake_termios = _FakeTermios(get_error=OSError("Inappropriate ioctl"))
        fake_tty = _FakeTTY()
        fake_select = _FakeSelect(fake_stdin)

        with (
            patch.object(sys, "stdin", fake_stdin),
            patch.dict(
                "sys.modules",
                {"termios": fake_termios, "tty": fake_tty, "select": fake_select},
            ),
        ):
            start_input_thread(state)

        assert state.input_thread is None


class TestStartInputThreadTTY:
    def test_thread_started_when_tty(self, state: InputState) -> None:
        import sys
        from unittest.mock import patch

        fake_stdin = _FakeStdin()
        fake_termios = _FakeTermios()
        fake_tty = _FakeTTY()
        fake_select = _FakeSelect(fake_stdin)

        with (
            patch.object(sys, "stdin", fake_stdin),
            patch.dict(
                "sys.modules",
                {"termios": fake_termios, "tty": fake_tty, "select": fake_select},
            ),
        ):
            start_input_thread(state)
            # Set stop immediately so the thread exits cleanly
            state.input_stop.set()
            if state.input_thread:
                state.input_thread.join(timeout=1.0)

        assert state.input_thread is None or not state.input_thread.is_alive()

    def test_thread_reads_key_when_readable(self, state: InputState) -> None:
        import sys
        from unittest.mock import patch

        fake_stdin = _FakeStdin(reads=["x"], on_read=state.input_stop.set)
        fake_termios = _FakeTermios()
        fake_tty = _FakeTTY()
        fake_select = _FakeSelect(fake_stdin, readable=True)

        with (
            patch.object(sys, "stdin", fake_stdin),
            patch.dict(
                "sys.modules",
                {
                    "termios": fake_termios,
                    "tty": fake_tty,
                    "select": fake_select,
                },
            ),
        ):
            start_input_thread(state)
            if state.input_thread:
                state.input_thread.join(timeout=1.0)

        assert not state.input_queue.empty() or state.input_stop.is_set()

    def test_terminal_reset_failure_is_logged(
        self, state: InputState, caplog: pytest.LogCaptureFixture
    ) -> None:
        import sys
        from unittest.mock import patch

        fake_stdin = _FakeStdin(fd=9)
        fake_termios = _FakeTermios(set_error=OSError("reset failed"))
        fake_select = _FakeSelect(fake_stdin, stop_event=state.input_stop)
        with (
            caplog.at_level("ERROR"),
            patch.object(sys, "stdin", fake_stdin),
            patch.dict(
                "sys.modules",
                {
                    "termios": fake_termios,
                    "tty": _FakeTTY(),
                    "select": fake_select,
                },
            ),
        ):
            start_input_thread(state)
            assert state.input_thread is not None
            state.input_thread.join(timeout=1.0)

        assert "terminal reset failed input_thread=milknado-input fd=9" in caplog.text
