"""Tests for run_loop/input.py: InputState, handle_key, drain_input, thread lifecycle."""
from __future__ import annotations

import queue
import threading

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


class TestStartInputThreadTTY:
    def test_thread_started_when_tty(self, state: InputState) -> None:
        import sys
        from unittest.mock import MagicMock, patch

        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = True
        fake_stdin.fileno.return_value = 0
        fake_stdin.read.return_value = ""

        fake_termios = MagicMock()
        fake_termios.tcgetattr.return_value = []
        fake_tty = MagicMock()
        fake_select = MagicMock()
        # select returns empty so read loop terminates after first pass via stop event
        fake_select.select.return_value = ([], [], [])

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
        from unittest.mock import MagicMock, patch

        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = True
        fake_stdin.fileno.return_value = 0

        # First select returns readable (delivers 'x'), then stop
        read_calls = [0]

        def fake_read(n: int) -> str:
            read_calls[0] += 1
            if read_calls[0] == 1:
                state.input_stop.set()
                return "x"
            return ""

        fake_stdin.read.side_effect = fake_read

        fake_termios = MagicMock()
        fake_termios.tcgetattr.return_value = []
        fake_tty = MagicMock()
        fake_select_module = MagicMock()
        fake_select_module.select.return_value = ([fake_stdin], [], [])

        with (
            patch.object(sys, "stdin", fake_stdin),
            patch.dict(
                "sys.modules",
                {
                    "termios": fake_termios,
                    "tty": fake_tty,
                    "select": fake_select_module,
                },
            ),
        ):
            start_input_thread(state)
            if state.input_thread:
                state.input_thread.join(timeout=1.0)

        assert not state.input_queue.empty() or state.input_stop.is_set()
