"""Tests for ralphify._output — ProcessResult, output collection, and formatting utilities."""

import pytest

from milknado.loop._output import (  # pyright: ignore[reportMissingTypeStubs]
    ProcessResult,
    collect_output,
    ensure_str,
    format_duration,
)


class TestProcessResult:
    def test_success_when_returncode_zero(self):
        assert ProcessResult(returncode=0).success is True

    def test_not_success_when_returncode_nonzero(self):
        assert ProcessResult(returncode=1).success is False
        assert ProcessResult(returncode=127).success is False
        assert ProcessResult(returncode=-1).success is False

    def test_not_success_when_timed_out(self):
        assert ProcessResult(returncode=None, timed_out=True).success is False

    def test_not_success_when_returncode_zero_but_timed_out(self):
        # Both conditions must hold: exit code 0 AND not timed out
        assert ProcessResult(returncode=0, timed_out=True).success is False

    def test_not_success_when_returncode_none(self):
        assert ProcessResult(returncode=None).success is False

    def test_timed_out_defaults_to_false(self):
        result = ProcessResult(returncode=0)
        assert result.timed_out is False


class TestEnsureStr:
    def test_str_passthrough(self):
        assert ensure_str("hello") == "hello"

    def test_bytes_decoded(self):
        assert ensure_str(b"hello\n") == "hello\n"

    def test_invalid_bytes_replaced(self):
        result = ensure_str(b"hello\xff")
        assert "hello" in result
        assert "\ufffd" in result

    def test_empty_str(self):
        assert ensure_str("") == ""

    def test_empty_bytes(self):
        assert ensure_str(b"") == ""


class TestCollectOutput:
    @pytest.mark.parametrize(
        "stdout, stderr, expected",
        [
            ("out\n", "err\n", "out\nerr\n"),
            ("out\n", "", "out\n"),
            ("", "err\n", "err\n"),
            (None, None, ""),
            (None, "err\n", "err\n"),
            ("out\n", None, "out\n"),
            ("", "", ""),
            (b"out\n", None, "out\n"),
            (None, b"err\n", "err\n"),
            (b"out\n", b"err\n", "out\nerr\n"),
            ("out\n", b"err\n", "out\nerr\n"),
        ],
        ids=[
            "both_strings",
            "stdout_only",
            "stderr_only",
            "both_none",
            "stdout_none",
            "stderr_none",
            "both_empty",
            "bytes_stdout",
            "bytes_stderr",
            "bytes_both",
            "mixed_str_and_bytes",
        ],
    )
    def test_collect_output(
        self, stdout: str | bytes | None, stderr: str | bytes | None, expected: str
    ):
        assert collect_output(stdout, stderr) == expected

    def test_stdout_without_trailing_newline_gets_separator(self):
        """When stdout doesn't end with a newline and stderr follows,
        a newline separator must be inserted to prevent garbled output."""
        assert collect_output("test passed", "warning: dep\n") == "test passed\nwarning: dep\n"

    def test_bytes_stdout_without_trailing_newline_gets_separator(self):
        """Same as above but with bytes input."""
        assert collect_output(b"test passed", b"warning\n") == "test passed\nwarning\n"

    def test_bytes_with_replacement(self):
        result = collect_output(b"hello\xff\n", None)
        assert "hello" in result
        assert "\ufffd" in result  # replacement character


class TestFormatDuration:
    def test_seconds(self):
        assert format_duration(5.3) == "5.3s"
        assert format_duration(0.1) == "0.1s"
        assert format_duration(59.9) == "59.9s"

    def test_minutes(self):
        assert format_duration(60) == "1m 0s"
        assert format_duration(90.5) == "1m 31s"
        assert format_duration(3599) == "59m 59s"

    def test_half_second_rounds_up_not_bankers(self):
        """Durations at exactly .5 seconds must always round UP (standard
        rounding) rather than using Python's banker's rounding which rounds
        to the nearest even integer — e.g. 90.5→90 loses half a second."""
        assert format_duration(60.5) == "1m 1s"
        assert format_duration(90.5) == "1m 31s"
        assert format_duration(150.5) == "2m 31s"
        # Odd totals already rounded up — verify they still do
        assert format_duration(61.5) == "1m 2s"
        assert format_duration(91.5) == "1m 32s"

    def test_zero(self):
        assert format_duration(0.0) == "0.0s"

    def test_boundary_at_60(self):
        assert format_duration(59.94) == "59.9s"
        assert format_duration(59.95) == "1m 0s"  # rounds to 60.0, so use minute format
        assert format_duration(60) == "1m 0s"
        assert format_duration(60.4) == "1m 0s"

    def test_hours(self):
        assert format_duration(3600) == "1h 0m"
        assert format_duration(5400) == "1h 30m"

    def test_multi_day(self):
        assert format_duration(90000) == "25h 0m"
