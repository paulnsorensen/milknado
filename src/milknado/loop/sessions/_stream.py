from __future__ import annotations

import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO

from milknado.domains.common import SessionEvent, redact_control_text
from milknado.loop._events import OutputStream
from milknado.loop._output import BoundedOutput
from milknado.loop.sessions._channel import SessionChannel
from milknado.loop.sessions._process import CAPTURE_LIMIT, POLL_INTERVAL, TERMINATE_GRACE, Line
from milknado.loop.sessions._protocol import ProtocolStep

_FAILURE_SUMMARY_LIMIT = 1024


@dataclass(slots=True)
class StreamContext:
    channel: SessionChannel
    stdout_tail: BoundedOutput
    stderr_tail: BoundedOutput
    log_handle: IO[str] | None
    on_stdout: Callable[[str], None]
    on_output_line: Callable[[str, OutputStream], None] | None
    on_reader_error: Callable[[str], None]
    sanitize: bool = False
    logged_chars: int = 0
    failure_captured: bool = False


def _safe_stderr(text: str) -> str:
    lower = text[:CAPTURE_LIMIT].lower()
    if "authentication" in lower or "unauthorized" in lower:
        return "provider authentication failed\n"
    if "invalid option" in lower or "unknown option" in lower:
        return "provider invalid option\n"
    if "permission denied" in lower:
        return "provider permission denied\n"
    return "provider stderr frame\n"


def _write_log(context: StreamContext, diagnostic: str, limit: int) -> None:
    if context.log_handle is None:
        return
    remaining = limit - context.logged_chars
    if remaining > 0:
        written = diagnostic[:remaining]
        _ = context.log_handle.write(written)
        _ = context.log_handle.flush()
        context.logged_chars += len(written)


def capture_failure(context: StreamContext | None, step: ProtocolStep) -> None:
    if context is None or not context.sanitize or context.failure_captured or not step.failed:
        return
    text = next(
        (event.text for event in step.events if event.kind == "error" and event.text.strip()),
        None,
    )
    if text is None:
        return
    redacted = " ".join(redact_control_text(text).splitlines())
    summary = f"provider failure: {redacted}"[: _FAILURE_SUMMARY_LIMIT - 1] + "\n"
    context.stdout_tail.append(summary)
    _write_log(context, summary, CAPTURE_LIMIT)
    context.failure_captured = True


def consume(item: Line, context: StreamContext) -> None:
    if item.text is None:
        return
    if item.stream == "reader_error":
        context.stderr_tail.append(item.text + "\n")
        context.on_reader_error(item.text)
        return
    diagnostic = item.text
    if context.sanitize:
        if item.stream == "stderr":
            diagnostic = _safe_stderr(item.text)
        else:
            diagnostic = "provider stdout frame\n"
    tail = context.stdout_tail if item.stream == "stdout" else context.stderr_tail
    tail.append(diagnostic)
    limit = (
        CAPTURE_LIMIT - _FAILURE_SUMMARY_LIMIT
        if context.sanitize
        else context.logged_chars + len(diagnostic)
    )
    _write_log(context, diagnostic, limit)
    if item.stream == "stderr":
        if context.on_output_line is not None:
            context.on_output_line(diagnostic, "stderr")
        context.channel.publish(
            SessionEvent(
                kind="error",
                text=diagnostic.rstrip("\r\n"),
                event_id="stderr",
                delta=True,
            )
        )
        return
    if context.on_output_line is not None:
        context.on_output_line(diagnostic, "stdout")
    context.on_stdout(item.text)


def drain(
    lines: queue.Queue[Line],
    eof_streams: set[str],
    context: StreamContext,
) -> None:
    deadline = time.monotonic() + TERMINATE_GRACE
    while len(eof_streams) < 2 and time.monotonic() < deadline:
        try:
            item = lines.get(timeout=POLL_INTERVAL)
        except queue.Empty:
            continue
        if item.text is None:
            eof_streams.add(item.stream)
        else:
            consume(item, context)
