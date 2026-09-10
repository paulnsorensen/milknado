from __future__ import annotations

import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO

from milknado.domains.common import SessionEvent
from milknado.loop._events import OutputStream
from milknado.loop.sessions._channel import SessionChannel
from milknado.loop.sessions._process import POLL_INTERVAL, TERMINATE_GRACE, BoundedTail, Line


@dataclass(slots=True)
class StreamContext:
    channel: SessionChannel
    stdout_tail: BoundedTail
    stderr_tail: BoundedTail
    log_handle: IO[str] | None
    on_stdout: Callable[[str], None]
    on_output_line: Callable[[str, OutputStream], None] | None
    on_reader_error: Callable[[str], None]


def consume(item: Line, context: StreamContext) -> None:
    if item.text is None:
        return
    if item.stream == "reader_error":
        context.stderr_tail.append(item.text + "\n")
        context.on_reader_error(item.text)
        return
    tail = context.stdout_tail if item.stream == "stdout" else context.stderr_tail
    tail.append(item.text)
    if context.log_handle is not None:
        _ = context.log_handle.write(item.text)
        _ = context.log_handle.flush()
    if item.stream == "stderr":
        if context.on_output_line is not None:
            context.on_output_line(item.text, "stderr")
        context.channel.publish(
            SessionEvent(
                kind="error",
                text=item.text.rstrip("\r\n"),
                event_id="stderr",
                delta=True,
            )
        )
        return
    if context.on_output_line is not None:
        context.on_output_line(item.text, "stdout")
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
