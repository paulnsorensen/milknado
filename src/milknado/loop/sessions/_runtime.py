from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from milknado.domains.common import SessionContext, SessionEvent, SessionInput
from milknado.loop._agent import (
    AgentResult,
    AgentRunSpec,
    _WindDownContext,  # pyright: ignore[reportPrivateUsage]
)
from milknado.loop._promise import has_promise_completion
from milknado.loop.sessions._channel import SessionChannel
from milknado.loop.sessions._factory import create_protocol
from milknado.loop.sessions._process import (
    POLL_INTERVAL,
    READER_QUEUE_LIMIT,
    BoundedTail,
    Line,
    cleanup_process,
    finish_process,
    log_path,
    prepare_wind_down,
    record_tool_count,
    start_process,
    start_readers,
    terminate,
    write_commands,
)
from milknado.loop.sessions._protocol import ProtocolStep, SessionProtocol
from milknado.loop.sessions._stream import StreamContext, consume, drain


@dataclass(slots=True)
class _SessionOutcome:
    done: bool = False
    failed: bool = False
    interrupted: bool = False
    result_text: str | None = None
    session_id: str | None = None
    tool_count: int = 0
    capped: bool = False
    timed_out: bool = False
    force_stopped: bool = False
    interrupt_requested: bool = False
    tool_ids: set[str] = field(default_factory=set)
    reader_failed: bool = False


def _publish_after_write_events(channel: SessionChannel, step: ProtocolStep) -> None:
    for event in step.after_write_events:
        channel.publish(event)


def _publish_rejected(channel: SessionChannel, command: SessionInput) -> None:
    text = command.text
    request_id = command.request_id
    channel.publish(SessionEvent(kind="user", text=text, event_id=request_id, state="rejected"))


@dataclass(slots=True)
class _SessionExecution:
    spec: AgentRunSpec
    channel: SessionChannel
    protocol: SessionProtocol
    start_step: ProtocolStep
    outcome: _SessionOutcome
    started_at: float
    lines: queue.Queue[Line]
    log_file: Path | None = None
    log_handle: IO[str] | None = None
    proc: subprocess.Popen[bytes] | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    threads: list[threading.Thread] = field(default_factory=list)
    eof_streams: set[str] = field(default_factory=set)
    stream_context: StreamContext | None = None
    wind_down: _WindDownContext | None = None

    def launch(self) -> None:
        self.wind_down = prepare_wind_down(self.spec)
        proc = start_process(
            self.protocol,
            self.spec.cwd or Path.cwd(),
            env=self.wind_down.env_overrides if self.wind_down is not None else None,
        )
        self.proc = proc
        self.threads = start_readers(proc, self.lines, self.stop, self.spec.iteration)
        self.stream_context = StreamContext(
            channel=self.channel,
            stdout_tail=BoundedTail(),
            stderr_tail=BoundedTail(),
            log_handle=self.log_handle,
            on_stdout=self.receive_line,
            on_output_line=self.spec.on_output_line,
            on_reader_error=self.mark_reader_error,
        )
        write_commands(proc, self.start_step.commands)
        _publish_after_write_events(self.channel, self.start_step)

    def remember_step(self, step: ProtocolStep) -> None:
        channel, outcome, wind_down = self.channel, self.outcome, self.wind_down
        for event in step.events:
            channel.publish(event)
            if event.kind == "tool":
                key = event.event_id or f"tool:{outcome.tool_count}:{event.text}"
                if key not in outcome.tool_ids:
                    outcome.tool_ids.add(key)
                    outcome.tool_count += 1
                    record_tool_count(wind_down, outcome.tool_count)
            outcome.interrupted = outcome.interrupted or event.state in {"interrupted", "aborted"}
        if step.result_text is not None:
            outcome.result_text = step.result_text
        if step.session_id is not None:
            outcome.session_id = step.session_id
        outcome.done = outcome.done or step.done
        outcome.failed = outcome.failed or step.failed
        outcome.interrupted = outcome.interrupted or step.interrupted
        context = channel.view().context
        if context is not None:
            channel.start(context, tuple(self.protocol.actions))

    def apply_step(self, step: ProtocolStep) -> None:
        assert self.proc is not None
        self.remember_step(step)
        write_commands(self.proc, step.commands)
        _publish_after_write_events(self.channel, step)

    def receive_line(self, text: str) -> None:
        self.apply_step(self.protocol.receive(text.encode("utf-8")))

    def mark_reader_error(self, text: str) -> None:
        self.outcome.failed = True
        self.outcome.reader_failed = True
        self.channel.publish(SessionEvent(kind="error", text=text, event_id="reader", delta=True))

    def submit_inputs(self) -> bool:
        commands = self.channel.drain()
        if not commands:
            return False
        for command in commands:
            was_done = self.outcome.done
            if command.action == "interrupt":
                self.outcome.interrupt_requested = True
            try:
                self.outcome.done = False
                self.apply_step(self.protocol.submit(command))
            except ValueError:
                self.outcome.done = was_done
                _publish_rejected(self.channel, command)
            except BrokenPipeError:
                self.outcome.done = was_done
                self.outcome.failed = True
                _publish_rejected(self.channel, command)
        return True

    def run(self) -> None:
        context = self.stream_context
        assert context is not None
        assert self.proc is not None
        deadline = self.started_at + self.spec.timeout if self.spec.timeout is not None else None
        force_stop = self.spec.force_stop_event
        while True:
            if force_stop is not None and force_stop.is_set():
                self.outcome.force_stopped = True
                break
            if deadline is not None and time.monotonic() >= deadline:
                self.outcome.timed_out = True
                break
            submitted = self.submit_inputs()
            if self.proc.poll() is not None:
                terminate(self.proc)
            if self.outcome.done and not submitted and self.lines.empty():
                time.sleep(POLL_INTERVAL)
                if self.submit_inputs():
                    continue
                break
            if self.proc.poll() is not None and len(self.eof_streams) == 2 and self.lines.empty():
                break
            wait = (
                min(POLL_INTERVAL, max(0.0, deadline - time.monotonic()))
                if deadline is not None
                else POLL_INTERVAL
            )
            try:
                item = self.lines.get(timeout=wait)
            except queue.Empty:
                continue
            if item.text is None:
                self.eof_streams.add(item.stream)
            else:
                consume(item, context)
            if self.outcome.reader_failed:
                break
            if self.spec.max_turns is not None and self.outcome.tool_count >= self.spec.max_turns:
                self.outcome.capped = True
                break
        assert self.proc is not None
        finish_process(
            self.proc,
            graceful=not (
                self.outcome.timed_out
                or self.outcome.force_stopped
                or not self.outcome.done
                or self.outcome.capped
            ),
        )
        drain(self.lines, self.eof_streams, context)

    def result(self) -> AgentResult:
        context = self.stream_context
        assert context is not None
        assert self.proc is not None
        returncode = self.proc.poll()
        if self.outcome.force_stopped or self.outcome.timed_out:
            returncode = None
        elif self.outcome.failed or (not self.outcome.done and not self.outcome.capped):
            returncode = returncode if returncode not in (None, 0) else 1
        elif returncode is None or returncode < 0:
            returncode = 0
        signal = self.spec.completion_signal
        completion_detected = bool(
            signal and has_promise_completion(self.outcome.result_text, signal)
        )
        return AgentResult(
            returncode=returncode,
            timed_out=self.outcome.timed_out,
            elapsed=time.monotonic() - self.started_at,
            log_file=self.log_file,
            session_id=self.outcome.session_id,
            result_text=self.outcome.result_text,
            captured_stdout=context.stdout_tail.text,
            completion_detected=completion_detected,
            captured_stderr=context.stderr_tail.text,
            force_stopped=self.outcome.force_stopped,
            interrupted=self.outcome.interrupted and self.outcome.interrupt_requested,
            tool_use_count=self.outcome.tool_count,
            turn_capped=self.outcome.capped,
        )

    def cleanup(self) -> None:
        if self.proc is not None:
            cleanup_process(self.proc, self.stop, tuple(self.threads))
        if self.log_handle is not None:
            self.log_handle.close()
        if self.wind_down is not None:
            self.wind_down.cleanup()


def _new_execution(spec: AgentRunSpec, channel: SessionChannel) -> _SessionExecution:
    cwd = spec.cwd or Path.cwd()
    protocol = create_protocol(tuple(spec.cmd), cwd)
    if protocol is None:
        raise ValueError(f"unsupported structured session command: {spec.cmd!r}")
    context = channel.view().context or SessionContext(family=Path(spec.cmd[0]).stem, cwd=str(cwd))
    channel.start(context, tuple(protocol.actions))
    started_at = time.monotonic()
    start_step = protocol.start(spec.prompt)
    execution = _SessionExecution(
        spec=spec,
        channel=channel,
        protocol=protocol,
        start_step=start_step,
        outcome=_SessionOutcome(),
        started_at=started_at,
        lines=queue.Queue(maxsize=READER_QUEUE_LIMIT),
    )
    execution.remember_step(start_step)
    if spec.log_dir is not None:
        execution.log_file = log_path(spec.log_dir, spec.iteration)
        execution.log_handle = execution.log_file.open("w", encoding="utf-8")
    return execution


def run_session(spec: AgentRunSpec, channel: SessionChannel) -> AgentResult:
    execution: _SessionExecution | None = None
    try:
        execution = _new_execution(spec, channel)
        execution.launch()
        execution.run()
        return execution.result()
    finally:
        if execution is not None:
            execution.cleanup()
        channel.close()
