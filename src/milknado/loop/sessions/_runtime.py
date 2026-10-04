from __future__ import annotations

import queue
import subprocess
import threading
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, cast

from milknado.domains.common import SessionContext, SessionEvent
from milknado.loop._agent import (
    AgentResult,
    AgentRunSpec,
    _build_spawn_env,  # pyright: ignore[reportPrivateUsage]
    _WindDownContext,  # pyright: ignore[reportPrivateUsage]
)
from milknado.loop._output import BoundedOutput
from milknado.loop._process_contract import WorkerHandle
from milknado.loop._process_gate import SpawnOptions
from milknado.loop._promise import has_promise_completion
from milknado.loop.sessions._channel import SessionChannel
from milknado.loop.sessions._factory import create_protocol
from milknado.loop.sessions._outcome import SessionOutcome
from milknado.loop.sessions._outcome import publish_events as _publish_events
from milknado.loop.sessions._process import (
    CAPTURE_LIMIT,
    POLL_INTERVAL,
    READER_QUEUE_LIMIT,
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
from milknado.loop.sessions._protocol import (
    ProtocolStep,
    ProviderFamily,
    ProviderSessionIdentity,
    SessionProtocol,
)
from milknado.loop.sessions._stream import StreamContext, consume, drain


@dataclass(slots=True)
class _SessionExecution:
    spec: AgentRunSpec
    channel: SessionChannel
    protocol: SessionProtocol
    process_invocation_id: str
    start_step: ProtocolStep
    outcome: SessionOutcome
    started_at: float
    lines: queue.Queue[Line]
    log_file: Path | None = None
    log_handle: IO[str] | None = None
    proc: subprocess.Popen[bytes] | None = None
    protected: WorkerHandle | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    threads: list[threading.Thread] = field(default_factory=list)
    eof_streams: set[str] = field(default_factory=set)
    stream_context: StreamContext | None = None
    wind_down: _WindDownContext | None = None

    def launch(self) -> None:
        self.wind_down = prepare_wind_down(self.spec)
        env = {
            **(self.spec.env or {}),
            **(self.wind_down.env_overrides if self.wind_down is not None else {}),
            "MILKNADO_INVOCATION_ID": self.process_invocation_id,
        }
        cwd = self.spec.cwd or Path.cwd()
        if self.spec.spawn_worker is not None:
            self.protected = self.spec.spawn_worker(
                SpawnOptions(
                    self.protocol.command,
                    cwd,
                    _build_spawn_env(env),
                    False,
                    subprocess.PIPE,
                    subprocess.PIPE,
                    subprocess.PIPE,
                    self.process_invocation_id,
                )
            )
            proc = cast(subprocess.Popen[bytes], self.protected.process)
        else:
            proc = start_process(self.protocol, cwd, env=env)
        self.proc = proc
        self.threads = start_readers(proc, self.lines, self.stop, self.spec.iteration)
        self.stream_context = StreamContext(
            channel=self.channel,
            stdout_tail=BoundedOutput(CAPTURE_LIMIT),
            stderr_tail=BoundedOutput(CAPTURE_LIMIT),
            log_handle=self.log_handle,
            on_stdout=self.receive_line,
            on_output_line=self.spec.on_output_line,
            on_reader_error=self.mark_reader_error,
        )
        write_commands(proc, self.start_step.commands)
        _publish_events(self.channel, self.start_step.after_write_events)

    def remember_step(self, step: ProtocolStep, *, publish_events: bool = True) -> None:
        channel, outcome, wind_down = self.channel, self.outcome, self.wind_down
        actions = tuple(self.protocol.actions)
        if step.session_id is not None:
            family = channel.view().context.family if channel.view().context else ""
            if family in {"claude", "codex"}:
                identity = ProviderSessionIdentity.from_step(cast(ProviderFamily, family), step)
                if identity is not None:
                    outcome.session_id = identity.session_id
            else:
                outcome.session_id = step.session_id
        if (context := channel.view().context) is not None and step.done:
            channel.start(context, actions, invocation_id=self.process_invocation_id)
        for event in step.events:
            if publish_events:
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
        outcome.done, outcome.failed = outcome.done or step.done, outcome.failed or step.failed
        outcome.interrupted = outcome.interrupted or step.interrupted
        if (context := channel.view().context) is not None:
            channel.start(context, actions, invocation_id=self.process_invocation_id)

    def apply_step(self, step: ProtocolStep) -> None:
        assert self.proc is not None
        self.remember_step(step)
        write_commands(self.proc, step.commands)
        _publish_events(self.channel, step.after_write_events)

    def receive_line(self, text: str) -> None:
        self.apply_step(self.protocol.receive(text.encode("utf-8")))

    def mark_reader_error(self, text: str) -> None:
        self.outcome.failed, self.outcome.reader_failed = True, True
        self.channel.publish(SessionEvent(kind="error", text=text, event_id="reader", delta=True))

    def submit_inputs(self) -> bool:
        assert self.proc is not None
        commands = self.channel.drain()
        if not commands:
            return False
        for command in commands:
            was_done = self.outcome.done
            if command.action == "interrupt":
                self.outcome.interrupt_requested = True
            try:
                self.outcome.done = False
                step = self.protocol.submit(command)
                _publish_events(self.channel, step.events)
                write_commands(self.proc, step.commands)
                if command.action in {"approve", "deny"} and step.after_write_events:
                    self.channel.confirm(command, "delivered")
                _publish_events(self.channel, step.after_write_events)
                self.remember_step(step, publish_events=False)
            except ValueError:
                self.outcome.done = was_done
                self.channel.confirm(command, "rejected")
            except BrokenPipeError:
                self.outcome.done = was_done
                self.outcome.failed = True
                self.channel.confirm(command, "unconfirmed")
        return True

    def run(self) -> None:
        context = self.stream_context
        assert context is not None
        assert self.proc is not None
        deadline = self.started_at + self.spec.timeout if self.spec.timeout is not None else None
        while True:
            if self.spec.force_stop_event is not None and self.spec.force_stop_event.is_set():
                self.outcome.force_stopped = True
                break
            if deadline is not None and time.monotonic() >= deadline:
                self.outcome.timed_out = True
                break
            submitted = self.submit_inputs()
            if self.proc.poll() is not None and self.protected is None:
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
        self._finish(context)

    def _finish(self, context: StreamContext) -> None:
        assert self.proc is not None
        if self.protected is not None:
            if not self.protected.complete(graceful=self.outcome.graceful):
                raise RuntimeError("worker cleanup remains unresolved")
        else:
            finish_process(self.proc, graceful=self.outcome.graceful)
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
        return AgentResult(
            returncode=returncode,
            timed_out=self.outcome.timed_out,
            elapsed=time.monotonic() - self.started_at,
            log_file=self.log_file,
            session_id=self.outcome.session_id,
            result_text=self.outcome.result_text,
            captured_stdout=context.stdout_tail.text,
            completion_detected=bool(
                signal and has_promise_completion(self.outcome.result_text, signal)
            ),
            captured_stderr=context.stderr_tail.text,
            force_stopped=self.outcome.force_stopped,
            interrupted=self.outcome.interrupted and self.outcome.interrupt_requested,
            tool_use_count=self.outcome.tool_count,
            turn_capped=self.outcome.capped,
        )

    def cleanup(self) -> None:
        with ExitStack() as cleanup:
            if self.wind_down is not None:
                _ = cleanup.callback(self.wind_down.cleanup)
            if self.log_handle is not None:
                _ = cleanup.callback(self.log_handle.close)
            if self.protected is not None:
                if not self.protected.cleanup(tuple(self.threads), stop=self.stop):
                    raise RuntimeError("worker cleanup remains unresolved")
            elif self.proc is not None:
                cleanup_process(self.proc, self.stop, tuple(self.threads))


def _new_execution(spec: AgentRunSpec, channel: SessionChannel) -> _SessionExecution:
    cwd = spec.cwd or Path.cwd()
    protocol = create_protocol(tuple(spec.cmd), cwd)
    if protocol is None:
        raise ValueError(f"unsupported structured session command: {spec.cmd!r}")
    context = channel.view().context or SessionContext(family=Path(spec.cmd[0]).stem, cwd=str(cwd))
    process_invocation_id = uuid.uuid4().hex
    start_step = protocol.start(spec.prompt)
    channel.start(context, tuple(protocol.actions), invocation_id=process_invocation_id)
    execution = _SessionExecution(
        spec=spec,
        channel=channel,
        protocol=protocol,
        process_invocation_id=process_invocation_id,
        start_step=start_step,
        outcome=SessionOutcome(),
        started_at=time.monotonic(),
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
