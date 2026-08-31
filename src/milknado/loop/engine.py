"""Run loop with structured event emission.

The core ``run_loop`` function is the autonomous agent loop.  It accepts
a ``RunConfig``, ``RunState``, and ``EventEmitter``, making it reusable
from both CLI and UI contexts.

The loop: run commands → assemble prompt → pipe to agent → repeat.
"""

from __future__ import annotations

import shlex
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, cast

from milknado.domains.common.agent_argv import validate_worker_argv
from milknado.loop._agent import (
    ActivityCallback,
    AgentResult,
    AgentRunSpec,
    OutputLineCallback,
    ToolUseCallback,
    execute_agent,
)
from milknado.loop._events import (
    AgentActivityData,
    BoundEmitter,
    CommandsCompletedData,
    CommandsStartedData,
    EventEmitter,
    EventType,
    IterationEndedData,
    IterationStartedData,
    NullEmitter,
    OutputStream,
    PromptAssembledData,
    RunStartedData,
    RunStoppedData,
    ToolUseData,
    TurnApproachingLimitData,
    TurnCappedData,
)
from milknado.loop._frontmatter import (
    FIELD_AGENT,
    FIELD_COMMANDS,
    RALPH_MARKER,
    parse_frontmatter,
)
from milknado.loop._output import format_duration
from milknado.loop._resolver import resolve_all, resolve_args
from milknado.loop._run_types import (
    Command,
    RunConfig,
    RunState,
    RunStatus,
)
from milknado.loop._runner import run_command
from milknado.loop.adapters import CLIAdapter, select_adapter
from milknado.loop.hooks import CombinedAgentHook

_PAUSE_POLL_INTERVAL = 0.25  # seconds between pause/resume checks
_RELATIVE_CMD_PREFIX = "./"  # commands starting with this run from the ralph directory


def _field_hint(field_name: str) -> str:
    return f"Check the '{field_name}' field in your {RALPH_MARKER} frontmatter."


def _footer_instruction(footer: str) -> str:
    return (
        "\n\n---\n\n"
        "IMPORTANT: When you make any git commits, include the following trailer "
        f"at the end of your commit message:\n\n{footer}"
    )


_VERIFIER_FEEDBACK_HEADER = (
    "\n\n---\n\n"
    "The previous iteration reported completion, but the completion check "
    "rejected it. Address this feedback before reporting completion again:\n\n"
)
_GUIDANCE_HEADER = "\n\n## Operator guidance\n\n"


def _wait_for_resume(state: RunState, emit: BoundEmitter) -> bool:
    """Block until the run is resumed or a stop is requested."""
    emit(EventType.RUN_PAUSED)
    while not state.wait_for_unpause(timeout=_PAUSE_POLL_INTERVAL):
        if state.stop_requested:
            break
    if state.stop_requested:
        state.status = RunStatus.STOPPED
        return False
    emit(EventType.RUN_RESUMED)
    return True


def _handle_control_signals(state: RunState, emit: BoundEmitter) -> bool:
    """Handle stop and pause requests at the top of each iteration."""
    if state.stop_requested:
        state.status = RunStatus.STOPPED
        return False
    if state.paused:
        if not _wait_for_resume(state, emit):
            return False
    return True


def _run_commands(
    commands: list[Command],
    ralph_dir: Path,
    project_root: Path,
    user_args: dict[str, str],
) -> dict[str, str]:
    """Execute all commands and return a dict of name→output.

    Commands with paths starting with ``./`` run relative to the ralph
    directory.  Other commands run from the project root.
    """
    results: dict[str, str] = {}
    quoted_args = {k: shlex.quote(v) for k, v in user_args.items()}
    for cmd in commands:
        run_str = resolve_args(cmd.run, quoted_args)
        if run_str.lstrip().startswith(_RELATIVE_CMD_PREFIX):
            cwd = ralph_dir
        else:
            cwd = project_root
        try:
            result = run_command(
                command=run_str,
                cwd=cwd,
                timeout=cmd.timeout,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Command '{cmd.name}' binary not found: {run_str!r}. "
                + f"{_field_hint(FIELD_COMMANDS)}"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"Command '{cmd.name}' has invalid syntax: {run_str!r}. "
                + f"{_field_hint(FIELD_COMMANDS)}"
            ) from exc
        output = result.output
        if result.timed_out:
            output += (
                f"\n\n[Command '{cmd.name}' timed out after "
                f"{format_duration(cmd.timeout)} — output may be incomplete]"
            )
        results[cmd.name] = output
    return results


def _build_ralph_context(config: RunConfig, state: RunState) -> dict[str, str]:
    """Build the context dict for ``{{ ralph.X }}`` placeholders."""
    ctx: dict[str, str] = {
        "name": config.ralph_dir.name,
        "iteration": str(state.iteration),
    }
    if config.max_iterations is not None:
        ctx["max_iterations"] = str(config.max_iterations)
    return ctx


def _assemble_prompt(
    config: RunConfig,
    state: RunState,
    command_outputs: dict[str, str],
    verifier_feedback: str | None = None,
    guidance: tuple[str, ...] = (),
) -> str:
    """Build the full prompt for one iteration.

    Uses ``config.prompt`` as the body when set (no file read, no
    frontmatter parse); otherwise reads the RALPH.md body.  Either way it
    resolves user args, command output, and context placeholders.

    ``verifier_feedback`` carries a rejecting ``completion_verifier``'s
    message from the prior iteration; when present it is appended verbatim
    so the agent can act on it this iteration.
    """
    if config.prompt is not None:
        prompt = config.prompt
    else:
        assert config.ralph_file is not None  # __post_init__ guarantees one is set
        raw = config.ralph_file.read_text(encoding="utf-8")
        _, prompt = parse_frontmatter(raw)
    ralph_context = _build_ralph_context(config, state)
    prompt = resolve_all(prompt, command_outputs, config.args, ralph_context)
    if verifier_feedback:
        prompt += _VERIFIER_FEEDBACK_HEADER + verifier_feedback
    if guidance:
        prompt += _GUIDANCE_HEADER + "\n\n".join(guidance)
    if config.commit_footer:
        prompt += _footer_instruction(config.commit_footer)
    return prompt


class _AgentCallbacks(NamedTuple):
    on_output_line: OutputLineCallback | None
    on_tool_use: ToolUseCallback | None
    on_activity: ActivityCallback | None


def _resolve_agent_command(config: RunConfig) -> tuple[list[str], CLIAdapter]:
    """Parse and validate the worker command before selecting its adapter."""
    try:
        cmd = shlex.split(config.agent)
    except ValueError as exc:
        raise ValueError(
            f"Invalid agent command syntax: {config.agent!r}. {_field_hint(FIELD_AGENT)}"
        ) from exc
    validate_worker_argv(cmd)
    return cmd, select_adapter(cmd)


def _build_agent_callbacks(
    config: RunConfig,
    state: RunState,
    emit: BoundEmitter,
    hooks: CombinedAgentHook | None,
) -> _AgentCallbacks:
    """Build the on_output_line, on_tool_use, and on_activity callbacks for one iteration."""

    def _on_output_line(line: str, stream: OutputStream) -> None:
        if emit.wants_agent_output_lines():
            emit.agent_output_line(line, stream, state.iteration)

    on_output_line = (
        _on_output_line if emit.wants_agent_output_lines() or config.log_dir is not None else None
    )

    on_tool_use = _build_tool_use_bridge(
        state=state,
        emit=emit,
        hooks=hooks,
        max_turns=config.max_turns,
        max_turns_grace=config.max_turns_grace,
    )

    def on_activity(data: dict[str, object]) -> None:
        emit(EventType.AGENT_ACTIVITY, AgentActivityData(raw=data, iteration=state.iteration))

    return _AgentCallbacks(on_output_line, on_tool_use, on_activity)


def _launch_agent(
    cmd: list[str],
    adapter: CLIAdapter,
    prompt: str,
    config: RunConfig,
    state: RunState,
    callbacks: _AgentCallbacks,
) -> AgentResult:
    # Capture full stdout only when somebody downstream actually needs the
    # bytes — log writing, or promise detection for adapters that cannot
    # work from ``agent.result_text`` alone.  Without this gate every
    # iteration would buffer the entire transcript even for verbose
    # streaming agents, regressing memory vs the prior tail-scan path.
    capture_stdout_for_promise = (
        config.stop_on_completion_signal and adapter.requires_full_stdout_for_completion
    )
    capture_stdout = config.log_dir is not None or capture_stdout_for_promise

    try:
        return execute_agent(
            AgentRunSpec(
                cmd,
                prompt,
                timeout=config.timeout,
                log_dir=config.log_dir,
                iteration=state.iteration,
                adapter=adapter,
                on_activity=callbacks.on_activity,
                on_output_line=callbacks.on_output_line,
                capture_result_text=True,
                capture_stdout=capture_stdout,
                completion_signal=(
                    config.completion_signal if config.stop_on_completion_signal else None
                ),
                max_turns=config.max_turns,
                max_turns_grace=config.max_turns_grace,
                on_tool_use=callbacks.on_tool_use,
                force_stop_event=state.force_stop_event,
                cwd=config.project_root,
            )
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Agent command not found: {config.agent!r}. {_field_hint(FIELD_AGENT)}"
        ) from exc


def _promise_completed(agent: AgentResult, adapter: CLIAdapter, config: RunConfig) -> bool:
    """Return whether the agent's exit + output satisfy the completion signal."""
    if not agent.success:
        return False
    if config.stop_on_completion_signal and agent.completion_detected:
        return True
    return bool(
        adapter.extract_completion_signal(
            result_text=agent.result_text,
            stdout=agent.captured_stdout,
            user_signal=config.completion_signal,
        )
    )


def _emit_turn_capped(
    agent: AgentResult, state: RunState, emit: BoundEmitter, hooks: CombinedAgentHook | None
) -> None:
    if not agent.turn_capped:
        return
    emit(
        EventType.ITERATION_TURN_CAPPED,
        TurnCappedData(iteration=state.iteration, count=agent.tool_use_count),
    )
    if hooks is not None:
        hooks.on_turn_capped(iteration=state.iteration, count=agent.tool_use_count)


def _classify_iteration_outcome(
    agent: AgentResult,
    state: RunState,
    promise_completed: bool,
    completion_signal: str,
    duration: str,
) -> tuple[EventType, str]:
    """Mark the state counter for this outcome and derive its (event_type, detail)."""
    if agent.timed_out:
        state.mark_timed_out()
        return EventType.ITERATION_TIMED_OUT, f"timed out after {duration}"
    if agent.turn_capped:
        state.mark_completed()
        detail = f"completed at turn cap ({agent.tool_use_count} tool uses, {duration})"
        return EventType.ITERATION_COMPLETED, detail
    if agent.success:
        state.mark_completed()
        if promise_completed:
            detail = (
                f"completed via promise tag <promise>{completion_signal}</promise> ({duration})"
            )
        else:
            detail = f"completed ({duration})"
        return EventType.ITERATION_COMPLETED, detail
    state.mark_failed()
    return EventType.ITERATION_FAILED, f"failed with exit code {agent.returncode} ({duration})"


def _build_ended_data(
    agent: AgentResult,
    state: RunState,
    config: RunConfig,
    duration: str,
    state_detail: str,
    emit: BoundEmitter,
) -> IterationEndedData:
    """Build the IterationEndedData, echoing raw output when peek is off."""
    ended_data = IterationEndedData(
        iteration=state.iteration,
        returncode=agent.returncode,
        duration=agent.elapsed,
        duration_formatted=duration,
        detail=state_detail,
        log_file=str(agent.log_file) if agent.log_file else None,
        result_text=agent.result_text,
    )
    if not emit.wants_agent_output_lines():
        # When peek was off, echo any captured raw output after the spinner
        # stops so blocking agents do not appear silent. Structured agents
        # already surface their parsed result_text, so avoid echoing raw JSON
        # unless we explicitly captured logs.
        if config.log_dir is not None:
            ended_data["echo_stdout"] = agent.captured_stdout
            ended_data["echo_stderr"] = agent.captured_stderr
        elif agent.result_text is None and agent.captured_stdout is not None:
            ended_data["echo_stdout"] = agent.captured_stdout
    return ended_data


def _notify_iteration_hooks(
    hooks: CombinedAgentHook | None,
    state: RunState,
    agent: AgentResult,
    promise_completed: bool,
    completion_signal: str | None,
) -> None:
    if hooks is None:
        return
    hooks.on_iteration_completed(
        iteration=state.iteration,
        result={
            "returncode": agent.returncode,
            "timed_out": agent.timed_out,
            "turn_capped": agent.turn_capped,
            "tool_use_count": agent.tool_use_count,
            "duration": agent.elapsed,
            "result_text": agent.result_text,
        },
    )
    if promise_completed:
        hooks.on_completion_signal(iteration=state.iteration, signal=cast(str, completion_signal))


def _run_agent_phase(
    prompt: str,
    config: RunConfig,
    state: RunState,
    emit: BoundEmitter,
    hooks: CombinedAgentHook | None,
) -> tuple[bool, bool]:
    """Run the agent subprocess, update state counters, and emit the result event.

    Returns ``(agent_succeeded, stop_for_completion_signal)``.
    """
    cmd, adapter = _resolve_agent_command(config)
    callbacks = _build_agent_callbacks(config, state, emit, hooks)
    agent = _launch_agent(cmd, adapter, prompt, config, state, callbacks)
    state.last_result_text = agent.result_text
    state.last_captured_stdout = agent.captured_stdout
    state.last_captured_stderr = agent.captured_stderr
    if getattr(agent, "force_stopped", False):
        state.status = RunStatus.STOPPED
        return False, False

    duration = format_duration(agent.elapsed)
    promise_completed = _promise_completed(agent, adapter, config)
    probe_completed = bool(
        agent.success and config.completion_probe is not None and config.completion_probe()
    )
    completion_detected = promise_completed or probe_completed
    if completion_detected:
        state.promise_completed = True

    _emit_turn_capped(agent, state, emit, hooks)

    event_type, state_detail = _classify_iteration_outcome(
        agent, state, completion_detected, config.completion_signal, duration
    )
    ended_data = _build_ended_data(agent, state, config, duration, state_detail, emit)
    emit(event_type, ended_data)
    _notify_iteration_hooks(hooks, state, agent, completion_detected, config.completion_signal)

    # Derive the stop_on_error signal from the authoritative classification so
    # it always agrees with _classify_iteration_outcome: only ITERATION_COMPLETED
    # is a success. This treats a graceful turn cap as success (the streaming
    # path SIGKILLs the child, so agent.success is False) without masking a real
    # timeout — the blocking path can report timed_out and turn_capped together
    # (post-hoc cap counting runs after a timeout), and that classifies as
    # ITERATION_TIMED_OUT, so stop_on_error still fires.
    iteration_succeeded = event_type == EventType.ITERATION_COMPLETED
    return iteration_succeeded, completion_detected and config.stop_on_completion_signal


def _build_tool_use_bridge(
    *,
    state: RunState,
    emit: BoundEmitter,
    hooks: CombinedAgentHook | None,
    max_turns: int | None,
    max_turns_grace: int,
) -> ToolUseCallback | None:
    """Return a ``ToolUseCallback`` that emits ``TOOL_USE`` and approaching-limit events.

    Collapses the per-tool-use notification shape expected by ``_agent``
    (``(tool_name, count)``) into structured events plus the hook
    notifications. Returns ``None`` when no subscriber cares — the
    streaming path then skips all per-line overhead.
    """
    if max_turns is None and hooks is None:
        return None

    # Clamp the grace below the cap.  ``RunConfig`` does not reject a grace
    # >= max_turns the way the CLI does, so an unclamped value would make
    # the threshold <= 0 and fire ITERATION_TURN_APPROACHING_LIMIT on the
    # first tool use.  Mirrors the wind-down shim's clamp.
    approaching_threshold = (
        (max_turns - min(max_turns_grace, max(max_turns - 1, 0)))
        if max_turns is not None and max_turns_grace > 0
        else None
    )
    approaching_fired = False

    def _on_tool_use(tool_name: str, count: int) -> None:
        nonlocal approaching_fired
        emit(
            EventType.TOOL_USE,
            ToolUseData(
                iteration=state.iteration,
                tool_name=tool_name,
                count=count,
            ),
        )
        if hooks is not None:
            hooks.on_tool_use(
                iteration=state.iteration,
                tool_name=tool_name,
                count=count,
            )
        if (
            not approaching_fired
            and approaching_threshold is not None
            and max_turns is not None
            and count >= approaching_threshold
            and count < max_turns
        ):
            approaching_fired = True
            emit(
                EventType.ITERATION_TURN_APPROACHING_LIMIT,
                TurnApproachingLimitData(
                    iteration=state.iteration,
                    count=count,
                    max_turns=max_turns,
                ),
            )
            if hooks is not None:
                hooks.on_turn_approaching_limit(
                    iteration=state.iteration,
                    count=count,
                    max_turns=max_turns,
                )

    return _on_tool_use


def _run_iteration(
    config: RunConfig,
    state: RunState,
    emit: BoundEmitter,
    hooks: CombinedAgentHook | None,
    verifier_feedback: str | None = None,
) -> tuple[bool, bool]:
    """Execute one iteration of the agent loop.

    Returns (should_continue, promise_would_complete):
      - should_continue: True if the loop should continue, False to break
      - promise_would_complete: True if an exit-0 + promise tag would
        complete the run (subject to the loop's completion verifier)
    """
    iteration = state.iteration

    emit(EventType.ITERATION_STARTED, IterationStartedData(iteration=iteration))
    if hooks is not None:
        hooks.on_iteration_started(iteration=iteration)

    command_outputs: dict[str, str] = {}
    if config.commands:
        emit(
            EventType.COMMANDS_STARTED,
            CommandsStartedData(iteration=iteration, count=len(config.commands)),
        )
        command_outputs = _run_commands(
            config.commands,
            config.ralph_dir,
            config.project_root,
            config.args,
        )
        emit(
            EventType.COMMANDS_COMPLETED,
            CommandsCompletedData(
                iteration=iteration,
                count=len(command_outputs),
            ),
        )
        if hooks is not None:
            hooks.on_commands_completed(iteration=iteration, outputs=command_outputs)

    prompt = _assemble_prompt(
        config, state, command_outputs, verifier_feedback, state.take_guidance()
    )
    emit(
        EventType.PROMPT_ASSEMBLED,
        PromptAssembledData(iteration=iteration, prompt_length=len(prompt)),
    )
    if hooks is not None:
        hooks.on_prompt_assembled(iteration=iteration, prompt=prompt)

    agent_succeeded, promise_would_complete = _run_agent_phase(prompt, config, state, emit, hooks)
    if state.status is RunStatus.STOPPED:
        return False, promise_would_complete

    if not agent_succeeded and config.stop_on_error:
        if state.try_commit_failure():
            emit.log_error("Stopping due to --stop-on-error.")
        return False, promise_would_complete

    cap = config.max_consecutive_failures
    if not agent_succeeded and cap is not None and state.consecutive_failures >= cap:
        state.status = RunStatus.FAILED
        emit.log_error(
            f"Stopping after {state.consecutive_failures} consecutive failed iterations "
            + "(max_consecutive_failures); the agent command may be unable to start."
        )
        return False, promise_would_complete
    return True, promise_would_complete


def _delay_if_needed(config: RunConfig, state: RunState, emit: BoundEmitter) -> None:
    """Sleep between iterations when a delay is configured.

    Uses :meth:`RunState.wait_for_stop` so that stop requests break the
    delay immediately rather than polling.
    """
    if config.delay > 0 and (
        config.max_iterations is None or state.iteration < config.max_iterations
    ):
        emit.log_info(f"Waiting {format_duration(config.delay)}...")
        _ = state.wait_for_stop(timeout=config.delay)


def run_loop(
    config: RunConfig,
    state: RunState,
    emitter: EventEmitter | None = None,
) -> None:
    """Execute the autonomous agent loop.

    Each iteration: run commands → assemble prompt → pipe to agent → repeat.
    """
    if emitter is None:
        emitter = NullEmitter()

    emit = BoundEmitter(emitter, state.run_id)
    state.status = RunStatus.RUNNING
    state.started_at = datetime.now(UTC)

    hooks = CombinedAgentHook(list(config.hooks)) if config.hooks else None

    if config.log_dir:
        config.log_dir.mkdir(parents=True, exist_ok=True)

    emit(
        EventType.RUN_STARTED,
        RunStartedData(
            ralph_name=config.ralph_dir.name,
            agent=config.agent,
            commands=len(config.commands),
            max_iterations=config.max_iterations,
            timeout=config.timeout,
            delay=config.delay,
        ),
    )

    pending_feedback: str | None = None
    verifier_rejected = False

    try:
        while True:
            if not _handle_control_signals(state, emit):
                break

            if config.max_iterations is not None and state.iteration >= config.max_iterations:
                state.close_guidance()
                if verifier_rejected and state.try_commit_failure():
                    emit.log_error(
                        "Completion verifier never accepted within the "
                        + f"{config.max_iterations}-iteration budget."
                    )
                break
            state.iteration += 1

            should_continue, promise_would_complete = _run_iteration(
                config, state, emit, hooks, pending_feedback
            )
            pending_feedback = None
            if state.stop_requested:
                state.status = RunStatus.STOPPED
                break
            if promise_would_complete:
                verdict = config.completion_verifier() if config.completion_verifier else None
                if (verdict is None or verdict.ok) and state.try_commit_soft_completion():
                    break
                if cast(RunStatus, cast(object, state.status)) is RunStatus.STOPPED:
                    break
                if verdict is not None and not verdict.ok:
                    verifier_rejected = True
                    pending_feedback = verdict.feedback
                    emit.log_info(f"Completion verifier rejected: {verdict.feedback}")
            if not should_continue:
                break

            _delay_if_needed(config, state, emit)

    except KeyboardInterrupt:
        state.status = RunStatus.STOPPED
    except Exception as exc:
        if state.try_commit_failure():
            tb = traceback.format_exc()
            emit.log_error(f"Run crashed: {exc}", traceback=tb)

    if state.status == RunStatus.RUNNING:
        _ = state.try_commit_completion()
    state.close_guidance()

    emit(
        EventType.RUN_STOPPED,
        RunStoppedData(
            reason=state.status.reason,
            total=state.total,
            completed=state.completed,
            failed=state.failed,
            timed_out_count=state.timed_out_count,
        ),
    )
