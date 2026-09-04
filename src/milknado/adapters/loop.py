from __future__ import annotations

import logging
import queue
import re
import shlex
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from milknado.adapters._loop_types import ReviewVerdict
from milknado.adapters._loop_types import RunHandleView as _RunHandle
from milknado.adapters._loop_types import (
    build_verify_prompt as _build_verify_prompt_impl,
)
from milknado.adapters._loop_types import event_text as _event_text
from milknado.domains.common import (
    CompletionTimeout,
    Gate,
    NodeAgentSession,
    ProgressEvent,
    TerminalRunOutcome,
    VerifySpecResult,
    build_resume_command,
)
from milknado.domains.execution import build_completion_verifier
from milknado.loop import EventType, QueueEmitter, RunConfig, RunManager, RunStatus

if TYPE_CHECKING:
    from milknado.loop._events import Event, EventData

MILKNADO_COMPLETION_SIGNAL: Final[str] = "MILKNADO_NODE_COMPLETE"

MAX_CONSECUTIVE_AGENT_FAILURES: Final[int] = 3  # Stop after three launch failures.


_logger = logging.getLogger(__name__)


class LoopAdapter:
    def __init__(self, agent: str = "") -> None:
        self._manager: RunManager = RunManager()
        self._queue: queue.Queue[Event[EventData]] = queue.Queue()
        self._emitter: QueueEmitter = QueueEmitter(self._queue)
        self._agent: str = agent

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        quality_gates: tuple[Gate, ...] | None,
        project_root: Path | None = None,
        commit_footer: str | None = None,
        base_oid: str | None = None,
        runtime_policy: object | None = None,
        run_id: str | None = None,
        completion_probe: Callable[[], bool] | None = None,
    ) -> _RunHandle:
        mcp_config = project_root / ".mcp.json" if project_root else None
        agent_cmd = agent
        session = cast(
            NodeAgentSession | None,
            getattr(runtime_policy, "session", None),
        )
        if session is not None:
            agent_cmd = build_resume_command(agent_cmd, session.family, session.session_id)
        # Only Claude supports --mcp-config; other CLIs reject it at launch.
        supports_mcp_flag = Path(shlex.split(agent_cmd)[0]).name == "claude"
        if mcp_config and mcp_config.exists() and supports_mcp_flag:
            agent_cmd = shlex.join([*shlex.split(agent_cmd), "--mcp-config", str(mcp_config)])
        config = RunConfig(
            agent=agent_cmd,
            ralph_dir=ralph_dir,
            ralph_file=ralph_file,
            project_root=project_root or ralph_dir,
            completion_signal=MILKNADO_COMPLETION_SIGNAL,
            stop_on_completion_signal=True,
            stop_on_error=True,
            log_dir=ralph_dir / ".ralph-logs",
            commit_footer=commit_footer,
            max_consecutive_failures=MAX_CONSECUTIVE_AGENT_FAILURES,
        )
        if completion_probe is not None:
            config.completion_probe = completion_probe
        if completion_probe is None:
            config.completion_verifier = build_completion_verifier(
                ralph_dir, quality_gates, base_oid=base_oid
            )
        return self._manager.create_run(config, emitter=self._emitter, run_id=run_id)

    def start_run(self, run_id: str) -> None:
        self._manager.start_run(run_id)

    def queue_guidance(self, run_id: str, text: str) -> bool:
        return self._manager.queue_guidance(run_id, text)

    def request_stop_run(self, run_id: str) -> None:
        self._manager.stop_run(run_id)

    def force_stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        return self._manager.force_stop_and_join(run_id, timeout)

    def stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        return self._manager.stop_and_join(run_id, timeout)

    def list_runs(self) -> Sequence[_RunHandle]:
        return self._manager.list_runs()

    def get_run(self, run_id: str) -> _RunHandle | None:
        return self._manager.get_run(run_id)

    def is_run_alive(self, run_id: str) -> bool:
        run = self._manager.get_run(run_id)
        return run is not None and (run.thread is None or run.thread.is_alive())

    def get_run_stdout(self, run_id: str) -> list[str]:
        run = self._manager.get_run(run_id)
        if run is None:
            return []
        text = run.state.last_captured_stdout
        if text is None:
            text = run.state.last_result_text
        return text.splitlines() if text is not None else []

    def get_run_failure_detail(self, run_id: str) -> str | None:
        """Last captured agent output for a failed run, flattened to one log line."""
        run = self._manager.get_run(run_id)
        if run is None:
            return None
        state = run.state
        for text in (
            getattr(state, "last_captured_stderr", None),
            getattr(state, "last_captured_stdout", None),
            getattr(state, "last_result_text", None),
        ):
            if isinstance(text, str) and text.strip():
                return " ".join(text.split())[-300:]
        return None

    def get_run_output_tail(self, run_id: str, max_lines: int) -> list[str]:
        """Return at most ``max_lines`` of output without splitting the whole text."""
        if max_lines <= 0:
            return []
        run = self._manager.get_run(run_id)
        if run is None:
            return []
        text = run.state.last_captured_stdout
        if text is None:
            text = run.state.last_result_text
        if not text:
            return []
        lines = text.rstrip("\r\n").rsplit("\n", max_lines)[-max_lines:]
        return [line.rstrip("\r") for line in lines]

    def get_run_guidance(self, run_id: str) -> tuple[str, ...]:
        run = self._manager.get_run(run_id)
        return () if run is None else run.state.pending_guidance

    def wait_for_next_completion(
        self,
        active_run_ids: set[str],
        timeout: float | None = None,
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]:
        start = time.monotonic()
        deadline = start + timeout if timeout is not None else None
        while True:
            remaining: float | None = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CompletionTimeout(
                        active_run_ids=active_run_ids,
                        waited_seconds=time.monotonic() - start,
                    )
            try:
                event = self._queue.get(timeout=remaining)
            except queue.Empty:
                raise CompletionTimeout(
                    active_run_ids=active_run_ids,
                    waited_seconds=time.monotonic() - start,
                ) from None
            if event.run_id not in active_run_ids:
                continue
            if event.type == EventType.ITERATION_STARTED:
                iteration_value = cast(Mapping[str, object], event.data).get("iteration", 0)
                iteration = iteration_value if isinstance(iteration_value, int) else 0
                return event.run_id, ProgressEvent(
                    run_id=event.run_id,
                    work=iteration,
                    total=0,
                    message=f"iteration {iteration} started",
                )
            if event.type != EventType.RUN_STOPPED:
                continue
            run = self._manager.get_run(event.run_id)
            if run is None:
                return event.run_id, "failed"
            status = run.state.status
            if status is RunStatus.COMPLETED:
                return event.run_id, "completed"
            if status is RunStatus.STOPPED:
                return event.run_id, "stopped"
            return event.run_id, "failed"

    def verify_spec(self, spec_text: str, graph_state: str) -> VerifySpecResult:
        if not self._agent:
            _logger.warning("verify_spec: no agent configured, returning done")
            return VerifySpecResult(outcome="done")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ralph_file = tmp_path / "ralph.md"
            _ = ralph_file.write_text(
                _build_verify_prompt(spec_text, graph_state),
                encoding="utf-8",
            )
            local_manager = RunManager()
            config = RunConfig(
                agent=self._agent,
                ralph_dir=tmp_path,
                ralph_file=ralph_file,
                project_root=tmp_path,
                completion_signal=MILKNADO_COMPLETION_SIGNAL,
                stop_on_completion_signal=True,
                max_consecutive_failures=MAX_CONSECUTIVE_AGENT_FAILURES,
            )
            local_run = local_manager.create_run(config)
            run_id = local_run.state.run_id
            ev_queue = cast(
                QueueEmitter,
                cast(object, local_run.emitter),
            ).queue
            local_manager.start_run(run_id)
            return _drain_verify_run(local_manager, run_id, ev_queue)

    def run_node_review(
        self,
        agent: str,
        prompt: str,
        worktree: Path,
        project_root: Path,
    ) -> ReviewVerdict:
        """Run one bounded, read-only reviewer turn in the pinned worktree."""
        del project_root
        import tempfile

        with tempfile.TemporaryDirectory(prefix="milknado-review-") as tmpdir:
            temp_root = Path(tmpdir)
            ralph_file = temp_root / "review.md"
            local_manager = RunManager()
            local_queue: queue.Queue[Event[EventData]] = queue.Queue()
            local_emitter = QueueEmitter(local_queue)
            _ = ralph_file.write_text(prompt, encoding="utf-8")
            config = RunConfig(
                agent=agent,
                ralph_dir=temp_root,
                ralph_file=ralph_file,
                project_root=worktree,
                completion_signal="MILKNADO_NODE_REVIEW_COMPLETE",
                stop_on_completion_signal=True,
                max_iterations=1,
                timeout=1800,
                log_dir=temp_root / ".ralph-logs",
            )
            run = local_manager.create_run(config, emitter=local_emitter)
            local_manager.start_run(run.state.run_id)
            return _drain_review_run(local_manager, run.state.run_id, local_queue)

    def generate_ralph_md(
        self,
        brief: str,
        quality_gates: tuple[Gate, ...] | None,
        output_path: Path,
        prior_findings: str = "",
        findings_round: int | None = None,
    ) -> Path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            content = _build_ralph_content(
                brief,
                quality_gates,
                prior_findings=prior_findings,
                findings_round=findings_round,
            )
            _ = output_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            from milknado.domains.common import RalphMarkdownWriteError

            raise RalphMarkdownWriteError(path=output_path, cause=exc) from exc
        return output_path


def _build_verify_prompt(spec_text: str, graph_state: object | None) -> str:
    return _build_verify_prompt_impl(spec_text, graph_state, MILKNADO_COMPLETION_SIGNAL)


def _drain_verify_run(
    local_manager: RunManager,
    run_id: str,
    ev_queue: queue.Queue[Event[EventData]],
) -> VerifySpecResult:
    _ITERATION_EVENTS = frozenset(
        {
            EventType.ITERATION_COMPLETED,
            EventType.ITERATION_FAILED,
        }
    )
    output_parts: list[str] = []
    deadline = time.monotonic() + 120.0
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _ = local_manager.stop_and_join(run_id, timeout=5.0)
                return VerifySpecResult(outcome="gaps", goal_delta="verification timed out")
            try:
                event = ev_queue.get(timeout=remaining)
            except queue.Empty:
                _ = local_manager.stop_and_join(run_id, timeout=5.0)
                return VerifySpecResult(outcome="gaps", goal_delta="verification timed out")
            if event.type in _ITERATION_EVENTS:
                text = _event_text(event.data, "result_text")
                if text:
                    output_parts.append(text)
            elif event.type == EventType.RUN_STOPPED:
                break
    except Exception as exc:
        _logger.exception("verify_spec drain failed for run_id=%s", run_id)
        try:
            _ = local_manager.stop_and_join(run_id, timeout=5.0)
        except Exception:
            _logger.exception("verify_spec stop failed for run_id=%s", run_id)
        return VerifySpecResult(outcome="gaps", goal_delta=f"verification failed: {exc}")
    return _parse_verify_output("\n".join(output_parts))


def _drain_review_run(
    local_manager: RunManager,
    run_id: str,
    ev_queue: queue.Queue[Event[EventData]],
) -> ReviewVerdict:
    """Drain one reviewer run and convert its final output into a verdict."""
    output_parts: list[str] = []
    deadline = time.monotonic() + 1800.0
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _ = local_manager.stop_and_join(run_id, timeout=5.0)
                return ReviewVerdict(
                    approved=False,
                    findings_md="reviewer timed out before producing a verdict",
                )
            try:
                event = ev_queue.get(timeout=remaining)
            except queue.Empty:
                _ = local_manager.stop_and_join(run_id, timeout=5.0)
                return ReviewVerdict(
                    approved=False,
                    findings_md="reviewer timed out before producing a verdict",
                )
            if event.type in {
                EventType.ITERATION_COMPLETED,
                EventType.ITERATION_FAILED,
            }:
                text = _event_text(event.data, "result_text", "echo_stdout")
                if text:
                    output_parts.append(text)
            elif event.type == EventType.RUN_STOPPED:
                break
    except Exception as exc:
        _logger.exception("node review drain failed for run_id=%s", run_id)
        try:
            _ = local_manager.stop_and_join(run_id, timeout=5.0)
        except Exception:
            _logger.exception("node review stop failed for run_id=%s", run_id)
        return ReviewVerdict(approved=False, findings_md=f"reviewer failed: {exc}")
    approved, findings_md = _parse_review_output("\n".join(output_parts))
    return ReviewVerdict(approved=approved, findings_md=findings_md)


def _parse_verify_output(output: str) -> VerifySpecResult:

    if "<result>done</result>" in output:
        return VerifySpecResult(outcome="done")
    if "<result>gaps</result>" in output:
        m = re.search(r"<goal_delta>(.*?)</goal_delta>", output, re.DOTALL)
        delta = m.group(1).strip() if m else None
        return VerifySpecResult(outcome="gaps", goal_delta=delta)
    _logger.warning("verify_spec: unparseable output, returning gaps")
    return VerifySpecResult(outcome="gaps", goal_delta="verification produced no explicit result")


def _build_ralph_content(
    brief: str,
    quality_gates: tuple[Gate, ...] | None,
    prior_findings: str = "",
    findings_round: int | None = None,
) -> str:
    if quality_gates is None:
        gates_section = (
            "## Quality Gates\n\n"
            "_(no gates configured — node cannot complete; "
            "add quality_gates to milknado.toml)_"
        )
    elif not quality_gates:
        gates_section = "## Quality Gates\n\n_(gates explicitly skipped for this flavor)_"
    else:
        gate_lines = "\n".join(f"- `{g.command}`" for g in quality_gates)
        gates_section = f"## Quality Gates\n\n{gate_lines}"
    round_label = f" (round {findings_round})" if findings_round is not None else ""
    findings_section = (
        f"## Prior review findings{round_label}\n\n{prior_findings.rstrip()}\n\n"
        if prior_findings.strip()
        else ""
    )
    return (
        f"{findings_section}{brief.rstrip()}\n\n"
        f"{gates_section}\n\n"
        "## Proposing follow-up work\n\n"
        "If this node cannot be finished without other work landing first "
        "(a missing prerequisite, or a refactor that must precede this "
        "change), do not force it into this loop. Register each item by "
        "calling the follow-up tracking tool "
        "(`mcp__milknado__milknado_track_follow_up` for Claude-family workers, "
        "`milknado_track_follow_up` for Gemini) with a one-line description, and "
        "keep your own change scoped to the files above. This is the Mikado "
        "discipline: register prerequisites as new graph nodes rather than "
        "widening this one.\n\n"
        "## Completion\n\n"
        "When every quality gate passes and this node is fully implemented,\n"
        f"emit `<promise>{MILKNADO_COMPLETION_SIGNAL}</promise>` on its own line\n"
        "so the run can stop before the iteration budget.\n"
    )


def _parse_review_output(output: str) -> tuple[bool, str]:
    match = re.search(r"<verdict>\s*(approve|reject|revise)\s*</verdict>", output, re.I)
    findings_md = output[: match.start()].strip() if match else output.strip()
    if match and match.group(1).lower() == "approve":
        return True, findings_md
    if match:
        return False, findings_md
    _logger.warning("run_node_review: unparseable reviewer output; treating as revise")
    return False, findings_md or "reviewer produced no parseable <verdict> tag"
