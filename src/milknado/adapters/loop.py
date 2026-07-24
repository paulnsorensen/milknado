from __future__ import annotations

import logging
import queue
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from milknado.domains.common import (
    CompletionTimeout,
    Gate,
    MikadoNode,
    ProgressEvent,
    TerminalRunOutcome,
    VerifySpecResult,
    build_resume_command,
)
from milknado.domains.execution import build_completion_verifier
from milknado.loop import EventType, QueueEmitter, RunConfig, RunManager, RunStatus

MILKNADO_COMPLETION_SIGNAL: Final[str] = "MILKNADO_NODE_COMPLETE"

# A worker that self-heals across iterations may fail a few times, but a
# command that cannot start (e.g. an unknown CLI flag) fails identically
# every ~1s forever. Three failures in a row ends the run loudly instead.
MAX_CONSECUTIVE_AGENT_FAILURES: Final[int] = 3


_logger = logging.getLogger(__name__)


class LoopAdapter:
    def __init__(self, agent: str = "") -> None:
        self._manager = RunManager()
        self._queue: queue.Queue[Any] = queue.Queue()
        self._emitter = QueueEmitter(self._queue)
        self._agent = agent

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: tuple[Gate, ...] | None,
        project_root: Path | None = None,
        commit_footer: str | None = None,
        base_oid: str | None = None,
        runtime_policy: Any | None = None,
        run_id: str | None = None,
    ) -> Any:
        mcp_config = project_root / ".mcp.json" if project_root else None
        agent_cmd = agent
        session = getattr(runtime_policy, "session", None)
        if session is not None:
            agent_cmd = build_resume_command(agent_cmd, session.family, session.session_id)
        # Only the claude CLI understands --mcp-config; omp/codex/gemini reject
        # the flag at launch and the worker crash-loops without ever starting.
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

    def list_runs(self) -> list[Any]:
        return self._manager.list_runs()

    def get_run(self, run_id: str) -> Any | None:
        return self._manager.get_run(run_id)

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
                data = event.data
                iteration = data.get("iteration", 0) if isinstance(data, dict) else 0
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
            ralph_file.write_text(
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
            ev_queue = local_run.emitter.queue
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
        import tempfile

        with tempfile.TemporaryDirectory(prefix="milknado-review-") as tmpdir:
            temp_root = Path(tmpdir)
            ralph_file = temp_root / "review.md"
            local_manager = RunManager()
            local_queue: queue.Queue[Any] = queue.Queue()
            local_emitter = QueueEmitter(local_queue)
            ralph_file.write_text(prompt, encoding="utf-8")
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
        node: MikadoNode,
        context: str,
        quality_gates: tuple[Gate, ...] | None,
        output_path: Path,
    ) -> Path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            content = _build_ralph_content(node, context, quality_gates)
            output_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            from milknado.domains.common import RalphMarkdownWriteError

            raise RalphMarkdownWriteError(path=output_path, cause=exc) from exc
        return output_path


def _build_verify_prompt(spec_text: str, graph_state: Any) -> str:
    graph_summary = str(graph_state) if graph_state is not None else "(no graph state)"
    return (
        "# Spec Verification\n\n"
        "Review whether the following spec has been fully implemented.\n\n"
        f"## Spec\n\n{spec_text}\n\n"
        f"## Graph State\n\n{graph_summary}\n\n"
        "## Instructions\n\n"
        "If the spec is fully satisfied, emit:\n"
        "<result>done</result>\n\n"
        "If there are unmet requirements, emit:\n"
        "<result>gaps</result>\n"
        "<goal_delta>\n"
        "Description of what is still missing\n"
        "</goal_delta>\n\n"
        "Then emit the completion signal to stop this run:\n"
        f"<promise>{MILKNADO_COMPLETION_SIGNAL}</promise>\n"
    )


def _drain_verify_run(
    local_manager: RunManager,
    run_id: str,
    ev_queue: queue.Queue[Any],
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
                local_manager.stop_and_join(run_id, timeout=5.0)
                return VerifySpecResult(outcome="gaps", goal_delta="verification timed out")
            try:
                event = ev_queue.get(timeout=remaining)
            except queue.Empty:
                local_manager.stop_and_join(run_id, timeout=5.0)
                return VerifySpecResult(outcome="gaps", goal_delta="verification timed out")
            if event.type in _ITERATION_EVENTS:
                text = event.data.get("result_text") or ""
                if text:
                    output_parts.append(text)
            elif event.type == EventType.RUN_STOPPED:
                break
    except Exception as exc:
        _logger.exception("verify_spec drain failed for run_id=%s", run_id)
        try:
            local_manager.stop_and_join(run_id, timeout=5.0)
        except Exception:
            _logger.exception("verify_spec stop failed for run_id=%s", run_id)
        return VerifySpecResult(outcome="gaps", goal_delta=f"verification failed: {exc}")
    return _parse_verify_output("\n".join(output_parts))


def _drain_review_run(
    local_manager: RunManager,
    run_id: str,
    ev_queue: queue.Queue[Any],
) -> ReviewVerdict:
    """Drain one reviewer run and convert its final output into a verdict."""
    output_parts: list[str] = []
    deadline = time.monotonic() + 1800.0
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                local_manager.stop_and_join(run_id, timeout=5.0)
                return ReviewVerdict(
                    approved=False,
                    findings_md="reviewer timed out before producing a verdict",
                )
            try:
                event = ev_queue.get(timeout=remaining)
            except queue.Empty:
                local_manager.stop_and_join(run_id, timeout=5.0)
                return ReviewVerdict(
                    approved=False,
                    findings_md="reviewer timed out before producing a verdict",
                )
            if event.type in {
                EventType.ITERATION_COMPLETED,
                EventType.ITERATION_FAILED,
            }:
                text = event.data.get("result_text") or event.data.get("echo_stdout") or ""
                if text:
                    output_parts.append(str(text))
            elif event.type == EventType.RUN_STOPPED:
                break
    except Exception as exc:
        _logger.exception("node review drain failed for run_id=%s", run_id)
        try:
            local_manager.stop_and_join(run_id, timeout=5.0)
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
    node: MikadoNode,
    context: str,
    quality_gates: tuple[Gate, ...] | None,
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
    return (
        f"# {node.description}\n\n"
        f"## Context\n\n{context}\n\n"
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


@dataclass(frozen=True)
class ReviewVerdict:
    approved: bool
    findings_md: str


def _parse_review_output(output: str) -> tuple[bool, str]:
    match = re.search(r"<verdict>\s*(approve|reject|revise)\s*</verdict>", output, re.I)
    findings_md = output[: match.start()].strip() if match else output.strip()
    if match and match.group(1).lower() == "approve":
        return True, findings_md
    if match:
        return False, findings_md
    _logger.warning("run_node_review: unparseable reviewer output; treating as revise")
    return False, findings_md or "reviewer produced no parseable <verdict> tag"
