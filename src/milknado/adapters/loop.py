from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from milknado.domains.common.errors import CompletionTimeout
from milknado.domains.common.protocols import ProgressEvent, VerifySpecResult
from milknado.domains.common.types import MikadoNode
from milknado.loop import (
    CompletionVerdict,
    EventType,
    QueueEmitter,
    RunConfig,
    RunManager,
    RunStatus,
)

MILKNADO_COMPLETION_SIGNAL: Final[str] = "MILKNADO_NODE_COMPLETE"

# Per-gate wall-clock cap, mirroring the loop's DEFAULT_COMMAND_TIMEOUT so a
# hanging gate cannot stall the completion verdict.
_GATE_TIMEOUT_SECONDS: Final[float] = 60.0
# Tail of a failing gate's combined output folded into the rejection feedback.
_FEEDBACK_TAIL_CHARS: Final[int] = 2000

_NO_CHANGE_FEEDBACK: Final[str] = (
    "you emitted the completion promise but produced no committed/stageable change"
)

_logger = logging.getLogger(__name__)


class LoopAdapter:
    def __init__(self, agent: str = "") -> None:
        self._manager = RunManager()
        self._queue: queue.Queue[Any] = queue.Queue()
        self._emitter = QueueEmitter(self._queue)
        self._agent = agent
        self._progress_buffer: list[ProgressEvent] = []
        self._progress_lock = threading.Lock()

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: list[str],
        project_root: Path | None = None,
    ) -> Any:
        mcp_config = project_root / ".mcp.json" if project_root else None
        agent_cmd = agent
        if mcp_config and mcp_config.exists():
            agent_cmd = f"{agent} --mcp-config {mcp_config}"
        config = RunConfig(
            agent=agent_cmd,
            ralph_dir=ralph_dir,
            ralph_file=ralph_file,
            project_root=ralph_dir,
            completion_signal=MILKNADO_COMPLETION_SIGNAL,
            stop_on_completion_signal=True,
            log_dir=ralph_dir / ".ralph-logs",
        )
        config.completion_verifier = _build_completion_verifier(ralph_dir, quality_gates)
        run = self._manager.create_run(config)
        run.add_listener(self._emitter)
        return run

    def start_run(self, run_id: str) -> None:
        self._manager.start_run(run_id)

    def stop_run(self, run_id: str) -> None:
        self._manager.stop_run(run_id)

    def list_runs(self) -> list[Any]:
        return self._manager.list_runs()

    def get_run(self, run_id: str) -> Any | None:
        return self._manager.get_run(run_id)

    def get_run_stdout(self, run_id: str) -> list[str]:
        run = self._manager.get_run(run_id)
        if run is None:
            return []
        stdout = getattr(run, "stdout", None)
        if isinstance(stdout, list):
            return stdout
        if isinstance(stdout, str):
            return stdout.splitlines()
        return []

    def wait_for_next_completion(
        self,
        active_run_ids: set[str],
        timeout: float | None = None,
    ) -> tuple[str, bool]:
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
            _PROGRESS = getattr(EventType, "PROGRESS", None)
            if _PROGRESS is not None and event.type == _PROGRESS:
                with self._progress_lock:
                    self._progress_buffer.append(
                        ProgressEvent(
                            run_id=event.run_id,
                            work=getattr(event, "work", 0),
                            total=getattr(event, "total", 0),
                            message=getattr(event, "message", ""),
                        )
                    )
                continue
            if event.type != EventType.RUN_STOPPED:
                continue
            if event.run_id not in active_run_ids:
                continue
            run = self._manager.get_run(event.run_id)
            success = run is not None and run.state.status == RunStatus.COMPLETED
            return event.run_id, success

    def poll_progress_events(self) -> list[ProgressEvent]:
        with self._progress_lock:
            result = list(self._progress_buffer)
            self._progress_buffer.clear()
        return result

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
            )
            local_run = local_manager.create_run(config)
            run_id = local_run.state.run_id
            ev_queue = local_run.emitter.queue
            local_manager.start_run(run_id)
            return _drain_verify_run(local_manager, run_id, ev_queue)

    def generate_ralph_md(
        self,
        node: MikadoNode,
        context: str,
        quality_gates: list[str],
        output_path: Path,
    ) -> Path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            content = _build_ralph_content(node, context, quality_gates)
            output_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            from milknado.domains.common.errors import RalphMarkdownWriteError

            raise RalphMarkdownWriteError(path=output_path, cause=exc) from exc
        return output_path


def _build_completion_verifier(
    worktree: Path,
    quality_gates: list[str],
) -> Callable[[], CompletionVerdict]:
    """Build the tier-2 completion verifier for a node's worktree.

    The returned closure is the authoritative harness-side check the engine
    consults when exit-0 + the completion promise would otherwise mark the
    node done. It re-runs every configured quality gate in *worktree* (the
    RALPH.md copy is only fast agent-side feedback) and confirms the branch
    actually produced a committed or stageable change. A failing gate or an
    empty diff rejects the completion with feedback for the next iteration.
    """
    gates = list(quality_gates)

    def verify() -> CompletionVerdict:
        gate_failure = _run_quality_gates(worktree, gates)
        if gate_failure is not None:
            return CompletionVerdict(ok=False, feedback=gate_failure)
        if _diff_is_empty(worktree):
            return CompletionVerdict(ok=False, feedback=_NO_CHANGE_FEEDBACK)
        return CompletionVerdict(ok=True, feedback="")

    return verify


def _run_quality_gates(worktree: Path, gates: list[str]) -> str | None:
    """Re-run each gate in *worktree*, stopping at the first non-zero exit.

    Returns rejection feedback (failing command + output tail) on the first
    failure or timeout, or ``None`` when every gate passes.
    """
    for command in gates:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=worktree,
                capture_output=True,
                text=True,
                timeout=_GATE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"quality gate `{command}` timed out after {_GATE_TIMEOUT_SECONDS:.0f}s"
        if result.returncode != 0:
            tail = (result.stdout + result.stderr)[-_FEEDBACK_TAIL_CHARS:].strip()
            return f"quality gate `{command}` failed (exit {result.returncode}):\n{tail}"
    return None


def _resolve_feature_branch(worktree: Path) -> str | None:
    """Return the branch the *worktree* forked from — the main worktree's branch.

    `git worktree list --porcelain` lists the main worktree first; its
    ``branch refs/heads/<name>`` line names the feature branch this worktree
    was created from, the same ref the executor later rebases onto. Returns
    ``None`` when the main worktree is detached or git cannot be queried.
    """
    try:
        listing = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=True,
            timeout=_GATE_TIMEOUT_SECONDS,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        _logger.warning("completion verifier: could not list worktrees in %s: %s", worktree, exc)
        return None
    # The first record (up to the first blank line) is the main worktree.
    first_record = listing.split("\n\n", 1)[0]
    for line in first_record.splitlines():
        if line.startswith("branch refs/heads/"):
            return line[len("branch refs/heads/") :].strip()
    return None


def _diff_is_empty(worktree: Path) -> bool:
    """True when the branch has produced no committed or stageable change.

    Committed work is measured against the merge-base with the feature branch
    (the fork point); stageable work is any dirty or untracked file in the
    worktree. Either one present means the worker produced change.
    """
    if _has_working_tree_change(worktree):
        return False
    feature_branch = _resolve_feature_branch(worktree)
    if feature_branch is None:
        # Clean tree and no resolvable fork point: with no base to diff against
        # we cannot prove the branch produced committed change, so treat it as
        # empty (a dirty tree already returned False above).
        return True
    return not _has_committed_change(worktree, feature_branch)


def _has_working_tree_change(worktree: Path) -> bool:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=True,
            timeout=_GATE_TIMEOUT_SECONDS,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        _logger.warning("completion verifier: git status failed in %s: %s", worktree, exc)
        return False
    return bool(status.strip())


def _has_committed_change(worktree: Path, feature_branch: str) -> bool:
    try:
        base = subprocess.run(
            ["git", "merge-base", "HEAD", feature_branch],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=True,
            timeout=_GATE_TIMEOUT_SECONDS,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--quiet", base, "HEAD"],
            cwd=worktree,
            timeout=_GATE_TIMEOUT_SECONDS,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        _logger.warning(
            "completion verifier: could not diff against %s in %s: %s",
            feature_branch,
            worktree,
            exc,
        )
        return False
    # `git diff --quiet` exits 1 when the diff is non-empty, 0 when identical.
    return diff.returncode != 0


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
                local_manager.stop_run(run_id)
                return VerifySpecResult(outcome="gaps", goal_delta="verification timed out")
            try:
                event = ev_queue.get(timeout=remaining)
            except queue.Empty:
                local_manager.stop_run(run_id)
                return VerifySpecResult(outcome="gaps", goal_delta="verification timed out")
            if event.type in _ITERATION_EVENTS:
                text = event.data.get("result_text") or ""
                if text:
                    output_parts.append(text)
            elif event.type == EventType.RUN_STOPPED:
                break
    except Exception as exc:
        _logger.warning("verify_spec error: %s", exc)
        try:
            local_manager.stop_run(run_id)
        except Exception:
            pass
        return VerifySpecResult(outcome="done")
    return _parse_verify_output("\n".join(output_parts))


def _parse_verify_output(output: str) -> VerifySpecResult:
    import re

    if "<result>done</result>" in output:
        return VerifySpecResult(outcome="done")
    if "<result>gaps</result>" in output:
        m = re.search(r"<goal_delta>(.*?)</goal_delta>", output, re.DOTALL)
        delta = m.group(1).strip() if m else None
        return VerifySpecResult(outcome="gaps", goal_delta=delta)
    _logger.warning("verify_spec: unparseable output, treating as done")
    return VerifySpecResult(outcome="done")


def _build_ralph_content(
    node: MikadoNode,
    context: str,
    quality_gates: list[str],
) -> str:
    gates = "\n".join(f"- `{g}`" for g in quality_gates)
    return (
        f"# {node.description}\n\n"
        f"## Context\n\n{context}\n\n"
        f"## Quality Gates\n\n{gates}\n\n"
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
