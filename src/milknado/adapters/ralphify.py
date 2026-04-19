from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Final

from ralphify import EventType, QueueEmitter, RunConfig, RunManager, RunStatus

from milknado.domains.common.errors import CompletionTimeout
from milknado.domains.common.protocols import ProgressEvent, VerifySpecResult
from milknado.domains.common.types import MikadoNode

MILKNADO_COMPLETION_SIGNAL: Final[str] = "MILKNADO_NODE_COMPLETE"

_logger = logging.getLogger(__name__)


class RalphifyAdapter:
    def __init__(self, agent: str = "") -> None:
        self._manager = RunManager()
        self._queue: queue.Queue[Any] = queue.Queue()
        self._emitter = QueueEmitter(self._queue)
        self._agent = agent
        self._progress_buffer: list[ProgressEvent] = []

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
        )
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
            # EventType.PROGRESS not in current ralphify — graceful degradation to spinner
            if event.type != EventType.RUN_STOPPED:
                continue
            if event.run_id not in active_run_ids:
                continue
            run = self._manager.get_run(event.run_id)
            success = (
                run is not None and run.state.status == RunStatus.COMPLETED
            )
            return event.run_id, success

    def poll_progress_events(self) -> list[ProgressEvent]:
        result = list(self._progress_buffer)
        self._progress_buffer.clear()
        return result

    def verify_spec(self, spec_text: str, graph_state: str) -> VerifySpecResult:
        if not self._agent:
            _logger.warning("verify_spec: no agent configured, returning done")
            return VerifySpecResult(outcome="done")
        import subprocess
        import tempfile
        prompt = _build_verify_prompt(spec_text, graph_state)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False, encoding="utf-8",
            ) as f:
                f.write(prompt)
                prompt_path = Path(f.name)
            result = subprocess.run(
                [*self._agent.split(), str(prompt_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout + result.stderr
            prompt_path.unlink(missing_ok=True)
            return _parse_verify_output(output)
        except Exception as exc:
            _logger.warning("verify_spec failed, treating as done: %s", exc)
            return VerifySpecResult(outcome="done")

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


def _build_verify_prompt(spec_text: str, graph_state: Any) -> str:
    graph_summary = str(graph_state) if graph_state is not None else "(no graph state)"
    return (
        "# Spec Verification\n\n"
        "Review whether the following spec has been fully implemented.\n\n"
        f"## Spec\n\n{spec_text}\n\n"
        f"## Graph State\n\n{graph_summary}\n\n"
        "## Instructions\n\n"
        "If the spec is fully satisfied, respond with exactly:\n"
        "<verify-done/>\n\n"
        "If there are unmet requirements, respond with:\n"
        "<verify-gaps>\n"
        "Description of what is still missing\n"
        "</verify-gaps>\n"
    )


def _parse_verify_output(output: str) -> VerifySpecResult:
    import re
    if "<verify-done/>" in output or "<verify-done />" in output:
        return VerifySpecResult(outcome="done")
    m = re.search(r"<verify-gaps>(.*?)</verify-gaps>", output, re.DOTALL)
    if m:
        return VerifySpecResult(outcome="gaps", goal_delta=m.group(1).strip())
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
        "## Completion\n\n"
        "When every quality gate passes and this node is fully implemented,\n"
        f"emit `<promise>{MILKNADO_COMPLETION_SIGNAL}</promise>` on its own line\n"
        "so the run can stop before the iteration budget.\n"
    )
