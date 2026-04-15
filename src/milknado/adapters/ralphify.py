from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

from ralphify import EventType, QueueEmitter, RunConfig, RunManager, RunStatus

from milknado.domains.common.types import MikadoNode


class RalphifyAdapter:
    def __init__(self) -> None:
        self._manager = RunManager()
        self._queue: queue.Queue[Any] = queue.Queue()
        self._emitter = QueueEmitter(self._queue)

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: list[str],
    ) -> Any:
        config = RunConfig(
            agent=agent,
            ralph_dir=ralph_dir,
            ralph_file=ralph_file,
            project_root=ralph_dir,
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
        self, active_run_ids: set[str],
    ) -> tuple[str, bool]:
        while True:
            event = self._queue.get()
            if event.type != EventType.RUN_STOPPED:
                continue
            if event.run_id not in active_run_ids:
                continue
            run = self._manager.get_run(event.run_id)
            success = (
                run is not None and run.state.status == RunStatus.COMPLETED
            )
            return event.run_id, success

    def generate_ralph_md(
        self,
        node: MikadoNode,
        context: str,
        quality_gates: list[str],
        output_path: Path,
    ) -> Path:
        content = _build_ralph_content(node, context, quality_gates)
        output_path.write_text(content)
        return output_path


def _build_ralph_content(
    node: MikadoNode,
    context: str,
    quality_gates: list[str],
) -> str:
    gates = "\n".join(f"- `{g}`" for g in quality_gates)
    return (
        f"# {node.description}\n\n"
        f"## Context\n\n{context}\n\n"
        f"## Quality Gates\n\n{gates}\n"
    )
