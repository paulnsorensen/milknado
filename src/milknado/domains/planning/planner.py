from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.agent_argv import build_planning_subprocess
from milknado.domains.planning.context import build_planning_context

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.graph.graph import MikadoGraph


@dataclass(frozen=True)
class PlanResult:
    success: bool
    exit_code: int
    context_path: Path | None = None


class Planner:
    def __init__(
        self,
        graph: MikadoGraph,
        crg: CrgPort,
        agent_command: str = "claude",
        *,
        agent_preset: str = "custom",
    ) -> None:
        self._graph = graph
        self._crg = crg
        self._agent_command = agent_command
        self._agent_preset = agent_preset

    def build_context(self, goal: str) -> str:
        return build_planning_context(goal, self._crg, self._graph)

    def launch(self, goal: str, project_root: Path) -> PlanResult:
        context = self.build_context(goal)
        context_path = self._write_context_file(
            context, project_root
        )
        exit_code = self._run_agent(context_path, project_root)
        return PlanResult(
            success=exit_code == 0,
            exit_code=exit_code,
            context_path=context_path,
        )

    def _write_context_file(
        self, context: str, project_root: Path
    ) -> Path:
        milknado_dir = project_root / ".milknado"
        milknado_dir.mkdir(parents=True, exist_ok=True)
        path = milknado_dir / "planning-context.md"
        path.write_text(context, encoding="utf-8")
        return path

    def _run_agent(
        self, context_path: Path, project_root: Path
    ) -> int:
        argv, extra = build_planning_subprocess(
            context_path, self._agent_preset, self._agent_command,
        )
        result = subprocess.run(argv, cwd=project_root, check=False, **extra)
        return result.returncode
