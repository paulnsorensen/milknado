from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
    ) -> None:
        self._graph = graph
        self._crg = crg
        self._agent_command = agent_command

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
        path.write_text(context)
        return path

    def _run_agent(
        self, context_path: Path, project_root: Path
    ) -> int:
        cmd = self._build_command(context_path)
        result = subprocess.run(
            cmd, cwd=project_root, check=False
        )
        return result.returncode

    def _build_command(self, context_path: Path) -> list[str]:
        parts = self._agent_command.split()
        parts.extend(["--print", str(context_path)])
        return parts
