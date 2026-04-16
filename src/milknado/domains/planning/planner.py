from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.agent_argv import build_planning_subprocess
from milknado.domains.planning.context import build_planning_context

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.graph import MikadoGraph


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
        planning_agent: str,
    ) -> None:
        self._graph = graph
        self._crg = crg
        self._planning_agent = planning_agent

    def launch(self, goal: str, project_root: Path) -> PlanResult:
        context = build_planning_context(goal, self._crg, self._graph)
        milknado_dir = project_root / ".milknado"
        milknado_dir.mkdir(parents=True, exist_ok=True)
        context_path = milknado_dir / "planning-context.md"
        context_path.write_text(context, encoding="utf-8")

        argv, extra = build_planning_subprocess(
            context_path, self._planning_agent,
        )
        result = subprocess.run(argv, cwd=project_root, check=False, **extra)
        exit_code = result.returncode
        return PlanResult(
            success=exit_code == 0,
            exit_code=exit_code,
            context_path=context_path,
        )
