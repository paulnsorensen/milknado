from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.agent_argv import build_planning_subprocess
from milknado.domains.planning.context import build_planning_context
from milknado.domains.planning.manifest import (
    apply_manifest_to_graph,
    parse_manifest_from_output,
)

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.graph import MikadoGraph


@dataclass(frozen=True)
class PlanResult:
    success: bool
    exit_code: int
    context_path: Path | None = None
    nodes_created: int = 0


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

    def launch(
        self,
        spec_path: Path,
        project_root: Path,
        *,
        execution_agent: str,
        allow_external_mcp: bool = False,
    ) -> PlanResult:
        context = build_planning_context(
            spec_path=spec_path,
            crg=self._crg,
            graph=self._graph,
            execution_agent=execution_agent,
        )
        milknado_dir = project_root / ".milknado"
        milknado_dir.mkdir(parents=True, exist_ok=True)
        context_path = milknado_dir / "planning-context.md"
        context_path.write_text(context, encoding="utf-8")

        argv, extra = build_planning_subprocess(
            context_path, self._planning_agent, allow_external_mcp=allow_external_mcp,
        )
        result = subprocess.run(
            argv,
            cwd=project_root,
            check=False,
            capture_output=True,
            **extra,
        )
        exit_code = result.returncode
        output = result.stdout if isinstance(result.stdout, str) else ""
        nodes_created = 0
        if exit_code == 0:
            manifest = parse_manifest_from_output(output)
            if manifest is not None:
                created_ids = apply_manifest_to_graph(self._graph, manifest)
                nodes_created = len(created_ids)
        return PlanResult(
            success=exit_code == 0,
            exit_code=exit_code,
            context_path=context_path,
            nodes_created=nodes_created,
        )
