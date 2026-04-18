from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.adapters.tilth import TilthAdapter
from milknado.domains.common.agent_argv import build_planning_subprocess
from milknado.domains.common.types import DegradationMarker
from milknado.domains.planning.budget import validate_manifest_budgets
from milknado.domains.planning.context import build_planning_context
from milknado.domains.planning.manifest import (
    apply_manifest_to_graph,
    parse_manifest_from_output,
)
from milknado.domains.planning.telemetry import record_budget_snapshot

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort, TilthPort
    from milknado.domains.graph import MikadoGraph


@dataclass(frozen=True)
class PlanResult:
    success: bool
    exit_code: int
    context_path: Path | None = None
    nodes_created: int = 0
    budget_violations: int = 0


class Planner:
    def __init__(
        self,
        graph: MikadoGraph,
        crg: CrgPort,
        planning_agent: str,
        tilth: TilthPort | None = None,
    ) -> None:
        self._graph = graph
        self._crg = crg
        self._planning_agent = planning_agent
        self._tilth = tilth if tilth is not None else TilthAdapter()

    def launch(
        self,
        spec_path: Path,
        project_root: Path,
        *,
        execution_agent: str,
        allow_external_mcp: bool = False,
    ) -> PlanResult:
        crg_degradation = self._ensure_crg_graph(project_root)
        context = build_planning_context(
            spec_path=spec_path,
            crg=self._crg,
            graph=self._graph,
            execution_agent=execution_agent,
            tilth=self._tilth,
            project_root=project_root,
            crg_degradation=crg_degradation,
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
        budget_violations = 0
        if exit_code == 0:
            manifest = parse_manifest_from_output(output)
            if manifest is not None:
                violations = validate_manifest_budgets(manifest)
                record_budget_snapshot(project_root, manifest, violations)
                budget_violations = len(violations)
                created_ids = apply_manifest_to_graph(self._graph, manifest)
                nodes_created = len(created_ids)
        return PlanResult(
            success=exit_code == 0,
            exit_code=exit_code,
            context_path=context_path,
            nodes_created=nodes_created,
            budget_violations=budget_violations,
        )

    def _ensure_crg_graph(self, project_root: Path) -> DegradationMarker | None:
        try:
            self._crg.ensure_graph(project_root)
        except Exception as exc:  # noqa: BLE001 - defensive: any graph build error degrades
            return DegradationMarker(
                source="crg",
                reason="ensure_graph_failed",
                detail=str(exc)[:500],
            )
        return None
