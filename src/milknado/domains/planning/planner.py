from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.adapters.tilth import TilthAdapter
from milknado.domains.common.agent_argv import build_planning_subprocess
from milknado.domains.planning.batching_bridge import (
    apply_batches_to_graph,
    run_batching,
)
from milknado.domains.planning.context import build_planning_context
from milknado.domains.planning.manifest import parse_manifest_from_output
from milknado.domains.planning.telemetry import record_batch_snapshot

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanResult:
    success: bool
    exit_code: int
    context_path: Path | None = None
    nodes_created: int = 0
    batch_count: int = 0
    oversized_count: int = 0
    solver_status: str = ""
    change_count: int = 0


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
        goal: str,
        project_root: Path,
        *,
        spec_path: Path | None = None,
    ) -> PlanResult:
        spec_text = _read_spec(spec_path)
        crg, crg_ok = _safe_ensure_crg(self._crg, project_root)
        tilth = TilthAdapter()
        context = build_planning_context(
            goal,
            crg if crg_ok else None,
            self._graph,
            spec_text=spec_text,
            tilth=tilth,
            scope=project_root,
        )
        milknado_dir = project_root / ".milknado"
        milknado_dir.mkdir(parents=True, exist_ok=True)
        context_path = milknado_dir / "planning-context.md"
        context_path.write_text(context, encoding="utf-8")

        argv, extra = build_planning_subprocess(
            context_path, self._planning_agent,
        )
        extra["stdout"] = subprocess.PIPE
        result = subprocess.run(argv, cwd=project_root, check=False, **extra)
        exit_code = result.returncode
        stdout = result.stdout or ""

        manifest = parse_manifest_from_output(stdout)
        if manifest is None:
            return PlanResult(
                success=exit_code == 0,
                exit_code=exit_code,
                context_path=context_path,
                nodes_created=0,
                batch_count=0,
                oversized_count=0,
                solver_status="NO_MANIFEST",
                change_count=0,
            )

        plan = run_batching(manifest, crg if crg_ok else None, project_root)
        created_ids = apply_batches_to_graph(
            self._graph, plan, manifest, parent_id=None,
        )
        record_batch_snapshot(project_root, manifest, plan)

        return PlanResult(
            success=exit_code == 0,
            exit_code=exit_code,
            context_path=context_path,
            nodes_created=len(created_ids),
            batch_count=len(plan.batches),
            oversized_count=sum(1 for b in plan.batches if b.oversized),
            solver_status=plan.solver_status,
            change_count=len(manifest.changes),
        )


def _read_spec(spec_path: Path | None) -> str | None:
    if spec_path is None:
        return None
    if not spec_path.exists():
        raise FileNotFoundError(f"spec_path does not exist: {spec_path}")
    if not spec_path.is_file():
        raise ValueError(f"spec_path is not a file: {spec_path}")
    return spec_path.read_text(encoding="utf-8")


def _safe_ensure_crg(
    crg: CrgPort, project_root: Path,
) -> tuple[CrgPort, bool]:
    try:
        crg.ensure_graph(project_root)
        return crg, True
    except Exception as exc:
        _logger.warning("CRG unavailable, running without graph context: %s", exc)
        return crg, False
