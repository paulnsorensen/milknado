from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.planning.batching_bridge import (
    apply_batches_to_graph,
    run_batching,
)
from milknado.domains.planning.context import build_planning_context
from milknado.domains.planning.manifest import (
    PlanChangeManifest,
    manifest_to_dict,
    parse_manifest_from_output,
)
from milknado.domains.planning.ports import PlanningPorts
from milknado.domains.planning.telemetry import record_batch_snapshot

if TYPE_CHECKING:
    from milknado.domains.batching import BatchPlan
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
    mega_batch_change_count: int | None = None


class Planner:
    def __init__(
        self,
        graph: MikadoGraph,
        crg: CrgPort,
        planning_agent: str,
        ports: PlanningPorts,
        planning_validation_hook: str | None = None,
        *,
        prompt_prepend: str | None = None,
    ) -> None:
        self._graph = graph
        self._crg = crg
        self._ports = ports
        self._planning_agent = planning_agent
        self._planning_validation_hook = (planning_validation_hook or "").strip() or None
        self._prompt_prepend = prompt_prepend

    def launch(
        self,
        goal: str,
        project_root: Path,
        *,
        spec_path: Path | None = None,
    ) -> PlanResult:
        spec_text = _read_spec(spec_path)
        crg, crg_ok = _safe_ensure_crg(self._crg, project_root)
        context_path = self._write_context(
            goal,
            project_root,
            crg if crg_ok else None,
            spec_text,
        )
        process = self._ports.process.run_agent(
            context_path,
            self._planning_agent,
            project_root,
        )
        manifest = parse_manifest_from_output(process.stdout)
        if manifest is None:
            return PlanResult(
                success=process.exit_code == 0,
                exit_code=process.exit_code,
                context_path=context_path,
                solver_status="NO_MANIFEST",
            )
        validation_error = self._run_validation_hook(manifest, project_root, context_path)
        if validation_error:
            _logger.warning("planner validation hook rejected manifest: %s", validation_error)
            return PlanResult(
                success=False,
                exit_code=1,
                context_path=context_path,
                solver_status="VALIDATION_FAILED",
                change_count=len(manifest.changes),
            )
        plan, created_count = self._apply_manifest(
            manifest,
            project_root,
            crg if crg_ok else None,
        )
        return PlanResult(
            success=process.exit_code == 0,
            exit_code=process.exit_code,
            context_path=context_path,
            nodes_created=created_count,
            batch_count=len(plan.batches),
            oversized_count=sum(1 for batch in plan.batches if batch.oversized),
            solver_status=plan.solver_status,
            change_count=len(manifest.changes),
            mega_batch_change_count=plan.mega_batch_change_count,
        )

    def _write_context(
        self,
        goal: str,
        project_root: Path,
        crg: CrgPort | None,
        spec_text: str | None,
    ) -> Path:
        context = build_planning_context(
            goal,
            crg,
            self._graph,
            spec_text=spec_text,
            prepend=self._prompt_prepend,
        )
        context_path = project_root / ".milknado" / "planning-context.md"
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_path.write_text(context, encoding="utf-8")
        return context_path

    def _apply_manifest(
        self,
        manifest: PlanChangeManifest,
        project_root: Path,
        crg: CrgPort | None,
    ) -> tuple[BatchPlan, int]:
        plan = run_batching(manifest, crg, project_root)
        existing_root = self._graph.get_root()
        parent_id = existing_root.id if existing_root is not None else None
        created = apply_batches_to_graph(
            self._graph,
            plan,
            manifest,
            parent_id=parent_id,
        )
        record_batch_snapshot(project_root, manifest, plan)
        return plan, len(created)

    def replan_with_delta(
        self,
        goal_delta: str,
        project_root: Path,
        spec_path: Path | None = None,
    ) -> PlanResult:
        return self.launch(goal_delta, project_root, spec_path=spec_path)

    def _run_validation_hook(
        self,
        manifest: PlanChangeManifest,
        project_root: Path,
        context_path: Path,
    ) -> str | None:
        if self._planning_validation_hook is None:
            return None
        payload = {
            "manifest": manifest_to_dict(manifest),
            "project_root": str(project_root),
            "context_path": str(context_path),
        }
        result = self._ports.process.run_validation(
            self._planning_validation_hook,
            payload,
            project_root,
        )
        if result.exit_code == 0:
            return None
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        return stderr or stdout or f"exit {result.exit_code}"


def _read_spec(spec_path: Path | None) -> str | None:
    if spec_path is None:
        return None
    if not spec_path.exists():
        raise FileNotFoundError(f"spec_path does not exist: {spec_path}")
    if not spec_path.is_file():
        raise ValueError(f"spec_path is not a file: {spec_path}")
    return spec_path.read_text(encoding="utf-8")


def _safe_ensure_crg(
    crg: CrgPort,
    project_root: Path,
) -> tuple[CrgPort, bool]:
    try:
        crg.ensure_graph(project_root)
        return crg, True
    except Exception as exc:
        _logger.warning("CRG unavailable, running without graph context: %s", exc)
        return crg, False
