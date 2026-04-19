from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.adapters.tilth import TilthAdapter
from milknado.domains.common.agent_argv import build_planning_subprocess
from milknado.domains.common.errors import PlanningFailed
from milknado.domains.planning.batching_bridge import (
    apply_batches_to_graph,
    run_batching,
)
from milknado.domains.planning.context import build_planning_context
from milknado.domains.planning.manifest import PlanChangeManifest, parse_manifest_from_output
from milknado.domains.planning.telemetry import record_batch_snapshot

_US_REF_RE = re.compile(r"\bUS-\d{3}\b")

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
    impl_change_count: int = 0
    test_change_count: int = 0
    multi_story_change_count: int = 0
    max_us_refs_per_change: int = 0
    distinct_path_count: int = 0


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

    def replan_with_delta(
        self,
        delta: str,
        project_root: Path,
        spec_path: Path | None,
    ) -> PlanResult:
        return self.launch(delta, project_root, spec_path=spec_path)

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
        extra["stderr"] = subprocess.PIPE
        result = subprocess.run(argv, cwd=project_root, check=False, **extra)
        checked = _check_planner_exit(result, context_path)
        if isinstance(checked, PlanResult):
            return checked
        manifest: PlanChangeManifest = checked

        manifest = _append_reuse_candidates(manifest, self._crg if crg_ok else None)
        plan = run_batching(manifest, crg if crg_ok else None, project_root)
        existing_root = self._graph.get_root()
        parent_id = existing_root.id if existing_root is not None else None
        created_ids = apply_batches_to_graph(
            self._graph, plan, manifest, parent_id=parent_id,
        )
        record_batch_snapshot(project_root, manifest, plan)

        quality = _summarise_manifest_quality(manifest)
        return PlanResult(
            success=True,
            exit_code=0,
            context_path=context_path,
            nodes_created=len(created_ids),
            batch_count=len(plan.batches),
            oversized_count=sum(1 for b in plan.batches if b.oversized),
            solver_status=plan.solver_status,
            change_count=len(manifest.changes),
            **quality,
        )


def _check_planner_exit(
    result: subprocess.CompletedProcess[str],
    context_path: Path,
) -> PlanChangeManifest | PlanResult:
    if result.returncode != 0:
        raise PlanningFailed(exit_code=result.returncode, stderr=result.stderr or "")
    manifest = parse_manifest_from_output(result.stdout or "")
    if manifest is None:
        return PlanResult(
            success=True, exit_code=0, context_path=context_path, solver_status="NO_MANIFEST"
        )
    return manifest


def _append_reuse_candidates(
    manifest: PlanChangeManifest,
    crg: CrgPort | None,
) -> PlanChangeManifest:
    if crg is None:
        return manifest
    from milknado.domains.batching import FileChange
    updated: list[FileChange] = []
    for change in manifest.changes:
        query = Path(change.path).stem
        try:
            hits = crg.semantic_search_nodes(query, top_n=5)
        except Exception as exc:
            _logger.warning("CRG reuse search failed for %s: %s", change.path, exc)
            hits = []
        if not hits:
            updated.append(change)
            continue
        lines = ["\n\n## Reuse candidates"]
        for h in hits:
            sym = h.get("symbol_name", "?")
            fpath = h.get("file_path", "?")
            summary = h.get("summary", "")
            lines.append(f"- `{sym}` ({fpath}): {summary}")
        updated.append(replace(change, description=change.description + "\n".join(lines)))
    return replace(manifest, changes=tuple(updated))


def _summarise_manifest_quality(manifest: PlanChangeManifest) -> dict[str, int]:
    if not manifest.changes:
        return {
            "impl_change_count": 0,
            "test_change_count": 0,
            "multi_story_change_count": 0,
            "max_us_refs_per_change": 0,
            "distinct_path_count": 0,
        }
    impl = sum(1 for c in manifest.changes if c.path.startswith("src/"))
    tests = sum(1 for c in manifest.changes if c.path.startswith("tests/"))
    us_per = [
        len(set(_US_REF_RE.findall(c.description or "")))
        for c in manifest.changes
    ]
    multi = sum(1 for n in us_per if n >= 2)
    return {
        "impl_change_count": impl,
        "test_change_count": tests,
        "multi_story_change_count": multi,
        "max_us_refs_per_change": max(us_per) if us_per else 0,
        "distinct_path_count": len({c.path for c in manifest.changes}),
    }


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
