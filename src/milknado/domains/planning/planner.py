from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.adapters.tilth import TilthAdapter
from milknado.domains.common.agent_argv import build_planning_subprocess
from milknado.domains.common.errors import (
    InsufficientTestCoverageError,
    MegaBatchAborted,
    MultiStoryBundlingError,
    PlanningFailed,
)
from milknado.domains.planning._manifest_quality import (
    append_reuse_candidates,
    get_bundled_changes,
    summarise_manifest_quality,
)
from milknado.domains.planning._plan_helpers import (
    build_coverage_delta,
    count_active_worktrees,
    detect_spec_hash_change,
    guard_existing_plan,
    hash_spec,
    read_spec,
    resolve_plan_mode,
    safe_ensure_crg,
)
from milknado.domains.planning.batching_bridge import (
    apply_batches_to_graph,
    run_batching,
)
from milknado.domains.planning.context import build_planning_context
from milknado.domains.planning.coverage import coverage_check
from milknado.domains.planning.manifest import PlanChangeManifest, parse_manifest_from_output
from milknado.domains.planning.telemetry import record_batch_snapshot

if TYPE_CHECKING:
    from milknado.domains.batching.change import BatchPlan
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
    verify_rounds_used: int = 0
    verify_round_cap_hit: bool = False
    coverage_gaps_remaining: int = 0


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
        max_verify_rounds: int = 3,
    ) -> PlanResult:
        return self.launch(
            delta,
            project_root,
            spec_path=spec_path,
            resuming=True,
            max_verify_rounds=max_verify_rounds,
        )

    def launch(
        self,
        goal: str,
        project_root: Path,
        *,
        spec_path: Path | None = None,
        max_verify_rounds: int = 3,
        resuming: bool = False,
        reset: bool = False,
        force_single_batch: bool = False,
        mega_batch_threshold: int = 5,
    ) -> PlanResult:
        spec_text = read_spec(spec_path)
        existing_node_count_at_plan_start = len(self._graph.get_all_nodes())
        guard_existing_plan(
            self._graph,
            resuming=resuming,
            reset=reset,
            project_root=project_root,
            spec_text=spec_text,
        )
        plan_mode_used = resolve_plan_mode(existing_node_count_at_plan_start, reset=reset)
        spec_hash_changed = detect_spec_hash_change(
            self._graph,
            spec_text,
            resuming=resuming,
        )
        orphaned_worktree_count = count_active_worktrees(project_root)
        crg, crg_ok = safe_ensure_crg(self._crg, project_root)
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

        argv, extra = build_planning_subprocess(context_path,
            self._planning_agent)
        extra["stdout"] = subprocess.PIPE
        extra["stderr"] = subprocess.PIPE
        result = subprocess.run(argv, cwd=project_root, check=False, **extra)
        checked = _check_planner_exit(result, context_path)
        if isinstance(checked, PlanResult):
            return checked
        manifest: PlanChangeManifest = checked

        verify_rounds_used = 1
        verify_round_cap_hit = False
        gaps = coverage_check(manifest)
        while gaps and verify_rounds_used < max_verify_rounds:
            delta_path = milknado_dir / f"coverage-delta-{verify_rounds_used}.md"
            delta_path.write_text(build_coverage_delta(gaps, context), encoding="utf-8")
            argv, extra = build_planning_subprocess(delta_path, self._planning_agent)
            extra["stdout"] = subprocess.PIPE
            extra["stderr"] = subprocess.PIPE
            re_result = subprocess.run(argv, cwd=project_root, check=False, **extra)
            re_checked = _check_planner_exit(re_result, context_path)
            verify_rounds_used += 1
            if isinstance(re_checked, PlanResult):
                break
            manifest = re_checked
            gaps = coverage_check(manifest)
        if gaps:
            if verify_rounds_used >= max_verify_rounds:
                verify_round_cap_hit = True
                _logger.warning(
                    "WARNING: verify-spec round cap hit (%d rounds)", verify_rounds_used
                )
            _logger.warning(
                "WARNING: %d impl changes lack test coverage: %s",
                len(gaps),
                ", ".join(gaps),
            )
        coverage_gaps_remaining = len(gaps)

        bundled = get_bundled_changes(manifest)
        if bundled:
            raise MultiStoryBundlingError(bundled)

        impl_count = sum(1 for c in manifest.changes if c.path.startswith("src/"))
        test_count = sum(1 for c in manifest.changes if c.path.startswith("tests/"))
        if gaps and impl_count > 0 and test_count / impl_count < 0.5:
            raise InsufficientTestCoverageError(list(gaps))

        manifest = append_reuse_candidates(manifest, self._crg if crg_ok else None)
        plan = run_batching(manifest, crg if crg_ok else None, project_root)
        _check_mega_batch(
            plan,
            force_single_batch=force_single_batch,
            threshold=mega_batch_threshold,
        )
        existing_root = self._graph.get_root()
        parent_id = existing_root.id if existing_root is not None else None
        created_ids = apply_batches_to_graph(self._graph,
            plan,
            manifest,
            parent_id=parent_id)
        record_batch_snapshot(
            project_root,
            manifest,
            plan,
            plan_mode_used=plan_mode_used,
            existing_node_count_at_plan_start=existing_node_count_at_plan_start,
            spec_hash_changed=spec_hash_changed,
            orphaned_worktree_count=orphaned_worktree_count,
            verify_rounds_used=verify_rounds_used,
            verify_round_cap_hit=verify_round_cap_hit,
            coverage_gaps_remaining=coverage_gaps_remaining,
            coverage_orphan_impl_changes=list(gaps),
        )

        if spec_text is not None:
            self._graph.set_spec_hash(hash_spec(spec_text))

        quality = summarise_manifest_quality(manifest)
        return PlanResult(
            success=True,
            exit_code=0,
            context_path=context_path,
            nodes_created=len(created_ids),
            batch_count=len(plan.batches),
            oversized_count=sum(1 for b in plan.batches if b.oversized),
            solver_status=plan.solver_status,
            change_count=len(manifest.changes),
            verify_rounds_used=verify_rounds_used,
            verify_round_cap_hit=verify_round_cap_hit,
            coverage_gaps_remaining=coverage_gaps_remaining,
            **quality,
        )


def _check_mega_batch(
    plan: BatchPlan,
    *,
    force_single_batch: bool,
    threshold: int,
) -> None:
    if (
        len(plan.batches) == 1
        and len(plan.batches[0].change_ids) > threshold
        and not force_single_batch
    ):
        raise MegaBatchAborted(len(plan.batches[0].change_ids), threshold)


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
