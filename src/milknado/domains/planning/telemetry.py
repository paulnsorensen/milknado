"""Append-only JSONL telemetry sink for batch planning snapshots (A5)."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from milknado.domains.batching import BatchPlan
from milknado.domains.planning.manifest import PlanChangeManifest

_US_PATTERN = re.compile(r"\bUS-\d+\b")

_logger = logging.getLogger(__name__)

_CALIBRATION_FILE = Path(".milknado") / "calibration.jsonl"


def record_batch_snapshot(
    project_root: Path,
    manifest: PlanChangeManifest,
    plan: BatchPlan,
    *,
    plan_mode_used: str = "fresh",
    existing_node_count_at_plan_start: int = 0,
    spec_hash_changed: bool | None = None,
    orphaned_worktree_count: int = 0,
    mega_batch_aborted: bool = False,
    mega_batch_threshold: int = 5,
    verify_rounds_used: int = 0,
    verify_round_cap_hit: bool = False,
    coverage_passes_used: int = 0,
    coverage_gaps_remaining: int = 0,
    coverage_us_refs_uncovered: list[str] | None = None,
    coverage_orphan_impl_changes: list[str] | None = None,
) -> None:
    """Append one JSONL record to <project_root>/.milknado/calibration.jsonl.

    Uses a single atomic os.write() to an O_APPEND fd so concurrent appends
    from multiple processes/threads don't interleave within a record.

    Never raises — telemetry must not block the planner.
    """
    record = _build_record(
        manifest,
        plan,
        plan_mode_used=plan_mode_used,
        existing_node_count_at_plan_start=existing_node_count_at_plan_start,
        spec_hash_changed=spec_hash_changed,
        orphaned_worktree_count=orphaned_worktree_count,
        mega_batch_aborted=mega_batch_aborted,
        mega_batch_threshold=mega_batch_threshold,
        verify_rounds_used=verify_rounds_used,
        verify_round_cap_hit=verify_round_cap_hit,
        coverage_passes_used=coverage_passes_used,
        coverage_gaps_remaining=coverage_gaps_remaining,
        coverage_us_refs_uncovered=coverage_us_refs_uncovered,
        coverage_orphan_impl_changes=coverage_orphan_impl_changes,
    )
    dest = project_root / _CALIBRATION_FILE
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(record) + "\n").encode("utf-8")
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except OSError as exc:
        _logger.warning("telemetry write failed, skipping: %s", exc)


def _build_record(
    manifest: PlanChangeManifest,
    plan: BatchPlan,
    *,
    plan_mode_used: str = "fresh",
    existing_node_count_at_plan_start: int = 0,
    spec_hash_changed: bool | None = None,
    orphaned_worktree_count: int = 0,
    mega_batch_aborted: bool = False,
    mega_batch_threshold: int = 5,
    verify_rounds_used: int = 0,
    verify_round_cap_hit: bool = False,
    coverage_passes_used: int = 0,
    coverage_gaps_remaining: int = 0,
    coverage_us_refs_uncovered: list[str] | None = None,
    coverage_orphan_impl_changes: list[str] | None = None,
) -> dict[str, object]:
    spreads = [s.spread for s in plan.spread_report]
    max_spread = max(spreads) if spreads else 0
    mean_spread = sum(spreads) / len(spreads) if spreads else 0.0
    us_counts = [len(_US_PATTERN.findall(c.description)) for c in manifest.changes]
    test_count = sum(
        1
        for c in manifest.changes
        if c.path.startswith("tests/")
        or c.path.startswith("test_")
        or "/test_" in c.path
        or "/tests/" in c.path
    )
    impl_count = len(manifest.changes) - test_count
    test_to_impl = (test_count / impl_count) if impl_count > 0 else 0.0
    return {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "change_count": len(manifest.changes),
        "batch_count": len(plan.batches),
        "oversized_count": sum(1 for b in plan.batches if b.oversized),
        "solver_status": plan.solver_status,
        "max_spread": max_spread,
        "mean_spread": mean_spread,
        "new_relationship_count": len(manifest.new_relationships),
        "spec_path": manifest.spec_path,
        "test_change_count": test_count,
        "impl_change_count": impl_count,
        "test_to_impl_ratio": test_to_impl,
        "max_us_refs_per_change": max(us_counts) if us_counts else 0,
        "multi_story_change_count": sum(1 for n in us_counts if n > 1),
        "distinct_path_count": len({c.path for c in manifest.changes}),
        "plan_mode_used": plan_mode_used,
        "existing_node_count_at_plan_start": existing_node_count_at_plan_start,
        "spec_hash_changed": spec_hash_changed,
        "orphaned_worktree_count": orphaned_worktree_count,
        "mega_batch_aborted": mega_batch_aborted,
        "mega_batch_threshold": mega_batch_threshold,
        "verify_rounds_used": verify_rounds_used,
        "verify_round_cap_hit": verify_round_cap_hit,
        "coverage_passes_used": coverage_passes_used,
        "coverage_gaps_remaining": coverage_gaps_remaining,
        "coverage_us_refs_uncovered": coverage_us_refs_uncovered or [],
        "coverage_orphan_impl_changes": coverage_orphan_impl_changes or [],
    }
