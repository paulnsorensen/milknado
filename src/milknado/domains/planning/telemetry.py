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
) -> None:
    """Append one JSONL record to <project_root>/.milknado/calibration.jsonl.

    Uses a single atomic os.write() to an O_APPEND fd so concurrent appends
    from multiple processes/threads don't interleave within a record.

    Never raises — telemetry must not block the planner.
    """
    record = _build_record(manifest, plan)
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


def _build_record(manifest: PlanChangeManifest, plan: BatchPlan) -> dict[str, object]:
    spreads = [s.spread for s in plan.spread_report]
    max_spread = max(spreads) if spreads else 0
    mean_spread = sum(spreads) / len(spreads) if spreads else 0.0
    us_counts = [len(_US_PATTERN.findall(c.description)) for c in manifest.changes]
    test_count = sum(
        1 for c in manifest.changes
        if c.path.startswith("tests/") or c.path.startswith("test_")
        or "/test_" in c.path or "/tests/" in c.path
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
    }
