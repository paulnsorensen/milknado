"""Tests for the calibration telemetry sink (A5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from milknado.domains.batching.change import (
    Batch,
    BatchPlan,
    FileChange,
    NewRelationship,
    SymbolRef,
    SymbolSpread,
)
from milknado.domains.planning.manifest import PlanChangeManifest
from milknado.domains.planning.telemetry import record_batch_snapshot

MANIFEST_VERSION = "milknado.plan.v2"


def _make_manifest(
    *,
    changes: tuple[FileChange, ...] = (),
    new_relationships: tuple[NewRelationship, ...] = (),
) -> PlanChangeManifest:
    return PlanChangeManifest(
        manifest_version=MANIFEST_VERSION,
        goal="Test goal",
        goal_summary="Test summary",
        spec_path=None,
        changes=changes,
        new_relationships=new_relationships,
    )


def _make_plan(
    *,
    batches: tuple[Batch, ...] = (),
    spread_report: tuple[SymbolSpread, ...] = (),
    solver_status: str = "OPTIMAL",
) -> BatchPlan:
    return BatchPlan(
        batches=batches,
        spread_report=spread_report,
        solver_status=solver_status,  # type: ignore
    )


def _make_change(cid: str, path: str = "src/foo.py") -> FileChange:
    return FileChange(id=cid, path=path, description="Some change")


def _make_batch(index: int, *, oversized: bool = False) -> Batch:
    return Batch(index=index, change_ids=("c1",), depends_on=(), oversized=oversized)


class TestHappyPath:
    def test_writes_single_record(self, tmp_path: Path) -> None:
        """One call → exactly one JSONL line."""
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(_make_batch(0),))
        record_batch_snapshot(tmp_path, manifest, plan)

        jsonl = (tmp_path / ".milknado" / "calibration.jsonl").read_text()
        lines = [ln for ln in jsonl.splitlines() if ln.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert isinstance(record, dict)

    def test_record_fields_present(self, tmp_path: Path) -> None:
        """All required fields appear in the written record."""
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(_make_batch(0),), solver_status="FEASIBLE")
        record_batch_snapshot(tmp_path, manifest, plan)

        line = (tmp_path / ".milknado" / "calibration.jsonl").read_text().strip()
        record = json.loads(line)
        assert "timestamp" in record
        assert "change_count" in record
        assert "batch_count" in record
        assert "oversized_count" in record
        assert "solver_status" in record
        assert "max_spread" in record
        assert "mean_spread" in record
        assert "new_relationship_count" in record
        assert "spec_path" in record
        assert "test_change_count" in record
        assert "impl_change_count" in record
        assert "test_to_impl_ratio" in record
        assert "max_us_refs_per_change" in record
        assert "multi_story_change_count" in record
        assert "distinct_path_count" in record


class TestManifestQualitySignals:
    def test_test_to_impl_ratio_split(self, tmp_path: Path) -> None:
        """Mixed src/ and tests/ paths produce correct test_to_impl_ratio."""
        changes = (
            FileChange(id="c1", path="src/foo.py", description="impl"),
            FileChange(id="c2", path="src/bar.py", description="impl"),
            FileChange(id="c3", path="tests/test_foo.py", description="test"),
        )
        manifest = _make_manifest(changes=changes)
        plan = _make_plan(batches=(_make_batch(0),))
        record_batch_snapshot(tmp_path, manifest, plan)
        record = json.loads((tmp_path / ".milknado" / "calibration.jsonl").read_text().strip())
        assert record["impl_change_count"] == 2
        assert record["test_change_count"] == 1
        assert record["test_to_impl_ratio"] == 0.5

    def test_us_refs_in_description(self, tmp_path: Path) -> None:
        """max_us_refs_per_change and multi_story_change_count count distinct US-NNN tokens."""
        changes = (
            FileChange(
                id="c1",
                path="src/a.py",
                description="Major TUI restructure covering US-001 US-004 US-005 US-006",
            ),
            FileChange(
                id="c2",
                path="src/b.py",
                description="Fix US-014 timeout issue",
            ),
            FileChange(
                id="c3",
                path="src/c.py",
                description="No US ref here",
            ),
        )
        manifest = _make_manifest(changes=changes)
        plan = _make_plan(batches=(_make_batch(0),))
        record_batch_snapshot(tmp_path, manifest, plan)
        record = json.loads((tmp_path / ".milknado" / "calibration.jsonl").read_text().strip())
        assert record["max_us_refs_per_change"] == 4
        assert record["multi_story_change_count"] == 1

    def test_distinct_path_count(self, tmp_path: Path) -> None:
        """Two changes sharing a path → distinct_path_count counts each path once."""
        changes = (
            FileChange(id="c1", path="src/foo.py", description="part 1"),
            FileChange(id="c2", path="src/foo.py", description="part 2"),
            FileChange(id="c3", path="src/bar.py", description="other"),
        )
        manifest = _make_manifest(changes=changes)
        plan = _make_plan(batches=(_make_batch(0),))
        record_batch_snapshot(tmp_path, manifest, plan)
        record = json.loads((tmp_path / ".milknado" / "calibration.jsonl").read_text().strip())
        assert record["distinct_path_count"] == 2
        assert record["change_count"] == 3

    def test_empty_manifest(self, tmp_path: Path) -> None:
        """Zero changes → all quality signals zeroed without errors."""
        manifest = _make_manifest(changes=())
        plan = _make_plan(batches=())
        record_batch_snapshot(tmp_path, manifest, plan)
        record = json.loads((tmp_path / ".milknado" / "calibration.jsonl").read_text().strip())
        assert record["impl_change_count"] == 0
        assert record["test_change_count"] == 0
        assert record["test_to_impl_ratio"] == 0.0
        assert record["distinct_path_count"] == 0

    def test_spec_path_recorded(self, tmp_path: Path) -> None:
        """manifest.spec_path lands in the record verbatim."""
        manifest = PlanChangeManifest(
            manifest_version=MANIFEST_VERSION,
            goal="g",
            goal_summary="gs",
            spec_path=".claude/specs/foo.md",
            changes=(_make_change("c1"),),
            new_relationships=(),
        )
        plan = _make_plan(batches=(_make_batch(0),))
        record_batch_snapshot(tmp_path, manifest, plan)
        record = json.loads((tmp_path / ".milknado" / "calibration.jsonl").read_text().strip())
        assert record["spec_path"] == ".claude/specs/foo.md"


class TestAppendBehavior:
    def test_second_call_appends(self, tmp_path: Path) -> None:
        """Two calls → two lines in the JSONL file."""
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(_make_batch(0),))
        record_batch_snapshot(tmp_path, manifest, plan)
        record_batch_snapshot(tmp_path, manifest, plan)

        lines = [
            ln
            for ln in (tmp_path / ".milknado" / "calibration.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert len(lines) == 2


class TestAutoCreateDirectory:
    def test_creates_milknado_dir(self, tmp_path: Path) -> None:
        """`.milknado/` is auto-created when missing."""
        assert not (tmp_path / ".milknado").exists()
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(_make_batch(0),))
        record_batch_snapshot(tmp_path, manifest, plan)
        assert (tmp_path / ".milknado" / "calibration.jsonl").exists()


class TestOversizedCount:
    def test_oversized_count_correct(self, tmp_path: Path) -> None:
        """1 oversized batch + 2 normal → oversized_count=1."""
        batches = (
            Batch(index=0, change_ids=("c1",), depends_on=(), oversized=True),
            Batch(index=1, change_ids=("c2",), depends_on=(0,), oversized=False),
            Batch(index=2, change_ids=("c3",), depends_on=(1,), oversized=False),
        )
        manifest = _make_manifest(
            changes=(
                _make_change("c1"),
                _make_change("c2", "src/bar.py"),
                _make_change("c3", "src/baz.py"),
            )
        )
        plan = _make_plan(batches=batches)
        record_batch_snapshot(tmp_path, manifest, plan)

        line = (tmp_path / ".milknado" / "calibration.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["oversized_count"] == 1
        assert record["batch_count"] == 3


class TestNewRelationshipCount:
    def test_new_relationship_count(self, tmp_path: Path) -> None:
        """new_relationship_count reflects len(manifest.new_relationships)."""
        rels = (
            NewRelationship(
                source_change_id="c1",
                dependant_change_id="c2",
                reason="new_call",
            ),
        )
        manifest = _make_manifest(
            changes=(_make_change("c1"), _make_change("c2", "src/bar.py")),
            new_relationships=rels,
        )
        plan = _make_plan(batches=(_make_batch(0),))
        record_batch_snapshot(tmp_path, manifest, plan)

        line = (tmp_path / ".milknado" / "calibration.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["new_relationship_count"] == 1


class TestDiskError:
    def test_no_exception_on_disk_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OSError on write → no exception raised, no record appended."""
        original_open = Path.open

        def bad_open(self: Path, *args: object, **kwargs: object) -> object:
            if "calibration.jsonl" in str(self):
                raise OSError("disk full")
            return original_open(self, *args, **kwargs)  # type: ignore

        monkeypatch.setattr(Path, "open", bad_open)

        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(_make_batch(0),))
        # Must not raise
        record_batch_snapshot(tmp_path, manifest, plan)


class TestTimestamp:
    def test_timestamp_is_utc_iso8601(self, tmp_path: Path) -> None:
        """Timestamp ends with +00:00 or Z (UTC ISO 8601)."""
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(_make_batch(0),))
        record_batch_snapshot(tmp_path, manifest, plan)

        line = (tmp_path / ".milknado" / "calibration.jsonl").read_text().strip()
        record = json.loads(line)
        ts: str = record["timestamp"]
        assert ts.endswith("+00:00") or ts.endswith("Z"), f"Expected UTC timestamp, got: {ts!r}"


class TestSpreadStats:
    def test_spread_stats_from_spread_report(self, tmp_path: Path) -> None:
        """max_spread and mean_spread reflect the spread_report values."""
        sym1 = SymbolRef(name="foo", file="src/foo.py")
        sym2 = SymbolRef(name="bar", file="src/bar.py")
        spread_report = (
            SymbolSpread(symbol=sym1, spread=3),
            SymbolSpread(symbol=sym2, spread=7),
        )
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(_make_batch(0),), spread_report=spread_report)
        record_batch_snapshot(tmp_path, manifest, plan)

        line = (tmp_path / ".milknado" / "calibration.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["max_spread"] == 7
        assert abs(record["mean_spread"] - 5.0) < 0.001

    def test_empty_spread_report(self, tmp_path: Path) -> None:
        """Empty spread_report → max_spread=0, mean_spread=0.0."""
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(_make_batch(0),), spread_report=())
        record_batch_snapshot(tmp_path, manifest, plan)

        line = (tmp_path / ".milknado" / "calibration.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["max_spread"] == 0
        assert record["mean_spread"] == 0.0
