"""Adversarial tests for the calibration telemetry sink.

Focus: permission errors, symlink traps, NaN/Infinity floats, concurrent writes.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from milknado.domains.batching.change import (
    Batch,
    BatchPlan,
    FileChange,
    SymbolRef,
    SymbolSpread,
)
from milknado.domains.planning.manifest import MANIFEST_VERSION, PlanChangeManifest
from milknado.domains.planning.telemetry import _build_record, record_batch_snapshot


def _make_manifest(changes: tuple[FileChange, ...] = ()) -> PlanChangeManifest:
    return PlanChangeManifest(
        manifest_version=MANIFEST_VERSION,
        goal="Test goal",
        goal_summary="Test summary",
        spec_path=None,
        changes=changes,
        new_relationships=(),
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


def _make_change(cid: str) -> FileChange:
    return FileChange(id=cid, path="src/foo.py", description="desc")


class TestPermissionErrors:
    def test_readonly_milknado_dir_does_not_raise(self, tmp_path: Path) -> None:
        """Write to a read-only .milknado/ should be swallowed, not crash."""
        milknado_dir = tmp_path / ".milknado"
        milknado_dir.mkdir()
        # Make directory read-only
        milknado_dir.chmod(0o444)
        try:
            manifest = _make_manifest(changes=(_make_change("c1"),))
            plan = _make_plan(batches=(Batch(index=0, change_ids=("c1",), depends_on=()),))
            record_batch_snapshot(tmp_path, manifest, plan)
            # Should not raise — telemetry must never block the planner
        finally:
            milknado_dir.chmod(0o755)  # restore for cleanup

    def test_no_write_permission_on_parent(self, tmp_path: Path) -> None:
        """Read-only project_root → mkdir fails → OSError caught silently."""
        # Make the project root itself read-only
        tmp_path.chmod(0o555)
        try:
            manifest = _make_manifest(changes=(_make_change("c1"),))
            plan = _make_plan()
            record_batch_snapshot(tmp_path, manifest, plan)
        finally:
            tmp_path.chmod(0o755)


class TestSymlinkToFullDevice:
    def test_symlink_to_dev_null_no_crash(self, tmp_path: Path) -> None:
        """Symlink calibration.jsonl → /dev/null — writes should succeed silently."""
        milknado_dir = tmp_path / ".milknado"
        milknado_dir.mkdir()
        calibration = milknado_dir / "calibration.jsonl"
        calibration.symlink_to("/dev/null")

        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(Batch(index=0, change_ids=("c1",), depends_on=()),))
        # Should not raise; /dev/null accepts all writes
        record_batch_snapshot(tmp_path, manifest, plan)

    @pytest.mark.skipif(
        not Path("/dev/full").exists(), reason="/dev/full not available on this platform"
    )
    def test_symlink_to_dev_full_does_not_raise(self, tmp_path: Path) -> None:
        """/dev/full always returns ENOSPC on write — must be caught silently."""
        milknado_dir = tmp_path / ".milknado"
        milknado_dir.mkdir()
        calibration = milknado_dir / "calibration.jsonl"
        calibration.symlink_to("/dev/full")

        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(Batch(index=0, change_ids=("c1",), depends_on=()),))
        # Must not raise — OSError from /dev/full must be caught
        record_batch_snapshot(tmp_path, manifest, plan)


class TestFloatEdgeCasesInSpread:
    def test_nan_spread_in_spread_report(self) -> None:
        """NaN in spread — _build_record must not crash."""
        sym = SymbolRef(name="foo", file="src/foo.py")
        spread_report = (SymbolSpread(symbol=sym, spread=float("nan")),)  # type: ignore
        # SymbolSpread.spread is typed as int; float("nan") is a float.
        # This probes whether the runtime allows it and whether max/mean crash.
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(spread_report=spread_report)
        # May raise TypeError in max() or propagate NaN — document behavior.
        try:
            record = _build_record(manifest, plan)
            # If it doesn't crash, NaN may propagate into the record.
            # json.dumps(NaN) raises ValueError in strict mode.
            try:
                dumped = json.dumps(record)
                parsed = json.loads(dumped)
                # If JSON serialized, max_spread might be "NaN" string or similar.
                _ = parsed
            except (ValueError, TypeError):
                pass  # NaN not JSON-serializable — silent failure in production
        except (TypeError, ValueError):
            pass  # Acceptable: runtime rejects NaN spread

    def test_infinity_spread_in_spread_report(self) -> None:
        """Infinity in spread — json.dumps(Infinity) raises ValueError in standard JSON."""
        sym = SymbolRef(name="foo", file="src/foo.py")
        spread_report = (SymbolSpread(symbol=sym, spread=float("inf")),)  # type: ignore
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(spread_report=spread_report)
        try:
            record = _build_record(manifest, plan)
            result = json.dumps(record)
            _ = result
        except (TypeError, ValueError):
            pass  # Expected: json.dumps rejects Infinity


class TestConcurrentWrites:
    def test_concurrent_append_all_lines_valid_json(self, tmp_path: Path) -> None:
        """Two threads writing simultaneously → all lines must be valid JSON."""
        errors: list[Exception] = []
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(Batch(index=0, change_ids=("c1",), depends_on=()),))

        def write_records() -> None:
            try:
                for _ in range(10):
                    record_batch_snapshot(tmp_path, manifest, plan)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write_records) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

        jsonl_path = tmp_path / ".milknado" / "calibration.jsonl"
        if jsonl_path.exists():
            for line in jsonl_path.read_text().splitlines():
                if line.strip():
                    # Every non-empty line must be valid JSON
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        pytest.fail(
                            f"Corrupted JSONL line from concurrent write: "
                            f"{exc}\nLine: {line!r}"
                        )


class TestPlanModeFields:
    def test_fresh_mode_defaults(self) -> None:
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan()
        record = _build_record(manifest, plan)
        assert record["plan_mode_used"] == "fresh"
        assert record["existing_node_count_at_plan_start"] == 0
        assert record["spec_hash_changed"] is None
        assert record["orphaned_worktree_count"] == 0

    def test_resume_mode_stored(self) -> None:
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan()
        record = _build_record(
            manifest, plan,
            plan_mode_used="resume",
            existing_node_count_at_plan_start=7,
            spec_hash_changed=True,
            orphaned_worktree_count=2,
        )
        assert record["plan_mode_used"] == "resume"
        assert record["existing_node_count_at_plan_start"] == 7
        assert record["spec_hash_changed"] is True
        assert record["orphaned_worktree_count"] == 2

    def test_reset_mode_with_no_hash_change(self) -> None:
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan()
        record = _build_record(
            manifest, plan,
            plan_mode_used="reset",
            existing_node_count_at_plan_start=3,
            spec_hash_changed=False,
            orphaned_worktree_count=0,
        )
        assert record["plan_mode_used"] == "reset"
        assert record["existing_node_count_at_plan_start"] == 3
        assert record["spec_hash_changed"] is False

    def test_plan_mode_fields_roundtrip_json(self, tmp_path: Path) -> None:
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(Batch(index=0, change_ids=("c1",), depends_on=()),))
        record_batch_snapshot(
            tmp_path, manifest, plan,
            plan_mode_used="resume",
            existing_node_count_at_plan_start=5,
            spec_hash_changed=True,
            orphaned_worktree_count=3,
        )
        line = (tmp_path / ".milknado" / "calibration.jsonl").read_text().strip()
        parsed = json.loads(line)
        assert parsed["plan_mode_used"] == "resume"
        assert parsed["existing_node_count_at_plan_start"] == 5
        assert parsed["spec_hash_changed"] is True
        assert parsed["orphaned_worktree_count"] == 3


class TestMegaBatchFields:
    def test_defaults_are_not_aborted(self) -> None:
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan()
        record = _build_record(manifest, plan)
        assert record["mega_batch_aborted"] is False
        assert record["mega_batch_threshold"] == 5

    def test_aborted_flag_stored(self) -> None:
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan()
        record = _build_record(manifest, plan, mega_batch_aborted=True, mega_batch_threshold=3)
        assert record["mega_batch_aborted"] is True
        assert record["mega_batch_threshold"] == 3

    def test_mega_batch_fields_roundtrip_json(self, tmp_path: Path) -> None:
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(Batch(index=0, change_ids=("c1",), depends_on=()),))
        record_batch_snapshot(
            tmp_path, manifest, plan,
            mega_batch_aborted=True,
            mega_batch_threshold=7,
        )
        line = (tmp_path / ".milknado" / "calibration.jsonl").read_text().strip()
        parsed = json.loads(line)
        assert parsed["mega_batch_aborted"] is True
        assert parsed["mega_batch_threshold"] == 7


class TestCoverageFields:
    def test_defaults_are_zero(self) -> None:
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan()
        record = _build_record(manifest, plan)
        assert record["verify_rounds_used"] == 0
        assert record["verify_round_cap_hit"] is False
        assert record["coverage_gaps_remaining"] == 0
        assert record["coverage_us_refs_uncovered"] == []

    def test_verify_fields_stored(self) -> None:
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan()
        record = _build_record(
            manifest, plan,
            verify_rounds_used=2,
            verify_round_cap_hit=True,
            coverage_gaps_remaining=3,
            coverage_us_refs_uncovered=["US-001", "US-002", "US-003"],
        )
        assert record["verify_rounds_used"] == 2
        assert record["verify_round_cap_hit"] is True
        assert record["coverage_gaps_remaining"] == 3
        assert record["coverage_us_refs_uncovered"] == ["US-001", "US-002", "US-003"]

    def test_none_uncovered_normalises_to_empty_list(self) -> None:
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan()
        record = _build_record(manifest, plan, coverage_us_refs_uncovered=None)
        assert record["coverage_us_refs_uncovered"] == []

    def test_coverage_fields_roundtrip_json(self, tmp_path: Path) -> None:
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = _make_plan(batches=(Batch(index=0, change_ids=("c1",), depends_on=()),))
        record_batch_snapshot(
            tmp_path, manifest, plan,
            verify_rounds_used=1,
            verify_round_cap_hit=False,
            coverage_gaps_remaining=2,
            coverage_us_refs_uncovered=["US-005", "US-007"],
        )
        line = (tmp_path / ".milknado" / "calibration.jsonl").read_text().strip()
        parsed = json.loads(line)
        assert parsed["verify_rounds_used"] == 1
        assert parsed["verify_round_cap_hit"] is False
        assert parsed["coverage_gaps_remaining"] == 2
        assert parsed["coverage_us_refs_uncovered"] == ["US-005", "US-007"]


class TestUnknownSolverStatus:
    def test_unknown_solver_status_string_stored(self, tmp_path: Path) -> None:
        """solver_status not in the known enum set is stored verbatim."""
        manifest = _make_manifest(changes=(_make_change("c1"),))
        plan = BatchPlan(
            batches=(),
            spread_report=(),
            solver_status="TIMEOUT",  # type: ignore
        )
        record_batch_snapshot(tmp_path, manifest, plan)
        line = (tmp_path / ".milknado" / "calibration.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["solver_status"] == "TIMEOUT"
