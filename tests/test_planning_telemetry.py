from __future__ import annotations

import json
from pathlib import Path

from milknado.domains.planning.budget import (
    BudgetViolation,
    ConfidenceLevel,
    DynamicMultiplier,
)
from milknado.domains.planning.manifest import (
    AtomTokenBudget,
    PlanAtom,
    PlanManifest,
)
from milknado.domains.planning.telemetry import (
    CALIBRATION_FILENAME,
    record_budget_snapshot,
)


def _atom(atom_id: str, *, with_budget: bool = True) -> PlanAtom:
    budget = (
        AtomTokenBudget(
            estimated_read_tokens=10_000,
            estimated_write_tokens=2_000,
            estimated_total_tokens=30_000,
            split_required=False,
        )
        if with_budget
        else None
    )
    return PlanAtom(
        id=atom_id, description=atom_id, depends_on=[], files=[],
        token_budget=budget,
    )


def _read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class TestRecordBudgetSnapshot:
    def test_creates_milknado_dir(self, tmp_path: Path) -> None:
        manifest = PlanManifest("v", [_atom("A1")])
        path = record_budget_snapshot(tmp_path, manifest, [])
        assert path == tmp_path / ".milknado" / CALIBRATION_FILENAME
        assert path.exists()

    def test_writes_one_record_per_atom(self, tmp_path: Path) -> None:
        manifest = PlanManifest("v", [_atom("A1"), _atom("A2"), _atom("A3")])
        path = record_budget_snapshot(tmp_path, manifest, [])
        records = _read_records(path)
        assert [r["atom_id"] for r in records] == ["A1", "A2", "A3"]

    def test_includes_budget_fields(self, tmp_path: Path) -> None:
        manifest = PlanManifest("v", [_atom("A1")])
        path = record_budget_snapshot(tmp_path, manifest, [])
        record = _read_records(path)[0]
        assert record["estimated_read_tokens"] == 10_000
        assert record["estimated_write_tokens"] == 2_000
        assert record["estimated_total_tokens"] == 30_000
        assert record["split_required"] is False

    def test_missing_budget_yields_null_fields(self, tmp_path: Path) -> None:
        manifest = PlanManifest("v", [_atom("A1", with_budget=False)])
        path = record_budget_snapshot(tmp_path, manifest, [])
        record = _read_records(path)[0]
        assert record["estimated_read_tokens"] is None
        assert record["estimated_write_tokens"] is None
        assert record["estimated_total_tokens"] is None

    def test_attaches_violations_by_atom_id(self, tmp_path: Path) -> None:
        manifest = PlanManifest("v", [_atom("A1"), _atom("A2")])
        violations = [
            BudgetViolation("A1", "write_pressure", "hot"),
            BudgetViolation("A1", "band_tight", "close"),
            BudgetViolation("A2", "missing_budget", "absent"),
        ]
        path = record_budget_snapshot(tmp_path, manifest, violations)
        records = _read_records(path)
        a1 = next(r for r in records if r["atom_id"] == "A1")
        a2 = next(r for r in records if r["atom_id"] == "A2")
        assert [v["kind"] for v in a1["violations"]] == ["write_pressure", "band_tight"]
        assert [v["kind"] for v in a2["violations"]] == ["missing_budget"]

    def test_appends_across_calls(self, tmp_path: Path) -> None:
        manifest = PlanManifest("v", [_atom("A1")])
        record_budget_snapshot(tmp_path, manifest, [])
        record_budget_snapshot(tmp_path, manifest, [])
        path = tmp_path / ".milknado" / CALIBRATION_FILENAME
        assert len(_read_records(path)) == 2

    def test_records_multiplier_and_confidence(self, tmp_path: Path) -> None:
        manifest = PlanManifest("v", [_atom("A1")])
        path = record_budget_snapshot(
            tmp_path, manifest, [],
            multiplier=DynamicMultiplier.HEAVY,
            confidence=ConfidenceLevel.LOW,
        )
        record = _read_records(path)[0]
        assert record["multiplier"] == "heavy"
        assert record["confidence"] == "low"

    def test_empty_manifest_writes_nothing(self, tmp_path: Path) -> None:
        manifest = PlanManifest("v", [])
        path = record_budget_snapshot(tmp_path, manifest, [])
        assert path.exists()
        assert path.read_text() == ""
